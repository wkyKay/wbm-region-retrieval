"""Invariants for isolated multilabel supervision and hard-label retrieval."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from experiments.supervised.suite import (Classifier, WholeMaps, exclusive, load_suite, prepare,
                                         split_protocol)
from experiments.supervised.evaluation import (classification_metrics, hard_containment, rank_predictions,
                                               retrieval_report)
from experiments.supervised.training import better_classifier, evaluate, train
from wbm_retrieval.common import ROOT, file_hash, read_json, seed_everything, write_csv, write_json
from wbm_retrieval.data import containment
from wbm_retrieval.evaluation import evaluate_ranking
from wbm_retrieval.models import Encoder, Projector
from wbm_retrieval.training import load_checkpoint, parameter_hash


class SupervisedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_backbone_matches_proposed_initialization_and_forward(self):
        seed_everything(2026)
        model = Classifier()
        seed_everything(2026)
        encoder, _ = Encoder(512), Projector(512, 128)
        self.assertEqual(parameter_hash(model.encoder), parameter_hash(encoder))
        self.assertEqual(model.encoder.stem[0].stride, (2, 2))
        self.assertEqual(model(torch.zeros(2, 1, 96, 96)).shape, (2, 8))

    def test_hard_containment_direction_empty_and_ties(self):
        q = np.array([1, 1, 0, 0, 0, 0, 0, 0])
        candidates = np.array([q, [1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0]])
        np.testing.assert_array_equal(hard_containment(q, candidates), [1, .5, 1])
        self.assertEqual(hard_containment(candidates[1], q[None])[0], 1)
        np.testing.assert_array_equal(hard_containment(q*0, candidates), [0, 0, 0])
        protocol = SimpleNamespace(queries=[1], candidates={1: [4, 3, 2]})
        ids = np.array([1, 2, 3, 4])
        predictions = np.stack([q, q, q, q])
        rows, _ = next(rank_predictions(ids, predictions, protocol))
        self.assertEqual([r[2] for r in rows], [2, 3, 4])

    def test_rank_does_not_read_ground_truth(self):
        class PredictionOnlyProtocol:
            queries, candidates = [1], {1: [2, 3]}
            @property
            def labels(self):
                raise AssertionError("Ground truth leaked into ranking")
        predicted = np.zeros((3, 8), dtype=np.uint8)
        rows, audit = next(rank_predictions([1, 2, 3], predicted, PredictionOnlyProtocol()))
        self.assertTrue(audit["empty_prediction"])
        self.assertEqual([r[3] for r in rows], [0, 0])

    def test_classification_known_values_and_absent_class(self):
        y = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]])
        p = y*.8+.1
        result = classification_metrics(y, p)
        self.assertEqual(result["metrics"]["subset_accuracy"], 1)
        self.assertEqual(result["metrics"]["micro_f1"], 1)
        self.assertEqual(result["metrics"]["hamming_loss"], 0)
        self.assertEqual(result["metrics"]["macro_f1"], .25)  # Eight labels, absent labels use zero_division=0.
        self.assertEqual(result["metrics"]["macro_ap"], 1)
        self.assertIsNone(result["per_class"][2]["ap"])
        result = classification_metrics(y, np.zeros_like(p))
        self.assertEqual(result["metrics"]["empty_prediction_rate"], 1)
        self.assertEqual(result["metrics"]["hamming_loss"], 3/16)

    def test_best_classifier_lexicographic_selection(self):
        self.assertTrue(better_classifier(.6, 5., .5, .1))
        self.assertTrue(better_classifier(.5, .1, .5, .2))
        self.assertFalse(better_classifier(.5, .2, .5, .1))
        self.assertFalse(better_classifier(.5, .1, .5, .1))

    def fixture(self, directory):
        folder = Path(directory)
        labels = np.zeros((20, 8), dtype=np.uint8)
        for i in range(20):
            labels[i, i % 8] = 1
            if i % 3 == 0:
                labels[i, (i+1) % 8] = 1
        maps = np.ones((20, 12, 12), dtype=np.uint8)
        for i in range(20):
            maps[i, 2+i%6:4+i%6, 3:9] = 2
        np.savez_compressed(folder / "source.npz", maps, labels)
        rows = []
        for i in range(20):
            row = {"sample_id": i, "original_index": i, "valid_index": i,
                   "split": "train" if i < 12 else "valid" if i < 16 else "test"}
            row.update({f"label_{k}": int(labels[i, k]) for k in range(8)})
            rows.append(row)
        write_csv(folder / "split.csv", rows, list(rows[0]))
        write_csv(folder / "queries.csv", [{"sample_id": i} for i in range(16, 20)], ["sample_id"])
        write_csv(folder / "candidates.csv", [{"query_id": q, "candidate_id": c} for q in range(16, 20)
                  for c in range(16, 20) if c != q], ["query_id", "candidate_id"])
        base = {"data_file": str(folder / "source.npz"), "split_manifest": str(folder / "split.csv"),
                "query_manifest": str(folder / "queries.csv"), "candidate_manifest": str(folder / "candidates.csv"),
                "expected_candidates": 3, "output_dir": str(folder / "legacy")}
        write_json(folder / "base.json", base)
        suite = read_json(ROOT / "configs/supervised_multilabel.json")
        suite.update(base_config=str(folder / "base.json"), output_root=str(folder / "new"), budgets=[.5],
                     device="cpu", num_workers=0, cpu_threads=2, train_diagnostic_maps=4,
                     inference_batch_size=4, valid_pool={"queries": 2, "candidates": 3, "seed": 2026})
        suite["training"].update(epochs=2, batch_size=4)
        suite["evaluation"]["bootstrap_samples"] = 0
        write_json(folder / "suite.json", suite)
        return load_suite(folder / "suite.json")

    def test_output_guards_and_cache_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(directory)
            with self.assertRaises(ValueError):
                load_suite(Path(directory) / "suite.json", output_root=directory)
            unknown = Path(directory) / "unknown"
            unknown.mkdir()
            (unknown / "existing.txt").write_text("preserve")
            with self.assertRaises(ValueError), exclusive(unknown):
                pass
            with exclusive(config["output_root"]):
                signature, _ = prepare(config)
                self.assertEqual(signature, prepare(config)[0])
                changed = copy.deepcopy(config)
                changed["training"]["learning_rate"] *= 2
                with self.assertRaises(ValueError):
                    prepare(changed)
                (Path(config["output_root"]) / "cache/train_ids.npy").write_bytes(b"tampered")
                with self.assertRaises(ValueError):
                    prepare(config)

    def test_training_resume_and_legacy_retrieval_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(directory)
            before = {p.name: file_hash(p) for p in Path(directory).glob("*.csv")}
            with exclusive(config["output_root"]):
                signature, protocol = prepare(config)
                train(config, protocol, signature, .5, stop_after=1)
                with self.assertRaises(FileExistsError):
                    train(config, protocol, signature, .5)
                train(config, protocol, signature, .5, resume=True)
                folder = Path(config["output_root"]) / "supervised_50pct"
                hashes = {p.name: file_hash(p) for p in (folder / "models").glob("*.pt")}
                train(config, protocol, signature, .5, resume=True)
                self.assertEqual(hashes, {p.name: file_hash(p) for p in (folder / "models").glob("*.pt")})
                state = load_checkpoint(folder / "models/last.pt")
                self.assertNotEqual(state["initial_parameter_sha256"], parameter_hash(state["model"]))
                evaluate(config, protocol, signature, .5, "valid")
                self.assertFalse((folder / "classification/test").exists())
                evaluate(config, protocol, signature, .5, "test")
                saved = read_json(folder / "retrieval/test/metrics.json")
                recomputed = evaluate_ranking(folder / "retrieval/test/rankings.csv", protocol, config["evaluation"], config["seed"])
                self.assertEqual(saved, recomputed)
                data = WholeMaps(config["output_root"], "train", rotation=True)
                torch.testing.assert_close(data[0][1], data[0][1])
                config2 = {**config, "output_root": str(Path(directory) / "continuous")}
                with exclusive(config2["output_root"]):
                    signature2, protocol2 = prepare(config2)
                    train(config2, protocol2, signature2, .5)
                other = load_checkpoint(Path(config2["output_root"]) / "supervised_50pct/models/last.pt")
                self.assertEqual(state["optimizer_steps"], other["optimizer_steps"])
                for name, value in state["model"].items():
                    torch.testing.assert_close(value, other["model"][name], rtol=0, atol=0)
            self.assertEqual(before, {p.name: file_hash(p) for p in Path(directory).glob("*.csv")})


if __name__ == "__main__":
    unittest.main()
