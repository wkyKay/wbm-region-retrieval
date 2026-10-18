"""Single-label transfer must retain saved inputs, frozen pools and official scores."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from wbm_retrieval.common import ROOT, csv_rows, file_hash, load_config, read_json, write_csv, write_json
from wbm_retrieval.data import prepare
from wbm_retrieval.diagnostics import source_snapshot
from wbm_retrieval.evaluation import evaluate_ranking
from wbm_retrieval.matching import map_similarity
from wbm_retrieval.models import Encoder, Projector
from wbm_retrieval.single_label import (load_saved_model, paired_comparison, prepare_top1, run_single_label,
                                        score_and_evaluate, summarize)
from wbm_retrieval.training import encoder_spec


def fixture(root, checkpoint=False):
    config = copy.deepcopy(read_json(ROOT / "configs/experiment_a.json"))
    config.update(output_dir=str(root / "source"), data_file=str(root / "data.npz"),
                  split_manifest=str(root / "split.csv"), query_manifest=str(root / "queries.csv"),
                  candidate_manifest=str(root / "candidates.csv"), device="cpu", num_workers=0)
    config["proposal"]["top_k"] = 6
    config["local"].pop("opening")
    config["evaluation"]["bootstrap_samples"] = 0
    signatures = [(i,) for i in range(8) for _ in range(2)] + [(0, 1), (2, 3)]
    labels, maps, rows = [], [], []
    for split in ("train", "valid", "test"):
        for signature in signatures:
            sample_id = len(rows)
            label = np.zeros(8, dtype=np.uint8)
            label[list(signature)] = 1
            raw = np.ones((32, 32), dtype=np.uint8)
            raw[3:12, 3:8 + signature[0]] = 2
            raw[22:25, 22:25] = 2
            if split == "valid" and sample_id == 18:
                raw[:] = 1  # Keep an empty-proposal query in the evaluation.
            maps.append(raw)
            labels.append(label)
            row = {"sample_id": sample_id, "original_index": sample_id, "valid_index": sample_id, "split": split}
            row.update({f"label_{i}": int(value) for i, value in enumerate(label)})
            rows.append(row)
    np.savez(config["data_file"], np.array(maps), np.array(labels))
    write_csv(config["split_manifest"], rows, list(rows[0]))
    write_csv(config["query_manifest"], [{"sample_id": 36}], ["sample_id"])
    write_csv(config["candidate_manifest"], [{"query_id": 36, "candidate_id": 37}], ["query_id", "candidate_id"])
    config["expected_candidates"] = 1
    prepare(config)
    if checkpoint:
        model = torch.nn.ModuleDict({"encoder": Encoder(), "projector": Projector(512, 128)})
        folder = root / "source/models/local"
        folder.mkdir(parents=True)
        torch.save({"model": model.state_dict(), "epoch": 7, "signature": "historical-code-signature",
                    "best_loss": .5, "embedding_dim": 512, "encoder_spec": encoder_spec(config)}, folder / "best.pt")
    return config


class SingleLabelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.settings = {"splits": ["valid"], "batch_size": 8, "num_workers": 0, "bootstrap": 10, "seed": 2026}

    def test_top1_preserves_inputs_single_labels_split_self_exclusion_and_empty_maps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(root)
            source, output = root / "source", root / "evaluation"
            before = source_snapshot(source)
            manifest = prepare_top1(source, output, config, "valid")
            self.assertEqual(manifest["queries"], 16)
            self.assertEqual(manifest["maps_without_region"], 1)
            coverage = csv_rows(output / "valid/region_coverage.csv")
            ids = {int(row["sample_id"]) for row in coverage}
            self.assertEqual(ids, set(range(18, 34)))
            edges = csv_rows(output / "valid/protocol/candidates.csv")
            self.assertEqual(len(edges), 16 * 15)
            self.assertTrue(all(int(row["query_id"]) != int(row["candidate_id"]) for row in edges))
            with np.load(source / "cache/valid_regions.npz") as old:
                descriptors = np.load(output / "valid/zernike47.npy")
                np.testing.assert_array_equal(descriptors[1], old["descriptor"][old["offsets"][1]])
            self.assertEqual(prepare_top1(source, output, config, "valid"), manifest)
            self.assertEqual(before, source_snapshot(source))
            (output / "valid/protocol/queries.csv").write_text("sample_id\n999\n")
            with self.assertRaises(ValueError):
                prepare_top1(source, output, config, "valid")

    def test_score_and_metric_equivalence_to_official_matcher_and_evaluator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(root)
            source, output = root / "source", root / "evaluation"
            prepare_top1(source, output, config, "valid")
            folder = output / "valid"
            coverage = csv_rows(folder / "region_coverage.csv")
            np.save(folder / "h512.npy", np.random.default_rng(2).normal(size=(16, 512)))
            ids = [int(row["sample_id"]) for row in coverage]
            labels = np.load(source / "cache/labels.npy")
            protocol = SimpleNamespace(queries=ids, candidates={q: [c for c in ids if c != q] for q in ids},
                                       labels={q: labels[q] for q in ids})
            for method in ("proposed", "handcrafted"):
                table, rows, _, _ = score_and_evaluate(folder, method, coverage, config, self.settings)
                official = evaluate_ranking(folder / "methods" / method / "rankings.csv", protocol,
                                            config["evaluation"], 2026)
                micro = next(row for row in table if row["averaging"] == "micro")
                for key, value in official["metrics"].items():
                    self.assertAlmostEqual(micro[key], value)
                features = np.load(folder / ("h512.npy" if method == "proposed" else "zernike47.npy"))
                rankings = csv_rows(folder / "methods" / method / "rankings.csv")
                for row in rankings:
                    qi, ci = ids.index(int(row["query_id"])), ids.index(int(row["candidate_id"]))
                    def region(index):
                        present = int(coverage[index]["has_region"])
                        return {"embedding": features[index:index+present], "area": np.array([13.] * present),
                                "position": np.zeros((present, 2)), "scale": np.ones(present)}
                    matching = {**config["matching"], "representation": "zernike" if method == "handcrafted" else "learned",
                                "zernike_distance_scale": 1.}
                    self.assertAlmostEqual(float(row["similarity_score"]), map_similarity(region(qi), region(ci), matching), places=12)

    def test_macro_differs_from_micro_and_missing_positives_remain_undefined(self):
        rows = [{"query_id": i, "label_id": label, "num_positives": positives, "has_region": 1,
                 "selected_fraction": 1., "mAP": value, "Top1Accuracy": float(value or 0)}
                for i, (label, positives, value) in enumerate([(0, 2, 1.), (0, 2, 1.), (0, 2, 1.), (1, 0, None), (2, 1, 0.)])]
        table, result = summarize(rows, "proposed", "valid", 10, 2026)
        self.assertEqual(result["micro"]["mAP"], .75)
        self.assertEqual(result["macro"]["mAP"], .5)
        self.assertEqual(result["valid_queries_by_metric"]["mAP"], 4)
        self.assertEqual(result["macro_classes_by_metric"]["mAP"], 2)
        self.assertEqual(result["macro_classes_by_metric"]["Top1Accuracy"], 3)
        differences = paired_comparison(rows, rows, self.settings)
        self.assertEqual(differences[0]["macro_mean_difference"], 0)

    def test_historical_checkpoint_inference_resume_and_isolation_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(root, checkpoint=True)
            source, output = root / "source", root / "evaluation"
            before = source_snapshot(source)
            best_hash = file_hash(source / "models/local/best.pt")
            # This deliberately cannot match a training signature made with new code.
            model, info = load_saved_model(source, config, torch.device("cpu"))
            self.assertEqual(info["historical_training_signature"], "historical-code-signature")
            self.assertFalse(model.training)
            del model
            with mock.patch("torch.optim.SGD", side_effect=AssertionError("No training allowed")), \
                    mock.patch("torch.Tensor.backward", side_effect=AssertionError("No backward allowed")):
                run_single_label(source, output, torch.device("cpu"), self.settings)
                hashes = {name: file_hash(output / "valid" / name) for name in ("h512.npy", "protocol/candidates.csv")}
                run_single_label(source, output, torch.device("cpu"), self.settings, resume=True)
            self.assertTrue(read_json(output / "source_audit.json")["source_unchanged"])
            self.assertEqual(read_json(output / "run_manifest.json")["status"], "complete")
            self.assertEqual(before, source_snapshot(source))
            self.assertEqual(best_hash, file_hash(source / "models/local/best.pt"))
            self.assertEqual(hashes, {name: file_hash(output / "valid" / name) for name in hashes})
            with self.assertRaises(ValueError):
                run_single_label(source, source / "bad", torch.device("cpu"), self.settings)
            with self.assertRaises(ValueError):
                run_single_label(source, output, torch.device("cpu"), {**self.settings, "seed": 9}, resume=True)


if __name__ == "__main__":
    unittest.main()
