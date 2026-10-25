"""G count scoring, actual 1000-negative CPU training, CLI, resume and isolation."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from test_core import fixture_config
from wbm_retrieval import ablation, ablation_ef
from wbm_retrieval.common import ROOT, csv_rows, file_hash, read_json, seed_everything, write_json
from wbm_retrieval.evaluation import evaluate_ranking
from wbm_retrieval.matching import map_similarity as legacy_similarity
from wbm_retrieval.training import load_checkpoint, parameter_hash
from experiments.g import training
from experiments.g.matching import batched_map_scores, count_similarity, map_similarity, method_settings
from experiments.g.suite import build_model, evaluate, exclusive, load_suite, prepare_suite, summarize


class GTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.suite, self.config = load_suite(ROOT / "configs/ablation_g.json", device="cpu", workers=0)

    def fixture(self, directory, epochs=1):
        config, labels = fixture_config(directory)
        config.update(input_size=96, batch_size=2, inference_batch_size=2)
        config["proposal"].update(min_area=5, top_k=4, padding=4)
        config["local"].update(epochs=epochs, num_negatives=1000, early_stopping={"enabled": False})
        config["local"]["opening"]["enabled"] = False
        config["matching"] = copy.deepcopy(self.config["matching"])
        suite = copy.deepcopy(self.suite)
        suite["valid_pool"].update(queries=4, candidates=5)
        suite["evaluate_every"] = 1
        return suite, config, labels

    @staticmethod
    def tokens(embedding, area):
        return dict(embedding=np.asarray(embedding, dtype=float), area=np.asarray(area),
                    position=np.zeros((len(area), 2)), scale=np.ones(len(area)))

    def test_count_direction_area_independence_threshold_and_legacy_score(self):
        settings = method_settings(self.config, "proposed")
        q = self.tokens(np.eye(3), [1, 4, 100])
        c = self.tokens([[.8, 0, .6], [0, .8, .6]], [800, 3])
        score, pairs = map_similarity(q, c, settings, details=True)
        self.assertEqual(score, 2 / 3)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(map_similarity(c, q, settings), 1)
        q["area"] = np.array([99999, 1, 1])
        self.assertEqual(map_similarity(q, c, settings), score)
        original = {**settings, "weight": "sqrt_area"}
        self.assertNotEqual(legacy_similarity(q, c, original), score)
        self.assertEqual(count_similarity(np.ones((3, 1)), settings), 1 / 3)
        self.assertEqual(count_similarity([[.9, .8], [.8, 0]], settings), .5)
        self.assertEqual(count_similarity([[0, 0]], settings), 0)
        self.assertEqual(count_similarity([[.5]], {**settings, "score_threshold": .5}), 1)
        self.assertEqual(count_similarity([[.5]], {**settings, "shape_threshold": .6}), 0)
        self.assertEqual(count_similarity(np.zeros((0, 2)), settings), 0)
        self.assertEqual(count_similarity(np.zeros((3, 0)), settings), 0)

    def test_batched_scalar_equivalence_ties_padding_and_both_representations(self):
        rng = np.random.default_rng(42)
        padded = np.array([[0, 1, -1], [2, -1, -1], [-1, -1, -1], [0, 1, 2]])
        for method in ("handcrafted", "proposed"):
            settings = method_settings(self.config, method)
            for nq in (0, 1, 3, 4):
                for raw in (rng.uniform(-1, 1, (nq, 3)), np.ones((nq, 3)), np.zeros((nq, 3))):
                    for threshold in (0, .5, 1):
                        current = {**settings, "score_threshold": threshold}
                        actual, counts = batched_map_scores(raw, padded, nq, current, return_counts=True)
                        expected = [count_similarity(raw[:, row[row >= 0]], current) for row in padded]
                        np.testing.assert_array_equal(actual, expected)
                        if nq:
                            np.testing.assert_allclose(actual, counts / nq)
            q = self.tokens([[1, 0], [0, 1], [.5, .5]], [1, 2, 9])
            c = self.tokens([[1, 0], [0, 1]], [2, 8])
            self.assertEqual(map_similarity(q, c, settings), 2 / 3)
        separate = copy.deepcopy(self.config)
        separate["matching"]["method_thresholds"]["handcrafted"]["score_threshold"] = .8
        self.assertEqual(method_settings(separate, "handcrafted")["score_threshold"], .8)
        self.assertEqual(method_settings(separate, "proposed")["score_threshold"], 0)

    def test_configuration_model_and_protected_outputs(self):
        self.assertEqual(self.config["local"]["num_negatives"], 1000)
        old_suite, old_config = ablation.load_suite(ROOT / "configs/ablation_abcd.json")
        ef_suite, ef_config = ablation_ef.load_suite(ROOT / "configs/ablation_ef.json")
        self.assertEqual(old_config["local"]["num_negatives"], 5000)
        self.assertEqual(ef_config["local"]["num_negatives"], 5000)
        self.assertEqual(self.suite["variants"]["G"], old_suite["variants"]["D"])
        seed_everything(2026)
        model = build_model(self.config, self.suite["variants"]["G"])
        self.assertEqual(sum(p.numel() for p in model.parameters()), 11235904)
        seed_everything(2026)
        self.assertEqual(parameter_hash(model), parameter_hash(ablation.build_model(old_config, old_suite["variants"]["D"])))
        for root in (old_config["output_dir"], ef_config["output_dir"], str(Path(ef_config["output_dir"]) / "G")):
            with self.assertRaises(ValueError):
                load_suite(ROOT / "configs/ablation_g.json", output_root=root)
        with tempfile.TemporaryDirectory() as directory:
            bad = copy.deepcopy(self.suite)
            bad["overrides"]["local"]["num_negatives"] = 5000
            path = Path(directory) / "bad.json"
            write_json(path, bad)
            with self.assertRaisesRegex(ValueError, "1000"):
                load_suite(path)
            prior = Path(directory) / "custom_history"
            prior.mkdir()
            write_json(prior / "suite_manifest.json", {"signature": "old"})
            with self.assertRaises(ValueError):
                load_suite(ROOT / "configs/ablation_g.json", output_root=str(prior / "G"))

    def test_cache_ranking_metrics_and_signature_guards(self):
        before = (ablation.code_hashes(), ablation_ef.code_hashes())
        with tempfile.TemporaryDirectory() as directory:
            suite, config, labels = self.fixture(directory)
            root = Path(config["output_dir"])
            with exclusive(root):
                signature = prepare_suite(suite, config)
                self.assertEqual(signature, prepare_suite(suite, config))
                result = evaluate(config, suite, "valid", root / "handcrafted/valid", rankings=True)
                pool = read_json(root / "valid_pool.json")
                protocol = SimpleNamespace(queries=pool["queries"], candidates={int(q): c for q, c in pool["candidates"].items()},
                                           labels=dict(enumerate(labels)))
                reference = evaluate_ranking(root / "handcrafted/valid/rankings.csv", protocol, config["evaluation"], 2026)
                self.assertEqual(result["metrics"], reference["metrics"])
                self.check_rankings(root / "handcrafted/valid")
                other = copy.deepcopy(config)
                other["matching"]["method_thresholds"]["proposed"]["score_threshold"] = .5
                with self.assertRaisesRegex(ValueError, "changed"):
                    prepare_suite(suite, other)
                Path(root / "cache/valid_B.npy").write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "Missing or modified"):
                    prepare_suite(suite, config)
        self.assertEqual(before, (ablation.code_hashes(), ablation_ef.code_hashes()))

    def check_rankings(self, folder):
        rows = csv_rows(folder / "rankings.csv")
        for row in rows:
            nq, count = int(row["query_regions"]), int(row["matched_pairs"])
            self.assertEqual(float(row["similarity_score"]), count / nq if nq else 0)
        for item in read_json(folder / "matches_top_k.json"):
            self.assertEqual(item["matched_pairs"], len(item["matches"]))
            self.assertAlmostEqual(item["similarity_score"], sum(m["similarity_contribution"] for m in item["matches"]))

    def test_actual_training_1000_negatives_and_exact_resume(self):
        before = (ablation.code_hashes(), ablation_ef.code_hashes())
        with tempfile.TemporaryDirectory() as directory:
            suite, config, _ = self.fixture(directory, epochs=2)
            root = Path(config["output_dir"])
            calls = []
            original_nce = training.nce_loss

            def checked_nce(anchors, positives, negatives, temperature):
                calls.append((torch.is_grad_enabled(), len(negatives)))
                return original_nce(anchors, positives, negatives, temperature)

            with exclusive(root), patch.object(training, "nce_loss", side_effect=checked_nce):
                signature = prepare_suite(suite, config)
                self.assertFalse(training.train_variant(config, suite, signature, "G", stop_after=1))
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    training.evaluate_variant(config, suite, signature, "G")
                self.assertTrue(training.train_variant(config, suite, signature, "G", resume=True))
                output = root / "seed_2026/G"
                resumed = load_checkpoint(output / "last.pt")
                self.assertNotEqual(resumed["initial_parameter_sha256"], parameter_hash({
                    k: v for k, v in resumed["model"].items() if not k.endswith("num_batches_tracked")}))
                self.assertTrue(any(grad for grad, _ in calls))
                self.assertTrue(any(not grad for grad, _ in calls))
                self.assertEqual({n for _, n in calls}, {1000})
                self.assertEqual(resumed["num_negatives"], 1000)
                checkpoint_hash = file_hash(output / "last.pt")
                (output / "training_summary.json").unlink()
                training.train_variant(config, suite, signature, "G", resume=True)
                self.assertEqual(checkpoint_hash, file_hash(output / "last.pt"))
                with self.assertRaises(FileExistsError):
                    training.train_variant(config, suite, signature, "G")
                training.evaluate_variant(config, suite, signature, "G")
                for kind in ("best_nce", "best_retrieval"):
                    self.check_rankings(output / "valid" / kind)
                self.assertFalse((output / "test").exists())
                training.evaluate_variant(config, suite, signature, "G", "test", "best_retrieval")
                self.assertEqual(checkpoint_hash, file_hash(output / "last.pt"))
                self.check_rankings(output / "test/best_retrieval")
                evaluate(config, suite, "valid", root / "handcrafted/valid", rankings=True)
                summarize(config, suite)
                rows = csv_rows(root / "seed_2026/comparison_valid.csv")
                self.assertEqual({row["variant"] for row in rows}, {"G", "handcrafted"})
            full_config = copy.deepcopy(config)
            full_config["output_dir"] = str(Path(directory) / "full")
            with exclusive(full_config["output_dir"]):
                full_signature = prepare_suite(suite, full_config)
                self.assertEqual(signature, full_signature)
                training.train_variant(full_config, suite, full_signature, "G")
                full = load_checkpoint(Path(full_config["output_dir"]) / "seed_2026/G/last.pt")
            self.assertEqual(resumed["optimizer_steps"], full["optimizer_steps"])
            self.assertEqual(resumed["scheduler"], full["scheduler"])
            torch.testing.assert_close(resumed["memory"], full["memory"], rtol=0, atol=0)
            for key in full["model"]:
                torch.testing.assert_close(resumed["model"][key], full["model"][key], rtol=0, atol=0)
        self.assertEqual(before, (ablation.code_hashes(), ablation_ef.code_hashes()))

    def test_cli_all_workers_test_and_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            suite, config, _ = self.fixture(directory)
            base = Path(directory) / "base.json"
            write_json(base, config)
            suite.update(base_config=str(base), output_root=str(Path(directory) / "cli"))
            suite["overrides"]["local"]["epochs"] = 1
            path = Path(directory) / "g.json"
            write_json(path, suite)
            command = [sys.executable, str(ROOT / "scripts/run_ablation_g.py"), "--config", str(path),
                       "--device", "cpu", "--num-workers", "1"]
            env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
            def run(*args, check=True):
                result = subprocess.run(command + list(args), cwd=ROOT, env=env, capture_output=True, text=True)
                if check:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result
            dry = json.loads(run("--dry-run").stdout)
            self.assertEqual(dry["config"]["local"]["num_negatives"], 1000)
            root = Path(suite["output_root"])
            self.assertFalse(root.exists())
            self.assertNotEqual(run("--stage", "test", check=False).returncode, 0)
            self.assertNotEqual(run("--variants", "E", check=False).returncode, 0)
            run("--stage", "all")
            output = root / "seed_2026/G"
            self.check_rankings(root / "handcrafted/valid")
            self.check_rankings(output / "valid/best_retrieval")
            self.assertFalse((output / "test").exists())
            before = file_hash(output / "last.pt")
            run("--stage", "all", "--resume")
            self.assertEqual(before, file_hash(output / "last.pt"))
            run("--stage", "test", "--variants", "G")
            self.check_rankings(root / "handcrafted/test")
            self.check_rankings(output / "test/best_retrieval")
            self.assertEqual(before, file_hash(output / "last.pt"))


if __name__ == "__main__":
    unittest.main()
