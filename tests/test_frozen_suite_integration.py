"""Consume the deduplicated full-gallery protocol through actual experiment suites."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from test_deduplicated_protocol import fixture
from experiments.proposed._shared import abcd_suite, ef_suite
from experiments.proposed._shared import ef_training
from experiments.proposed.g import suite as g_suite
from experiments.supervised.b import suite as supervised_suite
from wbm_retrieval.common import ROOT, csv_rows, file_hash, read_json, write_json
from wbm_retrieval.data import Protocol, decode_pool
from wbm_retrieval.exact_metrics import evaluate_rankings
from wbm_retrieval.models import Projector
from wbm_retrieval.retrieval_protocol import FullGalleryPool


class FrozenSuiteIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_abcd_ef_g_prepare_and_score_identical_frozen_valid_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, _, _, _, splits = fixture(root)
            base.update(input_size=96, output_dir=str(root / "base_output"))
            write_json(root / "base.json", base)
            protected = [Path(base["data_file"]), *list((root / "deduplicated").iterdir()),
                         *list((root / "retrieval").iterdir())]
            before = {p: file_hash(p) for p in protected}
            expected = Protocol(base).retrieval.valid
            payloads = []
            for name, module, config_path in (
                ("abcd", abcd_suite, "experiments/proposed/a/config.json"),
                ("ef", ef_suite, "experiments/proposed/e/config.json"),
                ("g", g_suite, "experiments/proposed/g/config.json"),
            ):
                with self.subTest(suite=name):
                    settings = copy.deepcopy(read_json(ROOT / config_path))
                    settings.update(base_config=str(root / "base.json"), output_root=str(root / name),
                                    selection_metric="ContainmentNDCG@10",
                                    valid_pool={"seed": 2026, "queries": 6, "candidates": 8})
                    write_json(root / f"{name}.json", settings)
                    suite, config = module.load_suite(root / f"{name}.json", device="cpu", workers=0)
                    with module.exclusive(config["output_dir"]):
                        module.prepare_suite(suite, config)
                        folder = Path(config["output_dir"])
                        payload = read_json(folder / "valid_pool.json")
                        self.assertNotIn("candidates", payload)
                        payloads.append(payload)
                        queries, pool = decode_pool(payload)
                        self.assertIsInstance(pool, FullGalleryPool)
                        self.assertEqual(queries, list(expected))
                        for split, ids in splits.items():
                            with np.load(folder / "cache" / f"{split}_regions.npz") as cache:
                                np.testing.assert_array_equal(cache["map_ids"], ids)
                        result = module.evaluate(config, suite, "valid", folder / "handcrafted/valid", rankings=True)
                        self.assertEqual(result["num_queries"], 6)
                        rows = csv_rows(folder / "handcrafted/valid/rankings.csv")
                        self.assertEqual(len(rows), 6 * 8)
                        for q in queries:
                            actual = [int(row["candidate_id"]) for row in rows if int(row["query_id"]) == q]
                            self.assertEqual(set(actual), set(expected[q]))
                            self.assertEqual(len(actual), len(set(actual)))
                            self.assertNotIn(q, actual)
                        self.assertFalse((folder / "handcrafted/test").exists())
            self.assertEqual(payloads[0], payloads[1])
            self.assertEqual(payloads[0], payloads[2])
            self.assertEqual(before, {p: file_hash(p) for p in protected})

    def test_supervised_preparation_keeps_compact_frozen_pool_and_unique_train_budgets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, _, _, _, splits = fixture(root)
            base.update(input_size=96, output_dir=str(root / "base_output"))
            write_json(root / "base.json", base)
            settings = read_json(ROOT / "experiments/supervised/b/config.json")
            settings.update(base_config=str(root / "base.json"), output_root=str(root / "supervised"),
                            valid_pool={"seed": 2026, "queries": 6, "candidates": 8},
                            budgets=[.5], device="cpu", num_workers=0)
            settings["evaluation"].pop("metric_version", None)
            settings["evaluation"]["ks"] = [1, 5, 10, 20]
            write_json(root / "suite.json", settings)
            config = supervised_suite.load_suite(root / "suite.json")
            self.assertNotIn("candidate_manifest", config)
            self.assertNotIn("query_manifest", config)
            with supervised_suite.exclusive(config["output_root"]):
                _, protocol = supervised_suite.prepare(config)
            folder = Path(config["output_root"])
            valid = supervised_suite.split_protocol(config, protocol, "valid")
            self.assertIsInstance(valid.candidates, FullGalleryPool)
            self.assertEqual(valid.queries, list(protocol.retrieval.valid))
            for q in valid.queries:
                np.testing.assert_array_equal(valid.candidates[q], protocol.retrieval.valid[q])
            self.assertTrue((folder / "protocol/valid_gallery.csv").is_file())
            self.assertFalse((folder / "protocol/valid_candidates.csv").exists())
            selected = {int(row["sample_id"]) for row in csv_rows(folder / "supervised_50pct/labeled_train_ids.csv")}
            self.assertTrue(selected)
            self.assertLessEqual(selected, set(splits["train"]))
            self.assertFalse(selected & set(splits["valid"]))
            self.assertFalse(selected & set(splits["test"]))
            self.assertIs(supervised_suite.split_protocol(config, protocol, "test"), protocol)
            self.assertFalse((folder / "supervised_50pct/models").exists())

    def test_exact_f_prepare_train_resume_and_full_gallery_test(self):
        """Real data, NCE, optimizer and evaluator with a cheap encoder for CPU CI."""
        class TinyEncoder(torch.nn.Module):
            def __init__(self, channels, dimension):
                super().__init__()
                self.linear = torch.nn.Linear(channels, dimension)

            def forward(self, inputs):
                return torch.nn.functional.normalize(self.linear(inputs.mean((-2, -1))), dim=1)

        def tiny_model(config, spec):
            return torch.nn.ModuleDict({
                "encoder": TinyEncoder(spec["input_channels"], config["embedding_dim"]),
                "projector": Projector(config["embedding_dim"], config["local"]["projector_dim"]),
            })

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, _, labels, _, splits = fixture(root, per_class=60)
            base.update(input_size=96, output_dir=str(root / "base_output"), batch_size=16,
                        inference_batch_size=32)
            base["evaluation"].update(metric_version="exact_v1", ks=[1, 5, 10], tie_seed=2026,
                                      bootstrap_samples=0)
            base["local"]["num_negatives"] = 8
            write_json(root / "base.json", base)
            settings = read_json(ROOT / "experiments/proposed/f/config.json")
            settings.update(base_config=str(root / "base.json"), output_root=str(root / "exact_f"),
                            selection_metric="mAP@10", evaluate_every=1,
                            valid_pool={"seed": 2026, "queries": 6, "candidates": 17})
            settings["overrides"]["local"]["epochs"] = 1
            write_json(root / "suite.json", settings)
            suite, config = ef_suite.load_suite(root / "suite.json", device="cpu", workers=0)
            protected = [Path(base["data_file"]), *list((root / "deduplicated").iterdir()),
                         *list((root / "retrieval").iterdir())]
            before = {p: file_hash(p) for p in protected}
            output = Path(config["output_dir"])
            with ef_suite.exclusive(output), patch.object(ef_training, "build_model", tiny_model):
                signature = ef_suite.prepare_suite(suite, config)
                self.assertEqual(signature, ef_suite.prepare_suite(suite, config))
                baseline = ef_suite.evaluate(config, suite, "valid", output / "handcrafted/valid", rankings=True)
                self.assertEqual(baseline["metric_version"], "exact_v1")
                self.assertEqual(baseline["num_queries"], 6)
                run = output / "seed_2026/F"
                # Crash after random evaluation, before the first optimization
                # step: a restart must retain the completed immutable report.
                with patch.object(ef_training.MemoryBank, "initialize", side_effect=InterruptedError("after random")):
                    with self.assertRaisesRegex(InterruptedError, "after random"):
                        ef_training.train_variant(config, suite, signature, "F")
                self.assertFalse((run / "last.pt").exists())
                random_files = {p: file_hash(p) for p in (run / "valid/random").rglob("*") if p.is_file()}
                self.assertTrue(random_files)
                self.assertFalse((run / "valid/epoch_001").exists())
                # Crash in the second critical window: the epoch report is
                # committed but its checkpoint is not. Retry must recompute in
                # a fresh attempt directory and reproduce the same scores.
                with patch.object(ef_training, "save_checkpoint", side_effect=InterruptedError("before checkpoint")):
                    with self.assertRaisesRegex(InterruptedError, "before checkpoint"):
                        ef_training.train_variant(config, suite, signature, "F", resume=True)
                self.assertFalse((run / "last.pt").exists())
                epoch_files = {p: file_hash(p) for p in (run / "valid/epoch_001").rglob("*") if p.is_file()}
                self.assertTrue(epoch_files)
                self.assertTrue(ef_training.train_variant(config, suite, signature, "F", resume=True))
                self.assertEqual(random_files, {p: file_hash(p) for p in random_files})
                self.assertEqual(epoch_files, {p: file_hash(p) for p in epoch_files})
                for name, expected_reports in (("random", 3), ("epoch_001", 2)):
                    attempts = sorted((run / "valid").glob(f"{name}*/metrics.json"))
                    self.assertEqual(len(attempts), expected_reports)
                    reference_metrics = read_json(attempts[0])["metrics"]
                    self.assertTrue(all(read_json(path)["metrics"] == reference_metrics for path in attempts))
                summary = read_json(run / "training_summary.json")
                self.assertGreater(summary["optimizer_steps"], 0)
                self.assertNotEqual(summary["initial_parameter_sha256"], summary["final_parameter_sha256"])
                self.assertIn("mAP@10", read_json(run / "history.json")[0])
                self.assertEqual(read_json(run / "run_manifest.json")["default_test_checkpoint"], "best_nce")
                checkpoint_hash = file_hash(run / "last.pt")
                (run / "training_summary.json").unlink()
                self.assertTrue(ef_training.train_variant(config, suite, signature, "F", resume=True))
                self.assertEqual(file_hash(run / "last.pt"), checkpoint_hash)
                ef_training.evaluate_variant(config, suite, signature, "F", "valid", "best_nce")
                self.assertFalse((run / "test").exists())
                ef_training.evaluate_variant(config, suite, signature, "F", "test", "best_nce")
                self.assertEqual(file_hash(run / "last.pt"), checkpoint_hash)
                protocol = Protocol(config)
                classes = np.unique(labels, axis=0, return_inverse=True)[1]
                bundle = SimpleNamespace(labels=labels, class_ids=classes,
                                         valid_ids=splits["valid"], reserved_ids=splits["test"])
                for split, query_count, relevant in (("valid", 6, 5), ("test", 36, 11)):
                    target = run / split / "best_nce"
                    report = read_json(target / "metrics.json")
                    self.assertEqual(report["metric_version"], "exact_v1")
                    self.assertEqual(report["num_queries"], query_count)
                    self.assertEqual(report["test_evaluated"], split == "test")
                    self.assertEqual(set(report["metrics"]),
                                     {f"{metric}@{k}" for metric in ("Hit", "Precision", "Recall", "mAP")
                                      for k in (1, 5, 10)})
                    rows = csv_rows(target / "rankings.csv")
                    self.assertEqual(len(rows), query_count * 10)
                    pool = getattr(protocol.retrieval, split)
                    rankings = {q: [int(row["candidate_id"]) for row in rows if int(row["query_id"]) == q]
                                for q in pool}
                    reference = evaluate_rankings(bundle, pool, rankings, [1, 5, 10],
                                                  root / f"reference_{split}", split_name=split)
                    self.assertEqual(report["metrics"], reference["metrics"])
                    self.assertTrue(all(int(row["relevant_in_pool"]) == relevant
                                        for row in csv_rows(target / "per_query.csv")))
                    self.assertTrue(all("AP@10" in row for row in read_json(target / "per_query.json")))
                ef_suite.summarize(config, suite)
                paired = read_json(output / "seed_2026/paired_valid.json")["comparisons"]
                self.assertEqual({row["metric"] for row in paired},
                                 {"Hit@10", "Precision@10", "Recall@10", "mAP@10"})
            self.assertEqual(before, {p: file_hash(p) for p in protected})


if __name__ == "__main__":
    unittest.main()
