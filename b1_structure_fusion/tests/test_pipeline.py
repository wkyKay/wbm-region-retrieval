"""Synthetic full-gallery integration; never accesses formal models or data."""
from contextlib import redirect_stdout
import csv
from io import StringIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from b1_structure_fusion import descriptors, pipeline, reporting, scoring, verify
from experiments.proposed.f_segmentation_geometry import pipeline as geo
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from wbm_retrieval.retrieval_protocol import FullGalleryPool


class BaseScores:
    def __init__(self):
        self.calls = []

    def components(self, query, candidates):
        self.calls.append((int(query), tuple(map(int, candidates))))
        base = np.where(np.asarray(candidates) == 11, .8, .9)
        return {"baseline": base, "optimal": base, "f_score": base,
                "optimal_f_score": base, "global_score": base,
                "angular_distance": np.zeros(len(base)), "density_distance": np.zeros(len(base))}


def fixture(split="valid"):
    ids = np.arange(12, dtype=np.int64)
    labels = np.tile([0, 1], (12, 1))
    labels[[0, 11]] = [1, 0]
    labels[2] = [1, 1]
    classes = np.array([0 if tuple(row) == (1, 0) else 1 if tuple(row) == (0, 1) else 2 for row in labels])
    bundle = SimpleNamespace(labels=labels, class_ids=classes, valid_ids=ids, reserved_ids=ids)
    pool = FullGalleryPool([0, 1, 2], ids, split, "synthetic_protocol")
    joint = np.zeros((12, 128))
    joint[:, -1] = 1.
    joint[[0, 11], -1] = 0.
    joint[[0, 11], 0] = 1.
    values = {"sample_ids": ids, "joint128": joint, "empty": np.zeros(12, dtype=bool)}
    settings = descriptors.load_settings()
    settings["ks"] = [1, 5, 10]
    return pool, bundle, values, settings


def snapshot(folder):
    return {str(path.relative_to(folder)): (file_hash(path), path.stat().st_mtime_ns)
            for path in Path(folder).rglob("*") if path.is_file()}


class PipelineTests(unittest.TestCase):
    def test_full_gallery_selection_test_and_independent_verification(self):
        pool, bundle, values, settings = fixture()
        scorer = BaseScores()
        refs = [{"split": "valid", "metric_version": "exact_v1", "group": "all", "metric": "Precision",
                 "K": "10", "value": .5},
                {"split": "test", "metric_version": "coverage_v1", "group": "all", "metric": "MeanCoverage",
                 "K": "10", "value": .99}]
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()) as console:
            output = Path(tmp) / "b1_valid"
            reports = pipeline.evaluate(output, "validsig", scorer, pool, bundle, settings["ks"], 2026,
                                        values, settings, reference=refs)
            self.assertEqual(len(reports), 18)
            self.assertEqual(scorer.calls[0][1], tuple(range(1, 12)))
            with (output / "valid/B0/rankings.csv").open() as handle:
                baseline_ids = [int(row["candidate_id"]) for row in csv.DictReader(handle) if row["query_id"] == "0"]
            with (output / "valid/J_0.25/rankings.csv").open() as handle:
                joint_ids = [int(row["candidate_id"]) for row in csv.DictReader(handle) if row["query_id"] == "0"]
            self.assertNotIn(11, baseline_ids)
            self.assertEqual(joint_ids[0], 11)
            selection = pipeline.select(output, "validsig", settings, context={"model": "frozen"})
            self.assertEqual(selection["selections"]["J"]["selected"]["variant"], "J_0.25")
            loaded = pipeline.load_selection(output / "selection.json", settings, {"model": "frozen"})
            self.assertEqual(loaded, selection)
            self.assertIn("缺少同协议报告", console.getvalue())
            self.assertIn("NA", console.getvalue())
            self.assertIn("valid 选中摘要", console.getvalue())
            self.assertIn("WBBS / 同 split 参考", console.getvalue())
            expected_refs = reporting.reference_index(refs, "valid")
            self.assertNotIn(("coverage_v1", "all", "MeanCoverage@10"), expected_refs)
            test_pool, test_bundle, test_values, _ = fixture("test")
            selected = pipeline.test_variants(loaded, settings)
            # --stage all uses one batch root: test must not replace valid caches/index/tables.
            valid_index = file_hash(output / "evaluation_index_valid.json")
            valid_table = file_hash(output / "comparison_valid.csv")
            test_output = output
            pipeline.evaluate(test_output, "testsig", BaseScores(), test_pool, test_bundle, settings["ks"], 2026,
                              test_values, settings, variants=selected)
            reporting.print_selection_summary(test_output, loaded, "test")
            self.assertIn("test 选中摘要", console.getvalue())
            index = read_json(test_output / "evaluation_index_test.json")
            self.assertEqual(file_hash(output / "evaluation_index_valid.json"), valid_index)
            self.assertEqual(file_hash(output / "comparison_valid.csv"), valid_table)
            self.assertEqual(pipeline.select(output, "validsig", settings, {"model": "frozen"}), selection)
            with (output / "comparison.csv").open() as handle:
                self.assertEqual({row["split"] for row in csv.DictReader(handle)}, {"valid", "test"})
            identities = []
            for name in selected:
                folder = test_output / "test" / name
                check = verify.verify_evaluation(folder, test_pool, test_bundle, settings["ks"],
                                                 index["variant_signatures"][name])
                self.assertEqual(check["num_queries"], 3)
                identity = read_json(folder / "run_identity.json")
                self.assertTrue(identity["started_at"].endswith("+08:00"))
                self.assertTrue(identity["run_id"].endswith("__"+identity["key"]))
                identities.append(identity["run_id"])
            self.assertEqual(len(set(identities)), len(identities))
            with self.assertRaisesRegex(ValueError, "context differ"):
                pipeline.load_selection(output / "selection.json", settings, {"model": "different"})
            selection["selections"]["J"]["selected"]["variant"] = "J_2"
            selection["signature"] = digest({k: v for k, v in selection.items() if k != "signature"})
            write_json(output / "selection.json", selection)
            with self.assertRaisesRegex(ValueError, "prescribed valid selection"):
                pipeline.load_selection(output / "selection.json", settings)

    def test_resume_preserves_attempts_and_identity_and_reuses_completed(self):
        pool, bundle, values, settings = fixture()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            output = Path(tmp) / "run"
            with patch.object(geo, "evaluate_rankings_file", side_effect=InterruptedError("after exact")):
                with self.assertRaises(InterruptedError):
                    pipeline.evaluate(output, "sig", BaseScores(), pool, bundle, settings["ks"], 2026, values, settings)
            attempts = list((output / "valid").glob(".*attempt_*"))
            before = {str(p): snapshot(p) for p in attempts}
            identities = snapshot(output / "identities")
            pipeline.evaluate(output, "sig", BaseScores(), pool, bundle, settings["ks"], 2026, values, settings)
            self.assertEqual(before, {str(p): snapshot(p) for p in attempts})
            self.assertEqual(identities, snapshot(output / "identities"))
            completed = snapshot(output / "valid/B0")
            with patch.object(BaseScores, "components", side_effect=AssertionError("completed results must be reused")):
                pipeline.evaluate(output, "sig", BaseScores(), pool, bundle, settings["ks"], 2026, values, settings)
            self.assertEqual(completed, snapshot(output / "valid/B0"))
            with self.assertRaisesRegex(ValueError, "signature changed"):
                pipeline.evaluate(output, "different", BaseScores(), pool, bundle, settings["ks"], 2026, values, settings)

    def test_descriptor_cache_uses_original_ids_and_rejects_corruption(self):
        maps = np.ones((7, 7, 7), dtype=np.uint8)
        maps[2, 1, 1] = 2
        maps[5, 1:4, 3] = 2
        settings = descriptors.load_settings()
        original = maps.copy()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            output = Path(tmp)
            values = pipeline.cache_descriptors(output, maps, np.array([2, 5]), "desc", settings)
            np.testing.assert_array_equal(values["original_count"], [1, 3])
            np.testing.assert_array_equal(values["sample_ids"], [2, 5])
            self.assertEqual(values["moments"].shape, (2, 4))
            np.testing.assert_array_equal(maps, original)
            saved = snapshot(output / "descriptors/valid")
            with patch.object(descriptors, "describe", side_effect=AssertionError("must reuse")):
                pipeline.cache_descriptors(output, maps, np.array([2, 5]), "desc", settings)
            self.assertEqual(saved, snapshot(output / "descriptors/valid"))
            with self.assertRaisesRegex(ValueError, "IDs differ"):
                pipeline.cache_descriptors(output, maps, np.array([5, 2]), "desc", settings)
            # A separate split has its own descriptor signature and never overwrites valid.
            test_values = pipeline.cache_descriptors(output, maps, np.array([1, 3]), "testdesc", settings, split="test")
            np.testing.assert_array_equal(test_values["original_count"], [0, 0])
            self.assertEqual(saved, snapshot(output / "descriptors/valid"))
            (output / "descriptors/valid/valid.npz").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                pipeline.cache_descriptors(output, maps, np.array([2, 5]), "desc", settings)

    def test_reject_unfrozen_test_incomplete_valid_grid_and_wrong_gallery(self):
        pool, bundle, values, settings = fixture("test")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "frozen valid selection"):
                pipeline.evaluate(tmp, "sig", BaseScores(), pool, bundle, settings["ks"], 2026, values, settings)
            valid, bundle, values, settings = fixture()
            with self.assertRaisesRegex(ValueError, "complete predeclared"):
                pipeline.evaluate(tmp, "sig", BaseScores(), valid, bundle, settings["ks"], 2026, values, settings,
                                  variants={"B0": scoring.valid_variants(settings)["B0"]})
            values["sample_ids"] = values["sample_ids"][:-1]
            with self.assertRaisesRegex(ValueError, "exactly the complete gallery"):
                pipeline.evaluate(tmp, "sig", BaseScores(), valid, bundle, settings["ks"], 2026, values, settings)

    def test_independent_metrics_full_denominators_and_zero_relevance(self):
        query = [1, 1]
        candidates = np.array([[1, 1], [1, 1], [1, 0], [0, 1]])
        ranked = candidates[[0, 2]]
        actual = verify.independent_query_metrics(query, candidates, ranked, [1, 2])
        self.assertEqual(actual["exact"]["AP@2"], .5)
        self.assertEqual(actual["coverage"]["MeanCoverage@2"], .75)
        self.assertEqual(actual["coverage"]["FullCoverageRecall@2"], .5)
        none = verify.independent_query_metrics([1, 1], candidates[2:], candidates[2:], [1])
        self.assertIsNone(none["exact"]["AP@1"])
        self.assertIsNone(none["coverage"]["FullCoverageRecall@1"])


if __name__ == "__main__":
    unittest.main()
