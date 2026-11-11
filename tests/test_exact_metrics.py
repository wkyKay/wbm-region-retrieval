"""Exact-combination formulas, fixed-pool denominators and cross-project parity."""

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from wbm_retrieval.common import ROOT, csv_rows, read_json
from wbm_retrieval import exact_metrics
from wbm_retrieval.retrieval_protocol import FullGalleryPool


REFERENCE = ROOT.parent / "compared_methods/shared/exact_metrics.py"


class ExactMetricTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        classes = np.array([0, 0, 0, 1, 1, 2, 2, 3])
        combinations = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
        self.bundle = SimpleNamespace(labels=combinations[classes], class_ids=classes,
                                      valid_ids=np.arange(8), reserved_ids=np.arange(8))
        self.pool = FullGalleryPool([0], np.arange(8), "valid", "frozen-test-pool")

    def tearDown(self):
        self.temporary.cleanup()

    def evaluate(self, rankings, ks=(1, 2, 3), pool=None, name="output", **kwargs):
        return exact_metrics.evaluate_rankings(self.bundle, self.pool if pool is None else pool,
                                              rankings, ks, self.root / name, **kwargs)

    def test_hand_computed_formulas_use_all_relevant_candidates(self):
        result = self.evaluate(iter([(0, [1, 3, 2])]))
        expected = {"Hit@1": 1., "Precision@1": 1., "Recall@1": .5, "mAP@1": .5,
                    "Hit@2": 1., "Precision@2": .5, "Recall@2": .5, "mAP@2": .5,
                    "Hit@3": 1., "Precision@3": 2/3, "Recall@3": 1., "mAP@3": 5/6}
        for field, value in expected.items():
            self.assertAlmostEqual(result["metrics"][field], value)
        self.assertEqual(result["retrieval_protocol_signature"], "frozen-test-pool")
        row = csv_rows(self.root / "output/per_query.csv")[0]
        self.assertEqual(row["relevant_in_pool"], "2")
        self.assertAlmostEqual(float(row["AP@3"]), 5/6)
        self.assertEqual(read_json(self.root / "output/metrics.json"), result)

    def test_extra_labels_partial_overlap_and_short_returns_are_not_extra_positives(self):
        # A+ B at rank 1 is not relevant to an A-only query; the omitted second
        # A-only candidate still contributes to R even after screening.
        result = self.evaluate({0: [5, 1]}, ks=(1, 3))
        self.assertEqual(result["metrics"]["Hit@1"], 0.)
        self.assertEqual(result["metrics"]["Precision@3"], 1/3)
        self.assertEqual(result["metrics"]["Recall@3"], .5)
        self.assertEqual(result["metrics"]["mAP@3"], .25)
        reverse = self.evaluate({5: [0, 6]}, (1, 2), pool={5: [0, 1, 6]}, name="reverse")
        self.assertEqual(reverse["metrics"]["Hit@1"], 0.)
        self.assertEqual(reverse["metrics"]["mAP@2"], .5)
        empty = self.evaluate({0: []}, name="empty")
        self.assertTrue(all(value == 0 for value in empty["metrics"].values()))

    def test_zero_relevant_queries_keep_hit_precision_but_not_recall_ap(self):
        result = self.evaluate({0: [1], 7: [0]}, (1,), pool={0: [1, 2, 7], 7: [0, 1, 2]})
        self.assertEqual(result["queries_without_relevant_candidates"], 1)
        self.assertEqual(result["metrics"]["Hit@1"], .5)
        self.assertEqual(result["metrics"]["Precision@1"], .5)
        self.assertEqual(result["metrics"]["Recall@1"], .5)
        self.assertEqual(result["metrics"]["mAP@1"], .5)
        self.assertEqual(result["metric_query_counts"]["mAP@1"], 1)
        self.assertEqual(result["class_macro_metric_class_counts"]["mAP@1"], 1)
        singleton = self.evaluate({7: []}, (1,), pool={7: [0, 1]}, name="singleton")
        self.assertIsNone(singleton["metrics"]["mAP@1"])
        self.assertIsNone(singleton["class_macro_metrics"]["Recall@1"])

    def test_query_macro_and_class_macro_have_distinct_weights(self):
        result = self.evaluate({0: [1], 1: [0], 5: [3]}, (1,),
                               pool={0: [1, 2, 3], 1: [0, 2, 3], 5: [0, 3, 6]})
        self.assertAlmostEqual(result["metrics"]["Hit@1"], 2/3)
        self.assertEqual(result["class_macro_metrics"]["Hit@1"], .5)
        self.assertAlmostEqual(result["metrics"]["mAP@1"], 1/3)
        self.assertEqual(result["class_macro_metrics"]["mAP@1"], .25)
        self.assertEqual(result["groups"]["single_label"]["num_queries"], 2)
        self.assertEqual(result["groups"]["mixed_label"]["metrics"]["Hit@1"], 0.)
        self.assertEqual(set(result["groups"]["by_label_count"]), {"1", "2"})

    def test_stream_coverage_self_duplicate_wrong_split_and_output_guards(self):
        for stream in ([], [(0, [1]), (0, [2])], [(1, [2])], [(0, [0])], [(0, [1, 1])], [(0, [8])]):
            with self.subTest(stream=stream), self.assertRaises(ValueError):
                self.evaluate(iter(stream))
        self.assertFalse((self.root / "output").exists())
        with self.assertRaises(ValueError):
            self.evaluate({0: [1]}, split_name="test")
        with self.assertRaisesRegex(ValueError, "K cannot exceed"):
            self.evaluate({0: [1]}, ks=[8])
        self.evaluate({0: [1]})
        original = (self.root / "output/metrics.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.evaluate({0: [2]})
        self.assertEqual((self.root / "output/metrics.json").read_bytes(), original)

    @unittest.skipUnless(REFERENCE.is_file(), "compared_methods sibling is not installed")
    def test_compared_methods_formula_outputs_and_source_are_identical(self):
        spec = importlib.util.spec_from_file_location("compared_exact_reference", REFERENCE)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        self.assertEqual(Path(exact_metrics.__file__).read_bytes(), REFERENCE.read_bytes())
        pool = FullGalleryPool([0, 5, 7], np.arange(8), "test", "shared-frozen-signature")
        rankings = [(0, [5, 1, 3]), (5, [0, 6]), (7, [])]
        actual = self.evaluate(iter(rankings), pool=pool, split_name="test")
        expected = reference.evaluate_rankings(self.bundle, pool, iter(rankings), [1, 2, 3],
                                               self.root / "reference", split_name="test")
        self.assertEqual(actual, expected)
        for filename in ("metrics.json", "per_query.csv", "per_class.csv"):
            self.assertEqual((self.root / "output" / filename).read_bytes(),
                             (self.root / "reference" / filename).read_bytes())

    def test_sha256_tie_policy_is_label_independent_and_permutation_invariant(self):
        ids = np.array([10, 4, 200, 1])
        expected = np.array([int.from_bytes(hashlib.sha256(
            f"exact_v1:tie:2026:9:{i}".encode("ascii")).digest()[:8], "big") for i in ids], dtype=np.uint64)
        np.testing.assert_array_equal(exact_metrics.tie_keys(9, ids), expected)
        order = exact_metrics.order_scores(ids, np.ones(4), 9)
        np.testing.assert_array_equal(ids[order], ids[np.lexsort((ids, expected))])
        permutation = np.array([2, 0, 3, 1])
        np.testing.assert_array_equal(ids[order], ids[permutation][
            exact_metrics.order_scores(ids[permutation], np.ones(4), 9)])
        np.testing.assert_array_equal(exact_metrics.order_scores([1, 2, 3], [.5, .5, -np.inf], 9,
                                                               secondary=[2., 1., 0.]), [1, 0, 2])


if __name__ == "__main__":
    unittest.main()
