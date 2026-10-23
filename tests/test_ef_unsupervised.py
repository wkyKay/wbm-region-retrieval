"""Independent metric oracles: ties, collapse, crowding and wafer exclusions."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch

path = Path(__file__).resolve().parents[1] / "scripts/test_ef_unsupervised.py"
spec = importlib.util.spec_from_file_location("ef_unsupervised", path)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class UnsupervisedMetricsTests(unittest.TestCase):
    def score(self, x, maps=None):
        x = np.asarray(x, dtype=np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if maps is None:
            maps = np.arange(len(x))
        return evaluation.score_features(x, x, np.asarray(maps), np.arange(len(x)), 90,
                                         torch.device("cpu"), 2, .07, 1e-6, .05)

    def test_uniform_ties_match_enumerating_every_possible_rank(self):
        result = evaluation.tie_metrics([0, 2], [8, 4], [8, 9])
        self.assertAlmostEqual(result["top1"][0], 1/8)
        self.assertAlmostEqual(result["recall_at_5"][0], 5/8)
        self.assertAlmostEqual(result["mean_rank"][0], 4.5)
        self.assertAlmostEqual(result["mrr"][1], np.mean(1/np.arange(3, 7)))
        self.assertAlmostEqual(result["recall_at_5"][1], .75)

    def test_collapsed_features_are_not_perfect_retrieval(self):
        rows = self.score([[1, 0]] * 8)
        for row in rows:
            self.assertEqual(row["tie_size"], 8)
            self.assertEqual(row["top1"], .125)
            self.assertEqual(row["hard_negative_margin"], 0)
            self.assertEqual(row["crowding_fraction"], 1)
        features = np.array([[1., 0.]] * 8)
        pairs = evaluation.cross_wafer_pairs(np.arange(8), 30, 42)
        distribution = evaluation.spectrum_metrics(features, pairs)
        self.assertEqual(distribution["effective_rank_covariance"], 0)
        self.assertEqual(distribution["cross_wafer_uniformity_t2"], 0)

    def test_top1_alone_hides_crowding(self):
        separated = self.score(np.eye(3))
        crowded = self.score([[1, .0, .0], [1, .01, .0], [1, .0, .01]])
        self.assertTrue(all(row["top1"] == 1 for row in separated + crowded))
        self.assertTrue(all(row["crowding_fraction"] == 0 for row in separated))
        self.assertTrue(all(row["crowding_fraction"] == 1 for row in crowded))
        self.assertTrue(all(row["hard_negative_margin"] < .001 for row in crowded))

    def test_same_wafer_other_regions_are_excluded(self):
        rows = self.score([[1, 0], [1, 0], [0, 1]], maps=[0, 0, 1])
        self.assertEqual([r["candidate_count"] for r in rows], [2, 2, 3])
        self.assertTrue(all(row["top1"] == 1 for row in rows))
        self.assertTrue(all(row["tie_size"] == 1 for row in rows))
        self.assertAlmostEqual(rows[0]["gallery_nce"], np.log1p(np.exp(-1/.07)), places=5)

    def test_pair_sampling_and_bootstrap_are_reproducible(self):
        maps = np.array([0, 0, 1, 2])
        a, b = evaluation.cross_wafer_pairs(maps, 100, 22)
        self.assertTrue(np.all(maps[a] != maps[b]))
        c, d = evaluation.cross_wafer_pairs(maps, 100, 22)
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)
        rows = self.score(np.eye(4), maps)
        self.assertEqual(evaluation.summarize(rows, 50, 22), evaluation.summarize(rows, 50, 22))
        with self.assertRaises(ValueError):
            evaluation.cross_wafer_pairs(np.zeros(4), 20, 1)


if __name__ == "__main__":
    unittest.main()
