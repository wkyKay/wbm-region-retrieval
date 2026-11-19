"""Equivalence tests against frozen SciPy WBBS; no retrieval-protocol changes."""
from itertools import product
import unittest

import numpy as np
from scipy.spatial.distance import cdist

from fgw_fusion.wbbs_backend import Scorer, SETTINGS, original_module


def make_scorer(groups, ids=None):
    groups = [np.asarray(x, dtype=np.float64).reshape(-1, 3) for x in groups]
    ids = np.arange(len(groups), dtype=np.int64) if ids is None else np.asarray(ids)
    return Scorer(ids, np.r_[0, np.cumsum([len(x) for x in groups])],
                  np.concatenate(groups), SETTINGS)


class WbbsBackendTests(unittest.TestCase):
    def test_random_pair_scores_and_symmetry_match_original_bits(self):
        rng = np.random.default_rng(2026)
        scorer = make_scorer([rng.random((n, 3)) for n in (1, 2, 7, 39, 260, 531)])
        pairs = list(product(range(6), repeat=2))
        self.assertEqual(scorer.audit_pairs(pairs)["pairs_checked"], 36)
        for q, c in pairs:
            self.assertEqual(scorer.score_pair_ids(q, c), scorer.score_pair_ids(c, q))

    def test_duplicates_preserve_bbp_count_but_keep_original_denominator(self):
        a = np.array([[.1, .3, .7], [.2, .4, .6], [.8, .7, .2]])
        groups = [a, a[[2, 0, 2, 1, 0]], np.repeat(a[[0]], 3, axis=0)]
        scorer = make_scorer(groups)
        original = original_module()
        for q, c in product(range(3), repeat=2):
            self.assertEqual(scorer._count(q, c), len(original.best_buddies(groups[q], groups[c])))
        scorer.audit_pairs(product(range(3), repeat=2))
        self.assertEqual(scorer.score_pair_ids(2, 2), 1 / 3)
        self.assertEqual(scorer.retained_counts.tolist(), [3, 5, 3])
        self.assertEqual(scorer.unique_counts.tolist(), [3, 3, 1])

    def test_cross_block_duplicate_first_rows_and_equal_distance(self):
        # More than one original block; repeats must not introduce BBPs.
        a = np.tile([[.25, .5, .5], [.75, .5, .5]], (150, 1))
        b = np.array([[.5, .5, .5], [.5, .6, .5]])
        scorer = make_scorer([a, b])
        scorer.audit_pairs([(0, 1), (1, 0), (0, 0)])
        np.testing.assert_array_equal(scorer._nearest(b, 0), np.argmin(cdist(b, scorer.unique[0], "sqeuclidean"), axis=1))
        self.assertGreater(scorer.fallback_rows, 0)

    def test_near_ties_one_ulp_and_tiny_distances_replay_original(self):
        x = .5
        groups = [np.array([[x, x, x], [0., 0., 0.]]),
                  np.array([[np.nextafter(x, 1.), x, x], [np.nextafter(x, 0.), x, x],
                            [1e-17, 0., 0.], [2e-17, 0., 0.]])]
        scorer = make_scorer(groups)
        scorer.audit_pairs([(0, 1), (1, 0)])
        expected = np.argmin(cdist(scorer.unique[0], scorer.unique[1], "sqeuclidean"), axis=1)
        np.testing.assert_array_equal(scorer._nearest(scorer.unique[0], 1), expected)
        self.assertGreater(scorer.fallback_rows, 0)

    def test_same_distance_tie_selects_first_source_row(self):
        target = np.array([[.75, .5, .5], [.25, .5, .5], [.5, .75, .5]])
        scorer = make_scorer([[[.5, .5, .5]], target])
        self.assertEqual(int(scorer._nearest(scorer.unique[0], 1)[0]), 0)
        scorer.audit_pairs([(0, 1), (1, 0)])

    def test_unbalanced_sets_partial_reverse_equals_full_reference_both_ways(self):
        rng = np.random.default_rng(17)
        groups = [rng.random((2, 3)), rng.random((1800, 3)),
                  np.tile([[.25, .5, .75], [.75, .5, .25]], (400, 1)),
                  np.r_[np.full((1, 3), .5), rng.random((1200, 3))]]
        scorer = make_scorer(groups)
        original = original_module()
        for q, c in product(range(len(groups)), repeat=2):
            expected = len(original.best_buddies(groups[q], groups[c]))
            self.assertEqual(scorer._count(q, c), expected)
            self.assertEqual(scorer._count(c, q), expected)
            self.assertEqual(scorer.score_pair_ids(q, c), scorer.score_pair_ids(c, q))
        scorer.audit_pairs(product(range(len(groups)), repeat=2))

    def test_empty_and_singleton_sets(self):
        scorer = make_scorer([[], [[.1, .2, .3]], [[.3, .4, .5]]])
        scorer.audit_pairs(product(range(3), repeat=2))
        for i in range(3):
            self.assertEqual(scorer.score_pair_ids(0, i), 0)
            self.assertEqual(scorer.score_pair_ids(i, 0), 0)
        self.assertEqual(scorer.score_pair_ids(1, 2), 1)

    def test_count_penalty_uses_full_retained_count(self):
        scorer = make_scorer([np.tile([[.4, .5, .6]], (2, 1)),
                              np.tile([[.4, .5, .6]], (7, 1))])
        self.assertEqual(scorer.score_pair_ids(0, 1), .5 * np.exp(-5. * 5 / 2061))
        scorer.audit_pairs([(0, 1)])

    def test_scores_preserve_candidate_input_order(self):
        scorer = make_scorer([[[.2, .3, .4]], [[.3, .4, .5]],
                              [[.3, .4, .5], [.7, .8, .9]]], ids=[42, 9, 108])
        expected = [scorer.score_pair_ids(42, x) for x in (108, 9)]
        np.testing.assert_array_equal(scorer.scores(42, [108, 9]), expected)
        self.assertEqual(scorer.scores(42, np.array([], dtype=int)).shape, (0,))

    def test_fixed_settings_and_input_validation(self):
        for change in ({"mu": 2}, {"alpha": 4}, {"block_size": 64}, {"mu": True}):
            with self.assertRaises(ValueError):
                Scorer([1], [0, 1], [[.1, .2, .3]], {**SETTINGS, **change})
        for ids, offsets, desc in [([1, 1], [0, 1, 1], [[.1, .2, .3]]),
                                   ([1], [0, 2], [[.1, .2, .3]]),
                                   ([1.], [0, 1], [[.1, .2, .3]]),
                                   ([1], [0, 1], [[np.nan, .2, .3]]),
                                   ([1], [0, 1], [[1.1, .2, .3]])]:
            with self.assertRaises(ValueError):
                Scorer(ids, offsets, desc, SETTINGS)
        scorer = make_scorer([[[0., 0., 0.]], [[1., 1., 1.]]])
        for candidates in ([0], [1, 1], [7], [1.]):
            with self.assertRaises(ValueError):scorer.scores(0, candidates)

    def test_audit_summary_default_and_optional_pair_evidence(self):
        scorer = make_scorer([[[.1, .2, .3]], [[.3, .4, .5]]])
        summary = scorer.audit_pairs([(0, 1), (1, 0)])
        self.assertEqual(summary["pairs_checked"], 2)
        self.assertNotIn("pairs", summary)
        evidence = scorer.audit_pairs([(0, 1)], include_rows=True)
        self.assertEqual(evidence["pairs_checked"], 1)
        self.assertEqual(evidence["pairs"][0]["score"], 1.)

    def test_feature_storage_is_readonly_and_not_aliased(self):
        original = np.array([[.1, .2, .3], [.4, .5, .6]])
        scorer = Scorer([1, 2], [0, 1, 2], original, SETTINGS)
        original[:] = 0
        self.assertEqual(scorer.descriptors[0, 0], .1)
        with self.assertRaises(ValueError):scorer.descriptors[0, 0] = 0


if __name__ == "__main__":
    unittest.main()
