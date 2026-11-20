import copy
import unittest

import numpy as np

from experiments.proposed.f_segmentation_geometry.scoring import pair_scores as old_pair_scores
from f_region_structure.scoring import (
    RegionPairScorer, frozen_grid, independent_distribution, independent_pair_scores,
    pair_scores, score_components,
)
from f_region_structure.selection import select_best


def sample_inputs(n=5):
    rng = np.random.default_rng(470)
    features = rng.normal(size=(n, 512)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    radial = rng.random((n, 16))
    radial /= radial.sum(axis=1, keepdims=True)
    joint = rng.random((n, 128))
    joint /= joint.sum(axis=1, keepdims=True)
    joint_all = joint.copy()
    joint[-1] = 0
    learned = {}
    for family in ("plain_mlp", "gated_mlp"):
        z = rng.normal(size=(n, 128)).astype(np.float32)
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        learned[family] = z
    return features, {"histograms": radial, "relative_area": np.linspace(.01, .3, n)}, {
        "joint": joint, "joint_all": joint_all}, learned


class ScoreTests(unittest.TestCase):
    def test_frozen_grid_is_exact_and_detached(self):
        a = frozen_grid()
        self.assertEqual(len(a), 25)
        self.assertEqual(sum(s["family"] == "add_j" for s in a.values()), 6)
        self.assertEqual(sum(s["family"] == "replace_radial" for s in a.values()), 7)
        self.assertEqual(sum(s["family"] == "concat" for s in a.values()), 9)
        a["baseline"]["area"] = False
        self.assertTrue(frozen_grid()["baseline"]["area"])

    def test_baseline_bitwise_matches_frozen_source(self):
        h, geo, desc, _ = sample_inputs()
        q = np.array([2, 0], dtype=np.int64)
        expected = old_pair_scores(h[q], h, geo["histograms"][q], geo["histograms"],
                                   geo["relative_area"][q], geo["relative_area"])
        actual = pair_scores(q, h, geo, desc, frozen_grid()["baseline"])
        np.testing.assert_array_equal(actual, expected)

    def test_all_variants_match_independent_scalar_formulas(self):
        h, geo, desc, learned = sample_inputs()
        scorer = RegionPairScorer(h, geo, desc, learned)
        q = np.array([0, 4])
        shared = scorer.components(q)
        for name, spec in frozen_grid().items():
            with self.subTest(name=name):
                actual = score_components(shared, spec)
                expected = independent_pair_scores(q, h, geo, desc, spec, learned)
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
                self.assertEqual(actual.shape, (2, len(h)))
                self.assertTrue(np.isfinite(actual).all())
                self.assertTrue(np.all((actual >= 0) & (actual <= 1+1e-7)))

    def test_concat_is_the_explicit_embedding_kernel(self):
        h, geo, desc, _ = sample_inputs()
        spec = frozen_grid()["concat_all_points_w0p25"]
        e = np.concatenate((np.sqrt(.75)*h.astype(np.float64),
                            np.sqrt(.25)*np.sqrt(desc["joint_all"])), axis=1)
        expected = np.clip(np.einsum("ik,jk->ij", e[[1, 3]], e), 0, 1)
        area_dist = .5*np.abs(np.log(geo["relative_area"][[1, 3]])[:, None]
                              - np.log(geo["relative_area"])[None])
        expected *= .85+.15*np.exp(-area_dist/1.5)
        actual = pair_scores(np.array([1, 3]), h, geo, desc, spec)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_empty_structure_is_missing_not_a_similarity_bonus(self):
        h, geo, desc, _ = sample_inputs()
        desc["joint"][:] = 0
        q = np.array([0, 3])
        scorer = RegionPairScorer(h, geo, desc)
        c = scorer.components(q)
        grid = frozen_grid()
        for name, spec in grid.items():
            if spec.get("descriptor") == "retained":
                expected = c["baseline"] if spec["family"] == "add_j" else c["cosine"]*c["area"]
                np.testing.assert_array_equal(score_components(c, spec), expected)
        self.assertEqual(float(c["structure"]["retained"]["valid"].sum()), 0)

    def test_joint_and_independent_distinguish_association_only(self):
        h, geo, desc, _ = sample_inputs(2)
        h[:] = h[0]
        geo["relative_area"][:] = .1
        p = np.zeros((2, 8, 4, 4))
        p[0, 0, 0, 0] = p[0, 7, 3, 3] = .5
        p[1, 0, 3, 3] = p[1, 7, 0, 0] = .5
        desc["joint_all"] = p.reshape(2, 128)
        independent = independent_distribution(desc["joint_all"])
        np.testing.assert_array_equal(independent[0], independent[1])
        self.assertEqual(independent.shape, (2, 128))
        q = np.array([0])
        j = pair_scores(q, h, geo, desc, frozen_grid()["replace_radial_all_points_l1p0"])
        i = pair_scores(q, h, geo, desc, frozen_grid()["replace_radial_independent_all_points_l1p0"])
        self.assertAlmostEqual(j[0, 1], np.exp(-1), places=6)
        self.assertAlmostEqual(i[0, 1], 1., places=6)

    def test_area_deletion_changes_only_area_factor(self):
        h, geo, desc, _ = sample_inputs()
        q = np.array([0, 1])
        c = RegionPairScorer(h, geo, desc).components(q)
        grid = frozen_grid()
        for w in ("0p25", "0p5", "0p75"):
            with_area = score_components(c, grid["concat_all_points_w"+w])
            no_area = score_components(c, grid["concat_all_points_no_area_w"+w])
            np.testing.assert_array_equal(with_area, no_area*c["area"])

    def test_test_subset_skips_unused_components_without_changing_scores(self):
        h, geo, desc, learned = sample_inputs()
        scorer = RegionPairScorer(h, geo, desc, learned)
        q = np.array([0, 4])
        full = scorer.components(q)
        for name, spec in frozen_grid().items():
            restricted = scorer.components(q, {name: spec})
            np.testing.assert_array_equal(score_components(full, spec), score_components(restricted, spec))
            if spec["family"] in ("baseline", "plain_mlp", "gated_mlp"):
                self.assertEqual(restricted["structure"], {})

    def test_learned_features_are_required_only_for_learned_scores(self):
        h, geo, desc, learned = sample_inputs()
        q = np.array([1])
        with self.assertRaisesRegex(ValueError, "Missing frozen"):
            pair_scores(q, h, geo, desc, frozen_grid()["plain_mlp"])
        actual = pair_scores(q, h, geo, desc, frozen_grid()["plain_mlp"], learned["plain_mlp"])
        np.testing.assert_array_equal(actual, np.clip(learned["plain_mlp"][q] @ learned["plain_mlp"].T, 0, 1))

    def test_rejects_invalid_inputs_or_modified_recipe(self):
        h, geo, desc, _ = sample_inputs()
        with self.assertRaises(ValueError):
            pair_scores(np.array([1, 1]), h, geo, desc, frozen_grid()["baseline"])
        bad = copy.deepcopy(desc)
        bad["joint"][0] *= 2
        with self.assertRaises(ValueError):
            RegionPairScorer(h, geo, bad)
        bad_geo = copy.deepcopy(geo)
        bad_geo["relative_area"][0] = 0
        with self.assertRaises(ValueError):
            RegionPairScorer(h, bad_geo, desc)
        bad_spec = copy.deepcopy(frozen_grid()["concat_all_points_w0p25"])
        bad_spec["structure_weight"] = .9
        with self.assertRaises(ValueError):
            pair_scores(np.array([0]), h, geo, desc, bad_spec)

    def test_no_regions_produces_empty_query_matrices(self):
        h, geo, desc, _ = sample_inputs()
        value = pair_scores(np.array([], dtype=np.int64), h, geo, desc, frozen_grid()["baseline"])
        self.assertEqual(value.shape, (0, 5))


class SelectionTests(unittest.TestCase):
    def rows(self, hits=1000):
        return {name: {"top10_hits": hits, "num_queries": 500, "split": "valid"} for name in frozen_grid()}

    def test_baseline_is_eligible_and_wins_exact_ties(self):
        selected = select_best(self.rows())
        self.assertEqual(selected["winner"], "baseline")
        self.assertFalse(selected["synthetic"])
        self.assertEqual(len(selected["ranking"]), 25)

    def test_single_integer_hit_beats_any_secondary_metric(self):
        rows = self.rows()
        rows["gated_mlp"].update(top10_hits=1001, map_at_100=0.)
        rows["baseline"]["map_at_100"] = 1.
        selected = select_best(rows)
        self.assertEqual(selected["winner"], "gated_mlp")
        self.assertEqual(selected["ranking"][0]["top10_hits"], 1001)

    def test_nonlearned_wins_tie_over_learned(self):
        rows = self.rows()
        for name in ("plain_mlp", "concat_all_points_w0p25"):
            rows[name]["top10_hits"] = 1001
        self.assertEqual(select_best(rows)["winner"], "concat_all_points_w0p25")

    def test_fewer_branches_then_lower_weight_break_ties(self):
        rows = self.rows()
        for name in ("concat_all_points_w0p25", "concat_all_points_no_area_w0p5",
                     "concat_all_points_no_area_w0p25"):
            rows[name]["top10_hits"] = 1001
        self.assertEqual(select_best(rows)["winner"], "concat_all_points_no_area_w0p25")

    def test_refuses_partial_grid_test_and_fractional_hits(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            select_best(rows, split="test")
        del rows["baseline"]
        with self.assertRaises(ValueError):
            select_best(rows)
        for wrong in (1000., True, -1, 5001):
            rows = self.rows()
            rows["baseline"]["top10_hits"] = wrong
            with self.assertRaises(ValueError):
                select_best(rows)
        rows = self.rows()
        rows["baseline"]["split"] = "test"
        with self.assertRaises(ValueError):
            select_best(rows)
        rows = self.rows()
        rows["baseline"]["num_queries"] = 499
        with self.assertRaises(ValueError):
            select_best(rows)


if __name__ == "__main__":
    unittest.main()
