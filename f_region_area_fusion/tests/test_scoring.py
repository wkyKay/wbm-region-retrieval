import copy
import math
import unittest
from unittest import mock

import numpy as np

from f_region_score_search.scoring import (
    OLD_SPEC, frozen_grid as prior_grid, score_components as prior_score,
)
from f_region_structure.scoring import RegionPairScorer
from f_region_area_fusion.scoring import (
    BASELINE, AreaPairScorer, checked_spec, entropy_weights, frozen_grid,
    independent_components, independent_pair_scores, independent_score_components,
    score_components,
)


def fixture():
    rng = np.random.default_rng(2026)
    h = rng.normal(size=(6, 512)).astype(np.float32)
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    h[1], h[2], h[5] = h[0], -h[0], 0.
    p = rng.random((6, 128))
    p /= p.sum(axis=1, keepdims=True)
    radial = rng.random((6, 16))
    radial /= radial.sum(axis=1, keepdims=True)
    data = {"histograms": radial, "relative_area": np.array([.001, .002, .01, .1, .5, 1.]),
            "owners": np.array([10, 10, 20, 21, 21, 30]), "joint_all": p, "joint": p.copy()}
    data["joint"][4:] = 0.  # Unused retained descriptor can be missing.
    return h, data, np.array([0, 2, 5], dtype=np.int64)


class ScoringTests(unittest.TestCase):
    def test_frozen_grid_has_29_unique_specs_and_three_deduplicated_zero_beta(self):
        grid = frozen_grid()
        self.assertEqual(len(grid), 29)
        self.assertEqual(list(grid)[:3], ["no_area_a0p4", "no_area_a0p5", "no_area_a0p6"])
        self.assertIn(BASELINE, grid)
        for family, count in (("no_area", 3), ("add", 12), ("mul", 12), ("equal3", 1), ("entropy", 1)):
            self.assertEqual(sum(s["family"] == family for s in grid.values()), count)
        for family in ("add", "mul"):
            self.assertEqual({(s["alpha"], s["beta"]) for s in grid.values() if s["family"] == family},
                             {(a, b) for a in (.4, .5, .6) for b in (.05, .1, .15, .25)})
        self.assertEqual(sum(s.get("beta") == 0 for s in grid.values()), 3)
        for spec in grid.values():
            self.assertEqual(checked_spec(spec), spec)
        grid[BASELINE]["beta"] = 0.
        self.assertEqual(frozen_grid()[BASELINE]["beta"], .15)

    def test_current_baseline_is_byte_identical_to_old_equal_addition(self):
        h, data, q = fixture()
        old_parts = RegionPairScorer(h, data, data).components(q, {"selected": OLD_SPEC})
        parts = AreaPairScorer(h, data, data).components(q)
        expected = prior_score(old_parts, prior_grid()["mixed_a0p5_e1p0"])
        actual = score_components(parts, frozen_grid()[BASELINE])
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.tobytes(), expected.tobytes())
        np.testing.assert_array_equal(parts["cosine"], old_parts["cosine"])
        np.testing.assert_array_equal(parts["structure_similarity"],
                                      np.exp(-.5 * old_parts["structure"]["all_points"]["distance"]))

    def test_all_29_match_independent_components_and_scalar_formulas(self):
        h, data, q = fixture()
        engine = AreaPairScorer(h, data, data)
        parts = engine.components(q)
        with mock.patch("f_region_area_fusion.scoring.entropy_weights", side_effect=AssertionError), \
             mock.patch("f_region_area_fusion.scoring.score_components", side_effect=AssertionError):
            reference = independent_components(q, h, data["relative_area"], data["joint_all"], data["owners"])
            expected = {name: independent_score_components(reference, spec)
                        for name, spec in frozen_grid().items()}
        for name, spec in frozen_grid().items():
            with self.subTest(name=name):
                actual = engine.score_components(parts, spec)
                np.testing.assert_allclose(actual, expected[name], atol=2e-7, rtol=1e-6)
                self.assertTrue(np.all((actual >= 0.) & (actual <= 1.)))
                self.assertEqual(actual.shape, (3, 6))

    def test_raw_area_similarity_is_not_recovered_from_rounded_B(self):
        h, data, q = fixture()
        parts = AreaPairScorer(h, data, data).components(q)
        r = data["relative_area"]
        expected = np.exp(-(.5 * np.abs(np.log(r[q])[:, None] - np.log(r)[None])) / 1.5)
        np.testing.assert_array_equal(parts["area_similarity"], expected)
        self.assertFalse(np.array_equal((parts["area"] - .85) / .15, expected))

    def test_closed_form_families_and_same_beta_area_sensitivity(self):
        parts = {"cosine": np.array([[.2, .2]]), "structure_similarity": np.array([[.8, .8]]),
                 "area_similarity": np.array([[.4, .6]]), "entropy_weights": np.array([[.2, .3, .5]])}
        grid = frozen_grid()
        m = .5
        np.testing.assert_allclose(score_components(parts, grid["no_area_a0p5"]), m)
        np.testing.assert_allclose(score_components(parts, grid["add_a0p5_b0p1"]), [[.49, .51]])
        np.testing.assert_allclose(score_components(parts, grid["mul_a0p5_b0p1"]), [[.47, .48]])
        np.testing.assert_allclose(score_components(parts, grid["equal3"]), [[1.4 / 3., 1.6 / 3.]])
        np.testing.assert_allclose(score_components(parts, grid["entropy"]), [[.48, .58]])
        self.assertAlmostEqual(np.diff(score_components(parts, grid["add_a0p5_b0p1"])).item(), .1 * .2)
        self.assertAlmostEqual(np.diff(score_components(parts, grid["mul_a0p5_b0p1"])).item(), .1 * m * .2)

    def test_entropy_known_distribution_constant_and_zero_channel(self):
        one = np.array([[1., 0., 0., 0.]])
        constant = np.full((1, 4), .3)
        zero = np.zeros((1, 4))
        mask = np.ones((1, 4), dtype=bool)
        # H(one)=0, H(constant)=log(4), exp(-H) ratio=4:1, zero mass is inactive.
        np.testing.assert_allclose(entropy_weights((one, constant, zero), mask), [[.8, .2, 0.]])
        np.testing.assert_allclose(entropy_weights((constant, constant / 2., constant * 2.), mask), 1. / 3.)
        np.testing.assert_array_equal(entropy_weights((zero, zero, zero), mask), np.full((1, 3), 1. / 3.))
        np.testing.assert_array_equal(entropy_weights((one, constant, zero), ~mask), np.full((1, 3), 1. / 3.))

    def test_entropy_excludes_all_same_map_tokens_without_excluding_other_queries(self):
        h, data, q = fixture()
        parts = AreaPairScorer(h, data, data).components(q)
        mask = data["owners"][q, None] != data["owners"][None]
        self.assertFalse(mask[0, 0])
        self.assertFalse(mask[0, 1])
        self.assertTrue(mask[0, 2])  # Another query's map remains in this row's reference.
        np.testing.assert_array_equal(parts["entropy_reference_counts"], [4, 5, 5])
        np.testing.assert_array_equal(parts["entropy_weights"], entropy_weights(
            tuple(parts[k] for k in ("cosine", "structure_similarity", "area_similarity")), mask))
        altered = copy.deepcopy(data)
        altered["joint_all"][1] = np.eye(1, 128, 17)[0]
        altered["relative_area"][1] = .9
        h2 = h.copy()
        h2[1] = -h2[0]
        changed = AreaPairScorer(h2, altered, altered).components(q)
        np.testing.assert_array_equal(changed["entropy_weights"][0], parts["entropy_weights"][0])

    def test_entropy_query_batching_and_candidate_scoring_do_not_change_weights(self):
        h, data, q = fixture()
        engine = AreaPairScorer(h, data, data)
        batch = engine.components(q)
        for row, token in enumerate(q):
            alone = engine.components(np.array([token]))
            # Different BLAS matrix shapes can change float32 cosine roundoff.
            np.testing.assert_allclose(alone["entropy_weights"][0], batch["entropy_weights"][row], atol=2e-7, rtol=0.)
        columns = np.array([1, 3, 5])
        full_reference = independent_components(q, h, data["relative_area"], data["joint_all"], data["owners"])
        subset_reference = independent_components(q, h, data["relative_area"], data["joint_all"], data["owners"],
                                                 candidate_indices=columns)
        np.testing.assert_array_equal(full_reference["entropy_weights"], subset_reference["entropy_weights"])
        expected = independent_score_components(full_reference, frozen_grid()["entropy"])[:, columns]
        np.testing.assert_array_equal(independent_score_components(subset_reference, frozen_grid()["entropy"]), expected)

    def test_no_padding_or_missing_allpoint_evidence_is_accepted(self):
        h, data, _ = fixture()
        for key, value in (("owners", np.array([10, 10, 20, 21, 21, -1])),
                           ("owners", data["owners"].astype(float)),
                           ("relative_area", np.array([.001, .002, .01, .1, .5, 0.])),
                           ("joint_all", np.zeros((6, 128)))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                bad = {**data, key: value}
                AreaPairScorer(h, bad, bad)
        # A zero CNN channel is valid; entropy gives it no weight.
        parts = AreaPairScorer(h, data, data).components(np.array([5]))
        self.assertEqual(parts["entropy_weights"][0, 0], 0.)

    def test_empty_query_and_no_other_map_reference_are_defined(self):
        h, data, _ = fixture()
        q = np.array([], dtype=np.int64)
        parts = AreaPairScorer(h, data, data).components(q)
        self.assertEqual(parts["entropy_weights"].shape, (0, 3))
        for spec in frozen_grid().values():
            self.assertEqual(score_components(parts, spec).shape, (0, 6))
            self.assertEqual(independent_pair_scores(q, h, data["relative_area"], data["joint_all"],
                                                    data["owners"], spec).shape, (0, 6))
        data["owners"][:] = 10
        parts = AreaPairScorer(h, data, data).components(np.array([0]))
        np.testing.assert_array_equal(parts["entropy_reference_counts"], [0])
        np.testing.assert_array_equal(parts["entropy_weights"], np.full((1, 3), 1. / 3.))

    def test_independent_components_can_skip_entropy_for_small_audits(self):
        h, data, q = fixture()
        selected = np.array([1, 4])
        parts = independent_components(q, h, data["relative_area"], data["joint_all"], data["owners"],
                                       candidate_indices=selected, with_entropy=False)
        self.assertNotIn("entropy_weights", parts)
        expected = AreaPairScorer(h, data, data).components(q)
        for name, spec in frozen_grid().items():
            if name == "entropy":
                with self.assertRaises(ValueError):
                    independent_score_components(parts, spec)
            else:
                np.testing.assert_allclose(independent_score_components(parts, spec),
                                           score_components(expected, spec)[:, selected], atol=2e-7, rtol=1e-6)

    def test_invalid_spec_fails_closed(self):
        baseline = frozen_grid()[BASELINE]
        for spec in (None, {}, "entropy", {"family": "entropy", "temperature": 1.},
                     {**baseline, "beta": 0.}, {**baseline, "alpha": .45},
                     {**baseline, "beta": True}, {**baseline, "alpha": np.nan},
                     {**baseline, "alpha": np.array([.5])}):
            with self.subTest(spec=str(spec)), self.assertRaises(ValueError):
                checked_spec(spec)

    def test_invalid_components_and_entropy_inputs_fail_closed(self):
        h, data, q = fixture()
        source = AreaPairScorer(h, data, data).components(q)
        for key, value in (("cosine", np.array([.1])), ("cosine", np.full((3, 6), np.nan)),
                           ("structure_similarity", np.full((3, 6), .5)),
                           ("area_similarity", np.zeros((3, 6))),
                           ("entropy_weights", np.ones((3, 3)))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                score_components({**source, key: value}, frozen_grid()["entropy"])
        with self.assertRaises(ValueError):
            score_components({}, frozen_grid()[BASELINE])
        channels = [np.ones((2, 3))] * 3
        for mask in (np.ones((2, 3)), np.ones((1, 3), dtype=bool)):
            with self.assertRaises(ValueError):
                entropy_weights(channels, mask)
        with self.assertRaises(ValueError):
            entropy_weights([np.full((2, 3), -1.)] * 3, np.ones((2, 3), dtype=bool))

    def test_independent_input_validation(self):
        h, data, q = fixture()
        base = [q, h, data["relative_area"], data["joint_all"], data["owners"]]
        invalid = [(0, np.array([0, 0])), (0, q.astype(float)), (0, np.array([6])),
                   (1, h * 2.), (1, h[:, :128]), (2, np.zeros(6)), (3, np.zeros((6, 128))),
                   (3, data["joint_all"] * .5), (4, np.full(6, -1))]
        for position, value in invalid:
            with self.subTest(position=position), self.assertRaises(ValueError):
                args = list(base)
                args[position] = value
                independent_components(*args)
        with self.assertRaises(ValueError):
            independent_components(*base, candidate_indices=np.array([-1]))
        with self.assertRaises(ValueError):
            independent_components(*base, with_entropy=1)


if __name__ == "__main__":
    unittest.main()
