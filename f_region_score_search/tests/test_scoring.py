import copy
import math
import unittest
from unittest import mock

import numpy as np

from f_region_structure.scoring import RegionPairScorer, score_components as old_score
from f_region_score_search.scoring import (
    OLD_SPEC, checked_spec, frozen_grid, independent_pair_scores, score_components,
)


def fixture():
    rng = np.random.default_rng(2026)
    h = rng.normal(size=(6, 512)).astype(np.float32)
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    h[1] = h[0]
    h[2] = -h[0]
    h[5] = 0.
    p = rng.random((6, 128))
    p /= p.sum(axis=1, keepdims=True)
    p[4:] = 0.
    radial = rng.random((6, 16))
    radial /= radial.sum(axis=1, keepdims=True)
    area = np.array([.001, .002, .01, .1, .5, 1.])
    engine = RegionPairScorer(h, {"histograms": radial, "relative_area": area},
                              {"joint": p, "joint_all": p})
    q = np.array([0, 4, 5], dtype=np.int64)
    return h, area, p, q, engine.components(q, specs={"selected": OLD_SPEC})


class ScoringTests(unittest.TestCase):
    def test_grid_is_frozen_complete_and_detached(self):
        grid = frozen_grid()
        self.assertEqual(len(grid), 15)
        self.assertEqual(next(iter(grid)), "product")
        mixed = [spec for spec in grid.values() if spec["family"] == "mixed"]
        self.assertEqual(len(mixed), 12)
        self.assertEqual(sum(spec["gate_zero"] for spec in mixed), 3)
        self.assertEqual({(s["alpha"], s["eta"]) for s in mixed if not s["gate_zero"]},
                         {(a, e) for a in (.25, .5, .75) for e in (.25, .5, 1.)})
        for spec in grid.values():
            self.assertEqual(checked_spec(spec), spec)
            self.assertEqual(spec["descriptor"], "all_points")
            self.assertEqual(spec["strength"], .5)
            self.assertTrue(spec["area"])
        grid["product"]["area"] = False
        self.assertTrue(frozen_grid()["product"]["area"])

    def test_product_exactly_calls_old_selected_score(self):
        *_, parts = fixture()
        expected = old_score(parts, OLD_SPEC)
        with mock.patch("f_region_score_search.scoring.original_score_components",
                        wraps=old_score) as source:
            actual = score_components(parts, frozen_grid()["product"])
            source.assert_called_once_with(parts, OLD_SPEC)
        np.testing.assert_array_equal(actual, expected)

    def test_all_fifteen_match_independent_scalar_formulas(self):
        h, area, p, q, parts = fixture()
        for name, spec in frozen_grid().items():
            with self.subTest(name=name):
                actual = score_components(parts, spec)
                expected = independent_pair_scores(q, h, area, p, spec)
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=2e-7)
                self.assertEqual(actual.shape, (len(q), len(h)))
                self.assertTrue(np.all((actual >= 0.) & (actual <= 1.)))

    def test_mixed_and_additive_endpoints_use_same_kernels(self):
        *_, parts = fixture()
        c, b = parts["cosine"], parts["area"]
        s = parts["structure"]["all_points"]
        t = np.exp(-.5*s["distance"])
        for name, spec in frozen_grid().items():
            if spec["family"] != "mixed":
                continue
            alpha, eta = spec["alpha"], spec["eta"]
            expected = b*((1.-eta)*c*t+eta*(alpha*c+(1.-alpha)*t))
            expected = np.where(s["valid"], expected, c*b)
            if spec["gate_zero"]:
                expected = np.where(c > 0., expected, 0.)
            with self.subTest(name=name):
                np.testing.assert_array_equal(score_components(parts, spec), expected)
        np.testing.assert_array_equal(score_components(parts, frozen_grid()["cnn_only"]), c*b)
        np.testing.assert_array_equal(score_components(parts, frozen_grid()["j_only"]),
                                      np.where(s["valid"], t*b, 0.))

    def test_zero_cosine_support_is_distinct_for_additive_and_gated(self):
        *_, parts = fixture()
        # q=0 versus token=2 have opposite CNN features but valid structure.
        self.assertEqual(parts["cosine"][0, 2], 0.)
        grid = frozen_grid()
        self.assertEqual(score_components(parts, grid["product"])[0, 2], 0.)
        self.assertGreater(score_components(parts, grid["mixed_a0p5_e1p0"])[0, 2], 0.)
        self.assertGreater(score_components(parts, grid["mixed_a0p5_e0p25"])[0, 2], 0.)
        self.assertEqual(score_components(parts, grid["gated_add_a0p5"])[0, 2], 0.)

    def test_missing_one_or_both_structures_does_not_add_constant_reward(self):
        *_, parts = fixture()
        missing = ~parts["structure"]["all_points"]["valid"]
        self.assertTrue(missing[0, 4])
        self.assertTrue(missing[1, 4])
        expected_cb = parts["cosine"]*parts["area"]
        for name, spec in frozen_grid().items():
            with self.subTest(name=name):
                expected = np.zeros_like(expected_cb) if name == "j_only" else expected_cb
                np.testing.assert_array_equal(score_components(parts, spec)[missing], expected[missing])

    def test_geometry_bounds_and_expected_closed_form(self):
        parts = {"cosine": np.array([[.4, 0.]]), "area": np.array([[.85, 1.]]),
                 "structure": {"all_points": {"distance": np.array([[1., 0.]]),
                                                 "valid": np.array([[True, True]])}}}
        grid = frozen_grid()
        self.assertAlmostEqual(score_components(parts, grid["product"])[0, 0],
                               .4*.85*math.exp(-.5))
        self.assertAlmostEqual(score_components(parts, grid["j_only"])[0, 0],
                               .85*math.exp(-.5))
        self.assertAlmostEqual(score_components(parts, grid["mixed_a0p25_e1p0"])[0, 0],
                               .85*(.25*.4+.75*math.exp(-.5)))

    def test_empty_query_batch_is_supported(self):
        h, area, p, _, _ = fixture()
        empty = np.empty(0, dtype=np.int64)
        parts = {"cosine": np.empty((0, 6)), "area": np.empty((0, 6)),
                 "structure": {"all_points": {"distance": np.empty((0, 6)),
                                                 "valid": np.empty((0, 6), dtype=bool)}}}
        for spec in frozen_grid().values():
            self.assertEqual(score_components(parts, spec).shape, (0, 6))
            self.assertEqual(independent_pair_scores(empty, h, area, p, spec).shape, (0, 6))

    def test_undeclared_or_malformed_specs_fail_closed(self):
        grid = frozen_grid()
        invalid = [None, "product", {}, {**grid["product"], "area": 1},
                   {**grid["product"], "strength": True},
                   {**grid["product"], "strength": np.nan},
                   {**grid["product"], "extra": 1},
                   {**grid["mixed_a0p5_e1p0"], "eta": 0.},
                   {**grid["mixed_a0p5_e1p0"], "alpha": .6},
                   {**grid["mixed_a0p5_e1p0"], "alpha": np.array([.5])}]
        for spec in invalid:
            with self.subTest(spec=str(spec)), self.assertRaises(ValueError):
                checked_spec(spec)

    def test_component_validation(self):
        *_, source = fixture()
        bad = []
        for key, replacement in (("cosine", np.array([1.])), ("area", np.zeros((3, 6))),
                                  ("cosine", np.full((3, 6), np.nan))):
            parts = copy.deepcopy(source)
            parts[key] = replacement
            bad.append(parts)
        for key, replacement in (("valid", np.ones((3, 6))),
                                 ("distance", np.full((3, 6), 1.1))):
            parts = copy.deepcopy(source)
            parts["structure"]["all_points"][key] = replacement
            bad.append(parts)
        bad.append({})
        for parts in bad:
            with self.assertRaises(ValueError):
                score_components(parts, frozen_grid()["product"])

    def test_reference_input_validation(self):
        h, area, p, q, _ = fixture()
        invalid = [(np.array([0, 0]), h, area, p), (np.array([6]), h, area, p),
                   (q.astype(float), h, area, p), (q, h[:, :128], area, p),
                   (q, h*2., area, p), (q, h, np.zeros_like(area), p),
                   (q, h, area, p*.5), (q, h, area, p[:, :127])]
        for args in invalid:
            with self.assertRaises(ValueError):
                independent_pair_scores(*args, frozen_grid()["product"])


if __name__ == "__main__":
    unittest.main()
