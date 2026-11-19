from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fgw_grid_search import scoring


def spec(family, ticks, denominator=10):
    return {"family": family, "weights": [v / denominator for v in ticks],
            "ticks": list(ticks), "denominator": denominator}


class GridScoringTests(unittest.TestCase):
    def test_default_grid_has_166_points_and_f_cannot_disappear(self):
        grid = scoring.specs()
        self.assertEqual(len(grid), 166)
        self.assertEqual(Counter(s["family"] for s in grid.values()),
                         {"sum_raw": 55, "sum_minmax": 55,
                          "weighted_product": 55, "legacy_fgw": 1})
        self.assertEqual(next(iter(grid)), "sum_raw_d10_f01_g00_w09")
        self.assertEqual(list(grid)[-1], "legacy_fgw")
        for s in grid.values():
            self.assertEqual(scoring.validate_spec(s), s)
            self.assertGreater(s["weights"][0], 0)
            if s["family"] != "legacy_fgw":
                self.assertEqual(sum(s["ticks"]), 10)
                self.assertGreaterEqual(s["ticks"][0], 1)

    def test_custom_grid_count_and_minimum_f(self):
        settings = scoring.validate_settings({"denominator": 20, "min_f_ticks": 3,
                                             "families": ["weighted_product"],
                                             "include_reference": False})
        grid = scoring.specs(settings)
        self.assertEqual(len(grid), 18 * 19 // 2)
        self.assertTrue(all(s["ticks"][0] >= 3 for s in grid.values()))
        self.assertEqual(len(scoring.specs({"denominator": 1})), 4)

    def test_default_config_load_and_no_mutable_alias(self):
        self.assertEqual(scoring.load_settings(), scoring.SETTINGS)
        settings = scoring.validate_settings({})
        settings["ks"].append(999)
        settings["families"].clear()
        self.assertEqual(scoring.load_settings(), scoring.SETTINGS)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.json"
            path.write_text(json.dumps({"denominator": 4, "min_f_ticks": 2}))
            self.assertEqual(len(scoring.specs(scoring.load_settings(path))), 19)
            path.write_text('{"denominator": 4, "denominator": 10}')
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                scoring.load_settings(path)

    def test_unknown_fields_bad_booleans_and_scientific_changes_rejected(self):
        cases = [{"unknown": 1}, {"denominator": 0}, {"denominator": 21},
                 {"denominator": True}, {"denominator": 10.0},
                 {"min_f_ticks": 0}, {"min_f_ticks": 11}, {"min_f_ticks": False},
                 {"families": []}, {"families": ["sum_raw", "sum_raw"]},
                 {"families": ["unknown"]}, {"families": ("sum_raw",)},
                 {"include_reference": 1}, {"include_reference": "false"},
                 {"objective": "test/exact/all/Precision@10"},
                 {"selection_split": "test"}, {"tie_seed": 0}, {"ks": [10]},
                 {"schema_version": True}, {"training_required": 0},
                 {"test_parameter_search": True}, {"posthoc_research": False},
                 {"family_priority": list(reversed(scoring.SETTINGS["family_priority"]))}]
        for settings in cases:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                scoring.validate_settings(settings)

    def test_invalid_or_modified_specs_rejected(self):
        base = spec("sum_raw", [2, 3, 5])
        invalid = []
        for key, value in [("weights", [.21, .3, .5]), ("ticks", [0, 5, 5]),
                           ("ticks", [2, True, 7]), ("denominator", True),
                           ("denominator", 9), ("family", "sum_log")]:
            changed = deepcopy(base)
            changed[key] = value
            invalid.append(changed)
        invalid += [dict(base, angular_gamma=0),
                    {"family": "legacy_fgw", "weights": [.5, .5, .5]},
                    {"family": "legacy_fgw", "weights": [True, .5, 1]},
                    spec("sum_raw", [0, 5, 5])]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                scoring.validate_spec(value)

    def test_raw_sum_and_legacy_reference_are_distinct(self):
        components = {"F": np.array([.2, .9, .6]), "G": np.array([.8, .1, .2]),
                      "WBBS": np.array([.05, .5, 1.])}
        actual = scoring.score_variant(components, spec("sum_raw", [2, 3, 5]))
        np.testing.assert_allclose(actual, [.305, .46, .68], rtol=0, atol=1e-15)
        legacy = scoring.score_variant(components, scoring.specs()["legacy_fgw"])
        np.testing.assert_array_equal(legacy, (.5 * components["F"] + .5 * components["G"])
                                      * components["WBBS"])

    def test_minmax_uses_all_candidates_and_constant_branch_is_zero(self):
        components = {"F": np.array([.4, .6, .7, .9]), "G": np.full(4, .7),
                      "WBBS": np.array([.1, .3, .8, .9])}
        result = scoring.score_variant(components, spec("sum_minmax", [5, 0, 5]))
        np.testing.assert_allclose(result, [0, .325, .7375, 1.], atol=1e-15, rtol=0)
        truncated = {key: value[:2] for key, value in components.items()}
        # This shows why the pipeline must supply a full pool, not old Top-K rows.
        wrong = scoring.score_variant(truncated, spec("sum_minmax", [5, 0, 5]))
        self.assertNotEqual(result[1], wrong[1])
        mixed = scoring.score_variant(components, spec("sum_minmax", [5, 5, 0]))
        np.testing.assert_allclose(mixed, [0, .2, .3, .5], atol=1e-15, rtol=0)
        constant = {key: np.full(3, .7) for key in components}
        np.testing.assert_array_equal(scoring.score_variant(constant, spec("sum_minmax", [4, 3, 3])),
                                      np.zeros(3))

    def test_product_skips_zero_exponents_and_preserves_active_zeros(self):
        components = {"F": np.array([.25, .36, 0., .81]), "G": np.zeros(4),
                      "WBBS": np.array([1., .25, .64, 0.])}
        score = scoring.score_variant(components, spec("weighted_product", [5, 0, 5]))
        np.testing.assert_allclose(score, [.5, .3, 0., 0.], rtol=0, atol=1e-15)
        score = scoring.score_variant(components, spec("weighted_product", [5, 5, 0]))
        np.testing.assert_array_equal(score, np.zeros(4))

    def test_pure_f_is_bitwise_identical_for_every_family_and_does_not_alias(self):
        local = np.array([0., np.nextafter(0., 1.), .31, np.nextafter(1., 0.), 1.])
        components = {"F": local, "G": np.zeros(5), "WBBS": np.zeros(5)}
        for family in ("sum_raw", "sum_minmax", "weighted_product"):
            with self.subTest(family=family):
                result = scoring.score_variant(components, spec(family, [10, 0, 0]))
                np.testing.assert_array_equal(result.view(np.uint64), local.view(np.uint64))
                self.assertFalse(np.shares_memory(result, local))
        constant = {"F": np.full(3, .7), "G": np.zeros(3), "WBBS": np.zeros(3)}
        np.testing.assert_array_equal(scoring.score_variant(constant, spec("sum_minmax", [10, 0, 0])),
                                      constant["F"])

    def test_components_are_checked_even_when_branch_weight_is_zero(self):
        good = {"F": [.2, .4], "G": [.1, .9], "WBBS": [.3, .2]}
        for bad in ([float("nan"), .1], [.2, 1.00001], [-.1, .3], [.2], [],
                    [[.2, .3]], [.2 + .1j, .3]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                scoring.score_variant(dict(good, G=bad), spec("sum_raw", [10, 0, 0]))
        with self.assertRaises(ValueError):
            scoring.score_variant({"F": [.2]}, spec("sum_raw", [10, 0, 0]))

    def test_scoring_never_changes_component_arrays(self):
        components = {"F": np.array([.2, .9]), "G": np.array([.8, .3]),
                      "WBBS": np.array([.5, .7])}
        original = deepcopy(components)
        for value in scoring.specs().values():
            score = scoring.score_variant(components, value)
            self.assertEqual(score.dtype, np.float64)
            self.assertTrue(np.isfinite(score).all())
            for key in components:
                np.testing.assert_array_equal(components[key], original[key])


if __name__ == "__main__":
    unittest.main()
