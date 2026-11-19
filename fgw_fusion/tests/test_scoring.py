"""Frozen formula, input, branch deletion and zero-WBBS tests."""
from copy import deepcopy
import unittest

import numpy as np
from fgw_fusion import scoring


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.values = {"optimal_f_score": np.array([.8, .2, .6, 1.]),
                       "global_score": np.array([.2, .8, 1., 0.]),
                       "angular_distance": np.array([0., 1., .3, .8]),
                       "f_score": np.array([.1, .4, .7, .2])}
        self.w = np.array([1., .5, .2, 0.])

    def test_three_frozen_formulas_and_deletion_normalization(self):
        formulas = {"FGW": np.array([.5, .25, .16, 0.]),
                    "GW": np.array([.2, .4, .2, 0.]),
                    "FW": np.array([.8, .1, .12, 0.])}
        for name, spec in scoring.specs().items():
            np.testing.assert_allclose(scoring.score_variant(self.values, self.w, spec), formulas[name], atol=1e-15, rtol=0)
            self.assertEqual(spec["local_weight"]+spec["global_weight"], 1.)
            self.assertEqual(spec["wbbs_power"], 1.)
            self.assertEqual(spec["angular_gamma"], 0.)

    def test_deleted_branch_greedy_and_angle_have_no_effect(self):
        for name, inactive in (("GW", "optimal_f_score"), ("FW", "global_score")):
            spec = scoring.specs()[name]
            before = scoring.score_variant(self.values, self.w, spec)
            altered = deepcopy(self.values)
            for key in (inactive, "angular_distance", "f_score"):
                altered[key] = 1-altered[key]
            np.testing.assert_array_equal(before, scoring.score_variant(altered, self.w, spec))
        altered = {**self.values, "angular_distance": np.full(4, np.nan), "f_score": np.full(4, np.nan)}
        np.testing.assert_array_equal(scoring.score_variant(self.values, self.w, scoring.specs()["FGW"]),
                                      scoring.score_variant(altered, self.w, scoring.specs()["FGW"]))

    def test_wbbs_zero_suppresses_and_one_preserves_base(self):
        for name, spec in scoring.specs().items():
            np.testing.assert_array_equal(scoring.score_variant(self.values, np.zeros(4), spec), np.zeros(4))
            result = scoring.score_variant(self.values, np.ones(4), spec)
            expected = spec["local_weight"]*self.values["optimal_f_score"]+spec["global_weight"]*self.values["global_score"]
            np.testing.assert_array_equal(result, expected)
            self.assertEqual(result.dtype, np.float64)

    def test_refuses_modified_recipe_and_spec(self):
        for key, value in (("tie_seed", 1), ("recipe", "tuned"), ("test_parameter_search", True)):
            settings = deepcopy(scoring.SETTINGS); settings[key] = value
            with self.assertRaises(ValueError):scoring.validate_settings(settings)
        for key, value in (("local_weight", .25), ("global_weight", .25), ("wbbs_power", 2.),
                           ("angular_gamma", .5), ("local_mode", "greedy"), ("wbbs_power", True)):
            spec = {**scoring.specs()["FGW"], key:value}
            with self.assertRaises(ValueError):scoring.score_variant(self.values, self.w, spec)
        with self.assertRaises(ValueError):scoring.specs("train")
        for ks in ([1,1], [5,1], [0], [True], []):
            with self.assertRaises(ValueError):scoring.validate_settings({**scoring.SETTINGS,"ks":ks})

    def test_refuses_nonfinite_out_of_range_and_misaligned_vectors(self):
        spec=scoring.specs()["FGW"]
        for bad in (np.nan,np.inf,-.01,1.01):
            for key in ("optimal_f_score","global_score"):
                values=deepcopy(self.values);values[key][0]=bad
                with self.assertRaises(ValueError):scoring.score_variant(values,self.w,spec)
            w=self.w.copy();w[0]=bad
            with self.assertRaises(ValueError):scoring.score_variant(self.values,w,spec)
        for w in ([.5], [[.5]*4]):
            with self.assertRaises(ValueError):scoring.score_variant(self.values,w,spec)
        with self.assertRaises(ValueError):scoring.score_variant({},self.w,spec)


if __name__ == "__main__":unittest.main()
