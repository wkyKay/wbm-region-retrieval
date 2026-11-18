"""Independent fixed-formula and frozen-boundary checks."""
from copy import deepcopy
import unittest

import numpy as np

from b1_ablation import scoring
from b1_structure_fusion import scoring as old_scoring


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.values = {"f_score": np.array([.1, .8, 0., 1.]),
                       "optimal_f_score": np.array([.4, .9, 0., 1.]),
                       "global_score": np.array([.7, .2, 1., 0.]),
                       "angular_distance": np.array([.3, .6, 0., 1.])}
        self.values["optimal"] = .5*self.values["optimal_f_score"]+.5*self.values["global_score"]
        self.values["baseline"] = .5*self.values["f_score"]+.5*self.values["global_score"]
        self.distances = {"J": np.array([0., .25, .5, 1.]), "I": np.array([1., .5, .25, 0.])}

    def test_all_fixed_formulas_with_distinct_inputs(self):
        v, d = self.values, self.distances
        a, j, independent = np.exp(-.5*v["angular_distance"]), np.exp(-2*d["J"]), np.exp(-2*d["I"])
        expected = {"no_local": v["global_score"]*a*j, "no_global": v["optimal_f_score"]*a*j,
                    "no_angle": (.5*v["optimal_f_score"]+.5*v["global_score"])*j,
                    "greedy": (.5*v["f_score"]+.5*v["global_score"])*a*j,
                    "structure_only": a*j,
                    "I_2": (.5*v["optimal_f_score"]+.5*v["global_score"])*a*independent}
        for name, spec in scoring.specs("test").items():
            np.testing.assert_array_equal(scoring.score_variant(v, d, spec), expected[name])
        self.assertEqual(len(scoring.specs("valid")), 5)
        self.assertNotIn("I_2", scoring.specs("valid"))

    def test_full_b1_is_original_frozen_arithmetic(self):
        actual = scoring.full_b1_score(self.values, self.distances)
        original = old_scoring.score_variant(self.values, self.distances, {"family": "J", "strength": 2.})
        np.testing.assert_array_equal(actual, original)
        self.assertTrue(np.all(actual <= self.values["optimal"]))

    def test_zero_distance_and_angle_boundaries_and_surviving_branch_normalization(self):
        v = deepcopy(self.values)
        v["angular_distance"] = np.zeros(4)
        d = {"J": np.zeros(4), "I": np.zeros(4)}
        np.testing.assert_array_equal(scoring.score_variant(v, d, scoring.specs("valid")["no_local"]), v["global_score"])
        np.testing.assert_array_equal(scoring.score_variant(v, d, scoring.specs("valid")["no_global"]), v["optimal_f_score"])
        np.testing.assert_array_equal(scoring.score_variant(v, d, scoring.specs("valid")["structure_only"]), np.ones(4))
        v["angular_distance"] = np.ones(4)
        v["f_score"] = np.zeros(4)
        v["optimal_f_score"] = np.zeros(4)
        v["global_score"] = np.zeros(4)
        np.testing.assert_array_equal(scoring.score_variant(v, d, scoring.specs("valid")["no_angle"]), np.zeros(4))
        np.testing.assert_allclose(scoring.score_variant(v, d, scoring.specs("valid")["structure_only"]), np.exp(-.5))

    def test_greedy_switch_is_only_local_matching_change(self):
        v = deepcopy(self.values)
        v["f_score"] = v["optimal_f_score"].copy()
        np.testing.assert_array_equal(scoring.score_variant(v, self.distances, scoring.specs("valid")["greedy"]),
                                      scoring.full_b1_score(v, self.distances))

    def test_unknown_spec_and_invalid_domains_rejected(self):
        spec = scoring.specs("valid")["no_local"]
        with self.assertRaisesRegex(ValueError, "Unknown"):
            scoring.score_variant(self.values, self.distances, {**spec, "strength": 1.})
        for bad in (np.nan, -1e-4, 1.0001):
            values = deepcopy(self.values)
            values["global_score"][0] = bad
            with self.assertRaises(ValueError):
                scoring.score_variant(values, self.distances, spec)
        with self.assertRaises(ValueError):
            scoring.score_variant(self.values, {"J": np.array([.5])}, spec)
        settings = scoring.load_settings()
        settings["structure_strength"] = 1.
        with self.assertRaisesRegex(ValueError, "frozen"):
            scoring.validate_settings(settings)
        with self.assertRaises(ValueError):
            scoring.specs("smoke")


if __name__ == "__main__":
    unittest.main()
