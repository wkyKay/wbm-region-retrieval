"""Scientific-formula and valid-only selection checks for whole-map G/J."""
from __future__ import annotations

import copy
import math
import unittest

import numpy as np

from gj_fusion.scoring import checked_spec, frozen_grid, score_variant
from gj_fusion.selection import role_aliases, select_candidates


class ScoringTests(unittest.TestCase):
    def test_grid_has_only_eight_unique_roles_and_is_fresh(self):
        grid = frozen_grid()
        self.assertEqual(len(grid), 8)
        self.assertEqual(len({tuple(sorted(x.items())) for x in grid.values()}), 8)
        self.assertEqual(sum(x["family"] == "endpoint" for x in grid.values()), 2)
        self.assertNotIn({"family": "geometric", "alpha": 0.5}, grid.values())
        grid["product"]["alpha"] = 99
        self.assertEqual(frozen_grid()["product"]["alpha"], 0.5)

    def test_all_formulas_against_scalar_math(self):
        g = np.array([0.0, 0.17, 0.8, 1.0])
        d = np.array([1.0, 0.02, 0.73, 0.0])
        for name, spec in frozen_grid().items():
            expected = []
            for global_score, distance in zip(g, d):
                j = math.exp(-2 * float(distance))
                a = spec["alpha"]
                if name == "G_only":
                    value = global_score
                elif name == "J_only":
                    value = j
                elif name == "product":
                    value = global_score * j
                elif spec["family"] == "additive":
                    value = a * global_score + (1-a) * j
                else:
                    value = global_score**a * j**(1-a)
                expected.append(value)
            with self.subTest(name=name):
                observed = score_variant(g, d, spec)
                self.assertEqual(observed.dtype, np.float64)
                np.testing.assert_allclose(observed, expected, rtol=1e-15, atol=1e-16)

    def test_endpoint_boundary_values_and_input_immutability(self):
        g = np.array([0.0, 1.0]); d = np.array([0.0, 1.0])
        np.testing.assert_array_equal(score_variant(g, d, frozen_grid()["G_only"]), g)
        np.testing.assert_array_equal(score_variant(g, d, frozen_grid()["J_only"]), np.exp(-2*d))
        output = score_variant(g, d, frozen_grid()["G_only"])
        output[0] = 0.8
        self.assertEqual(g[0], 0.0)
        np.testing.assert_array_equal(d, [0.0, 1.0])

    def test_active_geometric_zero_and_additive_compensation(self):
        for name in ("geo_a0p25", "product", "geo_a0p75"):
            self.assertEqual(score_variant([0.0], [0.2], frozen_grid()[name])[0], 0.0)
        self.assertGreater(score_variant([0.0], [0.2], frozen_grid()["add_a0p5"])[0], 0.0)

    def test_product_preserves_historical_operation(self):
        g = np.array([0.0, .13, .5, .9, 1.0])
        d = np.array([.2, .1, .8, .5, 1.0])
        product = score_variant(g, d, frozen_grid()["product"])
        np.testing.assert_array_equal(product, g*np.exp(-2*d))
        np.testing.assert_array_equal(np.argsort(product), np.argsort(np.sqrt(product)))
        geometric_half = np.sqrt(g) * np.sqrt(np.exp(-2*d))
        np.testing.assert_array_equal(np.argsort(product), np.argsort(geometric_half))

    def test_additive_and_product_can_reverse_order(self):
        # First candidate has excellent J but zero G; second is moderate in both.
        g = [0.0, 0.4]
        d = [0.0, -math.log(.4)/2]
        additive = score_variant(g, d, frozen_grid()["add_a0p5"])
        product = score_variant(g, d, frozen_grid()["product"])
        self.assertGreater(additive[0], additive[1])
        self.assertLess(product[0], product[1])

    def test_invalid_components_rejected_including_unused_endpoint(self):
        bad = ([math.nan], [math.inf], [-.001], [1.001], [True], [".5"],
               [1j], [[.5]], [], np.array([.5], dtype=object))
        for value in bad:
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError):
                    score_variant(value, [.5], frozen_grid()["J_only"])
                with self.assertRaises(ValueError):
                    score_variant([.5], value, frozen_grid()["G_only"])
        with self.assertRaises(ValueError):
            score_variant([.1, .2], [.1], frozen_grid()["product"])

    def test_spec_rejects_unfrozen_or_ambiguous_parameters(self):
        invalid = (None, {}, {"family": "additive", "alpha": True},
                   {"family": "endpoint", "alpha": False},
                   {"family": "additive", "alpha": .3},
                   {"family": "additive", "alpha": math.nan},
                   {"family": "geometric", "alpha": .5},
                   {"family": "product", "alpha": .25},
                   {"family": "endpoint", "alpha": .5},
                   {"family": "additive", "alpha": .5, "strength": 1})
        for spec in invalid:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                checked_spec(spec)


def result_grid(hits=2000):
    return {name: {"split": "valid", "top10_hits": hits, "num_queries": 500}
            for name in frozen_grid()}


class SelectionTests(unittest.TestCase):
    def test_all_tied_prefers_references_and_deduplicates_roles(self):
        selected = select_candidates(result_grid())
        self.assertEqual(selected["winner"], "product")
        self.assertEqual(selected["family_winners"], {"additive": "J_only", "geometric": "product"})
        self.assertEqual(selected["selected_variants"], ["product", "J_only", "G_only"])
        self.assertEqual(selected["variant_roles"]["product"],
                         ["overall_winner", "geometric_winner", "product_reference"])
        self.assertEqual([r["variant"] for r in selected["ranking"][:3]],
                         ["product", "J_only", "G_only"])

    def test_global_family_winners_and_lexicographic_ties(self):
        data = result_grid(1000)
        data["add_a0p25"]["top10_hits"] = 3000
        data["add_a0p75"]["top10_hits"] = 3000
        data["geo_a0p75"]["top10_hits"] = 2900
        selected = select_candidates(data)
        self.assertEqual(selected["winner"], "add_a0p25")
        self.assertEqual(selected["family_winners"],
                         {"additive": "add_a0p25", "geometric": "geo_a0p75"})
        self.assertEqual(selected["selected_variants"],
                         ["add_a0p25", "geo_a0p75", "product", "J_only", "G_only"])
        self.assertEqual(selected["ranking"][0]["precision_at_10"], .6)

    def test_both_families_include_shared_endpoints(self):
        data = result_grid()
        data["G_only"]["top10_hits"] = 3000
        selected = select_candidates(data)
        self.assertEqual(set(selected["family_winners"].values()), {"G_only"})
        for family in ("additive", "geometric"):
            self.assertTrue({"G_only", "J_only"} <= set(selected["family_candidates"][family]))

    def test_aliases_state_score_vs_rank_equivalence(self):
        aliases = role_aliases()
        self.assertEqual(aliases["geometric_alpha_0p5"]["equivalence"], "same_mathematical_ranking")
        self.assertEqual(aliases["geometric_alpha_0p5"]["stored_score"], "G*J")
        self.assertEqual(aliases["additive_alpha_0"]["variant"], "J_only")
        aliases["additive_alpha_0"]["variant"] = "bad"
        self.assertEqual(role_aliases()["additive_alpha_0"]["variant"], "J_only")

    def test_rejects_test_split_missing_split_and_partial_grid(self):
        with self.assertRaises(ValueError):
            select_candidates(result_grid(), split="test")
        for mutation in ("test", "missing_split", "missing_variant", "extra_variant"):
            data = result_grid()
            if mutation == "test":
                data["product"]["split"] = "test"
            elif mutation == "missing_split":
                del data["product"]["split"]
            elif mutation == "missing_variant":
                del data["product"]
            else:
                data["extra"] = copy.deepcopy(data["product"])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                select_candidates(data)

    def test_rejects_invalid_counts_hits_or_k(self):
        for key, bad_values in (("top10_hits", (True, -1, 5001, 2000.0, math.nan)),
                                ("num_queries", (True, 499, 501, 500.0)),
                                ("k", (True, 5, 10.0))):
            for value in bad_values:
                data = result_grid(); data["J_only"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    select_candidates(data)

    def test_secondary_metrics_do_not_select_and_inputs_are_preserved(self):
        data = result_grid()
        data["add_a0p5"]["test_precision"] = 1.0
        data["add_a0p5"]["coverage_ndcg"] = 1.0
        original = copy.deepcopy(data)
        selected = select_candidates(data)
        self.assertEqual(data, original)
        self.assertEqual(selected["winner"], "product")
        self.assertFalse(selected["test_parameter_search"])


if __name__ == "__main__":
    unittest.main()
