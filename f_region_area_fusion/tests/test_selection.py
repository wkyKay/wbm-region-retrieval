import copy
import unittest
from unittest.mock import patch

from f_region_area_fusion.scoring import frozen_grid
from f_region_area_fusion.selection import (BASELINE_VARIANT, RULE,
                                            select_top10, tie_key)


def result_grid(hits=1000, count=500):
    return {name: {"split": "valid", "top10_hits": hits, "num_queries": count}
            for name in frozen_grid()}


class SelectionTests(unittest.TestCase):
    def test_complete_grid_freezes_exactly_ten_by_integer_hits(self):
        results = result_grid()
        names = sorted(results)
        for index, name in enumerate(names):
            results[name]["top10_hits"] = 2000 + index
        selected = select_top10(results)
        expected = list(reversed(names))
        self.assertEqual(selected["selected_variants"], expected[:10])
        self.assertEqual(list(selected["selected_specs"]), expected[:10])
        self.assertEqual([r["variant"] for r in selected["ranking"]], expected)
        self.assertEqual([r["rank"] for r in selected["ranking"]], list(range(1, 30)))
        self.assertEqual(selected["winner"], expected[0])
        self.assertEqual(selected["ranking"][0]["precision_at_10"], 2028 / 5000)
        self.assertEqual(selected["num_candidates"], 29)
        self.assertEqual(selected["num_selected"], 10)
        self.assertEqual(selected["num_queries"], 500)
        self.assertFalse(selected["synthetic"])
        self.assertTrue(selected["posthoc_design"])
        self.assertFalse(selected["test_parameter_search"])

    def test_all_equal_prefers_baseline_then_lexicographic_names(self):
        selected = select_top10(result_grid())
        expected = [BASELINE_VARIANT, *sorted(set(frozen_grid()) - {BASELINE_VARIANT})]
        self.assertEqual([r["variant"] for r in selected["ranking"]], expected)
        self.assertEqual(selected["selected_variants"], expected[:10])
        self.assertEqual(tie_key(BASELINE_VARIANT), (False, BASELINE_VARIANT))
        self.assertEqual(RULE["tie_break"], ["baseline_first", "lexicographic_variant_name"])

    def test_tenth_place_ties_are_resolved_without_expanding_selection(self):
        results = result_grid(0)
        others = sorted(set(results) - {BASELINE_VARIANT})
        for name in others[:8]:
            results[name]["top10_hits"] = 3001
        for name in [BASELINE_VARIANT, others[8], others[9]]:
            results[name]["top10_hits"] = 3000
        selected = select_top10(results)
        self.assertEqual(selected["selected_variants"], [*others[:8], BASELINE_VARIANT, others[8]])
        self.assertEqual(selected["ranking"][10]["variant"], others[9])
        self.assertNotIn(others[9], selected["selected_specs"])

    def test_baseline_is_not_guaranteed_a_test_place(self):
        results = result_grid(0)
        others = sorted(set(results) - {BASELINE_VARIANT})
        for name in others[:10]:
            results[name]["top10_hits"] = 3001
        results[BASELINE_VARIANT]["top10_hits"] = 3000
        selected = select_top10(results)
        self.assertEqual(selected["selected_variants"], others[:10])
        self.assertEqual(selected["ranking"][10]["variant"], BASELINE_VARIANT)

    def test_mapping_order_and_secondary_metrics_cannot_change_choice(self):
        results = result_grid()
        expected = select_top10(results)
        for index, row in enumerate(results.values()):
            row["metrics"] = {"Precision@10": index / 100, "mAP@100": 1 - index / 100}
            row["coverage_metrics"] = {"Coverage-nDCG@10": index / 29}
            row["test_metrics"] = {"Precision@10": index / 29}
        self.assertEqual(select_top10(results), expected)
        self.assertEqual(select_top10(dict(reversed(list(results.items())))), expected)

    def test_selection_payload_is_detached_from_inputs_and_other_fields(self):
        grid = frozen_grid()
        pristine = copy.deepcopy(grid)
        results = result_grid()
        with patch("f_region_area_fusion.selection.frozen_grid", return_value=grid):
            selected = select_top10(results)
        results[BASELINE_VARIANT]["top10_hits"] = 0
        selected["selected_specs"][BASELINE_VARIANT]["family"] = "changed"
        selected["selection_rule"]["tie_break"].clear()
        self.assertEqual(grid, pristine)
        self.assertEqual(selected["ranking"][0]["top10_hits"], 1000)
        self.assertEqual(selected["ranking"][0]["spec"], pristine[BASELINE_VARIANT])
        self.assertTrue(RULE["tie_break"])
        selected["ranking"][0]["spec"]["family"] = "also_changed"
        self.assertEqual(grid, pristine)

    def test_incomplete_extra_or_nonmapping_results_are_rejected(self):
        missing = result_grid()
        missing.pop(BASELINE_VARIANT)
        extra = result_grid()
        extra["outside_grid"] = {"top10_hits": 1000, "num_queries": 500}
        for bad in (missing, extra, {}, [], None):
            with self.subTest(kind=type(bad).__name__), self.assertRaises(ValueError):
                select_top10(bad)

    def test_invalid_frozen_grid_is_rejected(self):
        grid = frozen_grid()
        without_baseline = dict(grid)
        without_baseline.pop(BASELINE_VARIANT)
        without_baseline["wrong_baseline"] = grid[BASELINE_VARIANT]
        shortened = dict(grid)
        shortened.pop(BASELINE_VARIANT)
        for invalid in (without_baseline, shortened):
            with patch("f_region_area_fusion.selection.frozen_grid", return_value=invalid):
                with self.assertRaises(ValueError):
                    select_top10({name: {"top10_hits": 1, "num_queries": 500} for name in invalid})

    def test_test_and_mixed_splits_are_rejected(self):
        for split in ("test", "train", None):
            with self.subTest(split=split), self.assertRaises(ValueError):
                select_top10(result_grid(), split=split)
            results = result_grid()
            results[BASELINE_VARIANT]["split"] = split
            with self.assertRaises(ValueError):
                select_top10(results)

    def test_bad_hit_counts_query_counts_and_rows_are_rejected(self):
        for field, bad in (("top10_hits", True), ("top10_hits", 1.), ("top10_hits", -1),
                           ("top10_hits", 5001), ("top10_hits", None),
                           ("num_queries", True), ("num_queries", 500.),
                           ("num_queries", 499), ("num_queries", None)):
            results = result_grid()
            results[BASELINE_VARIANT][field] = bad
            with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                select_top10(results)
        for bad in (None, [], 1):
            results = result_grid()
            results[BASELINE_VARIANT] = bad
            with self.subTest(row=bad), self.assertRaises(ValueError):
                select_top10(results)
        for bad in (True, 0, -1, 500., None):
            with self.subTest(expected_queries=bad), self.assertRaises(ValueError):
                select_top10(result_grid(), expected_queries=bad)

    def test_synthetic_query_count_is_explicit_but_still_selects_ten(self):
        results = result_grid(hits=12, count=2)
        selected = select_top10(results, expected_queries=2)
        self.assertTrue(selected["synthetic"])
        self.assertEqual(selected["ranking"][0]["precision_at_10"], .6)
        self.assertEqual(len(selected["selected_variants"]), 10)
        with self.assertRaises(ValueError):
            select_top10(results)


if __name__ == "__main__":
    unittest.main()
