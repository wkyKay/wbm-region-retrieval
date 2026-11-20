import unittest

from f_region_score_search.scoring import frozen_grid
from f_region_score_search.selection import RULE, select_best, tie_key


def result_grid(hits=1000, count=500):
    return {name: {"split": "valid", "top10_hits": hits, "num_queries": count}
            for name in frozen_grid()}


def find_variant(**fields):
    matches = [name for name, spec in frozen_grid().items()
               if all(spec.get(key) == value for key, value in fields.items())]
    if len(matches) != 1:
        raise AssertionError(f"Expected one frozen variant for {fields}: {matches}")
    return matches[0]


class SelectionTests(unittest.TestCase):
    def test_complete_grid_uses_integer_hits_and_ratio(self):
        results = result_grid()
        winner = find_variant(family="mixed", alpha=.75, eta=.5, gate_zero=False)
        results[winner]["top10_hits"] = 2501
        selection = select_best(results)
        self.assertEqual(selection["winner"], winner)
        self.assertEqual(selection["ranking"][0]["precision_at_10"], 2501 / 5000)
        self.assertEqual(selection["ranking"][0]["top10_hits"], 2501)
        self.assertEqual(selection["num_candidates"], 15)
        self.assertEqual(selection["num_queries"], 500)
        self.assertFalse(selection["synthetic"])
        self.assertEqual(selection["selection_rule"]["test_policy"], "valid_only_no_test")

    def test_product_wins_all_equal_and_results_are_detached(self):
        results = result_grid()
        selection = select_best(results)
        self.assertEqual(selection["winner"], "product")
        results["product"]["top10_hits"] = 0
        selection["selected_spec"]["family"] = "changed"
        selection["selection_rule"]["tie_break"].clear()
        self.assertEqual(selection["ranking"][0]["top10_hits"], 1000)
        self.assertEqual(selection["ranking"][0]["spec"]["family"], "product")
        self.assertTrue(RULE["tie_break"])
        self.assertEqual(frozen_grid()["product"]["family"], "product")

    def test_eta_then_alpha_then_gate_then_name_tie_order(self):
        results = result_grid()
        results["product"]["top10_hits"] = 0
        selection = select_best(results)
        ordered = [row["variant"] for row in selection["ranking"]]
        first = find_variant(family="mixed", alpha=.5, eta=.25, gate_zero=False)
        self.assertEqual(selection["winner"], first)
        # Smaller eta precedes a closer alpha at larger eta.
        low_eta = find_variant(family="mixed", alpha=.25, eta=.25, gate_zero=False)
        high_eta = find_variant(family="mixed", alpha=.5, eta=.5, gate_zero=False)
        self.assertLess(ordered.index(low_eta), ordered.index(high_eta))
        # Equal alpha distance is resolved by the stable variant name.
        symmetric = [find_variant(family="mixed", alpha=a, eta=.25, gate_zero=False)
                     for a in (.25, .75)]
        self.assertEqual(sorted(symmetric, key=ordered.index), sorted(symmetric))
        for alpha in (.25, .5, .75):
            ungated = find_variant(family="mixed", alpha=alpha, eta=1., gate_zero=False)
            gated = find_variant(family="mixed", alpha=alpha, eta=1., gate_zero=True)
            self.assertLess(ordered.index(ungated), ordered.index(gated))

    def test_endpoints_have_explicit_stable_tie_keys(self):
        grid = frozen_grid()
        endpoints = [find_variant(family=family) for family in ("cnn_only", "j_only")]
        for name in endpoints:
            self.assertEqual(tie_key(name, grid[name])[1:4], (2., 1., False))
        results = result_grid(0)
        for name in endpoints:
            results[name]["top10_hits"] = 1
        self.assertEqual(select_best(results)["winner"], min(endpoints))
        self.assertEqual(RULE["endpoint_tie_values"], {"eta": 2., "alpha_distance_from_0.5": 1.})

    def test_selection_is_independent_of_input_mapping_order(self):
        results = result_grid()
        self.assertEqual(select_best(results), select_best(dict(reversed(list(results.items())))))

    def test_incomplete_extra_or_nonmapping_grid_is_rejected(self):
        missing = result_grid()
        missing.pop("product")
        extra = result_grid()
        extra["unregistered"] = {"top10_hits": 1000, "num_queries": 500}
        for invalid in (missing, extra, {}, [], None):
            with self.subTest(invalid=type(invalid).__name__), self.assertRaises(ValueError):
                select_best(invalid)

    def test_test_or_mixed_split_is_rejected(self):
        with self.assertRaises(ValueError):
            select_best(result_grid(), split="test")
        for split in ("test", "train", None):
            mixed = result_grid()
            mixed["product"]["split"] = split
            with self.subTest(split=split), self.assertRaises(ValueError):
                select_best(mixed)

    def test_bad_counts_hits_or_candidate_records_are_rejected(self):
        for field, bad in (("top10_hits", True), ("top10_hits", 1.), ("top10_hits", -1),
                           ("top10_hits", 5001), ("top10_hits", None),
                           ("num_queries", True), ("num_queries", 500.),
                           ("num_queries", 499), ("num_queries", None)):
            results = result_grid()
            results["product"][field] = bad
            with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                select_best(results)
        for bad in (None, [], 1):
            results = result_grid()
            results["product"] = bad
            with self.subTest(record=bad), self.assertRaises(ValueError):
                select_best(results)
        for bad in (True, 0, -1, 500., None):
            with self.subTest(expected_queries=bad), self.assertRaises(ValueError):
                select_best(result_grid(), expected_queries=bad)

    def test_synthetic_count_is_explicit_and_still_complete(self):
        results = result_grid(hits=12, count=2)
        selection = select_best(results, expected_queries=2)
        self.assertTrue(selection["synthetic"])
        self.assertEqual(selection["ranking"][0]["precision_at_10"], .6)
        with self.assertRaises(ValueError):
            select_best(results)


if __name__ == "__main__":
    unittest.main()
