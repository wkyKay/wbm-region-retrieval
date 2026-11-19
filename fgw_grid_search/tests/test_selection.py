from copy import deepcopy
import json
import unittest

from fgw_grid_search import scoring, selection


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.settings = scoring.validate_settings({"denominator": 4})
        self.specs = scoring.specs(self.settings)
        self.rows = [{"variant": name, "family": spec["family"], "split": "valid",
                      "num_queries": 500, "precision10_hits": 2500,
                      "Precision@10": .5, "mAP@100": .2, "Coverage-nDCG@10": .8}
                     for name, spec in self.specs.items()]

    def named(self, family, ticks):
        return next(name for name, spec in self.specs.items()
                    if spec["family"] == family and spec.get("ticks") == ticks)

    def choose(self, rows=None):
        return selection.choose_winner(self.rows if rows is None else rows,
                                       self.specs, self.settings)

    def tie_only(self, names):
        rows = deepcopy(self.rows)
        for row in rows:
            if row["variant"] in names:
                row.update(precision10_hits=3000, **{"Precision@10": .6})
        return self.choose(rows)

    def test_default_tie_prefers_raw_pure_f_and_is_reproducible(self):
        record = self.choose()
        self.assertEqual(record["winner"], self.named("sum_raw", [4, 0, 0]))
        self.assertEqual(record, self.choose(list(reversed(self.rows))))
        self.assertFalse(record["secondary_metrics_used_for_selection"])
        self.assertFalse(record["test_used_for_selection"])
        self.assertEqual(record["num_variants"], len(self.specs))
        self.assertEqual(record["selection_split"], "valid")
        self.assertEqual(record["family_winners"]["legacy_fgw"], "legacy_fgw")
        json.dumps(record, allow_nan=False, sort_keys=True)

    def test_integer_hits_win_and_secondary_metrics_cannot_pick(self):
        target = self.named("sum_minmax", [1, 1, 2])
        rows = deepcopy(self.rows)
        for row in rows:
            if row["variant"] == target:
                row.update(precision10_hits=2501, **{"Precision@10": .5002,
                                                   "mAP@100": 0., "Coverage-nDCG@10": 0.})
            else:
                row.update(**{"mAP@100": 1., "Coverage-nDCG@10": 1.})
        self.assertEqual(self.choose(rows)["winner"], target)
        # Permitted floating summation noise cannot break an integer hit tie.
        for row in rows:
            row["precision10_hits"] = 2500
            row["Precision@10"] = .5 + (1e-13 if row["variant"] == target else 0.)
        self.assertEqual(self.choose(rows)["winner"], self.named("sum_raw", [4, 0, 0]))

    def test_fewer_branches_precede_family_priority(self):
        simple = self.named("sum_minmax", [3, 0, 1])
        complex_ = self.named("sum_raw", [2, 1, 1])
        self.assertEqual(self.tie_only([simple, complex_])["winner"], simple)

    def test_family_priority_precedes_f_weight(self):
        raw = self.named("sum_raw", [1, 0, 3])
        product = self.named("weighted_product", [3, 0, 1])
        minmax = self.named("sum_minmax", [3, 0, 1])
        self.assertEqual(self.tie_only([raw, product, minmax])["winner"], raw)
        self.assertEqual(self.tie_only([product, minmax])["winner"], product)

    def test_larger_f_then_smaller_w_is_predeclared(self):
        lower_f = self.named("sum_raw", [1, 0, 3])
        higher_f = self.named("sum_raw", [3, 0, 1])
        self.assertEqual(self.tie_only([lower_f, higher_f])["winner"], higher_f)
        lower_w = self.named("sum_raw", [1, 2, 1])
        higher_w = self.named("sum_raw", [1, 1, 2])
        self.assertEqual(self.tie_only([lower_w, higher_w])["winner"], lower_w)

    def test_family_winners_follow_same_valid_only_rule(self):
        target = self.named("weighted_product", [1, 2, 1])
        result = self.tie_only([target])
        self.assertEqual(result["family_winners"]["weighted_product"], target)
        self.assertEqual(result["family_winners"]["sum_raw"], self.named("sum_raw", [4, 0, 0]))
        self.assertEqual(result["winner"], target)

    def test_reference_can_win_and_optional_family_is_checked(self):
        result = self.tie_only(["legacy_fgw"])
        self.assertEqual(result["winner"], "legacy_fgw")
        rows = [{k: v for k, v in row.items() if k != "family"} for row in self.rows]
        self.assertEqual(self.choose(rows), self.choose())
        bad = deepcopy(self.rows)
        bad[0]["family"] = "weighted_product"
        with self.assertRaises(ValueError):
            self.choose(bad)

    def test_test_rows_and_test_derived_tie_breaks_rejected(self):
        for split in ("test", "train", "VALID"):
            rows = deepcopy(self.rows)
            rows[-1]["split"] = split
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "valid"):
                self.choose(rows)
        rows = deepcopy(self.rows)
        rows[0]["test_precision"] = 1.
        with self.assertRaises(ValueError):
            self.choose(rows)
        with self.assertRaises(ValueError):
            selection.choose_winner(self.rows, self.specs,
                                    dict(self.settings, tie_rule=["Coverage-nDCG@10"]))

    def test_missing_duplicate_unknown_and_changed_variants_rejected(self):
        bad_cases = [self.rows[:-1], self.rows + [self.rows[0]]]
        duplicate = deepcopy(self.rows)
        duplicate[-1] = deepcopy(duplicate[0])
        bad_cases.append(duplicate)
        unknown = deepcopy(self.rows)
        unknown[0]["variant"] = "unregistered"
        bad_cases.append(unknown)
        for rows in bad_cases:
            with self.subTest(length=len(rows)), self.assertRaises(ValueError):
                self.choose(rows)
        changed = deepcopy(self.specs)
        changed[next(iter(changed))]["weights"][0] += .01
        with self.assertRaises(ValueError):
            selection.choose_winner(self.rows, changed, self.settings)
        with self.assertRaises(ValueError):
            selection.choose_winner(self.rows, self.specs, dict(self.settings, denominator=5))

    def test_inconsistent_counts_or_non_integer_hits_rejected(self):
        cases = [("num_queries", 0), ("num_queries", True), ("num_queries", 499),
                 ("precision10_hits", 2500.), ("precision10_hits", True),
                 ("precision10_hits", -1), ("precision10_hits", 5001),
                 ("Precision@10", .51), ("mAP@100", float("nan")),
                 ("Coverage-nDCG@10", float("inf")), ("Precision@10", True)]
        for key, value in cases:
            rows = deepcopy(self.rows)
            rows[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.choose(rows)

    def test_selection_does_not_mutate_rows_or_settings(self):
        original_rows, original_settings = deepcopy(self.rows), deepcopy(self.settings)
        result = self.choose()
        result["settings"]["families"].clear()
        result["specifications"].clear()
        self.assertEqual(self.rows, original_rows)
        self.assertEqual(self.settings, original_settings)


if __name__ == "__main__":
    unittest.main()
