"""Frozen-valid boundary and real synthetic test evaluation through the public API."""
from copy import deepcopy
import multiprocessing
from pathlib import Path
import tempfile
import unittest

from fgw_grid_search import scoring, selection
from fgw_grid_search.state import signed
from fgw_grid_search.tests.test_pipeline import fixture, snapshot
from fgw_grid_search.tests.test_run_state import grid_rows
from fgw_topk_test import run, state
from wbm_retrieval.common import read_json


NAMES = [
    "sum_raw_d10_f01_g04_w05", "sum_minmax_d10_f02_g02_w06",
    "sum_raw_d10_f02_g03_w05", "sum_raw_d10_f01_g03_w06",
    "sum_raw_d10_f03_g03_w04", "sum_raw_d10_f03_g02_w05",
    "sum_minmax_d10_f03_g02_w05", "weighted_product_d10_f01_g05_w04",
    "weighted_product_d10_f02_g05_w03", "weighted_product_d10_f04_g03_w03",
]


def frozen_input():
    settings = scoring.validate_settings({})
    specs = scoring.specs(settings)
    rows = grid_rows(specs)
    for row in rows:
        if row["variant"] in NAMES:
            row["precision10_hits"] = 2800 - NAMES.index(row["variant"])
            row["Precision@10"] = row["precision10_hits"] / 5000
    decision = selection.choose_winner(rows, specs, settings)
    return signed({"version": "fgw_grid_search_v1", "parent_signature": "synthetic_parent",
        "settings_sha256": "synthetic_settings", "evaluation_index_sha256": "synthetic_index",
        "selected_spec": specs[decision["winner"]], **decision})


class FrozenSelectionTests(unittest.TestCase):
    def test_exact_ten_and_order_come_from_valid_and_original_input_unchanged(self):
        original = frozen_input()
        before = deepcopy(original)
        frozen = state.freeze_top10(original)
        self.assertEqual(frozen["ranked_variants"], NAMES)
        self.assertEqual(frozen["primary_variant"], NAMES[0])
        self.assertEqual(list(frozen["specs"]), NAMES)
        self.assertEqual(frozen["valid_selection"], original)
        self.assertFalse(frozen["test_parameter_search"])
        self.assertEqual(original, before)

    def test_forged_or_reordered_ranking_and_winner_are_rejected(self):
        for kind in ("reorder", "winner", "test_row", "duplicate", "missing"):
            value = deepcopy(frozen_input())
            value.pop("signature")
            if kind == "reorder": value["ranking"][0], value["ranking"][1] = value["ranking"][1], value["ranking"][0]
            if kind == "winner": value["winner"] = NAMES[1]
            if kind == "test_row": value["ranking"][0]["split"] = "test"
            if kind == "duplicate": value["ranking"][1] = deepcopy(value["ranking"][0])
            if kind == "missing": value["ranking"].pop()
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                state.freeze_top10(signed(value))

    def test_nonpositive_f_or_changed_formula_cannot_enter_fixed_test(self):
        value = deepcopy(frozen_input()); value.pop("signature")
        value["specifications"][NAMES[0]]["weights"][0] = 0
        with self.assertRaises(ValueError): state.freeze_top10(signed(value))

    def test_tenth_place_tie_keeps_predeclared_policy_without_expanding_test(self):
        value = frozen_input()
        fields = ("variant", "family", "split", "num_queries", "precision10_hits",
                  "Precision@10", "mAP@100", "Coverage-nDCG@10")
        rows = [{k: r[k] for k in fields} for r in value["ranking"]]
        tied = "weighted_product_d10_f02_g04_w04"
        for row in rows:
            if row["variant"] == tied:
                row["precision10_hits"] = 2791
                row["Precision@10"] = 2791 / 5000
        decision = selection.choose_winner(rows, value["specifications"], value["settings"])
        original = signed({"selected_spec": value["selected_spec"], **decision})
        frozen = state.freeze_top10(original)
        self.assertEqual(frozen["ranked_variants"], NAMES)
        self.assertNotIn(tied, frozen["ranked_variants"])


class FixedEvaluationTests(unittest.TestCase):
    def prepare(self, root):
        pool, bundle, _, _, provider = fixture(root, split="test")
        frozen = state.freeze_top10(frozen_input())
        return pool, bundle, provider, frozen

    def evaluate(self, output, values, workers=1):
        pool, bundle, provider, frozen = values
        return run.evaluate_fixed(output, "synthetic_top10_parent", provider, pool, bundle,
            frozen["specs"], frozen["valid_selection"]["settings"], workers=workers)

    def folders(self, output, frozen):
        return {name: output / "evaluations" / (output.name + "__" + name) / "test" / name
                for name in frozen["ranked_variants"]}

    def test_ten_real_reports_full_pool_and_immutable_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); values = self.prepare(root)
            output = root / "fgw_topk_test_one"
            result = self.evaluate(output, values)
            self.assertEqual([row["variant"] for row in result["summary_rows"]], NAMES)
            for folder in self.folders(output, values[3]).values():
                checked = read_json(folder / "verification.json")
                self.assertEqual(checked["num_queries"], 3)
                self.assertEqual(checked["full_candidates_checked"], 3 * 502)
                self.assertTrue(checked["all_queries_full_candidate_topk_exact"])
                for filename in ("metrics.json", "coverage_metrics.json"):
                    report = read_json(folder / filename)
                    self.assertEqual(report["split"], "test")
                    self.assertEqual(report["ks"], [1, 5, 10, 20, 50, 100, 500])
            before = snapshot(output)
            self.assertEqual(self.evaluate(output, values), result)
            self.assertEqual(snapshot(output), before)

    @unittest.skipUnless("fork" in multiprocessing.get_all_start_methods(), "requires fork")
    def test_parallel_scores_identical_and_batch_identities_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); values = self.prepare(root)
            a, b = root / "fgw_topk_test_batch_a", root / "fgw_topk_test_batch_b"
            first = self.evaluate(a, values, workers=1)
            second = self.evaluate(b, values, workers=2)
            self.assertEqual(first["summary_rows"], second["summary_rows"])
            af, bf = self.folders(a, values[3]), self.folders(b, values[3])
            keys = []
            for name in NAMES:
                self.assertEqual((af[name] / "rankings.csv").read_bytes(), (bf[name] / "rankings.csv").read_bytes())
                for folder in (af[name], bf[name]): keys.append(read_json(folder / "run_identity.json")["key"])
            self.assertEqual(len(set(keys)), 20)

    def test_changed_source_resume_fails_without_rewriting_completed_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); values = self.prepare(root)
            output = root / "fgw_topk_test_resume"
            self.evaluate(output, values)
            before = snapshot(output)
            values[2].signature = "changed_source"
            with self.assertRaises(ValueError): self.evaluate(output, values)
            self.assertEqual(snapshot(output), before)


if __name__ == "__main__":
    unittest.main()
