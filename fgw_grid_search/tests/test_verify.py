"""Independent formulas and full-pool validation reject plausible corruptions."""
from copy import deepcopy
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from fgw_grid_search import pipeline, scoring, verify
from fgw_grid_search.tests.test_pipeline import fixture, evaluate
from wbm_retrieval.common import read_json, write_json


def spec(family, ticks=(1, 1, 0)):
    return {"family": family, "weights": [x / sum(ticks) for x in ticks],
            "ticks": list(ticks), "denominator": sum(ticks)}


class VerificationTests(unittest.TestCase):
    def test_formulas_minmax_constant_product_zero_and_pure_f(self):
        components = {"F": np.array([.2, .5, .8]), "G": np.array([.4, .4, .4]), "WBBS": np.array([0., .2, .9])}
        np.testing.assert_allclose(verify.independent_score(components, spec("sum_raw")), [.3, .45, .6])
        np.testing.assert_allclose(verify.independent_score(components, spec("sum_minmax")), [0., .25, .5])
        np.testing.assert_allclose(verify.independent_score(components, spec("weighted_product", (1, 0, 1))), np.sqrt(components["F"] * components["WBBS"]))
        for family in ("sum_raw", "sum_minmax", "weighted_product"):
            np.testing.assert_array_equal(verify.independent_score(components, spec(family, (1, 0, 0))), components["F"])
        for bad in (spec("sum_raw", (0, 1, 1)), {"family": "legacy_fgw", "weights": [.5, .4, 1.]}, dict(spec("sum_raw"), ticks=[True, 1, 0])):
            with self.assertRaises(ValueError):
                verify.independent_score(components, bad)
        invalid = dict(components, WBBS=np.array([0., float("nan"), .2]))
        with self.assertRaises(ValueError):
            verify.independent_score(invalid, spec("sum_raw", (1, 0, 0)))

    def test_verifier_does_not_call_production_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = fixture(tmp)
            index = evaluate(Path(tmp) / "run", values)
            pool, bundle, settings, variants, provider = values
            with patch.object(scoring, "score_variant", side_effect=AssertionError("production scoring called")):
                for name in variants:
                    checked = verify.verify_evaluation(Path(tmp) / "run/valid" / name, pool, bundle, settings["ks"], index["variant_signatures"][name], provider)
                    self.assertEqual(checked["status"], "verified")
                    self.assertLess(checked["max_metric_absolute_error"], 1e-12)

    def test_metric_and_saved_component_corruption_rejected_without_completed_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = fixture(tmp)
            output = Path(tmp) / "run"
            index = evaluate(output, values)
            pool, bundle, settings, variants, provider = values
            name = next(iter(variants))
            folder = output / "valid" / name
            (folder / "completed.json").unlink()
            def check():
                return verify.verify_evaluation(folder, pool, bundle, settings["ks"], index["variant_signatures"][name], provider)
            self.assertEqual(check()["status"], "verified")
            original = read_json(folder / "coverage_metrics.json")
            damaged = deepcopy(original)
            damaged["metrics"]["Coverage-nDCG@10"] -= .1
            write_json(folder / "coverage_metrics.json", damaged)
            with self.assertRaisesRegex(ValueError, "Metric mismatch"):
                check()
            write_json(folder / "coverage_metrics.json", original)
            path = folder / "rankings.csv"
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle); fields = reader.fieldnames; rows = list(reader)
            rows[0]["global_score"] = "0.123"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "Saved component differs"):
                check()

    def test_unsaved_candidate_change_is_detected_by_full_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = fixture(tmp)
            output = Path(tmp) / "run"
            index = evaluate(output, values)
            pool, bundle, settings, variants, provider = values
            name = next(k for k, v in variants.items() if v["family"] == "sum_raw")
            folder = output / "valid" / name
            with (folder / "rankings.csv").open(newline="") as handle:
                top = {int(r["candidate_id"]) for r in csv.DictReader(handle) if int(r["query_id"]) == 0}
            missing = next(c for c in pool[0] if int(c) not in top)
            original = provider.scores
            def changed(query, candidates):
                components = original(query, candidates)
                if query == 0:
                    for key in components:
                        components[key][candidates == missing] = 1.
                return components
            with patch.object(provider, "scores", side_effect=changed):
                with self.assertRaisesRegex(ValueError, "full-candidate ordering"):
                    verify.verify_evaluation(folder, pool, bundle, settings["ks"], index["variant_signatures"][name], provider)


if __name__ == "__main__":
    unittest.main()
