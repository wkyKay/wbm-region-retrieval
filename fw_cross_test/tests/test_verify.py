"""Numerical and provenance checks for F-only/F*WBBS cross verification."""
from copy import deepcopy
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.proposed.f_segmentation_geometry import coverage_metrics
from fgw_cross_test.tests.test_verify import Pool, Provider, rewrite_csv, write_json
from fw_cross_test import verify
from wbm_retrieval.exact_metrics import order_scores


SPECS = {"F": {"family": "local_f", "formula": "F"},
         "FW": {"family": "product_fw", "formula": "F*WBBS"}}


def fixture(root, variant="FW", *, wrong_formula=None, tie_scores=False):
    root = Path(root)
    labels = np.array([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0],
                       [1, 0, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 0, 0],
                       [0, 0, 0, 1, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0],
                       [1, 1, 0, 0, 0], [0, 0, 0, 0, 1], [1, 1, 0, 0, 0],
                       [0, 0, 1, 1, 0]], dtype=int)
    taxonomy = {key: i for i, key in enumerate(sorted(set(map(tuple, labels))))}
    bundle = SimpleNamespace(labels=labels, class_ids=np.array([taxonomy[tuple(row)] for row in labels]))
    pool, ks = Pool(labels), [1, 3, 5]
    provider = Provider(root, pool)
    if tie_scores:
        for values in provider.values.values():
            values[:] = .4
    settings = {"ks": ks, "tie_seed": 2026}
    frozen = {"variants": ["F", "FW"], "specs": SPECS, "primary_variant": None,
              "selection_rule": "user_specified_posthoc_components",
              "test_parameter_search": False, "settings": settings}
    frozen["signature"] = verify._digest(frozen)
    frozen_path = root / "frozen_variants.json"
    write_json(frozen_path, frozen)
    folder = root / "evaluation"
    folder.mkdir()
    identity = {"run_id": "20261120T010203+0800__fixture", "started_at": "2026-11-20T01:02:03+08:00",
                "evaluation_signature": "evaluation_signature", "variant": variant, "split": "test"}
    manifest = {**identity, "signature": "evaluation_signature", "spec": SPECS[variant],
                "settings": settings, "ks": ks, "tie_seed": 2026, "num_queries": len(pool),
                "formal_protocol": False, "training_required": False,
                "posthoc_research": True, "test_parameter_search": False,
                "provider_source": {"signature": provider.signature, "completed_sha256": verify._sha(provider.marker)},
                "candidate_rule": verify.RULE, "derived_protocol": pool.metadata,
                "retrieval_protocol_signature": pool.protocol_signature,
                "base_protocol_signature": provider.protocol_signature,
                "frozen_variants": {"path": str(frozen_path), "sha256": verify._sha(frozen_path)}}
    write_json(folder / "run_identity.json", identity)
    write_json(folder / "evaluation_manifest.json", manifest)
    fields = ["query_id", "rank", "candidate_id", "similarity_score", "optimal_f_score", "global_score", "wbbs_score"]
    with (folder / "rankings.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for query in pool:
            candidates = pool[query]
            values = provider.scores(query, candidates)
            scores = values["F"].copy() if variant == "F" else values["F"] * values["WBBS"]
            if wrong_formula == "sqrt":
                scores = np.sqrt(scores)
            elif wrong_formula == "with_g":
                scores = scores * values["G"]
            elif wrong_formula == "minmax":
                scores = (values["F"] - values["F"].min()) / np.ptp(values["F"]) * values["WBBS"]
            order = order_scores(candidates, scores, query, 2026)[:max(ks)]
            for rank, i in enumerate(order, 1):
                writer.writerow((query, rank, int(candidates[i]), float(scores[i]),
                                 float(values["F"][i]), float(values["G"][i]), float(values["WBBS"][i])))
    coverage_metrics.evaluate_rankings_file(bundle, pool, folder / "rankings.csv", ks, folder,
        split_name="test", metadata={"signature": "evaluation_signature",
                                     "retrieval_protocol_signature": pool.protocol_signature})
    return folder, pool, bundle, ks, provider


def check(values):
    folder, pool, bundle, ks, provider = values
    return verify.verify_evaluation(folder, pool, bundle, ks, "evaluation_signature", provider)


class FWVerificationTests(unittest.TestCase):
    def test_both_variants_full_metrics_verified_without_production_evaluator(self):
        for variant in SPECS:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as root:
                values = fixture(root, variant)
                with patch.object(coverage_metrics, "query_metrics", side_effect=AssertionError("production metric called")):
                    result = check(values)
                self.assertEqual(result["status"], "verified")
                self.assertEqual(result["num_queries"], 13)
                self.assertEqual(result["ranking_rows"], 65)
                self.assertEqual(result["max_saved_score_absolute_error"], 0.)
                self.assertEqual(result["metric_query_counts"]["Coverage-nDCG@1"], 12)
                self.assertLess(result["max_metric_absolute_error"], 1e-12)

    def test_literal_product_rejects_sqrt_g_or_normalization(self):
        for mistake in ("sqrt", "with_g", "minmax"):
            with self.subTest(mistake=mistake), tempfile.TemporaryDirectory() as root:
                with self.assertRaisesRegex(ValueError, "score formula|full-candidate ordering"):
                    check(fixture(root, wrong_formula=mistake))

    def test_g_has_no_effect_and_zeros_and_f_endpoint_remain_exact(self):
        components = {"F": np.array([0., .5, .8]), "G": np.array([.1, .4, .8]), "WBBS": np.array([.9, 0., .9])}
        before = {k: v.copy() for k, v in components.items()}
        np.testing.assert_array_equal(verify.independent_score(components, SPECS["F"]), components["F"])
        expected = verify.independent_score(components, SPECS["FW"])
        np.testing.assert_array_equal(expected, components["F"] * components["WBBS"])
        for key, value in before.items():
            np.testing.assert_array_equal(components[key], value)
        components["G"][:] = 1.
        np.testing.assert_array_equal(verify.independent_score(components, SPECS["FW"]), expected)
        with self.assertRaisesRegex(ValueError, "Only the user-specified"):
            verify.independent_score(components, {"family": "weighted_product", "weights": [.5, 0., .5]})

    def test_ties_require_frozen_hash_order(self):
        with tempfile.TemporaryDirectory() as root:
            values = fixture(root, tie_scores=True)
            self.assertEqual(check(values)["status"], "verified")
            def swap(rows):
                for field in ("candidate_id", "similarity_score", "optimal_f_score", "global_score", "wbbs_score"):
                    rows[0][field], rows[1][field] = rows[1][field], rows[0][field]
            rewrite_csv(values[0] / "rankings.csv", swap)
            with self.assertRaisesRegex(ValueError, "full-candidate ordering"):
                check(values)

    def test_component_and_unsaved_full_pool_score_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            values = fixture(root)
            folder, pool, _, _, provider = values
            with (folder / "rankings.csv").open(newline="") as stream:
                returned = {int(row["candidate_id"]) for row in csv.DictReader(stream) if int(row["query_id"]) == 0}
            absent = next(c for c in pool[0] if int(c) not in returned)
            provider.values["F"][0, absent] = provider.values["WBBS"][0, absent] = 1.
            with self.assertRaisesRegex(ValueError, "full-candidate ordering"):
                check(values)
        with tempfile.TemporaryDirectory() as root:
            values = fixture(root)
            rewrite_csv(values[0] / "rankings.csv", lambda rows: rows[0].update(global_score="0.12345678"))
            with self.assertRaisesRegex(ValueError, "Saved component"):
                check(values)

    def test_candidate_and_formal_cutoff_guards(self):
        with tempfile.TemporaryDirectory() as root:
            values = fixture(root)
            original = Pool.__getitem__
            def missing(self, query):
                candidates = original(self, query)
                return candidates[1:] if query == 0 else candidates
            with patch.object(Pool, "__getitem__", missing):
                with self.assertRaisesRegex(ValueError, "candidate identities"):
                    check(values)
            path = values[0] / "evaluation_manifest.json"
            manifest = verify._json(path)
            manifest["formal_protocol"] = True
            write_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "all 7199"):
                check(values)

    def test_spec_posthoc_source_and_metric_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            values = fixture(root)
            path = values[0] / "evaluation_manifest.json"
            original = verify._json(path)
            for key, changed in (("spec", SPECS["F"]), ("test_parameter_search", True), ("posthoc_research", False)):
                damaged = deepcopy(original)
                damaged[key] = changed
                write_json(path, damaged)
                with self.assertRaisesRegex(ValueError, "two user-specified"):
                    check(values)
            write_json(path, original)
            path = values[0] / "coverage_metrics.json"
            original = verify._json(path)
            damaged = deepcopy(original)
            damaged["groups"]["mixed_label"]["metrics"]["MeanCoverage@3"] += .01
            write_json(path, damaged)
            with self.assertRaisesRegex(ValueError, "Metric mismatch"):
                check(values)
            damaged = deepcopy(original)
            damaged["metric_query_counts"]["Coverage-nDCG@1"] = 13
            write_json(path, damaged)
            with self.assertRaisesRegex(ValueError, "effective query count"):
                check(values)
            write_json(path, original)
            write_json(values[0] / "metrics.json", {})
            with self.assertRaisesRegex(ValueError, "exact reports"):
                check(values)


if __name__ == "__main__":
    unittest.main()
