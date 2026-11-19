"""Independent fusion formulas, full-pool identity and metric corruption checks."""
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from fgw_fusion import verify
from experiments.proposed.f_global_fusion import pipeline as fusion
from experiments.proposed.f_segmentation_geometry import pipeline as geo
from wbm_retrieval.common import read_json, write_json
from wbm_retrieval.exact_metrics import evaluate_rankings, order_scores
from wbm_retrieval.retrieval_protocol import FullGalleryPool


def spec(lw=.5, gw=.5):
    return {"local_weight": lw, "global_weight": gw, "wbbs_power": 1.,
            "angular_gamma": 0., "local_mode": "optimal"}


def fixture(folder):
    ids = np.arange(12, dtype=np.int64)
    labels = np.zeros((12, 8), dtype=int)
    labels[:, 0] = 1
    labels[[1, 3, 5, 7], 1] = 1
    labels[[2, 4, 6, 8], 0] = 0
    labels[[2, 4, 6, 8], 2] = 1
    _, classes = np.unique(labels, axis=0, return_inverse=True)
    bundle = SimpleNamespace(labels=labels, class_ids=classes, valid_ids=ids, reserved_ids=ids)
    pool = FullGalleryPool([0, 1, 2], ids, "valid", "synthetic_protocol")
    ks = [1, 5, 10]
    fields = ["query_id", "rank", "candidate_id", "similarity_score", "optimal_f_score", "global_score", "wbbs_score"]
    folder.mkdir(parents=True)
    with (folder / "rankings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for q in pool:
            candidates = np.asarray(pool[q])
            f = (candidates % 4) / 3
            g = ((candidates + q) % 5) / 4
            w = (candidates % 3) / 2
            scores = (.5*f + .5*g) * w
            order = order_scores(candidates, scores, q, 2026)[:10]
            for rank, i in enumerate(order, 1):
                writer.writerow(dict(zip(fields, (q, rank, candidates[i], scores[i], f[i], g[i], w[i]))))
    identity = {"schema_version": 1, "key": "fixture_FGW_valid", "split": "valid", "variant": "FGW",
                "run_id": "20261119T180000+0800__fixture_FGW_valid", "started_at": "2026-11-19T18:00:00+08:00"}
    write_json(folder / "run_identity.json", identity)
    write_json(folder / "evaluation_manifest.json", {**identity, "signature": "fixture_signature", "spec": spec(), "ks": ks})
    metadata = {"signature": "fixture_signature", "retrieval_protocol_signature": pool.protocol_signature}
    evaluate_rankings(bundle, pool, fusion.iter_rankings(folder / "rankings.csv"), ks, folder,
                      split_name="valid", metadata=metadata)
    geo.evaluate_rankings_file(bundle, pool, folder / "rankings.csv", ks, folder,
                               split_name="valid", metadata=metadata)
    geo.commit_marker(folder, "fixture_signature")
    return pool, bundle, ks


class VerificationTests(unittest.TestCase):
    def test_exact_fixed_formulas_and_zero_wbbs(self):
        rows = [{"optimal_f_score": .6, "global_score": .8, "wbbs_score": .2},
                {"optimal_f_score": 1., "global_score": 1., "wbbs_score": 0.}]
        np.testing.assert_allclose(verify.independent_saved_score(rows, spec()), [.14, 0], atol=1e-16)
        np.testing.assert_allclose(verify.independent_saved_score(rows, spec(0., 1.)), [.16, 0], atol=1e-16)
        np.testing.assert_allclose(verify.independent_saved_score(rows, spec(1., 0.)), [.12, 0], atol=1e-16)
        for bad in (dict(spec(), angular_gamma=.5), dict(spec(), wbbs_power=2.), spec(0., .5),
                    dict(spec(), local_mode="greedy"), dict(spec(), local_weight=True)):
            with self.assertRaises(ValueError):
                verify.independent_saved_score(rows, bad)
        for value in (-1e-14, 1.001, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                verify.independent_saved_score([dict(rows[0], wbbs_score=value)], spec())

    def test_full_pool_omission_and_close_numeric_rank_changes_are_not_accepted(self):
        result = verify.compare_full_ranking(1, [11, 12, 13], np.array([.8, .7, .9]), [11, 12])
        self.assertFalse(result["exact_order_match"])
        self.assertEqual(result["changes"][0]["recomputed_id"], 13)
        result = verify.compare_full_ranking(1, [11, 12], np.array([.5, .5+1e-14]), [11, 12])
        self.assertFalse(result["exact_order_match"])
        self.assertTrue(result["numeric_only"])

    def test_complete_reports_all_query_all_k_and_group_aggregates(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "FGW"
            pool, bundle, ks = fixture(folder)
            result = verify.verify_evaluation(folder, pool, bundle, ks, "fixture_signature")
            self.assertEqual(result["status"], "verified")
            self.assertEqual(result["num_queries"], 3)
            self.assertEqual(result["ranking_rows"], 30)
            self.assertGreater(result["aggregation_values_checked"], 100)
            self.assertLess(result["max_metric_absolute_error"], 1e-12)

    def test_hash_mutation_then_resigned_metric_mutation_are_both_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "FGW"
            pool, bundle, ks = fixture(folder)
            report = read_json(folder / "metrics.json")
            report["metrics"]["Precision@10"] += .1
            write_json(folder / "metrics.json", report)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify.verify_evaluation(folder, pool, bundle, ks, "fixture_signature")
            geo.commit_marker(folder, "fixture_signature")
            with self.assertRaisesRegex(ValueError, "Metric mismatch"):
                verify.verify_evaluation(folder, pool, bundle, ks, "fixture_signature")

    def test_wrong_component_is_detected_even_when_hashes_are_resealed(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "FGW"
            pool, bundle, ks = fixture(folder)
            path = folder / "rankings.csv"
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle); fields = reader.fieldnames; rows = list(reader)
            rows[0]["wbbs_score"] = "0"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            geo.commit_marker(folder, "fixture_signature")
            with self.assertRaises(AssertionError):
                verify.verify_evaluation(folder, pool, bundle, ks, "fixture_signature")

    def test_incomplete_candidate_pool_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "FGW"
            pool, bundle, ks = fixture(folder)
            class BadPool:
                gallery = pool.gallery
                candidates_per_query = pool.candidates_per_query
                split_name = pool.split_name
                def __iter__(self): return iter(pool)
                def __getitem__(self, q): return np.asarray(pool[q])[:-1]
            with self.assertRaisesRegex(ValueError, "complete gallery except self"):
                verify.verify_evaluation(folder, BadPool(), bundle, ks, "fixture_signature")


if __name__ == "__main__":
    unittest.main()
