"""Real evaluator fixtures with independent full-gallery corruption checks."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from experiments.proposed.f_segmentation_geometry.coverage_metrics import evaluate_rankings_file
from f_region_structure.verify import verify_evaluation
from wbm_retrieval.exact_metrics import evaluate_rankings, order_scores
from wbm_retrieval.retrieval_protocol import FullGalleryPool


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False))


def read_csv(path):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fixture(folder, *, only_single=False):
    folder = Path(folder)
    ids = np.arange(2, 34, 2, dtype=np.int64)  # original IDs are deliberately not row indices
    labels = np.zeros((34, 8), dtype=np.int64)
    first = np.eye(8, dtype=np.int64)[0]
    second = np.eye(8, dtype=np.int64)[1]
    mixed = first + second
    unique = np.eye(8, dtype=np.int64)[7]
    values = [first, second, mixed, unique] + [first] * 4 + [second] * 4 + [mixed] * 4
    labels[ids] = np.asarray(values)
    _, classes = np.unique(labels, axis=0, return_inverse=True)
    bundle = SimpleNamespace(labels=labels, class_ids=classes, valid_ids=ids, reserved_ids=ids)
    queries = ids[[0, 1, 3]] if only_single else ids[:4]
    pool = FullGalleryPool(queries, ids, "valid", "tiny_complete_protocol")
    ks, signature = [1, 5, 10], "tiny_region_evaluation"
    matrix = {(int(q), int(c)): float(((int(c)//2 + int(q)) % 5)/5) for q in pool for c in pool[q]}
    def provider(query):
        candidates = pool[query][::-1]  # independent ordering must not assume source row order
        return candidates, np.array([matrix[int(query), int(c)] for c in candidates])
    rankings, csv_rows = [], []
    for query in pool:
        candidates, scores = provider(query)
        order = order_scores(candidates, scores, query, 2026)[:10]
        ranked = candidates[order]
        rankings.append((query, ranked))
        csv_rows.extend({"query_id": query, "rank": rank, "candidate_id": int(c), "similarity_score": float(s)}
                        for rank, (c, s) in enumerate(zip(ranked, scores[order]), 1))
    write_csv(folder / "rankings.csv", ["query_id", "rank", "candidate_id", "similarity_score"], csv_rows)
    metadata = {"signature": signature, "retrieval_protocol_signature": pool.protocol_signature}
    evaluate_rankings(bundle, pool, rankings, ks, folder, metadata=metadata)
    evaluate_rankings_file(bundle, pool, folder / "rankings.csv", ks, folder, metadata=metadata)
    write_json(folder / "evaluation_manifest.json", {"signature": signature, "split": "valid", "ks": ks,
        "tie_seed": 2026, "variant": "synthetic_regions", "spec": {"family": "synthetic"}})
    write_json(folder / "run_identity.json", {"key": "synthetic_no_completion_required"})
    return pool, bundle, ks, signature, provider, matrix


class RegionVerificationTests(unittest.TestCase):
    def check(self, folder, values, provider=None):
        pool, bundle, ks, signature, original, _ = values
        return verify_evaluation(folder, pool, bundle, ks, signature, provider or original)

    def test_real_dual_evaluators_and_all_reports_verify_before_sealing(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = fixture(tmp)
            result = self.check(tmp, values)
            self.assertEqual(result["status"], "verified")
            self.assertEqual(result["num_queries"], 4)
            self.assertEqual(result["ranking_rows"], 40)
            self.assertEqual(result["full_candidates_checked"], 60)
            self.assertLess(result["max_metric_absolute_error"], 1e-12)
            self.assertEqual(result["max_saved_score_absolute_error"], 0.)
            self.assertIsInstance(result["top10_hits"], int)
            report = read_json(Path(tmp) / "metrics.json")
            self.assertAlmostEqual(result["top10_hits"]/40, report["metrics"]["Precision@10"])
            self.assertNotIn("completed.json", result["files"])
            self.assertNotIn("verification.json", result["files"])
            self.assertEqual(len(result["files"]), 9)
            self.assertIn("production full-candidate", result["scope"])
            # The unique label has no exact or coverage-relevant candidates.
            self.assertEqual(report["queries_without_relevant_candidates"], 1)

    def test_empty_mixed_group_and_undefined_metrics_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = fixture(tmp, only_single=True)
            self.assertEqual(self.check(tmp, values)["num_queries"], 3)
            report = read_json(Path(tmp) / "coverage_metrics.json")
            self.assertEqual(report["groups"]["mixed_label"]["num_queries"], 0)
            self.assertIsNone(report["groups"]["mixed_label"]["metrics"]["Coverage-nDCG@10"])

    def test_unsaved_candidate_matrix_change_and_incomplete_provider_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = fixture(tmp)
            pool, _, _, _, provider, matrix = values
            _, rows = read_csv(Path(tmp) / "rankings.csv")
            query = next(iter(pool))
            saved = {int(row["candidate_id"]) for row in rows if int(row["query_id"]) == query}
            missing = next(int(c) for c in pool[query] if c not in saved)
            matrix[query, missing] = 1.
            with self.assertRaisesRegex(ValueError, "full-candidate ordering"):
                self.check(tmp, values)
            def incomplete(q):
                ids, scores = provider(q)
                return ids[:-1], scores[:-1]
            with self.assertRaisesRegex(ValueError, "complete gallery"):
                self.check(tmp, values, incomplete)

    def test_ranking_order_saved_score_and_repeated_query_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = fixture(tmp)
            path = Path(tmp) / "rankings.csv"
            fields, original = read_csv(path)
            damaged = deepcopy(original)
            damaged[0]["candidate_id"], damaged[1]["candidate_id"] = damaged[1]["candidate_id"], damaged[0]["candidate_id"]
            write_csv(path, fields, damaged)
            with self.assertRaisesRegex(ValueError, "full-candidate ordering"):
                self.check(tmp, values)
            damaged = deepcopy(original)
            damaged[0]["similarity_score"] = str(float(damaged[0]["similarity_score"]) + .01)
            write_csv(path, fields, damaged)
            with self.assertRaisesRegex(ValueError, "Saved score"):
                self.check(tmp, values)
            write_csv(path, fields, original + original[:10])
            with self.assertRaisesRegex(ValueError, "repeated"):
                self.check(tmp, values)

    def test_every_aggregate_level_and_metric_counts_reject_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = fixture(tmp)
            path = Path(tmp) / "coverage_metrics.json"
            original = read_json(path)
            paths = [("metrics", "MeanCoverage@10"),
                     ("class_macro_metrics", "FullCoverage_mAP@10"),
                     ("groups", "single_label", "metrics", "Coverage-nDCG@5"),
                     ("groups", "mixed_label", "metrics", "FullCoveragePrecision@1"),
                     ("groups", "by_label_count", "2", "metrics", "MeanCoverage@5"),
                     ("metric_query_counts", "Coverage-nDCG@10"),
                     ("class_macro_metric_class_counts", "FullCoverageRecall@10"),
                     ("groups", "single_label", "num_classes")]
            for keys in paths:
                with self.subTest(path=keys):
                    changed = deepcopy(original)
                    target = changed
                    for key in keys[:-1]:
                        target = target[key]
                    target[keys[-1]] += 1
                    write_json(path, changed)
                    with self.assertRaises(ValueError):
                        self.check(tmp, values)
            write_json(path, original)
            self.assertEqual(self.check(tmp, values)["status"], "verified")

    def test_per_query_denominator_duplicate_and_per_class_value_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = fixture(tmp)
            path = Path(tmp) / "per_query.csv"
            fields, original = read_csv(path)
            damaged = deepcopy(original)
            damaged[0]["relevant_in_pool"] = str(int(damaged[0]["relevant_in_pool"]) + 1)
            write_csv(path, fields, damaged)
            with self.assertRaisesRegex(ValueError, "denominator"):
                self.check(tmp, values)
            write_csv(path, fields, original + original[:1])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                self.check(tmp, values)
            write_csv(path, fields, original)
            path = Path(tmp) / "per_class.csv"
            fields, rows = read_csv(path)
            rows[0]["Precision@10"] = "0.999"
            write_csv(path, fields, rows)
            with self.assertRaisesRegex(ValueError, "Metric mismatch"):
                self.check(tmp, values)

    def test_manifest_protocol_and_query_coverage_are_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = fixture(tmp)
            path = Path(tmp) / "evaluation_manifest.json"
            original = read_json(path)
            for key, value in (("signature", "wrong"), ("split", "test"), ("tie_seed", 1), ("ks", [1, 10])):
                write_json(path, dict(original, **{key: value}))
                with self.assertRaisesRegex(ValueError, "manifest"):
                    self.check(tmp, values)
            write_json(path, original)
            path = Path(tmp) / "coverage_metrics.json"
            report = read_json(path)
            report["metadata"]["retrieval_protocol_signature"] = "wrong"
            write_json(path, report)
            with self.assertRaisesRegex(ValueError, "protocol/signature"):
                self.check(tmp, values)


if __name__ == "__main__":
    unittest.main()
