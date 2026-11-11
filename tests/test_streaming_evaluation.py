"""Full-gallery evaluation consumes one query block and keeps legacy scores."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from wbm_retrieval import evaluation
from wbm_retrieval.common import csv_rows, write_csv
from wbm_retrieval.exact_metrics import order_scores
from wbm_retrieval.retrieval_protocol import FullGalleryPool


class StreamingEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.pool = FullGalleryPool([1, 3, 4], [1, 2, 3, 4, 5, 6], "test", "fixture")
        labels = {1: [1, 0], 2: [1, 0], 3: [0, 1], 4: [1, 1], 5: [1, 1], 6: [0, 1]}
        self.protocol = SimpleNamespace(queries=list(self.pool), candidates=self.pool, labels=labels)
        self.settings = {"ks": [1, 3, 5], "bootstrap_samples": 20}
        self.rows = []
        for q, candidates in self.pool.items():
            scored = [(int(c), float((q + 2 * int(c)) % 3)) for c in candidates]
            scored.sort(key=lambda value: (-value[1], value[0]))
            self.rows += [{"query_id": q, "rank": rank, "candidate_id": c, "similarity_score": score}
                          for rank, (c, score) in enumerate(scored, 1)]

    def write(self, path, rows):
        write_csv(path, rows, ["query_id", "rank", "candidate_id", "similarity_score"])

    def test_streamed_full_gallery_matches_legacy_metrics_and_query_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = SimpleNamespace(queries=self.protocol.queries,
                                     candidates={q: self.pool[q].tolist() for q in self.pool},
                                     labels=self.protocol.labels)
            for name in ("streamed", "legacy"):
                self.write(root / name / "rankings.csv", self.rows)
            result = evaluation.evaluate_ranking(root / "streamed/rankings.csv", self.protocol, self.settings, 2026)
            expected = evaluation.evaluate_ranking(root / "legacy/rankings.csv", legacy, self.settings, 2026)
            self.assertEqual(result["metrics"], expected["metrics"])
            self.assertEqual(result["query_bootstrap_95ci"], expected["query_bootstrap_95ci"])
            self.assertEqual(result["num_queries"], 3)
            self.assertEqual(csv_rows(root / "streamed/per_query_metrics.csv"),
                             csv_rows(root / "legacy/per_query_metrics.csv"))

    def test_evaluates_completed_query_before_consuming_the_whole_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rankings.csv"
            self.write(path, self.rows)
            with mock.patch.object(evaluation, "query_metrics", wraps=evaluation.query_metrics) as score_query:
                def records(_):
                    for index, row in enumerate(self.rows):
                        if index == 10:
                            self.assertGreaterEqual(score_query.call_count, 1,
                                                    "Full-gallery ranks were buffered for every query")
                        yield row
                with mock.patch.object(evaluation, "iter_csv", side_effect=records):
                    evaluation.evaluate_ranking(path, self.protocol, self.settings, 2026)
                self.assertEqual(score_query.call_count, 3)

    def test_missing_query_and_repeated_query_block_are_rejected(self):
        cases = {"missing": self.rows[:10],
                 "repeated_block": self.rows[:10] + self.rows[:5] + self.rows[10:],
                 "self_candidate": [{**self.rows[0], "candidate_id": self.rows[0]["query_id"]}] + self.rows[1:],
                 "nonfinite_score": [{**self.rows[0], "similarity_score": np.nan}] + self.rows[1:]}
        with tempfile.TemporaryDirectory() as directory:
            for name, rows in cases.items():
                with self.subTest(case=name):
                    path = Path(directory) / name / "rankings.csv"
                    self.write(path, rows)
                    with self.assertRaises(ValueError):
                        evaluation.evaluate_ranking(path, self.protocol, self.settings, 2026)
                    self.assertFalse((path.parent / "metrics.json").exists())

    def test_exact_csv_accepts_sha_tied_top_k_and_paired_ap_column(self):
        settings = {"metric_version": "exact_v1", "ks": [1, 3], "tie_seed": 2026, "bootstrap_samples": 0}
        rows = []
        for q in self.pool:
            ids = self.pool[q]
            ranked = ids[order_scores(ids, np.zeros(len(ids)), q)][:3]
            rows.extend({"query_id": q, "rank": rank, "candidate_id": int(c), "similarity_score": 0.}
                        for rank, c in enumerate(ranked, 1))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = []
            for method in ("proposed", "handcrafted"):
                path = root / "retrieval" / method / "rankings.csv"
                self.write(path, rows)
                reports.append(evaluation.evaluate_ranking(path, self.protocol, settings, 2026))
                self.assertEqual(reports[-1]["metric_version"], "exact_v1")
                self.assertEqual(reports[-1]["num_queries"], 3)
                per_query = csv_rows(path.with_name("per_query.csv"))
                self.assertTrue(all(row["returned"] == "3" and row["pool_size"] == "5" for row in per_query))
                self.assertTrue(all(row["relevant_in_pool"] == "1" for row in per_query))
                self.assertTrue(all("AP@3" in row for row in per_query))
            self.assertEqual(reports[0]["metrics"], reports[1]["metrics"])
            paired = evaluation.compare_paired({"output_dir": str(root), "evaluation": settings, "seed": 2026},
                                               metric="mAP@3")
            self.assertEqual(paired["metric"], "mAP@3")
            self.assertEqual(paired["mean_difference"], 0.)

    def test_exact_csv_rejects_invalid_ids_rank_numbers_and_wrong_sha_order(self):
        settings = {"metric_version": "exact_v1", "ks": [1, 3], "tie_seed": 2026, "bootstrap_samples": 0}
        rows = []
        for q in self.pool:
            ids = self.pool[q]
            ranked = ids[order_scores(ids, np.zeros(len(ids)), q)][:3]
            rows.extend({"query_id": q, "rank": rank, "candidate_id": int(c), "similarity_score": 0.}
                        for rank, c in enumerate(ranked, 1))
        wrong_tie = [{**rows[0], "candidate_id": rows[1]["candidate_id"]},
                     {**rows[1], "candidate_id": rows[0]["candidate_id"]}] + rows[2:]
        cases = {"duplicate": [rows[0], {**rows[1], "candidate_id": rows[0]["candidate_id"]}] + rows[2:],
                 "self": [{**rows[0], "candidate_id": rows[0]["query_id"]}] + rows[1:],
                 "unknown": [{**rows[0], "candidate_id": 999}] + rows[1:],
                 "rank": [{**rows[0], "rank": 2}] + rows[1:], "tie": wrong_tie,
                 "missing_query": rows[:6]}
        with tempfile.TemporaryDirectory() as directory:
            for name, records in cases.items():
                with self.subTest(case=name):
                    path = Path(directory) / name / "rankings.csv"
                    self.write(path, records)
                    with self.assertRaises(ValueError):
                        evaluation.evaluate_ranking(path, self.protocol, settings, 2026)
                    self.assertFalse(path.with_name("metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
