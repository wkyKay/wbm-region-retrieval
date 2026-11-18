"""Synthetic full-gallery proofs for the frozen J-only / angle-plus-J pair.

Fixtures use original sample IDs, small CPU galleries, and the specified SHA
tie protocol. No real descriptors, rankings, or run metadata are modified.
"""
from contextlib import redirect_stdout
from copy import deepcopy
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from b1_ablation import verify_structure_pair as subject
from b1_structure_fusion.scoring import StructuralDistances
from wbm_retrieval.retrieval_protocol import FullGalleryPool


FIELDS = ("query_id", "rank", "candidate_id", "similarity_score")


def independent_order(query, candidates, scores):
    """Protocol definition, independently of the function under test."""
    ids = np.asarray(candidates, dtype=np.int64)
    keys = np.array([
        int.from_bytes(hashlib.sha256(
            f"exact_v1:tie:2026:{int(query)}:{int(candidate)}".encode("ascii")
        ).digest()[:8], "big") for candidate in ids
    ], dtype=np.uint64)
    return np.lexsort((ids, keys, -np.asarray(scores, dtype=np.float64)))


def ranking_rows(query, candidates, scores, top_k):
    return [{"query_id": str(query), "rank": str(rank),
             "candidate_id": str(int(candidates[index])),
             "similarity_score": repr(float(scores[index]))}
            for rank, index in enumerate(
                independent_order(query, candidates, scores)[:top_k], 1)]


def query_fixture():
    query = 3
    candidates = np.array([10, 18, 41, 50], dtype=np.int64)
    gallery = np.array([3, 10, 18, 41, 50], dtype=np.int64)
    joint = np.array([.12, .20, .35, .65], dtype=np.float64)
    angular = np.array([.9, .05, .2, .7], dtype=np.float64)
    j = ranking_rows(query, candidates, np.exp(-joint), 3)
    aj = ranking_rows(query, candidates, np.exp(-.5 * angular) * np.exp(-2 * joint), 3)
    return query, candidates, gallery, joint, angular, j, aj


def stream_fixture():
    # Unsorted descriptor IDs ensure row numbers cannot substitute for sample IDs.
    ids = np.array([50, 3, 92, 18, 41, 10], dtype=np.int64)
    pool = FullGalleryPool([3, 41], ids, "valid", "synthetic_structure_pair_v1")
    joint = np.zeros((len(ids), 128), dtype=np.float64)
    joint[:, 0] = 1.
    moments = np.array([[.8, .2, .7, .3], [.1, .2, .3, .2], [.9, .9, .9, .9],
                        [.1, .2, .3, .2], [.4, .5, .6, .5], [.3, .2, .1, .2]])
    values = {"sample_ids": ids, "joint128": joint,
              "empty": np.zeros(len(ids), dtype=bool),
              "moments": moments, "outer_count": np.array([2, 4, 0, 7, 5, 3])}
    engine = StructuralDistances(ids, joint, values["empty"], device="cpu")
    positions = {int(sample): row for row, sample in enumerate(ids)}
    j_rows, aj_rows = [], []
    for query in pool:
        candidates = pool[query]
        distance = engine.for_query(query, candidates)["J"]
        qi = positions[query]
        ci = np.array([positions[int(c)] for c in candidates])
        angular = np.abs(moments[ci] - moments[qi]).mean(axis=1)
        angular *= (values["outer_count"][ci] > 0) & (values["outer_count"][qi] > 0)
        j_rows.extend(ranking_rows(query, candidates, np.exp(-distance), 3))
        aj_rows.extend(ranking_rows(query, candidates,
                                    np.exp(-.5 * angular) * np.exp(-2 * distance), 3))
    return pool, values, j_rows, aj_rows


def write_rankings(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def source_snapshot(folder):
    return {str(path.relative_to(folder)):
            (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
            for path in Path(folder).rglob("*") if path.is_file()}


class StructurePairQueryTests(unittest.TestCase):
    def test_scores_include_angle_and_preserve_fixed_lambda_formulas(self):
        joint = np.array([0., .25, 1.])
        angular = np.array([0., .6, 0.])
        scores = subject.compute_scores(joint, angular)
        self.assertEqual(set(scores), {"J_lambda1", "J_lambda2", "AJ"})
        np.testing.assert_array_equal(scores["J_lambda1"], np.exp(-joint))
        np.testing.assert_array_equal(scores["J_lambda2"], np.exp(-2 * joint))
        np.testing.assert_allclose(scores["AJ"], np.exp(-.5 * angular) * np.exp(-2 * joint),
                                   rtol=0, atol=1e-15)
        self.assertLess(scores["AJ"][1], scores["J_lambda2"][1])
        self.assertEqual(scores["AJ"][2], scores["J_lambda2"][2])

    def test_angular_original_ids_and_empty_outer_guard(self):
        values = {"sample_ids": np.array([18, 3, 41, 10]),
                  "moments": np.array([[.9, .8, .9, .8], [.1, .2, .1, .2],
                                       [.7, .6, .7, .6], [.5, .4, .5, .4]]),
                  "outer_count": np.array([0, 2, 3, 4])}
        np.testing.assert_allclose(subject.angular_distance(3, np.array([10, 18, 41]), values),
                                   [.3, 0., .5], rtol=0, atol=1e-15)
        np.testing.assert_array_equal(subject.angular_distance(18, np.array([3, 41, 10]), values),
                                      np.zeros(3))

    def test_complete_candidate_check_accepts_distinct_j_and_aj_orders(self):
        args = query_fixture()
        self.assertNotEqual(args[-2][0]["candidate_id"], args[-1][0]["candidate_id"])
        evidence = subject.check_query(*args, top_k=3)
        self.assertIsInstance(evidence, dict)
        self.assertTrue(evidence)

    def test_omitted_higher_scoring_candidate_outside_saved_topk_is_rejected(self):
        query, candidates, gallery, joint, angular, _, aj = query_fixture()
        joint[-1] = .01
        # All saved J rows remain legal and internally descending, but ID 50 wins.
        bad_j = ranking_rows(query, candidates[:-1], np.exp(-joint[:-1]), 3)
        good_aj = ranking_rows(query, candidates,
                               np.exp(-.5 * angular) * np.exp(-2 * joint), 3)
        with self.assertRaises(subject.VerificationMismatch) as caught:
            subject.check_query(query, candidates, gallery, joint, angular, bad_j, good_aj, top_k=3)
        self.assertIsInstance(caught.exception.evidence, dict)
        self.assertTrue(caught.exception.evidence)

    def test_truncated_candidate_pool_cannot_prove_full_gallery_topk(self):
        query, candidates, gallery, joint, angular, _, _ = query_fixture()
        candidates, joint, angular = candidates[:-1], joint[:-1], angular[:-1]
        j = ranking_rows(query, candidates, np.exp(-joint), 3)
        aj = ranking_rows(query, candidates, np.exp(-.5 * angular) * np.exp(-2 * joint), 3)
        with self.assertRaises(ValueError):
            subject.check_query(query, candidates, gallery, joint, angular, j, aj, top_k=3)

    def test_wrong_sha_order_of_exact_ties_is_rejected(self):
        query, candidates, gallery, _, _, _, _ = query_fixture()
        joint = np.full(4, .25)
        angular = np.zeros(4)
        j = ranking_rows(query, candidates, np.exp(-joint), 3)
        aj = ranking_rows(query, candidates, np.exp(-2 * joint), 3)
        j[0]["candidate_id"], j[1]["candidate_id"] = j[1]["candidate_id"], j[0]["candidate_id"]
        with self.assertRaises(subject.VerificationMismatch):
            subject.check_query(query, candidates, gallery, joint, angular, j, aj, top_k=3)

    def test_lambda_ranking_difference_is_fatal_even_one_float_ulp(self):
        query, candidates, gallery, _, _, _, _ = query_fixture()
        joint, angular = np.full(4, .25), np.zeros(4)
        one = np.full(4, .5)
        order = independent_order(query, candidates, one)
        two = np.full(4, .5)
        two[order[1]] = np.nextafter(.5, 1.)
        fake_scores = {"J_lambda1": one, "J_lambda2": two, "AJ": one.copy()}
        j = ranking_rows(query, candidates, one, 3)
        aj = ranking_rows(query, candidates, one, 3)
        with patch.object(subject, "compute_scores", return_value=fake_scores):
            with self.assertRaises(subject.VerificationMismatch) as caught:
                subject.check_query(query, candidates, gallery, joint, angular, j, aj, top_k=3)
        self.assertIn("lambda_topk_identity_changed", caught.exception.evidence["failure_kinds"])

    def test_saved_rows_reject_identity_rank_and_score_corruption(self):
        for corruption in ("duplicate_candidate", "self", "wrong_query", "wrong_rank",
                           "missing_rank", "score_nan", "wrong_score"):
            with self.subTest(corruption=corruption):
                query, candidates, gallery, joint, angular, j, aj = query_fixture()
                if corruption == "duplicate_candidate":
                    j[1]["candidate_id"] = j[0]["candidate_id"]
                elif corruption == "self":
                    j[0]["candidate_id"] = str(query)
                elif corruption == "wrong_query":
                    j[0]["query_id"] = "999"
                elif corruption == "wrong_rank":
                    j[0]["rank"] = "2"
                elif corruption == "missing_rank":
                    j.pop()
                elif corruption == "score_nan":
                    j[0]["similarity_score"] = "nan"
                else:
                    j[0]["similarity_score"] = ".9999"
                with self.assertRaises(ValueError):
                    subject.check_query(query, candidates, gallery, joint, angular, j, aj, top_k=3)


class StructurePairStreamingTests(unittest.TestCase):
    def make_sources(self, directory, j_rows, aj_rows):
        source = directory / "frozen_sources"
        source.mkdir()
        j_path, aj_path = source / "j_rankings.csv", source / "aj_rankings.csv"
        write_rankings(j_path, j_rows)
        write_rankings(aj_path, aj_rows)
        (source / "metadata.json").write_text(
            json.dumps({"protocol_signature": "synthetic_structure_pair_v1",
                        "candidate_rule": "all_gallery_except_query"}), encoding="utf-8")
        return source, j_path, aj_path

    def assert_failure_artifact(self, output):
        self.assertTrue(output.is_dir())
        failure = output / "failure.json"
        self.assertTrue(failure.is_file(), "Failure must persist evidence in the new output")
        document = json.loads(failure.read_text(encoding="utf-8"))
        self.assertIsInstance(document, dict)
        self.assertTrue(document)

    def test_all_queries_streamed_and_sources_untouched(self):
        pool, values, j_rows, aj_rows = stream_fixture()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            root = Path(tmp)
            source, j_path, aj_path = self.make_sources(root, j_rows, aj_rows)
            before = source_snapshot(source)
            output = root / "new_check"
            output.mkdir()
            (output / "unrelated.txt").write_text("retain this file", encoding="utf-8")
            summary = subject.verify_rankings(pool, values, j_path, aj_path, output,
                                              device="cpu", top_k=3)
            self.assertIsInstance(summary, dict)
            with (output / "per_query_checks.csv").open(newline="", encoding="utf-8") as handle:
                checks = list(csv.DictReader(handle))
            self.assertEqual(sorted(int(row["query_id"]) for row in checks), list(pool))
            self.assertEqual(len(checks), len(pool))
            self.assertEqual((output / "unrelated.txt").read_text(), "retain this file")
            self.assertEqual(source_snapshot(source), before)

    def test_missing_unexpected_or_repeated_query_blocks_fail_with_evidence(self):
        pool, values, good_j, good_aj = stream_fixture()
        first_query = str(next(iter(pool)))
        first_block = [row for row in good_j if row["query_id"] == first_query]
        for corruption in ("missing", "unexpected", "repeated"):
            for target in ("J", "AJ"):
                with self.subTest(corruption=corruption, target=target):
                    j_rows, aj_rows = deepcopy(good_j), deepcopy(good_aj)
                    rows = j_rows if target == "J" else aj_rows
                    if corruption == "missing":
                        rows[:] = [row for row in rows if row["query_id"] == first_query]
                    elif corruption == "unexpected":
                        extra = deepcopy(first_block)
                        for row in extra:
                            row["query_id"] = "92"
                        rows.extend(extra)
                    else:
                        rows.extend(deepcopy([row for row in rows if row["query_id"] == first_query]))
                    with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
                        root = Path(tmp)
                        source, j_path, aj_path = self.make_sources(root, j_rows, aj_rows)
                        before = source_snapshot(source)
                        output = root / "new_failed_check"
                        with self.assertRaises(ValueError):
                            subject.verify_rankings(pool, values, j_path, aj_path, output,
                                                    device="cpu", top_k=3)
                        self.assertEqual(source_snapshot(source), before)
                        self.assert_failure_artifact(output)

    def test_lambda_difference_records_all_queries_then_strictly_fails(self):
        pool, values, j_rows, aj_rows = stream_fixture()
        real_compute = subject.compute_scores
        calls = []

        def with_first_lambda_mismatch(joint, angular):
            scores = {key: value.copy() for key, value in real_compute(joint, angular).items()}
            query = list(pool)[len(calls)]
            calls.append(query)
            if len(calls) == 1:
                first = independent_order(query, pool[query], scores["J_lambda1"])[0]
                scores["J_lambda2"][first] = np.nextafter(scores["J_lambda2"][first], 0.)
            return scores

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            root = Path(tmp)
            source, j_path, aj_path = self.make_sources(root, j_rows, aj_rows)
            before = source_snapshot(source)
            output = root / "new_lambda_check"
            with patch.object(subject, "compute_scores", side_effect=with_first_lambda_mismatch):
                with self.assertRaises(subject.VerificationMismatch):
                    subject.verify_rankings(pool, values, j_path, aj_path, output,
                                            device="cpu", top_k=3)
            self.assertEqual(calls, list(pool), "Lambda mismatch must not skip other source checks")
            with (output / "per_query_checks.csv").open(newline="", encoding="utf-8") as handle:
                checked = list(csv.DictReader(handle))
            self.assertEqual(sorted(int(row["query_id"]) for row in checked), list(pool))
            self.assertEqual(source_snapshot(source), before)
            self.assert_failure_artifact(output)

    def test_existing_check_output_is_not_overwritten(self):
        pool, values, j_rows, aj_rows = stream_fixture()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            root = Path(tmp)
            source, j_path, aj_path = self.make_sources(root, j_rows, aj_rows)
            before = source_snapshot(source)
            output = root / "previous_check"
            output.mkdir()
            checks = output / "per_query_checks.csv"
            checks.write_text("previous evidence must remain byte identical\n", encoding="utf-8")
            saved = (checks.read_bytes(), checks.stat().st_mtime_ns)
            with self.assertRaises((ValueError, FileExistsError)):
                subject.verify_rankings(pool, values, j_path, aj_path, output,
                                        device="cpu", top_k=3)
            self.assertEqual((checks.read_bytes(), checks.stat().st_mtime_ns), saved)
            self.assertEqual(source_snapshot(source), before)

    def test_full_gallery_mismatch_is_saved_without_touching_sources(self):
        pool, values, j_rows, aj_rows = stream_fixture()
        # All J scores tie: replacing rank 3 by the omitted rank 4 is still
        # internally legal, but violates the full-gallery SHA tie ordering.
        query = next(iter(pool))
        candidates = pool[query]
        order = independent_order(query, candidates, np.ones(len(candidates)))
        j_rows[2]["candidate_id"] = str(int(candidates[order[3]]))
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            root = Path(tmp)
            source, j_path, aj_path = self.make_sources(root, j_rows, aj_rows)
            before = source_snapshot(source)
            output = root / "new_failed_check"
            with self.assertRaises(subject.VerificationMismatch) as caught:
                subject.verify_rankings(pool, values, j_path, aj_path, output,
                                        device="cpu", top_k=3)
            self.assertTrue(caught.exception.evidence)
            self.assertEqual(source_snapshot(source), before)
            self.assert_failure_artifact(output)


if __name__ == "__main__":
    unittest.main()
