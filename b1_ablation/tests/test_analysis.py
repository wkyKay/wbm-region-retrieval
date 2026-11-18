import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from b1_ablation import analysis


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def fixture(self):
        ids = [0, 1, 2, 3]
        a, b = (1, 0, 0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0, 0)
        gallery = {i: (a if i in (0, 1, 4, 5, 6, 7) else b) for i in range(12)}
        frozen = {q: gallery[q] for q in ids}
        classes, groups = analysis.group_indices(ids, frozen)
        ks = (1, 10)
        fields = analysis.metric_fields(ks, "exact")
        rows = []
        for q in ids:
            rows.append({"query_id": q, "class_id": 99 if q < 2 else 7,
                         "label_signature": "0" if q < 2 else "01000000",
                         "label_count": 1, "pool_size": 11, "returned": 10,
                         "relevant_in_pool": 5, **{field: 0.5 for field in fields}})
        analysis.write_csv(self.root / "per_query.csv", list(rows[0]), rows)
        metrics = {analysis.report_name(field): 0.5 for field in fields}
        report = {"num_queries": 4, "ks": [1, 10], "metrics": metrics, "class_macro_metrics": metrics,
                  "groups": {"single_label": {"metrics": metrics},
                             "by_label_count": {"1": {"metrics": metrics}}}}
        (self.root / "metrics.json").write_text(json.dumps(report))
        class_rows = [{"class_id": -100 + i, "label_signature": analysis.signature(label),
                       "num_queries": 2, **metrics} for i, label in enumerate(classes)]
        analysis.write_csv(self.root / "per_class.csv", list(class_rows[0]), class_rows)
        return ids, frozen, gallery, ks, classes, groups, rows

    def test_signature_formats_and_class_id_numbering_do_not_change_alignment(self):
        self.assertEqual(analysis.parse_signature("0|2|6"), analysis.parse_signature("10100010"))
        for invalid in ("0|0", "8", "00000000"):
            with self.assertRaises(ValueError):
                analysis.parse_signature(invalid)
        ids, frozen, gallery, ks, classes, groups, _ = self.fixture()
        loaded = analysis._load_family(self.root, "exact", ids, frozen, gallery, ks, classes, groups)
        np.testing.assert_allclose(loaded["aggregated"]["all"], 0.5)
        self.assertEqual(loaded["rows"][0]["class_id"], "99")

    def test_duplicate_query_wrong_labels_and_pool_size_are_rejected(self):
        ids, frozen, gallery, ks, classes, groups, rows = self.fixture()
        for corruption in ("duplicate", "labels", "pool"):
            corrupted = [dict(row) for row in rows]
            if corruption == "duplicate":
                corrupted.append(dict(rows[0]))
            elif corruption == "labels":
                corrupted[0]["label_signature"] = "1"
            else:
                corrupted[0]["pool_size"] = 10
            analysis.write_csv(self.root / "per_query.csv", list(rows[0]), corrupted)
            with self.assertRaises(ValueError):
                analysis._load_family(self.root, "exact", ids, frozen, gallery, ks, classes, groups)

    def test_aggregation_uses_class_equal_macro_and_query_weighted_overall(self):
        # The class means are 0 and 1, but the classes have sizes 1 and 3.
        a, b = (1, 0, 0, 0, 0, 0, 0, 0), (1, 1, 0, 0, 0, 0, 0, 0)
        frozen = {10: a, 20: b, 30: b, 40: b}
        classes, groups = analysis.group_indices([10, 20, 30, 40], frozen)
        values = np.array([[0., np.nan], [1., 0.], [1., 0.5], [1., 1.]])
        result = analysis.aggregate(values, classes, groups)
        np.testing.assert_allclose(result["all"], [0.75, 0.5])
        np.testing.assert_allclose(result["macro_class"], [0.5, 0.5])
        np.testing.assert_allclose(result["single_label"][0], 0.)
        np.testing.assert_allclose(result["mixed_label"], [1., 0.5])

    def test_bootstrap_reproducibility_shared_pairing_and_constant_effect(self):
        strata = {(0, 1): np.array([0, 2]), (1, 0): np.array([1, 3, 4])}
        delta = np.array([0., 0.1, 0.5, -0.2, 0.4])
        matrix = np.column_stack((delta, -delta, np.full(5, 0.25)))
        first = analysis.paired_bootstrap(matrix, strata, 751, 2026)
        second = analysis.paired_bootstrap(matrix, strata, 751, 2026)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertAlmostEqual(first["lower"][1], -first["upper"][0])
        self.assertAlmostEqual(first["upper"][1], -first["lower"][0])
        for key in ("difference", "lower", "upper"):
            self.assertAlmostEqual(first[key][2], 0.25)
        with self.assertRaises(ValueError):
            analysis.paired_bootstrap(matrix, {(1,): [0, 1, 2, 3]}, 10)

    def test_five_error_types_partition_top10_and_cases_keep_original_scores(self):
        query = (1, 1, 0, 0, 0, 0, 0, 0)
        patterns = [query, (1, 1, 1, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0, 0, 0),
                    (1, 0, 1, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0, 0, 0)]
        self.assertEqual([analysis.error_category(query, p) for p in patterns], list(analysis.ERRORS))
        labels = {0: query, **{i + 1: patterns[i % 5] for i in range(10)}}
        path = self.root / "rankings.csv"
        rows = [{"query_id": 0, "rank": i, "candidate_id": i, "score": 1. - i * 0.01,
                 "global_score": 0.7} for i in range(1, 11)]
        analysis.write_csv(path, list(rows[0]), rows)
        sink = io.StringIO()
        fields = ["method", "query_id", "k", "label_signature", "label_count"]
        fields += [k + "_count" for k in analysis.ERRORS] + [k + "_fraction" for k in analysis.ERRORS]
        writer = csv.DictWriter(sink, fieldnames=fields)
        values, cases, digest = analysis.stream_ranking_errors(
            path, "mock", [0], {0: query}, labels,
            {0: {"Precision@1": 1., "Precision@10": .2}},
            {0: {"FullCoveragePrecision@1": 1., "FullCoveragePrecision@10": .4}},
            {0}, writer, (1, 10))
        np.testing.assert_allclose(values[0, 1], [.2] * 5)
        np.testing.assert_allclose(values[0, 0], [1., 0., 0., 0., 0.])
        self.assertEqual(cases["0"][0]["global_score"], "0.7")
        self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
        rows[-1]["candidate_id"] = 1
        analysis.write_csv(path, list(rows[0]), rows)
        with self.assertRaises(ValueError):
            analysis.stream_ranking_errors(path, "mock", [0], {0: query}, labels,
                {0: {"Precision@1": 1., "Precision@10": .2}},
                {0: {"FullCoveragePrecision@1": 1., "FullCoveragePrecision@10": .4}},
                set(), writer, (1, 10))

    def test_case_selection_includes_both_directions_and_ties_by_original_id(self):
        result = analysis.select_cases([5, 10, 20, 30], {(1, 0): [0, 1, 2, 3]},
                                       np.array([[.2], [.2], [-.3], [-.1]]), ["baseline"])
        self.assertEqual([(r["direction"], r["query_id"]) for r in result],
                         [("improved", 5), ("degraded", 20)])

    def test_postprocess_end_to_end_and_resume_rejects_mutated_source(self):
        source = self.root / "source"
        source.mkdir()
        query = (1, 1, 0, 0, 0, 0, 0, 0)
        patterns = [query, (1, 1, 1, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0, 0, 0),
                    (1, 0, 1, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0, 0, 0)]
        labels = {0: query, **{i + 1: patterns[i % 5] for i in range(10)}}
        rankings = [{"query_id": 0, "rank": i, "candidate_id": i, "score": 1 - .01 * i}
                    for i in range(1, 11)]
        analysis.write_csv(source / "rankings.csv", list(rankings[0]), rankings)
        exact = np.asarray([1, 0, 0, 0, 0] * 2)
        full = np.asarray([1, 1, 0, 0, 0] * 2)
        coverage = np.asarray([1, 1, .5, .5, 0] * 2)
        family_metrics = {"exact": {}, "coverage": {}}
        for k in (1, 10):
            for family, hits, denominator in (("exact", exact, 2), ("coverage", full, 4)):
                prefix = "" if family == "exact" else "FullCoverage"
                metrics = family_metrics[family]
                metrics[prefix + "Hit@" + str(k)] = float(hits[:k].sum() > 0)
                metrics[prefix + "Precision@" + str(k)] = float(hits[:k].sum() / k)
                metrics[prefix + "Recall@" + str(k)] = float(hits[:k].sum() / denominator)
                metrics[prefix + "AP@" + str(k)] = sum(
                    hits[r] * sum(hits[:r + 1]) / (r + 1) for r in range(k)) / denominator
            dcg = sum(coverage[r] / np.log2(r + 2) for r in range(k))
            ideal = sorted(coverage, reverse=True)
            idcg = sum(ideal[r] / np.log2(r + 2) for r in range(k))
            family_metrics["coverage"]["Coverage-nDCG@" + str(k)] = dcg / idcg
            family_metrics["coverage"]["MeanCoverage@" + str(k)] = sum(coverage[:k]) / k
        for family, metrics in family_metrics.items():
            prefix = "" if family == "exact" else "coverage_"
            row = {"query_id": 0, "class_id": 123 if family == "exact" else 456,
                   "label_signature": "0|1" if family == "exact" else "11000000",
                   "label_count": 2, "pool_size": 10, "returned": 10,
                   ("relevant_in_pool" if family == "exact" else "full_coverage_in_pool"):
                       2 if family == "exact" else 4, **metrics}
            analysis.write_csv(source / (prefix + "per_query.csv"), list(row), [row])
            summary = {analysis.report_name(key): float(value) for key, value in metrics.items()}
            report = {"num_queries": 1, "ks": [1, 10], "metrics": summary, "class_macro_metrics": summary,
                      "groups": {"mixed_label": {"metrics": summary},
                                 "by_label_count": {"2": {"metrics": summary}}}}
            analysis.write_json(source / (prefix + "metrics.json"), report)
            class_row = {"class_id": 999, "label_signature": "11000000", "num_queries": 1, **summary}
            analysis.write_csv(source / (prefix + "per_class.csv"), list(class_row), [class_row])
        output = self.root / "postprocessed"
        sources = {"J_2": source, "B0": source}
        complete = analysis.postprocess(output, sources, labels, [0], list(labels),
                                        ks=(1, 10), bootstrap_replicates=20)
        self.assertEqual(complete["status"], "complete")
        self.assertEqual(complete["num_classes"], 1)
        bootstrap = json.loads((output / "paired_bootstrap.json").read_text())
        self.assertTrue(all(r["difference_pp"] == 0 for r in bootstrap["results"]))
        before = (output / "completed.json").stat().st_mtime_ns
        self.assertEqual(analysis.postprocess(output, sources, labels, [0], list(labels),
                                             ks=(1, 10), bootstrap_replicates=20), complete)
        self.assertEqual((output / "completed.json").stat().st_mtime_ns, before)
        with (source / "rankings.csv").open("a") as handle:
            handle.write("\\n")
        with self.assertRaisesRegex(ValueError, "source changed"):
            analysis.postprocess(output, sources, labels, [0], list(labels),
                                 ks=(1, 10), bootstrap_replicates=20)

    def test_completed_analysis_reuses_only_identical_sources_and_artifacts(self):
        source = self.root / "ranking.csv"
        source.write_text("frozen ranking")
        output = self.root / "analysis"
        output.mkdir()
        plan = {"seed": 2026}
        normalized = {"J_2": (self.root, self.root, source)}
        reports = {"J_2": {}}
        analysis.write_json(output / "analysis_plan.json", plan)
        analysis.write_json(output / "sources.json",
                            {"J_2": {str(source.resolve()): analysis.file_hash(source)}})
        marker = {"status": "complete", "files": {
            name: analysis.file_hash(output / name) for name in ("analysis_plan.json", "sources.json")}}
        analysis.write_json(output / "completed.json", marker)
        before = (output / "completed.json").stat().st_mtime_ns
        self.assertEqual(analysis.validate_completed(output, plan, normalized, reports), marker)
        self.assertEqual((output / "completed.json").stat().st_mtime_ns, before)
        source.write_text("changed ranking")
        with self.assertRaisesRegex(ValueError, "source changed"):
            analysis.validate_completed(output, plan, normalized, reports)
        source.write_text("frozen ranking")
        (output / "sources.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            analysis.validate_completed(output, plan, normalized, reports)

    def test_missing_sources_fail_before_creating_output(self):
        output = self.root / "new"
        labels = {i: (1, 0, 0, 0, 0, 0, 0, 0) for i in range(11)}
        with self.assertRaises(FileNotFoundError):
            analysis.postprocess(output, {"J_2": self.root / "missing", "B0": self.root / "absent"},
                                 labels, [0], list(labels), ks=(1, 10), bootstrap_replicates=10)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
