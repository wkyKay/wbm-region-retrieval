"""Catalog integrity checks; no model imports, training, or writes to source runs."""
import copy
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

CATALOG = Path(__file__).resolve().parents[1] / "results_catalog"
spec = importlib.util.spec_from_file_location("result_catalog_build", CATALOG / "build.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workspace = Path(self.tmp.name)
        for repo in ("compared_methods", "wbm-region-retrieval"):
            root = self.workspace / repo
            (root / "artifacts").mkdir(parents=True)
            for name in build.PROTOCOL_FILES:
                p = root / "protocol" / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("frozen fixture\n")
        self.root = self.workspace / "compared_methods"
        self.folder = self.root / "artifacts/example/valid"
        self.folder.mkdir(parents=True)
        self.report = self.folder / "metrics.json"
        self.marker = self.folder / "completed.json"
        build.dump(self.report, {"split": "valid", "num_queries": 500, "metric_version": "exact_v1",
                                 "retrieval_protocol_signature": build.BASE_PROTOCOL_SIGNATURE,
                                 "ks": [1, 5, 10, 20, 50, 100, 500], "metrics": {"Hit@1": .5}})
        self.seal()
        self.specs = {"runs": [{"key": "fixture_valid", "repository": "compared_methods",
                                "path": "artifacts/example/valid", "split": "valid",
                                "required_families": ["exact"]}]}

    def seal(self):
        build.dump(self.marker, {"files": {"metrics.json": build.sha(self.report)}})

    def collect(self, previous=None):
        with patch.object(build.subprocess, "check_output", return_value="fixture-head\n"):
            return build.collect(self.workspace, self.specs, previous)

    def test_id_survives_timestamp_change_without_source_write(self):
        before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in self.workspace.rglob("*") if p.is_file()}
        first = self.collect()
        after = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in self.workspace.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        stamp = self.marker.stat().st_mtime + 86400
        os.utime(self.marker, (stamp, stamp))
        second = self.collect(first)
        self.assertEqual(first["runs"][0]["run_id"], second["runs"][0]["run_id"])
        self.assertFalse(second["runs"][0]["id_time"]["is_execution_start"])

    def test_corrupt_or_missing_required_report_rejected(self):
        self.report.write_text("{}")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.collect()
        self.report.unlink()
        with self.assertRaisesRegex(ValueError, "report families"):
            self.collect()

    def test_protocol_and_split_mismatch_rejected(self):
        other = self.workspace / "wbm-region-retrieval/protocol" / build.PROTOCOL_FILES[0]
        other.write_text("different")
        with self.assertRaisesRegex(ValueError, "protocol files differ"):
            self.collect()
        other.write_text("frozen fixture\n")
        d = build.read(self.report)
        d["split"] = "test"
        build.dump(self.report, d)
        self.seal()
        with self.assertRaisesRegex(ValueError, "Query/split mismatch"):
            self.collect()

    def test_cross_combination_cannot_be_relabelled_normal(self):
        d = build.read(self.report)
        d["metadata"] = {"derived_protocol": {"protocol_version": "coverage_cross_combination_v1"}}
        build.dump(self.report, d)
        self.seal()
        with self.assertRaisesRegex(ValueError, "Candidate protocol mismatch"):
            self.collect()

    def test_old_protocol_signature_rejected_even_with_same_query_count(self):
        d = build.read(self.report)
        d["retrieval_protocol_signature"] = "different-protocol"
        build.dump(self.report, d)
        self.seal()
        with self.assertRaisesRegex(ValueError, "protocol signature mismatch"):
            self.collect()

    def test_changed_completed_run_cannot_reuse_key(self):
        first = self.collect()
        d = build.read(self.report)
        d["metrics"]["Hit@1"] = .6
        build.dump(self.report, d)
        self.seal()
        with self.assertRaisesRegex(ValueError, "Completed run changed"):
            self.collect(first)


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = build.read(CATALOG / "snapshot.json")
        cls.output = build.render(cls.snapshot)

    def test_partition_and_values_match_source_snapshot(self):
        lookup = {r["run_id"]: r for r in self.snapshot["runs"]}
        seen = set()
        for row in csv.DictReader(io.StringIO(self.output["metrics.csv"])):
            identity = tuple(row[k] for k in ["run_id", "metric_version", "group", "metric", "K"])
            self.assertNotIn(identity, seen)
            seen.add(identity)
            run = lookup[row["run_id"]]
            data = run["reports"][row["metric_version"].removesuffix("_v1")]["data"]
            if row["group"] == "all":
                source = data["metrics"]
            elif row["group"] == "class_macro":
                source = data["class_macro_metrics"]
            else:
                source = data["groups"]
                for part in row["group"].split("/"):
                    source = source[part]
                source = source["metrics"]
            value = source[row["metric"] + "@" + row["K"]]
            self.assertEqual(None if not row["value"] else float(row["value"]), value)
        expected_count = sum(len(d["data"]["metrics"]) + len(d["data"]["class_macro_metrics"]) +
                             sum(len(v["metrics"]) for k, v in d["data"]["groups"].items() if k != "by_label_count") +
                             sum(len(v["metrics"]) for v in d["data"]["groups"]["by_label_count"].values())
                             for r in self.snapshot["runs"] for d in r["reports"].values())
        self.assertEqual(len(seen), expected_count)

    def test_paper_tables_separate_protocols_and_posthoc(self):
        main = list(csv.DictReader(io.StringIO(self.output["tables/methods.csv"])))
        self.assertEqual({r["protocol"] for r in main}, {"retrieval_v1"})
        self.assertEqual({r["posthoc"] for r in main}, {"False"})
        self.assertEqual(len({r["run_id"] for r in main}), 7)
        cross = list(csv.DictReader(io.StringIO(self.output["tables/cross_combination.csv"])))
        self.assertEqual({r["protocol"] for r in cross}, {"coverage_cross_combination_v1"})
        self.assertEqual({r["metric_version"] for r in cross}, {"coverage_v1"})
        fusion = list(csv.DictReader(io.StringIO(self.output["tables/fusion.csv"])))
        self.assertEqual({r["key"] for r in fusion if r["posthoc"] == "True"}, {"f_combination_test"})

    def test_repeated_render_is_byte_identical_and_preserves_null(self):
        self.assertEqual(self.output, build.render(copy.deepcopy(self.snapshot)))
        groups = dict(build.metric_groups({"metrics": {"mAP@1": None}, "groups": {}}))
        self.assertIsNone(groups["all"]["mAP@1"])


if __name__ == "__main__":
    unittest.main()
