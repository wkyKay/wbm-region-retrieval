"""Catalog integrity checks; no model imports, training, or writes to source runs."""
import copy
from contextlib import redirect_stdout
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


class SerializationTests(unittest.TestCase):
    def test_compact_unicode_float_nested_roundtrip_and_default_format(self):
        value = {"说明": "晶圆／融合 🌐", "metrics": {"Precision@10": 0.5501999999999999,
                 "mAP@100": 0.20514388384343732, "tiny": 1e-12, "negative_zero": -0.0},
                 "nested": [{"original": [None, True, False, 2751, {"文本": "第一行\n第二行"}]}]}
        with tempfile.TemporaryDirectory() as tmp:
            pretty, compact = Path(tmp) / "default.json", Path(tmp) / "compact.json"
            build.dump(pretty, value)
            build.dump(compact, value, compact=True)
            self.assertEqual(pretty.read_text(), json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            self.assertEqual(compact.read_text(), json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
            self.assertEqual(build.read(pretty), value)
            self.assertEqual(build.read(compact), value)
            self.assertIn("晶圆", compact.read_text())
            self.assertEqual(compact.read_text().count("\n"), 1)
            self.assertLess(compact.stat().st_size, pretty.stat().st_size)

    def test_both_formats_reject_nonfinite_values_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            for compact in (False, True):
                for number in (float("nan"), float("inf"), float("-inf")):
                    path = Path(tmp) / "rejected.json"
                    with self.subTest(compact=compact, number=number), self.assertRaises(ValueError):
                        build.dump(path, {"metric": number}, compact=compact)
                    self.assertFalse(path.exists())

    def test_collect_cli_writes_compact_snapshot_without_changing_values_or_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp).resolve() / "catalog"
            workspace = Path(tmp).resolve() / "workspace"
            sources = {"runs": [{"key": "synthetic"}]}
            previous = {"runs": [], "schema_version": 1}
            snapshot = {"schema_version": 1, "runs": [{"key": "synthetic", "value": .5501999999999999}],
                        "metadata": {"说明": "仅排版压缩", "nested": [None, {"value": 1e-12}]}}
            generated = {"registry.csv": "key\nsynthetic\n", "tables/example.csv": "value\n0.5501999999999999\n"}
            build.dump(folder / "sources.json", sources)
            build.dump(folder / "snapshot.json", previous)
            source_bytes = (folder / "sources.json").read_bytes()
            with patch("sys.argv", ["build.py", "collect", "--workspace", str(workspace), "--catalog", str(folder)]), \
                    patch.object(build, "collect", return_value=snapshot) as collect, \
                    patch.object(build, "render", return_value=generated) as render, \
                    patch.object(build, "dump", wraps=build.dump) as dump, redirect_stdout(io.StringIO()):
                build.main()
            collect.assert_called_once_with(workspace, sources, previous)
            render.assert_called_once_with(snapshot)
            dump.assert_called_once_with(folder / "snapshot.json", snapshot, compact=True)
            self.assertEqual(build.read(folder / "snapshot.json"), snapshot)
            self.assertEqual((folder / "snapshot.json").read_text().count("\n"), 1)
            self.assertEqual((folder / "sources.json").read_bytes(), source_bytes)
            for name, text in generated.items():
                self.assertEqual((folder / name).read_text(), text)


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
        files = {"metrics.json": build.sha(self.report)}
        identity = self.folder / "run_identity.json"
        if identity.exists():
            files[identity.name] = build.sha(identity)
        build.dump(self.marker, {"files": files})

    def identity(self, **updates):
        path = self.folder / "run_identity.json"
        value = {"schema_version": 1, "key": "fixture_valid", "split": "valid",
                 "started_at": "2026-11-17T09:30:00+08:00",
                 "run_id": "20261117T093000+0800__fixture_valid"}
        value.update(updates)
        build.dump(path, value)
        self.specs["runs"][0].update(identity_schema_version=1, identity=str(path.relative_to(self.root)))
        self.seal()
        return path

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

    def test_cross_valid_uses_its_frozen_candidate_identity_and_range(self):
        self.report.unlink()
        self.report = self.folder / "coverage_metrics.json"
        derived = {"protocol_version": "coverage_cross_combination_v1",
                   "base_protocol_signature": build.BASE_PROTOCOL_SIGNATURE,
                   "candidate_identity_sha256": build.CROSS_SPLITS["valid"]["candidate_identity_sha256"]}
        report = {"split": "valid", "num_queries": 500, "metric_version": "coverage_v1",
                  "ks": [1, 5, 10, 20, 50, 100, 500], "metrics": {"Coverage-nDCG@10": .5},
                  "metadata": {"derived_protocol": derived}}
        build.dump(self.report, report)
        build.dump(self.marker, {"files": {self.report.name: build.sha(self.report)}})
        self.specs["runs"][0].update(required_families=["coverage"], protocol="coverage_cross_combination_v1")
        first = self.collect()
        row = first["runs"][0]
        self.assertEqual((row["query_count"], row["gallery_count"], row["candidates_per_query"]), (500, 3599, "3499-3585"))
        derived["candidate_identity_sha256"] = build.CROSS_SPLITS["test"]["candidate_identity_sha256"]
        build.dump(self.report, report)
        build.dump(self.marker, {"files": {self.report.name: build.sha(self.report)}})
        with self.assertRaisesRegex(ValueError, "candidate identity changed"):
            self.collect()

    def test_cross_test_keeps_original_candidate_identity_and_range(self):
        self.report.unlink()
        self.report = self.folder / "coverage_metrics.json"
        report = {"split": "test", "num_queries": 7199, "metric_version": "coverage_v1",
                  "ks": [1, 5, 10, 20, 50, 100, 500], "metrics": {"Coverage-nDCG@10": .5},
                  "metadata": {"derived_protocol": {"protocol_version": "coverage_cross_combination_v1",
                    "base_protocol_signature": build.BASE_PROTOCOL_SIGNATURE,
                    "candidate_identity_sha256": build.CROSS_SPLITS["test"]["candidate_identity_sha256"]}}}
        build.dump(self.report, report)
        build.dump(self.marker, {"files": {self.report.name: build.sha(self.report)}})
        self.specs["runs"][0].update(split="test", required_families=["coverage"], protocol="coverage_cross_combination_v1")
        first = self.collect()
        row = first["runs"][0]
        self.assertEqual((row["query_count"], row["gallery_count"], row["candidates_per_query"]), (7199, 7199, "6999-7170"))
        self.assertEqual(self.collect(first)["runs"], first["runs"])

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

    def test_persisted_start_identity_is_bound_and_stable(self):
        identity = self.identity()
        first = self.collect()
        run = first["runs"][0]
        self.assertEqual(run["run_id"], "20261117T093000+0800__fixture_valid")
        self.assertEqual(run["id_time"], {"value": "2026-11-17T09:30:00+08:00", "basis": "started_at",
                                         "source": "artifacts/example/valid/run_identity.json",
                                         "is_execution_start": True})
        self.assertEqual(run["provenance"]["identity"]["sha256"], build.sha(identity))
        before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in self.workspace.rglob("*") if p.is_file()}
        self.collect(first)
        after = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in self.workspace.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        for path in (identity, self.marker):
            stamp = path.stat().st_mtime + 86400
            os.utime(path, (stamp, stamp))
        second = self.collect(first)
        fresh = self.collect()
        self.assertEqual(first["runs"][0], second["runs"][0])
        self.assertEqual(first["runs"][0], fresh["runs"][0])

    def test_declared_identity_cannot_fall_back_to_mtime(self):
        self.identity().unlink()
        with self.assertRaisesRegex(ValueError, "Missing declared run identity"):
            self.collect()
        self.specs["runs"][0].pop("identity")
        with self.assertRaisesRegex(ValueError, "requires identity_schema_version"):
            self.collect()

    def test_identity_requires_schema_timezone_matching_key_and_split(self):
        cases = [({"schema_version": 2}, "schema/key/split"),
                 ({"schema_version": True}, "schema/key/split"),
                 ({"key": "other"}, "schema/key/split"),
                 ({"split": "test"}, "schema/key/split"),
                 ({"started_at": "2026-11-17T09:30:00"}, "seconds and \\+08:00"),
                 ({"started_at": "2026-11-17T01:30:00+00:00"}, "seconds and \\+08:00"),
                 ({"started_at": "2026-02-30T09:30:00+08:00"}, "Invalid started_at"),
                 ({"run_id": "20261117T093001+0800__fixture_valid"}, "does not match"),
                 ({"run_id": "20261117T093000+0800__other"}, "does not match")]
        for updates, message in cases:
            with self.subTest(updates=updates):
                self.identity(**updates)
                with self.assertRaisesRegex(ValueError, message):
                    self.collect()

    def test_identity_must_be_completed_and_immutable(self):
        path = self.identity()
        first = self.collect()
        value = build.read(path)
        value["started_at"] = "2026-11-17T09:31:00+08:00"
        value["run_id"] = "20261117T093100+0800__fixture_valid"
        build.dump(path, value)
        with self.assertRaisesRegex(ValueError, "identity hash mismatch"):
            self.collect(first)
        self.seal()
        with self.assertRaisesRegex(ValueError, "Completed run changed"):
            self.collect(first)

    def test_registered_identity_cannot_be_removed_or_relabelled(self):
        self.identity()
        first = self.collect()
        self.specs["runs"][0].pop("identity")
        self.specs["runs"][0].pop("identity_schema_version")
        with self.assertRaisesRegex(ValueError, "Registered run identity changed"):
            self.collect(first)

    def test_manifest_with_new_identity_requires_explicit_declaration(self):
        manifest = self.folder / "manifest.json"
        build.dump(manifest, {"run_id": "20261117T093000+0800__fixture_valid",
                              "started_at": "2026-11-17T09:30:00+08:00"})
        self.specs["runs"][0]["manifest"] = str(manifest.relative_to(self.root))
        with self.assertRaisesRegex(ValueError, "explicit identity source"):
            self.collect()

    def test_identity_path_cannot_escape_the_run(self):
        self.identity()
        for name in ("../run_identity.json", str(self.folder / "run_identity.json"), "run_identity.json"):
            with self.subTest(name=name):
                self.specs["runs"][0]["identity"] = name
                with self.assertRaisesRegex(ValueError, "repository-relative|inside the completed run"):
                    self.collect()

    def test_artificial_outputs_are_inventoried_without_changing_artifacts(self):
        output = self.root / "artificial/experiment_20261117T093000+0800"
        output.mkdir(parents=True)
        build.dump(output / "status.json", {"status": "running"})
        result = self.collect()
        entries = {item["path"]: item for item in result["inventory"] if item["repository"] == "compared_methods"}
        self.assertEqual(entries["artifacts/example"]["registered_results"], ["fixture_valid"])
        self.assertEqual(entries[str(output.relative_to(self.root))]["recorded_status"], "running")
        self.assertEqual(entries[str(output.relative_to(self.root))]["role"], "not_in_current_tables")


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
        source_runs = {r["key"]: r for r in self.snapshot["runs"]}
        expected_posthoc = {
            r["key"] for r in self.snapshot["runs"]
            if r["category"] in ("fusion", "fusion_ablation", "posthoc") and r.get("posthoc")
        }
        self.assertIn("f_combination_test", expected_posthoc)
        self.assertEqual({r["key"] for r in fusion if r["posthoc"] == "True"}, expected_posthoc)
        for row in fusion:
            self.assertEqual(row["posthoc"], str(bool(source_runs[row["key"]].get("posthoc"))))

    def test_repeated_render_is_byte_identical_and_preserves_null(self):
        self.assertEqual(self.output, build.render(copy.deepcopy(self.snapshot)))
        groups = dict(build.metric_groups({"metrics": {"mAP@1": None}, "groups": {}}))
        self.assertIsNone(groups["all"]["mAP@1"])

    def test_compact_snapshot_roundtrip_preserves_all_records_and_rendered_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snapshot.json"
            build.dump(path, self.snapshot, compact=True)
            restored = build.read(path)
            self.assertEqual(restored, self.snapshot)
            self.assertEqual(restored["runs"], self.snapshot["runs"])
            self.assertEqual(build.render(restored), self.output)

    def test_existing_generated_files_are_unchanged(self):
        for name, content in self.output.items():
            with self.subTest(name=name):
                self.assertEqual((CATALOG / name).read_text(), content)

    def test_cross_valid_remains_in_csv_but_not_test_overview(self):
        snapshot = copy.deepcopy(self.snapshot)
        item = copy.deepcopy(next(r for r in snapshot["runs"] if r["category"] == "cross_combination"))
        item.update(key="synthetic_cross_valid", run_id="20261121T010000+0800__synthetic_cross_valid",
                    split="valid", query_count=500, gallery_count=3599, candidates_per_query="3499-3585")
        snapshot["runs"].append(item)
        rendered = build.render(snapshot)
        self.assertNotIn("`synthetic_cross_valid`", rendered["CURRENT_RESULTS.md"])
        cross = list(csv.DictReader(io.StringIO(rendered["tables/cross_combination.csv"])))
        new_rows = [r for r in cross if r["key"] == "synthetic_cross_valid"]
        self.assertTrue(new_rows)
        self.assertEqual({r["split"] for r in new_rows}, {"valid"})

    def test_new_fusion_test_is_appended_without_valid_or_cross_contamination(self):
        snapshot = copy.deepcopy(self.snapshot)
        template = next(r for r in snapshot["runs"] if r["key"] == "f_combination_test")
        for key, split, protocol in [("new_fusion_test", "test", "retrieval_v1"),
                                     ("new_fusion_valid", "valid", "retrieval_v1"),
                                     ("new_fusion_cross", "test", "coverage_cross_combination_v1")]:
            item = copy.deepcopy(template)
            item.update(key=key, run_id="20261117T093000+0800__" + key, split=split,
                        protocol=protocol, category="fusion_ablation", display=key)
            snapshot["runs"].append(item)
        text = build.render(snapshot)["CURRENT_RESULTS.md"]
        section = text.split("## 融合及消融\n", 1)[1].split("## 跨组合 test", 1)[0]
        self.assertIn("`new_fusion_test`", section)
        self.assertNotIn("`new_fusion_valid`", section)
        self.assertNotIn("`new_fusion_cross`", section)
        self.assertLess(section.index("`f_combination_test`"), section.index("`new_fusion_test`"))
        self.assertNotIn("最后一行组合", section)


if __name__ == "__main__":
    unittest.main()
