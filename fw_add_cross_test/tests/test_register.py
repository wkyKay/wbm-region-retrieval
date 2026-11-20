"""Coverage-only cross-combination registration and immutable catalogue transactions."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fgw_grid_search import scoring
from fw_add_cross_test import register
from wbm_retrieval.common import file_hash, read_json, write_json


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name).resolve()
        self.root = self.workspace / "wbm-region-retrieval"
        self.output = self.root / "artificial/fw_add_cross_test_20261120T120000+0800"
        self.settings = deepcopy(scoring.SETTINGS)
        self.names = register.VARIANTS.copy()
        specs = deepcopy(register.SPECS)
        self.frozen = {"variants": self.names.copy(), "specs": deepcopy(specs),
                       "selection_rule": "user_specified_posthoc_components", "primary_variant": None,
                       "settings": deepcopy(self.settings), "test_parameter_search": False}
        self.manifest = {"stage": "test", "context": {"split": "test", "settings": deepcopy(self.settings),
            "variants": self.names.copy(), "specs": specs, "source_run": "artificial/frozen_source",
            "selection_rule": "user_specified_posthoc_components", "primary_variant": None,
            "evaluation_paths": register.evaluation_paths(self.output, self.names)}}
        self.derived = {"protocol_version": register.CROSS_PROTOCOL, "candidate_rule": register.CANDIDATE_RULE,
            "candidate_identity_sha256": register.CANDIDATE_IDENTITY,
            "base_protocol_signature": register.catalog.BASE_PROTOCOL_SIGNATURE,
            "candidate_count_min": 6999, "candidate_count_max": 7170, "filter_uses_ground_truth": True}
        self.manifest["context"]["derived_protocol"] = self.derived
        # The run-state implementation independently owns sealing/cache validation.
        # Here it is mocked so registration tests never touch formal sources or score data.
        mocked = patch.object(register, "verify_run", return_value=(self.manifest, self.frozen))
        mocked.start(); self.addCleanup(mocked.stop)
        write_json(self.output / "run_manifest.json", self.manifest)
        write_json(self.output / "frozen_variants.json", self.frozen)
        for name, spec in specs.items():
            folder = register.evaluation_folder(self.output, name)
            signature = "synthetic_" + name
            key = "fixture_20261120T120000_test_" + name
            identity = {"schema_version": 1, "key": key, "variant": name, "split": "test",
                        "evaluation_signature": signature, "started_at": "2026-11-20T12:00:00+08:00",
                        "run_id": "20261120T120000+0800__" + key}
            write_json(folder / "run_identity.json", identity)
            write_json(folder / "evaluation_manifest.json", {**identity, "signature": signature,
                "stage": "test", "spec": spec, "settings": self.settings,
                "protocol": register.CROSS_PROTOCOL, "derived_protocol": self.derived})
            report = {"split": "test", "num_queries": 7199, "ks": self.settings["ks"],
                      "metadata": {"signature": signature,
                          "derived_protocol": self.derived},
                      "queries_without_full_coverage_candidates": 999,
                      "metric_query_counts": {f"{metric}@{k}": count for k in self.settings["ks"]
                          for metric, count in (("Coverage-nDCG", 7000), ("MeanCoverage", 7199),
                          ("FullCoverageHit", 7199), ("FullCoveragePrecision", 7199),
                          ("FullCoverageRecall", 6200), ("FullCoverage_mAP", 6200))}}
            write_json(folder / "coverage_metrics.json", {**report, "metric_version": "coverage_v1"})
            write_json(folder / "completed.json", {"status": "completed", "signature": signature,
                "files": {p.name: file_hash(p) for p in folder.iterdir()}})
        self.catalog = self.root / "results_catalog"
        old_source = {"key": "historical_fixed", "repository": "wbm-region-retrieval", "path": "old/path"}
        self.sources = {"schema_version": 1, "runs": [old_source]}
        self.previous = {"source_spec_sha256": self.fingerprint(self.sources), "runs": [{**old_source,
            "provenance": {"description": "保留原始来源", "nested": [.25, None, {"original": True}]}}]}
        self.publish_catalog(self.sources, self.previous)

    @staticmethod
    def fingerprint(sources):
        return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def render(snapshot):
        keys = ",".join(row["key"] for row in snapshot["runs"])
        return {"registry.csv": "keys\n" + keys + "\n", "metrics.csv": "count\n" + str(len(snapshot["runs"])) + "\n"}

    def publish_catalog(self, sources, previous):
        write_json(self.catalog / "sources.json", sources)
        write_json(self.catalog / "snapshot.json", previous)
        for name, text in self.render(previous).items():
            (self.catalog / name).write_text(text)

    def snapshot(self):
        return {p.name: p.read_bytes() for p in self.catalog.iterdir() if p.is_file() and not p.name.startswith(".")}

    def collect(self, workspace, sources, previous):
        self.assertEqual(workspace, self.workspace)
        self.assertEqual(previous, self.previous)
        self.assertEqual(self.snapshot(), self.original)
        old_keys = {row["key"] for row in previous["runs"]}
        return {"source_spec_sha256": self.fingerprint(sources), "runs": deepcopy(previous["runs"]) +
                [deepcopy(row) for row in sources["runs"] if row["key"] not in old_keys]}

    def test_proposal_single_equal_addition_formula_no_new_primary_and_zero_write_resume(self):
        before = self.snapshot()
        path = register.write_proposal(self.output, root=self.root)
        rows = read_json(path)["runs"]
        self.assertEqual(len(rows), 1)
        self.assertEqual([row["variant"] for row in rows], ["F-WBBS-add-cross/" + name for name in self.names])
        self.assertEqual([row["valid_selected_primary"] for row in rows], [False])
        self.assertTrue(all("valid_rank" not in row for row in rows))
        self.assertTrue(all(row["selection_rule"] == "user_specified_posthoc_components" for row in rows))
        self.assertTrue(all(row["supervision"] == "self_supervised" and row["posthoc"] for row in rows))
        self.assertTrue(all(row["split"] == "test" and row["required_families"] == ["coverage"] for row in rows))
        self.assertEqual(rows[0]["path"], str(register.evaluation_folder(self.output, self.names[0]).relative_to(self.root)))
        self.assertEqual(rows[0]["selection"], str((self.output / "frozen_variants.json").relative_to(self.root)))
        stamp = path.stat().st_mtime_ns
        self.assertEqual(register.write_proposal(self.output, root=self.root), path)
        self.assertEqual(path.stat().st_mtime_ns, stamp)
        self.assertEqual(self.snapshot(), before)

    def test_wrong_order_missing_formula_primary_or_test_selection_rejected(self):
        original = deepcopy(self.frozen)
        mutations = [lambda d: d["variants"].append("FW"),
                     lambda d: d["variants"].pop(),
                     lambda d: d["specs"]["FW_ADD"].update(weights=[0.6, 0.0, 0.4]),
                     lambda d: d["specs"]["FW_ADD"].update(family="sum_minmax"),
                     lambda d: d.update(primary_variant=self.names[0]),
                     lambda d: d.update(test_parameter_search=True)]
        for change in mutations:
            with self.subTest(change=change):
                self.frozen.clear(); self.frozen.update(deepcopy(original)); change(self.frozen)
                with self.assertRaisesRegex(ValueError, r"exactly the requested 0.5F\+0.5WBBS"):
                    register.proposal(self.output, root=self.root)

    def test_non_test_pool_or_changed_evaluation_paths_rejected(self):
        self.manifest["context"]["evaluation_paths"][self.names[0]] = "../other"
        with self.assertRaisesRegex(ValueError, "evaluation paths"):
            register.proposal(self.output, root=self.root)
        self.manifest["context"]["evaluation_paths"] = register.evaluation_paths(self.output, self.names)
        path = register.evaluation_folder(self.output, self.names[0]) / "coverage_metrics.json"
        report = read_json(path); report["num_queries"] = 500; write_json(path, report)
        with self.assertRaisesRegex(ValueError, "formal cross-combination test protocol"):
            register.proposal(self.output, root=self.root)

    def test_original_settings_and_no_exact_reports_are_enforced(self):
        original = deepcopy(self.manifest["context"]["settings"])
        self.manifest["context"]["settings"]["ks"] = [1, 10]
        with self.assertRaisesRegex(ValueError, "original settings"):
            register.proposal(self.output, root=self.root)
        self.manifest["context"]["settings"] = original
        folder = register.evaluation_folder(self.output, self.names[0])
        write_json(folder / "metrics.json", {"metric_version": "exact_v1"})
        with self.assertRaisesRegex(ValueError, "coverage only"):
            register.proposal(self.output, root=self.root)

    def test_candidate_identity_and_filter_metadata_must_match_frozen_pool(self):
        for key, value in (("candidate_identity_sha256", "different"),
                           ("candidate_count_min", 7198),
                           ("base_protocol_signature", "different"),
                           ("filter_uses_ground_truth", False)):
            with self.subTest(key=key):
                original = self.derived[key]
                self.derived[key] = value
                with self.assertRaisesRegex(ValueError, "candidate protocol"):
                    register.proposal(self.output, root=self.root)
                self.derived[key] = original

    def test_metric_protocol_and_valid_query_counts_cannot_be_relabelled(self):
        path = register.evaluation_folder(self.output, self.names[0]) / "coverage_metrics.json"
        original = read_json(path)
        report = deepcopy(original)
        report["metadata"]["derived_protocol"]["candidate_identity_sha256"] = "ordinary_pool"
        write_json(path, report)
        with self.assertRaisesRegex(ValueError, "formal cross-combination test protocol"):
            register.proposal(self.output, root=self.root)
        report = deepcopy(original)
        report["metric_query_counts"]["Coverage-nDCG@10"] = 7199
        write_json(path, report)
        with self.assertRaisesRegex(ValueError, "zero-denominator rules"):
            register.proposal(self.output, root=self.root)
        write_json(path, original)

    def test_incomplete_run_rejected_before_proposal_or_catalogue_writes(self):
        before = self.snapshot()
        with patch.object(register, "verify_run", side_effect=ValueError("Run is incomplete")), \
                patch.object(register.catalog, "collect") as collect:
            with self.assertRaisesRegex(ValueError, "incomplete"):
                register.apply_proposal(self.output, root=self.root)
            collect.assert_not_called()
        self.assertEqual(self.snapshot(), before)
        self.assertFalse((self.output / "catalog_sources.json").exists())

    def test_apply_preserves_old_record_compacts_snapshot_and_is_idempotent(self):
        self.original = self.snapshot()
        with patch.object(register.catalog, "collect", side_effect=self.collect) as collect, \
                patch.object(register.catalog, "render", side_effect=self.render):
            result = register.apply_proposal(self.output, root=self.root)
            self.assertEqual(result["new_results"], 1)
            snapshot_path = self.catalog / "snapshot.json"
            current = read_json(snapshot_path)
            self.assertEqual(current["runs"][0], self.previous["runs"][0])
            self.assertEqual(snapshot_path.read_bytes(), register._compact_bytes(current))
            self.assertIn("保留原始来源".encode(), snapshot_path.read_bytes())
            backup = Path(result["backup"])
            self.assertEqual(read_json(backup / "manifest.json")["status"], "committed")
            for name, content in self.original.items():
                self.assertEqual((backup / "files" / name).read_bytes(), content)
            before = self.snapshot()
            self.assertEqual(register.apply_proposal(self.output, root=self.root)["status"], "already_registered")
            self.assertEqual(self.snapshot(), before)
            self.assertEqual(collect.call_count, 1)

    def test_changed_old_record_rejected_before_any_catalogue_mutation(self):
        self.original = self.snapshot()
        def changed(*args):
            value = self.collect(*args)
            value["runs"][0]["provenance"]["nested"][0] = .5
            return value
        with patch.object(register.catalog, "collect", side_effect=changed), \
                patch.object(register.catalog, "render", side_effect=self.render):
            with self.assertRaisesRegex(ValueError, "existing catalogue record changed"):
                register.apply_proposal(self.output, root=self.root)
        self.assertEqual(self.snapshot(), self.original)
        self.assertFalse((self.output / "catalog_backup").exists())

    def test_missing_collected_row_and_file_size_limit_fail_before_writes(self):
        self.original = self.snapshot()
        def missing(*args):
            value = self.collect(*args); value["runs"].pop(); return value
        with patch.object(register.catalog, "collect", side_effect=missing), \
                patch.object(register.catalog, "render", side_effect=self.render):
            with self.assertRaisesRegex(ValueError, "omitted or duplicated"):
                register.apply_proposal(self.output, root=self.root)
        with patch.object(register.catalog, "collect", side_effect=self.collect), \
                patch.object(register.catalog, "render", side_effect=self.render), \
                patch.object(register, "MAX_CATALOG_FILE_BYTES", 100):
            with self.assertRaisesRegex(ValueError, "100 MiB file limit"):
                register.apply_proposal(self.output, root=self.root)
        self.assertEqual(self.snapshot(), self.original)
        self.assertFalse((self.output / "catalog_backup").exists())

    def test_injected_write_failure_rolls_back_all_catalogue_files(self):
        self.original = self.snapshot()
        atomic_write = register._atomic_write
        failed = False
        def fail_once(path, content):
            nonlocal failed
            if path == self.catalog / "metrics.csv" and not failed:
                failed = True
                raise OSError("injected registration write failure")
            atomic_write(path, content)
        with patch.object(register.catalog, "collect", side_effect=self.collect), \
                patch.object(register.catalog, "render", side_effect=self.render), \
                patch.object(register, "_atomic_write", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "injected registration"):
                register.apply_proposal(self.output, root=self.root)
        self.assertEqual(self.snapshot(), self.original)
        backup, = list((self.output / "catalog_backup").iterdir())
        self.assertEqual(read_json(backup / "manifest.json")["status"], "rolled_back")
        self.assertFalse((self.catalog / ".registration.lock").exists())

    def test_conflicting_key_and_stale_catalogue_fail_before_collect(self):
        item = deepcopy(register.proposal(self.output, root=self.root)["runs"][0])
        item["display"] = "conflicting source"
        sources = {"schema_version": 1, "runs": [item]}
        previous = {"source_spec_sha256": self.fingerprint(sources), "runs": [item]}
        self.publish_catalog(sources, previous)
        with patch.object(register.catalog, "collect") as collect, \
                patch.object(register.catalog, "render", side_effect=self.render):
            with self.assertRaisesRegex(ValueError, "Conflicting catalogue key"):
                register.apply_proposal(self.output, root=self.root)
            collect.assert_not_called()
            (self.catalog / "metrics.csv").write_text("stale")
            with self.assertRaisesRegex(ValueError, "stale"):
                register.apply_proposal(self.output, root=self.root)
            collect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
