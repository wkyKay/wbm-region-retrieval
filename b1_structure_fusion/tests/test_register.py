"""Registration stays optional, verifies frozen outputs, and rolls back shared writes."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from b1_structure_fusion import descriptors, register, scoring
from wbm_retrieval.common import digest


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(register._json_bytes(value))


def signed(value):
    return {**value, "signature": digest(value)}


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workspace = Path(self.tmp.name).resolve()
        self.root = self.workspace / "wbm-region-retrieval"
        self.output = self.root / "artificial/fixture_valid_20261117T093000+0800"
        self.settings = descriptors.load_settings()
        self.context = {"settings": self.settings}
        self.specs = scoring.valid_variants(self.settings)
        manifest = signed({"version": "b1_structure_fusion_v1", "stage": "valid", "context": self.context})
        dump(self.output / "run_manifest.json", manifest)
        self.inputs = signed({"context": self.context, "split": "valid"})
        dump(self.output / "inputs_valid.json", self.inputs)
        hashes, signatures, verification, values = {}, {}, {}, {}
        for name, spec in self.specs.items():
            folder = self.output / "valid" / name
            signature = digest({"evaluation": self.inputs["signature"], "variant": name, "spec": spec})
            key = "fixture_" + name.replace(".", "p") + "_valid"
            identity = {"schema_version": 1, "key": key, "split": "valid", "variant": name,
                        "started_at": "2026-11-17T09:30:00+08:00", "run_id": "20261117T093000+0800__" + key}
            dump(folder / "run_identity.json", identity)
            dump(folder / "evaluation_manifest.json", {**identity, "signature": signature,
                 "parent_signature": self.inputs["signature"], "spec": spec})
            report_base = {"split": "valid", "num_queries": 500, "ks": self.settings["ks"],
                           "metadata": {"signature": signature,
                                        "retrieval_protocol_signature": register.catalog.BASE_PROTOCOL_SIGNATURE}}
            dump(folder / "metrics.json", {**report_base, "metric_version": "exact_v1", "metrics": {"Precision@10": .5}})
            dump(folder / "coverage_metrics.json", {**report_base, "metric_version": "coverage_v1", "metrics": {"MeanCoverage@10": .8}})
            for filename in ("per_query.csv", "coverage_per_query.csv", "rankings.csv"):
                (folder / filename).write_text("synthetic fixture only\n")
            dump(folder / "completed.json", {"signature": signature, "files": {
                p.name: register.catalog.sha(p) for p in folder.iterdir()}})
            hashes[name] = register.catalog.sha(folder / "completed.json")
            signatures[name] = signature
            values[name] = .5
            verification[name] = {"status": "verified", "split": "valid", "variant": name,
                "num_queries": 500, "ranking_rows": 500 * max(self.settings["ks"]),
                "completion_sha256": hashes[name], "ranking_sha256": register.catalog.sha(folder / "rankings.csv")}
        self.index = {"signature": self.inputs["signature"], "split": "valid", "variants": self.specs,
                      "variant_signatures": signatures, "completed_sha256": hashes}
        dump(self.output / "evaluation_index_valid.json", self.index)
        dump(self.output / "verification_valid.json", verification)
        choices = {}
        for family in ("R", "T", "I", "J"):
            candidates = [{"variant": name, "strength": spec["strength"], "value": values[name]}
                          for name, spec in self.specs.items() if spec["family"] in ("B0", family)]
            choices[family] = {"selected": candidates[0], "candidates": candidates}
        selection = signed({"split": "valid", "test_evaluated": False, "settings": self.settings,
                            "context": self.context, "primary_family": "J", "selections": choices,
                            "completed_sha256": hashes, "variant_signatures": signatures})
        dump(self.output / "selection.json", selection)
        dump(self.output / "status.json", {"status": "completed", "test_evaluated": False})
        self.catalog = self.root / "results_catalog"
        self.sources = {"schema_version": 1, "runs": []}
        self.previous = {"source_spec_sha256": self.fingerprint(self.sources), "runs": []}
        dump(self.catalog / "sources.json", self.sources)
        dump(self.catalog / "snapshot.json", self.previous)
        for name, text in self.render(self.previous).items():
            (self.catalog / name).write_text(text)

    @staticmethod
    def fingerprint(sources):
        return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def render(snapshot):
        keys = ",".join(item["key"] for item in snapshot["runs"])
        return {"registry.csv": "keys\n" + keys + "\n", "metrics.csv": f"count\n{len(snapshot['runs'])}\n"}

    def snapshot(self):
        return {p.name: p.read_bytes() for p in self.catalog.iterdir() if p.is_file() and not p.name.startswith(".")}

    def collect(self, workspace, sources, previous):
        self.assertEqual(workspace, self.workspace)
        self.assertEqual(previous, self.previous)
        self.assertEqual(self.snapshot(), self.original)
        return {"source_spec_sha256": self.fingerprint(sources), "runs": sources["runs"]}

    def test_proposal_is_complete_idempotent_and_does_not_touch_shared_catalogue(self):
        before = self.snapshot()
        attempt = self.output / "valid/.failed_attempt"
        attempt.mkdir()
        (attempt / "partial.csv").write_text("unfinished")
        path = register.write_proposal(self.output, root=self.root)
        value = register.catalog.read(path)
        self.assertEqual(len(value["runs"]), 18)
        self.assertEqual({row["variant"] for row in value["runs"]}, {"B1/" + name for name in self.specs})
        self.assertTrue(all(row["identity_schema_version"] == 1 for row in value["runs"]))
        self.assertTrue(all(row["category"] == "fusion_ablation" for row in value["runs"]))
        stamp = path.stat().st_mtime_ns
        self.assertEqual(register.write_proposal(self.output, root=self.root), path)
        self.assertEqual(path.stat().st_mtime_ns, stamp)
        self.assertEqual(before, self.snapshot())

    def test_failed_or_unverified_outputs_are_not_proposed(self):
        dump(self.output / "status.json", {"status": "failed"})
        with self.assertRaisesRegex(ValueError, "completed, verified"):
            register.write_proposal(self.output, root=self.root)
        dump(self.output / "status.json", {"status": "completed", "test_evaluated": False})
        checked = register.catalog.read(self.output / "verification_valid.json")
        checked["B0"]["ranking_sha256"] = "changed"
        dump(self.output / "verification_valid.json", checked)
        with self.assertRaisesRegex(ValueError, "verification is missing, stale"):
            register.write_proposal(self.output, root=self.root)

    def test_changed_selection_or_source_signature_is_rejected(self):
        path = self.output / "run_manifest.json"
        value = register.catalog.read(path)
        value["context"] = {"settings": {}}
        dump(path, value)
        with self.assertRaisesRegex(ValueError, "Invalid signed"):
            register.write_proposal(self.output, root=self.root)

    def test_test_only_uses_external_frozen_selection_and_only_selected_variants(self):
        selection_path = self.output / "selection.json"
        selection = register.catalog.read(selection_path)
        output = self.root / "artificial/fixture_test_20261117T094000+0800"
        dump(output / "run_manifest.json", signed({"version": "b1_structure_fusion_v1", "stage": "test",
             "context": self.context, "selection_file": str(selection_path)}))
        inputs = signed({"context": self.context, "split": "test"})
        dump(output / "inputs_test.json", inputs)
        specs = register.pipeline.test_variants(selection, self.settings)
        signatures, hashes, verification = {}, {}, {}
        for name, spec in specs.items():
            source = self.output / "valid" / name
            target = output / "test" / name
            target.mkdir(parents=True)
            for path in source.iterdir():
                if path.name != "completed.json":
                    (target / path.name).write_bytes(path.read_bytes())
            signature = digest({"evaluation": inputs["signature"], "variant": name, "spec": spec})
            key = "fixture_" + name + "_test"
            identity = {"schema_version": 1, "key": key, "split": "test", "variant": name,
                        "started_at": "2026-11-17T09:40:00+08:00", "run_id": "20261117T094000+0800__" + key}
            dump(target / "run_identity.json", identity)
            dump(target / "evaluation_manifest.json", {**identity, "signature": signature,
                 "parent_signature": inputs["signature"], "spec": spec})
            for filename in ("metrics.json", "coverage_metrics.json"):
                report = register.catalog.read(target / filename)
                report.update(split="test", num_queries=7199)
                report["metadata"]["signature"] = signature
                dump(target / filename, report)
            dump(target / "completed.json", {"signature": signature, "files": {
                p.name: register.catalog.sha(p) for p in target.iterdir()}})
            signatures[name] = signature
            hashes[name] = register.catalog.sha(target / "completed.json")
            verification[name] = {"status": "verified", "split": "test", "variant": name,
                "num_queries": 7199, "ranking_rows": 7199 * max(self.settings["ks"]),
                "completion_sha256": hashes[name], "ranking_sha256": register.catalog.sha(target / "rankings.csv")}
        dump(output / "verification_test.json", verification)
        dump(output / "evaluation_index_test.json", {"signature": inputs["signature"], "split": "test",
             "variants": specs, "variant_signatures": signatures, "completed_sha256": hashes})
        dump(output / "status.json", {"status": "completed", "test_evaluated": True})
        value = register.proposal(output, root=self.root)
        self.assertEqual(len(value["runs"]), 2)
        self.assertEqual({item["variant"] for item in value["runs"]}, {"B1/B0", "B1/J_only"})
        self.assertTrue(all(item["selection"] == str(selection_path.relative_to(self.root)) for item in value["runs"]))
        self.assertTrue(all(item["split"] == "test" for item in value["runs"]))

    def test_apply_verifies_in_memory_then_backs_up_and_is_idempotent(self):
        self.original = self.snapshot()
        with patch.object(register.catalog, "collect", side_effect=self.collect) as collect, \
                patch.object(register.catalog, "render", side_effect=self.render):
            result = register.apply_proposal(self.output, workspace=self.workspace, root=self.root)
            self.assertEqual(result["new_results"], 18)
            self.assertEqual(collect.call_count, 1)
            backup = Path(result["backup"])
            audit = register.catalog.read(backup / "manifest.json")
            self.assertEqual(audit["status"], "committed")
            for name, content in self.original.items():
                self.assertEqual((backup / "files" / name).read_bytes(), content)
            current = self.snapshot()
            second = register.apply_proposal(self.output, workspace=self.workspace, root=self.root)
            self.assertEqual(second["status"], "already_registered")
            self.assertEqual(self.snapshot(), current)
            self.assertEqual(collect.call_count, 1)

    def test_write_failure_rolls_back_catalogue_and_preserves_backup(self):
        self.original = self.snapshot()
        original_write = register._atomic_write
        failed = False

        def fail_once(path, content):
            nonlocal failed
            if path == self.catalog / "metrics.csv" and not failed:
                failed = True
                raise OSError("injected write failure")
            original_write(path, content)

        with patch.object(register.catalog, "collect", side_effect=self.collect), \
                patch.object(register.catalog, "render", side_effect=self.render), \
                patch.object(register, "_atomic_write", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "injected write failure"):
                register.apply_proposal(self.output, root=self.root)
        self.assertEqual(self.snapshot(), self.original)
        backups = list((self.output / "catalog_backup").iterdir())
        self.assertEqual(len(backups), 1)
        self.assertEqual(register.catalog.read(backups[0] / "manifest.json")["status"], "rolled_back")
        self.assertFalse((self.catalog / ".registration.lock").exists())

    def test_conflicting_key_and_stale_catalogue_rejected_before_collect(self):
        proposed = register.proposal(self.output, root=self.root)
        conflicting = copy.deepcopy(proposed["runs"][0])
        conflicting["display"] = "changed"
        sources = {"schema_version": 1, "runs": [conflicting]}
        snapshot = {"source_spec_sha256": self.fingerprint(sources), "runs": sources["runs"]}
        dump(self.catalog / "sources.json", sources)
        dump(self.catalog / "snapshot.json", snapshot)
        for name, text in self.render(snapshot).items():
            (self.catalog / name).write_text(text)
        with patch.object(register.catalog, "collect") as collect, patch.object(register.catalog, "render", side_effect=self.render):
            with self.assertRaisesRegex(ValueError, "Conflicting catalogue key"):
                register.apply_proposal(self.output, root=self.root)
            self.assertFalse(collect.called)
            (self.catalog / "metrics.csv").write_text("stale")
            with self.assertRaisesRegex(ValueError, "stale"):
                register.apply_proposal(self.output, root=self.root)


if __name__ == "__main__":
    unittest.main()
