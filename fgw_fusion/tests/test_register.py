"""Six-evaluation registration, complete evidence gates and atomic rollback."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fgw_fusion import register, scoring
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
        self.output = self.root / "artificial/fgw_fusion_20261119T180000+0800"
        source = self.root / "artificial/frozen_source"
        dump(source / "run_manifest.json", {"frozen": True})
        self.settings = dict(scoring.SETTINGS)
        self.specs = scoring.specs()
        self.context = {"plan": {"source_root": str(source), "settings": self.settings,
            "mode": "all", "splits": ["valid", "test"]}, "code": {},
            "source_manifest_sha256": register.file_hash(source / "run_manifest.json")}
        self.manifest = signed({"version": "fgw_fusion_v1", "context": self.context})
        dump(self.output / "run_manifest.json", self.manifest)
        dump(self.output / "source_protection_before.json", {})
        dump(self.output / "source_protection_after.json", {"status": "unchanged", "files": 0, "inventory_digest": digest({})})
        for split in ("valid", "test"):
            count = 500 if split == "valid" else 7199
            inp = signed({"parent_signature": self.manifest["signature"], "split": split,
                "specs": self.specs, "code": {}, "provenance": {"protocol_signature": register.catalog.BASE_PROTOCOL_SIGNATURE}})
            dump(self.output / f"inputs_{split}.json", inp)
            hashes, signatures, verification = {}, {}, {}
            for name, spec in self.specs.items():
                folder = self.output / split / name
                signature = digest({"evaluation": inp["signature"], "variant": name, "spec": spec,
                    "split": split, "settings": self.settings, "device": "cpu",
                    "retrieval_protocol_signature": register.catalog.BASE_PROTOCOL_SIGNATURE})
                key = f"fixture_{split}_{name}"
                identity = {"schema_version": 1, "key": key, "split": split, "variant": name,
                    "evaluation_signature": signature, "started_at": "2026-11-19T18:00:00+08:00",
                    "run_id": "20261119T180000+0800__" + key}
                dump(folder / "run_identity.json", identity)
                dump(folder / "evaluation_manifest.json", {**identity, "signature": signature,
                    "parent_signature": inp["signature"], "spec": spec,
                    "ks": self.settings["ks"], "tie_seed": self.settings["tie_seed"]})
                report = {"split": split, "num_queries": count, "ks": self.settings["ks"],
                    "metadata": {"signature": signature, "retrieval_protocol_signature": register.catalog.BASE_PROTOCOL_SIGNATURE}}
                dump(folder / "metrics.json", {**report, "metric_version": "exact_v1", "metrics": {"Precision@10": .5}})
                dump(folder / "coverage_metrics.json", {**report, "metric_version": "coverage_v1", "metrics": {"Coverage-nDCG@10": .8}})
                for filename in ("per_query.csv", "coverage_per_query.csv", "rankings.csv"):
                    (folder / filename).write_text("synthetic integrity fixture only\n")
                dump(folder / "completed.json", {"signature": signature, "files": {
                    p.name: register.file_hash(p) for p in folder.iterdir()}})
                hashes[name] = register.file_hash(folder / "completed.json")
                signatures[name] = signature
                verification[name] = {"status": "verified", "split": split, "variant": name,
                    "num_queries": count, "ranking_rows": count * max(self.settings["ks"]),
                    "completion_sha256": hashes[name], "ranking_sha256": register.file_hash(folder / "rankings.csv")}
            dump(self.output / f"evaluation_index_{split}.json", {"signature": inp["signature"], "split": split,
                "variants": self.specs, "variant_signatures": signatures, "completed_sha256": hashes})
            dump(self.output / f"verification_{split}.json", verification)
            dump(self.output / f"source_audit_{split}.json", signed({"status": "verified", "input_context_signature": digest(self.context)}))
            analysis = self.output / "analysis" / split
            dump(analysis / "analysis_plan.json", {"anchor": "FGW", "ks": self.settings["ks"],
                "bootstrap_replicates": 10000, "seed": 2026})
            for name in ("metrics.csv", "paired_bootstrap.csv"):
                (analysis / name).write_text("synthetic integrity fixture only\n")
            refs = {name: {str(self.output / split / name / "metrics.json"):
                          register.file_hash(self.output / split / name / "metrics.json")} for name in self.specs}
            dump(analysis / "sources.json", refs)
            dump(analysis / "completed.json", {"status": "complete", "split": split, "num_queries": count,
                "num_classes": 37, "methods": list(self.specs),
                "files": {p.name: register.file_hash(p) for p in analysis.iterdir()}})
        dump(self.output / "status.json", {"status": "completed"})
        self.seal_root()
        self.catalog = self.root / "results_catalog"
        self.sources = {"schema_version": 1, "runs": []}
        self.previous = {"source_spec_sha256": self.fingerprint(self.sources), "runs": []}
        dump(self.catalog / "sources.json", self.sources)
        dump(self.catalog / "snapshot.json", self.previous)
        for name, text in self.render(self.previous).items():
            (self.catalog / name).write_text(text)

    def seal_root(self):
        marker = self.output / "completed.json"
        dump(marker, {"status": "completed", "signature": self.manifest["signature"],
            "files": {str(p.relative_to(self.output)): register.file_hash(p)
                      for p in self.output.rglob("*") if p.is_file() and p != marker}})

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

    def test_proposal_six_new_self_supervised_identities_idempotent(self):
        before = self.snapshot()
        path = register.write_proposal(self.output, root=self.root)
        value = register.catalog.read(path)
        self.assertEqual(len(value["runs"]), 6)
        self.assertEqual({row["variant"] for row in value["runs"]}, {"FGW/FGW", "FGW/GW", "FGW/FW"})
        self.assertEqual({row["split"] for row in value["runs"]}, {"valid", "test"})
        self.assertTrue(all(row["supervision"] == "self_supervised" and row["posthoc"] for row in value["runs"]))
        self.assertTrue(all(row["selection"] is None and row["identity_schema_version"] == 1 for row in value["runs"]))
        stamp = path.stat().st_mtime_ns
        self.assertEqual(register.write_proposal(self.output, root=self.root), path)
        self.assertEqual(path.stat().st_mtime_ns, stamp)
        self.assertEqual(before, self.snapshot())

    def test_incomplete_or_changed_root_bound_evidence_is_rejected(self):
        marker = self.output / "completed.json"
        value = register.catalog.read(marker)
        del value["files"]["source_audit_test.json"]
        dump(marker, value)
        with self.assertRaisesRegex(ValueError, "omits required"):
            register.proposal(self.output, root=self.root)
        self.seal_root()
        (self.output / "analysis/test/paired_bootstrap.csv").write_text("corrupt")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            register.proposal(self.output, root=self.root)

    def test_resealed_stale_audit_and_bad_protection_fail_semantically(self):
        path = self.output / "source_audit_test.json"
        dump(path, signed({"status": "verified", "input_context_signature": "stale"}))
        self.seal_root()
        with self.assertRaisesRegex(ValueError, "stale successful source audit"):
            register.proposal(self.output, root=self.root)
        dump(path, signed({"status": "verified", "input_context_signature": digest(self.context)}))
        dump(self.output / "source_protection_after.json", {"status": "unchanged", "files": 0, "inventory_digest": "stale"})
        self.seal_root()
        with self.assertRaisesRegex(ValueError, "Source protection"):
            register.proposal(self.output, root=self.root)

    def test_resealed_stale_verification_and_changed_source_rejected(self):
        path = self.output / "verification_valid.json"
        value = register.catalog.read(path); value["FGW"]["ranking_sha256"] = "stale"
        dump(path, value); self.seal_root()
        with self.assertRaisesRegex(ValueError, "stale independent verification"):
            register.proposal(self.output, root=self.root)
        source = Path(self.context["plan"]["source_root"]) / "run_manifest.json"
        dump(source, {"frozen": False})
        with self.assertRaisesRegex(ValueError, "source manifest changed"):
            register.proposal(self.output, root=self.root)

    def test_apply_collects_before_writes_backs_up_and_is_idempotent(self):
        self.original = self.snapshot()
        with patch.object(register.catalog, "collect", side_effect=self.collect) as collect, \
                patch.object(register.catalog, "render", side_effect=self.render):
            result = register.apply_proposal(self.output, workspace=self.workspace, root=self.root)
            self.assertEqual(result["new_results"], 6)
            self.assertEqual(collect.call_count, 1)
            backup = Path(result["backup"])
            self.assertEqual(register.catalog.read(backup / "manifest.json")["status"], "committed")
            for name, data in self.original.items():
                self.assertEqual((backup / "files" / name).read_bytes(), data)
            current = self.snapshot()
            self.assertEqual(register.apply_proposal(self.output, workspace=self.workspace, root=self.root)["status"], "already_registered")
            self.assertEqual(self.snapshot(), current)
            self.assertEqual(collect.call_count, 1)

    def test_write_failure_restores_all_shared_files(self):
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

    def test_conflicting_key_and_stale_catalog_rejected_before_collection(self):
        item = copy.deepcopy(register.proposal(self.output, root=self.root)["runs"][0])
        item["display"] = "conflicting existing source"
        sources = {"schema_version": 1, "runs": [item]}
        previous = {"source_spec_sha256": self.fingerprint(sources), "runs": sources["runs"]}
        dump(self.catalog / "sources.json", sources); dump(self.catalog / "snapshot.json", previous)
        for name, text in self.render(previous).items(): (self.catalog / name).write_text(text)
        with patch.object(register.catalog, "collect") as collect, patch.object(register.catalog, "render", side_effect=self.render):
            with self.assertRaisesRegex(ValueError, "Conflicting catalogue key"):
                register.apply_proposal(self.output, root=self.root)
            self.assertFalse(collect.called)
            (self.catalog / "metrics.csv").write_text("stale")
            with self.assertRaisesRegex(ValueError, "stale"):
                register.apply_proposal(self.output, root=self.root)


if __name__ == "__main__":
    unittest.main()
