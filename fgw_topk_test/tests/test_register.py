"""Top-10 registration semantics and transactions, using temporary metadata only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fgw_grid_search import scoring, selection
from fgw_topk_test import register
from wbm_retrieval.common import file_hash, read_json, write_json


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name).resolve()
        self.root = self.workspace / "wbm-region-retrieval"
        self.output = self.root / "artificial/fgw_topk_test_20261119T220000+0800"
        self.settings = deepcopy(scoring.SETTINGS)
        all_specs = scoring.specs(self.settings)
        rows = [{"variant": name, "split": "valid", "num_queries": 500,
                 "precision10_hits": 5000 - i, "Precision@10": (5000 - i) / 5000,
                 "mAP@100": .25, "Coverage-nDCG@10": .75}
                for i, name in enumerate(all_specs)]
        chosen = selection.choose_winner(rows, all_specs, self.settings)
        self.names = [row["variant"] for row in chosen["ranking"][:10]]
        specs = {name: all_specs[name] for name in self.names}
        self.frozen = {"top_k": 10, "ranked_variants": self.names.copy(), "specs": specs,
                       "valid_selection": chosen, "primary_variant": chosen["winner"],
                       "selection_split": "valid", "test_parameter_search": False}
        self.manifest = {"stage": "test", "context": {"split": "test", "settings": self.settings,
            "specs": specs, "source_run": "artificial/frozen_source",
            "evaluation_paths": register.evaluation_paths(self.output, self.names)}}
        # The run-state implementation independently owns sealing/cache validation.
        # Here it is mocked so registration tests never touch formal sources or score data.
        mocked = patch.object(register, "verify_run", return_value=(self.manifest, self.frozen))
        mocked.start(); self.addCleanup(mocked.stop)
        write_json(self.output / "run_manifest.json", self.manifest)
        write_json(self.output / "frozen_selection.json", self.frozen)
        for name, spec in specs.items():
            folder = register.evaluation_folder(self.output, name)
            signature = "synthetic_" + name
            key = "fixture_20261119T220000_test_" + name
            identity = {"schema_version": 1, "key": key, "variant": name, "split": "test",
                        "evaluation_signature": signature, "started_at": "2026-11-19T22:00:00+08:00",
                        "run_id": "20261119T220000+0800__" + key}
            write_json(folder / "run_identity.json", identity)
            write_json(folder / "evaluation_manifest.json", {**identity, "signature": signature,
                "stage": "test", "spec": spec, "settings": self.settings})
            report = {"split": "test", "num_queries": 7199, "ks": self.settings["ks"],
                      "metadata": {"signature": signature,
                          "retrieval_protocol_signature": register.catalog.BASE_PROTOCOL_SIGNATURE}}
            write_json(folder / "metrics.json", {**report, "metric_version": "exact_v1"})
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

    def test_proposal_exact_ten_rank_order_primary_identity_and_zero_write_resume(self):
        before = self.snapshot()
        path = register.write_proposal(self.output, root=self.root)
        rows = read_json(path)["runs"]
        self.assertEqual(len(rows), 10)
        self.assertEqual([row["variant"] for row in rows], ["FGW-top10-test/" + name for name in self.names])
        self.assertEqual([row["valid_rank"] for row in rows], list(range(1, 11)))
        self.assertEqual([row["valid_selected_primary"] for row in rows], [True] + [False] * 9)
        self.assertTrue(all(row["supervision"] == "self_supervised" and row["posthoc"] for row in rows))
        self.assertTrue(all(row["split"] == "test" and row["required_families"] == ["exact", "coverage"] for row in rows))
        self.assertEqual(rows[0]["path"], str(register.evaluation_folder(self.output, self.names[0]).relative_to(self.root)))
        self.assertEqual(rows[0]["selection"], str((self.output / "frozen_selection.json").relative_to(self.root)))
        stamp = path.stat().st_mtime_ns
        self.assertEqual(register.write_proposal(self.output, root=self.root), path)
        self.assertEqual(path.stat().st_mtime_ns, stamp)
        self.assertEqual(self.snapshot(), before)

    def test_wrong_order_missing_formula_primary_or_test_selection_rejected(self):
        original = deepcopy(self.frozen)
        mutations = [lambda d: d["ranked_variants"].reverse(),
                     lambda d: d["ranked_variants"].pop(),
                     lambda d: d.update(primary_variant=self.names[1]),
                     lambda d: d.update(test_parameter_search=True)]
        for change in mutations:
            with self.subTest(change=change):
                self.frozen.clear(); self.frozen.update(deepcopy(original)); change(self.frozen)
                with self.assertRaisesRegex(ValueError, "exactly the frozen valid Top 10"):
                    register.proposal(self.output, root=self.root)

    def test_non_test_pool_or_changed_evaluation_paths_rejected(self):
        self.manifest["context"]["evaluation_paths"][self.names[0]] = "../other"
        with self.assertRaisesRegex(ValueError, "evaluation paths"):
            register.proposal(self.output, root=self.root)
        self.manifest["context"]["evaluation_paths"] = register.evaluation_paths(self.output, self.names)
        path = register.evaluation_folder(self.output, self.names[0]) / "metrics.json"
        report = read_json(path); report["num_queries"] = 500; write_json(path, report)
        with self.assertRaisesRegex(ValueError, "formal frozen test protocol"):
            register.proposal(self.output, root=self.root)

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
            self.assertEqual(result["new_results"], 10)
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
