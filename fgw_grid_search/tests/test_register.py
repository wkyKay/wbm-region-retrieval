"""Registration gates and transactional catalogue recovery in temporary trees."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fgw_grid_search import cache, register, state
from fgw_grid_search.tests.test_run_state import metadata_provider, sealed_fixture, seal_root
from wbm_retrieval.common import file_hash, read_json, write_json


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name).resolve()
        self.root = self.workspace / "wbm-region-retrieval"
        self.output, self.manifest, self.index, self.chosen = sealed_fixture(self.root)
        mocked_cache = patch.object(cache, "FullScoreCache", return_value=metadata_provider(self.output, self.index))
        mocked_cache.start(); self.addCleanup(mocked_cache.stop)
        self.catalog = self.root / "results_catalog"
        self.sources = {"schema_version": 1, "runs": []}
        self.previous = {"source_spec_sha256": self.fingerprint(self.sources), "runs": []}
        write_json(self.catalog / "sources.json", self.sources)
        write_json(self.catalog / "snapshot.json", self.previous)
        for name, text in self.render(self.previous).items():
            (self.catalog / name).write_text(text)

    @staticmethod
    def fingerprint(sources):
        return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def render(snapshot):
        keys = ",".join(row["key"] for row in snapshot["runs"])
        return {"registry.csv": "keys\n" + keys + "\n", "metrics.csv": "count\n" + str(len(snapshot["runs"])) + "\n"}

    def snapshot(self):
        return {p.name: p.read_bytes() for p in self.catalog.iterdir() if p.is_file() and not p.name.startswith(".")}

    def collect(self, workspace, sources, previous):
        self.assertEqual(workspace, self.workspace)
        self.assertEqual(previous, self.previous)
        self.assertEqual(self.snapshot(), self.original)
        return {"source_spec_sha256": self.fingerprint(sources), "runs": sources["runs"]}

    def test_positive_sealed_grid_proposal_identity_selection_and_idempotence(self):
        before = self.snapshot()
        path = register.write_proposal(self.output, root=self.root)
        proposal = read_json(path)
        self.assertEqual(len(proposal["runs"]), 1)
        row = proposal["runs"][0]
        self.assertTrue(row["key"].startswith("fgw_grid_"))
        self.assertEqual(row["split"], "valid")
        self.assertEqual(row["supervision"], "self_supervised")
        self.assertTrue(row["posthoc"])
        self.assertEqual(row["required_families"], ["exact", "coverage"])
        self.assertEqual(row["selection"], str((self.output / "selection.json").relative_to(self.root)))
        stamp = path.stat().st_mtime_ns
        self.assertEqual(register.write_proposal(self.output, root=self.root), path)
        self.assertEqual(path.stat().st_mtime_ns, stamp)
        self.assertEqual(self.snapshot(), before)

    def test_incomplete_run_cannot_prepare_or_apply_registration(self):
        before = self.snapshot()
        status = read_json(self.output / "status.json")
        status["status"] = "running"
        write_json(self.output / "status.json", status)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            register.proposal(self.output, root=self.root)
        with patch.object(register.catalog, "collect") as collect:
            with self.assertRaisesRegex(ValueError, "incomplete"):
                register.apply_proposal(self.output, root=self.root)
            collect.assert_not_called()
        self.assertEqual(self.snapshot(), before)
        self.assertFalse((self.output / "catalog_sources.json").exists())

    def test_resealed_incomplete_full_pool_verification_is_rejected(self):
        name = next(iter(self.index["variants"]))
        folder = self.output / "valid" / name
        checked = read_json(folder / "verification.json")
        checked["full_candidates_checked"] -= 1
        write_json(folder / "verification.json", checked)
        completed = read_json(folder / "completed.json")
        completed["files"]["verification.json"] = file_hash(folder / "verification.json")
        write_json(folder / "completed.json", completed)
        self.index["completed_sha256"][name] = file_hash(folder / "completed.json")
        index_path = self.output / "evaluation_index_valid.json"
        write_json(index_path, self.index)
        write_json(self.output / "selection.json", state.select_payload(self.manifest, self.index, index_path))
        seal_root(self.output, self.manifest, len(self.index["variants"]))
        with self.assertRaisesRegex(ValueError, "complete frozen pool"):
            register.proposal(self.output, root=self.root)

    def test_apply_collects_first_then_backs_up_commits_and_repeats_without_writes(self):
        self.original = self.snapshot()
        with patch.object(register.catalog, "collect", side_effect=self.collect) as collect, \
                patch.object(register.catalog, "render", side_effect=self.render):
            result = register.apply_proposal(self.output, root=self.root)
            self.assertEqual(result["new_results"], 1)
            backup = Path(result["backup"])
            self.assertEqual(read_json(backup / "manifest.json")["status"], "committed")
            for name, content in self.original.items():
                self.assertEqual((backup / "files" / name).read_bytes(), content)
            current = self.snapshot()
            self.assertEqual(register.apply_proposal(self.output, root=self.root)["status"], "already_registered")
            self.assertEqual(self.snapshot(), current)
            self.assertEqual(collect.call_count, 1)

    def test_injected_write_failure_restores_catalogue_and_retains_backup(self):
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
        backups = list((self.output / "catalog_backup").iterdir())
        self.assertEqual(len(backups), 1)
        self.assertEqual(read_json(backups[0] / "manifest.json")["status"], "rolled_back")
        self.assertFalse((self.catalog / ".registration.lock").exists())

    def test_conflicting_key_and_stale_catalogue_fail_before_collect(self):
        item = deepcopy(register.proposal(self.output, root=self.root)["runs"][0])
        item["display"] = "conflicting identity"
        sources = {"schema_version": 1, "runs": [item]}
        previous = {"source_spec_sha256": self.fingerprint(sources), "runs": sources["runs"]}
        write_json(self.catalog / "sources.json", sources)
        write_json(self.catalog / "snapshot.json", previous)
        for name, text in self.render(previous).items(): (self.catalog / name).write_text(text)
        with patch.object(register.catalog, "collect") as collect, patch.object(register.catalog, "render", side_effect=self.render):
            with self.assertRaisesRegex(ValueError, "Conflicting catalogue key"):
                register.apply_proposal(self.output, root=self.root)
            collect.assert_not_called()
            (self.catalog / "metrics.csv").write_text("stale")
            with self.assertRaisesRegex(ValueError, "stale"):
                register.apply_proposal(self.output, root=self.root)
            collect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
