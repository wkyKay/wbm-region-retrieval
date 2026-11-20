"""No-data CLI guards and formal-shaped synthetic selection integrity checks."""
from contextlib import nullcontext, redirect_stdout
from datetime import datetime
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from f_region_structure import run, scoring, selection, state
from wbm_retrieval.common import digest, file_hash, read_json, write_json


def snapshot(folder):
    return {str(path.relative_to(folder)): (file_hash(path), path.stat().st_mtime_ns)
            for path in Path(folder).rglob("*") if path.is_file()}


def selection_fixture(output):
    """Fake 500-count records exercise orchestration only, never real evaluation."""
    output.mkdir(parents=True)
    context = "synthetic_context_for_selection_guard"
    grid, results = scoring.frozen_grid(), {}
    for name, spec in grid.items():
        folder = output / "valid" / name
        folder.mkdir(parents=True)
        signature = digest({"context": context, "split": "valid", "variant": name, "spec": spec})
        write_json(folder / "evaluation_manifest.json", {"signature": signature, "split": "valid",
                   "variant": name, "spec": spec, "fixture_only": True})
        write_json(folder / "verification.json", {"status": "verified", "top10_hits": 2000,
                   "num_queries": 500, "fixture_only": True})
        state.seal(folder, signature)
        results[name] = {"top10_hits": 2000, "num_queries": 500,
                        "completed_sha256": file_hash(folder / "completed.json")}
    index = {"split": "valid", "specs": grid, "results": results, "context_signature": context}
    write_json(output / "evaluation_index_valid.json", index)
    chosen = selection.select_best(results)
    record = {**chosen, "context_signature": context, "valid_index_sha256": file_hash(output / "evaluation_index_valid.json"),
              "models": {}, "fixture_only": True}
    write_json(output / "selection.json", state.signed(record))
    return context, record, index


class CommandGuardTests(unittest.TestCase):
    def test_dry_run_never_loads_sources_or_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = snapshot(root)
            with patch.object(run, "ROOT", root), redirect_stdout(StringIO()) as text, \
                    patch.object(run.sources, "load", side_effect=AssertionError("dry-run read sources")):
                self.assertEqual(run.main(["--dry-run", "--output-root", "artificial/new_run"]), 0)
            plan = json.loads(text.getvalue())
            self.assertEqual(len(plan["grid"]), 25)
            self.assertEqual(plan["stage"], "all")
            self.assertEqual(snapshot(root), before)
            self.assertFalse((root / "artificial").exists())

    def test_test_requires_explicit_resume_before_io(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(run, "ROOT", Path(tmp)):
            with patch.object(run.sources, "load", side_effect=AssertionError("guard did not run first")):
                with self.assertRaisesRegex(ValueError, "Test requires --resume-root"):
                    run.main(["--stage", "test", "--dry-run"])
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_existing_outside_and_nested_outputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(run, "ROOT", Path(tmp)):
            root = Path(tmp)
            existing = root / "artificial" / "old"
            existing.mkdir(parents=True)
            write_json(existing / "run_manifest.json", {"fixture": True})
            before = snapshot(root)
            for path in ("artificial/old", "artifacts/new", "artificial", "artificial/old/nested"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    run.main(["--dry-run", "--output-root", path])
            with self.assertRaisesRegex(ValueError, "Resume requires"):
                run.main(["--dry-run", "--resume-root", "artificial/missing"])
            self.assertEqual(snapshot(root), before)

    def test_resume_rejects_code_signature_change_before_cache_or_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "artificial" / "run"
            output.mkdir(parents=True)
            source_identity = {"frozen_checkpoint": "best_nce", "geometry": {"fixture": True}}
            matching = {"weight": "sqrt_area"}
            context = {"version": run.VERSION, "source": source_identity, "code": {"run.py": "old_hash"},
                       "grid": scoring.frozen_grid(), "selection_rule": selection.RULE, "device": "cpu",
                       "numpy": run.np.__version__, "torch": run.torch.__version__,
                       "training": {"epochs": 20, "batch_size": 512, "seed": 2026, "checkpoint_rule": "final_epoch"},
                       "matching": matching, "posthoc_design": True, "test_parameter_search": False}
            write_json(output / "run_manifest.json", state.signed({"context": context}))
            before = snapshot(root)
            with patch.object(run, "ROOT", root), patch.object(run.geo, "source_lock", return_value=nullcontext()), \
                    patch.object(run.sources, "load", return_value=({"matching": matching}, {}, {}, object(), source_identity)), \
                    patch.object(run.sources, "feature_source", return_value=None), \
                    patch.object(run, "code_identity", return_value={"run.py": "new_hash"}), \
                    patch.object(run.sources, "prepare", side_effect=AssertionError("changed code reached computation")):
                with self.assertRaisesRegex(ValueError, "Source/code/environment changed"):
                    run.main(["--stage", "test", "--device", "cpu", "--resume-root", "artificial/run"])
            self.assertEqual(snapshot(root), before)


class FrozenSelectionTests(unittest.TestCase):
    def test_complete_grid_is_frozen_and_signed_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            context, original, _ = selection_fixture(output)
            self.assertEqual(run.frozen_selection(output, context)["winner"], "baseline")
            changed = read_json(output / "selection.json")
            changed["winner"] = "plain_mlp"
            write_json(output / "selection.json", changed)
            with self.assertRaisesRegex(ValueError, "Invalid signed metadata"):
                run.frozen_selection(output, context)
            changed = dict(original, winner="plain_mlp")
            write_json(output / "selection.json", state.signed(changed))
            with self.assertRaisesRegex(ValueError, "Selection disagrees"):
                run.frozen_selection(output, context)
            write_json(output / "selection.json", state.signed(dict(original, synthetic=True)))
            with self.assertRaisesRegex(ValueError, "synthetic"):
                run.frozen_selection(output, context)

    def test_incomplete_or_changed_valid_artifacts_refuse_test_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            context, _, _ = selection_fixture(output)
            file = output / "valid" / "baseline" / "verification.json"
            file.write_text(file.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "Changed or missing"):
                run.frozen_selection(output, context)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            context, _, _ = selection_fixture(output)
            (output / "valid" / "baseline" / "completed.json").unlink()
            with self.assertRaisesRegex(ValueError, "incomplete or changed"):
                run.frozen_selection(output, context)

    def test_rehashed_partial_grid_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            context, chosen, index = selection_fixture(output)
            del index["results"]["plain_mlp"]
            write_json(output / "evaluation_index_valid.json", index)
            chosen["valid_index_sha256"] = file_hash(output / "evaluation_index_valid.json")
            write_json(output / "selection.json", state.signed(chosen))
            with self.assertRaisesRegex(ValueError, "complete frozen"):
                run.frozen_selection(output, context)


class IdentityTests(unittest.TestCase):
    def test_local_time_identity_and_immutable_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run+name"
            fixed = datetime(2026, 11, 20, 15, 3, 7, tzinfo=state.TZ)
            with patch.object(state, "now", return_value=fixed):
                identity = state.identity(output, "valid", "plain_mlp", "signature")
            self.assertRegex(identity["key"], r"^[A-Za-z0-9_.-]+$")
            self.assertEqual(identity["started_at"], "2026-11-20T15:03:07+08:00")
            self.assertEqual(identity["run_id"], "20261120T150307+0800__" + identity["key"])
            before = snapshot(output)
            with patch.object(state, "now", side_effect=AssertionError("resume must preserve actual time")):
                self.assertEqual(state.identity(output, "valid", "plain_mlp", "signature"), identity)
            self.assertEqual(snapshot(output), before)
            with self.assertRaisesRegex(ValueError, "identity changed"):
                state.identity(output, "valid", "plain_mlp", "different_signature")


if __name__ == "__main__":
    unittest.main()
