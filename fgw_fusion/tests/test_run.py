"""Runner isolation and read-only reuse of a completed experiment."""
from contextlib import ExitStack, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fgw_fusion import register, run
from wbm_retrieval.common import file_hash, write_json


def snapshot(folder):
    return {str(p.relative_to(folder)): (file_hash(p), p.stat().st_mtime_ns)
            for p in folder.rglob("*") if p.is_file()}


class RunnerTests(unittest.TestCase):
    def test_dry_run_exposes_fixed_three_variants_without_loading_or_writing(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(run, "ROOT", Path(tmp)):
            with patch.object(run.oldrun, "load_inputs", side_effect=AssertionError("loaded")), \
                    redirect_stdout(StringIO()) as output:
                self.assertEqual(run.main(["--dry-run", "--workers", "2"]), 0)
            plan = json.loads(output.getvalue())
            self.assertEqual(plan["splits"], ["valid", "test"])
            self.assertEqual(set(plan["specs"]), {"FGW", "GW", "FW"})
            self.assertEqual(plan["specs"]["FGW"]["local_weight"], .5)
            self.assertEqual(plan["specs"]["FGW"]["global_weight"], .5)
            self.assertTrue(all(item["angular_gamma"] == 0 for item in plan["specs"].values()))
            self.assertEqual(plan["device"], "cpu")
            self.assertEqual(plan["source_device"], "cuda")
            self.assertFalse(plan["parameter_search"])
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_worker_bounds_rejected_before_sources(self):
        with patch.object(run.oldrun, "load_inputs", side_effect=AssertionError("loaded")):
            for workers in ("0", "-1", "33"):
                with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, "bounded W workers"):
                    run.main(["--dry-run", "--workers", workers])

    def test_outputs_reject_protected_nested_existing_and_unsigned_resume(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(run, "ROOT", Path(tmp)):
            root = Path(tmp)
            for rel in ("artifacts/fgw_fusion_x", "protocol/fgw_fusion_x", "artificial/old_name", "artificial"):
                with self.subTest(path=rel), self.assertRaises(ValueError):
                    run.choose_output(run.parser().parse_args(["--output-root", rel]))
            source = root / "artificial/fgw_fusion_original"
            source.mkdir(parents=True)
            write_json(source / "run_manifest.json", {})
            with self.assertRaisesRegex(ValueError, "nest inside"):
                run.choose_output(run.parser().parse_args(["--output-root", str(source / "fgw_fusion_child")]))
            with self.assertRaises(FileExistsError):
                run.choose_output(run.parser().parse_args(["--output-root", str(source)]))
            with self.assertRaisesRegex(ValueError, "signed manifest"):
                run.choose_output(run.parser().parse_args(["--resume-root", "artificial/fgw_fusion_missing"]))
            allowed = run.choose_output(run.parser().parse_args(["--resume-root", str(source)]))
            self.assertEqual(allowed, source)

    def completed_fixture(self, stack, root):
        source = root / "artificial/frozen_source"
        reference = root / "artificial/frozen_reference"
        output = root / "artificial/fgw_fusion_completed"
        source.mkdir(parents=True); reference.mkdir(parents=True)
        write_json(source / "run_manifest.json", run.signed({"context": {}}))
        write_json(reference / "run_manifest.json", run.signed({"context": {}}))
        for target, name, value in ((run, "ROOT", root), (run.sources, "SOURCE", source),
                (run.sources, "REFERENCE", reference), (run.sources, "PROTECTED", []),
                (run.sources, "WBBS_RUNS", {"valid": root / "wbbs_valid", "test": root / "wbbs_test"})):
            stack.enter_context(patch.object(target, name, value))
        with redirect_stdout(StringIO()) as capture:
            run.main(["--output-root", str(output), "--dry-run"])
        plan = json.loads(capture.getvalue()); plan.pop("output")
        provenance = {"reference": {"reports": {}}, "protocol_signature": "frozen"}
        w_provenance = {"protocol_signature": "frozen", "source": "frozen_W"}
        contexts = {s: (object(), object(), {}, {}, provenance) for s in ("valid", "test")}
        code = {"fgw_fusion/run.py": "frozen_code_hash"}
        context = {"plan": plan, "code": code,
            "source_manifest_sha256": file_hash(source / "run_manifest.json"),
            "reference_manifest_sha256": file_hash(reference / "run_manifest.json"),
            "inputs": {s: {**provenance, "wbbs": w_provenance} for s in contexts},
            "environment": {"numpy": run.np.__version__, "scipy": run.scipy.__version__,
                            "torch": run.torch.__version__, "python": run.sys.version}}
        output.mkdir()
        write_json(output / "run_manifest.json", run.signed({"version": run.VERSION, "context": context}))
        write_json(output / "source_protection_before.json", {})
        write_json(output / "status.json", {"status": "completed"})
        write_json(output / "completed.json", {"status": "completed", "frozen": True})
        write_json(output / "catalog_sources.json", {"already_prepared": True})
        # The real previous run already owns its advisory lock file.
        (output / ".ablation.lock").touch()
        stack.enter_context(patch.object(run.oldrun, "load_inputs", return_value=(object(), contexts)))
        stack.enter_context(patch.object(run.oldregister, "proposal", return_value={}))
        stack.enter_context(patch.object(run.sources, "load_wbbs", return_value=(object(), {}, w_provenance)))
        stack.enter_context(patch.object(run, "identity", return_value=code))
        stack.enter_context(patch.object(run, "build_cache", side_effect=AssertionError("completed cache recomputed")))
        stack.enter_context(patch.object(run.pipeline, "evaluate", side_effect=AssertionError("completed evaluation recomputed")))
        stack.enter_context(patch.object(run, "postprocess", side_effect=AssertionError("completed analysis recomputed")))
        stack.enter_context(patch.object(run.sources, "inventory", side_effect=AssertionError("completed protection rewritten")))
        return output

    def test_completed_resume_verifies_then_returns_without_mutating_any_file(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            output = self.completed_fixture(stack, Path(tmp))
            before = snapshot(output)
            proposal = stack.enter_context(patch.object(register, "proposal", return_value={"runs": []}))
            with redirect_stdout(StringIO()):
                self.assertEqual(run.main(["--resume-root", str(output)]), 0)
                self.assertEqual(run.main(["--resume-root", str(output)]), 0)
            self.assertEqual(snapshot(output), before)
            self.assertEqual(proposal.call_count, 2)
            self.assertEqual(proposal.call_args.args, (output,))

    def test_completed_resume_rejects_changed_context_before_reuse(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            output = self.completed_fixture(stack, Path(tmp))
            manifest = json.loads((output / "run_manifest.json").read_text())
            manifest.pop("signature"); manifest["context"]["code"] = {"changed": "source"}
            write_json(output / "run_manifest.json", run.signed(manifest))
            before = snapshot(output)
            proposal = stack.enter_context(patch.object(register, "proposal", side_effect=AssertionError("stale context accepted")))
            with redirect_stdout(StringIO()), self.assertRaisesRegex(ValueError, "Resume code/settings/source/environment changed"):
                run.main(["--resume-root", str(output)])
            proposal.assert_not_called()
            self.assertEqual(snapshot(output), before)

    def test_completed_resume_propagates_failed_evidence_without_rewriting_status(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            output = self.completed_fixture(stack, Path(tmp))
            before = snapshot(output)
            stack.enter_context(patch.object(register, "proposal", side_effect=ValueError("invalid completion evidence")))
            with redirect_stdout(StringIO()), self.assertRaisesRegex(ValueError, "invalid completion evidence"):
                run.main(["--resume-root", str(output)])
            self.assertEqual(snapshot(output), before)


if __name__ == "__main__":
    unittest.main()
