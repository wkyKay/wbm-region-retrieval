import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from b1_ablation import run, register

class RunnerTests(unittest.TestCase):
    def test_dry_run_never_loads_sources_or_creates_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(run, "ROOT", root), patch.object(run, "load_inputs", side_effect=AssertionError("loaded")):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(run.main(["--dry-run"]), 0)
                self.assertIn('"I_2"', output.getvalue())
                self.assertEqual(list(root.iterdir()), [])

    def test_outputs_reject_sources_nested_and_resume_without_manifest(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(run, "ROOT", Path(directory)):
            root = Path(directory)
            for name in ("artifacts/b1_ablation_x", "artificial/old_run", "artificial"):
                with self.assertRaises(ValueError):
                    run.choose_output(run.parser().parse_args(["--output-root", name]))
            parent = root / "artificial/b1_ablation_original"
            parent.mkdir(parents=True)
            (parent / "run_manifest.json").write_text("{}")
            with self.assertRaises(ValueError):
                run.choose_output(run.parser().parse_args(["--output-root", str(parent / "b1_ablation_child")]))
            with self.assertRaises(ValueError):
                run.choose_output(run.parser().parse_args(["--resume-root", "artificial/b1_ablation_absent"]))

    def test_source_inventory_detects_content_and_mtime(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(run, "ROOT", Path(directory)):
            root = Path(directory)
            source = root / "source"; source.mkdir()
            code = root / "b1_structure_fusion"; code.mkdir()
            (code / "old.py").write_text("original")
            data = source / "rankings.csv"; data.write_text("old")
            before = run.inventory(source)
            data.write_text("new")
            self.assertNotEqual(before, run.inventory(source))
            self.assertEqual((code / "old.py").read_text(), "original")

    def test_registration_rejects_incomplete_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); out = root / "artificial/b1_ablation_incomplete"
            out.mkdir(parents=True); (out / "status.json").write_text('{"status":"running"}')
            with self.assertRaises(ValueError): register.proposal(out, root=root)
            self.assertEqual([p.name for p in out.iterdir()], ["status.json"])

if __name__ == "__main__": unittest.main()
