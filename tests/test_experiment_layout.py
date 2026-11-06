"""Experiment routing, configuration resolution and legacy-source compatibility."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments._shared.entrypoints import ROOT, launch


class ExperimentLayoutTests(unittest.TestCase):
    def test_old_entrypoints_and_aliases_are_removed(self):
        obsolete = ["configs", "experiments/g", "wbm_retrieval/methods",
                    "wbm_retrieval/ablation.py", "wbm_retrieval/ablation_training.py",
                    "wbm_retrieval/ablation_ef.py", "wbm_retrieval/ablation_ef_training.py",
                    "experiments/supervised/suite.py", "experiments/supervised/training.py",
                    "run_experiment.py", "scripts/run_ablation.py", "scripts/run_ablation_ef.py",
                    "scripts/run_ablation_g.py", "scripts/run_supervised_multilabel.py",
                    "experiments/compatibility.json"]
        for name in obsolete:
            with self.subTest(path=name):
                self.assertFalse(os.path.lexists(ROOT / name))
        self.assertFalse(any(p.is_symlink() for p in (ROOT / "experiments").rglob("*")))

    def test_all_entries_resolve_their_configs_without_writes(self):
        catalog = json.loads((ROOT / "experiments/catalog.json").read_text())
        for name, entry in catalog.items():
            with self.subTest(name=name), contextlib.redirect_stdout(io.StringIO()) as out:
                launch(name, ["--dry-run"])
                result = json.loads(out.getvalue())
                self.assertTrue((ROOT / entry["config"]).is_file())
                if entry.get("variant"):
                    self.assertEqual(list(result["variants"]), [entry["variant"]])
                if entry["kind"] == "main_method":
                    plan = result["stages"]
                    if name == "handcrafted/a":
                        self.assertEqual(plan, ["prepare", "retrieve", "evaluate"])
                        self.assertEqual(result["methods"], ["handcrafted"])
                    elif name == "proposed/main":
                        self.assertNotIn("train-global", plan)
                        self.assertEqual(result["methods"], ["proposed"])
                    else:
                        self.assertNotIn("train-local", plan)
                        self.assertEqual(result["methods"], ["global_5pct", "global_10pct", "global_50pct"])

    def test_scoped_entries_cannot_run_another_variant(self):
        for flag in ("--variants", "--var", "--v"):
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                launch("proposed/a", [flag, "B", "--dry-run"])

    def test_main_method_executes_only_its_own_stages(self):
        with patch("experiments._shared.entrypoints.invoke") as invoke:
            launch("handcrafted/a", ["--stage", "all"])
        calls = [call.args[1] for call in invoke.call_args_list]
        stages = [args[args.index("--stage") + 1] for args in calls]
        self.assertEqual(stages, ["prepare", "retrieve", "evaluate"])
        for args in calls[1:]:
            self.assertEqual(args[args.index("--methods") + 1:], ["handcrafted"])
        with patch("experiments._shared.entrypoints.invoke") as invoke:
            launch("supervised/a", ["--stage", "test"])
        self.assertEqual([c.args[1][c.args[1].index("--stage")+1] for c in invoke.call_args_list],
                         ["retrieve", "evaluate"])

    def test_cli_from_another_working_directory_and_explicit_test(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(ROOT / "experiments/proposed/g/run.py"),
                 "--config", "experiments/proposed/g/config.json", "--stage", "test",
                 "--test-checkpoint", "best_nce", "--dry-run"], cwd=directory,
                check=True, text=True, capture_output=True)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["stage"], "test")
            self.assertEqual(list(plan["variants"]), ["G"])
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_handcrafted_count_runs_real_evaluation_without_training(self):
        from test_ablation_g import GTests
        from wbm_retrieval.common import write_json
        helper = GTests()
        helper.setUp()
        with tempfile.TemporaryDirectory() as directory:
            suite, config, _ = helper.fixture(directory)
            base = Path(directory) / "base.json"
            write_json(base, config)
            suite["base_config"] = str(base)
            suite["output_root"] = str(Path(directory) / "count-output")
            path = Path(directory) / "suite.json"
            write_json(path, suite)
            with contextlib.redirect_stdout(io.StringIO()), patch(
                    "experiments.proposed.g.training.train_variant", side_effect=AssertionError("must not train")):
                launch("handcrafted/b", ["--config", str(path), "--stage", "all"])
            output = Path(suite["output_root"])
            self.assertTrue((output / "handcrafted/valid/metrics.json").is_file())
            self.assertFalse((output / "seed_2026").exists())
            self.assertFalse(list(output.rglob("*.pt")))


if __name__ == "__main__":
    unittest.main()
