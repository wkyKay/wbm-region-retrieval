"""Scientific E/F invariants, real optimizer steps, CLI workers and exact resume."""

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from test_core import fixture_config
from wbm_retrieval import ablation as legacy
from wbm_retrieval.ablation_ef import (RegionViews, build_model, evaluate, fixed_window_patch,
    load_suite, prepare_suite, summarize, validate_reference)
from wbm_retrieval.ablation_ef_training import evaluate_variant, train_variant
from wbm_retrieval.common import ROOT, csv_rows, file_hash, read_json, seed_everything, write_json
from wbm_retrieval.evaluation import evaluate_ranking
from wbm_retrieval.training import load_checkpoint, parameter_hash


class EFTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.suite, self.config = load_suite(ROOT / "configs/ablation_ef.json", device="cpu", workers=0)

    def fixture(self, directory, epochs=1):
        config, labels = fixture_config(directory)
        # Real 52x52 input geometry for F, with the same frozen synthetic IDs.
        with np.load(config["data_file"]) as archive:
            maps = np.ones((24, 52, 52), dtype=np.uint8)
            maps[:, :16, :16] = archive["arr_0"]
        np.savez(config["data_file"], maps, labels)
        config.update(input_size=96, batch_size=2, inference_batch_size=2)
        config["proposal"].update(min_area=5, top_k=4, padding=4)
        config["local"].update(epochs=epochs, num_negatives=8)
        config["local"]["opening"]["enabled"] = False
        config["local"]["early_stopping"] = {"enabled": False}
        suite = copy.deepcopy(self.suite)
        suite["valid_pool"].update(queries=4, candidates=5)
        suite["evaluate_every"] = 1
        Path(config["output_dir"]).mkdir()
        return suite, config, labels

    def test_configuration_and_protected_outputs(self):
        with self.assertRaises(ValueError):
            load_suite(ROOT / "configs/ablation_ef.json", output_root="artifacts/ablation_abcd_20261019/new")
        with tempfile.TemporaryDirectory() as directory:
            changed = copy.deepcopy(self.suite)
            changed["variants"]["E"]["max_shift"] = 3
            path = Path(directory) / "bad.json"
            write_json(path, changed)
            with self.assertRaises(ValueError):
                load_suite(path)
        result = subprocess.run([sys.executable, str(ROOT / "scripts/run_ablation_ef.py"),
                                 "--config", str(ROOT / "configs/ablation_ef.json"), "--dry-run"],
                                capture_output=True, text=True, check=True)
        self.assertEqual(set(json.loads(result.stdout)["variants"]), {"E", "F"})
        result = subprocess.run([sys.executable, str(ROOT / "scripts/run_ablation_ef.py"), "--stage", "test"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("explicit --variants", result.stderr)

    def test_f_matches_legacy_window_semantics_and_audits_clipping(self):
        raw = np.ones((52, 52), dtype=np.uint8)
        mask = np.zeros_like(raw)
        mask[20:30, 1:9] = 1
        mask[25, 9:52] = 1  # Asymmetric connected region; centroid window clips the tail.
        raw[mask > 0] = 2
        raw[35, 12] = 2  # Context defect outside the target.
        raw[23, 11] = 3  # Not defect or valid.
        actual, audit = fixed_window_patch(raw, mask)
        self.assertTrue(audit["clipped"])
        self.assertLess(audit["retained_fraction"], 1)
        expected = np.zeros((3, 96, 96), dtype=np.uint8)
        # Independent output-pixel oracle, following the old 52-to-96 mapping.
        for row in range(96):
            for col in range(96):
                r = row * 52 // 96 + audit["window_row"]
                c = col * 52 // 96 + audit["window_col"]
                if 0 <= r < 52 and 0 <= c < 52:
                    expected[:, row, col] = [raw[r, c] == 2, mask[r, c] > 0, raw[r, c] in (1, 2)]
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(np.all(actual[1] <= actual[0]))
        self.assertTrue(np.all(actual[0] <= actual[2]))
        self.assertGreater(actual[0].sum(), actual[1].sum())
        self.assertTrue((actual[:, :, 0] == 0).all())
        center = np.zeros_like(raw)
        center[24:28, 24:28] = 1
        self.assertEqual(fixed_window_patch(raw, center)[1]["retained_fraction"], 1)

    def test_initialization_params_and_feature_shapes(self):
        states, rng_states = {}, {}
        for name in "DEF":
            seed_everything(2026)
            with patch("torch.cuda.manual_seed_all") as cuda_seed:
                model = build_model(self.config, self.suite["variants"][name]).eval()
                cuda_seed.assert_not_called()
            rng_states[name] = torch.get_rng_state().clone()
            states[name] = {k: v.clone() for k, v in model.state_dict().items()}
            self.assertEqual(sum(p.numel() for p in model.parameters()), 11242176 if name == "F" else 11235904)
            channels = 3 if name == "F" else 1
            with torch.no_grad():
                spatial = model["encoder"].stages(model["encoder"].stem(torch.ones(2, channels, 96, 96)))
            self.assertEqual(tuple(spatial.shape), (2, 512, 6, 6))
        for name in "EF":
            torch.testing.assert_close(rng_states["D"], rng_states[name], rtol=0, atol=0)
        for key in states["D"]:
            torch.testing.assert_close(states["D"][key], states["E"][key], rtol=0, atol=0)
            if key != "encoder.stem.0.weight":
                torch.testing.assert_close(states["D"][key], states["F"][key], rtol=0, atol=0)
        old_suite, old_config = legacy.load_suite(ROOT / "configs/ablation_abcd.json")
        seed_everything(2026)
        old = legacy.build_model(old_config, old_suite["variants"]["D"])
        for key in states["D"]:
            torch.testing.assert_close(states["D"][key], old.state_dict()[key], rtol=0, atol=0)

    def test_views_cache_and_reference_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            suite, config, _ = self.fixture(directory)
            root = Path(config["output_dir"])
            signature = prepare_suite(suite, config)
            self.assertEqual(signature, prepare_suite(suite, config))
            old_suite, _ = legacy.load_suite(ROOT / "configs/ablation_abcd.json")
            old_suite["valid_pool"] = suite["valid_pool"]
            old_suite["evaluate_every"] = 1
            old_config = {**config, "output_dir": str(Path(directory) / "old")}
            Path(old_config["output_dir"]).mkdir()
            legacy.prepare_suite(old_suite, old_config)
            self.assertEqual(read_json(root / "valid_pool.json"), read_json(Path(old_config["output_dir"]) / "valid_pool.json"))
            for name, old_name in (("D", "D"), ("E", "C")):
                new = RegionViews(root / "cache", "train", suite["variants"][name], 2026, epoch=2)
                old = legacy.RegionViews(Path(old_config["output_dir"]) / "cache", "train", old_suite["variants"][old_name], 2026, epoch=2)
                for i in range(len(new)):
                    torch.testing.assert_close(new[i][1], old[i][1], rtol=0, atol=0)
                    torch.testing.assert_close(new[i][2], old[i][2], rtol=0, atol=0)
            f = RegionViews(root / "cache", "train", suite["variants"]["F"], 2026)
            _, original, view, _ = f[0]
            self.assertEqual(tuple(original.shape), (3, 96, 96))
            self.assertTrue(any(torch.equal(view, torch.rot90(original, k, (-2, -1))) for k in range(4)))
            self.assertIsNone(f.__getstate__()["patches"])
            audit = read_json(root / "cache/F_window_audit.json")
            self.assertEqual(audit["train"]["tokens"], len(f))
            self.assertEqual(len(csv_rows(root / "cache/train_F_window_audit.csv")), len(f))
            # Invalid historical settings fail before loading historical weights/reports.
            manifest = Path(old_config["output_dir"]) / "suite_manifest.json"
            bad = read_json(manifest)
            bad["settings"]["batch_size"] = 999
            write_json(manifest, bad)
            with self.assertRaisesRegex(ValueError, "settings differ"):
                validate_reference(config, suite, old_config["output_dir"])
            with (root / "cache/valid_B.npy").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "modified"):
                prepare_suite(suite, config)

    def test_real_training_e_f_evaluation_and_finished_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            suite, config, labels = self.fixture(directory)
            root = Path(config["output_dir"])
            signature = prepare_suite(suite, config)
            evaluate(config, suite, "valid", root / "handcrafted/valid", rankings=True)
            for name in "EF":
                self.assertTrue(train_variant(config, suite, signature, name))
                output = root / f"seed_{config['seed']}" / name
                summary = read_json(output / "training_summary.json")
                self.assertNotEqual(summary["initial_parameter_sha256"], summary["final_parameter_sha256"])
                self.assertGreater(summary["optimizer_steps"], 0)
                checkpoint_hash = file_hash(output / "last.pt")
                (output / "training_summary.json").unlink()
                self.assertTrue(train_variant(config, suite, signature, name, resume=True))
                self.assertEqual(file_hash(output / "last.pt"), checkpoint_hash)
                with self.assertRaises(FileExistsError):
                    train_variant(config, suite, signature, name)
                evaluate_variant(config, suite, signature, name)
                self.assertFalse((output / "test").exists())
                pool = read_json(root / "valid_pool.json")
                protocol = SimpleNamespace(queries=pool["queries"], candidates={int(q): c for q, c in pool["candidates"].items()},
                                           labels={i: value for i, value in enumerate(labels)})
                target = output / "valid/best_retrieval"
                reference_folder = root / f"reference_evaluator_{name}"
                reference_folder.mkdir()
                shutil.copyfile(target / "rankings.csv", reference_folder / "rankings.csv")
                reference = evaluate_ranking(reference_folder / "rankings.csv", protocol, config["evaluation"], config["seed"])
                self.assertEqual(reference["metrics"], read_json(target / "metrics.json")["metrics"])
                evaluate_variant(config, suite, signature, name, "test", "best_nce")
                self.assertEqual(file_hash(output / "last.pt"), checkpoint_hash)
                self.assertEqual(len(csv_rows(output / "test/best_nce/rankings.csv")), 6)
            summarize(config, suite)
            pairs = read_json(root / f"seed_{config['seed']}/paired_valid.json")["comparisons"]
            self.assertTrue(any(r["left"] == "E" and r["right"] == "F" for r in pairs))
            self.assertTrue(any(r["right"] == "handcrafted" for r in pairs))
            summarize(config, suite, "test")
            pairs = read_json(root / f"seed_{config['seed']}/paired_test.json")["comparisons"]
            self.assertTrue(all(r["checkpoint"] == "best_nce" for r in pairs))

    def test_interrupted_resume_exact_for_both_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            suite, config, _ = self.fixture(directory, epochs=2)
            signature = prepare_suite(suite, config)
            full_config = {**config, "output_dir": str(Path(directory) / "full")}
            Path(full_config["output_dir"]).mkdir()
            full_signature = prepare_suite(suite, full_config)
            self.assertEqual(signature, full_signature)
            for name in "EF":
                self.assertFalse(train_variant(config, suite, signature, name, stop_after=1))
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    evaluate_variant(config, suite, signature, name)
                train_variant(config, suite, signature, name, resume=True)
                train_variant(full_config, suite, full_signature, name)
                a = load_checkpoint(Path(config["output_dir"]) / f"seed_{config['seed']}/{name}/last.pt")
                b = load_checkpoint(Path(full_config["output_dir"]) / f"seed_{config['seed']}/{name}/last.pt")
                self.assertEqual(a["scheduler"], b["scheduler"])
                self.assertEqual(a["optimizer_steps"], b["optimizer_steps"])
                torch.testing.assert_close(a["memory"], b["memory"], rtol=0, atol=0)
                for key in a["model"]:
                    torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)

    def test_cli_with_spawn_workers_and_read_only_legacy_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            suite, config, _ = self.fixture(directory)
            old_suite, _ = legacy.load_suite(ROOT / "configs/ablation_abcd.json")
            old_suite.update(valid_pool=suite["valid_pool"], evaluate_every=1)
            old_config = {**config, "output_dir": str(Path(directory) / "old")}
            Path(old_config["output_dir"]).mkdir()
            old_signature = legacy.prepare_suite(old_suite, old_config)
            from wbm_retrieval.ablation_training import train_variant as old_train, evaluate_variant as old_evaluate
            old_train(old_config, old_suite, old_signature, "D")
            old_evaluate(old_config, old_suite, old_signature, "D")
            old_root = Path(old_config["output_dir"])
            before = {str(p): file_hash(p) for p in old_root.rglob("*") if p.is_file()}
            base_path, suite_path = Path(directory) / "base.json", Path(directory) / "suite.json"
            write_json(base_path, config)
            suite.update(base_config=str(base_path), output_root=str(Path(directory) / "cli"))
            suite["overrides"]["local"]["epochs"] = 1
            write_json(suite_path, suite)
            command = [sys.executable, str(ROOT / "scripts/run_ablation_ef.py"), "--config", str(suite_path),
                       "--device", "cpu", "--num-workers", "2", "--reference-abcd-root", str(old_root)]
            env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
            result = subprocess.run(command, capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            root = Path(suite["output_root"])
            pairs = read_json(root / f"seed_{config['seed']}/paired_valid.json")["comparisons"]
            self.assertEqual({r["left"] for r in pairs if r["right"] == "D"}, {"E", "F"})
            for path, sha in before.items():
                self.assertEqual(file_hash(path), sha)
            self.assertEqual(before.keys(), {str(p) for p in old_root.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main()
