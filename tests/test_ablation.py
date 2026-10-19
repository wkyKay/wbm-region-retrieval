"""A/B/C/D scientific invariants plus actual CPU training / resume smoke tests."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from test_core import fixture_config
from wbm_retrieval.ablation import (RegionViews, build_model, evaluate, exclusive, load_suite,
                                   prepare_suite, resize_then_pad, summarize, valid_pool)
from wbm_retrieval.ablation_training import evaluate_variant, train_variant
from wbm_retrieval.common import ROOT, csv_rows, file_hash, read_json, seed_everything
from wbm_retrieval.evaluation import evaluate_ranking
from wbm_retrieval.matching import map_similarity
from wbm_retrieval.regions import token_patch
from wbm_retrieval.training import load_checkpoint, parameter_hash


class AblationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.suite, self.config = load_suite(ROOT / "configs/ablation_abcd.json", device="cpu", workers=0)

    def fixture(self, directory):
        config, labels = fixture_config(directory)
        config.update(input_size=96, batch_size=2, inference_batch_size=2)
        config["proposal"].update(min_area=5, top_k=4, padding=4)
        config["local"].update(epochs=1, num_negatives=8, opening={"enabled": False})
        config["local"]["early_stopping"] = {"enabled": False}
        suite = copy.deepcopy(self.suite)
        suite["valid_pool"].update(queries=4, candidates=5)
        suite["evaluate_every"] = 1
        return suite, config, labels

    def test_resize_pad_and_original_baseline(self):
        mask = np.zeros((52, 52), np.uint8)
        mask[4:8, 7:17] = 1
        actual = resize_then_pad(mask)
        self.assertEqual(actual.shape, (96, 96))
        self.assertTrue(np.isin(actual, [0, 1]).all())
        points = np.argwhere(actual)
        self.assertEqual(points[:, 1].min(), 4)
        self.assertEqual(points[:, 1].max(), 91)
        np.testing.assert_array_equal(actual, np.pad(token_patch(mask, 88, 0), 4))
        np.testing.assert_array_equal(actual, resize_then_pad(np.roll(mask, (8, 9), (0, 1))))
        self.assertFalse(np.array_equal(actual, token_patch(mask, 96, 4)))
        for side in range(80, 89):
            centered = resize_then_pad(mask, 96, side)
            shifted = resize_then_pad(mask, 96, side, (2, -2))
            self.assertEqual(centered.sum(), shifted.sum())
        with self.assertRaises(ValueError):
            resize_then_pad(mask, 96, 88, (10, 0))

    def test_identical_initialization_and_stride_feature_maps(self):
        hashes = []
        for name in "ABCD":
            seed_everything(2026)
            model = build_model(self.config, self.suite["variants"][name]).eval()
            hashes.append(parameter_hash(model))
            with torch.no_grad():
                feature_map = model["encoder"].stages(model["encoder"].stem(torch.ones(1, 1, 96, 96)))
            side = 6 if name == "D" else 3
            self.assertEqual(feature_map.shape, (1, 512, side, side))
            self.assertEqual(model["encoder"].stem[3].stride, 2)
        self.assertEqual(len(set(hashes)), 1)

    def test_pool_label_blind_fixed_and_self_excluded(self):
        settings = dict(seed=2026, queries=4, candidates=5)
        ids = list(range(8, 16))
        pool = valid_pool(ids, settings)
        self.assertEqual(pool, valid_pool(ids[::-1], settings))
        for q, candidates in pool["candidates"].items():
            self.assertEqual(len(set(candidates)), 5)
            self.assertNotIn(int(q), candidates)
            self.assertTrue(set(candidates) <= set(ids))
        with self.assertRaises(ValueError):
            valid_pool(ids, dict(seed=2026, queries=9, candidates=5))

    def test_cache_views_scoring_and_tamper_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            suite, config, labels = self.fixture(directory)
            root = Path(config["output_dir"])
            with exclusive(root):
                signature = prepare_suite(suite, config)
                self.assertEqual(signature, prepare_suite(suite, config))
                with self.assertRaises(RuntimeError):
                    with exclusive(root):
                        pass
                for name in "ABCD":
                    data = RegionViews(root / "cache", "valid", suite["variants"][name], 2026)
                    _, original, view, _ = data[0]
                    self.assertEqual(tuple(original.shape), (1, 96, 96))
                    torch.testing.assert_close(view, data[0][2])
                    self.assertTrue(np.isin(view.numpy(), [0, 1]).all())
                    mask = np.load(root / "cache/valid_masks.npy")[0]
                    expected = token_patch(mask, 96, 4) if name == "A" else resize_then_pad(mask)
                    np.testing.assert_array_equal(original[0], expected)
                    # Dataset access does not mutate the standard input or the raw mask.
                    self.assertEqual(data.__getstate__()["patches"], None)
                result = evaluate(config, suite, "valid", root / "hc", rankings=True)
                pool = read_json(root / "valid_pool.json")
                protocol = SimpleNamespace(queries=pool["queries"], candidates={int(q): c for q, c in pool["candidates"].items()},
                                           labels={i: label for i, label in enumerate(labels)})
                reference = evaluate_ranking(root / "hc/rankings.csv", protocol, config["evaluation"], 2026)
                self.assertEqual(reference["metrics"], result["metrics"])
                with np.load(root / "cache/valid_regions.npz") as archive:
                    meta = dict(archive)
                grouped = {}
                for i, a, b in zip(meta["map_ids"], meta["offsets"][:-1], meta["offsets"][1:]):
                    grouped[int(i)] = dict(area=meta["area"][a:b], embedding=meta["descriptor"][a:b],
                                           position=np.zeros((b-a, 2)), scale=np.ones(b-a))
                match = {**config["matching"], "representation": "zernike", "zernike_distance_scale": 1.}
                for row in csv_rows(root / "hc/rankings.csv"):
                    expected = map_similarity(grouped[int(row["query_id"])], grouped[int(row["candidate_id"])], match)
                    self.assertAlmostEqual(expected, float(row["similarity_score"]), places=12)
                changed = copy.deepcopy(config)
                changed["batch_size"] = 4
                with self.assertRaises(ValueError):
                    prepare_suite(suite, changed)

    def test_real_training_all_four_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            suite, config, _ = self.fixture(directory)
            root = Path(config["output_dir"])
            with exclusive(root):
                signature = prepare_suite(suite, config)
                pool_hash = file_hash(root / "valid_pool.json")
                initial = []
                for name in "ABCD":
                    self.assertTrue(train_variant(config, suite, signature, name))
                    output = root / f"seed_{config['seed']}" / name
                    checkpoint_hash = file_hash(output / "last.pt")
                    # Simulate interruption between committing the checkpoint and its JSON report.
                    (output / "training_summary.json").unlink()
                    self.assertTrue(train_variant(config, suite, signature, name, resume=True))
                    self.assertEqual(checkpoint_hash, file_hash(output / "last.pt"))
                    with self.assertRaises(FileExistsError):
                        train_variant(config, suite, signature, name)
                    summary = read_json(output / "training_summary.json")
                    initial.append(summary["initial_parameter_sha256"])
                    self.assertNotEqual(initial[-1], summary["final_parameter_sha256"])
                    self.assertGreater(summary["optimizer_steps"], 0)
                    evaluate_variant(config, suite, signature, name)
                self.assertEqual(len(set(initial)), 1)
                self.assertEqual(pool_hash, file_hash(root / "valid_pool.json"))
                self.assertFalse(any((root / f"seed_{config['seed']}" / name / "test").exists() for name in "ABCD"))
                summarize(config, suite)
                rows = csv_rows(root / f"seed_{config['seed']}" / "comparison_valid.csv")
                self.assertEqual(len(rows), 4 * 3 * 3)
                # Explicit test does not update weights and evaluates the original frozen pool.
                output = root / f"seed_{config['seed']}" / "B"
                before = file_hash(output / "last.pt")
                evaluate_variant(config, suite, signature, "B", "test", "best_retrieval")
                self.assertEqual(before, file_hash(output / "last.pt"))
                self.assertEqual(len(csv_rows(output / "test/best_retrieval/rankings.csv")), 6)

    def test_interrupted_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            suite, config, _ = self.fixture(directory)
            config["local"]["epochs"] = 2
            root = Path(config["output_dir"])
            with exclusive(root):
                signature = prepare_suite(suite, config)
                self.assertFalse(train_variant(config, suite, signature, "B", stop_after=1))
                self.assertTrue(train_variant(config, suite, signature, "B", resume=True))
                resumed = load_checkpoint(root / f"seed_{config['seed']}" / "B/last.pt")
            other = copy.deepcopy(config)
            other["output_dir"] = str(Path(directory) / "uninterrupted")
            with exclusive(other["output_dir"]):
                other_signature = prepare_suite(suite, other)
                self.assertEqual(signature, other_signature)
                self.assertTrue(train_variant(other, suite, other_signature, "B"))
                full = load_checkpoint(Path(other["output_dir"]) / f"seed_{config['seed']}" / "B/last.pt")
            self.assertEqual(resumed["optimizer_steps"], full["optimizer_steps"])
            self.assertEqual(resumed["scheduler"], full["scheduler"])
            torch.testing.assert_close(resumed["memory"], full["memory"], rtol=0, atol=0)
            for key in full["model"]:
                torch.testing.assert_close(resumed["model"][key], full["model"][key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
