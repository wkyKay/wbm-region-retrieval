import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cnn_refinement.augmentation import (ContextViews, context_dropout, original_window,
                                         prepare_context_windows)
from experiments.proposed._shared.ef_suite import RegionViews, fixed_window_patch
from wbm_retrieval.common import read_json
from wbm_retrieval.regions import resize_nearest


class ContextDropoutTests(unittest.TestCase):
    def fixture(self, path):
        raw = np.ones((52, 52), dtype=np.uint8)
        raw[:2] = 0
        raw[8:12, 7:13] = 2
        raw[20:24, 19:25] = 2
        mask = np.zeros((52, 52), dtype=np.uint8)
        mask[8:12, 7:13] = 1
        window = original_window(raw, mask)
        patch, _ = fixed_window_patch(raw, mask)
        np.save(path / "train_F.npy", patch[None])
        np.save(path / "windows.npy", window[None])
        return raw, mask, window, patch

    def test_exact_window_at_edges_and_centroid_rounding(self):
        raw = np.ones((52, 52), dtype=np.uint8)
        for row, col in ((0, 0), (50, 50), (25, 25), (0, 25)):
            target = np.zeros_like(raw)
            target[row:row+2, col:col+2] = 1
            current = raw.copy()
            current[target > 0] = 2
            actual = np.stack([resize_nearest(c, 96) for c in original_window(current, target)])
            np.testing.assert_array_equal(actual, fixed_window_patch(current, target)[0])

    def test_only_context_deleted_target_and_support_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, window, _ = self.fixture(Path(folder))
            for probability in (0., .5, 1.):
                changed, stats = context_dropout(window, np.random.default_rng(9), probability)
                np.testing.assert_array_equal(changed[1:], window[1:])
                self.assertTrue(np.all(changed[0] <= window[0]))
                self.assertTrue(np.all(changed[0] >= window[1]))
                self.assertEqual(stats["removed_pixels"], int((window[0]-changed[0]).sum()))
                if probability == 1:
                    np.testing.assert_array_equal(changed[0], window[1])
                if probability == 0:
                    np.testing.assert_array_equal(changed, window)
                    self.assertEqual(stats["no_op"], 1)
                self.assertEqual(stats["no_op"], int(stats["removed_pixels"] == 0))

    def test_bernoulli_fraction_and_repeatability(self):
        window = np.ones((3, 52, 52), dtype=np.uint8)
        window[1] = 0
        window[1, 25:27, 25:27] = 1
        a, stats = context_dropout(window, np.random.default_rng(2026))
        b, _ = context_dropout(window, np.random.default_rng(2026))
        np.testing.assert_array_equal(a, b)
        self.assertLess(abs(stats["removed_pixels"] / stats["context_pixels"] - .5), .04)

    def test_views_keep_first_view_and_original_rotation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            _, _, window, _ = self.fixture(path)
            spec = {"preprocess": "fixed_window", "scale_range": None}
            baseline = RegionViews(path, "train", spec, 2026)
            altered = ContextViews(path, "train", spec, 2026, path / "windows.npy")
            for epoch in range(4):
                baseline.epoch = altered.epoch = epoch
                _, original, view, stats = altered[0]
                _, expected_original, original_view, _ = baseline[0]
                self.assertTrue(torch.equal(original, expected_original))
                self.assertTrue(torch.equal(view[1:], original_view[1:]))
                self.assertTrue(torch.all(view[0] <= original_view[0]))
                turns = int(np.random.default_rng(np.random.SeedSequence([2026, epoch, 0])).integers(4))
                changed, _ = context_dropout(window, np.random.default_rng(
                    np.random.SeedSequence([2026, epoch, 0, 527])))
                resized = np.stack([resize_nearest(c, 96) for c in changed])
                expected = np.rot90(resized, turns, axes=(-2, -1))
                np.testing.assert_array_equal(view.numpy(), expected)
                self.assertEqual(stats["context_pixels"], int(((window[0]>0)&(window[1]==0)).sum()))
            self.assertIsNone(altered.__getstate__()["windows"])

    def test_full_cache_audit_and_recovery_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw, mask, _, patch = self.fixture(root)
            dataset = {"patches": patch[None], "prepared_marker": {"signature": "fixture"},
                       "map_ids": np.array([0]), "offsets": np.array([0, 1]),
                       "token_region_ids": np.array([1]), "region_masks": mask[None],
                       "area": np.array([int(mask.sum())])}
            inputs = {"datasets": {"train": dataset}, "maps": raw[None],
                      "source_manifest": {"signature": "fixture-source"}}
            destination = root / "prepared"
            path = prepare_context_windows(inputs, destination)
            self.assertEqual(read_json(destination / "train_windows52.json")["tokens_checked"], 1)
            self.assertEqual(prepare_context_windows(inputs, destination), path)
            (destination / "train_windows52.json").unlink()
            self.assertEqual(prepare_context_windows(inputs, destination), path)
            corrupted = np.load(path)
            corrupted[0, 0, 0, 0] ^= 1
            np.save(path, corrupted)
            with self.assertRaisesRegex(ValueError, "changed"):
                prepare_context_windows(inputs, destination)

    def test_invalid_channel_semantics_are_rejected(self):
        window = np.zeros((3, 52, 52), dtype=np.uint8)
        window[1, 2, 2] = 1
        with self.assertRaisesRegex(ValueError, "subset"):
            context_dropout(window, np.random.default_rng(0))

    def test_no_context_is_a_recorded_no_op(self):
        window = np.zeros((3, 52, 52), dtype=np.uint8)
        window[:, 20:23, 20:23] = 1
        changed, stats = context_dropout(window, np.random.default_rng(2026))
        np.testing.assert_array_equal(changed, window)
        self.assertEqual(stats["no_op"], 1)
        self.assertEqual(stats["no_context"], 1)


if __name__ == "__main__":
    unittest.main()
