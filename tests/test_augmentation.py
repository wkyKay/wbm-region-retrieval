"""Opening must remove small protrusions without changing proposals or global views."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wbm_retrieval.augmentation import region_view, validate_opening
from wbm_retrieval.common import ROOT, read_json
from wbm_retrieval.regions import augment


class OpeningTests(unittest.TestCase):
    def setUp(self):
        self.settings = {"enabled": True, "probability": 1., "kernel": "cross3", "iterations": 1,
                         "min_retained_fraction": .8}
        self.base = {"rotation": False, "max_shift": 0, "dropout": 0., "noise": 0.}

    def test_removes_spur_preserves_original_and_retention(self):
        image = np.zeros((1, 96, 96), dtype=np.uint8)
        image[0, 20:70, 20:70] = 1
        image[0, 45, 70:85] = 1
        before = image.copy()
        view, info = region_view(image, np.random.default_rng(42), self.base, self.settings)
        np.testing.assert_array_equal(image, before)
        self.assertEqual((info["attempted"], info["accepted"], info["changed"], info["fallback"]), (1, 1, 1, 0))
        self.assertLess(view.sum(), image.sum())
        self.assertGreaterEqual(view.sum() / image.sum(), .8)
        self.assertEqual(view[0, 45, 84], 0)
        self.assertTrue(view.flags.c_contiguous)

    def test_empty_or_excessive_removal_reverts_before_rotation(self):
        thin = np.zeros((1, 96, 96), dtype=np.uint8)
        thin[0, 45, 10:90] = 1
        fragile = thin.copy()
        fragile[0, 40:50, 10:20] = 1
        for image in (thin, fragile):
            view, info = region_view(image, np.random.default_rng(42), self.base, self.settings)
            self.assertEqual(info["fallback"], 1)
            self.assertLess(info["candidate_retention"], .8)
            np.testing.assert_array_equal(view, image)

    def test_disabled_opening_preserves_old_view_and_rng(self):
        image = np.eye(96, dtype=np.uint8)[None]
        for settings in (None, {**self.settings, "enabled": False}):
            left, right = np.random.default_rng(9), np.random.default_rng(9)
            actual, info = region_view(image, left, {**self.base, "rotation": True}, settings)
            expected = augment(image, right, **{**self.base, "rotation": True})
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(left.random(), right.random())
            self.assertEqual(info["attempted"], 0)

    def test_local_scope_and_fixed_dataset_view(self):
        from wbm_retrieval.methods.proposed import PatchDataset
        config = read_json(ROOT / "configs/experiment_a.json")
        self.assertEqual(config["proposal"]["top_k"], 4)
        self.assertNotIn("opening", config["augmentation"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "patches.npy"
            images = np.zeros((4, 1, 96, 96), dtype=np.uint8)
            images[:, :, 20:70, 20:70] = 1
            np.save(path, images)
            data = PatchDataset(path, config["augmentation"], 2026, opening=config["local"]["opening"])
            np.testing.assert_array_equal(data[0][2], data[0][2])
            self.assertEqual(data[0][3], data[0][3])
            np.testing.assert_array_equal(data[0][1], images[0])
        bad = copy.deepcopy(self.settings)
        bad["iterations"] = 2
        with self.assertRaises(ValueError):
            validate_opening(bad)


if __name__ == "__main__":
    unittest.main()
