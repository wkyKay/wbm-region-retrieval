"""Tiny synthetic tests only: no real-data training or retrieval."""
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from f_region_structure import learning


def fixture(seed=11, count=12):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(count, 512)).astype(np.float32)
    views = np.stack((base, base + rng.normal(0, .1, base.shape)), axis=1).astype(np.float32)
    views /= np.linalg.norm(views, axis=-1, keepdims=True)
    joint = np.zeros((count, 128), dtype=np.float64)
    joint[:, :8] = rng.random((count, 8))
    joint /= joint.sum(axis=1, keepdims=True)
    joint[0] = 0.
    area = np.exp(rng.uniform(-8, -2, count))
    ids = np.repeat(np.arange((count + 1) // 2), 2)[:count].astype(np.int64)
    return views, joint, area, ids


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = fixture()

    def tearDown(self):
        self.temp.cleanup()

    def train(self, name="plain_mlp", directory=None):
        return learning.fit(*self.data, directory or self.root / name, family=name,
                            epochs=2, batch_size=5, seed=2026)

    def test_two_families_complete_and_roundtrip(self):
        for family in learning.FAMILIES:
            with self.subTest(family=family):
                saved = [a.copy() for a in self.data]
                model = self.train(family)
                h, joint, area, ids = self.data
                actual = learning.encode(model, h[:, 0], joint, area, batch_size=4)
                loaded = learning.encode(model.checkpoint_path, h[:, 0], joint, area, batch_size=4)
                np.testing.assert_array_equal(actual, loaded)
                self.assertEqual(actual.shape, (len(h), 128))
                np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1., atol=2e-6)
                for a, original in zip(self.data, saved):
                    np.testing.assert_array_equal(a, original)
                manifest = model.training_metadata
                self.assertEqual(manifest["missing_structure_count"], 1)
                self.assertFalse(manifest["test_used"])
                self.assertFalse(manifest["settings"]["labels_used"])
                self.assertEqual(manifest["settings"]["checkpoint_rule"], "last_epoch")
                self.assertEqual(manifest["settings"]["area_normalization"]["fit_split"], "train")
                self.assertEqual(manifest["settings"]["epochs"], 2)
                self.assertTrue(manifest["started_at"].endswith("+08:00"))
                with (self.root / family / "loss.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual([r["epoch"] for r in rows], ["1", "2"])
                self.assertTrue(all(float(r["loss"]) > 0 for r in rows))

    def test_cpu_same_seed_reproducibility(self):
        first = self.train(directory=self.root / "first")
        second = self.train(directory=self.root / "second")
        for key, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
        self.assertEqual((self.root / "first/loss.csv").read_text(), (self.root / "second/loss.csv").read_text())

    def test_same_wafer_regions_are_not_positives_or_negatives(self):
        z = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
        ids = torch.tensor([11, 11, 12])
        loss, missing = learning.contrastive_loss(z, z, ids, temperature=1.)
        # First two rows contain their own positive and the other wafer only;
        # last row contains its positive plus the two other-wafer negatives.
        expected = (2 * np.log1p(np.exp(-1)) + np.log1p(2 * np.exp(-1))) / 3
        self.assertAlmostEqual(float(loss), expected, places=6)
        self.assertEqual(missing, 0)
        alone, missing = learning.contrastive_loss(z, z, torch.tensor([11, 11, 11]))
        self.assertEqual(float(alone), 0.)
        self.assertEqual(missing, 3)

    def test_geometry_normalization_frozen_at_inference(self):
        model = self.train()
        h, joint, area, _ = self.data
        mean, std = model.area_mean.clone(), model.area_std.clone()
        learning.encode(model, h[:, 0], joint, np.ones_like(area))
        torch.testing.assert_close(model.area_mean, mean, rtol=0, atol=0)
        torch.testing.assert_close(model.area_std, std, rtol=0, atol=0)
        self.assertAlmostEqual(float(mean), float(np.log(area).astype(np.float32).mean(dtype=np.float64)), places=6)

    def test_output_protection_and_train_only(self):
        for split in ("valid", "test"):
            with self.assertRaisesRegex(ValueError, "train"):
                learning.fit(*self.data, self.root / split, family="plain_mlp", split=split)
            self.assertFalse((self.root / split).exists())
        self.train()
        before = (self.root / "plain_mlp/final.pt").read_bytes()
        with self.assertRaises(FileExistsError):
            self.train()
        self.assertEqual(before, (self.root / "plain_mlp/final.pt").read_bytes())

    def test_incomplete_or_tampered_checkpoint_rejected(self):
        model = self.train()
        path = Path(model.checkpoint_path)
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "changed"):
            learning.load_model(path)

    def test_input_validation_before_writes(self):
        h, joint, area, ids = self.data
        bad_cases = [
            (h * 2, joint, area, ids),
            (h, joint * 2, area, ids),
            (h, joint, area * 0, ids),
            (h, joint, np.ones_like(area) * 1.1, ids),
            (h, joint, area, np.zeros_like(ids)),
            (h, joint, area, ids.astype(float)),
        ]
        for n, data in enumerate(bad_cases):
            with self.subTest(n=n), self.assertRaises(ValueError):
                learning.fit(*data, self.root / str(n), family="plain_mlp", epochs=2)
            self.assertFalse((self.root / str(n)).exists())
        for key, value in (("epochs", 0), ("batch_size", 1), ("seed", True)):
            with self.assertRaises(ValueError):
                learning.fit(*self.data, self.root / "invalid", family="plain_mlp", **{key: value})

    def test_empty_encoding_and_explicit_missing_bit(self):
        model = learning.FusionModel("plain_mlp")
        self.assertEqual(learning.encode(model, np.empty((0, 512)), np.empty((0, 128)), np.empty(0)).shape, (0, 128))
        root, log_area, missing = learning._geometry(self.data[1], self.data[2], len(self.data[1]))
        self.assertEqual(float(missing.sum()), 1.)
        self.assertEqual(float(root[0].sum()), 0.)
        np.testing.assert_allclose(np.linalg.norm(root[1:], axis=1), 1., atol=1e-6)


if __name__ == "__main__":
    unittest.main()
