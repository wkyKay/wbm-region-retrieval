import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cnn_augmentation.augmentation import PaperViews, VARIANTS, augment_patch, _crop_parameters
from experiments.proposed._shared.ef_suite import RegionViews
from wbm_retrieval.training import loader


def patch_fixture():
    patch = np.zeros((3, 96, 96), dtype=np.uint8)
    patch[2, 3:94, 2:95] = 1
    patch[0, 10:80:3, 8:89] = 1
    patch[0, 40:57, 43:60] = 1
    patch[1, 40:57, 43:60] = 1
    return patch


class FixedRNG:
    def __init__(self, *, uniforms=(), integers=(), random_value=0.):
        self.uniforms = iter(uniforms)
        self.integer_values = iter(integers)
        self.random_value = random_value

    def uniform(self, *args):
        return next(self.uniforms)

    def integers(self, *args, **kwargs):
        return next(self.integer_values)

    def random(self, size=None):
        return self.random_value if size is None else np.full(size, self.random_value)


class AugmentationTests(unittest.TestCase):
    def test_all_variants_preserve_channel_semantics_and_input_bytes(self):
        original = patch_fixture()
        preserved = original.tobytes()
        for variant in VARIANTS:
            for seed in range(12):
                changed, info = augment_patch(original, variant, np.random.default_rng(seed))
                self.assertEqual(changed.dtype, np.float32)
                self.assertEqual(changed.shape, original.shape)
                self.assertTrue(changed.flags.c_contiguous)
                self.assertTrue(np.isin(changed, [0., 1.]).all())
                self.assertTrue(np.all(changed[1] <= changed[0]))
                self.assertTrue(np.all(changed[0] <= changed[2]))
                self.assertLessEqual(info["target_retention_fraction"], 1.)
                self.assertLessEqual(info["support_retention_fraction"], 1.)
                self.assertEqual(original.tobytes(), preserved)

    def test_cutout_gate_and_old_edge_behavior(self):
        patch = np.ones((3, 96, 96), dtype=np.uint8)
        skipped, info = augment_patch(patch, "cutout", FixedRNG(random_value=.9))
        np.testing.assert_array_equal(skipped, patch)
        self.assertEqual((info["applied"], info["no_op"]), (0, 1))
        # Four overlapping holes: top/left clip before adding side19; end clips to9.
        changed, info = augment_patch(patch, "cutout", FixedRNG(integers=[0, 0, 96, 96, 0, 0, 96, 96]))
        expected = patch.copy()
        expected[:, :19, :19] = 0
        expected[:, 87:, 87:] = 0
        np.testing.assert_array_equal(changed, expected)
        self.assertEqual(info["cutout_holes"], 4)
        self.assertEqual(info["changed_positions"], 19*19+9*9)
        self.assertEqual(info["target_retained_source_pixels"], 96*96-19*19-9*9)

    def test_crop_preserves_046_start_conversion_and_area_magnification_audit(self):
        rng = FixedRNG(uniforms=[.5, 0.], integers=[7, 9])
        self.assertEqual(_crop_parameters(rng), (6, 8, 68, 68, 1))
        patch = np.ones((3, 96, 96), dtype=np.uint8)
        changed, info = augment_patch(patch, "crop", FixedRNG(uniforms=[.5, 0.], integers=[7, 9]))
        self.assertEqual(info["target_after_pixels"], 96*96)
        self.assertEqual(info["target_retained_source_pixels"], 68*68)
        self.assertAlmostEqual(info["target_retention_fraction"], 68*68/(96*96))
        self.assertEqual(info["no_op"], 1)  # Appearance unchanged, source field still narrower.
        np.testing.assert_array_equal(changed, patch)

    def test_crop_has_only_geometric_retry_and_full_square_fallback(self):
        rng = FixedRNG(uniforms=[1., np.log(1.1)]*10)
        self.assertEqual(_crop_parameters(rng), (0, 0, 96, 96, 11))

    def test_continuous_rotation_uses_author_center_not_np_rot90(self):
        patch = patch_fixture()
        changed, info = augment_patch(patch, "rotation", FixedRNG(uniforms=[90.]))
        expected = np.zeros_like(patch)
        for row in range(96):
            for col in range(1, 96):
                expected[:, 96-col, row] = patch[:, row, col]
        np.testing.assert_array_equal(changed, expected)
        self.assertFalse(np.array_equal(changed, np.rot90(patch, axes=(-2, -1))))
        self.assertEqual(info["angle_degrees"], 90.)

    def test_shift_preserves_continuous_offsets_and_zero_padding(self):
        patch = patch_fixture()
        changed, info = augment_patch(patch, "shift", FixedRNG(uniforms=[.25, -.25]))
        expected = np.zeros_like(patch)
        expected[:, :72, 24:] = patch[:, 24:, :72]
        np.testing.assert_array_equal(changed, expected)
        # .51 pixel becomes a one-pixel nearest shift; pre-truncating would do nothing.
        shifted, _ = augment_patch(patch, "shift", FixedRNG(uniforms=[.51/96, 0.]))
        expected = np.zeros_like(patch)
        expected[:, :, 1:] = patch[:, :, :-1]
        np.testing.assert_array_equal(shifted, expected)
        self.assertEqual(info["shift_fraction_x"], .25)

    def test_noise_can_remove_and_add_defects_with_consistent_target(self):
        original = patch_fixture()
        replacements = (np.indices((96, 96)).sum(axis=0) % 2).astype(np.uint8)
        changed, info = augment_patch(original, "noise", FixedRNG(integers=[replacements]))
        expected = original.copy()
        expected[0, original[2] > 0] = replacements[original[2] > 0]
        expected[1] &= expected[0]
        np.testing.assert_array_equal(changed, expected)
        np.testing.assert_array_equal(changed[2], original[2])
        self.assertGreater(info["noise_added_defects"], 0)
        self.assertGreater(info["noise_removed_defects"], 0)
        self.assertEqual(info["noise_selected_pixels"], int(original[2].sum()))
        self.assertEqual(info["noise_changed_pixels"], int(np.count_nonzero(changed[0] != original[0])))

    def test_noise_selected_and_actual_change_rates_are_distinct(self):
        original = np.ones((3, 96, 96), dtype=np.uint8)
        selected, changed = 0, 0
        rng = np.random.default_rng(2026)
        for _ in range(128):
            _, info = augment_patch(original, "noise", rng)
            selected += info["noise_selected_pixels"]
            changed += info["noise_changed_pixels"]
        self.assertLess(abs(selected/(128*96*96)-.05), .001)
        self.assertLess(abs(changed/(128*96*96)-.025), .001)

    def test_empty_targets_reported_without_retry_or_fabrication(self):
        original = np.zeros((3, 96, 96), dtype=np.uint8)
        original[2] = 1
        original[:2, :4, :4] = 1
        changed, info = augment_patch(original, "shift", FixedRNG(uniforms=[-.25, -.25]))
        self.assertEqual(info["newly_empty_target"], 1)
        self.assertEqual(info["empty_target_after"], 1)
        self.assertEqual(info["target_retention_fraction"], 0.)
        self.assertEqual(changed[1].sum(), 0.)
        original[1] = 0
        _, info = augment_patch(original, "noise", np.random.default_rng(2))
        self.assertEqual(info["empty_target_before"], 1)
        self.assertEqual(info["target_retention_eligible"], 0)
        self.assertEqual(info["newly_empty_target"], 0)

    def test_bad_channel_inputs_and_variant_fail(self):
        patch = patch_fixture()
        patch[1, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "subset"):
            augment_patch(patch, "noise", np.random.default_rng(2))
        with self.assertRaisesRegex(ValueError, "Unknown"):
            augment_patch(patch_fixture(), "crop+rotation", np.random.default_rng(2))

    def test_dataset_uses_independent_rng_and_keeps_original_first_view(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            np.save(path/"train_F.npy", np.stack([patch_fixture()]*3))
            spec = {"preprocess": "fixed_window", "scale_range": None}
            baseline = RegionViews(path, "train", spec, 2026, epoch=7)
            before_np, before_torch, before_python = np.random.get_state(), torch.get_rng_state(), random.getstate()
            for variant in VARIANTS:
                data = PaperViews(path, "train", spec, 2026, variant, epoch=7)
                index, original, view, stats = data[1]
                _, reference, reference_view, _ = baseline[1]
                self.assertEqual(index, 1)
                self.assertTrue(torch.equal(original, reference))
                turns = int(np.random.default_rng(np.random.SeedSequence([2026, 7, 1])).integers(4))
                np.testing.assert_array_equal(reference_view.numpy(), np.rot90(reference.numpy(), turns, axes=(-2, -1)))
                second = PaperViews(path, "train", spec, 2026, variant, epoch=7)[1]
                self.assertTrue(torch.equal(view, second[2]))
                self.assertEqual(stats, second[3])
                self.assertIsNone(data.__getstate__()["patches"])
            after_np = np.random.get_state()
            self.assertEqual(before_np[0], after_np[0])
            np.testing.assert_array_equal(before_np[1], after_np[1])
            self.assertEqual(before_np[2:], after_np[2:])
            self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))
            self.assertEqual(before_python, random.getstate())

    def test_loader_worker_count_does_not_change_samples_or_shared_negative_rng(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            np.save(path/"train_F.npy", np.stack([patch_fixture()]*4))
            spec = {"preprocess": "fixed_window", "scale_range": None}
            dataset = PaperViews(path, "train", spec, 2026, "noise", epoch=23)
            cfg = {"batch_size": 2, "device": "cpu", "num_workers": 0}
            expected = list(loader(dataset, cfg, 2049, True))
            cfg["num_workers"] = 1
            actual = list(loader(dataset, cfg, 2049, True))
            for a, b in zip(expected, actual):
                for index in range(3):
                    self.assertTrue(torch.equal(a[index], b[index]))
                self.assertEqual(a[3].keys(), b[3].keys())
                for key in a[3]:
                    self.assertTrue(torch.equal(a[3][key], b[3][key]))


if __name__ == "__main__":
    unittest.main()
