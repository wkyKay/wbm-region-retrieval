"""Mathematical checks for frozen-F rotation and radial inference ablations."""
import unittest

import numpy as np

from scripts.evaluate_f_inference_ablations import (
    pool_rotation_features,
    radial_distance,
    radial_histogram,
)


class RadialHistogramTests(unittest.TestCase):
    def setUp(self):
        rows, cols = np.indices((25, 25))
        self.radius2 = (rows - 12) ** 2 + (cols - 12) ** 2
        self.support = self.radius2 <= 100

    def test_center_and_outer_boundary_are_included(self):
        center = np.zeros_like(self.support)
        center[12, 12] = True
        edge = np.zeros_like(self.support)
        edge[12, 22] = True  # rho == 1 exactly: must remain in last bin.
        expected_center, expected_edge = np.zeros(16), np.zeros(16)
        expected_center[0], expected_edge[-1] = 1, 1
        np.testing.assert_array_equal(radial_histogram(center, self.support), expected_center)
        np.testing.assert_array_equal(radial_histogram(edge, self.support), expected_edge)
        np.testing.assert_allclose(radial_histogram(center | edge, self.support),
                                   .5 * (expected_center + expected_edge))

    def test_rotation_and_translation_invariance(self):
        rows, cols = np.indices(self.support.shape)
        mask = self.support & (rows < 13) & (cols > 8) & (self.radius2 > 9)
        expected = radial_histogram(mask, self.support)
        self.assertAlmostEqual(expected.sum(), 1)
        for k in range(4):
            np.testing.assert_allclose(radial_histogram(np.rot90(mask, k),
                                                        np.rot90(self.support, k)), expected)
        # Embed both arrays with an offset: center must follow the wafer, not canvas.
        shifted_mask, shifted_support = np.zeros((40, 43), bool), np.zeros((40, 43), bool)
        shifted_mask[3:28, 7:32], shifted_support[3:28, 7:32] = mask, self.support
        np.testing.assert_allclose(radial_histogram(shifted_mask, shifted_support), expected)

    def test_concentric_disk_and_ring_are_distinct(self):
        disk = self.radius2 <= 9
        ring = (self.radius2 >= 49) & (self.radius2 <= 64)
        np.testing.assert_allclose(np.argwhere(disk).mean(0), np.argwhere(ring).mean(0))
        disk_hist = radial_histogram(disk, self.support)
        ring_hist = radial_histogram(ring, self.support)
        self.assertGreater(radial_distance(disk_hist, ring_hist), .3)

    def test_degenerate_single_support_point(self):
        point = np.array([[True]])
        expected = np.zeros(16)
        expected[0] = 1
        np.testing.assert_array_equal(radial_histogram(point, point), expected)

    def test_invalid_geometry_is_rejected(self):
        mask = self.radius2 <= 9
        cases = [(np.zeros_like(mask), self.support), (mask, np.zeros_like(mask)),
                 (~self.support, self.support), (mask[:2], self.support)]
        for bad_mask, bad_support in cases:
            with self.subTest(shape=bad_mask.shape), self.assertRaises(ValueError):
                radial_histogram(bad_mask, bad_support)
        with self.assertRaises(ValueError):
            radial_histogram(mask, self.support, bins=1)


class RadialDistanceTests(unittest.TestCase):
    def test_wasserstein_one_hot_distance_and_broadcast(self):
        bins = 16
        left_indices, right_indices = np.array([0, 3, 15]), np.array([1, 8])
        left, right = np.eye(bins)[left_indices], np.eye(bins)[right_indices]
        expected = abs(left_indices[:, None] - right_indices[None, :]) / bins
        actual = radial_distance(left[:, None], right[None])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        np.testing.assert_allclose(actual, radial_distance(right[:, None], left[None]).T)
        np.testing.assert_array_equal(radial_distance(left, left), np.zeros(len(left)))
        self.assertTrue(np.all((actual >= 0) & (actual <= 1)))

    def test_probability_mass_transport(self):
        left, right = np.zeros(16), np.zeros(16)
        left[0], left[4], right[2] = .5, .5, 1
        # Half the mass moves two bins right, half two bins left.
        self.assertAlmostEqual(radial_distance(left, right), 2 / 16)

    def test_invalid_distributions_are_rejected(self):
        valid = np.ones(16) / 16
        for invalid in (np.zeros(16), -valid, np.full(16, np.nan), np.ones(8) / 8):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                radial_distance(invalid, valid)


class RotationPoolingTests(unittest.TestCase):
    def test_unit_features_are_averaged_before_final_normalization(self):
        # Raw averaging would favour x because of its larger norms.
        rotations = np.array([[[10., 0.]], [[0., 1.]], [[3., 0.]], [[0., 2.]]])
        pooled = pool_rotation_features(rotations)
        np.testing.assert_allclose(pooled, [[2 ** -.5, 2 ** -.5]], atol=1e-14)

    def test_cyclic_rotation_order_and_scale_invariance(self):
        rng = np.random.default_rng(2026)
        rotations = rng.uniform(.1, 1, size=(4, 7, 11))
        expected = pool_rotation_features(rotations)
        np.testing.assert_allclose(np.linalg.norm(expected, axis=1), np.ones(7), atol=1e-14)
        for k in range(4):
            np.testing.assert_allclose(pool_rotation_features(np.roll(rotations, k, axis=0)),
                                       expected, atol=1e-14)
        factors = rng.uniform(.01, 100, size=(4, 7, 1))
        np.testing.assert_allclose(pool_rotation_features(rotations * factors), expected, atol=1e-14)

    def test_invalid_rotation_arrays_are_rejected(self):
        for invalid in (np.ones((3, 2, 5)), np.ones((4, 5)), np.full((4, 2, 5), np.nan)):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                pool_rotation_features(invalid)


if __name__ == '__main__':
    unittest.main()
