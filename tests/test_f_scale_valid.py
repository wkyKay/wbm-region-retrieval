"""Numerical invariants for the isolated F scale comparison."""
import unittest

import numpy as np

from scripts.evaluate_f_scale_valid import apply_scale, scale_descriptor, scale_distance


class ScaleGeometryTests(unittest.TestCase):
    def test_population_covariance_and_inside_root_regularizer(self):
        support = np.ones((5, 7), bool)
        mask = np.zeros_like(support)
        mask[1:3, 2:5] = True
        descriptor, eigenvalues = scale_descriptor(mask, support)
        np.testing.assert_allclose(eigenvalues, [2/3, 1/4], atol=1e-15)
        np.testing.assert_allclose(descriptor, [np.sqrt(6/35), np.sqrt((2/3+1)/35), np.sqrt(1.25/35)])

    def test_rotation_translation_and_support_normalization(self):
        mask = np.zeros((11, 11), bool)
        mask[2:5, 3:9] = True
        support = np.ones_like(mask)
        expected, _ = scale_descriptor(mask, support)
        for k in range(4):
            actual, _ = scale_descriptor(np.rot90(mask, k), np.rot90(support, k))
            np.testing.assert_allclose(actual, expected, atol=1e-15)
        moved = np.roll(mask, (3, 1), axis=(0, 1))
        np.testing.assert_allclose(scale_descriptor(moved, support)[0], expected)
        # Same full pixels, four times wafer support: all relative lengths halve.
        large_mask, large_support = np.zeros((22, 22), bool), np.ones((22, 22), bool)
        large_mask[:11, :11] = mask
        np.testing.assert_allclose(scale_descriptor(large_mask, large_support)[0], expected/2)

    def test_degenerate_axis_is_finite_and_empty_is_rejected(self):
        support = np.ones((7, 7), bool)
        mask = np.zeros_like(support)
        mask[3, 1:6] = True
        descriptor, eigenvalues = scale_descriptor(mask, support)
        self.assertEqual(eigenvalues[1], 0)
        self.assertAlmostEqual(descriptor[2], 1/7)
        self.assertTrue(np.isfinite(np.log(descriptor)).all())
        with self.assertRaises(ValueError):
            scale_descriptor(np.zeros_like(mask), support)
        with self.assertRaises(ValueError):
            scale_descriptor(mask, ~mask)

    def test_nominal_linear_scale_sensitivity_and_smoothing_limit(self):
        base = np.array([.2, .3, .1])
        for factor in (.5, 1., 1.25, 2.):
            for kind in ('area', 'axes'):
                self.assertAlmostEqual(scale_distance(base, factor*base, kind), abs(np.log(factor)))
        # Original-pixel isotropic rescaling: retaining +1 dampens axis response.
        eigen = np.array([9., .25])
        original = np.r_[.2, np.sqrt(eigen+1)]
        enlarged = np.r_[.4, np.sqrt(4*eigen+1)]
        self.assertAlmostEqual(scale_distance(original, enlarged, 'area'), np.log(2))
        self.assertLess(scale_distance(original, enlarged, 'axes'), np.log(2))

    def test_broadcast_symmetry_and_area_axes_distinguish_extent(self):
        a = np.array([[.2, .3, .1], [.2, .2, .15]])
        area = scale_distance(a[:, None], a[None], 'area')
        axes = scale_distance(a[:, None], a[None], 'axes')
        np.testing.assert_array_equal(area, np.zeros((2, 2)))
        np.testing.assert_array_equal(axes, axes.T)
        self.assertGreater(axes[0, 1], 0)

    def test_fusion_preserves_identity_zero_weight_and_bounds(self):
        scores = np.array([0., .2, .7, 1.])
        np.testing.assert_array_equal(apply_scale(scores, np.zeros(4)), scores)
        np.testing.assert_array_equal(apply_scale(scores, np.arange(4), weight=0), scores)
        result = apply_scale(scores, np.array([0., 1., 10., 1000.]))
        self.assertTrue(np.all(result <= scores))
        self.assertTrue(np.all(result >= .85*scores))
        self.assertEqual(result[0], 0)
        with self.assertRaises(ValueError):
            apply_scale(scores, -np.ones(4))
        with self.assertRaises(ValueError):
            scale_distance(np.zeros(3), np.ones(3), 'area')


if __name__ == '__main__':
    unittest.main()
