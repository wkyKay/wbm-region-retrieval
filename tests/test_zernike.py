"""Mathematical Zernike values and task-level shape/augmentation invariants."""

import unittest

import numpy as np

from wbm_retrieval.methods.handcrafted import handcrafted_descriptor, moment_orders
from wbm_retrieval.matching import pair_components
from wbm_retrieval.common import ROOT, load_config
from wbm_retrieval.regions import augment


class ZernikeTests(unittest.TestCase):
    def test_known_four_point_moments(self):
        mask = np.zeros((9, 9), dtype=np.uint8)
        mask[2, 4] = mask[6, 4] = mask[4, 2] = mask[4, 6] = 1
        expected = {(2, 0): 3., (2, 2): 0., (3, 1): 0., (3, 3): 0.,
                    (4, 0): 5., (4, 2): 0., (4, 4): 5.}
        values = handcrafted_descriptor(mask, 4)
        np.testing.assert_allclose(values, [expected[order] for order in moment_orders(4)], atol=1e-6)
        rotated = np.zeros_like(mask)
        rotated[2, 2] = rotated[2, 6] = rotated[6, 2] = rotated[6, 6] = 1
        np.testing.assert_allclose(values, handcrafted_descriptor(rotated, 4), atol=1e-6)

    def test_complete_ring_is_different_from_arc_and_disk(self):
        yy, xx = np.indices((65, 65))
        radius = np.hypot(yy - 32, xx - 32)
        ring = (radius >= 18) & (radius <= 24)
        arc = ring & (xx < 38)
        disk = radius <= 24
        a = handcrafted_descriptor(ring)
        self.assertEqual(a.shape, (47,))
        np.testing.assert_allclose(a, handcrafted_descriptor(np.rot90(ring)), atol=1e-6)
        np.testing.assert_allclose(a, handcrafted_descriptor(np.pad(ring, ((3, 7), (8, 2)))), atol=1e-6)
        self.assertGreater(np.linalg.norm(a - handcrafted_descriptor(arc)), .1)
        self.assertGreater(np.linalg.norm(a - handcrafted_descriptor(disk)), .1)
        with self.assertRaises(ValueError):
            handcrafted_descriptor(np.zeros((4, 4)))

    def test_euclidean_scores_use_magnitudes_not_cosine(self):
        config = load_config(ROOT / "configs/experiment_a.json")
        tokens = {"embedding": np.ones((1, 47)), "position": np.zeros((1, 2)), "scale": np.ones(1)}
        candidate = {**tokens, "embedding": np.full((1, 47), 2.)}
        settings = {**config["matching"], "representation": "zernike", "zernike_distance_scale": 1.}
        shape = pair_components(tokens, candidate, settings)[1]
        self.assertAlmostEqual(float(shape[0, 0]), np.exp(-np.sqrt(47)))

    def test_rotation_and_safe_translation_never_truncate_foreground(self):
        image = np.zeros((1, 24, 24), dtype=np.float32)
        image[0, :20, :2] = 1
        image[0, 18:20, :18] = 1
        for seed in range(30):
            transformed = augment(image, np.random.default_rng(seed), max_shift=5)
            self.assertEqual(transformed.sum(), image.sum())
            np.testing.assert_allclose(handcrafted_descriptor(transformed[0]), handcrafted_descriptor(image[0]), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
