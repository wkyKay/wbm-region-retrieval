"""Check the ablation against the existing scalar matcher, including ties."""
import unittest

import numpy as np

from scripts.evaluate_f_bidirectional import bidirectional_scores
from wbm_retrieval.matching import greedy_assignment


class BidirectionalTests(unittest.TestCase):
    settings = dict(lambda_pos=0, lambda_scale=0, shape_threshold=0,
                    score_threshold=0, weight="sqrt_area")

    def reference(self, matrix, area, settings):
        matrix = np.clip(matrix, 0, 1).copy()
        matrix[matrix < settings["shape_threshold"]] = 0
        weights = (np.ones_like(area) if settings["weight"] == "uniform" else
                   np.sqrt(area) if settings["weight"] == "sqrt_area" else area)
        pairs = greedy_assignment(matrix, settings["score_threshold"])
        return sum(weights[i]*score for i, _, score in pairs)/sum(weights) if len(area) else 0

    def test_independent_directions_and_padding(self):
        rng = np.random.default_rng(2026)
        for weight in ("uniform", "sqrt_area", "area"):
            settings = {**self.settings, "weight": weight, "shape_threshold": .2, "score_threshold": .3}
            for nq in range(5):
                for nc in range(5):
                    # Quantization deliberately introduces exact score ties.
                    matrix = rng.integers(-1, 6, size=(nq, nc))/5
                    qa, ca = rng.integers(1, 100, nq), rng.integers(1, 100, nc)
                    padded = np.zeros((1, 4, 4))
                    padded[0, :nq, :nc] = matrix
                    qp, cp = np.zeros((1, 4)), np.zeros((1, 4))
                    qp[0, :nq], cp[0, :nc] = qa, ca
                    a, b, h = bidirectional_scores(padded, qp, cp, settings)
                    self.assertAlmostEqual(a[0], self.reference(matrix, qa, settings))
                    self.assertAlmostEqual(b[0], self.reference(matrix.T, ca, settings))
                    swapped = bidirectional_scores(padded.transpose(0, 2, 1), cp, qp, settings)[2]
                    np.testing.assert_allclose(h, swapped)
                    self.assertTrue(0 <= h[0] <= 1)

    def test_extra_candidate_region_penalized(self):
        a, b, h = bidirectional_scores(np.array([[[1., .2]]]), np.array([[25.]]),
                                      np.array([[25., 25.]]), self.settings)
        np.testing.assert_allclose(a, [1])
        np.testing.assert_allclose(b, [.5])
        np.testing.assert_allclose(h, [2/3])

    def test_empty_source(self):
        a, b, h = bidirectional_scores(np.empty((2, 0, 4)), np.empty((2, 0)),
                                      np.ones((2, 4)), self.settings)
        np.testing.assert_array_equal([a, b, h], np.zeros((3, 2)))

    def test_zero_scores(self):
        scores = bidirectional_scores(np.zeros((2, 4, 4)), np.ones((2, 4)),
                                      np.ones((2, 4)), self.settings)
        np.testing.assert_array_equal(scores, np.zeros((3, 2)))


if __name__ == "__main__":
    unittest.main()
