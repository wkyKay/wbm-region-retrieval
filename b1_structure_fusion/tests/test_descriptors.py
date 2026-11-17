"""Independent mathematical checks for the removable B1 experiment."""
from copy import deepcopy
import math
import unittest

import numpy as np
import torch

from b1_structure_fusion import descriptors as d
from b1_structure_fusion import scoring as s
from experiments.proposed.f_fusion_combination_test.pipeline import combined_score
from experiments.proposed.f_segmentation.tensor_voting import tensor_vote, retained_points


def direct_tensor_reference(points, sigma):
    """Literal pairwise reflected-matrix expression, independent of the fast contraction."""
    result = []
    identity = np.eye(2)
    for point in points:
        tensor = np.zeros((2, 2))
        for other in points:
            delta = point-other
            length2 = float(delta @ delta)
            if length2 == 0:
                continue
            direction = delta / math.sqrt(length2)
            rr = np.outer(direction, direction)
            reflection = identity-2*rr
            tensor += math.exp(-length2/sigma) * reflection @ identity @ (identity-.5*rr) @ reflection.T
        result.append(tensor)
    return np.asarray(result)


def analytical_saliencies(tensors):
    # Closed 2x2 eigenvalue formula, not the eigensolver used in production.
    a, b, c = tensors[:, 0, 0], tensors[:, 0, 1], tensors[:, 1, 1]
    gap = np.sqrt((a-c)**2 + 4*b*b)
    return np.column_stack((gap, (a+c-gap)/2))


def independent_deposit(points):
    """Direct corner enumeration; production uses a three-axis contraction."""
    result = np.zeros((8, 4, 4), dtype=np.float64)
    for point in points:
        choices = []
        for value, size in zip(point, (8, 4, 4)):
            coordinate = value*(size-1)
            low = int(math.floor(coordinate))
            frac = coordinate-low
            choices.append([(low, 1-frac)] + ([(low+1, frac)] if frac else []))
        for i, wi in choices[0]:
            for j, wj in choices[1]:
                for k, wk in choices[2]:
                    result[i, j, k] += wi*wj*wk/len(points)
    return result.reshape(-1)


class DescriptorTests(unittest.TestCase):
    def test_literal_tensor_and_before_filter_normalization(self):
        raw = np.ones((15, 15), dtype=np.uint8)
        raw[1, 1] = 0
        raw[1, 2] = 3
        points = np.array([[2, 3], [2, 4], [2, 5], [3, 5], [6, 7], [13, 12]], dtype=np.float64)
        raw[tuple(points.astype(int).T)] = 2
        reference = direct_tensor_reference(points, 9.)
        np.testing.assert_allclose(tensor_vote(points, block_size=2), reference, rtol=0, atol=2e-15)
        saliency = analytical_saliencies(reference)
        maximum = saliency.max(0)
        keep = ~((saliency[:, 0] < .32*maximum[0]) & (saliency[:, 1] < .32*maximum[1]))
        self.assertFalse(keep[-1], "Fixture must include a removed point")
        support = np.argwhere((raw == 1) | (raw == 2))
        center = (support.min(0)+support.max(0))/2
        rho = np.linalg.norm(points-center, axis=1) / np.linalg.norm(support-center, axis=1).max()
        features = np.column_stack((rho, saliency/maximum))[keep]
        actual = d.describe(raw)
        np.testing.assert_allclose(actual["saliency_max"], maximum, rtol=0, atol=2e-15)
        np.testing.assert_allclose(actual["joint128"], independent_deposit(features), rtol=0, atol=2e-15)
        self.assertEqual(actual["original_count"], len(points))
        self.assertEqual(actual["retained_count"], int(keep.sum()))

    def test_strict_and_threshold_and_zero_channels(self):
        saliency = np.array([[1., 1.], [.32, 0.], [0., .32], [.31, .31]])
        np.testing.assert_array_equal(retained_points(saliency), [True, True, True, False])
        raw = np.ones((7, 7), dtype=np.uint8)
        raw[3, 3] = 2
        one = d.describe(raw)
        self.assertFalse(one["empty"])
        self.assertEqual(one["retained_count"], 1)
        np.testing.assert_array_equal(one["saliency_max"], [0., 0.])
        self.assertEqual(one["joint128"][0], 1.)
        raw[3, 3] = 1
        empty = d.describe(raw)
        self.assertTrue(empty["empty"])
        np.testing.assert_array_equal(empty["joint128"], 0.)
        with self.assertRaisesRegex(ValueError, "support"):
            d.describe(np.zeros((3, 3)))

    def test_soft_bin_nodes_midpoints_and_mass(self):
        endpoint = d.soft_histogram(np.array([[0., 0., 0.], [1., 1., 1.]]))
        expected = np.zeros((8, 4, 4))
        expected[0, 0, 0] = expected[7, 3, 3] = .5
        np.testing.assert_array_equal(endpoint, expected)
        midpoint = d.soft_histogram(np.array([[1/14, 1/6, 1/6]]))
        expected = np.zeros((8, 4, 4))
        expected[:2, :2, :2] = 1/8
        np.testing.assert_array_equal(midpoint, expected)
        rng = np.random.default_rng(2026)
        points = rng.random((31, 3))
        actual = d.soft_histogram(points).reshape(-1)
        np.testing.assert_allclose(actual, independent_deposit(points), rtol=0, atol=1e-16)
        self.assertAlmostEqual(float(actual.sum()), 1., places=14)
        self.assertTrue(np.all(actual >= 0))
        for bad in (np.array([[np.nan, 0., 0.]]), np.array([[1.01, 0., 0.]])):
            with self.assertRaises(ValueError):
                d.soft_histogram(bad)

    def test_joint_relationship_control_has_identical_marginals(self):
        first = d.soft_histogram(np.array([[0., 0., 0.], [1., 1., 1.]])).reshape(-1)
        second = d.soft_histogram(np.array([[0., 1., 1.], [1., 0., 0.]])).reshape(-1)
        distributions = d.distributions(np.stack([first, second]))
        for family in ("R", "T", "I"):
            np.testing.assert_array_equal(distributions[family][0], distributions[family][1])
            self.assertEqual(d.hellinger(distributions[family][0], distributions[family][1:])[0], 0.)
        self.assertEqual(d.hellinger(first, second[None])[0], 1.)
        np.testing.assert_array_equal(distributions["I"][0].reshape(8, 4, 4).sum((1, 2)), distributions["R"][0])
        np.testing.assert_array_equal(distributions["I"][0].reshape(8, 4, 4).sum(0).reshape(-1), distributions["T"][0])

    def test_hellinger_distance_definition_and_empty_atom(self):
        q = np.array([.25, .75])
        candidates = np.array([[.25, .75], [.75, .25], [0., 0.]])
        value = d.hellinger(q, candidates, c_empty=np.array([False, False, True]))
        self.assertEqual(value[0], 0.)
        self.assertAlmostEqual(value[1], math.sqrt(1-math.sqrt(3)/2), places=14)
        self.assertEqual(value[2], 1.)
        self.assertEqual(d.hellinger(candidates[2], candidates, True, np.array([False, False, True]))[-1], 0.)
        self.assertAlmostEqual(value[1], d.hellinger(candidates[1], q[None])[0], places=15)
        with self.assertRaisesRegex(ValueError, "flags"):
            d.hellinger(q, candidates)
        with self.assertRaises(ValueError):
            d.hellinger(np.array([.1, .1]), candidates[:1])

    def test_original_map_quarter_rotation_and_small_components(self):
        raw = np.ones((19, 19), dtype=np.uint8)
        raw[2:5, 3:8] = 2
        raw[12:16, 14] = 2
        baseline = d.describe(raw)
        for k in (1, 2, 3):
            rotated = d.describe(np.rot90(raw, k))
            np.testing.assert_allclose(rotated["joint128"], baseline["joint128"], rtol=0, atol=3e-14)
            self.assertEqual(rotated["retained_count"], baseline["retained_count"])
        # No min_area or Top-4 filter is applied: even a new singleton votes.
        raw[4, 9] = 2
        changed = d.describe(raw)
        self.assertEqual(changed["original_count"], baseline["original_count"]+1)
        self.assertGreater(float(np.abs(changed["joint128"]-baseline["joint128"]).sum()), 0.)

    def test_config_validation_and_default_18_variants(self):
        settings = d.load_settings()
        variants = s.valid_variants(settings)
        self.assertEqual(len(variants), 18)
        self.assertEqual(sum(v["strength"] == 0 for v in variants.values()), 1)
        self.assertEqual(variants["J_only"], {"family": "J_only", "strength": 1.})
        for key, value in (("bins", [4, 4, 4]), ("distance", "squared_hellinger"),
                           ("lambda_grid", [0, 1, .5]), ("block_size", 0)):
            bad = deepcopy(settings)
            bad[key] = value
            with self.assertRaises(ValueError):
                d.validate_settings(bad)


class ScoringTests(unittest.TestCase):
    def test_exact_baseline_reversion_and_full_product(self):
        values = {"optimal": np.array([.2, .5, .9]), "angular_distance": np.array([.8, .4, 0.])}
        baseline = combined_score(values)
        np.testing.assert_array_equal(s.baseline_score(values), baseline)
        self.assertIs(s.score_variant(baseline, {}, {"family": "B0", "strength": 0}), baseline)
        for family in d.FAMILIES:
            np.testing.assert_array_equal(s.score_variant(values, {}, {"family": family, "strength": 0}), baseline)
        distance = np.array([0., .5, 1.])
        actual = s.score_variant(values, {"J": distance}, {"family": "J", "strength": 2.})
        np.testing.assert_array_equal(actual, baseline*np.exp(-2*distance))
        np.testing.assert_array_equal(s.score_variant(np.zeros(3), {"J": distance}, {"family": "J_only", "strength": 1.}), np.exp(-distance))
        with self.assertRaises(ValueError):
            s.score_variant(values, {"J": distance}, {"family": "J", "strength": -1})

    def test_original_id_mapping_cpu_distances_and_explicit_empty_flags(self):
        rng = np.random.default_rng(39)
        joint = rng.random((4, 128))
        joint /= joint.sum(1, keepdims=True)
        joint[2] = 0
        ids = np.array([901, 4, 77, 33])
        empty = np.array([False, False, True, False])
        cache = s.StructuralDistances(ids, joint, empty, "cpu")
        got = cache.for_query(33, np.array([77, 901, 4]))
        for family, matrix in d.distributions(joint).items():
            expected = d.hellinger(matrix[3], matrix[[2, 0, 1]], False, empty[[2, 0, 1]])
            np.testing.assert_array_equal(got[family], expected)
        with self.assertRaisesRegex(ValueError, "Missing"):
            cache.for_query(0, np.array([901]))
        with self.assertRaises(ValueError):
            s.StructuralDistances(np.array([1, 1, 2, 3]), joint, empty)
        with self.assertRaises(ValueError):
            s.StructuralDistances(ids, joint, np.zeros(4, dtype=bool))
        with self.assertRaises(ValueError):
            s.StructuralDistances(ids, joint, empty, "mps")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA validation requires a GPU")
    def test_cuda_float64_matches_cpu_and_zero_is_exact(self):
        rng = np.random.default_rng(2026)
        joint = rng.random((12, 128))
        joint /= joint.sum(1, keepdims=True)
        joint[3] = 0
        ids = np.arange(12)*13
        empty = np.zeros(12, dtype=bool)
        empty[3] = True
        cpu = s.StructuralDistances(ids, joint, empty, "cpu")
        gpu = s.StructuralDistances(ids, joint, empty, "cuda")
        for query in (ids[0], ids[3]):
            candidates = ids[ids != query]
            a, b = cpu.for_query(query, candidates), gpu.for_query(query, candidates)
            for family in d.FAMILIES:
                np.testing.assert_allclose(a[family], b[family], rtol=0, atol=3e-15)
            base = np.linspace(.3, .9, len(candidates))
            np.testing.assert_array_equal(s.score_variant(base, b, {"family": "J", "strength": 0}), base)


if __name__ == "__main__":
    unittest.main()
