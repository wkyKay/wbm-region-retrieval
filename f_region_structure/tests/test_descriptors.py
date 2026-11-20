"""Conservation, original-coordinate and region-integrity checks."""
import unittest
from unittest.mock import patch

import numpy as np

from b1_structure_fusion.descriptors import describe, load_settings
from experiments.proposed.f_segmentation.tensor_voting import tensor_vote
from experiments.proposed.f_segmentation_geometry.scoring import geometry
from f_region_structure.descriptors import describe_regions


def two_regions():
    raw = np.ones((31, 31), dtype=np.uint8)
    labels = np.zeros(raw.shape, dtype=np.uint8)
    raw[3:8, 4:9] = 2
    labels[3:8, 4:9] = 1
    raw[22, 19:26] = 2
    labels[22, 19:26] = 2
    return raw, labels


class RegionDescriptorTests(unittest.TestCase):
    def test_global_pooling_conserves_existing_joint(self):
        raw, labels = two_regions()
        with patch("f_region_structure.descriptors.tensor_vote", wraps=tensor_vote) as vote:
            regions = describe_regions(raw, labels)
        self.assertEqual(vote.call_count, 1, "TV must run once for the full original map")
        whole = describe(raw)
        weights = regions["retained_count"] / regions["retained_count"].sum()
        np.testing.assert_allclose(weights @ regions["joint"], whole["joint128"], rtol=0, atol=3e-16)
        np.testing.assert_array_equal(regions["saliency_max"], whole["saliency_max"])
        self.assertEqual(regions["global_retained_count"], whole["retained_count"])
        self.assertEqual(regions["global_original_count"], 32)
        self.assertEqual(regions["selected_fraction"], 1.)
        np.testing.assert_array_equal(regions["original_count"], [25, 7])
        np.testing.assert_allclose(regions["joint"].sum(1), 1., rtol=0, atol=3e-16)
        np.testing.assert_allclose(regions["shape"].sum(1), 1., rtol=0, atol=3e-16)

    def test_all_point_version_is_no_filter_global_descriptor(self):
        raw, labels = two_regions()
        regions = describe_regions(raw, labels)
        no_filter = load_settings()
        no_filter["noise_threshold"] = 0.
        whole = describe(raw, settings=no_filter)
        weights = regions["original_count"] / regions["original_count"].sum()
        np.testing.assert_allclose(weights @ regions["joint_all"], whole["joint128"], rtol=0, atol=3e-16)
        np.testing.assert_allclose(regions["joint_all"].reshape(2, 8, 4, 4).sum(1).reshape(2, 16),
                                   regions["shape_all"], rtol=0, atol=0)
        np.testing.assert_array_equal(regions["empty_all"], False)

    def test_geometry_matches_original_full_masks(self):
        raw, labels = two_regions()
        raw[0, 0] = 0
        raw[0, 1] = 3
        result = describe_regions(raw, labels)
        for token in (1, 2):
            histogram, fraction = geometry(labels == token, (raw == 1) | (raw == 2))
            np.testing.assert_array_equal(result["radial_hist16"][token-1], histogram)
            self.assertEqual(result["relative_area"][token-1], fraction)

    def test_radius_is_wafer_relative_not_region_relative(self):
        center = np.ones((31, 31), dtype=np.uint8)
        edge = center.copy()
        center_labels = np.zeros(center.shape, dtype=np.uint8)
        edge_labels = center_labels.copy()
        center[15, 13:18] = 2
        center_labels[15, 13:18] = 1
        edge[3, 13:18] = 2
        edge_labels[3, 13:18] = 1
        a, b = describe_regions(center, center_labels), describe_regions(edge, edge_labels)
        np.testing.assert_allclose(a["shape_all"], b["shape_all"], rtol=0, atol=3e-16)
        self.assertGreater(float(np.abs(a["joint_all"]-b["joint_all"]).sum()), 1.)
        self.assertGreater(float(np.abs(a["radial_hist16"]-b["radial_hist16"]).sum()), 1.)

    def test_unassigned_small_components_still_vote(self):
        raw = np.ones((25, 25), dtype=np.uint8)
        labels = np.zeros(raw.shape, dtype=np.uint8)
        raw[10, 8:13] = 2
        labels[10, 8:13] = 1
        isolated = describe_regions(raw, labels)
        raw[12, 9] = raw[12, 11] = 2  # two <5-pixel components, outside the F token
        with_context = describe_regions(raw, labels)
        np.testing.assert_array_equal(isolated["original_count"], with_context["original_count"])
        self.assertEqual(with_context["global_original_count"], 7)
        self.assertAlmostEqual(with_context["selected_fraction"], 5/7)
        self.assertGreater(float(np.abs(isolated["joint_all"]-with_context["joint_all"]).sum()), .01)

    def test_denoised_empty_region_stays_explicit_and_all_points_survive(self):
        raw = np.ones((61, 61), dtype=np.uint8)
        labels = np.zeros(raw.shape, dtype=np.uint8)
        raw[5:25, 5:25] = 2
        labels[5:25, 5:25] = 1
        for row, column in ((45, 45), (44, 45), (46, 45), (45, 44), (45, 46)):
            raw[row, column], labels[row, column] = 2, 2
        settings = load_settings()
        settings["noise_threshold"] = .9  # deliberately empty the weak five-pixel token
        result = describe_regions(raw, labels, settings=settings)
        self.assertTrue(result["empty"][1])
        self.assertEqual(result["quality"][1], 0.)
        np.testing.assert_array_equal(result["joint"][1], 0.)
        np.testing.assert_array_equal(result["shape"][1], 0.)
        self.assertAlmostEqual(float(result["joint_all"][1].sum()), 1.)
        self.assertFalse(result["empty_all"][1])
        np.testing.assert_array_equal(result["quality"], result["retained_count"]/result["original_count"])

    def test_empty_token_maps_have_stable_shapes(self):
        for defect in (False, True):
            raw = np.ones((9, 9), dtype=np.uint8)
            if defect:
                raw[4, 4] = 2  # a genuine no-token F map after min_area filtering
            result = describe_regions(raw, np.zeros(raw.shape, dtype=np.uint8))
            for key, width in (("joint", 128), ("joint_all", 128), ("shape", 16),
                               ("shape_all", 16), ("radial_hist16", 16)):
                self.assertEqual(result[key].shape, (0, width))
            for key in ("original_count", "retained_count", "quality", "empty", "empty_all", "relative_area"):
                self.assertEqual(result[key].shape, (0,))
            self.assertEqual(result["global_original_count"], int(defect))

    def test_small_regions_and_invalid_labels_are_rejected(self):
        raw = np.ones((12, 12), dtype=np.uint8)
        raw[5, 3:8] = 2
        labels = np.zeros(raw.shape, dtype=np.uint8)
        labels[5, 3:8] = 1
        self.assertEqual(describe_regions(raw, labels)["original_count"][0], 5)
        bad = []
        bad.append(labels.astype(float))
        bad.append(labels.astype(bool))
        bad.append(labels[:-1])
        bad.append(labels * 2)
        bad.append(labels * 5)
        negative = labels.astype(np.int16)
        negative[0, 0] = -1
        bad.append(negative)
        foreign = labels.copy()
        foreign[0, 0] = 1
        bad.append(foreign)
        partial = labels.copy()
        partial[5, 7] = 0
        bad.append(partial)
        for value in bad:
            with self.subTest(dtype=value.dtype, shape=value.shape, total=value.sum()):
                with self.assertRaises(ValueError):
                    describe_regions(raw, value)
        for kwargs in ({"min_area": True}, {"min_area": 0}, {"top_k": 5}, {"top_k": 1.5}):
            with self.assertRaises(ValueError):
                describe_regions(raw, labels, **kwargs)

    def test_partial_and_disconnected_cc8_tokens_are_rejected(self):
        raw, labels = two_regions()
        partial = labels.copy()
        partial[3, 4] = 0  # still area >= 5: must fail on full component integrity
        with self.assertRaisesRegex(ValueError, "complete original cc8"):
            describe_regions(raw, partial)
        merged = labels.copy()
        merged[merged == 2] = 1
        with self.assertRaisesRegex(ValueError, "complete original cc8"):
            describe_regions(raw, merged)

    def test_invalid_raw_maps_and_input_immutability(self):
        raw, labels = two_regions()
        original_raw, original_labels = raw.copy(), labels.copy()
        describe_regions(raw, labels)
        np.testing.assert_array_equal(raw, original_raw)
        np.testing.assert_array_equal(labels, original_labels)
        for invalid in (np.zeros((0, 3)), np.zeros((3, 3)), np.full((3, 3), 4), np.full((3, 3), np.nan)):
            with self.assertRaises(ValueError):
                describe_regions(invalid, np.zeros(invalid.shape, dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
