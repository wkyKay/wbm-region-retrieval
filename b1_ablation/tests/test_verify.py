"""Independent source reconstruction and verification failure examples."""
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from b1_ablation import verify


class SourceVerificationTests(unittest.TestCase):
    def test_original_descriptor_matches_two_point_analytic_result(self):
        raw = np.ones((9,9),dtype=np.uint8)
        raw[4,3] = raw[4,5] = 2
        result = verify.independent_descriptor(raw)
        decay = np.exp(-4/9)
        # One other voter: eigenvalues c and c/2, so both saliencies c/2.
        np.testing.assert_allclose(result["saliency_max"],[decay/2,decay/2],atol=1e-14)
        self.assertEqual(result["original_count"],2)
        self.assertEqual(result["retained_count"],2)
        self.assertAlmostEqual(result["joint128"].sum(),1)
        hist = result["joint128"].reshape(8,4,4)
        self.assertEqual(np.count_nonzero(hist[:,:,:3]),0)
        self.assertEqual(np.count_nonzero(hist[:,:3,:]),0)
        rho = 1/np.sqrt(32)
        position = 7*rho
        expected = np.zeros((8,4,4))
        lower = int(np.floor(position))
        expected[lower,3,3] = 1-(position-lower)
        expected[lower+1,3,3] = position-lower
        np.testing.assert_allclose(hist,expected,atol=1e-14)

    def test_original_descriptor_independent_against_frozen_implementation(self):
        # Several shapes, an isolated point, and dense patch exercise eigenvalue
        # normalization, denoising and soft-bin accumulation independently.
        from b1_structure_fusion.descriptors import describe
        raw = np.ones((31,31),dtype=np.uint8)
        raw[3:24,9] = 2
        raw[17:23,18:25] = 2
        raw[28,28] = 2
        result = verify.independent_descriptor(raw,block_size=5)
        reference = describe(raw)
        for key in ("empty","original_count","retained_count"):
            self.assertEqual(result[key],reference[key])
        for key in ("joint128","saliency_max"):
            np.testing.assert_allclose(result[key],reference[key],atol=2e-11,rtol=2e-11)

    def test_empty_descriptor(self):
        result = verify.independent_descriptor(np.ones((5,5),dtype=np.uint8))
        self.assertTrue(result["empty"])
        self.assertEqual(result["joint128"].sum(),0)
        self.assertEqual(result["retained_count"],0)

    def test_full_pool_detects_omitted_higher_candidate(self):
        # Saved rows are internally descending and legal, but ID 13 outranks both.
        result = verify.compare_full_ranking(1,[11,12,13],np.array([.8,.7,.9]),[11,12])
        self.assertFalse(result["exact_order_match"])
        self.assertEqual(result["mismatch_count"],2)
        self.assertFalse(result["numeric_only"])
        self.assertEqual(result["changes"][0]["recomputed_id"],13)

    def test_numeric_ties_are_reported_not_accepted(self):
        result = verify.compare_full_ranking(1,[11,12],np.array([.5,.5+1e-14]),[11,12])
        self.assertFalse(result["exact_order_match"])
        self.assertTrue(result["numeric_only"])
        self.assertTrue(result["changes"])

    def test_independent_sha_order_matches_frozen_protocol(self):
        from wbm_retrieval.exact_metrics import order_scores
        ids = np.array([8,2,19,5],dtype=np.int64)
        scores = np.array([.4,.4,.9,.4])
        np.testing.assert_array_equal(verify.independent_order(7,ids,scores),
                                      order_scores(ids,scores,7,2026))

    def test_duplicate_query_rows_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"queries.csv"
            path.write_text("query_id,Precision@10\n3,0.5\n3,0.6\n")
            with self.assertRaisesRegex(ValueError,"Duplicate per-query"):
                verify._unique_records(path)

    def test_fixed_component_formula(self):
        f,g,a,j,i = .6,.8,.3,.2,.4
        b0 = (.5*f+.5*g)*np.exp(-.5*a)
        row = {"f_score":.5,"optimal_f_score":f,"global_score":g,"angular_distance":a,
               "joint_distance":j,"independent_distance":i,"structure_distance":j,
               "b0_score":b0,"full_b1_score":b0*np.exp(-2*j)}
        spec = {"family":"J","strength":2.,"local_mode":"none","local_weight":0.,
                "global_weight":1.,"angular_gamma":.5}
        np.testing.assert_allclose(verify.independent_saved_score([row],spec),
                                   [g*np.exp(-.5*a)*np.exp(-2*j)],atol=1e-14)
        spec["global_weight"] = 0.
        np.testing.assert_allclose(verify.independent_saved_score([row],spec),
                                   [np.exp(-.5*a)*np.exp(-2*j)],atol=1e-14)
        row["full_b1_score"] = .99
        with self.assertRaises(AssertionError):
            verify.independent_saved_score([row],spec)


if __name__ == "__main__":
    unittest.main()
