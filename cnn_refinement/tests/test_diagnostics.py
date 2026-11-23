from copy import deepcopy
import unittest
import numpy as np

from cnn_refinement.diagnostics import context_effect,dispersion


class DiagnosticTests(unittest.TestCase):
    def test_context_common_cohort_and_map_clustering(self):
        original=[]
        for sid,regions in ((10,2),(20,1),(30,1)):
            for rid in range(regions):
                for view in ('context_drop','target_only','rot90'):
                    original.append({'space':'h512','view':view,'map_id':sid,'region_id':rid,
                        'token_index':sid+rid,'label_count':2,'label_signature':'0|1',
                        'is_noop':sid==30 and view=='context_drop','has_context':True,
                        'valid_region_input':True,'source_context_pixels':5,'source_target_pixels':10,
                        'source_all_defect_changed_pixels':0 if sid==30 and view=='context_drop' else 3,
                        'cosine_distance':.3,'rotation_distance':.1,'distance_minus_rotation':.2,
                        'larger_than_rotation':1.})
        changed=deepcopy(original)
        for row in changed:
            row['cosine_distance']-=.2 if row['map_id']==10 else .1
            row['distance_minus_rotation']-=.2 if row['map_id']==10 else .1
        result=context_effect(original,changed,repetitions=128)
        for view in ('context_drop','target_only'):
            value=result['results']['h512/'+view+'/mixed']
            self.assertEqual(value['maps'],2)
            self.assertEqual(value['regions'],3)
            self.assertAlmostEqual(value['effects']['cosine_distance']['difference'],-.15)
        changed[0]['source_context_pixels']=9
        with self.assertRaises(ValueError):context_effect(original,changed,repetitions=20)

    def test_collapse_statistics_distinguish_identical_and_diverse_vectors(self):
        collapsed=dispersion(np.ones((20,8)))
        self.assertTrue(collapsed['zero_normalized_variance'])
        diverse=dispersion(np.random.default_rng(2).normal(size=(20,8)))
        self.assertFalse(diverse['zero_normalized_variance'])
        self.assertGreater(diverse['covariance_effective_rank'],2)
        self.assertGreater(diverse['normalized_mean_dimension_variance'],0)
        with self.assertRaisesRegex(ValueError,'Zero-norm'):dispersion(np.zeros((20,8)))


if __name__=='__main__':unittest.main()
