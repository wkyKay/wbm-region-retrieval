import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from wbm_retrieval.exact_metrics import order_scores,tie_keys
from fgj_cross_test.scoring import score_all,score_variant
from fgj_cross_test.state import SPECS,VARIANTS


class FormulaTests(unittest.TestCase):
    def setUp(self):
        self.components={'F':np.array([0.,1.,.3]),'G':np.array([1.,0.,.7]),
                         'joint_distance':np.array([0.,1.,.5]),'WBBS':np.array([.2,.8,.1])}

    def test_original_scale_and_frozen_strength_have_exact_endpoints(self):
        s=score_all(self.components)
        j=np.array([1.,np.exp(-2.),np.exp(-1.)])
        np.testing.assert_array_equal(s['J_ONLY'],j)
        np.testing.assert_array_equal(s['GJ_ADD'],np.array([1.,.5*np.exp(-2.),.35+.5*np.exp(-1.)]))
        np.testing.assert_array_equal(s['FJ_ADD'],np.array([.5,.5+.5*np.exp(-2.),.15+.5*np.exp(-1.)]))
        for name,key in [('F_ONLY','F'),('G_ONLY','G'),('WBBS','WBBS')]:
            np.testing.assert_array_equal(s[name],self.components[key])
        for name in VARIANTS:
            np.testing.assert_array_equal(score_variant(self.components,name,SPECS[name]),s[name])

    def test_rejects_formula_search_or_component_substitution(self):
        with self.assertRaises(ValueError):score_variant(self.components,'GJ_ADD',{'formula':'0.6*G+0.4*J'})
        with self.assertRaises(ValueError):score_all({**self.components,'J':np.ones(3)})
        for invalid in [np.array([np.nan,0.,1.]),np.array([-1.,0.,1.]),np.array([2.,0.,1.]),np.array([.1])]:
            with self.subTest(value=invalid),self.assertRaises(ValueError):
                score_all({**self.components,'F':invalid})

    def test_shared_tie_keys_preserve_original_order_for_every_formula(self):
        ids=np.array([11,3,20,7,99,4]);q=1
        c={k:np.resize(v,len(ids)) for k,v in self.components.items()}
        keys=tie_keys(q,ids,2026)
        for scores in score_all(c).values():
            np.testing.assert_array_equal(np.lexsort((ids,keys,-scores)),order_scores(ids,scores,q,2026))

    def test_filtering_before_topk_keeps_stronger_eligible_candidates(self):
        ids=np.array([5,6,7,8]); eligible=np.array([False,False,True,True])
        c={'F':np.array([1.,.9,.8,.7]),'G':np.array([1.,.9,.8,.7]),
           'WBBS':np.array([1.,.9,.8,.7]),'joint_distance':np.array([0.,.1,.2,.3])}
        for full in score_all(c).values():
            filtered=full[eligible];candidates=ids[eligible]
            selected=candidates[order_scores(candidates,filtered,1)[:2]]
            self.assertEqual(set(selected),{7,8})
            self.assertEqual(len(ids[order_scores(ids,full,1)[:2]][eligible[:2]]),0)


class DescriptiveOrderingTests(unittest.TestCase):
    def test_reports_metric_specific_ties_without_selecting_test_winner(self):
        from fgj_cross_test.run import summarize
        from wbm_retrieval.common import read_json
        values=[.8,.8,.4,.3,.2,.1]
        rows={n:{'coverage_metrics':{'Coverage-nDCG@10':v,'MeanCoverage@10':1-v}}
              for n,v in zip(VARIANTS,values)}
        with tempfile.TemporaryDirectory() as t:
            summarize(t,rows)
            with (Path(t)/'metric_orderings.csv').open() as handle:
                ranking=list(csv.DictReader(handle))
            first=[r for r in ranking if r['metric']=='Coverage-nDCG@10']
            self.assertEqual([int(r['rank']) for r in first],[1,1,3,4,5,6])
            other=[r for r in ranking if r['metric']=='MeanCoverage@10']
            self.assertEqual(other[0]['variant'],'WBBS')
            design=read_json(Path(t)/'comparison_design.json')
            self.assertIsNone(design['primary_variant'])
            self.assertFalse(design['test_selects_parameters'])

    def test_real_pipeline_batches_all_formulas_and_reuses_sealed_children(self):
        from fgj_cross_test import pipeline
        from fgj_cross_test.tests.test_verify import fixture
        from wbm_retrieval.common import digest,read_json
        with tempfile.TemporaryDirectory() as root:
            source=Path(root)/'synthetic_source';source.mkdir()
            _,pool,bundle,ks,_,provider,binding=fixture(source)
            data=dict(pool=pool,bundle=bundle,source_binding=binding,
                      source_binding_signature=digest(binding))
            output=Path(root)/'pipeline';output.mkdir()
            with patch.object(pipeline,'SETTINGS',{'ks':ks,'tie_seed':2026}):
                results=pipeline.evaluate(output,data,'synthetic-context',provider.scores,formal=False)
                self.assertEqual(len(provider.calls),2*len(pool))
                self.assertEqual(set(results),set(VARIANTS))
                checked=read_json(output/'shared_verification.json')
                self.assertEqual(checked['shared']['components_calls'],len(pool))
                provider.calls.clear()
                again=pipeline.evaluate(output,data,'synthetic-context',provider.scores,formal=False)
                self.assertEqual(again,results)
                self.assertEqual(provider.calls,[])
                final=output/'test'/'GJ_ADD'
                final.rename(final.parent/'.GJ_ADD_attempt_interrupted')
                recovered=pipeline.evaluate(output,data,'synthetic-context',provider.scores,formal=False)
                self.assertEqual(recovered,results)
                self.assertEqual(provider.calls,[])
                self.assertTrue((final/'completed.json').exists())


if __name__=='__main__':unittest.main()
