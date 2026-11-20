"""Full-gallery integration and source-isolated runner boundary checks."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from b1_structure_fusion.tests.test_pipeline import fixture, snapshot
from gj_fusion import pipeline, run, scoring
from wbm_retrieval.common import read_json


class PipelineTests(unittest.TestCase):
    def data(self):
        pool,bundle,desc,_=fixture()
        features=np.zeros((12,512),dtype=np.float64)
        features[:,0]=1.
        features[11,0]=.8
        features[11,1]=.6
        return dict(pool=pool,bundle=bundle,descriptors=desc,
                    global_ids=desc['sample_ids'],global_features=features)

    def test_complete_grid_verified_and_completed_resume_read_only(self):
        data=self.data()
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(StringIO()):
            out=Path(tmp)/'run'
            results=pipeline.evaluate(out,data,'synthetic',scoring.frozen_grid(),'cpu',[1,5,10])
            self.assertEqual(set(results),set(scoring.frozen_grid()))
            self.assertTrue(all(r['num_queries']==3 for r in results.values()))
            for name in results:
                check=read_json(out/'valid'/name/'verification.json')
                self.assertTrue(check['all_queries_full_candidate_topk_exact'])
                self.assertEqual(check['full_candidates_checked'],33)
                self.assertEqual(read_json(out/'valid'/name/'run_identity.json')['variant'],name)
            before=snapshot(out)
            repeated=pipeline.evaluate(out,data,'synthetic',scoring.frozen_grid(),'cpu',[1,5,10])
            self.assertEqual(results,repeated)
            self.assertEqual(before,snapshot(out))

    def test_partial_valid_and_nonfrozen_spec_rejected(self):
        data=self.data()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                pipeline.evaluate(tmp,data,'synthetic',{'product':scoring.frozen_grid()['product']},'cpu',[1,5,10])
            grid=scoring.frozen_grid();grid['product']['alpha']=.6
            with self.assertRaises(ValueError):
                pipeline.evaluate(tmp,data,'synthetic',grid,'cpu',[1,5,10])

    def test_test_requires_valid_before_loading_inputs(self):
        args=run.parser().parse_args(['--stage','test'])
        with patch.object(run.inputs,'load_inputs') as loader:
            with self.assertRaisesRegex(ValueError,'valid-run'):
                run.run_stage(args,'test')
            loader.assert_not_called()

    def test_reject_output_in_source_and_existing_directory(self):
        args=run.parser().parse_args(['--stage','valid','--output-root','artifacts/forbidden'])
        with self.assertRaises(ValueError):run.output_path(args,'valid')
        args=run.parser().parse_args(['--stage','valid','--output-root','artificial'])
        with self.assertRaises(ValueError):run.output_path(args,'valid')


if __name__=='__main__':unittest.main()
