import csv
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import tempfile
import unittest
import numpy as np
from cnn_augmentation.tests.test_evaluation import fixture
from cnn_rotation_fusion import evaluation as ev
from cnn_rotation_fusion import run
from cnn_rotation_fusion.state import GRID, THETA0, signed, record, read_json, sha, code_binding


class CoreTests(unittest.TestCase):
    def test_grid_and_dependency_identity(self):
        self.assertEqual(len(GRID),12)
        self.assertEqual({(s['alpha'],s['beta']) for s in GRID.values()},
                         {(a,b) for a in (.4,.5,.6) for b in (.05,.1,.15,.25)})
        binding=code_binding()
        for name in ('cnn_rotation_fusion/evaluation.py','cnn_rotation_fusion/inputs.py','cnn_rotation_fusion/run.py'):
            self.assertTrue(any(path.endswith(name) for path in binding))
        self.assertTrue(all(sha(path)==value for path,value in binding.items()))

    def test_pair_level_weights_retain_exact_theta0(self):
        from cnn_augmentation.evaluation import Provider as Original
        with tempfile.TemporaryDirectory() as temp:
            data,feature=fixture(temp)
            pool=data['protocol'].retrieval.valid
            original=Original(data['cache'],pool)
            current=ev.Provider(data['cache'],pool)
            for query in pool:
                np.testing.assert_array_equal(original.score(query,'fusion')[1],current.score(query,THETA0)[1])
                parts=current.parts(query)
                for name,spec in GRID.items():
                    expected=(1-spec['beta'])*(spec['alpha']*parts['cosine']+(1-spec['alpha'])*parts['structure_similarity'])+spec['beta']*parts['area_similarity']
                    np.testing.assert_array_equal(current.pair_scores(parts,name),np.clip(expected,0,1))
            with self.assertRaises(ValueError):current.pair_scores(parts,{'family':'add','alpha':.7,'beta':.2})

    def test_full_score_and_ordinary_cross_with_source_immutability(self):
        with tempfile.TemporaryDirectory() as temp:
            data,feature=fixture(temp)
            data.update(features={'baseline':feature,'rotation':feature},references={})
            source_hash=sha(feature)
            output=Path(temp)/'run'
            names=[THETA0,'add_a0p5_b0p15']
            rows=ev.evaluate_variants(output,data,'rotation','valid',names,['valid','cross_valid'],'root',1,formal=False,ks=[1,5,10])
            self.assertEqual(len(rows),4)
            self.assertEqual(sha(feature),source_hash)
            for row in rows.values():
                self.assertTrue(row['register']);ev.verify_evaluation_row(row)
                check=read_json(Path(row['path'])/'verification.json')
                self.assertTrue(check['all_queries_full_candidate_topk_exact'])
                self.assertEqual(check['full_candidates_checked'],14*(30 if row['stage']=='cross_valid' else 39))
            resumed=ev.evaluate_variants(output,data,'rotation','valid',names,['valid','cross_valid'],'root',1,formal=False,ks=[1,5,10])
            self.assertEqual(rows,resumed)
            bad=deepcopy(next(iter(rows.values())));bad['spec']={'family':'add','alpha':.5,'beta':.25}
            with self.assertRaises(ValueError):ev.verify_evaluation_row(bad)
            with self.assertRaises(ValueError):ev.evaluate_variants(output,data,'rotation','valid',names,['valid'],'changed',1,formal=False,ks=[1,5,10])

    def test_spawn_score_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            data,feature=fixture(temp,'test');pool=data['protocol'].retrieval.test
            providers=ev.compute_scores(Path(temp)/'out',data['cache'],pool,data['config']['matching'],feature,'context',[THETA0],2)
            provider=ev.Provider(data['cache'],pool)
            for query in pool:
                ids,values=provider.score(query,THETA0)
                np.testing.assert_array_equal(values,providers[THETA0].scores(query,ids))

    def selection_fixture(self,temp,hits):
        rows={}
        for name in GRID:
            folder=Path(temp)/name;folder.mkdir()
            with (folder/'per_query.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['query_id','Precision@10']);writer.writeheader()
                writer.writerows({'query_id':q,'Precision@10':hits[name]/10} for q in range(500))
            record(folder/'metrics.json',{'metrics':{'Precision@10':hits[name]/10},'metric_query_counts':{'Precision@10':500}})
            for model in ('baseline','rotation'):
                rows[model+'/valid/'+name]={'model':model,'stage':'valid','variant':name,'spec':GRID[name],'metrics':str(folder/'metrics.json')}
        return {'evaluations':rows}

    def test_integer_selection_ties_and_no_test_access(self):
        with tempfile.TemporaryDirectory() as temp:
            index=self.selection_fixture(temp,{name:5 for name in GRID})
            selected=run.selection_data(index)
            self.assertEqual(selected['winner'],THETA0)
            self.assertEqual(selected['selected_variants'],[THETA0])
            del index['evaluations']['rotation/valid/'+THETA0]
            with self.assertRaises(ValueError):run.selection_data(index)
        with tempfile.TemporaryDirectory() as temp:
            names=sorted(set(GRID)-{THETA0});hits={name:5 for name in GRID};hits[names[0]]=hits[names[1]]=6
            selected=run.selection_data(self.selection_fixture(temp,hits))
            self.assertEqual(selected['winner'],names[0]);self.assertFalse(selected['test_used'])
        with patch.object(run,'run_context',return_value={'signature':'ctx'}), patch.object(run,'verify_selection',side_effect=ValueError('gate')):
            with patch('cnn_rotation_fusion.inputs.load') as load:
                with self.assertRaisesRegex(ValueError,'gate'):run.test('/tmp/not-a-run')
                load.assert_not_called()
