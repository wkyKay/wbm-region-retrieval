from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from copy import deepcopy
import unittest

import numpy as np

from cnn_augmentation import evaluation as ev
from f_region_structure.state import seal
from wbm_retrieval.common import file_hash, read_json, write_json
from wbm_retrieval.retrieval_protocol import FullGalleryPool


def fixture(root, split='valid'):
    root=Path(root)
    rng=np.random.default_rng(2026)
    # Distinct IDs for the other split permit strict cache/split identification.
    ids=np.arange(0,40) if split=='valid' else np.arange(40,80)
    n=len(ids)*2
    features=rng.normal(size=(n,512))
    features/=np.linalg.norm(features,axis=1,keepdims=True)
    probability=rng.uniform(size=(n,128));probability/=probability.sum(1,keepdims=True)
    arrays={'map_ids':ids,'offsets':np.arange(0,n+1,2),'area':rng.integers(5,200,n).astype(float),
            'owners':np.repeat(ids,2),'relative_area':rng.uniform(.01,.2,n),
            'joint_all':probability,'joint':probability,'histograms':np.ones((n,16))/16}
    folder=root/'source';folder.mkdir(parents=True)
    np.savez(folder/'regions.npz',**arrays)
    np.save(folder/'h512.npy',features)
    seal(folder,'source')
    cache={**arrays,'features':np.load(folder/'h512.npy',mmap_mode='r'),'folder':folder}
    labels=np.zeros((80,8),dtype=np.uint8)
    labels[np.arange(80)%4==0,0]=1
    labels[np.arange(80)%4==1,1]=1
    labels[np.arange(80)%4==2,:2]=1
    labels[np.arange(80)%4==3,2]=1
    _,classes=np.unique(labels,axis=0,return_inverse=True)
    valid=FullGalleryPool(np.arange(0,40,3),np.arange(40),'valid','toy')
    test=FullGalleryPool(np.arange(40,80),np.arange(40,80),'test','toy')
    bundle=SimpleNamespace(labels=labels,class_ids=classes,valid_ids=valid.gallery,reserved_ids=test.gallery)
    data={'cache':cache,'bundle':bundle,'protocol':SimpleNamespace(retrieval=SimpleNamespace(valid=valid,test=test)),
          'config':{'matching':ev.normalize_matching()},'source_binding':{str((folder/'completed.json').resolve()):file_hash(folder/'completed.json')},
          'replay_references':{}}
    return data,folder/'h512.npy'


class EvaluationTests(unittest.TestCase):
    def test_scientific_dependency_identity_is_complete_and_hashed(self):
        identity=ev.code_identity()
        self.assertGreaterEqual(len(identity),20)
        for name in ('evaluation.py','scoring.py','exact_metrics.py','cross_combination.py','verify.py'):
            self.assertTrue(any(Path(path).name==name for path in identity))
        self.assertTrue(all(Path(path).is_absolute() and file_hash(path)==sha for path,sha in identity.items()))

    def test_end_to_end_full_gallery_cross_rerank_and_effective_n(self):
        with tempfile.TemporaryDirectory() as temp:
            data,path=fixture(temp)
            source_hash=file_hash(path)
            index=ev.evaluate_model(Path(temp)/'run',data,'crop',path,'context',workers=1,formal=False,ks=[1,5,10])
            self.assertEqual(index['new_evaluations'],4)
            self.assertEqual(file_hash(path),source_hash)
            self.assertEqual(set(index['evaluations']),{'valid/cnn_only','valid/fusion','cross_valid/cnn_only','cross_valid/fusion'})
            cross=index['evaluations']['cross_valid/cnn_only']
            folder=Path(cross['path'])
            check=read_json(folder/'verification.json')
            self.assertEqual(check['full_candidates_checked'],14*30)
            self.assertTrue(check['all_queries_full_candidate_topk_exact'])
            self.assertFalse((folder/'metrics.json').exists())
            counts=cross['metric_query_counts']['coverage']
            self.assertLess(counts['FullCoverage_mAP@10'],14)
            self.assertLess(counts['Coverage-nDCG@10'],14)
            self.assertEqual(counts['MeanCoverage@10'],14)
            baseline=ev.Provider(data['cache'],data['protocol'].retrieval.valid,data['config']['matching'])
            saved=ev.CachedProvider(index['score_cache']['fusion']['path'])
            for q in baseline.pool:
                ids,scores=baseline.score(q,'fusion')
                np.testing.assert_array_equal(saved.scores(q,ids),scores)
            reused=ev.evaluate_model(Path(temp)/'run',data,'crop',path,'context',workers=1,formal=False,ks=[1,5,10])
            self.assertEqual(reused,index)

    def test_test_split_spawn_workers_and_immutable_context(self):
        with tempfile.TemporaryDirectory() as temp:
            data,path=fixture(temp,'test')
            result=ev.evaluate_model(Path(temp)/'run',data,'noise',path,'context',workers=2,formal=False,ks=[1,5,10])
            self.assertEqual(result['split'],'test')
            self.assertEqual(result['evaluations']['test/fusion']['metric_query_counts']['exact']['mAP@10'],40)
            with self.assertRaises(ValueError):
                ev.evaluate_model(Path(temp)/'run',data,'noise',path,'changed',workers=1,formal=False,ks=[1,5,10])

    def test_cnn_only_ignores_geometry_and_preserves_greedy_area_denominator(self):
        parts={'cosine':np.array([[.2,.7],[.8,.1]],dtype=np.float32),
               'structure_similarity':np.zeros((2,2)), 'area_similarity':np.zeros((2,2))}
        np.testing.assert_array_equal(ev.Provider.pair_scores(parts,'cnn_only'),parts['cosine'])
        with tempfile.TemporaryDirectory() as temp:
            data,_=fixture(temp)
            p=ev.Provider(data['cache'],data['protocol'].retrieval.valid)
            q=next(iter(p.pool));original=p.parts(q)
            changed={k:(1-v if k!='cosine' else v) for k,v in original.items()}
            np.testing.assert_array_equal(p.score(q,'cnn_only',original)[1],p.score(q,'cnn_only',changed)[1])
            result=ev.audit_formulas(p,data['bundle'])
            self.assertEqual(result['num_classes'],4)
            self.assertEqual(result['variants']['cnn_only']['candidate_pairs'],80)

    def test_reject_changed_baseline_or_unnormalized_features(self):
        with tempfile.TemporaryDirectory() as temp:
            data,path=fixture(temp)
            new=Path(temp)/'changed.npy'
            np.save(new,np.asarray(data['cache']['features'])[::-1])
            with self.assertRaisesRegex(ValueError,'Baseline must retain'):
                ev.evaluate_model(Path(temp)/'run',data,'baseline',new,'context',workers=1,formal=False,ks=[1,5,10])
            np.save(new,np.asarray(data['cache']['features'])*2)
            with self.assertRaisesRegex(ValueError,'normalized'):
                ev.replaced_cache(data['cache'],new)

    def test_all_baseline_roles_keep_canonical_identity_and_create_no_score_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            data,path=fixture(temp)
            oldroot=Path(temp)/'old'
            old=ev.evaluate_model(oldroot,data,'crop',path,'old-context',workers=1,formal=False,ks=[1,5,10])
            for item in old['evaluations'].values():item['model']='baseline'
            index=oldroot/'evaluation/baseline/evaluation_index_valid.json'
            write_json(index,old)
            write_json(index.parent/'model_context_valid.json',{'token_layout':ev.token_layout(data['cache']),
                       'h512_sha256':file_hash(path)})
            data['baseline_evaluation_root']=oldroot
            data['source_binding'].update({str(index.resolve()):file_hash(index), str(path.resolve()):file_hash(path)})
            for item in old['evaluations'].values():
                p=Path(item['path'])/'completed.json';data['source_binding'][str(p.resolve())]=file_hash(p)
            with patch('cnn_refinement.register.validate_model_split',return_value=old):
                result=ev.reference_baseline(Path(temp)/'new',data,'valid','new-context')
            self.assertEqual(result['new_evaluations'],0)
            self.assertEqual(result['historical_references'],4)
            for key,item in result['evaluations'].items():
                self.assertEqual(item['key'],old['evaluations'][key]['key'])
                self.assertFalse(item['register']);self.assertTrue(item['historical_reference'])
            self.assertFalse((Path(temp)/'new/evaluation/baseline/score_cache').exists())
            from cnn_augmentation import register
            inputs={'context_signature':'new-context','baseline_evaluation_root':str(oldroot),'source_binding':data['source_binding']}
            with patch('cnn_refinement.register.validate_model_split',return_value=old):
                self.assertEqual(register._baseline_reference(Path(temp)/'new','valid',{'signature':'new-context'},inputs,result),result)
                changed=deepcopy(result);changed['evaluations']['valid/cnn_only']['key']='duplicate_new_key'
                with self.assertRaisesRegex(ValueError,'existing baseline role'):
                    register._baseline_reference(Path(temp)/'new','valid',{'signature':'new-context'},inputs,changed)

    def test_evaluation_identity_namespaces_distinct_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            a=ev._identity(Path(temp)/'run1/evaluation/crop','test','cnn_only','science1')
            b=ev._identity(Path(temp)/'run2/evaluation/crop','test','cnn_only','science2')
            self.assertNotEqual(a['key'],b['key'])


if __name__=='__main__':unittest.main()
