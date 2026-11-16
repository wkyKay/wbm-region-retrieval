"""One-off handcrafted A test using the existing EF handcrafted evaluator, read-only cache."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np

p=argparse.ArgumentParser()
for name in ['project','compared','source','output']:p.add_argument('--'+name,required=True)
a=p.parse_args()
project,compared,source,output=[Path(getattr(a,n)).resolve() for n in ['project','compared','source','output']]
sys.path[:0]=[str(project),str(compared)]
from wbm_retrieval.common import load_config,read_json,write_json,file_hash,digest
from wbm_retrieval.data import Protocol,source_fingerprint
from wbm_retrieval.regions import extract_regions
from experiments.handcrafted.a.descriptor import handcrafted_descriptor
from experiments.proposed._shared.ef_suite import evaluate
from wbm_retrieval.diagnostics import batched_map_scores
from wbm_retrieval.matching import map_similarity
from shared.coverage_metrics import evaluate_rankings_file

assert not output.exists(),'Output must be fresh'
assert output!=source and source not in output.parents and output not in source.parents
output.mkdir(parents=True)
started=time.monotonic()
config=load_config(project/'experiments/handcrafted/a/config.json')
assert config['data_protocol']=='deduplicated_v1' and config['evaluation']['metric_version']=='exact_v1'
assert config['matching']['weight']=='sqrt_area'
manifest=read_json(source/'suite_manifest.json');prepared=read_json(source/'cache/prepared.json')
assert prepared['signature']==manifest['signature']==digest({k:v for k,v in manifest.items() if k!='signature'})
assert source_fingerprint(config)==manifest['sources']
for key in ['proposal','matching','handcrafted']:
    assert config[key]==manifest['settings'][key],key
for name,h in manifest['code'].items():assert file_hash(project/name)==h,name
for name in ['cache/labels.npy','cache/test_regions.npz','cache/test_masks.npy']:
    assert file_hash(source/name)==prepared['files'][name],name
config['output_dir']=str(source) # Read-only input cache; evaluate has an explicit independent output.
config['device']='cpu'
protocol=Protocol(config)
assert len(protocol.queries)==7199 and protocol.candidates.candidates_per_query==7198
for name in ['test_queries.csv','test_gallery.csv','metadata.json']:
    assert file_hash(project/'protocol/retrieval_v1'/name)==file_hash(compared/'protocol/retrieval_v1'/name)
with np.load(source/'cache/test_regions.npz') as z:meta=dict(z)
np.testing.assert_array_equal(meta['map_ids'],protocol.candidates.gallery)
labels=np.load(source/'cache/labels.npy')
with np.load(config['data_file']) as z:
    maps=z['arr_0'];np.testing.assert_array_equal(labels,z['arr_1'])
mask_cache=np.load(source/'cache/test_masks.npy',mmap_mode='r')
selected=np.unique(np.r_[np.linspace(0,len(meta['map_ids'])-1,128,dtype=int),np.flatnonzero(np.diff(meta['offsets'])==0)])
groups={}
for i in selected:
    q=int(meta['map_ids'][i]);regions=extract_regions(maps[q],**config['proposal'])
    start,end=meta['offsets'][i:i+2];assert end-start==len(regions)
    for j,r in enumerate(regions):
        np.testing.assert_array_equal(r['mask'],mask_cache[start+j])
        assert r['area']==meta['area'][start+j]
        np.testing.assert_array_equal(handcrafted_descriptor(r['mask'],12),meta['descriptor'][start+j])
    groups[q]={'embedding':meta['descriptor'][start:end], 'area':meta['area'][start:end],
        'position':np.array([r['position'] for r in regions]).reshape(-1,2), 'scale':np.array([r['scale'] for r in regions])}
padded=np.full((len(selected),4),-1,dtype=int)
for j,i in enumerate(selected):
    start,end=meta['offsets'][i:i+2];padded[j,:end-start]=np.arange(start,end)
features=meta['descriptor'].astype(np.float64)
checked=0;maxdiff=0.
for i in selected[:16]:
    q=int(meta['map_ids'][i]);start,end=meta['offsets'][i:i+2]
    sims=np.exp(-np.linalg.norm(features[start:end,None]-features[None],axis=-1)/config['handcrafted']['distance_scale'])
    actual=batched_map_scores(sims,padded,meta['area'][start:end],config['matching'])
    settings={**config['matching'],'representation':'zernike','zernike_distance_scale':1.0}
    expected=np.array([map_similarity(groups[q],groups[int(meta['map_ids'][j])],settings) for j in selected])
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
    maxdiff=max(maxdiff,float(np.max(np.abs(actual-expected))));checked+=len(expected)
print(f'Cache recomputation passed: {len(selected)} maps; scalar/batched parity: {checked} pairs, maxdiff={maxdiff}',flush=True)
paths=[source/'suite_manifest.json',source/'cache/prepared.json',source/'cache/labels.npy',source/'cache/test_regions.npz',source/'cache/test_masks.npy',source/'seed_2026/F/best_nce.pt',source/'seed_2026/F/last.pt']
paths += [project/name for name in manifest['code']]
paths += [compared/'shared/coverage_metrics.py',Path(__file__).resolve()]
snapshot={str(p):{'sha256':file_hash(p),'mtime_ns':p.stat().st_mtime_ns} for p in paths}
write_json(output/'run_manifest.json',{'method':'handcrafted_a','training_required':False,'source_cache':str(source),
    'cache_signature':prepared['signature'],'config':config,'query_count':7199,'gallery_count':7199,
    'candidates_per_query':7198,'candidate_rule':'all_gallery_except_query','ranking_limit':500,
    'representation':'Zernike47','implementation':'existing ef_suite.evaluate(model=None); same handcrafted A descriptor and matching',
    'sources':manifest['sources'],'snapshot':snapshot,'audit':{'reconstructed_maps':len(selected),'scalar_batched_pairs':checked,'max_abs_error':maxdiff}})
write_json(output/'config.json',config)
test=output/'test'
result=evaluate(config,manifest['suite'],'test',test,model=None,rankings=True,bootstrap=config['evaluation']['bootstrap_samples'])
classes=sorted({tuple(labels[i]) for i in protocol.ids['train']});mapping={c:i for i,c in enumerate(classes)}
class_ids=np.full(len(labels),-1,dtype=int)
for q in protocol.ids['test']:class_ids[q]=mapping[tuple(labels[q])]
bundle=SimpleNamespace(labels=labels,class_ids=class_ids)
coverage=evaluate_rankings_file(bundle,protocol.candidates,test/'rankings.csv',config['evaluation']['ks'],test,split_name='test',
    metadata={'method':'handcrafted_a','ranking_source_sha256':file_hash(test/'rankings.csv'),
        'representation':'Zernike47','candidate_rule':'all_gallery_except_query','ranking_limit':500,'smoke':False,
        'retrieval_protocol_signature':protocol.retrieval.signature})
for name,state in snapshot.items():
    path=Path(name);assert file_hash(path)==state['sha256'] and path.stat().st_mtime_ns==state['mtime_ns'],name
write_json(output/'completed.json',{'method':'handcrafted_a','split':'test','smoke':False,'query_count':7199,
    'gallery_count':7199,'candidates_per_query':7198,'source_files_unchanged':True,'seconds':time.monotonic()-started,
    'files':{str(p.relative_to(output)):file_hash(p) for p in test.rglob('*') if p.is_file()}})
print(json.dumps({'exact':result['metrics'],'coverage':coverage['metrics']},indent=2),flush=True)
