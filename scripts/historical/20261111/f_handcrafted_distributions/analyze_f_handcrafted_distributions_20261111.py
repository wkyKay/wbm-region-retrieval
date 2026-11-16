"""Full frozen-test score histograms for F and handcrafted A; no training/tuning."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

p=argparse.ArgumentParser()
for n in ('project','feature-run','handcrafted-run','output'):
    p.add_argument('--'+n,required=True)
a=p.parse_args()
project,feature_run,hand_run,output=[Path(getattr(a,n)).resolve() for n in ('project','feature_run','handcrafted_run','output')]
sys.path.insert(0,str(project))
from wbm_retrieval.common import file_hash,read_json,write_json
from wbm_retrieval.diagnostics import batched_map_scores

assert not output.exists()
output.mkdir(parents=True)
started=time.monotonic()
torch.set_num_threads(4)
fm=read_json(feature_run/'run_manifest.json')
hm=read_json(hand_run/'run_manifest.json')
source=Path(fm['source_run'])
assert source==Path(hm['source_cache'])
assert fm['cache_signature']==hm['cache_signature']
prepared=read_json(source/'cache/prepared.json')
feature_path=feature_run/'test_features.npy'
assert file_hash(feature_path)==read_json(feature_run/'bundle_hashes.json')['test_features.npy']
assert file_hash(source/'cache/test_regions.npz')==prepared['files']['cache/test_regions.npz']
assert file_hash(source/'cache/labels.npy')==prepared['files']['cache/labels.npy']
with np.load(source/'cache/test_regions.npz') as z:
    meta=dict(z)
maps,offsets,area=meta['map_ids'],meta['offsets'],meta['area']
h=np.load(feature_path)
d=meta['descriptor'].astype(np.float64)
labels=np.load(source/'cache/labels.npy')[maps].astype(int)
owners=np.repeat(maps,np.diff(offsets))
assert len(h)==len(d)==offsets[-1] and len(maps)==7199
settings=hm['config']['matching']
assert settings['shape_threshold']==settings['score_threshold']==settings['lambda_pos']==settings['lambda_scale']==0
assert hm['config']['handcrafted']['distance_scale']==1.0 and settings['weight']=='sqrt_area'
snapshot={str(f):{'sha256':file_hash(f),'mtime_ns':f.stat().st_mtime_ns} for f in
    [feature_path,source/'cache/test_regions.npz',source/'cache/labels.npy',feature_run/'run_manifest.json',
     hand_run/'run_manifest.json',project/'wbm_retrieval/diagnostics.py',Path(__file__).resolve()]}
write_json(output/'manifest.json',{'source_run':str(source),'F_checkpoint_sha256':fm['checkpoint_sha256'],
    'F_region_score':'clip(cosine(h512),0,1)','handcrafted_region_score':'exp(-L2(Zernike47)/1.0)',
    'matching':settings,'test_wafer_count':len(maps),'test_region_count':len(h),
    'same_wafer_region_pairs_excluded':True,'map_candidate_rule':'all_gallery_except_query',
    'pair_sampling':'none: exhaustive directed pairs','statistics_bins':1000,'display_bins':100,
    'label_groups':'Wafer labels define whole-map relevance only, not region ground truth',
    'snapshot':snapshot})

BINS=1000
edges=np.linspace(0,1,BINS+1)
def empty():
    return {'count':0,'sum':0.,'sumsq':0.,'min':float('inf'),'max':-float('inf'),'counts':np.zeros(BINS,dtype=np.int64)}
regional={name:empty() for name in ('F','Handcrafted')}
whole={name:{group:empty() for group in ('all','exact','not_exact','full_coverage','not_full_coverage')}
       for name in ('F','Handcrafted')}
def cpu_add(acc,v):
    if not len(v):return
    assert np.isfinite(v).all() and v.min()>=-1e-12 and v.max()<=1+1e-12
    v=np.clip(v,0,1)
    acc['count']+=len(v);acc['sum']+=float(v.sum());acc['sumsq']+=float(v@v)
    acc['min']=min(acc['min'],float(v.min()));acc['max']=max(acc['max'],float(v.max()))
    indices=np.minimum((v*BINS).astype(np.int64),BINS-1)
    acc['counts']+=np.bincount(indices,minlength=BINS)
def gpu_add(acc,v):
    acc['count']+=v.numel();acc['sum']+=v.sum().item();acc['sumsq']+=(v*v).sum().item()
    acc['min']=min(acc['min'],v.min().item());acc['max']=max(acc['max'],v.max().item())
    acc['counts']+=torch.histc(v,bins=BINS,min=0,max=1).cpu().numpy().astype(np.int64)

H=torch.from_numpy(h).to('cuda:0',dtype=torch.float64)
D=torch.from_numpy(d).to('cuda:0',dtype=torch.float64)
owner_tensor=torch.from_numpy(owners).to('cuda:0')
padded=np.full((len(maps),int(np.diff(offsets).max())),-1,dtype=int)
for i,(start,end) in enumerate(zip(offsets[:-1],offsets[1:])):
    padded[i,:end-start]=np.arange(start,end)
parity={'queries':0,'region_pairs':0,'F_max_abs_error':0.,'handcrafted_max_abs_error':0.,'map_scores_max_abs_error':0.}

with torch.inference_mode():
    for begin in range(0,len(maps),64):
        finish=min(begin+64,len(maps))
        left,right=int(offsets[begin]),int(offsets[finish])
        cos=(H[left:right]@H.T).clamp(0,1)
        distance=torch.cdist(D[left:right],D,p=2,compute_mode='donot_use_mm_for_euclid_dist')
        hand=torch.exp(-distance)
        mask=owner_tensor[left:right,None]!=owner_tensor[None,:]
        gpu_add(regional['F'],cos[mask])
        gpu_add(regional['Handcrafted'],hand[mask])
        fblock,hblock=cos.cpu().numpy(),hand.cpu().numpy()
        for i in range(begin,finish):
            start,end=offsets[i:i+2]
            fs,hs=fblock[start-left:end-left],hblock[start-left:end-left]
            fscore=batched_map_scores(fs,padded,area[start:end],settings)
            hscore=batched_map_scores(hs,padded,area[start:end],settings)
            if i<16:
                fref=np.clip(h[start:end]@h.T,0,1)
                href=np.exp(-np.linalg.norm(d[start:end,None]-d[None],axis=-1))
                np.testing.assert_allclose(fs,fref,atol=1e-12,rtol=1e-12)
                np.testing.assert_allclose(hs,href,atol=1e-12,rtol=1e-12)
                parity['F_max_abs_error']=max(parity['F_max_abs_error'],float(np.max(np.abs(fs-fref))))
                parity['handcrafted_max_abs_error']=max(parity['handcrafted_max_abs_error'],float(np.max(np.abs(hs-href))))
                for actual,ref in ((fscore,fref),(hscore,href)):
                    expected=batched_map_scores(ref,padded,area[start:end],settings)
                    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
                    parity['map_scores_max_abs_error']=max(parity['map_scores_max_abs_error'],float(np.max(np.abs(actual-expected))))
                parity['queries']+=1;parity['region_pairs']+=int(end-start)*len(h)
            keep=np.arange(len(maps))!=i
            exact=np.all(labels==labels[i],axis=1)
            full=(labels@labels[i])==labels[i].sum()
            masks={'all':keep,'exact':keep&exact,'not_exact':keep&~exact,
                   'full_coverage':keep&full,'not_full_coverage':keep&~full}
            for name,values in (('F',fscore),('Handcrafted',hscore)):
                for group,select in masks.items():cpu_add(whole[name][group],values[select])
        if finish%512==0 or finish==len(maps):
            print(f'Distributions: {finish}/{len(maps)} query wafers',flush=True)

expected_regions=int(len(h)**2-np.square(np.diff(offsets)).sum())
expected_maps=len(maps)*(len(maps)-1)
assert expected_regions==813105740 and expected_maps==51818402
for method in regional:
    assert regional[method]['count']==regional[method]['counts'].sum()==expected_regions
    assert whole[method]['all']['count']==whole[method]['all']['counts'].sum()==expected_maps
    for pos,neg in (('exact','not_exact'),('full_coverage','not_full_coverage')):
        np.testing.assert_array_equal(whole[method][pos]['counts']+whole[method][neg]['counts'],whole[method]['all']['counts'])
assert regional['F']['counts'][:500].sum()==12

def summarize(acc):
    count=acc['count'];mean=acc['sum']/count
    cum=np.cumsum(acc['counts'])
    quantiles={}
    for q in (.01,.05,.25,.5,.75,.95,.99):
        j=min(int(np.searchsorted(cum,q*count,side='left')),BINS-1)
        quantiles[str(q)]=[float(edges[j]),float(edges[j+1])]
    return {'count':count,'mean':mean,'std':float(np.sqrt(max(0,acc['sumsq']/count-mean**2))),
            'min':acc['min'],'max':acc['max'],'quantile_bin_intervals':quantiles,
            'fraction_below_05':int(acc['counts'][:500].sum())/count,
            'fraction_at_least_09':int(acc['counts'][900:].sum())/count}
summary={'region':{k:summarize(v) for k,v in regional.items()},
         'whole_map':{k:{g:summarize(v) for g,v in groups.items()} for k,groups in whole.items()},
         'quantile_note':'Quantiles are enclosed by width-0.001 histogram bins; moments use all raw scores.',
         'parity_verification':parity,'seconds':time.monotonic()-started}
histograms={'edges':edges}
histograms.update({f'region_{k}':v['counts'] for k,v in regional.items()})
histograms.update({f'map_{k}_{g}':v['counts'] for k,groups in whole.items() for g,v in groups.items()})
np.savez_compressed(output/'histograms.npz',**histograms)
write_json(output/'statistics.json',summary)
for name,state in snapshot.items():
    f=Path(name)
    assert file_hash(f)==state['sha256'] and f.stat().st_mtime_ns==state['mtime_ns'],name
write_json(output/'completed.json',{'success':True,'training_performed':False,'source_files_unchanged':True,
    'region_pairs_per_method':expected_regions,'map_pairs_per_method':expected_maps,
    'statistics_sha256':file_hash(output/'statistics.json'),'histograms_sha256':file_hash(output/'histograms.npz')})
print(json.dumps(summary,indent=2),flush=True)
