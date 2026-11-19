"""Read-only frozen F/G/WBBS inputs and independent branch checks."""
from pathlib import Path
import csv
import numpy as np
from b1_ablation import run as oldrun
from b1_ablation.verify import independent_order, SOURCE_ATOL
from experiments.proposed.f_global_fusion import pipeline as fusion
from wbm_retrieval.common import digest, file_hash, read_json

ROOT=Path(__file__).resolve().parents[1]
WBBS_BASE=ROOT.parent/'compared_methods/artifacts/formal_exact_20261109_120738/evaluation'
WBBS_RUNS={'valid':WBBS_BASE/'wbbs_valid','test':WBBS_BASE/'wbbs_test_r2'}
SOURCE=ROOT/'artificial/b1_structure_all_20261117T193244+0800'
REFERENCE=ROOT/'artificial/b1_ablation_20261118T150805+0800'
PROTECTED=[ROOT/'artificial'/n for n in ('b1_ablation_20261118T183806+0800','b1_ablation_20261118T190927+0800','b1_ablation_structure_pair_check_20261118T203238+0800')]
WBBS_PARAMETERS={'sigma_d':9.0,'noise_threshold':0.32,'alpha':5.0,'mu':1.0,'total_dies':2061,'block_size':256}

def load_wbbs(split,pool,source_context):
    from .wbbs_backend import Scorer
    folder=WBBS_RUNS[split]
    manifest=fusion.signed_manifest(folder/'evaluation_manifest.json',compared=True)
    done=read_json(folder/'completed.json')
    if (manifest.get('method')!='wbbs' or manifest.get('split')!=split or manifest.get('smoke') is not False
            or manifest.get('variant')!='global' or manifest.get('candidate_rule')!='all_gallery_except_query'
            or manifest.get('query_count')!=len(pool) or manifest.get('gallery_count')!=len(pool.gallery)
            or manifest.get('candidates_per_query')!=pool.candidates_per_query
            or done.get('evaluation_signature')!=manifest['signature'] or done.get('split')!=split):
        raise ValueError('WBBS source must be complete frozen full-gallery evaluation')
    expected={'data_file':source_context['sources']['data_file'],'split_file':source_context['sources']['split_manifest'],
        'retrieval_protocol':{k.split('/',1)[1]:v for k,v in source_context['sources'].items() if k.startswith('retrieval_protocol/')}}
    if manifest['sources']!=expected or manifest['settings']['parameters']!=WBBS_PARAMETERS:
        raise ValueError('WBBS data/protocol/parameters differ')
    original_code=ROOT.parent/'compared_methods/methods/wbbs/wbbs.py'
    if file_hash(original_code)!=manifest['code']['methods/wbbs/wbbs.py']:
        raise ValueError('Original WBBS scoring/descriptor implementation changed')
    fusion.checked_files(folder,done['files'])
    prepared=folder/split/'prepared'
    metadata=read_json(prepared/'prepared_metadata.json')
    if (metadata['parameters']!=WBBS_PARAMETERS or metadata['protocol_signature']!=pool.protocol_signature
            or metadata['files']['features.npz']!=file_hash(prepared/'features.npz')):
        raise ValueError('WBBS prepared provenance differs')
    with np.load(prepared/'features.npz',allow_pickle=False) as z: arrays=dict(z)
    if not np.array_equal(arrays['sample_ids'],pool.gallery):
        raise ValueError('WBBS gallery IDs/order differ from frozen pool')
    backend=Scorer(arrays['sample_ids'],arrays['offsets'],arrays['descriptors'],WBBS_PARAMETERS)
    provenance={'source_root':str(folder),'manifest_sha256':file_hash(folder/'evaluation_manifest.json'),
        'completed_sha256':file_hash(folder/'completed.json'),'features_sha256':file_hash(prepared/'features.npz'),
        'metadata_sha256':file_hash(prepared/'prepared_metadata.json'),'original_code_sha256':file_hash(original_code),
        'source_evaluation_signature':manifest['signature'],'settings':WBBS_PARAMETERS,'protocol_signature':pool.protocol_signature}
    return backend,arrays,provenance

def selected_queries(pool,labels):
    result={}
    for q in pool:
        label=tuple(map(int,labels[int(q)]))
        result[label]=min(int(q),result.get(label,int(q)))
    return sorted(result.values())

def independent_fg(query,candidates,scorer):
    """F region geometry rebuilt from cache, optimum solved by subset dynamic programming."""
    m=scorer.meta; qr=scorer.rows[int(query)]
    q=np.arange(m['offsets'][qr],m['offsets'][qr+1]); n=len(candidates)
    whole=np.clip(scorer.global_features[[scorer.global_rows[int(c)] for c in candidates]] @
                  scorer.global_features[scorer.global_rows[int(query)]],0,1)
    if not len(q):return np.zeros(n),whole
    cosine=np.clip(scorer.features[q] @ scorer.features.T,0,1)
    cdf=np.cumsum(m['histograms'],axis=1)
    radial=np.abs(cdf[q,None,:]-cdf[None,:,:]).sum(axis=2)/16
    area=.5*np.abs(np.log(m['relative_area'][q,None]/m['relative_area'][None,:]))
    pairs=cosine*np.exp(-radial)*(.85+.15*np.exp(-area/1.5))
    slots=scorer.padded[[scorer.rows[int(c)] for c in candidates]]
    matrix=pairs[:,np.maximum(slots,0)].transpose(1,0,2)*(slots>=0)[:,None,:]
    weights=np.sqrt(m['area'][q]); width=slots.shape[1]
    dp={0:np.zeros(n)}
    for qi in range(len(q)):
        nxt={mask:value.copy() for mask,value in dp.items()}
        for mask,value in dp.items():
            for cj in range(width):
                if not(mask>>cj)&1:
                    target=mask|(1<<cj); trial=value+weights[qi]*matrix[:,qi,cj]
                    nxt[target]=np.maximum(nxt[target],trial) if target in nxt else trial
        dp=nxt
    return np.maximum.reduce(list(dp.values()))/weights.sum(),whole

def audit_fg(pool,bundle,scorer,provider):
    """Per-class full-gallery F/G formula, assignment, fusion and exact Top500 order audit."""
    from .scoring import specs,score_variant
    audits=[]
    for q in selected_queries(pool,bundle.labels):
        c=np.asarray(pool[q]); f,g=independent_fg(q,c,scorer); values=scorer.components(q,c);w=provider.scores(q,c)
        np.testing.assert_allclose(f,values['optimal_f_score'],atol=SOURCE_ATOL,rtol=SOURCE_ATOL)
        np.testing.assert_allclose(g,values['global_score'],atol=SOURCE_ATOL,rtol=SOURCE_ATOL)
        for name,spec in specs().items():
            explicit=(spec['local_weight']*f+spec['global_weight']*g)*w
            actual=score_variant(values,w,spec)
            if not np.array_equal(c[independent_order(q,c,explicit)[:500]],c[independent_order(q,c,actual)[:500]]):
                raise ValueError(f'Independent full-gallery fused ordering differs: {q}/{name}')
        audits.append({'query_id':q,'candidate_count':len(c),'F_max_error':float(np.max(np.abs(f-values['optimal_f_score']))),
                       'G_max_error':float(np.max(np.abs(g-values['global_score']))),'all_three_top500_exact':True})
    return {'status':'verified','selection':'minimum original ID in each full-label combination',
        'query_count':len(audits),'queries':audits,'scope':'independent region-geometry formula and subset-DP optimum, frozen G dot product, full-gallery three-variant Top500; W native audit recorded separately'}

def verify_wbbs_replay(pool,provider):
    """Every new full-gallery W row must reproduce the old immutable W Top500 and scores."""
    from itertools import groupby
    path=WBBS_RUNS[pool.split_name]/pool.split_name/'rankings.csv'
    checked=0; maxerr=0.
    with path.open(newline='') as f:
        groups=groupby(csv.DictReader(f),key=lambda r:int(r['query_id']))
        for q in pool:
            sid,it=next(groups); rows=list(it);c=np.asarray(pool[q]);w=provider.scores(q,c)
            if sid!=int(q) or len(rows)!=500:raise ValueError('WBBS source query/rank order changed')
            ranked=np.array([int(r['candidate_id']) for r in rows])
            order=independent_order(int(q),c,w,2026)[:500]
            if not np.array_equal(c[order],ranked):raise ValueError(f'WBBS source Top500 mismatch: {q}')
            saved=np.array([float(r['similarity_score'] if 'similarity_score' in r else r['score']) for r in rows])
            err=float(np.max(np.abs(w[order]-saved)));maxerr=max(maxerr,err)
            if not np.array_equal(w[order],saved):raise ValueError(f'WBBS source score changed: {q}, {err}')
            checked+=1
        if next(groups,None) is not None:raise ValueError('Extra WBBS source query')
    return {'status':'verified','num_queries':checked,'candidates_per_query':pool.candidates_per_query,
        'top_k':500,'max_score_error':maxerr,'ranking_sha256':file_hash(path),
        'scope':'every query, exact saved Top500 identity/order and score equality after scoring all candidates'}

def inventory(source=SOURCE,reference=REFERENCE):
    files=oldrun.inventory(source,reference,PROTECTED)
    for folder in WBBS_RUNS.values():
        for p in sorted(folder.rglob('*')):
            if p.is_file():files[str(p)]={'sha256':file_hash(p),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
    for folder in (ROOT/'b1_ablation',ROOT.parent/'compared_methods/methods/wbbs',ROOT/'protocol',ROOT.parent/'compared_methods/protocol'):
        for p in sorted(folder.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                files[str(p)]={'sha256':file_hash(p),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
    return files
