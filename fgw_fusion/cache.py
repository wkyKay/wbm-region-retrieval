"""Resumable disk-backed full-candidate W scores; exact symmetry, no shortlist."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing as mp
from pathlib import Path
import tempfile
import time
import numpy as np
from wbm_retrieval.common import digest,file_hash,read_json,write_json

_BACKEND=None
_GALLERY=None

def _job(task):
    q,candidates,audit=task
    started=time.perf_counter()
    scores=_BACKEND.scores(q,candidates)
    checked=None
    if audit:
        checked=_BACKEND.audit_pairs((q,int(c)) for c in _GALLERY if int(c)!=q)
    return q,np.asarray(candidates,dtype=np.int64),scores,checked,time.perf_counter()-started

def _ids(values):
    a=np.asarray(values)
    if a.ndim!=1 or a.dtype.kind not in 'iu' or len(set(map(int,a)))!=len(a):
        raise ValueError('Unique integer sample IDs required')
    return a.astype(np.int64,copy=False)

def _row_candidates(query,gallery,positions,symmetric):
    if symmetric:return gallery[positions[int(query)]+1:]
    return gallery[gallery!=query]

def _read_row(folder,signature,query,candidates):
    marker=read_json(folder/'completed.json')
    if (marker['signature']!=signature or marker['query_id']!=int(query)
            or marker['candidate_ids_digest']!=digest(list(map(int,candidates)))
            or file_hash(folder/'scores.npz')!=marker['scores_sha256']):
        raise ValueError('WBBS row cache signature/hash mismatch: '+str(folder))
    with np.load(folder/'scores.npz',allow_pickle=False) as z:ids,scores=z['candidate_ids'],z['scores']
    if (not np.array_equal(ids,candidates) or scores.shape!=ids.shape or not np.isfinite(scores).all()
            or np.any(scores<0) or np.any(scores>1)):
        raise ValueError('Invalid cached WBBS row')
    return scores,marker

def build_cache(folder,signature,pool,backend,*,workers=8,audit_ids=()):
    global _BACKEND,_GALLERY
    folder=Path(folder);gallery=_ids(pool.gallery);queries=_ids(list(pool));positions={int(q):i for i,q in enumerate(gallery)}
    if any(int(q) not in positions for q in queries) or pool.candidates_per_query!=len(gallery)-1:
        raise ValueError('Expected full-gallery except-self pool')
    symmetric=set(map(int,queries))==set(map(int,gallery))
    audit_ids=set(map(int,audit_ids))
    if not audit_ids<=set(map(int,queries)):raise ValueError('Audit queries outside frozen pool')
    plan={'signature':signature,'query_ids':list(map(int,queries)),'gallery_ids':list(map(int,gallery)),
        'symmetric_upper_triangle':symmetric,'audit_ids':sorted(audit_ids),'dtype':'float64','backend':'exact_ckdtree_native_tie_fallback_v1'}
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'plan.json').exists():
        if read_json(folder/'plan.json')!=plan:raise ValueError('WBBS cache scientific plan changed')
    else:write_json(folder/'plan.json',plan)
    if (folder/'completed.json').exists():
        marker=read_json(folder/'completed.json')
        if marker.get('status')!='completed' or marker['signature']!=signature or marker['plan_sha256']!=file_hash(folder/'plan.json'):
            raise ValueError('WBBS completed cache provenance mismatch')
        if file_hash(folder/'scores.npy')!=marker['scores_sha256']:raise ValueError('WBBS matrix hash mismatch')
        audits=marker.get('native_audits',{})
        if (set(audits)!=set(map(str,audit_ids)) or any(a.get('status')!='verified' or a.get('pairs_checked')!=len(gallery)-1 for a in audits.values())
                or marker.get('query_count')!=len(queries) or marker.get('gallery_count')!=len(gallery)
                or marker.get('represented_directed_pairs')!=len(queries)*(len(gallery)-1)):
            raise ValueError('Completed WBBS cache audit/identity incomplete')
        return CachedScores(folder),marker
    rows=folder/'rows';rows.mkdir(exist_ok=True);pending=[];audits={};row_hashes={}
    for q in queries:
        q=int(q);c=_row_candidates(q,gallery,positions,symmetric);target=rows/str(q)
        if target.exists():
            _,marker=_read_row(target,signature,q,c)
            if q in audit_ids:
                if not isinstance(marker.get('native_audit'),dict) or marker['native_audit'].get('status')!='verified' or marker['native_audit'].get('pairs_checked')!=len(gallery)-1:raise ValueError('Native audit missing in cached row')
                audits[str(q)]=marker['native_audit']
            row_hashes[str(q)]=file_hash(target/'completed.json')
        else:pending.append((q,c,q in audit_ids))
    print(f'WBBS {pool.split_name}: {len(queries)-len(pending)}/{len(queries)} cached rows, workers={workers}',flush=True)
    _BACKEND,_GALLERY=backend,gallery
    start=time.perf_counter();done=len(queries)-len(pending)
    def persist(result):
        nonlocal done
        q,c,scores,audit,seconds=result
        if scores.shape!=c.shape or not np.isfinite(scores).all() or np.any(scores<0) or np.any(scores>1):
            raise ValueError('Invalid fresh WBBS scores')
        attempt=Path(tempfile.mkdtemp(prefix=f'.{q}_',dir=rows))
        with (attempt/'scores.npz').open('xb') as f:np.savez(f,candidate_ids=c,scores=scores)
        marker={'signature':signature,'query_id':q,'candidate_ids_digest':digest(list(map(int,c))),
                'scores_sha256':file_hash(attempt/'scores.npz'),'native_audit':audit,'seconds':seconds}
        write_json(attempt/'completed.json',marker);attempt.rename(rows/str(q));done+=1
        row_hashes[str(q)]=file_hash(rows/str(q)/'completed.json')
        if audit is not None:audits[str(q)]=audit
        write_json(folder/'status.json',{'status':'running','rows_done':done,'rows_total':len(queries),
            'native_audits_done':len(audits),'elapsed_seconds':time.perf_counter()-start})
        if done%50==0 or done==len(queries):
            print(f'WBBS {pool.split_name}: {done}/{len(queries)} rows; native audits={len(audits)}; elapsed={time.perf_counter()-start:.1f}s',flush=True)
    if workers==1:
        for task in pending:persist(_job(task))
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork')) as executor:
            tasks=iter(pending);futures=set()
            for _ in range(min(2*workers,len(pending))):futures.add(executor.submit(_job,next(tasks)))
            while futures:
                finished,futures=wait(futures,return_when=FIRST_COMPLETED)
                for future in finished:
                    persist(future.result())
                    task=next(tasks,None)
                    if task is not None:futures.add(executor.submit(_job,task))
    if set(audits)!=set(map(str,audit_ids)) or any(a['status']!='verified' for a in audits.values()):
        raise ValueError('All specified full-gallery native W audits must pass')
    temp=Path(tempfile.mkdtemp(prefix='.matrix_',dir=folder))/'scores.npy'
    matrix=np.lib.format.open_memmap(temp,mode='w+',dtype=np.float64,shape=(len(queries),len(gallery)))
    matrix[:]=np.nan;qpositions={int(q):i for i,q in enumerate(queries)}
    for q in queries:
        q=int(q);c=_row_candidates(q,gallery,positions,symmetric);s,marker=_read_row(rows/str(q),signature,q,c)
        ix=np.array([positions[int(x)] for x in c],dtype=int);matrix[qpositions[q],ix]=s
        if symmetric:
            qi=np.array([qpositions[int(x)] for x in c],dtype=int);matrix[qi,positions[q]]=s
    for q in queries:
        line=np.asarray(matrix[qpositions[int(q)]]);mask=gallery!=q
        if not np.isfinite(line[mask]).all() or not np.isnan(line[~mask]).all():raise ValueError('Incomplete WBBS matrix')
    matrix.flush();del matrix
    if (folder/'scores.npy').exists():
        orphan=Path(tempfile.mkdtemp(prefix='.preserved_unsealed_matrix_',dir=folder))
        (folder/'scores.npy').rename(orphan/'scores.npy')
        write_json(orphan/'reason.json',{'reason':'matrix published before completion seal; preserved during verified-row recovery'})
    temp.rename(folder/'scores.npy')
    marker={'status':'completed','signature':signature,'plan_sha256':file_hash(folder/'plan.json'),
        'scores_sha256':file_hash(folder/'scores.npy'),'query_count':len(queries),'gallery_count':len(gallery),
        'candidate_count':len(gallery)-1,'scored_pairs':sum(len(_row_candidates(int(q),gallery,positions,symmetric)) for q in queries),
        'represented_directed_pairs':len(queries)*(len(gallery)-1),'native_audits':audits,'row_marker_hashes':row_hashes,
        'seconds_this_attempt':time.perf_counter()-start,'symmetry':'WBBS raw score is symmetric; F/G are scored separately per query'}
    write_json(folder/'completed.json',marker);write_json(folder/'status.json',{'status':'completed','rows_done':len(queries)})
    return CachedScores(folder),marker

class CachedScores:
    def __init__(self,folder):
        self.folder=Path(folder);plan=read_json(self.folder/'plan.json')
        self.rows={int(q):i for i,q in enumerate(plan['query_ids'])}
        self.columns={int(q):i for i,q in enumerate(plan['gallery_ids'])}
        self.matrix=np.load(self.folder/'scores.npy',mmap_mode='r',allow_pickle=False)
        if self.matrix.shape!=(len(self.rows),len(self.columns)) or self.matrix.dtype!=np.float64:
            raise ValueError('WBBS matrix shape/dtype differs from plan')
    def scores(self,query,candidates):
        c=_ids(candidates)
        if int(query) not in self.rows or int(query) in set(map(int,c)) or any(int(x) not in self.columns for x in c):
            raise ValueError('Invalid W cache query/candidates')
        values=np.asarray(self.matrix[self.rows[int(query)],[self.columns[int(x)] for x in c]])
        if not np.isfinite(values).all() or np.any(values<0) or np.any(values>1):raise ValueError('Invalid W cache values')
        return values
