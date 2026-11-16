"""Independent all-query metric checks for six valid runs and the selected test."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
for n in ('run','data','protocol'):
    p.add_argument('--'+n,required=True)
a=p.parse_args()
r,protocol=Path(a.run),Path(a.protocol)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):
            h.update(chunk)
    return h.hexdigest()
def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))
def read(path):
    return json.loads(path.read_text())
marker=read(r/'completed.json')
assert marker['smoke'] is False and marker['query_count']==7199
for name,h in marker['files'].items():
    assert sha(r/name)==h,name
with np.load(a.data) as z:
    labels=z['arr_1'].astype(int)
ks=[1,5,10,20,50,100,500]
def check(row,key,value):
    if value is None:
        assert row[key]==''
    else:
        np.testing.assert_allclose(float(row[key]),value,atol=1e-12,rtol=1e-10,err_msg=key)
def verify(dest,split):
    queries=[int(x['sample_id']) for x in rows(protocol/f'{split}_queries.csv')]
    gallery=np.array([int(x['sample_id']) for x in rows(protocol/f'{split}_gallery.csv')])
    gallery_set=set(gallery)
    assert (len(queries),len(gallery))==((500,3599) if split=='valid' else (7199,7199))
    exact={int(x['query_id']):x for x in rows(dest/'per_query.csv')}
    coverage={int(x['query_id']):x for x in rows(dest/'coverage_per_query.csv')}
    assert set(exact)==set(coverage)==set(queries)
    seen=[]
    with (dest/'rankings.csv').open() as f:
        for q,group in itertools.groupby(csv.DictReader(f),key=lambda x:int(x['query_id'])):
            rank=list(group)
            ids=np.array([int(x['candidate_id']) for x in rank])
            scores=np.array([float(x['similarity_score']) for x in rank])
            assert len(rank)==len(set(ids))==500 and q not in ids and set(ids)<=gallery_set
            assert [int(x['rank']) for x in rank]==list(range(1,501))
            assert np.isfinite(scores).all() and np.all(scores[:-1]>=scores[1:])
            assert scores.min()>=0 and scores.max()<=1
            ties=np.array([int.from_bytes(hashlib.sha256(f'exact_v1:tie:2026:{q}:{i}'.encode()).digest()[:8],'big') for i in ids],dtype=np.uint64)
            np.testing.assert_array_equal(np.lexsort((ids,ties,-scores)),np.arange(500))
            candidates=gallery[gallery!=q]
            assert int(exact[q]['pool_size'])==int(coverage[q]['pool_size'])==len(gallery)-1
            rel=np.all(labels[ids]==labels[q],axis=1)
            count=np.all(labels[candidates]==labels[q],axis=1).sum()
            cov=labels[ids]@labels[q]/labels[q].sum()
            full=labels[candidates]@labels[q]/labels[q].sum()
            ideal=np.sort(full)[::-1]
            flags=cov==1
            full_count=(full==1).sum()
            for k in ks:
                hits=rel[:k].sum()
                pr=np.cumsum(rel[:k])/np.arange(1,k+1)
                for key,v in [('Hit',float(hits>0)),('Precision',hits/k),('Recall',hits/count if count else None),
                              ('AP',(pr*rel[:k]).sum()/count if count else None)]:
                    check(exact[q],f'{key}@{k}',v)
                discount=1/np.log2(np.arange(2,k+2))
                idcg=ideal[:k]@discount
                fh=flags[:k].sum()
                fp=np.cumsum(flags[:k])/np.arange(1,k+1)
                for key,v in [('Coverage-nDCG',cov[:k]@discount/idcg if idcg else None),
                    ('MeanCoverage',cov[:k].mean()),('FullCoverageHit',float(fh>0)),('FullCoveragePrecision',fh/k),
                    ('FullCoverageRecall',fh/full_count if full_count else None),
                    ('FullCoverageAP',(fp*flags[:k]).sum()/full_count if full_count else None)]:
                    check(coverage[q],f'{key}@{k}',v)
            seen.append(q)
            if len(seen)%1000==0:
                print(f'Verifying {dest.name}/{split}: {len(seen)}/{len(queries)}',flush=True)
    assert seen==queries
    for filename,records in [('metrics.json',exact),('coverage_metrics.json',coverage)]:
        for key,v in read(dest/filename)['metrics'].items():
            field=key.replace('FullCoverage_mAP','FullCoverageAP').replace('mAP','AP')
            values=[float(x[field]) for x in records.values() if x[field]!='']
            np.testing.assert_allclose(v,np.mean(values),atol=1e-12,rtol=1e-10)
    result={'directory':str(dest.relative_to(r)),'split':split,'queries':len(seen),'gallery':len(gallery),
            'candidates_per_query':len(gallery)-1,'ranking_rows':len(seen)*500}
    print('Verified '+json.dumps(result),flush=True)
    return result
selection=read(r/'selection_before_test.json')
assert sha(r/'selection_before_test.json')==marker['selection_sha256']
best=max(selection['grid_results'],key=lambda x:(x['Coverage-nDCG@10'],-x['threshold']))
assert best['threshold']==selection['selected_threshold']==marker['selected_threshold']
reports=[]
for result in selection['grid_results']:
    path=r/result['directory']
    assert sha(path/'coverage_metrics.json')==result['coverage_metrics_sha256']
    reports.append(verify(path,'valid'))
reports.append(verify(r/'test','test'))
summary={'queries':7199,'total_queries_checked':sum(x['queries'] for x in reports),
         'ranking_rows_checked':sum(x['ranking_rows'] for x in reports),'reports':reports,
         'all_report_hashes_verified':True,'exact_and_coverage_per_query_and_aggregate_recomputed':True,
         'self_exclusion_and_sha_ties_verified':True,'valid_selection_recomputed':True,
         'selected_threshold':selection['selected_threshold']}
with (r/'independent_verification.json').open('x') as f:
    json.dump(summary,f,indent=2)
print(json.dumps(summary,indent=2),flush=True)
