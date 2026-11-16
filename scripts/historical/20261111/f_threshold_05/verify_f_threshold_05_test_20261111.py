"""Independently recalculate handcrafted test metrics from frozen labels and saved rankings."""
import argparse,csv,hashlib,itertools,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--data',required=True);p.add_argument('--protocol',required=True);a=p.parse_args()
r=Path(a.run);protocol=Path(a.protocol);dest=r/'test'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def rows(p):
 with p.open() as f:return list(csv.DictReader(f))
marker=json.loads((r/'completed.json').read_text());assert marker['smoke'] is False
for name,h in marker['files'].items():assert sha(r/name)==h,name
queries=[int(x['sample_id']) for x in rows(protocol/'test_queries.csv')]
gallery=np.array([int(x['sample_id']) for x in rows(protocol/'test_gallery.csv')]);gallery_set=set(gallery)
assert len(queries)==len(gallery)==7199
with np.load(a.data) as z:labels=z['arr_1'].astype(int)
exact={int(x['query_id']):x for x in rows(dest/'per_query.csv')};coverage={int(x['query_id']):x for x in rows(dest/'coverage_per_query.csv')}
assert set(exact)==set(coverage)==set(queries)
ks=[1,5,10,20,50,100,500];seen=[]
def check(row,key,v):
 if v is None:assert row[key]==''
 else:np.testing.assert_allclose(float(row[key]),v,rtol=1e-10,atol=1e-12,err_msg=key)
with (dest/'rankings.csv').open() as f:
 for q,group in itertools.groupby(csv.DictReader(f),key=lambda x:int(x['query_id'])):
  rank=list(group);ids=np.array([int(x['candidate_id']) for x in rank]);scores=np.array([float(x['similarity_score']) for x in rank])
  assert len(rank)==len(set(ids))==500 and q not in ids and set(ids)<=gallery_set
  assert [int(x['rank']) for x in rank]==list(range(1,501))
  assert np.isfinite(scores).all() and np.all(scores[:-1]>=scores[1:]) and scores.min()>=0 and scores.max()<=1
  ties=np.array([int.from_bytes(hashlib.sha256(f'exact_v1:tie:2026:{q}:{i}'.encode()).digest()[:8],'big') for i in ids],dtype=np.uint64)
  np.testing.assert_array_equal(np.lexsort((ids,ties,-scores)),np.arange(500))
  candidates=gallery[gallery!=q];assert len(candidates)==7198
  assert int(exact[q]['pool_size'])==int(coverage[q]['pool_size'])==7198
  rel=np.all(labels[ids]==labels[q],axis=1);R=np.all(labels[candidates]==labels[q],axis=1).sum()
  cov=labels[ids]@labels[q]/labels[q].sum();full=labels[candidates]@labels[q]/labels[q].sum()
  ideal=np.sort(full)[::-1];flags=cov==1;full_R=(full==1).sum()
  for k in ks:
   hits=rel[:k].sum();pr=np.cumsum(rel[:k])/np.arange(1,k+1)
   for key,v in [('Hit',float(hits>0)),('Precision',hits/k),('Recall',hits/R if R else None),('AP',(pr*rel[:k]).sum()/R if R else None)]:check(exact[q],f'{key}@{k}',v)
   discount=1/np.log2(np.arange(2,k+2));idcg=ideal[:k]@discount
   fh=flags[:k].sum();fp=np.cumsum(flags[:k])/np.arange(1,k+1)
   for key,v in [('Coverage-nDCG',cov[:k]@discount/idcg if idcg else None),('MeanCoverage',cov[:k].mean()),('FullCoverageHit',float(fh>0)),('FullCoveragePrecision',fh/k),('FullCoverageRecall',fh/full_R if full_R else None),('FullCoverageAP',(fp*flags[:k]).sum()/full_R if full_R else None)]:check(coverage[q],f'{key}@{k}',v)
  seen.append(q)
  if len(seen)%1000==0:print(f'Independent verification: {len(seen)}/7199',flush=True)
assert seen==queries
for filename,records in [('metrics.json',exact),('coverage_metrics.json',coverage)]:
 j=json.loads((dest/filename).read_text())
 for key,v in j['metrics'].items():
  field=key.replace('FullCoverage_mAP','FullCoverageAP').replace('mAP','AP')
  vals=[float(x[field]) for x in records.values() if x[field]!='']
  np.testing.assert_allclose(v,np.mean(vals),rtol=1e-10,atol=1e-12)
report={'queries':len(seen),'gallery':len(gallery),'candidates_per_query':7198,'ranking_rows':len(seen)*500,
 'all_report_hashes_verified':True,'exact_and_coverage_per_query_and_aggregate_recomputed':True,'self_exclusion_and_sha_ties_verified':True}
with (r/'independent_verification.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report,indent=2),flush=True)
