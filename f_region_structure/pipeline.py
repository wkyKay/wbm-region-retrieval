"""Shared region-pair computation, unchanged greedy matching, full-pool evaluation."""
from contextlib import ExitStack
from concurrent.futures import ProcessPoolExecutor
import csv
from itertools import groupby
import multiprocessing
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.proposed.f_segmentation_geometry.coverage_metrics import evaluate_rankings_file
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from wbm_retrieval.diagnostics import batched_map_scores
from wbm_retrieval.exact_metrics import evaluate_rankings, order_scores
from . import scoring, verify
from .state import identity, record, seal, verified

KS = [1,5,10,20,50,100,500]


def bundle_for(labels, protocol):
    _, classes = np.unique(labels,axis=0,return_inverse=True)
    return SimpleNamespace(labels=labels,class_ids=classes,valid_ids=protocol.ids['valid'],
                           reserved_ids=protocol.ids['test'])


class Provider:
    def __init__(self, cache, pool, matching, learned=None):
        self.cache, self.pool, self.matching = cache,pool,matching
        self.learned = learned or {}
        self.engine = scoring.RegionPairScorer(cache['features'],cache,cache,learned_features=self.learned)
        self.positions = {int(s):i for i,s in enumerate(cache['map_ids'])}
        if set(self.positions) != set(map(int,pool.gallery)):
            raise ValueError('Features must cover exactly the full fixed gallery')
        self.padded = np.full((len(self.positions),4),-1,dtype=np.int64)
        for i,(a,b) in enumerate(zip(cache['offsets'][:-1],cache['offsets'][1:])):
            self.padded[i,:b-a] = np.arange(a,b)

    def tokens(self,query):
        i = self.positions[int(query)]
        return np.arange(self.cache['offsets'][i],self.cache['offsets'][i+1])

    def parts(self,query,specs=None):
        return self.engine.components(self.tokens(query),specs=specs)

    def score(self,query,spec,parts=None):
        tokens = self.tokens(query)
        candidates = np.asarray(self.pool[query],dtype=np.int64)
        pair = self.engine.score_components(self.parts(query,{'selected':spec}) if parts is None else parts,spec)
        candidate_rows = self.padded[[self.positions[int(c)] for c in candidates]]
        result = batched_map_scores(pair,candidate_rows,self.cache['area'][tokens],self.matching)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite region aggregation')
        return candidates,result


def _scalar_greedy(pair,areas):
    if pair.size == 0:
        return 0.
    pairs = sorted(((float(pair[i,j]),i,j) for i in range(pair.shape[0]) for j in range(pair.shape[1])),
                   key=lambda v:(-v[0],v[1],v[2]))
    usedq,usedc,total = set(),set(),0.
    weights = np.sqrt(np.asarray(areas,dtype=float))
    for value,i,j in pairs:
        if value>0 and i not in usedq and j not in usedc:
            usedq.add(i); usedc.add(j)
            total += value*weights[i]
    return total/weights.sum() if len(weights) else 0.


def audit_formulas(provider,specs,bundle,output):
    """Independent pair formulas and scalar greedy over 20 candidates per query class."""
    representatives = {}
    for q in provider.pool:
        representatives.setdefault(int(bundle.class_ids[q]),int(q))
    maximum = 0.
    count = 0
    cache = provider.cache
    for query in representatives.values():
        candidates = np.asarray(provider.pool[query])[:20]
        qt = provider.tokens(query)
        ct = np.concatenate([provider.tokens(c) for c in candidates])
        selected = np.concatenate([qt,ct])
        features = np.asarray(cache['features'][selected])
        geo = {k:cache[k][selected] for k in ('histograms','relative_area')}
        desc = {k:cache[k][selected] for k in ('joint','joint_all')}
        learned = {k:v[selected] for k,v in provider.learned.items()}
        parts = provider.parts(query,specs)
        for name,spec in specs.items():
            direct = scoring.independent_pair_scores(np.arange(len(qt)),features,geo,desc,spec,
                                                     learned_features=learned)
            production = provider.engine.score_components(parts,spec)[:,selected]
            error = float(np.max(np.abs(direct-production))) if direct.size else 0.
            if error>2e-6:
                raise ValueError(f'Independent region formula mismatch {name}/{query}: {error}')
            all_ids,all_scores = provider.score(query,spec,parts)
            lookup = {int(c):float(s) for c,s in zip(all_ids,all_scores)}
            start = len(qt)
            for candidate in candidates:
                width = len(provider.tokens(candidate))
                # Check matching against the exact production matrix. Independent
                # dot/GEMM rounding may otherwise flip nearly tied region pairs.
                value = _scalar_greedy(production[:,start:start+width],cache['area'][qt])
                start += width
                error = max(error,abs(value-lookup[int(candidate)]))
                if abs(value-lookup[int(candidate)])>2e-6:
                    raise ValueError('Independent greedy aggregation mismatch')
                count += 1
            maximum = max(maximum,error)
    record(Path(output)/f'formula_audit_{provider.pool.split_name}.json', {
        'status':'verified','query_ids':list(representatives.values()),'candidate_pairs':count,
        'max_absolute_error':maximum,'scope':'independent region formulas within tolerance; scalar greedy on identical production matrix; first 20 fixed candidates per class representative'})


def _ranking_rows(path):
    with Path(path).open(newline='') as handle:
        for query,rows in groupby(csv.DictReader(handle),key=lambda r:int(r['query_id'])):
            yield query,np.array([int(row['candidate_id']) for row in rows],dtype=np.int64)


def report(folder,pool,bundle,ks,metadata):
    evaluate_rankings(bundle,pool,_ranking_rows(folder/'rankings.csv'),ks,folder,
                      split_name=pool.split_name,metadata=metadata)
    evaluate_rankings_file(bundle,pool,folder/'rankings.csv',ks,folder,
                           split_name=pool.split_name,metadata=metadata)


def evaluate(output,cache,pool,bundle,matching,context_signature,specs,learned=None):
    output = Path(output)
    split = pool.split_name
    grid = scoring.frozen_grid()
    if split not in ('valid','test') or not specs or any(name not in grid or spec!=grid[name] for name,spec in specs.items()):
        raise ValueError('Invalid split or scientific score specifications')
    if (split=='valid' and specs!=grid) or (split=='test' and len(specs)!=1):
        raise ValueError('Valid requires the complete grid; test permits only one frozen winner')
    provider = Provider(cache,pool,matching,learned)
    audit_path = output/f'formula_audit_{split}.json'
    if not audit_path.exists():
        audit_formulas(provider,specs,bundle,output)
    signatures,folders,pending = {},{},{}
    for name,spec in specs.items():
        signature = digest({'context':context_signature,'split':split,'variant':name,'spec':spec})
        signatures[name] = signature
        folder = output/split/name
        folders[name] = folder
        if verified(folder,signature):
            continue
        if folder.exists() and any(folder.iterdir()):
            interrupted = output/'interrupted_evaluations'
            interrupted.mkdir(exist_ok=True)
            preserved = Path(tempfile.mkdtemp(prefix=split+'__'+name+'__',dir=interrupted))
            folder.rename(preserved/'evaluation')
        folder.mkdir(parents=True,exist_ok=True)
        run_identity = identity(output,split,name,signature)
        manifest = {**run_identity,'signature':signature,'spec':spec,'ks':KS,'tie_seed':2026,
                    'retrieval_protocol_signature':pool.protocol_signature,'num_queries':len(pool),
                    'gallery_count':len(pool.gallery),'candidate_rule':'all_gallery_except_query',
                    'ranking_scope':'top_max_k','posthoc_research':True,'test_parameter_search':False,
                    'checkpoint_kind':'best_nce','source_seed':2026,'matching':'original_F_greedy_sqrt_area',
                    'training_required':spec['family'] in ('plain_mlp','gated_mlp')}
        record(folder/'run_identity.json',run_identity)
        record(folder/'evaluation_manifest.json',manifest)
        pending[name] = spec
    if pending:
        with ExitStack() as stack:
            writers = {}
            for name in pending:
                # An interrupted, unsealed ranking is regenerated; completed rankings are immutable.
                writer = csv.writer(stack.enter_context((folders[name]/'rankings.csv').open('w',newline='')))
                writer.writerow(['query_id','rank','candidate_id','similarity_score'])
                writers[name] = writer
            for number,query in enumerate(pool,1):
                parts = provider.parts(query,pending)
                for name,spec in pending.items():
                    candidates,scores = provider.score(query,spec,parts)
                    order = order_scores(candidates,scores,int(query),2026)[:max(KS)]
                    writers[name].writerows((int(query),rank,int(candidates[i]),float(scores[i]))
                                           for rank,i in enumerate(order,1))
                if number%100==0 or number==len(pool):
                    print(f'{split} rankings {number}/{len(pool)}; {len(pending)} variants',flush=True)
                    write_json(output/'status.json',{'status':'running','phase':'rankings','split':split,'done':number,'total':len(pool)})
        for name,spec in pending.items():
            folder = folders[name]
            metadata = read_json(folder/'evaluation_manifest.json')
            report(folder,pool,bundle,KS,metadata)
            checked = verify.verify_evaluation(folder,pool,bundle,KS,signatures[name],
                       lambda q,s=spec:provider.score(q,s),tie_seed=2026)
            record(folder/'verification.json',checked)
            seal(folder,signatures[name])
            metrics = read_json(folder/'metrics.json')['metrics']
            coverage = read_json(folder/'coverage_metrics.json')['metrics']
            print(f"{split} {name}: P@10={100*metrics['Precision@10']:.4f}% "
                  f"mAP@100={100*metrics['mAP@100']:.4f}% C-nDCG@10={100*coverage['Coverage-nDCG@10']:.4f}%",flush=True)
    results = {}
    for name,folder in folders.items():
        checked = read_json(folder/'verification.json')
        results[name] = {'top10_hits':int(checked['top10_hits']),'num_queries':len(pool),
                         'metrics':read_json(folder/'metrics.json')['metrics'],
                         'coverage_metrics':read_json(folder/'coverage_metrics.json')['metrics'],
                         'completed_sha256':file_hash(folder/'completed.json')}
    record(output/f'evaluation_index_{split}.json',{'split':split,'specs':specs,'results':results,
                                                  'context_signature':context_signature})
    return results
