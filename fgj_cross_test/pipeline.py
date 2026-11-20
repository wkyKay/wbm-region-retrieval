"""Stream six rankings, then verify all six in one shared full-candidate pass."""
from contextlib import ExitStack
import csv
from pathlib import Path
import tempfile
import time

import numpy as np
from experiments.proposed.f_segmentation_geometry import coverage_metrics
from wbm_retrieval.common import file_hash, read_json, write_json
from wbm_retrieval.exact_metrics import tie_keys
from . import scoring
from .state import (VARIANTS, SPECS, SETTINGS, PROTOCOL, CANDIDATE_RULE, identity,
                    variant_signature, record, seal, verified)

FIELDS = ['query_id','rank','candidate_id','similarity_score','optimal_f_score',
          'global_score','joint_distance','wbbs_score']


def evaluate(output, data, context_signature, provider, *, formal=True):
    from .verify import verify_evaluations
    output, pool, bundle = Path(output), data['pool'], data['bundle']
    ks = SETTINGS['ks']; seed = SETTINGS['tie_seed']
    if pool.split_name != 'test' or pool.candidates_per_query < max(ks):
        raise ValueError('Invalid cross test pool')
    if formal and (len(pool) != 7199 or len(pool.gallery) != 7199):
        raise ValueError('Formal evaluation requires all frozen test queries')
    parent = output/'test'; parent.mkdir(parents=True, exist_ok=True)
    signatures = {n: variant_signature(context_signature,n) for n in VARIANTS}
    finals = {n:parent/n for n in VARIANTS}
    # Finish an interrupted publication only when the entire verified batch is
    # available. Old incomplete attempts stay untouched for provenance.
    shared_path=output/'shared_verification.json'
    if shared_path.exists():
        shared=read_json(shared_path)
        recover={}
        for name in VARIANTS:
            if finals[name].exists():continue
            candidates=[]
            for p in parent.glob('.'+name+'_attempt_*'):
                if (p/'completed.json').exists() and verified(p,signatures[name]):
                    if read_json(p/'verification.json')==shared.get('variants',{}).get(name):
                        candidates.append(p)
            if len(candidates)>1:raise ValueError('Ambiguous sealed attempts; no automatic overwrite')
            if candidates:recover[name]=candidates[0]
        missing=[n for n in VARIANTS if not finals[n].exists()]
        if missing and set(recover)==set(missing):
            for name,p in recover.items():p.rename(finals[name])
    # A failed shared verification publishes no new children. A fully sealed
    # batch can be reused on a signature-identical resume.
    complete = [verified(finals[n],signatures[n]) for n in VARIANTS]
    if any(complete) and not all(complete):
        raise ValueError('Partially published batch requires explicit inspection; no overwrite')
    if not all(complete):
        folders = {}
        for name in VARIANTS:
            ident = identity(output,name,signatures[name])
            folder = Path(tempfile.mkdtemp(prefix='.'+name+'_attempt_',dir=parent))
            folders[name] = folder
            record(folder/'run_identity.json',ident)
            record(folder/'evaluation_manifest.json',dict(**ident,
                signature=signatures[name],context_signature=context_signature,
                spec=SPECS[name],settings=SETTINGS,ks=ks,tie_seed=seed,
                stage='test',protocol=PROTOCOL,candidate_rule=CANDIDATE_RULE,
                derived_protocol=pool.metadata,retrieval_protocol_signature=pool.protocol_signature,
                base_protocol_signature=pool.metadata['base_protocol_signature'],
                num_queries=len(pool),gallery_count=len(pool.gallery),
                formal_protocol=formal,structure_strength=2.0,f_version='cc8_optimal_radial_area',
                training_required=False,posthoc_research=True,test_parameter_search=False,
                selection_rule='user_specified_posthoc_components',primary_variant=None,
                normalization_scope='none',ranking_scope='top_max_k',
                source_binding_signature=data['source_binding_signature']))
        began = time.perf_counter()
        with ExitStack() as stack:
            writers = {}
            for name,folder in folders.items():
                w=csv.writer(stack.enter_context((folder/'rankings.csv').open('x',newline='')))
                w.writerow(FIELDS); writers[name]=w
            for number,q in enumerate(pool,1):
                q=int(q); candidates=np.asarray(pool[q])
                components=scoring.component_vectors(provider(q,candidates))
                scores=scoring.score_all(components)
                hashes=tie_keys(q,candidates,seed)
                for name in VARIANTS:
                    order=np.lexsort((candidates,hashes,-scores[name]))[:max(ks)]
                    writers[name].writerows((q,rank,int(candidates[i]),float(scores[name][i]),
                        float(components['F'][i]),float(components['G'][i]),
                        float(components['joint_distance'][i]),float(components['WBBS'][i]))
                        for rank,i in enumerate(order,1))
                if number%100==0 or number==len(pool):
                    print(f'Cross six-formula full-pool ranking: {number}/{len(pool)}',flush=True)
                    write_json(output/'status.json',dict(status='running',phase='ranking',
                        queries_done=number,queries_total=len(pool)))
        ranking_seconds=time.perf_counter()-began
        metrics_seconds={}
        for name,folder in folders.items():
            print('Original coverage_v1 metrics: '+name,flush=True)
            write_json(output/'status.json',dict(status='running',phase='metrics',variant=name))
            began=time.perf_counter()
            metadata=read_json(folder/'evaluation_manifest.json')
            metadata.update(candidates_scored='full_filtered_pool',smoke=False)
            coverage_metrics.evaluate_rankings_file(bundle,pool,folder/'rankings.csv',ks,folder,
                split_name='test',metadata=metadata)
            metrics_seconds[name]=time.perf_counter()-began
        print('Shared independent verification: all six formulas and all candidates',flush=True)
        write_json(output/'status.json',dict(status='running',phase='shared_verification'))
        began=time.perf_counter()
        checked=verify_evaluations(folders,pool,bundle,ks,signatures,provider,
                                  data['source_binding'],tie_seed=seed)
        verify_seconds=time.perf_counter()-began
        record(output/'shared_verification.json',checked)
        for name,folder in folders.items():
            record(folder/'verification.json',checked['variants'][name])
            record(folder/'timing.json',dict(ranking_batch_wall_seconds=ranking_seconds,
                ranking_batch_variants=VARIANTS,metrics_wall_seconds=metrics_seconds[name],
                shared_verification_wall_seconds=verify_seconds,
                batch_times_are_not_per_variant=True))
            seal(folder,signatures[name])
        for name,folder in folders.items():
            if finals[name].exists(): raise FileExistsError(finals[name])
            folder.rename(finals[name])
    results={}
    for name,folder in finals.items():
        report=read_json(folder/'coverage_metrics.json')
        results[name]=dict(coverage_metrics=report['metrics'],
            metric_query_counts=report['metric_query_counts'],signature=signatures[name],
            completed_sha256=file_hash(folder/'completed.json'))
        displayed=[]
        for key in ('Coverage-nDCG@10','MeanCoverage@10','FullCoveragePrecision@10','FullCoverage_mAP@100'):
            if key in report['metrics']:
                value=report['metrics'][key]
                displayed.append(key+'='+(f'{100*value:.4f}%' if value is not None else 'NA'))
        print(name+': '+', '.join(displayed),flush=True)
    record(output/'evaluation_index_test.json',dict(context_signature=context_signature,
        split='test',variants=VARIANTS,specs=SPECS,results=results))
    return results
