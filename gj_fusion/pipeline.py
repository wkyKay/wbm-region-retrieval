"""Stream complete candidate scores and save verified Top-500 dual evaluations."""
from contextlib import ExitStack
import csv
from pathlib import Path
import tempfile
import time

import numpy as np

from b1_structure_fusion.scoring import StructuralDistances
from experiments.proposed.f_global_fusion import pipeline as fusion
from experiments.proposed.f_segmentation_geometry import pipeline as geo
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from wbm_retrieval.exact_metrics import evaluate_rankings, order_scores
from . import scoring, verify
from .state import identity, record

KS = [1, 5, 10, 20, 50, 100, 500]
FIELDS = ['query_id', 'rank', 'candidate_id', 'similarity_score', 'global_score', 'joint_distance']


class Components:
    """Use the original G arithmetic and frozen B1 distance backend, without F."""
    def __init__(self, data, device='cuda'):
        ids, self.features = data['global_ids'], data['global_features']
        self.rows = {int(sample): i for i, sample in enumerate(ids)}
        d = data['descriptors']
        self.distances = StructuralDistances(d['sample_ids'], d['joint128'], d['empty'], device)

    def __call__(self, query, candidates):
        rows = [self.rows[int(c)] for c in candidates]
        g = np.clip(self.features[rows] @ self.features[self.rows[int(query)]], 0, 1)
        d = self.distances.for_query(query, candidates)['J']
        return np.asarray(g, dtype=np.float64), np.asarray(d, dtype=np.float64)


def variant_signature(context_signature, split, name, spec):
    return digest(dict(context=context_signature, split=split, variant=name, spec=spec))


def evaluate(output, data, context_signature, specs, device='cuda', ks=KS):
    output = Path(output)
    pool, bundle = data['pool'], data['bundle']
    split = pool.split_name
    grid = scoring.frozen_grid()
    if split not in ('valid', 'test') or not specs or any(grid.get(k) != v for k,v in specs.items()):
        raise ValueError('Unknown split or non-frozen score specification')
    if split == 'valid' and specs != grid:
        raise ValueError('Valid must evaluate the entire eight-candidate grid')
    if max(ks) > pool.candidates_per_query:
        raise ValueError('K exceeds full candidate pool')
    parent = output / split
    parent.mkdir(parents=True, exist_ok=True)
    components = Components(data, device)
    audit = verify.audit_components(data, components)
    record(output/f'component_audit_{split}.json', dict(**audit,context_signature=context_signature))
    pending, signatures = [], {}
    for name, spec in specs.items():
        signature = variant_signature(context_signature, split, name, spec)
        signatures[name] = signature
        target = parent / name
        if geo.completed(target, signature):
            continue
        ident = identity(output, split, name, signature)
        attempt = Path(tempfile.mkdtemp(prefix='.'+name+'_attempt_', dir=parent))
        record(attempt/'run_identity.json', ident)
        record(attempt/'evaluation_manifest.json', dict(**ident, signature=signature,
            context_signature=context_signature, spec=spec, ks=list(ks), tie_seed=2026,
            retrieval_protocol_signature=pool.protocol_signature,
            num_queries=len(pool), gallery_count=len(pool.gallery),
            candidate_rule='all_gallery_except_query', scoring_device=device,
            structure_strength=2., angular_component=False, local_component=False,
            training_required=False, posthoc_research=True, test_parameter_search=False,
            test_evaluated=split == 'test', source_seed=2026, checkpoint_kind='best_nce',
            score_scale='original G and exp(-2D_J); no calibration'))
        pending.append((name, spec, target, attempt, ident))
    if pending:
        started = time.perf_counter()
        gallery_set = set(map(int, pool.gallery))
        with ExitStack() as stack:
            writers = []
            for name, spec, _, attempt, _ in pending:
                writer = csv.writer(stack.enter_context((attempt/'rankings.csv').open('w', newline='')))
                writer.writerow(FIELDS)
                writers.append((name, spec, writer))
            for number, query in enumerate(pool, 1):
                candidates = np.asarray(pool[query])
                if len(candidates) != len(gallery_set)-1 or set(map(int,candidates)) != gallery_set-{int(query)}:
                    raise ValueError('Candidate identities differ from the full gallery except query')
                g, d = components(query, candidates)
                for name, spec, writer in writers:
                    scores = scoring.score_variant(g, d, spec)
                    order = order_scores(candidates, scores, int(query), 2026)[:max(ks)]
                    writer.writerows((int(query), rank, int(candidates[i]), float(scores[i]),
                                      float(g[i]), float(d[i])) for rank,i in enumerate(order,1))
                if number % 100 == 0 or number == len(pool):
                    print(f'G/J {split} full-gallery scores: {number}/{len(pool)}, {len(pending)} variants', flush=True)
                    write_json(output/'status.json', dict(status='running',split=split,phase='ranking',
                               queries_done=number,queries_total=len(pool)))
        ranking_seconds = time.perf_counter()-started
        for name, spec, target, attempt, ident in pending:
            print(f'G/J {split} metrics and independent verification: {name}', flush=True)
            write_json(output/'status.json',dict(status='running',split=split,phase='metrics_and_verify',variant=name))
            metadata = dict(signature=signatures[name], run_id=ident['run_id'], started_at=ident['started_at'],
                variant=name, spec=spec, candidates_scored='full_pool', ranking_scope='top_max_k',
                retrieval_protocol_signature=pool.protocol_signature, posthoc_research=True,
                test_parameter_search=False)
            began = time.perf_counter()
            evaluate_rankings(bundle,pool,fusion.iter_rankings(attempt/'rankings.csv'),ks,attempt,
                              split_name=split,metadata=metadata)
            geo.evaluate_rankings_file(bundle,pool,attempt/'rankings.csv',ks,attempt,
                                       split_name=split,metadata=metadata)
            checked = verify.verify_evaluation(attempt,pool,bundle,ks,signatures[name],components,tie_seed=2026)
            write_json(attempt/'verification.json', checked)
            write_json(attempt/'timing.json', dict(ranking_batch_wall_seconds=ranking_seconds,
                ranking_batch_variants=[p[0] for p in pending], ranking_time_is_shared_not_per_variant=True,
                metrics_and_verification_seconds=time.perf_counter()-began))
            geo.commit_marker(attempt,signatures[name])
            attempt.rename(target)
    results = {}
    for name, spec in specs.items():
        target = parent/name
        exact = read_json(target/'metrics.json')
        coverage = read_json(target/'coverage_metrics.json')
        checked = read_json(target/'verification.json')
        results[name] = dict(split=split,num_queries=len(pool),top10_hits=checked['top10_hits'],
            spec=spec, metrics=exact['metrics'], coverage_metrics=coverage['metrics'],
            signature=signatures[name],completed_sha256=file_hash(target/'completed.json'))
        ap = exact['metrics'].get('mAP@100')
        ap_text = 'NA' if ap is None else f'{100*ap:.4f}%'
        print(f"G/J {split}/{name}: P@10={100*exact['metrics']['Precision@10']:.4f}% "
              f"mAP@100={ap_text} "
              f"Coverage-nDCG@10={100*coverage['metrics']['Coverage-nDCG@10']:.4f}%",flush=True)
    record(output/f'evaluation_index_{split}.json', dict(context_signature=context_signature,
           split=split,variants=specs,results=results))
    return results
