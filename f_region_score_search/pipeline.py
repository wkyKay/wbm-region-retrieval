"""Same full-gallery matcher, independent checks, and isolated valid reports."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import csv
import multiprocessing
from pathlib import Path
import tempfile

import numpy as np

from f_region_structure.pipeline import Provider as OldProvider, _scalar_greedy, report
from f_region_structure.verify import verify_evaluation
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from wbm_retrieval.diagnostics import batched_map_scores
from wbm_retrieval.exact_metrics import order_scores
from . import scoring
from .state import identity, record, seal, verified

KS = [1, 5, 10, 20, 50, 100, 500]


class Provider(OldProvider):
    def __init__(self, cache, pool, matching):
        if pool.split_name != 'valid':
            raise ValueError('This search only permits valid')
        super().__init__(cache, pool, matching)

    def parts(self, query):
        return self.engine.components(self.tokens(query), specs={'selected': scoring.OLD_SPEC})

    def score(self, query, spec, parts=None):
        tokens = self.tokens(query)
        candidates = np.asarray(self.pool[query], dtype=np.int64)
        pair = scoring.score_components(self.parts(query) if parts is None else parts, spec)
        rows = self.padded[[self.positions[int(c)] for c in candidates]]
        scores = batched_map_scores(pair, rows, self.cache['area'][tokens], self.matching)
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite candidate score')
        return candidates, scores


def audit_formulas(provider, bundle, output, specs):
    representatives = {}
    for q in provider.pool:
        representatives.setdefault(int(bundle.class_ids[q]), int(q))
    maximum, count = 0., 0
    cache = provider.cache
    for q in representatives.values():
        candidates = np.asarray(provider.pool[q])[:20]
        qt = provider.tokens(q)
        ct = np.concatenate([provider.tokens(c) for c in candidates])
        selected = np.concatenate([qt, ct])
        parts = provider.parts(q)
        for spec in specs.values():
            actual = scoring.score_components(parts, spec)[:, selected]
            reference = scoring.independent_pair_scores(
                np.arange(len(qt)), np.asarray(cache['features'][selected]),
                cache['relative_area'][selected], cache['joint_all'][selected], spec)
            error = float(np.max(np.abs(actual-reference))) if actual.size else 0.
            if error > 2e-6:
                raise ValueError('Independent region-pair formula mismatch')
            ids, values = provider.score(q, spec, parts)
            lookup = dict(zip(map(int, ids), map(float, values)))
            start = len(qt)
            for c in candidates:
                width = len(provider.tokens(c))
                scalar = _scalar_greedy(actual[:, start:start+width], cache['area'][qt])
                start += width
                error = max(error, abs(scalar-lookup[int(c)]))
                if abs(scalar-lookup[int(c)]) > 2e-6:
                    raise ValueError('Independent scalar greedy mismatch')
                count += 1
            maximum = max(maximum, error)
    record(Path(output)/'formula_audit_valid.json', {
        'status': 'verified', 'query_ids': list(representatives.values()),
        'candidate_pairs': count, 'max_absolute_error': maximum,
        'scope': 'Independent region formulas; scalar greedy on identical production matrices; first 20 candidates per class representative.'})


_WORKER = None


def _initialize_worker(cache, pool, matching, bundle):
    global _WORKER
    _WORKER = (Provider(cache, pool, matching), bundle)


def _finish(task):
    folder, name, spec, signature, ks = task
    folder = Path(folder)
    provider, bundle = _WORKER
    metadata = read_json(folder/'evaluation_manifest.json')
    report(folder, provider.pool, bundle, ks, metadata)
    checked = verify_evaluation(folder, provider.pool, bundle, ks, signature,
                               lambda q: provider.score(q, spec), tie_seed=2026)
    record(folder/'verification.json', checked)
    seal(folder, signature)
    return name


def evaluate(output, cache, pool, bundle, matching, context_signature, workers=4):
    if pool.split_name != 'valid' or not 1 <= workers <= 8:
        raise ValueError('Only valid with 1..8 workers is supported')
    output = Path(output)
    specs = scoring.frozen_grid()
    provider = Provider(cache, pool, matching)
    if not (output/'formula_audit_valid.json').exists():
        audit_formulas(provider, bundle, output, specs)
    folders, signatures, pending = {}, {}, {}
    for name, spec in specs.items():
        signature = digest({'context': context_signature, 'split': 'valid', 'variant': name, 'spec': spec})
        folder = output/'valid'/name
        folders[name], signatures[name] = folder, signature
        if verified(folder, signature):
            continue
        if folder.exists() and any(folder.iterdir()):
            parent = output/'interrupted_evaluations'
            parent.mkdir(exist_ok=True)
            saved = Path(tempfile.mkdtemp(prefix=name+'_', dir=parent))
            folder.rename(saved/'evaluation')
        folder.mkdir(parents=True, exist_ok=True)
        run_identity = identity(output, name, signature)
        record(folder/'run_identity.json', run_identity)
        record(folder/'evaluation_manifest.json', {
            **run_identity, 'signature': signature, 'spec': spec, 'ks': KS, 'tie_seed': 2026,
            'retrieval_protocol_signature': pool.protocol_signature, 'num_queries': len(pool),
            'gallery_count': len(pool.gallery), 'candidate_rule': 'all_gallery_except_query',
            'ranking_scope': 'top_max_k', 'posthoc_research': True, 'test_parameter_search': False,
            'test_evaluated': False, 'training_required': False, 'checkpoint_kind': 'best_nce',
            'source_seed': 2026, 'matching': 'original_F_greedy_sqrt_area'})
        pending[name] = spec
    if pending:
        with ExitStack() as stack:
            writers = {}
            for name in pending:
                writers[name] = csv.writer(stack.enter_context((folders[name]/'rankings.csv').open('w', newline='')))
                writers[name].writerow(['query_id', 'rank', 'candidate_id', 'similarity_score'])
            for number, q in enumerate(pool, 1):
                parts = provider.parts(q)
                for name, spec in pending.items():
                    ids, scores = provider.score(q, spec, parts)
                    order = order_scores(ids, scores, int(q), 2026)[:max(KS)]
                    writers[name].writerows((int(q), rank, int(ids[i]), float(scores[i]))
                                            for rank, i in enumerate(order, 1))
                if number % 50 == 0 or number == len(pool):
                    print(f'valid rankings {number}/{len(pool)}; {len(pending)} variants', flush=True)
                    write_json(output/'status.json', {'status': 'running', 'phase': 'rankings',
                        'split': 'valid', 'done': number, 'total': len(pool), 'test_evaluated': False})
        tasks = [(str(folders[n]), n, s, signatures[n], KS) for n, s in pending.items()]
        if workers == 1:
            _initialize_worker(cache, pool, matching, bundle)
            finished = map(_finish, tasks)
            for n in finished:
                print('valid verified: '+n, flush=True)
        else:
            with ProcessPoolExecutor(max_workers=min(workers, len(tasks)),
                                     mp_context=multiprocessing.get_context('spawn'),
                                     initializer=_initialize_worker,
                                     initargs=(cache, pool, matching, bundle)) as executor:
                futures = [executor.submit(_finish, task) for task in tasks]
                for i, future in enumerate(as_completed(futures), 1):
                    name = future.result()
                    print(f'valid verified {i}/{len(tasks)}: {name}', flush=True)
                    write_json(output/'status.json', {'status': 'running', 'phase': 'verification',
                        'split': 'valid', 'done': i, 'total': len(tasks), 'test_evaluated': False})
    results = {}
    for name, folder in folders.items():
        checked = read_json(folder/'verification.json')
        results[name] = {'top10_hits': int(checked['top10_hits']), 'num_queries': len(pool),
                        'metrics': read_json(folder/'metrics.json')['metrics'],
                        'coverage_metrics': read_json(folder/'coverage_metrics.json')['metrics'],
                        'completed_sha256': file_hash(folder/'completed.json')}
    record(output/'evaluation_index_valid.json', {'split': 'valid', 'specs': specs,
           'results': results, 'context_signature': context_signature})
    return results
