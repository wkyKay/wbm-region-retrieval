"""One frozen test scorer, streamed ranking shards, and independent verification."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import math
import multiprocessing
from pathlib import Path
import shutil
import tempfile

import numpy as np

from f_region_score_search import scoring
from f_region_structure.pipeline import Provider as OldProvider, _scalar_greedy, report
from f_region_structure.verify import verify_evaluation
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from wbm_retrieval.diagnostics import batched_map_scores
from wbm_retrieval.exact_metrics import order_scores
from .state import identity, record, seal, verified


VARIANT = 'mixed_a0p5_e1p0'
SPEC = scoring.frozen_grid()[VARIANT]
KS = [1, 5, 10, 20, 50, 100, 500]
_MATCHING = {'lambda_pos': 0., 'lambda_scale': 0., 'shape_threshold': 0.,
             'score_threshold': 0., 'weight': 'sqrt_area'}
_HEADER = ['query_id', 'rank', 'candidate_id', 'similarity_score']


class Provider(OldProvider):
    """The selected valid formula only; no variant argument or test search."""
    def __init__(self, cache, pool, matching):
        queries = np.asarray(list(pool))
        gallery = np.asarray(pool.gallery)
        if (pool.split_name != 'test' or queries.ndim != 1 or gallery.ndim != 1
                or queries.dtype.kind not in 'iu' or gallery.dtype.kind not in 'iu'
                or len(queries) < 2 or len(np.unique(queries)) != len(queries)
                or len(np.unique(gallery)) != len(gallery)
                or len(queries) != len(gallery) or set(map(int, queries)) != set(map(int, gallery))
                or pool.candidates_per_query != len(gallery)-1):
            raise ValueError('Only complete test queries and full gallery except self are permitted')
        if any(matching.get(key) != value for key, value in _MATCHING.items()):
            raise ValueError('The original greedy sqrt-area matcher must remain fixed')
        super().__init__(cache, pool, matching)

    def parts(self, query):
        return self.engine.components(self.tokens(query), specs={'selected': scoring.OLD_SPEC})

    def score(self, query, parts=None):
        tokens = self.tokens(query)
        candidates = np.asarray(self.pool[query], dtype=np.int64)
        if len(candidates) != self.pool.candidates_per_query or int(query) in candidates:
            raise ValueError('Candidate pool must exclude self and retain the full gallery')
        pair = scoring.score_components(self.parts(query) if parts is None else parts, SPEC)
        rows = self.padded[[self.positions[int(c)] for c in candidates]]
        values = batched_map_scores(pair, rows, self.cache['area'][tokens], self.matching)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite test candidate score')
        return candidates, values


def _representatives(provider, bundle):
    representatives = {}
    for q in provider.pool:
        representatives.setdefault(int(bundle.class_ids[q]), int(q))
    return list(representatives.values())


def audit_formulas(provider, bundle, output, context_signature):
    """Check regional formulas independently, then scalar greedy on identical pairs."""
    representatives = _representatives(provider, bundle)
    path = Path(output)/'formula_audit_test.json'
    expected_count = len(representatives)*min(20, provider.pool.candidates_per_query)
    if path.exists():
        previous = read_json(path)
        fixed = {'status': 'verified', 'split': 'test', 'variant': VARIANT,
                 'context_signature': context_signature, 'query_ids': representatives,
                 'candidate_pairs': expected_count}
        error = previous.get('max_absolute_error')
        if (any(previous.get(k) != v for k, v in fixed.items())
                or isinstance(error, bool) or not isinstance(error, (int, float))
                or not math.isfinite(error) or not 0 <= error <= 2e-6):
            raise ValueError('Existing formula audit differs from the frozen test evaluation')
        return previous
    maximum, count = 0., 0
    cache = provider.cache
    for q in representatives:
        candidates = np.asarray(provider.pool[q])[:20]
        qt = provider.tokens(q)
        ct = np.concatenate([provider.tokens(c) for c in candidates])
        selected = np.concatenate([qt, ct])
        parts = provider.parts(q)
        actual = scoring.score_components(parts, SPEC)[:, selected]
        reference = scoring.independent_pair_scores(
            np.arange(len(qt)), np.asarray(cache['features'][selected]),
            cache['relative_area'][selected], cache['joint_all'][selected], SPEC)
        error = float(np.max(np.abs(actual-reference))) if actual.size else 0.
        if error > 2e-6:
            raise ValueError('Independent regional test formula mismatch')
        ids, values = provider.score(q, parts)
        lookup = dict(zip(map(int, ids), map(float, values)))
        start = len(qt)
        for c in candidates:
            width = len(provider.tokens(c))
            scalar = _scalar_greedy(actual[:, start:start+width], cache['area'][qt])
            start += width
            difference = abs(scalar-lookup[int(c)])
            if difference > 2e-6:
                raise ValueError('Independent scalar greedy test mismatch')
            error = max(error, difference)
            count += 1
        maximum = max(maximum, error)
    result = {'status': 'verified', 'split': 'test', 'variant': VARIANT,
              'context_signature': context_signature, 'query_ids': representatives,
              'candidate_pairs': count, 'max_absolute_error': maximum,
              'scope': 'Independent region formulas within tolerance; scalar greedy on identical '
                       'production matrices; first 20 fixed candidates per class representative.'}
    record(path, result)
    return result


_WORKER = None


def _initialize_worker(cache, pool, matching):
    global _WORKER
    _WORKER = Provider(cache, pool, matching)


def _rank_shard(task):
    index, queries, filename, k = task
    path = Path(filename)
    provider = _WORKER
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        for number, q in enumerate(queries, 1):
            ids, values = provider.score(q)
            order = order_scores(ids, values, int(q), 2026)[:k]
            if len(order) != k:
                raise ValueError('A test shard cannot provide the requested complete Top-K')
            writer.writerows((int(q), rank, int(ids[i]), float(values[i]))
                             for rank, i in enumerate(order, 1))
            if number % 250 == 0 or number == len(queries):
                print(f'test ranking shard {index+1}: {number}/{len(queries)}', flush=True)
    return {'index': index, 'queries': queries, 'rows': len(queries)*k,
            'path': str(path), 'sha256': file_hash(path)}


def _merge_shards(destination, tasks, results):
    """Merge complete, hash-checked shards in the original fixed query order."""
    if len(results) != len(tasks) or len({r['index'] for r in results}) != len(tasks):
        raise ValueError('Incomplete or duplicated test ranking shards')
    by_index = {r['index']: r for r in results}
    for index, queries, filename, k in tasks:
        result = by_index.get(index)
        if (result is None or result.get('queries') != queries or result.get('path') != filename
                or result.get('rows') != len(queries)*k or file_hash(filename) != result.get('sha256')):
            raise ValueError('A test ranking shard changed or has the wrong query partition')
    with Path(destination).open('wb') as output:
        output.write((','.join(_HEADER)+'\r\n').encode())
        for _, _, filename, _ in tasks:
            with Path(filename).open('rb') as handle:
                shutil.copyfileobj(handle, output)


def _rank_all(output, folder, cache, pool, matching, workers, ks):
    queries = list(map(int, pool))
    partitions = [part.tolist() for part in np.array_split(np.asarray(queries), min(workers, len(queries)))]
    write_json(Path(output)/'status.json', {'status': 'running', 'phase': 'rankings',
               'split': 'test', 'done': 0, 'total': len(queries), 'test_evaluated': True})
    # Shards are temporary new scratch files, never artifacts in a sealed run.
    with tempfile.TemporaryDirectory(prefix='f_region_score_test_rank_') as scratch:
        tasks = [(i, q, str(Path(scratch)/f'{i:03d}.csv'), max(ks)) for i, q in enumerate(partitions)]
        results = []
        if workers == 1:
            _initialize_worker(cache, pool, matching)
            results = [_rank_shard(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=len(tasks), mp_context=multiprocessing.get_context('spawn'),
                                     initializer=_initialize_worker, initargs=(cache, pool, matching)) as executor:
                futures = [executor.submit(_rank_shard, task) for task in tasks]
                for future in as_completed(futures):
                    results.append(future.result())
                    write_json(Path(output)/'status.json', {'status': 'running', 'phase': 'rankings',
                               'split': 'test', 'done': sum(len(r['queries']) for r in results),
                               'total': len(queries), 'test_evaluated': True})
        _merge_shards(Path(folder)/'rankings.csv', tasks, results)


def evaluate(output, cache, pool, bundle, matching, context_signature, workers=4):
    """Evaluate only the one valid-selected formula on the complete test pool.

    The root input loader binds the formal 7199-query protocol; this function
    additionally requires every gallery wafer to be a query. Smaller complete
    pools are supported solely for explicitly synthetic integration checks.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError('workers must be an integer between 1 and 8')
    provider = Provider(cache, pool, matching)
    if max(KS) > pool.candidates_per_query:
        raise ValueError('The test gallery is smaller than the fixed maximum K')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    signature = digest({'context': context_signature, 'split': 'test', 'variant': VARIANT, 'spec': SPEC})
    folder = output/'test'/VARIANT
    audit_formulas(provider, bundle, output, context_signature)
    if not verified(folder, signature):
        if folder.exists() and any(folder.iterdir()):
            parent = output/'interrupted_evaluations'
            parent.mkdir(exist_ok=True)
            saved = Path(tempfile.mkdtemp(prefix=VARIANT+'_', dir=parent))
            folder.rename(saved/'evaluation')
        folder.mkdir(parents=True, exist_ok=True)
        run_identity = identity(output, VARIANT, signature)
        record(folder/'run_identity.json', run_identity)
        metadata = {**run_identity, 'signature': signature, 'spec': SPEC, 'ks': KS, 'tie_seed': 2026,
                    'retrieval_protocol_signature': pool.protocol_signature, 'num_queries': len(pool),
                    'gallery_count': len(pool.gallery), 'candidate_rule': 'all_gallery_except_query',
                    'ranking_scope': 'top_max_k', 'posthoc_research': True, 'test_parameter_search': False,
                    'selection_split': 'valid', 'test_evaluated': True, 'training_required': False,
                    'checkpoint_kind': 'best_nce', 'source_seed': 2026,
                    'matching': 'original_F_greedy_sqrt_area'}
        record(folder/'evaluation_manifest.json', metadata)
        _rank_all(output, folder, cache, pool, matching, workers, KS)
        write_json(output/'status.json', {'status': 'running', 'phase': 'reports',
                   'split': 'test', 'test_evaluated': True})
        report(folder, pool, bundle, KS, metadata)
        write_json(output/'status.json', {'status': 'running', 'phase': 'verification',
                   'split': 'test', 'test_evaluated': True})
        checked = verify_evaluation(folder, pool, bundle, KS, signature,
                                    provider.score, tie_seed=2026)
        record(folder/'verification.json', checked)
        seal(folder, signature)
        print('test verified: '+VARIANT, flush=True)
    checked = read_json(folder/'verification.json')
    results = {VARIANT: {'top10_hits': int(checked['top10_hits']), 'num_queries': len(pool),
                        'metrics': read_json(folder/'metrics.json')['metrics'],
                        'coverage_metrics': read_json(folder/'coverage_metrics.json')['metrics'],
                        'completed_sha256': file_hash(folder/'completed.json')}}
    record(output/'evaluation_index_test.json', {'split': 'test', 'specs': {VARIANT: SPEC},
           'results': results, 'context_signature': context_signature})
    return results
