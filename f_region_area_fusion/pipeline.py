"""Batch regional components, stream rankings, and replay every full candidate pool."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import csv
import multiprocessing
from pathlib import Path
import tempfile

import numpy as np

from f_region_structure.pipeline import Provider as OldProvider, _scalar_greedy, report
from f_region_structure.verify import verify_evaluation
from f_region_score_test.pipeline import _merge_shards
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from wbm_retrieval.diagnostics import batched_map_scores
from wbm_retrieval.exact_metrics import order_scores
from . import scoring
from .state import identity, record, seal, verified

KS = [1, 5, 10, 20, 50, 100, 500]
MATCHING = {'lambda_pos': 0., 'lambda_scale': 0., 'shape_threshold': 0.,
            'score_threshold': 0., 'weight': 'sqrt_area'}


class Provider(OldProvider):
    def __init__(self, cache, pool, matching):
        queries = np.asarray(list(pool))
        gallery = np.asarray(pool.gallery)
        if (pool.split_name not in ('valid', 'test') or queries.ndim != 1 or gallery.ndim != 1
                or queries.dtype.kind not in 'iu' or gallery.dtype.kind not in 'iu'
                or len(queries) < 1 or len(gallery) < 2
                or len(np.unique(queries)) != len(queries) or len(np.unique(gallery)) != len(gallery)
                or not set(map(int, queries)) <= set(map(int, gallery))
                or pool.candidates_per_query != len(gallery)-1):
            raise ValueError('A full gallery except self is required')
        if pool.split_name == 'test' and set(map(int, queries)) != set(map(int, gallery)):
            raise ValueError('Test requires every gallery sample as a query')
        if any(matching.get(k) != v for k, v in MATCHING.items()):
            raise ValueError('The original greedy sqrt-area matcher must remain fixed')
        super().__init__(cache, pool, matching)
        self.engine = scoring.AreaPairScorer(cache['features'], cache, cache)

    def parts(self, query):
        return self.engine.components(self.tokens(query))

    def score(self, query, spec, parts=None):
        qt = self.tokens(query)
        ids = np.asarray(self.pool[query], dtype=np.int64)
        if len(ids) != self.pool.candidates_per_query or int(query) in ids:
            raise ValueError('Candidate pool must retain all gallery samples except self')
        pair = scoring.score_components(self.parts(query) if parts is None else parts, spec)
        rows = self.padded[[self.positions[int(c)] for c in ids]]
        values = batched_map_scores(pair, rows, self.cache['area'][qt], self.matching)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite candidate score')
        return ids, values


def audit_formulas(provider, bundle, output, specs, context_signature):
    split = provider.pool.split_name
    representatives = {}
    for q in provider.pool:
        representatives.setdefault(int(bundle.class_ids[q]), int(q))
    path = Path(output)/f'formula_audit_{split}.json'
    fixed = {'status': 'verified', 'split': split, 'specs': specs,
             'context_signature': context_signature, 'query_ids': list(representatives.values()),
             'candidate_pairs': len(representatives)*min(20, provider.pool.candidates_per_query)*len(specs)}
    if path.exists():
        previous = read_json(path)
        error = previous.get('max_absolute_error')
        if (any(previous.get(k) != v for k, v in fixed.items())
                or isinstance(error, bool) or not isinstance(error, (int, float))
                or not np.isfinite(error) or not 0 <= error <= 2e-6):
            raise ValueError('Existing formula audit differs from this frozen stage')
        return previous
    cache = provider.cache
    maximum = 0.
    for q in representatives.values():
        ids = np.asarray(provider.pool[q])[:20]
        qt = provider.tokens(q)
        ct = np.concatenate([provider.tokens(c) for c in ids])
        selected = np.concatenate([qt, ct])
        parts = provider.parts(q)
        if 'entropy' in specs:
            reference_parts = scoring.independent_components(qt, np.asarray(cache['features']),
                cache['relative_area'], cache['joint_all'], cache['owners'],
                candidate_indices=selected)
        else:
            reference_parts = scoring.independent_components(np.arange(len(qt)),
                np.asarray(cache['features'][selected]), cache['relative_area'][selected],
                cache['joint_all'][selected], cache['owners'][selected], with_entropy=False)
        for name, spec in specs.items():
            actual = scoring.score_components(parts, spec)[:, selected]
            # Entropy must use the complete candidate-region population, never
            # the audit subset. Its independent implementation excludes owners.
            reference = scoring.independent_score_components(reference_parts, spec)
            error = float(np.max(np.abs(actual-reference))) if actual.size else 0.
            if error > 2e-6:
                raise ValueError(f'Independent regional formula mismatch: {name}/{q}: {error}')
            all_ids, values = provider.score(q, spec, parts)
            lookup = dict(zip(map(int, all_ids), map(float, values)))
            start = len(qt)
            for c in ids:
                width = len(provider.tokens(c))
                scalar = _scalar_greedy(actual[:, start:start+width], cache['area'][qt])
                start += width
                difference = abs(scalar-lookup[int(c)])
                if difference > 2e-6:
                    raise ValueError(f'Independent scalar greedy mismatch: {name}/{q}/{c}')
                error = max(error, difference)
            maximum = max(maximum, error)
    value = {**fixed, 'max_absolute_error': maximum,
             'scope': 'Independent regional formulas; entropy uses complete gallery region scores excluding all query-owned regions; scalar greedy on production matrices; first 20 candidates per class representative.'}
    record(path, value)
    return value


_WORKER = None


def _initialize_worker(cache, pool, matching, bundle=None):
    global _WORKER
    _WORKER = (Provider(cache, pool, matching), bundle)


def _rank_shard(task):
    index, queries, paths, specs, k = task
    provider, _ = _WORKER
    with ExitStack() as stack:
        writers = {name: csv.writer(stack.enter_context(Path(path).open('w', newline='')))
                   for name, path in paths.items()}
        for number, q in enumerate(queries, 1):
            parts = provider.parts(q)
            for name, spec in specs.items():
                ids, values = provider.score(q, spec, parts)
                order = order_scores(ids, values, int(q), 2026)[:k]
                if len(order) != k:
                    raise ValueError('A ranking shard cannot provide the complete Top-K')
                writers[name].writerows((int(q), rank, int(ids[i]), float(values[i]))
                                       for rank, i in enumerate(order, 1))
            interval = 25 if provider.pool.split_name == 'valid' else 100
            if number % interval == 0 or number == len(queries):
                print(f'{provider.pool.split_name} ranking shard {index+1}: {number}/{len(queries)}; {len(specs)} variants', flush=True)
    return {name: {'index': index, 'queries': queries, 'rows': len(queries)*k,
                   'path': path, 'sha256': file_hash(path)} for name, path in paths.items()}


def rank_all(output, folders, specs, cache, pool, matching, workers, ks):
    partitions = [p.tolist() for p in np.array_split(np.asarray(list(pool)), min(workers, len(pool)))]
    write_json(Path(output)/'status.json', {'status': 'running', 'phase': 'rankings',
               'split': pool.split_name, 'done': 0, 'total': len(pool), 'variants': list(specs)})
    with tempfile.TemporaryDirectory(prefix='f_region_area_fusion_rank_') as scratch:
        tasks = [(i, queries, {n: str(Path(scratch)/f'{i:03d}_{n}.csv') for n in specs}, specs, max(ks))
                 for i, queries in enumerate(partitions)]
        results = []
        if workers == 1:
            _initialize_worker(cache, pool, matching)
            results = [_rank_shard(t) for t in tasks]
        else:
            with ProcessPoolExecutor(max_workers=len(tasks), mp_context=multiprocessing.get_context('spawn'),
                    initializer=_initialize_worker, initargs=(cache, pool, matching)) as executor:
                futures = [executor.submit(_rank_shard, task) for task in tasks]
                for future in as_completed(futures):
                    results.append(future.result())
                    done = sum(len(next(iter(r.values()))['queries']) for r in results)
                    write_json(Path(output)/'status.json', {'status': 'running', 'phase': 'rankings',
                        'split': pool.split_name, 'done': done, 'total': len(pool), 'variants': list(specs)})
        for name in specs:
            selected_tasks = [(i, queries, paths[name], k) for i, queries, paths, _, k in tasks]
            _merge_shards(folders[name]/'rankings.csv', selected_tasks, [r[name] for r in results])


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


def evaluate(output, cache, pool, bundle, matching, context_signature, specs, workers=8,
             frozen_selection=None):
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError('workers must be an integer between 1 and 8')
    split = pool.split_name
    grid = scoring.frozen_grid()
    if not specs or any(n not in grid or s != grid[n] for n, s in specs.items()):
        raise ValueError('All specifications must belong to the frozen 29-candidate grid')
    if split == 'valid':
        if specs != grid or frozen_selection is not None:
            raise ValueError('Valid requires the complete grid and no preselected subset')
    elif split == 'test':
        if (not isinstance(frozen_selection, dict) or len(specs) != 10
                or specs != frozen_selection.get('selected_specs')
                or list(specs) != frozen_selection.get('selected_variants')):
            raise ValueError('Test requires exactly the frozen valid Top-10 in selection order')
    else:
        raise ValueError('Only valid/test evaluation is permitted')
    provider = Provider(cache, pool, matching)
    if max(KS) > pool.candidates_per_query:
        raise ValueError('Gallery is smaller than the fixed maximum K')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    audit_formulas(provider, bundle, output, specs, context_signature)
    folders, signatures, pending = {}, {}, {}
    for name, spec in specs.items():
        signature = digest({'context': context_signature, 'split': split, 'variant': name, 'spec': spec})
        folder = output/split/name
        folders[name], signatures[name] = folder, signature
        if verified(folder, signature):
            continue
        if folder.exists() and any(folder.iterdir()):
            parent = output/'interrupted_evaluations'
            parent.mkdir(exist_ok=True)
            saved = Path(tempfile.mkdtemp(prefix=split+'__'+name+'__', dir=parent))
            folder.rename(saved/'evaluation')
        folder.mkdir(parents=True, exist_ok=True)
        run_identity = identity(output, split, name, signature)
        record(folder/'run_identity.json', run_identity)
        record(folder/'evaluation_manifest.json', {
            **run_identity, 'signature': signature, 'spec': spec, 'ks': KS, 'tie_seed': 2026,
            'retrieval_protocol_signature': pool.protocol_signature, 'num_queries': len(pool),
            'gallery_count': len(pool.gallery), 'candidate_rule': 'all_gallery_except_query',
            'ranking_scope': 'top_max_k', 'posthoc_research': True, 'test_parameter_search': False,
            'selection_split': 'valid', 'test_evaluated': split == 'test', 'training_required': False,
            'checkpoint_kind': 'best_nce', 'source_seed': 2026, 'matching': 'original_F_greedy_sqrt_area',
            'valid_selection_rank': (frozen_selection['selected_variants'].index(name)+1
                                     if split == 'test' else None)})
        pending[name] = spec
    if pending:
        rank_all(output, folders, pending, cache, pool, matching, workers, KS)
        tasks = [(str(folders[n]), n, s, signatures[n], KS) for n, s in pending.items()]
        write_json(output/'status.json', {'status': 'running', 'phase': 'reports_and_full_replay',
                    'split': split, 'done': 0, 'total': len(tasks)})
        if workers == 1:
            _initialize_worker(cache, pool, matching, bundle)
            for name in map(_finish, tasks):
                print(split+' verified: '+name, flush=True)
        else:
            with ProcessPoolExecutor(max_workers=min(workers, len(tasks)),
                    mp_context=multiprocessing.get_context('spawn'), initializer=_initialize_worker,
                    initargs=(cache, pool, matching, bundle)) as executor:
                futures = [executor.submit(_finish, task) for task in tasks]
                for i, future in enumerate(as_completed(futures), 1):
                    name = future.result()
                    print(f'{split} verified {i}/{len(tasks)}: {name}', flush=True)
                    write_json(output/'status.json', {'status': 'running', 'phase': 'reports_and_full_replay',
                        'split': split, 'done': i, 'total': len(tasks)})
    results = {}
    for name, folder in folders.items():
        checked = read_json(folder/'verification.json')
        results[name] = {'top10_hits': int(checked['top10_hits']), 'num_queries': len(pool),
                        'metrics': read_json(folder/'metrics.json')['metrics'],
                        'coverage_metrics': read_json(folder/'coverage_metrics.json')['metrics'],
                        'completed_sha256': file_hash(folder/'completed.json')}
    record(output/f'evaluation_index_{split}.json', {'split': split, 'specs': specs,
           'results': results, 'context_signature': context_signature})
    return results
