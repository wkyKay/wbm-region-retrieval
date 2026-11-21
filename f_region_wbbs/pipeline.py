"""Shared full-candidate scores, immutable caches and independently checked reports."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from itertools import groupby, zip_longest
import csv
import multiprocessing
from pathlib import Path
import time

import numpy as np

from experiments.proposed.f_segmentation_geometry import coverage_metrics
from f_region_structure.pipeline import report
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from wbm_retrieval.exact_metrics import order_scores
from . import scoring
from .state import KS, record, seal, verified, identity, evaluation_signature

FIELDS = ['query_id', 'rank', 'candidate_id', 'similarity_score']
_WORKER = None


def _init_scores(cache_folder, count_folder, pool, paths, specs):
    global _WORKER
    from .prepare import read_cache
    from .counts import read_counts
    cache = read_cache(cache_folder)
    # Parent create/read_counts already verified every sealed count byte.
    matrices, tokens = read_counts(count_folder, check=False)
    provider = scoring.Provider(cache, pool, matrices, tokens)
    scores = {n: np.load(path, mmap_mode='r+') for n, path in paths.items()}
    positions = {int(s): i for i, s in enumerate(pool.gallery)}
    _WORKER = provider, scores, positions, specs


def _score_shard(task):
    provider, matrices, positions, specs = _WORKER
    shard, rows = task
    for number, (row, q) in enumerate(rows, 1):
        parts = provider.parts(q)
        for name, spec in specs.items():
            ids, scores = provider.score(q, spec, parts)
            matrices[name][row, :] = 0.
            matrices[name][row, [positions[int(c)] for c in ids]] = scores
        if number % 100 == 0 or number == len(rows):
            print(f'Full scores shard {shard}: {number}/{len(rows)} queries, {len(specs)} recipes', flush=True)
    for matrix in matrices.values():
        matrix.flush()
    return len(rows)


class CachedProvider:
    """Only reads a sealed score matrix; no new region scoring during ranking audit."""
    def __init__(self, folder, *, check=True):
        folder = Path(folder)
        meta = read_json(folder/'metadata.json')
        self.signature = meta['signature']
        self.marker = folder/'completed.json'
        if check and not verified(folder, self.signature):
            raise ValueError('Score cache is incomplete')
        with np.load(folder/'identities.npz', allow_pickle=False) as archive:
            self.query_ids = archive['queries']
            self.gallery_ids = archive['gallery']
        self.protocol_signature = meta['retrieval_protocol_signature']
        self.matrix = np.load(folder/'scores.npy', mmap_mode='r')
        self.query_positions = {int(q): i for i, q in enumerate(self.query_ids)}
        self.gallery_positions = {int(c): i for i, c in enumerate(self.gallery_ids)}
        if self.matrix.shape != (len(self.query_ids), len(self.gallery_ids)) or self.matrix.dtype != np.float64:
            raise ValueError('Score matrix dimensions or dtype differ')

    def scores(self, query, ids):
        return np.asarray(self.matrix[self.query_positions[int(query)],
                          [self.gallery_positions[int(c)] for c in ids]])

    def full_row(self, query):
        ids = self.gallery_ids[self.gallery_ids != query]
        return ids, self.scores(query, ids)


def compute_scores(output, cache, pool, context_signature, specs, workers=16):
    output = Path(output)
    folders = {n: output/'score_cache'/pool.split_name/n for n in specs}
    paths, pending = {}, {}
    for name, spec in specs.items():
        folder = folders[name]
        signature = digest({'context': context_signature, 'split': pool.split_name,
                            'variant': name, 'spec': spec, 'score_cache_version': 1})
        if verified(folder, signature):
            continue
        if folder.exists():
            raise FileExistsError('Incomplete score cache requires a fresh run: '+str(folder))
        folder.mkdir(parents=True)
        matrix = np.lib.format.open_memmap(folder/'scores.npy', mode='w+', dtype=np.float64,
                   shape=(len(pool), len(pool.gallery)))
        matrix[:] = np.nan
        matrix.flush()
        del matrix
        np.savez(folder/'identities.npz', queries=np.asarray(list(pool)), gallery=np.asarray(pool.gallery))
        record(folder/'metadata.json', {'signature': signature, 'context_signature': context_signature,
               'variant': name, 'spec': spec, 'split': pool.split_name,
               'retrieval_protocol_signature': pool.protocol_signature,
               'num_queries': len(pool), 'gallery_count': len(pool.gallery),
               'dtype': 'float64', 'self_scores': 'zero; never ranked',
               'source_cache_sha256': file_hash(cache['folder']/'completed.json'),
               'counts_sha256': file_hash(output/'counts'/pool.split_name/'completed.json')})
        paths[name], pending[name] = str(folder/'scores.npy'), spec
    if pending:
        rows = list(enumerate(map(int, pool)))
        partitions = [list(p) for p in np.array_split(np.asarray(rows, dtype=np.int64), min(workers, len(rows)))]
        tasks = [(i, [(int(row), int(q)) for row, q in part]) for i, part in enumerate(partitions)]
        started = time.perf_counter()
        with ProcessPoolExecutor(max_workers=len(tasks), mp_context=multiprocessing.get_context('spawn'),
                initializer=_init_scores, initargs=(str(cache['folder']), str(output/'counts'/pool.split_name), pool, paths, pending)) as executor:
            for done in as_completed([executor.submit(_score_shard, task) for task in tasks]):
                done.result()
        for name in pending:
            matrix = np.load(paths[name], mmap_mode='r')
            for start in range(0, len(matrix), 128):
                chunk = matrix[start:start+128]
                if not np.isfinite(chunk).all() or np.any(chunk < 0) or np.any(chunk > 1):
                    raise ValueError('Incomplete or invalid full score cache')
            del matrix
            seal(folders[name], read_json(folders[name]/'metadata.json')['signature'])
        record(output/f'score_timing_{pool.split_name}.json', {'wall_seconds': time.perf_counter()-started,
               'variants': list(pending), 'workers': workers, 'candidate_pairs_per_variant': len(pool)*pool.candidates_per_query})
    return {name: CachedProvider(folder) for name, folder in folders.items()}


def replay_baseline(output, provider, historical, split, pool=None):
    from fgw_grid_search.verify import independent_order
    path = Path(output)/f'baseline_replay_{split}.json'
    if path.exists():
        saved = read_json(path)
        if saved['source_sha256'] != file_hash(Path(historical)/'rankings.csv') or saved['score_cache_sha256'] != file_hash(provider.marker):
            raise ValueError('Baseline replay source changed')
        return saved
    seen = []
    with (Path(historical)/'rankings.csv').open(newline='') as f:
        for q, records in groupby(csv.DictReader(f), key=lambda r: int(r['query_id'])):
            rows = list(records)
            ids = provider.gallery_ids[provider.gallery_ids != q] if pool is None else np.asarray(pool[q])
            scores = provider.scores(q, ids)
            order = independent_order(q, ids, scores, 2026)[:500]
            if (len(rows) != 500 or not np.array_equal(ids[order], [int(r['candidate_id']) for r in rows])
                    or not np.array_equal(scores[order], [float(r['similarity_score']) for r in rows])):
                raise ValueError(f'Original baseline Top500 did not replay exactly: {split}/{q}')
            seen.append(q)
    if not np.array_equal(seen, provider.query_ids):
        raise ValueError('Baseline query IDs/order differs')
    result = {'status': 'verified', 'split': split, 'num_queries': len(seen),
              'ranking_rows': len(seen)*500, 'top500_identical': True, 'max_score_error': 0.,
              'source': str(Path(historical)/'rankings.csv'),
              'source_sha256': file_hash(Path(historical)/'rankings.csv'),
              'score_cache_sha256': file_hash(provider.marker)}
    record(path, result)
    return result


def _evaluate_one(task):
    output, stage, name, spec, pool, bundle, context_signature, source_binding, selection = task
    from .verify import verify_evaluation
    output = Path(output)
    if stage.startswith('cross_'):
        from .protocol import make_cross_pool
        pool = make_cross_pool(pool, bundle.labels)
    folder = output/stage/name
    signature = evaluation_signature(context_signature, stage, name)
    if not verified(folder, signature):
        if folder.exists():
            raise FileExistsError('Incomplete evaluation; preserve and inspect: '+str(folder))
        folder.mkdir(parents=True)
        provider = CachedProvider(output/'score_cache'/pool.split_name/name)
        ident = identity(output, stage, name, signature)
        cross = stage.startswith('cross_')
        manifest = {**ident, 'signature': signature, 'context_signature': context_signature,
            'variant': name, 'spec': spec, 'ks': KS, 'tie_seed': 2026,
            'num_queries': len(pool), 'gallery_count': len(pool.gallery),
            'candidate_rule': 'all_gallery_except_query_and_exact_label_combination' if cross else 'all_gallery_except_query',
            'retrieval_protocol_signature': pool.protocol_signature,
            'base_protocol_signature': provider.protocol_signature,
            'provider_source': {'signature': provider.signature, 'completed_sha256': file_hash(provider.marker)},
            'source_binding_sha256': digest(source_binding),
            'ranking_scope': 'top_max_k', 'training_required': False, 'posthoc_research': True,
            'test_parameter_search': False, 'selection_split': 'valid', 'selection_protocol': 'coverage_cross_combination_v1', 'formal_protocol': True,
            'primary_variant': None if selection is None else selection['winner'],
            'best_regional_wbbs': None if selection is None else selection['best_regional_wbbs'],
            'frozen_selection_sha256': None if selection is None else file_hash(output/'selection.json')}
        if cross:
            manifest['derived_protocol'] = pool.metadata
        record(folder/'run_identity.json', ident)
        record(folder/'evaluation_manifest.json', manifest)
        started = time.perf_counter()
        with (folder/'rankings.csv').open('x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(FIELDS)
            for number, q in enumerate(pool, 1):
                ids = np.asarray(pool[q])
                scores = provider.scores(q, ids)
                order = order_scores(ids, scores, int(q), 2026)[:500]
                writer.writerows((int(q), rank, int(ids[i]), float(scores[i])) for rank, i in enumerate(order, 1))
                if number % 1500 == 0:
                    print(f'{stage}/{name} rank {number}/{len(pool)}', flush=True)
        ranking_seconds = time.perf_counter()-started
        if cross:
            coverage_metrics.evaluate_rankings_file(bundle, pool, folder/'rankings.csv', KS, folder,
                split_name=pool.split_name, metadata=manifest)
        else:
            report(folder, pool, bundle, KS, manifest)
        checked = verify_evaluation(folder, pool, bundle, KS, signature, provider, source_binding, tie_seed=2026)
        record(folder/'verification.json', checked)
        record(folder/'timing.json', {'ranking_seconds': ranking_seconds,
               'total_wall_seconds': time.perf_counter()-started,
               'verification_scope': 'independent all-query full-cache ordering and metrics; regional formula audit separate'})
        seal(folder, signature)
        print(f'Completed and verified {stage}/{name}', flush=True)
    coverage = read_json(folder/'coverage_metrics.json')
    result = {'coverage_metrics': coverage['metrics'], 'completed_sha256': file_hash(folder/'completed.json')}
    if not stage.startswith('cross_'):
        exact = read_json(folder/'metrics.json')
        result.update(metrics=exact['metrics'], top10_hits=read_json(folder/'verification.json')['top10_hits'])
    return name, result


def evaluate(output, stage, specs, pool, bundle, context_signature, source_binding, selection=None, workers=8):
    worker_pool = pool.base if stage.startswith('cross_') else pool
    tasks = [(str(output), stage, n, spec, worker_pool, bundle, context_signature, source_binding, selection) for n, spec in specs.items()]
    results = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=multiprocessing.get_context('spawn')) as executor:
        for future in as_completed([executor.submit(_evaluate_one, t) for t in tasks]):
            name, result = future.result()
            results[name] = result
    results = {n: results[n] for n in specs}
    record(Path(output)/f'evaluation_index_{stage}.json', {'context_signature': context_signature,
           'stage': stage, 'split': pool.split_name, 'specs': specs, 'results': results})
    return results
