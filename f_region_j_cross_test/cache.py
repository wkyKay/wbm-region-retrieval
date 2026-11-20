"""Compute the original regional scores once, using disjoint query workers."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from f_region_area_fusion.pipeline import Provider as OriginalProvider
from wbm_retrieval.common import file_hash, read_json, write_json
from .state import VARIANT, SPEC, record, cache_signature

FILES = ('ids.npy', 'scores.npy', 'metadata.json')
_WORK = None


class ScoreCache:
    def __init__(self, folder, context_signature):
        self.folder = Path(folder)
        self.marker = self.folder / 'completed.json'
        marker = read_json(self.marker)
        self.signature = cache_signature(context_signature)
        if (marker.get('status') != 'completed' or marker.get('signature') != self.signature
                or set(marker.get('files', {})) != set(FILES)):
            raise ValueError('Invalid complete regional score cache')
        for name, expected in marker['files'].items():
            if file_hash(self.folder / name) != expected:
                raise ValueError('Score cache hash mismatch: ' + name)
        meta = read_json(self.folder / 'metadata.json')
        if (meta.get('context_signature') != context_signature or meta.get('variant') != VARIANT
                or meta.get('spec') != SPEC or meta.get('dtype') != 'float64'
                or meta.get('self_score') != 'NaN' or not meta.get('base_protocol_signature')):
            raise ValueError('Regional cache formula/context differs')
        self.query_ids = np.load(self.folder / 'ids.npy', allow_pickle=False)
        self.gallery_ids = self.query_ids.copy()
        self.matrix = np.load(self.folder / 'scores.npy', mmap_mode='r', allow_pickle=False)
        ids = self.query_ids
        if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(np.unique(ids)) != len(ids)
                or np.any(ids < 0) or len(ids) < 2 or self.matrix.shape != (len(ids), len(ids))
                or self.matrix.dtype != np.float64 or meta.get('query_count') != len(ids)
                or meta.get('gallery_count') != len(ids) or meta.get('full_pairs') != len(ids)*(len(ids)-1)):
            raise ValueError('Regional cache identities/shape differ')
        for row in range(len(ids)):
            values = self.matrix[row]
            if (not np.isnan(values[row]) or np.count_nonzero(np.isnan(values)) != 1
                    or not np.isfinite(np.delete(values, row)).all()
                    or np.nanmin(values) < 0 or np.nanmax(values) > 1 + 1e-12):
                raise ValueError('Incomplete or invalid regional score row')
        self.rows = {int(q): i for i, q in enumerate(ids)}
        self.protocol_signature = meta['base_protocol_signature']
        self.split = 'test'

    def scores(self, query, candidates):
        candidates = np.asarray(candidates)
        if (candidates.ndim != 1 or candidates.dtype.kind not in 'iu'
                or len(np.unique(candidates)) != len(candidates) or int(query) in candidates):
            raise ValueError('Expected unique nonself candidate IDs')
        try:
            row = self.rows[int(query)]
            columns = np.asarray([self.rows[int(c)] for c in candidates], dtype=np.int64)
        except KeyError as error:
            raise ValueError('Unknown score cache identity') from error
        return np.asarray(self.matrix[row, columns])

    def full_row(self, query):
        ids = self.gallery_ids[self.gallery_ids != int(query)]
        return ids, self.scores(query, ids)


class AuditProvider:
    """Keep original full-matrix arithmetic; select cross candidates afterwards."""
    def __init__(self, data):
        self.original = OriginalProvider(data['cache'], data['base_pool'], data['matching'])
        self.pool, self.cache = data['pool'], data['cache']

    def tokens(self, query):
        return self.original.tokens(query)

    def parts(self, query):
        return self.original.parts(query)

    def score(self, query, spec, parts=None):
        ids, values = self.original.score(query, spec, parts)
        candidates = np.asarray(self.pool[query])
        mask = np.isin(ids, candidates, assume_unique=True)
        if not np.array_equal(ids[mask], candidates):
            raise ValueError('Filtered candidate order differs')
        return candidates, values[mask]


def _initialize(cache, base_pool, matching, folder):
    global _WORK
    if any(os.environ.get(k) != '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('Frozen numerical environment requires one BLAS thread per worker')
    ids = np.load(Path(folder) / 'ids.npy', allow_pickle=False)
    matrix = np.load(Path(folder) / 'scores.npy', mmap_mode='r+', allow_pickle=False)
    _WORK = (OriginalProvider(cache, base_pool, matching), matrix,
             {int(q): i for i, q in enumerate(ids)})


def _score_shard(task):
    index, queries = task
    provider, matrix, rows = _WORK
    for number, query in enumerate(queries, 1):
        ids, values = provider.score(int(query), SPEC)
        if (len(ids) != len(rows)-1 or int(query) in ids or values.dtype != np.float64
                or not np.isfinite(values).all()):
            raise ValueError('Original full-pool regional scores are incomplete')
        columns = np.asarray([rows[int(c)] for c in ids], dtype=np.int64)
        matrix[rows[int(query)], columns] = values
        if number % 100 == 0 or number == len(queries):
            print(f'Regional scoring shard {index+1}: {number}/{len(queries)}', flush=True)
    matrix.flush()
    return {'index': index, 'queries': list(map(int, queries)), 'count': len(queries)}


def build_cache(output, data, context_signature, workers=8):
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError('Use one to eight independent query workers')
    output = Path(output)
    folder = output / 'score_cache'
    if folder.exists():
        return ScoreCache(folder, context_signature), {'reused': True}
    started = time.perf_counter()
    ids = np.asarray(list(data['base_pool']), dtype=np.int64)
    if not np.array_equal(ids, data['base_pool'].gallery):
        raise ValueError('Test query and gallery ordering must match')
    attempt = Path(tempfile.mkdtemp(prefix='.score_cache_attempt_', dir=output))
    np.save(attempt / 'ids.npy', ids, allow_pickle=False)
    matrix = np.lib.format.open_memmap(attempt / 'scores.npy', mode='w+', dtype=np.float64,
                                      shape=(len(ids), len(ids)))
    matrix[:] = np.nan
    matrix.flush()
    del matrix
    parts = [p.tolist() for p in np.array_split(ids, min(workers, len(ids))) if len(p)]
    tasks = list(enumerate(parts))
    args = (data['cache'], data['base_pool'], data['matching'], str(attempt))
    results = []
    if workers == 1:
        _initialize(*args)
        results = [_score_shard(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=len(tasks), mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_initialize, initargs=args) as executor:
            futures = [executor.submit(_score_shard, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
                write_json(output / 'status.json', {'status': 'running', 'phase': 'regional_score_cache',
                       'queries_done': sum(r['count'] for r in results), 'queries_total': len(ids)})
    completed = [q for result in sorted(results, key=lambda x: x['index']) for q in result['queries']]
    if completed != ids.tolist():
        raise ValueError('Query shards did not cover the frozen gallery exactly once')
    record(attempt / 'metadata.json', {'context_signature': context_signature, 'variant': VARIANT,
           'spec': SPEC, 'base_protocol_signature': data['base_pool'].protocol_signature,
           'query_count': len(ids), 'gallery_count': len(ids), 'dtype': 'float64', 'self_score': 'NaN',
           'full_pairs': len(ids)*(len(ids)-1), 'scope': 'Original full-gallery regional scorer; no changed arithmetic'})
    record(attempt / 'completed.json', {'status': 'completed', 'signature': cache_signature(context_signature),
           'files': {name: file_hash(attempt / name) for name in FILES}})
    ScoreCache(attempt, context_signature)
    if folder.exists():
        raise FileExistsError('Concurrent cache publication')
    attempt.rename(folder)
    return ScoreCache(folder, context_signature), {'reused': False, 'workers': workers,
           'full_pairs': len(ids)*(len(ids)-1), 'wall_seconds': time.perf_counter()-started}
