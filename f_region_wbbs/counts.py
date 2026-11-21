"""Resumable exact regional BBP counts, shared by all frozen score recipes.

Valid stores only fixed-query tokens by all gallery tokens. Test computes every
upper-triangle entry including its real diagonal and mirrors the count only.
One completed block records a SHA for each owned row segment; partially written
unmarked segments are recomputed. No score or directional F match is mirrored.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import multiprocessing
from pathlib import Path
import os
import tempfile
import time

import numpy as np

from f_region_structure.state import now, record, seal, verified
from wbm_retrieval.common import digest, file_hash, read_json
from .prepare import read_cache
from .scoring import MODES, RegionScorer, cache_layout

BLOCK_ROWS = 16
VERSION = 1
_COUNTER = None
_MATRICES = None
_TOKENS = None
_TRIANGLE = None


def _sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _request(cache, pool, context_signature):
    if not isinstance(context_signature, str) or not context_signature:
        raise ValueError('A frozen context signature is required')
    ids, offsets, _ = cache_layout(cache, max_regions=4)
    gallery, queries = np.asarray(pool.gallery), np.asarray(list(pool))
    if (pool.split_name not in ('valid', 'test') or gallery.dtype.kind not in 'iu'
            or queries.dtype.kind not in 'iu' or gallery.ndim != 1 or queries.ndim != 1
            or not len(queries) or len(np.unique(queries)) != len(queries)
            or not np.array_equal(gallery, ids) or not np.isin(queries, ids).all()):
        raise ValueError('Counts require the unchanged full-gallery ordinary valid/test pool')
    positions = {int(s): i for i, s in enumerate(ids)}
    tokens = np.concatenate([np.arange(offsets[positions[int(q)]], offsets[positions[int(q)]+1],
                                     dtype=np.int64) for q in queries])
    triangle = pool.split_name == 'test'
    if triangle:
        if not np.array_equal(queries, ids):
            raise ValueError('Test upper triangle requires all gallery maps in original order')
        tokens = np.arange(int(offsets[-1]), dtype=np.int64)
    source = Path(cache['folder']).resolve()
    metadata = {'version': VERSION, 'context_signature': context_signature,
        'split': pool.split_name, 'retrieval_protocol_signature': pool.protocol_signature,
        'source_cache': str(source), 'source_cache_completed_sha256': file_hash(source/'completed.json'),
        'point_modes': list(MODES), 'dtype': 'uint16', 'query_tokens': len(tokens),
        'gallery_tokens': int(offsets[-1]), 'query_tokens_sha256': _sha(tokens),
        'query_ids_sha256': _sha(queries.astype(np.int64)), 'gallery_ids_sha256': _sha(ids),
        'map_offsets_sha256': _sha(offsets), 'upper_triangle': triangle,
        'diagonal': 'original exact mutual-nearest-neighbour count; never forced',
        'block_rows': BLOCK_ROWS, 'pair_arithmetic': 'frozen exact cKDTree with first-raster cdist tie replay'}
    metadata['signature'] = digest(metadata)
    return metadata, tokens, queries.astype(np.int64), ids, offsets


@contextmanager
def _lock(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another process owns this count-cache run') from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _blocks(n):
    return [(i//BLOCK_ROWS, i, min(i+BLOCK_ROWS, n)) for i in range(0, n, BLOCK_ROWS)]


def _block_path(folder, block):
    return Path(folder)/'rows'/f'{block:06d}.json'


def _init_worker(folder, cache_folder, inherited):
    global _COUNTER, _MATRICES, _TOKENS, _TRIANGLE
    if not inherited:
        _COUNTER = RegionScorer(read_cache(cache_folder))
    folder = Path(folder)
    metadata = read_json(folder/'metadata.json')
    _TRIANGLE = metadata['upper_triangle']
    _TOKENS = np.load(folder/'query_tokens.npy', mmap_mode='r')
    _MATRICES = {mode: np.load(folder/f'{mode}.npy', mmap_mode='r+') for mode in MODES}


def _process_block(task):
    block, start, stop = task
    started = time.perf_counter()
    n = _COUNTER.n
    before = {mode: (_COUNTER.calls[mode], _COUNTER.seconds[mode]) for mode in MODES}
    records = []
    for row in range(start, stop):
        query = int(_TOKENS[row])
        first = row if _TRIANGLE else 0
        entry = {'row': row, 'query_token': query, 'first_column': first, 'sha256': {}}
        for mode in MODES:
            values = _COUNTER.count_matrix(np.array([query], dtype=np.int64), np.arange(first, n, dtype=np.int64), mode)[0]
            _MATRICES[mode][row, first:] = values
            if _TRIANGLE:
                _MATRICES[mode][first:, row] = values
            entry['sha256'][mode] = _sha(values)
        records.append(entry)
    for matrix in _MATRICES.values():
        matrix.flush()
    return {'block': block, 'start': start, 'stop': stop, 'rows': records,
        'wall_seconds': time.perf_counter()-started,
        'point_modes': {mode: {'count_calls': _COUNTER.calls[mode]-before[mode][0],
            'count_seconds': _COUNTER.seconds[mode]-before[mode][1]} for mode in MODES}}


def _row_check(folder, metadata, tokens, matrices, task):
    block, start, stop = task
    path = _block_path(folder, block)
    if not path.exists():
        return False
    marker = read_json(path)
    if (marker.get('signature') != metadata['signature'] or marker.get('status') != 'completed'
            or (marker.get('block'), marker.get('start'), marker.get('stop')) != task
            or len(marker.get('rows', [])) != stop-start):
        raise ValueError('Count block identity differs: '+str(path))
    for row, entry in zip(range(start, stop), marker['rows']):
        first = row if metadata['upper_triangle'] else 0
        if (entry.get('row') != row or entry.get('query_token') != int(tokens[row])
                or entry.get('first_column') != first or set(entry.get('sha256', {})) != set(MODES)):
            raise ValueError('Count row identity differs')
        for mode, matrix in matrices.items():
            values = matrix[row, first:]
            if _sha(values) != entry['sha256'][mode]:
                raise ValueError('A completed count row changed')
            if metadata['upper_triangle'] and not np.array_equal(values, matrix[first:, row]):
                raise ValueError('A completed count mirror changed')
    return True


def _read_shape(folder, metadata):
    if metadata.get('signature') != digest({k: v for k, v in metadata.items() if k != 'signature'}):
        raise ValueError('Count metadata signature differs')
    if (metadata.get('version') != VERSION or metadata.get('point_modes') != list(MODES)
            or metadata.get('dtype') != 'uint16' or metadata.get('block_rows') != BLOCK_ROWS
            or metadata.get('split') not in ('valid', 'test')
            or metadata.get('upper_triangle') != (metadata['split'] == 'test')):
        raise ValueError('Count metadata recipe differs')
    tokens = np.load(folder/'query_tokens.npy', mmap_mode='r')
    shape = (metadata['query_tokens'], metadata['gallery_tokens'])
    if (tokens.dtype != np.int64 or tokens.shape != (shape[0],)
            or len(np.unique(tokens)) != len(tokens) or np.any(tokens < 0) or np.any(tokens >= shape[1])
            or _sha(tokens) != metadata['query_tokens_sha256']
            or (metadata['upper_triangle'] and not np.array_equal(tokens, np.arange(shape[1])))):
        raise ValueError('Count query-token identities differ')
    matrices = {mode: np.load(folder/f'{mode}.npy', mmap_mode='r') for mode in MODES}
    if any(m.dtype != np.uint16 or m.shape != shape for m in matrices.values()):
        raise ValueError('Count matrix shape or dtype differs')
    with np.load(folder/'identities.npz', allow_pickle=False) as identities:
        if any(_sha(identities[key]) != metadata[field] for key, field in
               [('queries', 'query_ids_sha256'), ('gallery', 'gallery_ids_sha256'), ('offsets', 'map_offsets_sha256')]):
            raise ValueError('Count map identities differ')
    return matrices, tokens


def read_counts(folder, *, check=True):
    """Read sealed count arrays as read-only mmap; check=False is parent-verified only."""
    folder = Path(folder)
    metadata = read_json(folder/'metadata.json')
    if check and not verified(folder, metadata['signature']):
        raise ValueError('Count matrices are not complete')
    marker = read_json(folder/'completed.json')
    required = {'metadata.json', 'query_tokens.npy', 'identities.npz', 'timing.json',
                *(f'{mode}.npy' for mode in MODES),
                *(str(_block_path(folder, b).relative_to(folder)) for b, _, _ in _blocks(metadata['query_tokens']))}
    if (marker.get('status') != 'completed' or marker.get('signature') != metadata['signature']
            or set(marker.get('files', {})) != required):
        raise ValueError('Count seal inventory is incomplete or unexpected')
    return _read_shape(folder, metadata)


def create(output, cache, pool, context_signature, workers=24):
    """Create or resume exclusively owned row blocks, then seal both full arrays."""
    global _COUNTER
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer')
    cache = read_cache(cache['folder'])
    metadata, tokens, queries, gallery, offsets = _request(cache, pool, context_signature)
    parent = Path(output).resolve()/'counts'
    folder = parent/pool.split_name
    if folder.is_relative_to(Path(cache['folder']).resolve()):
        raise ValueError('Count output must not overwrite its source cache')
    started = time.perf_counter()
    with _lock(parent/('.'+pool.split_name+'.lock')):
        if folder.exists():
            if read_json(folder/'metadata.json') != metadata:
                raise ValueError('Count cache request differs from its frozen identity')
            if verified(folder, metadata['signature']):
                return read_counts(folder, check=False)
        else:
            attempt = Path(tempfile.mkdtemp(prefix='.'+pool.split_name+'_counts_', dir=parent))
            record(attempt/'metadata.json', metadata)
            np.save(attempt/'query_tokens.npy', tokens)
            np.savez(attempt/'identities.npz', queries=queries, gallery=gallery, offsets=offsets)
            (attempt/'rows').mkdir()
            for mode in MODES:
                matrix = np.lib.format.open_memmap(attempt/f'{mode}.npy', mode='w+', dtype=np.uint16,
                          shape=(len(tokens), int(offsets[-1])))
                matrix[:] = np.iinfo(np.uint16).max
                matrix.flush()
                del matrix
            attempt.rename(folder)
        matrices, tokens = _read_shape(folder, metadata)
        tasks = _blocks(len(tokens))
        wanted_blocks = {f'{b:06d}.json' for b, _, _ in tasks}
        if {p.name for p in (folder/'rows').iterdir()} - wanted_blocks:
            raise ValueError('Unexpected count-block state')
        pending = [task for task in tasks if not _row_check(folder, metadata, tokens, matrices, task)]
        resumed = len(tasks)-len(pending)
        build_seconds = 0.
        if pending:
            # Linux fork shares immutable descriptor arrays and built KD trees.
            # Every worker modifies disjoint matrix cells and its private counters.
            inherited = 'fork' in multiprocessing.get_all_start_methods()
            method = 'fork' if inherited else 'spawn'
            if inherited:
                _COUNTER = RegionScorer(cache)
                build_seconds = _COUNTER.build_seconds
            try:
                with ProcessPoolExecutor(max_workers=min(workers, len(pending)),
                        mp_context=multiprocessing.get_context(method), initializer=_init_worker,
                        initargs=(str(folder), str(cache['folder']), inherited)) as executor:
                    futures = {executor.submit(_process_block, task): task for task in pending}
                    for number, future in enumerate(as_completed(futures), 1):
                        result = future.result()
                        record(_block_path(folder, result['block']),
                               {'status': 'completed', 'signature': metadata['signature'], **result})
                        if number % 25 == 0 or number == len(pending):
                            print(f'{pool.split_name} exact BB counts: {resumed+number}/{len(tasks)} blocks', flush=True)
            finally:
                _COUNTER = None
        for task in tasks:
            if not _row_check(folder, metadata, tokens, matrices, task):
                raise ValueError('Count matrix has unfinished rows')
        for mode, matrix in matrices.items():
            totals = np.diff(cache['point_sets'][mode]['offsets'])
            for first in range(0, len(tokens), 64):
                bounds = np.minimum(totals[tokens[first:first+64], None], totals[None])
                if np.any(matrix[first:first+64] > bounds):
                    raise ValueError('Completed BB count exceeds complete point counts')
        timing_path = folder/'timing.json'
        if not timing_path.exists():
            records = [read_json(_block_path(folder, b)) for b, _, _ in tasks]
            record(timing_path, {'completed_at': now().isoformat(), 'invocation_wall_seconds': time.perf_counter()-started,
                'workers_requested': workers, 'blocks_resumed': resumed, 'blocks_total': len(tasks),
                'backend_build_seconds_this_invocation': build_seconds,
                'point_modes': {mode: {'count_calls': sum(r['point_modes'][mode]['count_calls'] for r in records),
                    'summed_worker_count_seconds': sum(r['point_modes'][mode]['count_seconds'] for r in records)} for mode in MODES},
                'logical_matrix_shape': [len(tokens), int(offsets[-1])],
                'scope': 'exact counts only; upper triangle and mirrored storage only for test'})
        seal(folder, metadata['signature'])
        return read_counts(folder, check=False)


def benchmark(cache, pool, query_ids=None, candidates_per_query=20, workers_estimate=24):
    """Serial diagnostic only, with an explicitly ideal parallel extrapolation."""
    engine = RegionScorer(cache)
    ids, offsets, _ = cache_layout(cache, max_regions=4)
    positions = {int(q): i for i, q in enumerate(ids)}
    chosen = list(pool)[:37] if query_ids is None else list(query_ids)
    if (len(set(chosen)) != len(chosen) or not set(chosen) <= set(map(int, pool))
            or isinstance(candidates_per_query, bool) or candidates_per_query < 1
            or isinstance(workers_estimate, bool) or workers_estimate < 1):
        raise ValueError('Benchmark requires unique eligible queries and positive diagnostic limits')
    started = time.perf_counter()
    map_pairs = 0
    checked = {mode: 0 for mode in MODES}
    original = __import__('fgw_fusion.wbbs_backend', fromlist=['original_module']).original_module()
    for q in chosen:
        qi = positions[int(q)]
        qt = np.arange(offsets[qi], offsets[qi+1], dtype=np.int64)
        for c in np.sort(pool[q])[:candidates_per_query]:
            ci = positions[int(c)]
            ct = np.arange(offsets[ci], offsets[ci+1], dtype=np.int64)
            for mode in MODES:
                counts = engine.count_matrix(qt, ct, mode)
                # Independent oracle work is excluded from the engine count timer.
                for i, qtoken in enumerate(qt):
                    for j, ctoken in enumerate(ct):
                        if checked[mode] < 64:
                            backend = engine.backends[mode]
                            expected = len(original.best_buddies(backend._original_rows(qtoken),
                                backend._original_rows(ctoken), mu=1., block_size=256))
                            if counts[i, j] != expected:
                                raise ValueError('Real-data exact BBP benchmark oracle differs')
                            checked[mode] += 1
            map_pairs += 1
    statistics = engine.statistics()
    query_tokens = sum(int(offsets[positions[int(q)]+1]-offsets[positions[int(q)]]) for q in pool)
    full_pairs = engine.n*(engine.n+1)//2 if pool.split_name == 'test' else query_tokens*engine.n
    statistics.update(status='verified', diagnostic_only=True, split=pool.split_name,
        query_ids=list(map(int, chosen)), map_pairs=map_pairs, oracle_count_pairs=checked,
        total_wall_seconds=time.perf_counter()-started, full_split_count_pairs_per_mode=full_pairs,
        ideal_parallel_workers=workers_estimate,
        ideal_parallel_seconds={mode: (engine.seconds[mode]/engine.calls[mode]*full_pairs/workers_estimate
                                  if engine.calls[mode] else None) for mode in MODES},
        extrapolation_scope='ideal scaling only; excludes dispatch, synchronization, I/O, memory contention, setup and oracle')
    return statistics
