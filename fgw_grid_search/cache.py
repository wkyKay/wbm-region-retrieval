"""Atomic, resumable full F/G/WBBS matrices, with immutable finished reads.

Incomplete resumes validate every sealed row. Finished reads validate all final
matrix/ID hashes and semantics; they do not revisit large row NPZ files or write
status. Unsealed attempt directories are always preserved.
"""
from pathlib import Path
import tempfile
import numpy as np
from wbm_retrieval.common import digest, file_hash, read_json, write_json

COMPONENTS = ('F', 'G', 'WBBS')
VERSION = 'fgw_full_components_v1'


def ids(values):
    a = np.asarray(values)
    if (a.ndim != 1 or not len(a) or a.dtype.kind not in 'iu'
            or len(set(map(int, a))) != len(a)
            or any(int(x) < 0 or int(x) > np.iinfo(np.int64).max for x in a)):
        raise ValueError('Expected nonempty unique nonnegative original integer IDs')
    return a.astype(np.int64, copy=False)


def full_candidates(query, candidates, gallery, queries):
    if isinstance(query, (bool, np.bool_)) or not isinstance(query, (int, np.integer)):
        raise ValueError('Query must be an original integer ID')
    q = int(query)
    c = ids(candidates)
    if (q not in queries or len(c) != len(gallery) - 1
            or set(map(int, c)) != set(map(int, gallery)) - {q}):
        raise ValueError('Expected the entire frozen gallery excluding only this query')
    return q, c


def unit_scores(value, shape, name, *, strict_dtype=False):
    raw = np.asarray(value)
    if (raw.dtype.kind not in 'fiu' or raw.shape != shape
            or (strict_dtype and raw.dtype != np.float64)):
        raise ValueError('Invalid component shape/dtype: ' + name)
    result = raw.astype(np.float64, copy=False)
    if not np.isfinite(result).all() or np.any(result < 0) or np.any(result > 1):
        raise ValueError('Component values must be finite in [0,1]: ' + name)
    return result


def validate_matrix(matrix, queries, gallery):
    if matrix.dtype != np.float64 or matrix.shape != (len(queries), len(gallery)):
        raise ValueError('Full matrix shape/dtype differs from frozen identity')
    for i, q in enumerate(queries):
        mask = gallery != q
        if not np.isnan(matrix[i, ~mask]).all():
            raise ValueError('Self cells must be NaN')
        unit_scores(matrix[i, mask], (len(gallery) - 1,), 'matrix', strict_dtype=True)


def _pool_plan(signature, pool):
    if not isinstance(signature, str) or not signature:
        raise ValueError('A nonempty scientific signature is required')
    gallery, queries = ids(pool.gallery), ids(list(pool))
    if (not set(map(int, queries)) <= set(map(int, gallery))
            or pool.candidates_per_query != len(gallery) - 1 or len(gallery) < 2):
        raise ValueError('Expected a complete frozen except-self pool')
    for q in queries:
        _, c = full_candidates(int(q), pool[int(q)], gallery, set(map(int, queries)))
        if not np.array_equal(c, gallery[gallery != q]):
            raise ValueError('Pool candidates must follow frozen gallery order')
    split = pool.split_name
    if (split not in ('valid', 'test') or not isinstance(pool.protocol_signature, str)
            or not pool.protocol_signature):
        raise ValueError('Invalid frozen split/protocol identity')
    return {'version': VERSION, 'signature': signature, 'split': split,
            'protocol_signature': pool.protocol_signature,
            'query_ids': queries.tolist(), 'gallery_ids': gallery.tolist(),
            'candidate_rule': 'all_gallery_except_query', 'components': list(COMPONENTS),
            'dtype': 'float64', 'shape': [len(queries), len(gallery)]}


def _read_row(folder, plan, plan_hash, query, candidates):
    marker = read_json(folder / 'completed.json')
    if (marker.get('status') != 'completed' or marker.get('signature') != plan['signature']
            or marker.get('plan_sha256') != plan_hash or marker.get('query_id') != query
            or marker.get('candidate_ids_digest') != digest(candidates.tolist())
            or marker.get('row_sha256') != file_hash(folder / 'row.npz')):
        raise ValueError('Sealed component row identity/hash differs: ' + str(folder))
    with np.load(folder / 'row.npz', allow_pickle=False) as z:
        if set(z.files) != {'ids', 'f', 'g', 'w'}:
            raise ValueError('Unexpected row fields')
        if z['ids'].dtype != np.int64 or not np.array_equal(z['ids'], candidates):
            raise ValueError('Row candidate IDs/order changed')
        values = {name: unit_scores(z[field], candidates.shape, name, strict_dtype=True).copy()
                  for name, field in zip(COMPONENTS, ('f', 'g', 'w'))}
    return values, marker


class FullScoreCache:
    """Strict read-only finished cache. scores() accepts a whole candidate row."""
    def __init__(self, folder):
        self.folder = Path(folder)
        self.marker = self.folder / 'completed' / 'completed.json'
        self.completion = read_json(self.marker)
        self.plan = read_json(self.folder / 'plan.json')
        p, m = self.plan, self.completion
        self.query_ids, self.gallery_ids = ids(p['query_ids']), ids(p['gallery_ids'])
        self.signature, self.split = p['signature'], p['split']
        self.protocol_signature = p['protocol_signature']
        if (p.get('version') != VERSION or p.get('dtype') != 'float64'
                or p.get('candidate_rule') != 'all_gallery_except_query'
                or p.get('components') != list(COMPONENTS)
                or self.split not in ('valid', 'test') or not isinstance(self.protocol_signature, str)
                or not self.protocol_signature or not isinstance(self.signature, str) or not self.signature
                or p.get('shape') != [len(self.query_ids), len(self.gallery_ids)]
                or not set(self.query_ids) <= set(self.gallery_ids)
                or len(self.gallery_ids) < 2 or m.get('status') != 'completed'
                or m.get('signature') != self.signature
                or m.get('plan_sha256') != file_hash(self.folder / 'plan.json')
                or m.get('plan_digest') != digest(p)
                or m.get('query_count') != len(self.query_ids)
                or m.get('gallery_count') != len(self.gallery_ids)
                or m.get('candidate_count') != len(self.gallery_ids) - 1
                or m.get('pairs_per_component') != len(self.query_ids) * (len(self.gallery_ids) - 1)
                or set(m.get('row_marker_hashes', {})) != set(map(str, self.query_ids))):
            raise ValueError('Completed full-score cache provenance differs')
        expected = {name + '.npy' for name in COMPONENTS} | {'query_ids.npy', 'gallery_ids.npy'}
        if set(m.get('files', {})) != expected:
            raise ValueError('Finished cache omits/introduces files')
        for name, sha in m['files'].items():
            path = self.marker.parent / name
            if path.is_symlink() or file_hash(path) != sha:
                raise ValueError('Finished cache file hash changed: ' + str(path))
        for name, expected_ids in [('query_ids', self.query_ids), ('gallery_ids', self.gallery_ids)]:
            actual = np.load(self.marker.parent / (name + '.npy'), allow_pickle=False)
            if actual.dtype != np.int64 or not np.array_equal(actual, expected_ids):
                raise ValueError('Finished original ID identity/order differs')
        self.rows = {int(q): i for i, q in enumerate(self.query_ids)}
        self.columns = {int(q): i for i, q in enumerate(self.gallery_ids)}
        self.query_ids.setflags(write=False)
        self.gallery_ids.setflags(write=False)
        self.matrices = {}
        for name in COMPONENTS:
            a = np.load(self.marker.parent / (name + '.npy'), mmap_mode='r', allow_pickle=False)
            validate_matrix(a, self.query_ids, self.gallery_ids)
            self.matrices[name] = a

    def scores(self, query, candidate_ids):
        q, c = full_candidates(query, candidate_ids, self.gallery_ids, self.rows)
        columns = [self.columns[int(x)] for x in c]
        return {name: np.asarray(a[self.rows[q], columns]) for name, a in self.matrices.items()}


def build_cache(folder, signature, pool, scorer, wbbs_provider, *, audit_callback=None):
    """Compute each full row once, seal rows, atomically publish all matrices."""
    folder = Path(folder)
    plan = _pool_plan(signature, pool)
    if (folder / 'completed').exists():
        result = FullScoreCache(folder)
        if result.plan != plan:
            raise ValueError('Completed cache scientific plan changed')
        return result, result.completion
    if folder.exists() and not (folder / 'plan.json').exists() and any(folder.iterdir()):
        raise ValueError('Refuse to adopt an unsealed existing cache directory')
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'plan.json').exists():
        if read_json(folder / 'plan.json') != plan:
            raise ValueError('Incomplete cache scientific plan changed')
    else:
        write_json(folder / 'plan.json', plan)
    plan_hash = file_hash(folder / 'plan.json')
    gallery, queries = np.array(plan['gallery_ids'], dtype=np.int64), np.array(plan['query_ids'], dtype=np.int64)
    rows = folder / 'rows'
    rows.mkdir(exist_ok=True)
    row_hashes = {}
    for q_ in queries:
        q = int(q_)
        c = gallery[gallery != q]
        row = rows / str(q)
        if row.exists():
            _read_row(row, plan, plan_hash, q, c)
        else:
            raw = scorer.components(q, c)
            values = {'F': unit_scores(raw['optimal_f_score'], c.shape, 'F'),
                      'G': unit_scores(raw['global_score'], c.shape, 'G'),
                      'WBBS': unit_scores(wbbs_provider.scores(q, c), c.shape, 'WBBS')}
            evidence = audit_callback(q, c, values) if audit_callback is not None else None
            attempt = Path(tempfile.mkdtemp(prefix=f'.attempt_{q}_', dir=rows))
            with (attempt / 'row.npz').open('xb') as handle:
                np.savez(handle, ids=c, f=values['F'], g=values['G'], w=values['WBBS'])
            marker = {'status': 'completed', 'signature': signature, 'plan_sha256': plan_hash,
                      'query_id': q, 'candidate_ids_digest': digest(c.tolist()),
                      'row_sha256': file_hash(attempt / 'row.npz'), 'audit': evidence}
            write_json(attempt / 'completed.json', marker)
            if row.exists():
                raise ValueError('Concurrent row publication detected; attempt preserved')
            attempt.rename(row)
        row_hashes[str(q)] = file_hash(row / 'completed.json')
    attempt = Path(tempfile.mkdtemp(prefix='.attempt_matrices_', dir=folder))
    matrices = {name: np.lib.format.open_memmap(attempt / (name + '.npy'), mode='w+',
                dtype=np.float64, shape=tuple(plan['shape'])) for name in COMPONENTS}
    for a in matrices.values():
        a[:] = np.nan
    for i, q_ in enumerate(queries):
        q = int(q_)
        mask = gallery != q
        values, _ = _read_row(rows / str(q), plan, plan_hash, q, gallery[mask])
        for name in COMPONENTS:
            matrices[name][i, mask] = values[name]
    for a in matrices.values():
        validate_matrix(a, queries, gallery)
        a.flush()
    matrices.clear()
    for name, a in [('query_ids', queries), ('gallery_ids', gallery)]:
        with (attempt / (name + '.npy')).open('xb') as handle:
            np.save(handle, a, allow_pickle=False)
    marker = {'status': 'completed', 'signature': signature, 'plan_sha256': plan_hash,
              'plan_digest': digest(plan), 'query_count': len(queries), 'gallery_count': len(gallery),
              'candidate_count': len(gallery) - 1, 'pairs_per_component': len(queries) * (len(gallery) - 1),
              'files': {p.name: file_hash(p) for p in sorted(attempt.glob('*.npy'))},
              'row_marker_hashes': row_hashes,
              'validation': 'incomplete resume validates each row; finished open validates every final matrix and ID file'}
    write_json(attempt / 'completed.json', marker)
    if (folder / 'completed').exists():
        raise ValueError('Concurrent matrix publication detected; attempt preserved')
    attempt.rename(folder / 'completed')
    return FullScoreCache(folder), marker
