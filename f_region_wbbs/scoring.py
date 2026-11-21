"""Exact regional WBBS counts and frozen C/structure/area score fusion.

Point descriptors keep full raster order and counts. Only the existing exact
search backend removes duplicate rows internally, retaining first-row ties.
No labels, point sampling, approximate search, or candidate truncation enter.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
import time

import numpy as np

from fgw_fusion.wbbs_backend import Scorer as ExactBBP, SETTINGS, original_module
from f_region_evidence.scoring import (MATCHING, Provider as JProvider,
    cache_layout, score_components as j_score_components, batched_score)
from f_region_structure.scoring import _indices

MODES = ('all_points', 'wbbs_denoised')
COUNT_ALPHAS = (0., 5.)
CNN_WEIGHTS = (.25, .4, .6)
TOTAL_DIES = 2061
MAX_POINTS = np.iinfo(np.uint16).max
COORD_DTYPE = np.dtype([('row', np.int64), ('column', np.int64)])


def checked_spec(spec):
    if isinstance(spec, Mapping) and dict(spec) == {'family': 'j_baseline'}:
        return dict(spec)
    keys = {'family', 'point_mode', 'count_alpha', 'cnn_weight', 'area_weight'}
    if (not isinstance(spec, Mapping) or set(spec) != keys or spec['family'] != 'wbbs'
            or spec['point_mode'] not in MODES):
        raise ValueError('Expected the frozen regional J or WBBS specification')
    for key, allowed in (('count_alpha', COUNT_ALPHAS), ('cnn_weight', CNN_WEIGHTS),
                         ('area_weight', (.25,))):
        value = spec[key]
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value not in allowed):
            raise ValueError('Unfrozen regional WBBS parameter: '+key)
    return dict(spec)


def _point_sets(cache):
    ids, offsets, area = cache_layout(cache, max_regions=4)
    n = int(offsets[-1])
    if np.any(area != np.rint(area)) or np.any(area > MAX_POINTS):
        raise ValueError('Original region point counts must be exact uint16-sized integers')
    sources = cache.get('point_sets')
    if not isinstance(sources, Mapping) or set(sources) != set(MODES):
        raise ValueError('Both frozen all-point and original global-denoised point sets are required')
    output = {}
    for mode in MODES:
        source = sources[mode]
        off = np.asarray(source['offsets'])
        desc = np.asarray(source['descriptors'])
        xy = np.asarray(source['coordinates'])
        if (desc.dtype != np.float64 or desc.ndim != 2 or desc.shape[1] != 3
                or not np.isfinite(desc).all() or np.any(desc < 0.) or np.any(desc > 1.)
                or off.dtype.kind not in 'iu' or off.shape != (n+1,)
                or off[0] != 0 or off[-1] != len(desc) or np.any(off > len(desc))
                or xy.dtype.kind not in 'iu' or xy.shape != (len(desc), 2)
                or np.any(xy < 0) or np.any(xy > np.iinfo(np.int64).max)):
            raise ValueError('Invalid float64 regional point descriptors, coordinates, or offsets')
        off = off.astype(np.int64, copy=False)
        counts = np.diff(off)
        if (np.any(counts < 0) or np.any(counts > MAX_POINTS)
                or np.any(counts > area) or (mode == 'all_points' and not np.array_equal(counts, area))):
            raise ValueError('Point counts do not preserve the complete frozen original regions')
        xy = np.ascontiguousarray(xy, dtype=np.int64)
        for token in range(n):
            first, last = off[token:token+2]
            positions = xy[first:last]
            ordered = ((positions[1:, 0] > positions[:-1, 0])
                       | ((positions[1:, 0] == positions[:-1, 0])
                          & (positions[1:, 1] > positions[:-1, 1])))
            if not ordered.all():
                raise ValueError('Each region must retain unique points in original raster order')
        output[mode] = {'offsets': off, 'descriptors': desc, 'coordinates': xy, 'counts': counts}
    full, kept = (output[mode] for mode in MODES)
    for token in range(n):
        a, b = full['offsets'][token:token+2]; c, d = kept['offsets'][token:token+2]
        xy, selected = full['coordinates'][a:b], kept['coordinates'][c:d]
        positions = np.searchsorted(xy.view(COORD_DTYPE).ravel(), selected.view(COORD_DTYPE).ravel())
        if (np.any(positions >= len(xy)) or not np.array_equal(xy[positions], selected)
                or full['descriptors'][a:b][positions].tobytes() != kept['descriptors'][c:d].tobytes()):
            raise ValueError('Global-denoised rows must be an unchanged raster subset of all-point rows')
    return output


def _token(value, n):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not 0 <= int(value) < n):
        raise ValueError('Unknown original region token')
    return int(value)


class RegionScorer:
    """One exact immutable KD backend per point population; reusable BBP counts."""
    def __init__(self, cache):
        started = time.perf_counter()
        self.point_sets = _point_sets(cache)
        self.n = len(cache['features'])
        self.backends = {mode: ExactBBP(np.arange(self.n, dtype=np.int64), source['offsets'],
                                       source['descriptors'], SETTINGS)
                         for mode, source in self.point_sets.items()}
        self.build_seconds = time.perf_counter()-started
        self.calls = {mode: 0 for mode in MODES}
        self.seconds = {mode: 0. for mode in MODES}

    def count(self, query_token, candidate_token, mode):
        if mode not in MODES:
            raise ValueError('Unknown point population')
        q, c = _token(query_token, self.n), _token(candidate_token, self.n)
        started = time.perf_counter()
        count = self.backends[mode]._count(q, c)
        self.calls[mode] += 1
        self.seconds[mode] += time.perf_counter()-started
        if not 0 <= count <= min(self.point_sets[mode]['counts'][q], self.point_sets[mode]['counts'][c]):
            raise ValueError('BBP count exceeds full original point counts')
        return count

    def count_matrix(self, query_tokens, candidate_tokens, mode):
        q, c = _indices(query_tokens, self.n), _indices(candidate_tokens, self.n)
        if mode not in MODES:
            raise ValueError('Unknown point population')
        result = np.empty((len(q), len(c)), dtype=np.uint16)
        for i, qi in enumerate(q):
            for j, ci in enumerate(c):
                result[i, j] = self.count(qi, ci, mode)
        return result

    def reference_pairs(self, query_token, candidate_token, mode):
        """Original-raster local point index pairs, for small explanations/audits."""
        if mode not in MODES:
            raise ValueError('Unknown point population')
        q, c = _token(query_token, self.n), _token(candidate_token, self.n)
        backend = self.backends[mode]
        return original_module().best_buddies(backend._original_rows(q), backend._original_rows(c),
                                             mu=1., block_size=256)

    def statistics(self):
        return {'backend': 'frozen_exact_ckdtree_eps0_first_raster_cdist_near_tie_replay',
                'build_seconds': self.build_seconds, 'regions': self.n,
                'point_modes': {mode: {'point_rows': int(source['counts'].sum()),
                    'unique_descriptor_rows': int(self.backends[mode].unique_counts.sum()),
                    'empty_regions': int((source['counts'] == 0).sum()),
                    'singleton_regions': int((source['counts'] == 1).sum()),
                    'maximum_region_points': int(source['counts'].max(initial=0)),
                    'count_calls': self.calls[mode], 'count_seconds': self.seconds[mode],
                    'near_tie_fallback_rows': int(self.backends[mode].fallback_rows)}
                    for mode, source in self.point_sets.items()}}


def wbbs_scores(counts, query_counts, candidate_counts, alpha):
    """Same scalar WBBS arithmetic, using full counts and one shared B matrix."""
    if (isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real)
            or alpha not in COUNT_ALPHAS):
        raise ValueError('Only frozen count penalties alpha0 and alpha5 are supported')
    b, q, c = map(np.asarray, (counts, query_counts, candidate_counts))
    if (b.ndim != 2 or q.ndim != 1 or c.ndim != 1 or b.shape != (len(q), len(c))
            or any(x.dtype.kind not in 'iu' for x in (b, q, c))
            or any(np.any(x < 0) or np.any(x > MAX_POINTS) for x in (b, q, c))):
        raise ValueError('Invalid BBP counts or complete region point counts')
    q, c = q.astype(np.int64, copy=False), c.astype(np.int64, copy=False)
    minimum = np.minimum(q[:, None], c[None])
    if np.any(b > minimum):
        raise ValueError('BBP count exceeds the smaller full point count')
    base = np.divide(b, minimum, out=np.zeros(b.shape, dtype=np.float64), where=minimum > 0)
    if alpha == 0.:
        return base
    difference = np.abs(q[:, None]-c[None])
    # Scalar np.exp and operand order follow unchanged WBBS score_pair exactly.
    unique, inverse = np.unique(difference, return_inverse=True)
    penalty = np.array([np.exp(-float(alpha)*int(d)/TOTAL_DIES) for d in unique])
    return base*penalty[inverse].reshape(difference.shape)


def score_components(parts, spec):
    spec = checked_spec(spec)
    if spec['family'] == 'j_baseline':
        return j_score_components(parts)
    c, a = np.asarray(parts['cosine']), np.asarray(parts['area_similarity'])
    w = np.asarray(parts['wbbs_scores'][spec['point_mode']][float(spec['count_alpha'])])
    if (c.ndim != 2 or a.shape != c.shape or w.shape != c.shape
            or any(x.dtype.kind not in 'fi' or not np.isfinite(x).all()
                   or np.any(x < 0.) or np.any(x > 1.) for x in (c, a, w))):
        raise ValueError('Invalid C/W/A region pair matrices')
    alpha, beta = spec['cnn_weight'], spec['area_weight']
    mixed = alpha*c + (1.-alpha)*w
    return np.clip((1.-beta)*mixed + beta*a, 0., 1.)


class Provider(JProvider):
    """Read full point-count caches, preserving original directional F matching."""
    def __init__(self, cache, pool, count_matrices=None, count_query_tokens=None, matching=None):
        cache_layout(cache, max_regions=4)
        super().__init__(cache, pool, matching=matching)
        self.point_sets = _point_sets(cache)
        self.count_matrices = count_matrices
        self.counter = None
        self.count_rows = None
        if count_matrices is None:
            if count_query_tokens is not None:
                raise ValueError('Count row identities require corresponding count matrices')
        else:
            if not isinstance(count_matrices, Mapping) or set(count_matrices) != set(MODES):
                raise ValueError('Both sealed count populations are required')
            tokens = (np.arange(self.engine.n, dtype=np.int64) if count_query_tokens is None
                      else _indices(count_query_tokens, self.engine.n))
            for matrix in count_matrices.values():
                if np.asarray(matrix).dtype != np.uint16 or matrix.shape != (len(tokens), self.engine.n):
                    raise ValueError('Count matrices must be uint16 with explicit query-token rows and full gallery columns')
            self.count_rows = {int(token): row for row, token in enumerate(tokens)}

    def parts(self, query):
        qt = self.tokens(query)
        parts = self.engine.components(qt)
        parts['bb_counts'], parts['wbbs_scores'] = {}, {}
        if self.count_matrices is None and self.counter is None:
            self.counter = RegionScorer(self.cache)
        for mode in MODES:
            if self.count_matrices is None:
                b = self.counter.count_matrix(qt, np.arange(self.engine.n, dtype=np.int64), mode)
            else:
                if any(int(t) not in self.count_rows for t in qt):
                    raise ValueError('The count cache lacks a required query region')
                b = self.count_matrices[mode][[self.count_rows[int(t)] for t in qt]]
            counts = self.point_sets[mode]['counts']
            parts['bb_counts'][mode] = b
            parts['wbbs_scores'][mode] = {alpha: wbbs_scores(b, counts[qt], counts, alpha)
                                          for alpha in COUNT_ALPHAS}
        return parts

    def score(self, query, spec, parts=None):
        spec = checked_spec(spec)
        if parts is None and spec['family'] == 'j_baseline':
            parts = self.engine.components(self.tokens(query))
        pair = score_components(self.parts(query) if parts is None else parts, spec)
        qt = self.tokens(query)
        if pair.shape != (len(qt), self.engine.n):
            raise ValueError('Regional scoring requires the complete gallery matrix')
        ids = np.asarray(self.pool[query])
        if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or int(query) in ids
                or len(np.unique(ids)) != len(ids) or not set(map(int, ids)) <= set(self.positions)):
            raise ValueError('Candidate identities must be unique eligible gallery samples')
        rows = self.padded[[self.positions[int(c)] for c in ids]]
        return ids, batched_score(pair, rows, self.area[qt])

    def explain(self, query, candidate, spec, parts=None):
        """Original greedy region correspondences, with W/full-count evidence."""
        spec = checked_spec(spec)
        if int(candidate) not in set(map(int, self.pool[query])):
            raise ValueError('Explanation candidate is outside the eligible pool')
        qt, ct = self.tokens(query), self.tokens(candidate)
        if parts is None:
            parts = (self.engine.components(qt) if spec['family'] == 'j_baseline'
                     else self.parts(query))
        raw = score_components(parts, spec)
        if raw.shape != (len(qt), self.engine.n):
            raise ValueError('Explanation requires complete-gallery components')
        pair, weights = raw[:, ct], np.sqrt(self.area[qt])
        denominator = float(weights.sum())
        used_q, used_c, matches, total = set(), set(), [], 0.
        for flat in np.argsort(-pair.ravel(), kind='stable'):
            qi, ci = divmod(int(flat), len(ct))
            value = float(pair[qi, ci])
            if value <= 0.:
                break
            if qi in used_q or ci in used_c:
                continue
            weighted = float(weights[qi]*value)
            total += weighted
            used_q.add(qi); used_c.add(ci)
            row = {'query_token': int(qt[qi]), 'candidate_token': int(ct[ci]),
                   'query_local_index': qi, 'candidate_local_index': ci,
                   'pair_score': value, 'C': float(parts['cosine'][qi, ct[ci]]),
                   'A': float(parts['area_similarity'][qi, ct[ci]]),
                   'query_weight': float(weights[qi]), 'weighted_score': weighted,
                   'contribution': weighted/denominator}
            if spec['family'] == 'j_baseline':
                row['T'] = float(parts['structure_similarity'][qi, ct[ci]])
            else:
                mode, alpha = spec['point_mode'], spec['count_alpha']
                counts = self.point_sets[mode]['counts']
                nq, nc = int(counts[qt[qi]]), int(counts[ct[ci]])
                b = int(parts['bb_counts'][mode][qi, ct[ci]])
                row.update(W=float(parts['wbbs_scores'][mode][alpha][qi, ct[ci]]),
                           mutual_point_pairs=b, query_points=nq, candidate_points=nc,
                           BBP=b/min(nq, nc) if min(nq, nc) else 0.,
                           count_alpha=float(alpha),
                           count_penalty=float(np.exp(-float(alpha)*abs(nq-nc)/TOTAL_DIES)))
            matches.append(row)
        return {'query_id': int(query), 'candidate_id': int(candidate), 'spec': spec,
                'score': total/denominator if denominator else 0.,
                'query_weight_denominator': denominator, 'matches': matches,
                'unmatched_query_tokens': [int(t) for i, t in enumerate(qt) if i not in used_q],
                'unmatched_candidate_tokens': [int(t) for i, t in enumerate(ct) if i not in used_c]}
