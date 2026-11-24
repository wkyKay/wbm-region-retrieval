"""Rotation-CNN primary scores and traceable regional correspondences.

The original operation order, greedy matching and complete query sqrt-area
denominator are retained. Candidate filtering never changes pair computation.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np

from f_region_structure.scoring import _features, _indices, _probabilities
from wbm_retrieval.diagnostics import batched_map_scores

MATCHING = {'lambda_pos': 0., 'lambda_scale': 0., 'shape_threshold': 0.,
            'score_threshold': 0., 'weight': 'sqrt_area'}
REJECTIONS = ('none', 'q75', 'q90', 'q95')
SPEC = {'family': 'add', 'alpha': .4, 'beta': .15}
TAU_ONE_POLICY = 'zero_evidence_when_background_quantile_equals_one'


def _arrays(cache):
    h = _features(cache['features'], width=512)
    n = len(h)
    p = _probabilities(cache['joint_all'], n, 128, 'all-point J', False)
    r = np.asarray(cache['relative_area'], dtype=np.float64)
    if (r.shape != (n,) or not np.isfinite(r).all() or np.any(r <= 0.)
            or np.any(r > 1.)):
        raise ValueError('Relative areas must be in (0,1]')
    return h, r, np.sqrt(p)


def cache_layout(cache, *, max_regions=8):
    """Validate real tokens and their original-wafer ownership, never labels."""
    ids, offsets = np.asarray(cache['map_ids']), np.asarray(cache['offsets'])
    n = len(cache['features'])
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or np.any(ids < 0)
            or len(np.unique(ids)) != len(ids) or offsets.dtype.kind not in 'iu'
            or offsets.shape != (len(ids)+1,) or offsets[0] != 0
            or offsets[-1] != n or np.any(np.diff(offsets.astype(np.int64)) < 0)
            or np.any(np.diff(offsets.astype(np.int64)) > max_regions)):
        raise ValueError('Invalid map IDs or region offsets')
    area = np.asarray(cache['area'], dtype=np.float64)
    if (area.shape != (n,) or not np.isfinite(area).all() or np.any(area <= 0.)):
        raise ValueError('Actual region areas must be positive')
    owners = np.repeat(ids, np.diff(offsets.astype(np.int64)))
    if 'owners' in cache and not np.array_equal(np.asarray(cache['owners']), owners):
        raise ValueError('Region owners disagree with map offsets')
    return ids.astype(np.int64, copy=False), offsets.astype(np.int64, copy=False), area


def score_components(parts):
    """Preserve 0.85 * (0.4*C + 0.6*T) + 0.15*A byte-for-byte."""
    c, t, a = (np.asarray(parts[k]) for k in
               ('cosine', 'structure_similarity', 'area_similarity'))
    if (c.ndim != 2 or t.shape != c.shape or a.shape != c.shape
            or any(x.dtype.kind not in 'fi' or not np.isfinite(x).all()
                   or np.any(x < 0.) or np.any(x > 1.) for x in (c, t, a))):
        raise ValueError('Invalid regional score components')
    mixed = .4 * c + (1. - .4) * t
    return np.clip((1. - .15) * mixed + .15 * a, 0., 1.)


class RegionPairScorer:
    """Compute only the active C/T/A components, without gallery entropy.

    An optional reference cache is used for train-only calibration. With no
    reference, the full-gallery matrix shape matches the frozen scorer.
    """
    def __init__(self, cache, reference=None):
        self.features, self.relative_area, self.roots = _arrays(cache)
        self.n = len(self.features)
        if reference is None:
            self.reference_features = self.features
            self.reference_area = self.relative_area
            self.reference_roots = self.roots
        else:
            self.reference_features, self.reference_area, self.reference_roots = _arrays(reference)

    def components(self, query_indices):
        q = _indices(query_indices, self.n)
        # Preserve the source feature dtype and its exact GEMM operation order.
        cosine = np.clip(self.features[q] @ self.reference_features.T, 0, 1)
        distance = np.sqrt(np.maximum(.5 * np.square(
            self.roots[q, None] - self.reference_roots[None]).sum(-1), 0.))
        distance = np.clip(distance, 0, 1)
        area_distance = .5 * np.abs(np.log(self.relative_area[q])[:, None]
                                   - np.log(self.reference_area)[None])
        return {'cosine': cosine, 'joint_distance': distance,
                'structure_similarity': np.exp(-.5 * distance),
                'area_similarity': np.exp(-area_distance / 1.5)}

    def scores(self, query_indices):
        return score_components(self.components(query_indices))


def net_evidence(scores, tau=None):
    """A null threshold consumes evidence, never renormalizes matched regions.

    tau==1 has no score strictly above background and yields zero evidence.
    The none branch returns the original matrix without any arithmetic.
    """
    scores = np.asarray(scores)
    if (scores.ndim != 2 or scores.dtype.kind not in 'fi'
            or not np.isfinite(scores).all() or np.any(scores < 0.) or np.any(scores > 1.)):
        raise ValueError('Pair scores must be a finite matrix in [0,1]')
    if tau is None:
        return scores
    tau = np.asarray(tau, dtype=np.float64)
    if (tau.shape != (len(scores),) or not np.isfinite(tau).all()
            or np.any(tau < 0.) or np.any(tau > 1.)):
        raise ValueError('One background threshold in [0,1] is required per query region')
    result = np.zeros(scores.shape, dtype=np.float64)
    np.divide(scores - tau[:, None], 1. - tau[:, None], out=result,
              where=tau[:, None] < 1.)
    return np.maximum(result, 0.)


def batched_score(pair_scores, candidate_indices, query_area):
    """Original greedy matcher, now with up to eight real regions per map."""
    pair_scores = np.asarray(pair_scores)
    candidate_indices = np.asarray(candidate_indices)
    area = np.asarray(query_area, dtype=np.float64)
    if (pair_scores.ndim != 2 or len(pair_scores) > 8
            or candidate_indices.ndim != 2 or candidate_indices.shape[1] > 8
            or candidate_indices.dtype.kind not in 'iu'
            or np.any(candidate_indices < -1) or np.any(candidate_indices >= pair_scores.shape[1])
            or area.shape != (len(pair_scores),) or not np.isfinite(area).all()
            or np.any(area <= 0.) or not np.isfinite(pair_scores).all()
            or np.any(pair_scores < 0.) or np.any(pair_scores > 1.)):
        raise ValueError('Invalid up-to-eight-region matcher input')
    return batched_map_scores(pair_scores, candidate_indices, area, MATCHING)


class Provider:
    def __init__(self, cache, pool, matching=None, thresholds=None):
        if matching is not None and dict(matching) != MATCHING:
            raise ValueError('The frozen original greedy sqrt-area matcher is required')
        self.cache, self.pool, self.matching = cache, pool, dict(MATCHING)
        ids, self.offsets, self.area = cache_layout(cache)
        self.positions = {int(sid): row for row, sid in enumerate(ids)}
        gallery = np.asarray(pool.gallery)
        if (gallery.ndim != 1 or gallery.dtype.kind not in 'iu'
                or len(np.unique(gallery)) != len(gallery)
                or set(map(int, gallery)) != set(self.positions)
                or not set(map(int, pool)) <= set(self.positions)):
            raise ValueError('Cache must contain the complete gallery exactly once')
        self.padded = np.full((len(ids), int(np.max(np.diff(self.offsets), initial=0))),
                              -1, dtype=np.int64)
        for row in range(len(ids)):
            a, b = self.offsets[row:row+2]
            self.padded[row, :b-a] = np.arange(a, b)
        self.engine = RegionPairScorer(cache)
        self.thresholds = {}
        if thresholds is not None:
            if not isinstance(thresholds, Mapping) or not set(thresholds) <= set(REJECTIONS[1:]):
                raise ValueError('Unknown background quantile')
            for name, value in thresholds.items():
                value = np.asarray(value, dtype=np.float64)
                if (value.shape != (self.engine.n,) or not np.isfinite(value).all()
                        or np.any(value < 0.) or np.any(value > 1.)):
                    raise ValueError('Thresholds must cover every gallery token')
                self.thresholds[name] = value

    def tokens(self, query):
        row = self.positions[int(query)]
        a, b = self.offsets[row:row+2]
        return np.arange(a, b, dtype=np.int64)

    def parts(self, query):
        return self.engine.components(self.tokens(query))

    def _tau(self, query, rejection):
        if rejection not in REJECTIONS:
            raise ValueError('Unknown rejection variant')
        if rejection == 'none':
            return None
        if rejection not in self.thresholds:
            raise ValueError('Requested train background quantile has not been calibrated')
        return self.thresholds[rejection][self.tokens(query)]

    def score(self, query, rejection='none', parts=None):
        ids = np.asarray(self.pool[query])
        if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or int(query) in ids
                or len(np.unique(ids)) != len(ids) or not set(map(int, ids)) <= set(self.positions)):
            raise ValueError('Invalid candidate identities')
        qt = self.tokens(query)
        pair = score_components(self.parts(query) if parts is None else parts)
        if pair.shape != (len(qt), self.engine.n):
            raise ValueError('Pair components must use the complete gallery')
        evidence = net_evidence(pair, self._tau(query, rejection))
        rows = self.padded[[self.positions[int(c)] for c in ids]]
        return ids, batched_score(evidence, rows, self.area[qt])

    def explain(self, query, candidate, rejection='none', parts=None):
        """Return sparse accepted correspondences and explicit unmatched tokens."""
        if int(candidate) not in set(map(int, self.pool[query])):
            raise ValueError('Explanation candidate is outside the eligible pool')
        qt, ct = self.tokens(query), self.tokens(candidate)
        parts = self.parts(query) if parts is None else parts
        raw = score_components(parts)
        if raw.shape != (len(qt), self.engine.n):
            raise ValueError('Explanation requires complete-gallery components')
        tau = self._tau(query, rejection)
        pair = net_evidence(raw, tau)[:, ct]
        weights = np.sqrt(self.area[qt])
        denominator = float(weights.sum())
        used_q, used_c, matches = set(), set(), []
        total = 0.
        for flat in np.argsort(-pair.ravel(), kind='stable'):
            qi, ci = divmod(int(flat), len(ct))
            value = float(pair[qi, ci])
            if value <= 0.:
                break
            if qi in used_q or ci in used_c:
                continue
            weighted = float(weights[qi] * value)
            total += weighted
            used_q.add(qi)
            used_c.add(ci)
            matches.append({'query_token': int(qt[qi]), 'candidate_token': int(ct[ci]),
                            'query_local_index': qi, 'candidate_local_index': ci,
                            'raw_score': float(raw[qi, ct[ci]]), 'net_evidence': value,
                            'tau': None if tau is None else float(tau[qi]),
                            'C': float(parts['cosine'][qi, ct[ci]]),
                            'T': float(parts['structure_similarity'][qi, ct[ci]]),
                            'A': float(parts['area_similarity'][qi, ct[ci]]),
                            'query_weight': float(weights[qi]),
                            'weighted_evidence': weighted,
                            'contribution': weighted / denominator})
        return {'query_id': int(query), 'candidate_id': int(candidate), 'rejection': rejection,
                'score': total / denominator if denominator else 0.,
                'query_weight_denominator': denominator, 'matches': matches,
                'unmatched_query_tokens': [int(t) for i, t in enumerate(qt) if i not in used_q],
                'unmatched_candidate_tokens': [int(t) for i, t in enumerate(ct) if i not in used_c]}
