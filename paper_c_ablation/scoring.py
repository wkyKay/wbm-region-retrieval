"""Frozen C1--C5 controls for the paper's regional J+F formula.

The primary and no-A branches call the historical arithmetic directly. All
variants share the original C/T/A matrices; only independent-J adds a second
Hellinger computation. No labels, fitting, calibration or parameter selection
are performed here.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real

import numpy as np

from f_region_area_fusion.pipeline import MATCHING as _OLD_MATCHING
from f_region_area_fusion.pipeline import Provider as AreaProvider
from f_region_area_fusion.scoring import _channels
from f_region_area_fusion.scoring import frozen_grid as area_grid
from f_region_area_fusion.scoring import score_components as area_score
from f_region_evidence.scoring import RegionPairScorer as FrozenChannels
from f_region_evidence.scoring import cache_layout
from f_region_structure.scoring import _probabilities, independent_distribution
from wbm_retrieval.diagnostics import batched_map_scores


MATCHING = dict(_OLD_MATCHING)
LEGACY_UNUSED_MATCHING = {'sigma_pos': .35, 'sigma_scale': .5}
VARIANTS = ('primary', 'no_c', 'no_t', 'no_a', 'independent_j', 'reuse_candidate')
PRIMARY_SPEC = area_grid()['add_a0p4_b0p25']
NO_AREA_SPEC = area_grid()['no_area_a0p4']


def normalize_matching(matching=None):
    """Accept only the five active settings or the exact seven-key legacy form.

    Historical geometry configurations retain sigma_pos=.35 and sigma_scale=.5.
    Both corresponding lambdas are zero, so the frozen regional scorer never
    uses those widths. Return a fresh active-settings copy without mutating the
    source configuration that remains bound to the historical signatures.
    """
    if matching is None:
        return dict(MATCHING)
    core, legacy = set(MATCHING), set(MATCHING) | set(LEGACY_UNUSED_MATCHING)
    if not isinstance(matching, Mapping) or set(matching) not in (core, legacy):
        raise ValueError('The original matcher settings must remain fixed; unknown or partial legacy keys')
    expected = MATCHING if set(matching) == core else {**MATCHING, **LEGACY_UNUSED_MATCHING}
    for key, target in expected.items():
        value = matching[key]
        good = (isinstance(value, str) and value == target) if isinstance(target, str) else (
            isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
            and math.isfinite(value) and value == target)
        if not good:
            raise ValueError('The original matcher settings must remain fixed: '+key)
    return dict(MATCHING)


def grid():
    """Fresh, ordered specifications; every declared control must be reported."""
    return {name: {'variant': name,
                   'distribution': 'independent' if name == 'independent_j' else 'joint',
                   'matching': 'reuse_candidate' if name == 'reuse_candidate' else 'greedy_one_to_one'}
            for name in VARIANTS}


def checked_spec(spec):
    """Reject extra keys and undeclared changes to a named control."""
    if isinstance(spec, Mapping):
        for expected in grid().values():
            if (set(spec) == set(expected)
                    and all(isinstance(spec[k], str) and spec[k] == v
                            for k, v in expected.items())):
                return expected
    raise ValueError('Score specification is outside the frozen six-variant grid')


def score_components(parts, spec):
    """Score actual region tokens with fixed weights and original arithmetic."""
    spec = checked_spec(spec)
    c, t, a = _channels(parts)
    name = spec['variant']
    if name in ('primary', 'reuse_candidate'):
        return area_score(parts, PRIMARY_SPEC)
    if name == 'no_a':
        return area_score(parts, NO_AREA_SPEC)
    if name == 'no_c':
        return np.clip((.45 * t + .25 * a) / .70, 0., 1.)
    if name == 'no_t':
        return np.clip((.30 * c + .25 * a) / .55, 0., 1.)
    try:
        independent = np.asarray(parts['independent_structure_similarity'])
    except (KeyError, TypeError) as error:
        raise ValueError('Independent-J requires its own structure similarity') from error
    # The old channel validator checks the new T's shape, mass-derived bounds,
    # finiteness and range without changing the original C/A array dtypes.
    independent_parts = dict(parts, structure_similarity=independent)
    return area_score(independent_parts, PRIMARY_SPEC)


def reuse_candidate_scores(pair, candidate_indices, query_area):
    """Each query region independently takes its best real candidate region.

    Candidate tokens may be reused. Argmax ties use the first candidate token,
    matching the existing token order. The denominator includes every selected
    query region, including rows with no positive evidence. Empty maps score 0.
    """
    pair = np.asarray(pair)
    rows = np.asarray(candidate_indices)
    areas = np.asarray(query_area, dtype=np.float64)
    if (pair.ndim != 2 or pair.dtype.kind not in 'fi'
            or not np.isfinite(pair).all() or np.any(pair < 0.) or np.any(pair > 1.)
            or len(pair) > 4 or rows.ndim != 2 or rows.shape[1] > 4
            or rows.dtype.kind not in 'iu' or np.any(rows < -1)
            or np.any(rows >= pair.shape[1]) or areas.shape != (len(pair),)
            or not np.isfinite(areas).all() or np.any(areas <= 0.)):
        raise ValueError('Invalid frozen Top-4 many-to-one matcher inputs')
    if not len(pair) or not rows.shape[1]:
        return np.zeros(len(rows), dtype=np.float64)
    scores = pair[:, np.maximum(rows, 0)].transpose(1, 0, 2)
    scores = np.where((rows >= 0)[:, None, :], scores, 0.)
    chosen = np.argmax(scores, axis=2)
    best = np.take_along_axis(scores, chosen[:, :, None], axis=2)[:, :, 0]
    weights = np.sqrt(areas)
    total = np.zeros(len(rows), dtype=np.float64)
    # Declare query-index summation order explicitly rather than depending on a
    # backend reduction. No successful-match-only renormalization is allowed.
    for q in range(len(pair)):
        total += weights[q] * best[:, q]
    return total / weights.sum()


class Provider(AreaProvider):
    """Original full-gallery provider with shared frozen channels and six specs."""

    def __init__(self, cache, pool, matching=None):
        matching = normalize_matching(matching)
        cache_layout(cache, max_regions=4)
        super().__init__(cache, pool, matching)
        # Same GEMM dtype, raw-area formula and direct Hellinger reduction as the
        # historical provider, without unused radial/retained/entropy matrices.
        self.engine = FrozenChannels(cache)
        independent = independent_distribution(cache['joint_all'])
        self.independent_probabilities = _probabilities(
            independent, self.engine.n, 128, 'independent all-point J', False)
        self.independent_roots = np.sqrt(self.independent_probabilities)

    def parts(self, query):
        qt = self.tokens(query)
        parts = self.engine.components(qt)
        root = self.independent_roots
        distance = np.sqrt(np.maximum(
            .5 * np.square(root[qt, None] - root[None]).sum(-1), 0.))
        distance = np.clip(distance, 0., 1.)
        parts['independent_joint_distance'] = distance
        parts['independent_structure_similarity'] = np.exp(-.5 * distance)
        parts['query_tokens'] = qt.copy()
        return parts

    def score(self, query, spec, parts=None):
        spec = checked_spec(spec)
        qt = self.tokens(query)
        ids = np.asarray(self.pool[query], dtype=np.int64)
        if (len(ids) != self.pool.candidates_per_query or int(query) in ids
                or len(np.unique(ids)) != len(ids)
                or set(map(int, ids)) != set(self.positions) - {int(query)}):
            raise ValueError('Candidate pool must retain all gallery samples except self')
        parts = self.parts(query) if parts is None else parts
        if ('query_tokens' in parts
                and not np.array_equal(np.asarray(parts['query_tokens']), qt)):
            raise ValueError('Shared parts belong to a different query')
        pair = score_components(parts, spec)
        if pair.shape != (len(qt), self.engine.n):
            raise ValueError('Shared parts must cover the complete gallery region tokens')
        rows = self.padded[[self.positions[int(c)] for c in ids]]
        if spec['matching'] == 'reuse_candidate':
            values = reuse_candidate_scores(pair, rows, self.cache['area'][qt])
        else:
            values = batched_map_scores(pair, rows, self.cache['area'][qt], self.matching)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite candidate score')
        return ids, values
