"""Label-free background quantiles from one fixed train-only region reference."""
from __future__ import annotations

import hashlib
import numpy as np

from .scoring import RegionPairScorer, TAU_ONE_POLICY, cache_layout

QUANTILES = (.75, .90, .95)
REFERENCE_SIZE = 4096
REFERENCE_SEED = 2026
REFERENCE_RULE = 'sorted_train_owners_sample_without_replacement_then_one_uniform_region_per_owner'
QUANTILE_METHOD = 'linear'


def _identity(values):
    return hashlib.sha256(np.asarray(values, dtype='<i8').tobytes()).hexdigest()


def select_reference(train_cache, size=REFERENCE_SIZE, seed=REFERENCE_SEED):
    """Select distinct train wafers first, preventing fragmented-wafer bias.

    The caller binds and proves the cache is the frozen train split. Selection
    uses only IDs/offsets; labels, valid and test data are never consulted.
    """
    ids, offsets, _ = cache_layout(train_cache, max_regions=4)
    if (isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer))
            or size < 1 or isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError('Reference size and seed must be positive/nonnegative integers')
    positions = {int(sid): row for row, sid in enumerate(ids)}
    eligible = np.sort(ids[np.diff(offsets) > 0])
    if len(eligible) < size:
        raise ValueError('Insufficient distinct nonempty train wafers for the fixed reference')
    rng = np.random.default_rng(int(seed))
    owners = np.sort(rng.choice(eligible, size=int(size), replace=False))
    local = np.array([rng.integers(int(offsets[positions[int(sid)]+1]
                                      - offsets[positions[int(sid)]])) for sid in owners], dtype=np.int64)
    tokens = np.array([offsets[positions[int(sid)]] for sid in owners], dtype=np.int64) + local
    return {'token_indices': tokens, 'owner_ids': owners, 'local_indices': local,
            'size': int(size), 'seed': int(seed), 'rule': REFERENCE_RULE,
            'train_map_ids_sha256': _identity(ids), 'train_offsets_sha256': _identity(offsets),
            'reference_identity_sha256': _identity(np.column_stack((owners, local, tokens)))}


def _checked_reference(train_cache, reference):
    if not isinstance(reference, dict):
        raise ValueError('Expected a frozen train reference record')
    expected = select_reference(train_cache, reference.get('size'), reference.get('seed'))
    if set(reference) != set(expected):
        raise ValueError('Unexpected reference fields')
    for key, value in expected.items():
        if isinstance(value, np.ndarray):
            actual = np.asarray(reference[key])
            if actual.dtype.kind not in 'iu' or not np.array_equal(actual, value):
                raise ValueError('Reference identity differs from deterministic train selection')
        elif reference[key] != value:
            raise ValueError('Reference metadata differs from deterministic train selection')
    return expected


def calibrate(cache, train_cache, reference, quantiles=QUANTILES):
    """Return (thresholds, metadata), scoring each query against fixed train tokens.

    Quantiles are computed independently per query region, never fitted to the
    query split or its gallery score distribution. All three proposal families
    must receive the same frozen reference record.
    """
    quantiles = tuple(quantiles)
    if (quantiles != QUANTILES or any(isinstance(x, (bool, np.bool_)) for x in quantiles)):
        raise ValueError('Only the frozen q75/q90/q95 calibration is permitted')
    reference = _checked_reference(train_cache, reference)
    ids, offsets, _ = cache_layout(cache)
    train_ids = np.asarray(train_cache['map_ids'])
    if np.intersect1d(ids, train_ids).size:
        raise ValueError('Calibration queries must be disjoint from the train wafer population')
    tokens = reference['token_indices']
    selected = {name: np.asarray(train_cache[name])[tokens]
                for name in ('features', 'relative_area', 'joint_all')}
    engine = RegionPairScorer(cache, reference=selected)
    names = ('q75', 'q90', 'q95')
    thresholds = {name: np.empty(engine.n, dtype=np.float64) for name in names}
    for row in range(len(ids)):
        a, b = offsets[row:row+2]
        if a == b:
            continue
        values = engine.scores(np.arange(a, b, dtype=np.int64))
        result = np.quantile(values, quantiles, axis=1, method=QUANTILE_METHOD)
        for index, name in enumerate(names):
            thresholds[name][a:b] = result[index]
    metadata = {'schema_version': 1, 'reference': reference,
                'quantiles': dict(zip(names, quantiles)), 'quantile_method': QUANTILE_METHOD,
                'scope': 'per_query_region_against_fixed_train_reference_only',
                'labels_used': False, 'train_query_disjoint': True,
                'reference_shared_across_proposals_and_splits': True,
                'query_maps': len(ids), 'query_regions': engine.n,
                'region_reference_pairs': engine.n * reference['size'],
                'tau_equal_one_policy': TAU_ONE_POLICY,
                'tau_equal_one_counts': {name: int(np.sum(value == 1.))
                                        for name, value in thresholds.items()}}
    return thresholds, metadata
