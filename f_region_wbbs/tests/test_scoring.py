import copy
import unittest
from unittest import mock

import numpy as np

from f_region_evidence.scoring import Provider as JProvider
from f_region_evidence.tests.test_scoring import fixture as region_fixture, scalar_greedy, Pool
from f_region_wbbs import scoring
from f_region_wbbs.state import grid


def fixture(counts=(2, 4, 0, 1), dtype=np.float64):
    cache, pool = region_fixture(counts, dtype)
    rng = np.random.default_rng(143)
    full, kept, full_xy, kept_xy = [], [], [], []
    for token, size in enumerate(cache['area']):
        desc = rng.random((size, 3))
        xy = np.column_stack((np.arange(size)//7, np.arange(size)%7)).astype(np.int64)
        # Include empty/singleton denoised regions without changing original area.
        which = np.arange(size)[::2] if token > 1 else np.arange(token)
        full.append(desc); full_xy.append(xy)
        kept.append(desc[which]); kept_xy.append(xy[which])
    cache['point_sets'] = {}
    for mode, desc, xy in [('all_points', full, full_xy), ('wbbs_denoised', kept, kept_xy)]:
        cache['point_sets'][mode] = {
            'offsets': np.r_[0, np.cumsum([len(v) for v in desc])].astype(np.int64),
            'descriptors': np.concatenate(desc) if desc else np.empty((0, 3), dtype=np.float64),
            'coordinates': np.concatenate(xy) if xy else np.empty((0, 2), dtype=np.int64)}
    return cache, pool


def reference(cache, mode, token):
    source = cache['point_sets'][mode]
    a, b = source['offsets'][token:token+2]
    return scoring.original_module().WaferFeatures(source['coordinates'][a:b],
                                                  source['descriptors'][a:b], b-a)


class ScoringTests(unittest.TestCase):
    def test_both_point_modes_all_pair_counts_scores_bitwise_original(self):
        cache, _ = fixture()
        engine = scoring.RegionScorer(cache)
        tokens = np.arange(engine.n)
        original = scoring.original_module()
        for mode in scoring.MODES:
            counts = engine.count_matrix(tokens, tokens, mode)
            np.testing.assert_array_equal(counts, counts.T)
            totals = np.diff(cache['point_sets'][mode]['offsets'])
            for alpha in scoring.COUNT_ALPHAS:
                actual = scoring.wbbs_scores(counts, totals, totals, alpha)
                for q in tokens:
                    for c in tokens:
                        qf, cf = reference(cache, mode, q), reference(cache, mode, c)
                        pairs = original.best_buddies(qf.descriptors, cf.descriptors, mu=1., block_size=256)
                        self.assertEqual(int(counts[q, c]), len(pairs))
                        np.testing.assert_array_equal(engine.reference_pairs(q, c, mode), pairs)
                        expected = original.score_pair(qf, cf, {**scoring.SETTINGS, 'alpha': alpha})
                        self.assertEqual(actual[q, c].tobytes(), np.float64(expected).tobytes())
        self.assertEqual(engine.statistics()['point_modes']['all_points']['count_calls'], engine.n**2)

    def test_duplicate_and_equal_distance_ties_keep_raster_and_full_counts(self):
        cache, _ = fixture((1, 1))
        for mode in scoring.MODES:
            source = cache['point_sets'][mode]
            n = len(source['descriptors'])
            source['descriptors'] = np.full((n, 3), .25, dtype=np.float64)
        engine = scoring.RegionScorer(cache)
        self.assertEqual(engine.count(0, 0, 'all_points'), 1)
        self.assertEqual(engine.count(0, 1, 'all_points'), 1)
        np.testing.assert_array_equal(engine.reference_pairs(0, 1, 'all_points'), [(0, 0)])
        n = cache['area'][0]
        actual = scoring.wbbs_scores(np.array([[1]], dtype=np.uint16), np.array([n]), np.array([n]), 0.)
        self.assertEqual(actual.item(), 1/n)
        self.assertLess(actual.item(), 1.)  # Diagonal must not be forced to one.

    def test_thirteen_variants_match_original_j_and_scalar_outer_greedy(self):
        for dtype in (np.float32, np.float64):
            cache, pool = fixture(dtype=dtype)
            provider, old = scoring.Provider(cache, pool), JProvider(cache, pool)
            for q in pool:
                parts = provider.parts(q)
                for spec in grid().values():
                    ids, actual = provider.score(q, spec, parts)
                    if spec['family'] == 'j_baseline':
                        self.assertEqual(old.score(q)[1].tobytes(), actual.tobytes())
                    pair = scoring.score_components(parts, spec)
                    expected = [scalar_greedy(pair[:, provider.tokens(c)], cache['area'][provider.tokens(q)]) for c in ids]
                    np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0.)
                    for c, value in zip(ids, actual):
                        report = provider.explain(q, c, spec, parts)
                        self.assertEqual(report['score'], value)
                        self.assertAlmostEqual(sum(m['contribution'] for m in report['matches']), value)
                        if spec['family'] == 'wbbs':
                            self.assertTrue(all('W' in m and 'T' not in m for m in report['matches']))

    def test_j_does_not_construct_or_compute_wbbs_backend(self):
        cache, pool = fixture()
        with mock.patch.object(scoring, 'RegionScorer', side_effect=AssertionError):
            actual = scoring.Provider(cache, pool).score(pool.gallery[0], {'family': 'j_baseline'})[1]
        self.assertEqual(actual.tobytes(), JProvider(cache, pool).score(pool.gallery[0])[1].tobytes())

    def test_explicit_nonordered_query_rows_and_candidate_filter_keep_scores(self):
        cache, pool = fixture()
        direct = scoring.Provider(cache, pool)
        tokens = np.arange(direct.engine.n)[::-1]
        counter = scoring.RegionScorer(cache)
        direct.counter = counter
        matrices = {mode: counter.count_matrix(tokens, np.arange(len(tokens)), mode) for mode in scoring.MODES}
        class Filtered(Pool):
            def __getitem__(self, q):
                return super().__getitem__(q)[::2]
        cached = scoring.Provider(cache, Filtered(pool.gallery), matrices, tokens)
        with mock.patch.object(scoring, 'RegionScorer', side_effect=AssertionError):
            for q in pool:
                direct_parts, cached_parts = direct.parts(q), cached.parts(q)
                for spec in grid().values():
                    ids, expected = direct.score(q, spec, direct_parts)
                    selected, actual = cached.score(q, spec, cached_parts)
                    np.testing.assert_array_equal(selected, ids[::2])
                    self.assertEqual(actual.tobytes(), expected[::2].tobytes())

    def test_empty_region_maps_and_finite_frozen_specs(self):
        cache, pool = fixture((0, 0))
        provider = scoring.Provider(cache, pool)
        for spec in grid().values():
            self.assertEqual(provider.score(pool.gallery[0], spec)[1].tolist(), [0.])
            self.assertEqual(provider.explain(pool.gallery[0], pool.gallery[1], spec)['matches'], [])
        spec = grid()['all_bb_a04']
        for key, value in [('cnn_weight', .5), ('area_weight', .3), ('count_alpha', 1.),
                           ('count_alpha', np.nan), ('count_alpha', False), ('point_mode', 'sampled')]:
            with self.assertRaises(ValueError):
                scoring.checked_spec({**spec, key: value})
        with self.assertRaises(ValueError):
            scoring.checked_spec({'family': 'global_wbbs'})

    def test_bad_points_counts_coordinates_or_original_area_fail_closed(self):
        cache, _ = fixture()
        mutations = []
        def bad(mode, key, transform):
            value = copy.deepcopy(cache)
            value['point_sets'][mode][key] = transform(value['point_sets'][mode][key])
            mutations.append(value)
        bad('all_points', 'descriptors', lambda x: x.astype(np.float32))
        bad('all_points', 'descriptors', lambda x: np.full_like(x, np.nan))
        bad('all_points', 'coordinates', lambda x: np.zeros_like(x))
        bad('all_points', 'coordinates', lambda x: x[::-1].copy())
        bad('all_points', 'offsets', lambda x: np.r_[x[:-1], x[-1]-1])
        bad('wbbs_denoised', 'descriptors', lambda x: 1.-x)
        mutations.append({**cache, 'area': np.full(len(cache['area']), 65536)})
        for value in mutations:
            with self.subTest(value=len(mutations)), self.assertRaises(ValueError):
                scoring.RegionScorer(value)
        bigger, _ = fixture((5, 1))
        with self.assertRaises(ValueError):
            scoring.RegionScorer(bigger)

    def test_cached_count_bounds_shape_and_query_identity_fail_closed(self):
        cache, pool = fixture()
        n = len(cache['features'])
        matrices = {mode: np.zeros((n, n), dtype=np.uint16) for mode in scoring.MODES}
        for tokens in (np.zeros(n, dtype=np.int64), np.arange(n)+1):
            with self.assertRaises(ValueError):
                scoring.Provider(cache, pool, matrices, tokens)
        matrices['all_points'] = np.full((n, n), 65535, dtype=np.uint16)
        with self.assertRaises(ValueError):
            scoring.Provider(cache, pool, matrices).parts(pool.gallery[0])
        matrices['all_points'] = matrices['all_points'].astype(np.uint32)
        with self.assertRaises(ValueError):
            scoring.Provider(cache, pool, matrices)


if __name__ == '__main__':
    unittest.main()
