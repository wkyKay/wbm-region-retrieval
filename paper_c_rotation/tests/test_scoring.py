"""Synthetic scientific controls; no formal data, model inference or experiment."""
import copy
import math
import unittest
from unittest import mock

import numpy as np

from f_region_area_fusion.pipeline import Provider as HistoricalProvider
from f_region_area_fusion.scoring import frozen_grid as historical_grid
from f_region_evidence.scoring import RegionPairScorer as FrozenChannels
from wbm_retrieval.exact_metrics import order_scores
from wbm_retrieval.retrieval_protocol import FullGalleryPool
from paper_c_rotation import scoring


def fixture(dtype=np.float32, split='valid'):
    rng = np.random.default_rng(2026)
    ids = np.array([5, 11, 16, 29, 40], dtype=np.int64)
    counts = np.array([0, 1, 2, 4, 1], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(counts)]
    n = int(offsets[-1])
    h = rng.normal(size=(n, 512)).astype(dtype)
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    h[1], h[2], h[-1] = h[0], -h[0], 0.
    joint = rng.random((n, 128))
    joint /= joint.sum(axis=1, keepdims=True)
    radial = rng.random((n, 16))
    radial /= radial.sum(axis=1, keepdims=True)
    area = np.array([5, 9, 16, 25, 36, 49, 64, 81], dtype=np.int64)
    cache = {'map_ids': ids, 'offsets': offsets, 'features': h, 'area': area,
             'relative_area': area / 1000., 'joint': joint.copy(), 'joint_all': joint,
             'histograms': radial, 'owners': np.repeat(ids, counts)}
    cache['joint'][::2] = 0.  # Unused retained distributions can be empty.
    pool = FullGalleryPool(ids, ids, split, 'synthetic_paper_c_rotation')
    return cache, pool


def scalar_aggregate(pair, area, reuse=False):
    """Small independent matching reference, deliberately scalar."""
    if not pair.size:
        return 0.
    weights = [math.sqrt(float(x)) for x in area]
    if reuse:
        total = math.fsum(weights[q] * max(map(float, pair[q]))
                          for q in range(len(pair)))
    else:
        edges = sorted(((-float(pair[q, c]), q, c)
                        for q in range(len(pair)) for c in range(pair.shape[1])))
        seen_q, seen_c, total = set(), set(), 0.
        for negative, q, c in edges:
            if negative < 0. and q not in seen_q and c not in seen_c:
                seen_q.add(q)
                seen_c.add(c)
                total += -negative * weights[q]
    return total / math.fsum(weights)


class ScoringTests(unittest.TestCase):
    def test_grid_is_fresh_ordered_and_rejects_mutated_science(self):
        expected = ('primary', 'no_c', 'no_t', 'no_a', 'independent_j', 'reuse_candidate')
        specs = scoring.grid()
        self.assertEqual(tuple(specs), expected)
        for name, spec in specs.items():
            self.assertEqual(scoring.checked_spec(spec), spec)
            self.assertEqual(spec['variant'], name)
        specs['primary']['matching'] = 'reuse_candidate'
        self.assertEqual(scoring.grid()['primary']['matching'], 'greedy_one_to_one')
        invalid = [None, 'primary', specs['primary'],
                   {**scoring.grid()['primary'], 'alpha': .4},
                   {**scoring.grid()['independent_j'], 'distribution': 'joint'}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                scoring.checked_spec(value)

    def test_primary_and_no_a_replay_historical_scores_and_rankings_exactly(self):
        for dtype in (np.float32, np.float64):
            cache, pool = fixture(dtype)
            before = {k: v.copy() for k, v in cache.items()}
            old = HistoricalProvider(cache, pool, scoring.MATCHING)
            new = scoring.Provider(cache, pool, scoring.MATCHING)
            for q in pool:
                parts = new.parts(q)
                for name, historic in (('primary', 'add_a0p4_b0p15'),
                                       ('no_a', 'no_area_a0p4')):
                    with self.subTest(dtype=dtype, query=q, variant=name):
                        a_ids, actual = new.score(q, scoring.grid()[name], parts)
                        b_ids, expected = old.score(q, historical_grid()[historic])
                        np.testing.assert_array_equal(a_ids, b_ids)
                        self.assertEqual(actual.dtype, expected.dtype)
                        self.assertEqual(actual.tobytes(), expected.tobytes())
                        np.testing.assert_array_equal(order_scores(a_ids, actual, int(q), 2026),
                                                      order_scores(b_ids, expected, int(q), 2026))
            for key in before:
                np.testing.assert_array_equal(cache[key], before[key])

    def test_exact_seven_key_legacy_matching_replays_without_mutating_sources(self):
        legacy = {**scoring.MATCHING, 'sigma_pos': .35, 'sigma_scale': .5}
        original = dict(legacy)
        self.assertEqual(scoring.normalize_matching(legacy), scoring.MATCHING)
        for dtype in (np.float32, np.float64):
            cache, pool = fixture(dtype)
            old = HistoricalProvider(cache, pool, legacy)
            new = scoring.Provider(cache, pool, legacy)
            for q in pool:
                parts = new.parts(q)
                for name, historic in (('primary', 'add_a0p4_b0p15'), ('no_a', 'no_area_a0p4')):
                    a_ids, actual = new.score(q, scoring.grid()[name], parts)
                    b_ids, expected = old.score(q, historical_grid()[historic])
                    np.testing.assert_array_equal(a_ids, b_ids)
                    self.assertEqual(actual.dtype, expected.dtype)
                    self.assertEqual(actual.tobytes(), expected.tobytes())
                    np.testing.assert_array_equal(order_scores(a_ids, actual, int(q), 2026),
                                                  order_scores(b_ids, expected, int(q), 2026))
        self.assertEqual(legacy, original)
        self.assertEqual(set(legacy), set(scoring.MATCHING) | {'sigma_pos', 'sigma_scale'})

    def test_matching_compatibility_is_limited_to_exact_historical_extras(self):
        legacy = {**scoring.MATCHING, 'sigma_pos': .35, 'sigma_scale': .5}
        invalid = [[], {**legacy, 'sigma_pos': .4}, {**legacy, 'sigma_scale': 1.},
                   {**scoring.MATCHING, 'sigma_pos': .35},
                   {**scoring.MATCHING, 'sigma_scale': .5},
                   {**legacy, 'unrecognized': 0.}, {**scoring.MATCHING, 'other': .5},
                   {**legacy, 'lambda_pos': False}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                scoring.normalize_matching(value)

    def test_deletion_formulas_keep_relative_weights_and_do_not_refit(self):
        parts = {'cosine': np.array([[.2, .8]], dtype=np.float32),
                 'structure_similarity': np.array([[.7, .9]]),
                 'area_similarity': np.array([[.3, .6]])}
        expected = {'no_c': [( .51*.7 + .15*.3)/.66, (.51*.9 + .15*.6)/.66],
                    'no_t': [( .34*.2 + .15*.3)/.49, (.34*.8 + .15*.6)/.49],
                    'no_a': [.4*.2 + .6*.7, .4*.8 + .6*.9]}
        for name, values in expected.items():
            np.testing.assert_allclose(scoring.score_components(parts, scoring.grid()[name]),
                                       [values], atol=6e-8, rtol=0.)
        for name, removed in (('no_c', 'cosine'), ('no_t', 'structure_similarity'),
                              ('no_a', 'area_similarity')):
            altered = copy.deepcopy(parts)
            altered[removed][:] = 1.
            np.testing.assert_array_equal(scoring.score_components(parts, scoring.grid()[name]),
                                          scoring.score_components(altered, scoring.grid()[name]))

    def test_independence_removes_only_radius_shape_association_per_region(self):
        cache, pool = fixture(np.float64)
        cube = np.zeros((8, 8, 4, 4))
        # Two regions: the same radial and full joint stick/ball marginals,
        # but opposite radial-to-shape assignment. Stick/ball remain coupled.
        cube[:, 0, 0, 0] = cube[:, 7, 3, 3] = .5
        cube[1] = 0.
        cube[1, 0, 3, 3] = cube[1, 7, 0, 0] = .5
        cache['joint_all'] = cube.reshape(8, 128)
        provider = scoring.Provider(cache, pool)
        independent = provider.independent_probabilities.reshape(8, 8, 4, 4)
        np.testing.assert_array_equal(independent[0], independent[1])
        np.testing.assert_array_equal(independent.sum(axis=(2, 3)), cube.sum(axis=(2, 3)))
        np.testing.assert_array_equal(independent.sum(axis=1), cube.sum(axis=1))
        self.assertEqual(independent[0, 0, 0, 3], 0.)  # Not p(r)*p(s)*p(b).
        q = 11  # This map owns region token 0.
        parts = provider.parts(q)
        self.assertEqual(parts['joint_distance'][0, 1], 1.)
        self.assertEqual(parts['independent_joint_distance'][0, 1], 0.)
        self.assertEqual(parts['independent_structure_similarity'][0, 1], 1.)
        self.assertEqual(parts['structure_similarity'][0, 1], np.exp(-.5))
        c, a = parts['cosine'][0, 1], parts['area_similarity'][0, 1]
        actual = scoring.score_components(parts, scoring.grid()['independent_j'])[0, 1]
        self.assertEqual(actual, .85*(.4*c+.6)+.15*a)

    def test_shared_parts_compute_frozen_channels_once_for_all_six_variants(self):
        cache, pool = fixture()
        provider = scoring.Provider(cache, pool)
        with mock.patch.object(provider.engine, 'components', wraps=provider.engine.components) as call:
            parts = provider.parts(29)
            for spec in scoring.grid().values():
                provider.score(29, spec, parts)
            self.assertEqual(call.call_count, 1)
        original = FrozenChannels(cache).components(provider.tokens(29))
        for key in ('cosine', 'structure_similarity', 'area_similarity'):
            self.assertEqual(parts[key].dtype, original[key].dtype)
            self.assertEqual(parts[key].tobytes(), original[key].tobytes())

    def test_all_variants_match_scalar_aggregation_with_empty_maps(self):
        cache, pool = fixture(split='test')
        provider = scoring.Provider(cache, pool)
        for q in pool:
            parts = provider.parts(q)
            qt = provider.tokens(q)
            for name, spec in scoring.grid().items():
                pair = scoring.score_components(parts, spec)
                ids, actual = provider.score(q, spec, parts)
                expected = [scalar_aggregate(pair[:, provider.tokens(c)], cache['area'][qt],
                                             reuse=name == 'reuse_candidate') for c in ids]
                np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=0.)
                for c, value in zip(ids, actual):
                    if not len(qt) or not len(provider.tokens(c)):
                        self.assertEqual(value, 0.)

    def test_reuse_preserves_full_query_denominator_and_real_token_mask(self):
        # One usable candidate token can serve both query regions. The second
        # query's zero evidence must still contribute sqrt(9) to the denominator.
        pair = np.array([[.8, .6], [.0, .9]])
        rows = np.array([[0, -1, -1, -1], [-1, -1, -1, -1], [0, 1, -1, -1]])
        result = scoring.reuse_candidate_scores(pair, rows, np.array([4., 9.]))
        np.testing.assert_allclose(result, [.8*2/5, 0., (.8*2+.9*3)/5], atol=1e-16)
        shared = np.array([[.8], [.9]])
        result = scoring.reuse_candidate_scores(shared, np.array([[0, -1]]), [4., 9.])
        self.assertAlmostEqual(result[0], (.8*2+.9*3)/5)
        self.assertGreater(result[0], scalar_aggregate(shared, [4., 9.]))

    def test_greedy_ties_keep_original_query_then_candidate_order(self):
        cache, pool = fixture()
        provider = scoring.Provider(cache, pool)
        q = 16  # query areas 9,16; candidate 11 has one region.
        parts = provider.parts(q)
        for key in ('cosine', 'structure_similarity', 'area_similarity'):
            parts[key][:] = 1.
        ids, greedy = provider.score(q, scoring.grid()['primary'], parts)
        _, reused = provider.score(q, scoring.grid()['reuse_candidate'], parts)
        i = list(ids).index(11)
        self.assertEqual(greedy[i], 3/7)
        self.assertEqual(reused[i], 1.)

    def test_all_empty_gallery_is_defined_for_every_variant(self):
        cache, pool = fixture()
        cache['offsets'] = np.zeros(len(cache['map_ids'])+1, dtype=np.int64)
        for key in ('features', 'area', 'relative_area', 'joint', 'joint_all', 'histograms', 'owners'):
            cache[key] = cache[key][:0]
        provider = scoring.Provider(cache, pool)
        for q in pool:
            parts = provider.parts(q)
            for spec in scoring.grid().values():
                ids, values = provider.score(q, spec, parts)
                np.testing.assert_array_equal(values, np.zeros(len(ids)))

    def test_rejects_changed_matching_partial_test_and_malformed_sources(self):
        cache, pool = fixture(split='test')
        with self.assertRaisesRegex(ValueError, 'remain fixed'):
            scoring.Provider(cache, pool, {**scoring.MATCHING, 'weight': 'area'})
        partial = FullGalleryPool(pool.gallery[:2], pool.gallery, 'test', 'synthetic')
        with self.assertRaisesRegex(ValueError, 'every gallery'):
            scoring.Provider(cache, partial)
        for key in ('joint_all', 'owners', 'area'):
            bad = copy.deepcopy(cache)
            bad[key][0] = 0
            with self.subTest(key=key), self.assertRaises(ValueError):
                scoring.Provider(bad, pool)
        provider = scoring.Provider(cache, pool)
        with self.assertRaisesRegex(ValueError, 'different query'):
            provider.score(29, scoring.grid()['primary'], provider.parts(16))
        parts = provider.parts(29)
        parts['independent_structure_similarity'][:] = np.nan
        with self.assertRaises(ValueError):
            provider.score(29, scoring.grid()['independent_j'], parts)


if __name__ == '__main__':
    unittest.main()
