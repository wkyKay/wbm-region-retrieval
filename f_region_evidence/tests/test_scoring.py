import copy
import unittest
from unittest import mock

import numpy as np

from f_region_area_fusion.scoring import AreaPairScorer, score_components as old_score
from f_region_area_fusion.scoring import independent_components, independent_score_components
from f_region_evidence import scoring
from wbm_retrieval.diagnostics import batched_map_scores


class Pool:
    def __init__(self, ids):
        self.gallery = np.asarray(ids)
        self.split_name = 'valid'
        self.candidates_per_query = len(ids)-1

    def __iter__(self):
        return iter(self.gallery)

    def __getitem__(self, query):
        return self.gallery[self.gallery != query]


def fixture(counts=(3, 8, 0, 1, 2), dtype=np.float64, start=10):
    rng = np.random.default_rng(2026)
    ids = np.arange(start, start+len(counts), dtype=np.int64)
    offsets = np.r_[0, np.cumsum(counts)]
    n = int(offsets[-1])
    h = rng.normal(size=(n, 512)).astype(dtype)
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    p = rng.random((n, 128))
    p /= p.sum(axis=1, keepdims=True)
    radial = rng.random((n, 16))
    radial /= radial.sum(axis=1, keepdims=True)
    area = rng.integers(5, 101, n)
    cache = {'map_ids': ids, 'offsets': offsets, 'owners': np.repeat(ids, counts),
             'features': h, 'joint_all': p, 'joint': p.copy(), 'histograms': radial,
             'area': area, 'relative_area': area/1000.}
    return cache, Pool(ids)


def scalar_greedy(pair, area):
    weights = [float(np.sqrt(x)) for x in area]
    if not weights:
        return 0.
    pairs = [(float(pair[i, j]), i, j) for i in range(len(area)) for j in range(pair.shape[1])]
    pairs.sort(key=lambda row: (-row[0], row[1], row[2]))
    used_q, used_c, total = set(), set(), 0.
    for value, qi, ci in pairs:
        if value > 0. and qi not in used_q and ci not in used_c:
            total += weights[qi] * value
            used_q.add(qi)
            used_c.add(ci)
    return total / sum(weights)


class ScoringTests(unittest.TestCase):
    def test_none_reproduces_old_components_and_scores_byte_for_byte(self):
        for dtype in (np.float32, np.float64):
            cache, pool = fixture((3, 4, 0, 1, 2), dtype)
            provider = scoring.Provider(cache, pool)
            old = AreaPairScorer(cache['features'], cache, cache)
            for query in pool:
                qt = provider.tokens(query)
                before = old.components(qt)
                with mock.patch('f_region_area_fusion.scoring.entropy_weights', side_effect=AssertionError):
                    after = provider.parts(query)
                for channel in ('cosine', 'structure_similarity', 'area_similarity'):
                    self.assertEqual(before[channel].dtype, after[channel].dtype)
                    self.assertEqual(before[channel].tobytes(), after[channel].tobytes())
                expected_pair = old_score(before, scoring.SPEC)
                actual_pair = scoring.score_components(after)
                self.assertEqual(expected_pair.tobytes(), actual_pair.tobytes())
                ids, actual = provider.score(query, parts=after)
                rows = provider.padded[[provider.positions[int(c)] for c in ids]]
                expected = batched_map_scores(expected_pair, rows, cache['area'][qt], scoring.MATCHING)
                self.assertEqual(expected.tobytes(), actual.tobytes())

    def test_components_match_independent_old_scalar_reference(self):
        cache, _ = fixture()
        q = np.array([0, 3, 9], dtype=np.int64)
        actual = scoring.RegionPairScorer(cache).scores(q)
        parts = independent_components(q, cache['features'], cache['relative_area'],
                                       cache['joint_all'], cache['owners'], with_entropy=False)
        expected = independent_score_components(parts, scoring.SPEC)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.)

    def test_rejection_zeroes_weak_pairs_without_removing_query_weight(self):
        raw = np.array([[.9], [.4]])
        tau = np.array([.5, .5])
        effective = scoring.net_evidence(raw, tau)
        np.testing.assert_array_equal(effective, [[.8], [0.]])
        score = scoring.batched_score(effective, np.array([[0]]), np.array([1., 9.]))
        self.assertEqual(score.item(), .2)  # .8 / (sqrt(1)+sqrt(9)), not .8.
        self.assertIs(scoring.net_evidence(raw), raw)
        np.testing.assert_array_equal(raw, [[.9], [.4]])

    def test_tau_one_rejects_even_equal_background_without_nan(self):
        result = scoring.net_evidence(np.array([[1., .7], [.8, .4]]), np.array([1., .4]))
        np.testing.assert_array_equal(result[0], [0., 0.])
        self.assertAlmostEqual(result[1, 0], 2./3.)
        self.assertEqual(result[1, 1], 0.)
        self.assertTrue(np.isfinite(result).all())

    def test_eight_region_padded_batch_matches_independent_greedy(self):
        cache, pool = fixture()
        n = len(cache['features'])
        thresholds = {'q75': np.linspace(.35, .85, n)}
        provider = scoring.Provider(cache, pool, thresholds=thresholds)
        for q in pool:
            qt = provider.tokens(q)
            pair = scoring.net_evidence(provider.engine.scores(qt), thresholds['q75'][qt])
            ids, actual = provider.score(q, 'q75')
            expected = [scalar_greedy(pair[:, provider.tokens(c)], cache['area'][qt]) for c in ids]
            np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0.)

    def test_explanations_match_scores_and_disclose_unmatched_regions(self):
        cache, pool = fixture()
        n = len(cache['features'])
        thresholds = {'q75': np.linspace(.35, .85, n)}
        provider = scoring.Provider(cache, pool, thresholds=thresholds)
        for q in pool:
            parts = provider.parts(q)
            for rejection in ('none', 'q75'):
                ids, values = provider.score(q, rejection, parts)
                for candidate, score in zip(ids, values):
                    report = provider.explain(q, candidate, rejection, parts)
                    self.assertEqual(report['score'], float(score))
                    self.assertAlmostEqual(sum(m['contribution'] for m in report['matches']), float(score))
                    matched = [m['query_token'] for m in report['matches']]
                    self.assertEqual(len(matched), len(set(matched)))
                    self.assertEqual(sorted(matched+report['unmatched_query_tokens']), provider.tokens(q).tolist())

    def test_ties_use_original_query_then_candidate_order(self):
        cache, pool = fixture((3, 2))
        provider = scoring.Provider(cache, pool)
        q, c = pool.gallery
        parts = {k: np.ones((3, 5)) for k in ('cosine', 'structure_similarity', 'area_similarity')}
        report = provider.explain(q, c, parts=parts)
        self.assertEqual([(m['query_local_index'], m['candidate_local_index']) for m in report['matches']],
                         [(0, 0), (1, 1)])
        self.assertEqual(report['unmatched_query_tokens'], [2])

    def test_candidate_filtering_does_not_recompute_pair_components(self):
        cache, pool = fixture()
        provider = scoring.Provider(cache, pool)
        q = int(pool.gallery[0])
        parts = provider.parts(q)
        ids, full = provider.score(q, parts=parts)
        class FilteredPool(Pool):
            def __getitem__(self, query):
                return super().__getitem__(query)[::2]
        filtered = scoring.Provider(cache, FilteredPool(pool.gallery))
        with mock.patch.object(filtered.engine, 'components', side_effect=AssertionError):
            chosen, scores = filtered.score(q, parts=parts)
        np.testing.assert_array_equal(chosen, ids[::2])
        np.testing.assert_array_equal(scores, full[::2])

    def test_empty_gallery_regions_and_empty_queries_are_defined(self):
        cache, pool = fixture((0, 0))
        provider = scoring.Provider(cache, pool)
        for q in pool:
            self.assertEqual(provider.score(q)[1].tolist(), [0.])
            report = provider.explain(q, pool[q][0])
            self.assertEqual(report['score'], 0.)
            self.assertEqual(report['matches'], [])

    def test_invalid_cache_thresholds_and_matching_fail_closed(self):
        cache, pool = fixture()
        for key, value in (('relative_area', np.zeros(len(cache['area']))),
                           ('joint_all', np.zeros_like(cache['joint_all'])),
                           ('owners', np.zeros_like(cache['owners'])),
                           ('area', np.zeros_like(cache['area']))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                scoring.Provider({**cache, key: value}, pool)
        with self.assertRaises(ValueError):
            scoring.Provider(cache, pool, {**scoring.MATCHING, 'weight': 'uniform'})
        with self.assertRaises(ValueError):
            scoring.Provider(cache, pool, thresholds={'q75': np.ones(1)})
        with self.assertRaises(ValueError):
            scoring.Provider(cache, pool).score(pool.gallery[0], 'q90')
        oversized, oversized_pool = fixture((9, 1))
        with self.assertRaises(ValueError):
            scoring.Provider(oversized, oversized_pool)
        for tau in (np.array([np.nan]), np.array([1.1]), np.array([-1.]), np.ones(2)):
            with self.assertRaises(ValueError):
                scoring.net_evidence(np.ones((1, 2)), tau)


if __name__ == '__main__':
    unittest.main()
