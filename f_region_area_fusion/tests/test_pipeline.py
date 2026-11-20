"""Full scoring/report/replay integration on explicitly synthetic pools."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from f_region_score_test.tests.test_pipeline import fixture as old_fixture, snapshot
from f_region_structure.pipeline import _scalar_greedy
from wbm_retrieval.common import file_hash, read_json
from wbm_retrieval.retrieval_protocol import FullGalleryPool
from f_region_area_fusion import pipeline, scoring, selection
from f_region_area_fusion.state import verified


def fixture(split='valid'):
    cache, pool, bundle = old_fixture()
    cache['joint_all'][3] = 1./128.
    cache['joint'] = cache['joint_all'].copy()
    pool = FullGalleryPool(pool.gallery, pool.gallery, split, 'synthetic_area_fusion')
    return cache, pool, bundle


class ProviderTests(unittest.TestCase):
    def test_full_gallery_scalar_matching_and_immutable_sources(self):
        cache, pool, _ = fixture()
        original = {k: v.copy() for k, v in cache.items()}
        provider = pipeline.Provider(cache, pool, pipeline.MATCHING)
        for q in pool:
            parts = provider.parts(q)
            qt = provider.tokens(q)
            np.testing.assert_array_equal(parts['entropy_reference_counts'],
                                          np.full(len(qt), len(cache['owners'])-len(qt)))
            for spec in scoring.frozen_grid().values():
                pair = scoring.score_components(parts, spec)
                ids, actual = provider.score(q, spec, parts)
                self.assertEqual(set(map(int, ids)), set(map(int, pool.gallery))-{int(q)})
                expected = [_scalar_greedy(pair[:, provider.tokens(c)], cache['area'][qt]) for c in ids]
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.)
                for c, value in zip(ids, actual):
                    if not len(qt) or not len(provider.tokens(c)):
                        self.assertEqual(value, 0.)
        for k in original:
            np.testing.assert_array_equal(cache[k], original[k])

    def test_rejects_partial_test_and_changed_matching(self):
        cache, pool, _ = fixture('test')
        bad_pool = FullGalleryPool(pool.gallery[:3], pool.gallery, 'test', 'synthetic')
        with self.assertRaisesRegex(ValueError, 'every gallery'):
            pipeline.Provider(cache, bad_pool, pipeline.MATCHING)
        with self.assertRaisesRegex(ValueError, 'matcher must remain fixed'):
            pipeline.Provider(cache, pool, {**pipeline.MATCHING, 'weight': 'area'})


class EvaluationTests(unittest.TestCase):
    def test_all_valid_serial_parallel_and_frozen_test_ten(self):
        cache, valid, bundle = fixture()
        _, test, _ = fixture('test')
        with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline, 'KS', [1, 5, 10]), redirect_stdout(StringIO()):
            roots = [Path(tmp)/'serial', Path(tmp)/'parallel']
            outcomes = [pipeline.evaluate(root, cache, valid, bundle, pipeline.MATCHING,
                            'synthetic_valid', scoring.frozen_grid(), workers)
                        for root, workers in zip(roots, [1, 2])]
            self.assertEqual(len(outcomes[0]), 29)
            for name in scoring.frozen_grid():
                self.assertEqual(file_hash(roots[0]/'valid'/name/'rankings.csv'),
                                 file_hash(roots[1]/'valid'/name/'rankings.csv'))
                self.assertTrue(verified(roots[0]/'valid'/name))
            chosen = selection.select_top10(outcomes[0], expected_queries=len(valid))
            test_root = Path(tmp)/'test_run'
            result = pipeline.evaluate(test_root, cache, test, bundle, pipeline.MATCHING,
                'synthetic_test', chosen['selected_specs'], 2, frozen_selection=chosen)
            self.assertEqual(list(result), chosen['selected_variants'])
            self.assertEqual(len(result), 10)
            for rank, name in enumerate(chosen['selected_variants'], 1):
                folder = test_root/'test'/name
                self.assertTrue(verified(folder))
                evidence = read_json(folder/'verification.json')
                self.assertEqual(evidence['full_candidates_checked'], len(test)*(len(test.gallery)-1))
                self.assertTrue(evidence['all_queries_full_candidate_topk_exact'])
                self.assertEqual(evidence['max_saved_score_absolute_error'], 0.)
                self.assertEqual(read_json(folder/'evaluation_manifest.json')['valid_selection_rank'], rank)
            before = snapshot(test_root)
            reused = pipeline.evaluate(test_root, cache, test, bundle, pipeline.MATCHING,
                'synthetic_test', chosen['selected_specs'], 1, frozen_selection=chosen)
            self.assertEqual(reused, result)
            self.assertEqual(before, snapshot(test_root))

    def test_subset_and_unfrozen_test_rejected_before_writes(self):
        cache, valid, bundle = fixture()
        _, test, _ = fixture('test')
        subset = dict(list(scoring.frozen_grid().items())[:10])
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'fresh'
            for pool, specs, choice in [(valid, subset, None), (test, subset, None),
                    (test, subset, {'selected_specs': subset, 'selected_variants': list(reversed(subset))})]:
                with self.assertRaises(ValueError):
                    pipeline.evaluate(target, cache, pool, bundle, pipeline.MATCHING, 'sig',
                                      specs, 1, frozen_selection=choice)
                self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
