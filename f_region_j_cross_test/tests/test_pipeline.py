"""Real regional scorer integration with parallel cache and cross-only reports."""
import csv
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np

from f_region_area_fusion.tests.test_pipeline import fixture as original_fixture
from f_region_area_fusion.pipeline import MATCHING, Provider
from fgw_cross_test.state import cross_pool
from fgw_cross_test.verify import independent_order
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from f_region_j_cross_test import cache, pipeline
from f_region_j_cross_test.state import VARIANT, SPEC


def fixture():
    values, base, bundle = original_fixture('test')
    bundle.labels = np.pad(bundle.labels, ((0, 0), (0, 6)))
    return {'cache': values, 'base_pool': base, 'pool': cross_pool(base, bundle.labels),
            'bundle': bundle, 'matching': MATCHING}


class RegionalCrossPipelineTests(unittest.TestCase):
    def test_failed_final_semantic_gate_preserves_evidence_without_completed_root(self):
        from f_region_j_cross_test import run
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_json(root/'status.json', {'status': 'completed'})
            write_json(root/'retained_result.json', {'immutable': 'result'})
            with patch.object(run, 'verify_run', side_effect=ValueError('semantic mismatch')):
                with self.assertRaisesRegex(ValueError, 'semantic mismatch'):
                    run.seal_and_verify(root, 'signature')
            self.assertFalse((root/'completed.json').exists())
            failed = read_json(root/'status.json')
            self.assertEqual(failed['status'], 'failed')
            saved = Path(failed['retained_evidence'])
            self.assertEqual(read_json(saved/'status.json')['status'], 'completed')
            self.assertEqual(read_json(saved/'completed.json')['signature'], 'signature')
            self.assertEqual(read_json(root/'retained_result.json'), {'immutable': 'result'})

    def test_serial_and_spawned_score_caches_are_bit_identical_to_original(self):
        data = fixture()
        original = Provider(data['cache'], data['base_pool'], data['matching'])
        with tempfile.TemporaryDirectory() as raw:
            roots = [Path(raw)/'one', Path(raw)/'two']
            for root in roots:
                root.mkdir()
            outputs = [cache.build_cache(root, data, 'context', workers)[0]
                       for root, workers in zip(roots, [1, 2])]
            np.testing.assert_array_equal(outputs[0].matrix, outputs[1].matrix)
            for q in data['base_pool']:
                ids, scores = original.score(q, SPEC)
                np.testing.assert_array_equal(outputs[0].scores(q, ids), scores)
            with patch.object(cache, '_score_shard', side_effect=AssertionError('completed cache recomputation')):
                _, timing = cache.build_cache(roots[0], data, 'context', 2)
            self.assertTrue(timing['reused'])

    def test_filtering_keeps_original_full_matrix_arithmetic_and_excludes_same_labels(self):
        data = fixture()
        provider = cache.AuditProvider(data)
        for q in data['pool']:
            ids, scores = provider.score(q, SPEC)
            before, original = provider.original.score(q, SPEC)
            mask = np.isin(before, ids)
            np.testing.assert_array_equal(ids, data['pool'][q])
            np.testing.assert_array_equal(scores, original[mask])
            self.assertTrue(np.any(data['bundle'].labels[ids] != data['bundle'].labels[q], axis=1).all())

    def test_self_and_missing_scores_cannot_be_hidden_by_resigning_files(self):
        data = fixture()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            provider, _ = cache.build_cache(root, data, 'context', 1)
            folder = root/'score_cache'
            values = np.load(folder/'scores.npy', mmap_mode='r+')
            values[0, 1] = np.nan
            values.flush()
            marker = read_json(provider.marker)
            marker['files']['scores.npy'] = file_hash(folder/'scores.npy')
            write_json(provider.marker, marker)
            with self.assertRaisesRegex(ValueError, 'score row'):
                cache.ScoreCache(folder, 'context')

    def test_real_coverage_pipeline_and_complete_reuse_without_region_rescoring(self):
        data = fixture()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            upstream = root/'source.json'
            write_json(upstream, {'frozen': True})
            data['source_binding'] = {str(upstream): file_hash(upstream)}
            data['source_binding_signature'] = digest(data['source_binding'])
            provider, _ = cache.build_cache(root, data, 'context', 1)
            def identity(output, signature):
                return {'run_id': '20261120T210203+0800__fixture', 'started_at': '2026-11-20T21:02:03+08:00',
                        'evaluation_signature': signature, 'variant': VARIANT, 'split': 'test'}
            with patch.object(pipeline, 'SETTINGS', {'ks': [1, 5, 10], 'tie_seed': 2026}), \
                 patch.object(pipeline, 'identity', identity):
                result = pipeline.evaluate(root, data, 'context', provider, formal=False)
                folder = pipeline.evaluation_folder(root)
                self.assertEqual(result['num_queries'], len(data['pool']))
                self.assertFalse((folder/'metrics.json').exists())
                self.assertEqual(read_json(folder/'verification.json')['max_saved_score_absolute_error'], 0.)
                (root/'evaluation_index_test.json').unlink()
                with patch.object(provider, 'scores', side_effect=AssertionError('recompute complete evaluation')):
                    self.assertEqual(result, pipeline.evaluate(root, data, 'context', provider, formal=False))
                self.assertTrue((root/'evaluation_index_test.json').exists())

    def test_ordinary_replay_demands_exact_order_and_scores(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            marker = root/'marker.json'
            write_json(marker, {'status': 'completed'})
            ids, scores = np.arange(1, 601), np.linspace(.1, .9, 600)
            class ProviderFixture:
                query_ids = np.array([0])
                def full_row(self, q):return ids, scores
            provider = ProviderFixture()
            provider.marker = marker
            order = independent_order(0, ids, scores, 2026)[:500]
            path = root/'rankings.csv'
            with path.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(pipeline.RANKING_FIELDS)
                writer.writerows((0, rank, int(ids[i]), float(scores[i])) for rank, i in enumerate(order, 1))
            data = {'source_metadata': {'source_winner_folder': str(root)}}
            receipt = pipeline.historical_replay(root, provider, data, 'context')
            self.assertEqual(receipt['max_score_error'], 0.)
            (root/'historical_replay.json').unlink()
            scores[order[0]] -= .0001
            with self.assertRaisesRegex(ValueError, 'exact replay'):
                pipeline.historical_replay(root, provider, data, 'context')


if __name__ == '__main__':
    unittest.main()
