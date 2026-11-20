"""Synthetic complete-test integration, including real spawned ranking shards."""
from contextlib import redirect_stdout
import copy
from io import StringIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from f_region_score_search import scoring
from f_region_structure.pipeline import _scalar_greedy, bundle_for
from wbm_retrieval.common import file_hash, read_json
from wbm_retrieval.retrieval_protocol import FullGalleryPool
from f_region_score_test import pipeline
from f_region_score_test.state import verified


MATCHING = {'lambda_pos': 0., 'lambda_scale': 0., 'shape_threshold': 0.,
            'score_threshold': 0., 'weight': 'sqrt_area'}


def fixture():
    rng = np.random.default_rng(319)
    ids = np.arange(7, 43, 2, dtype=np.int64)
    counts = np.tile([0, 1, 2, 4, 1, 3], 3)
    offsets = np.r_[0, np.cumsum(counts)]
    n = int(offsets[-1])
    h = rng.uniform(.01, 1., (n, 512)).astype(np.float32)
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    h[1] = -h[0]
    h[-1] = 0.
    radial = rng.random((n, 16))
    radial /= radial.sum(1, keepdims=True)
    joint = rng.random((n, 128))
    joint /= joint.sum(1, keepdims=True)
    joint[3] = 0.
    area = rng.integers(5, 100, n)
    cache = {'map_ids': ids, 'offsets': offsets, 'area': area, 'features': h,
             'histograms': radial, 'relative_area': area/1000., 'joint': joint.copy(),
             'joint_all': joint, 'owners': np.repeat(ids, counts)}
    labels = np.zeros((ids[-1]+1, 2), dtype=np.int64)
    labels[ids] = np.tile([[1, 0], [0, 1], [1, 1]], (6, 1))
    bundle = bundle_for(labels, SimpleNamespace(ids={'valid': ids, 'test': ids}))
    pool = FullGalleryPool(ids, ids, 'test', 'synthetic_complete_test')
    return cache, pool, bundle


def snapshot(folder):
    return {str(p.relative_to(folder)): (file_hash(p), p.stat().st_mtime_ns)
            for p in Path(folder).rglob('*') if p.is_file()}


class ProviderTests(unittest.TestCase):
    def test_fixed_additive_matches_scalar_greedy_and_masks_empty_regions(self):
        cache, pool, _ = fixture()
        untouched = {name: value.copy() for name, value in cache.items()}
        provider = pipeline.Provider(cache, pool, MATCHING)
        self.assertEqual(pipeline.VARIANT, 'mixed_a0p5_e1p0')
        self.assertEqual(pipeline.SPEC, scoring.frozen_grid()['mixed_a0p5_e1p0'])
        for q in pool:
            parts = provider.parts(q)
            pairs = scoring.score_components(parts, pipeline.SPEC)
            qt = provider.tokens(q)
            ids, actual = provider.score(q, parts)
            self.assertEqual(set(map(int, ids)), set(map(int, pool.gallery))-{int(q)})
            expected = [_scalar_greedy(pairs[:, provider.tokens(c)], cache['area'][qt]) for c in ids]
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.)
            for c, value in zip(ids, actual):
                if len(qt) == 0 or len(provider.tokens(c)) == 0:
                    self.assertEqual(value, 0.)
        for name, value in untouched.items():
            np.testing.assert_array_equal(cache[name], value)

    def test_rejects_valid_partial_test_and_changed_matcher(self):
        cache, pool, _ = fixture()
        ids = pool.gallery
        invalid = [FullGalleryPool(ids, ids, 'valid', 'synthetic'),
                   FullGalleryPool(ids[:3], ids, 'test', 'synthetic')]
        for candidate in invalid:
            with self.assertRaisesRegex(ValueError, 'complete test'):
                pipeline.Provider(cache, candidate, MATCHING)
        for change in ({'weight': 'area'}, {'score_threshold': .2}, {'lambda_pos': 1.}):
            with self.assertRaisesRegex(ValueError, 'matcher must remain fixed'):
                pipeline.Provider(cache, pool, {**MATCHING, **change})


class EvaluationTests(unittest.TestCase):
    def test_serial_and_real_spawned_rankings_match_and_sealed_reuse_is_readonly(self):
        cache, pool, bundle = fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline, 'KS', [1, 5, 10]), redirect_stdout(StringIO()):
            folders = [Path(tmp)/'serial', Path(tmp)/'parallel']
            outputs = [pipeline.evaluate(folder, cache, pool, bundle, MATCHING, 'synthetic_context', workers)
                       for folder, workers in zip(folders, [1, 2])]
            for output, folder in zip(outputs, folders):
                self.assertEqual(set(output), {pipeline.VARIANT})
                evaluation = folder/'test'/pipeline.VARIANT
                self.assertTrue(verified(evaluation))
                check = read_json(evaluation/'verification.json')
                self.assertEqual(check['num_queries'], 18)
                self.assertEqual(check['ranking_rows'], 180)
                self.assertEqual(check['full_candidates_checked'], 18*17)
                self.assertTrue(check['all_queries_full_candidate_topk_exact'])
                self.assertEqual(check['max_saved_score_absolute_error'], 0.)
                manifest = read_json(evaluation/'evaluation_manifest.json')
                self.assertTrue(manifest['test_evaluated'])
                self.assertFalse(manifest['test_parameter_search'])
                self.assertFalse(manifest['training_required'])
                self.assertEqual(manifest['selection_split'], 'valid')
                audit = read_json(folder/'formula_audit_test.json')
                self.assertEqual(len(audit['query_ids']), 3)
                self.assertEqual(audit['candidate_pairs'], 3*17)
                self.assertLessEqual(audit['max_absolute_error'], 2e-6)
                self.assertFalse((folder/'valid').exists())
                self.assertFalse((folder/'selection.json').exists())
                index = read_json(folder/'evaluation_index_test.json')
                self.assertEqual(index['specs'], {pipeline.VARIANT: pipeline.SPEC})
            self.assertEqual(file_hash(folders[0]/'test'/pipeline.VARIANT/'rankings.csv'),
                             file_hash(folders[1]/'test'/pipeline.VARIANT/'rankings.csv'))
            for key in ('top10_hits', 'metrics', 'coverage_metrics'):
                self.assertEqual(outputs[0][pipeline.VARIANT][key], outputs[1][pipeline.VARIANT][key])
            before = snapshot(folders[0])
            with patch.object(pipeline, '_rank_all', side_effect=AssertionError('ranking recomputed')):
                pipeline.evaluate(folders[0], cache, pool, bundle, MATCHING, 'synthetic_context', 1)
            self.assertEqual(before, snapshot(folders[0]))

    def test_interrupted_attempt_is_preserved_and_identity_is_reused(self):
        cache, pool, bundle = fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline, 'KS', [1, 5, 10]), redirect_stdout(StringIO()):
            output = Path(tmp)/'run'
            with patch.object(pipeline, 'verify_evaluation', side_effect=InterruptedError('injected')):
                with self.assertRaises(InterruptedError):
                    pipeline.evaluate(output, cache, pool, bundle, MATCHING, 'synthetic_context', 1)
            folder = output/'test'/pipeline.VARIANT
            before = snapshot(folder)
            identities = snapshot(output/'identities')
            pipeline.evaluate(output, cache, pool, bundle, MATCHING, 'synthetic_context', 1)
            attempts = list((output/'interrupted_evaluations').glob('*/evaluation'))
            self.assertEqual(len(attempts), 1)
            self.assertEqual(before, snapshot(attempts[0]))
            self.assertEqual(identities, snapshot(output/'identities'))
            self.assertTrue(verified(folder))

    def test_changed_audit_context_or_completed_artifact_is_rejected(self):
        cache, pool, bundle = fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline, 'KS', [1, 5, 10]), redirect_stdout(StringIO()):
            output = Path(tmp)/'run'
            pipeline.evaluate(output, cache, pool, bundle, MATCHING, 'synthetic_context', 1)
            with self.assertRaisesRegex(ValueError, 'formula audit differs'):
                pipeline.evaluate(output, cache, pool, bundle, MATCHING, 'different_context', 1)
            ranking = output/'test'/pipeline.VARIANT/'rankings.csv'
            with ranking.open('a') as handle:
                handle.write('tampered\n')
            with self.assertRaisesRegex(ValueError, 'Changed or missing completed artifact'):
                pipeline.evaluate(output, cache, pool, bundle, MATCHING, 'synthetic_context', 1)

    def test_invalid_workers_and_too_small_gallery_rejected_before_writes(self):
        cache, pool, bundle = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'run'
            for workers in (0, 9, True, 1.5):
                with self.assertRaises(ValueError):
                    pipeline.evaluate(output, cache, pool, bundle, MATCHING, 'synthetic_context', workers)
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError, 'smaller than the fixed maximum K'):
                pipeline.evaluate(output, cache, pool, bundle, MATCHING, 'synthetic_context', 1)
            self.assertFalse(output.exists())

    def test_merge_rejects_incomplete_duplicate_or_corrupted_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            shard = folder/'part.csv'
            shard.write_bytes(b'1,1,2,0.5\r\n')
            tasks = [(0, [1], str(shard), 1)]
            good = {'index': 0, 'queries': [1], 'path': str(shard), 'rows': 1, 'sha256': file_hash(shard)}
            for invalid in ([], [good, good], [{**good, 'queries': [2]}], [{**good, 'rows': 2}]):
                with self.assertRaises(ValueError):
                    pipeline._merge_shards(folder/'rankings.csv', tasks, invalid)
            shard.write_bytes(b'1,1,2,0.6\r\n')
            with self.assertRaisesRegex(ValueError, 'shard changed'):
                pipeline._merge_shards(folder/'rankings.csv', tasks, [good])


if __name__ == '__main__':
    unittest.main()
