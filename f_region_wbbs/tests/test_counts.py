from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from f_region_wbbs import counts, scoring
from f_region_wbbs.tests.test_scoring import fixture
from wbm_retrieval.common import read_json


class Pool:
    def __init__(self, ids, split='valid', queries=None):
        self.gallery = np.asarray(ids, dtype=np.int64)
        self.queries = self.gallery if queries is None else np.asarray(queries, dtype=np.int64)
        self.split_name, self.protocol_signature = split, 'fixed_protocol'

    def __iter__(self):
        return iter(self.queries)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, q):
        return self.gallery[self.gallery != q]


class CountTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cache, _ = fixture((2, 2, 2, 2, 2, 2, 2, 2, 2, 2))
        folder = self.root/'source'
        folder.mkdir()
        (folder/'completed.json').write_text('{"source":"immutable fixture"}')
        self.cache['folder'] = folder
        self.reader = mock.patch.object(counts, 'read_cache', return_value=self.cache)
        self.reader.start()
        self.addCleanup(self.reader.stop)

    def create(self, split='valid', queries=None, output='run'):
        pool = Pool(self.cache['map_ids'], split, queries)
        matrices, tokens = counts.create(self.root/output, self.cache, pool, 'context', workers=2)
        return matrices, tokens, pool

    def test_rectangular_query_token_rows_and_sealed_readonly_arrays(self):
        ids = self.cache['map_ids']
        matrices, tokens, pool = self.create(queries=ids[[7, 0, 3]])
        np.testing.assert_array_equal(tokens, [14, 15, 0, 1, 6, 7])
        oracle = scoring.RegionScorer(self.cache)
        for mode in scoring.MODES:
            expected = oracle.count_matrix(tokens, np.arange(oracle.n), mode)
            np.testing.assert_array_equal(matrices[mode], expected)
            self.assertFalse(matrices[mode].flags.writeable)
        folder = self.root/'run/counts/valid'
        reread, same = counts.read_counts(folder)
        np.testing.assert_array_equal(same, tokens)
        with mock.patch.object(counts, 'RegionScorer', side_effect=AssertionError):
            counts.create(self.root/'run', self.cache, pool, 'context', workers=2)
        self.assertEqual(read_json(folder/'timing.json')['point_modes']['all_points']['count_calls'], 6*20)

    def test_triangle_parallel_mirroring_real_diagonal_and_exact_pair_budget(self):
        # Repeated descriptors make diagonal BBP counts differ from point counts.
        for mode in scoring.MODES:
            self.cache['point_sets'][mode]['descriptors'][:] = .25
        matrices, tokens, _ = self.create('test')
        np.testing.assert_array_equal(tokens, np.arange(20))
        for matrix in matrices.values():
            np.testing.assert_array_equal(matrix, matrix.T)
        np.testing.assert_array_equal(np.diag(matrices['all_points']), np.ones(20))
        oracle = scoring.RegionScorer(self.cache)
        for mode in scoring.MODES:
            np.testing.assert_array_equal(matrices[mode], oracle.count_matrix(tokens, tokens, mode))
        timing = read_json(self.root/'run/counts/test/timing.json')
        self.assertEqual(timing['point_modes']['all_points']['count_calls'], 20*21//2)
        self.assertEqual(timing['blocks_total'], 2)

    def test_resume_recomputes_only_unmarked_segment_and_retains_sealed_rows(self):
        matrices, tokens, pool = self.create('test')
        expected = {mode: np.array(matrix) for mode, matrix in matrices.items()}
        folder = self.root/'run/counts/test'
        first_marker = (folder/'rows/000000.json').read_bytes()
        for name in ('completed.json', 'timing.json', 'rows/000001.json'):
            (folder/name).unlink()
        for mode in scoring.MODES:
            matrix = np.load(folder/f'{mode}.npy', mmap_mode='r+')
            matrix[16:, 16:] = 65535
            matrix.flush()
        resumed, _ = counts.create(self.root/'run', self.cache, pool, 'context', workers=2)
        for mode in scoring.MODES:
            np.testing.assert_array_equal(resumed[mode], expected[mode])
        self.assertEqual((folder/'rows/000000.json').read_bytes(), first_marker)
        self.assertEqual(read_json(folder/'timing.json')['blocks_resumed'], 1)

    def test_completed_corruption_and_marked_incomplete_corruption_refused(self):
        _, _, pool = self.create('test')
        folder = self.root/'run/counts/test'
        matrix = np.load(folder/'all_points.npy', mmap_mode='r+')
        matrix[0, 0] += 1
        matrix.flush()
        with self.assertRaises(ValueError):
            counts.read_counts(folder)
        with self.assertRaises(ValueError):
            counts.create(self.root/'run', self.cache, pool, 'context')
        (folder/'completed.json').unlink()
        with self.assertRaisesRegex(ValueError, 'completed count row changed'):
            counts.create(self.root/'run', self.cache, pool, 'context')

    def test_context_source_protocol_and_query_order_changes_cannot_reuse(self):
        _, _, pool = self.create()
        for key, value in [('protocol_signature', 'other'), ('queries', pool.queries[::-1])]:
            old = getattr(pool, key)
            setattr(pool, key, value)
            with self.assertRaises(ValueError):
                counts.create(self.root/'run', self.cache, pool, 'context')
            setattr(pool, key, old)
        with self.assertRaises(ValueError):
            counts.create(self.root/'run', self.cache, pool, 'other_context')
        (self.cache['folder']/'completed.json').write_text('{"source":"changed"}')
        with self.assertRaises(ValueError):
            counts.create(self.root/'run', self.cache, pool, 'context')

    def test_incomplete_test_queries_and_empty_queries_rejected(self):
        ids = self.cache['map_ids']
        for split, queries in [('test', ids[:-1]), ('test', ids[::-1]), ('valid', [])]:
            with self.assertRaises(ValueError):
                self.create(split, queries)

    def test_microbenchmark_oracle_and_work_estimate_are_diagnostic_only(self):
        pool = Pool(self.cache['map_ids'])
        result = counts.benchmark(self.cache, pool, query_ids=pool.gallery[:2], candidates_per_query=3)
        self.assertEqual(result['status'], 'verified')
        self.assertTrue(result['diagnostic_only'])
        self.assertEqual(result['map_pairs'], 6)
        self.assertEqual(result['full_split_count_pairs_per_mode'], 400)
        self.assertEqual(result['point_modes']['all_points']['count_calls'], 24)
        self.assertEqual(result['oracle_count_pairs']['all_points'], 24)


if __name__ == '__main__':
    unittest.main()
