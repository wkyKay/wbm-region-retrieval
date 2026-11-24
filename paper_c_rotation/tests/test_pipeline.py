"""Real on-disk caches and spawned workers for the new ablation pipeline."""
from contextlib import redirect_stdout
import csv
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from f_region_area_fusion.pipeline import Provider as HistoricalProvider
from f_region_area_fusion.scoring import frozen_grid as historical_grid
from f_region_structure.state import seal
from wbm_retrieval.common import file_hash, read_json
from wbm_retrieval.exact_metrics import order_scores
from paper_c_rotation import pipeline, scoring
from paper_c_rotation.tests.test_scoring import fixture


def disk_fixture(root, split='valid', *, all_empty=False):
    """The worker must reopen source files rather than pickle parent memmaps."""
    cache, pool = fixture(split=split)
    if all_empty:
        cache['offsets'] = np.zeros(len(cache['map_ids'])+1, dtype=np.int64)
        for key in ('features', 'area', 'relative_area', 'joint', 'joint_all', 'histograms', 'owners'):
            cache[key] = cache[key][:0]
    folder = Path(root)/'source'/split
    folder.mkdir(parents=True)
    np.save(folder/'h512.npy', cache.pop('features'), allow_pickle=False)
    np.savez(folder/'regions.npz', **cache)
    # An irrelevant payload verifies that the lean loader does not open views.
    (folder/'h_views.npy').write_bytes(b'not a NumPy file; never a scoring input')
    seal(folder, 'synthetic_source_cache')
    return pipeline.read_scoring_cache(folder), pool


def snapshot(root):
    return {str(p.relative_to(root)): (file_hash(p), p.stat().st_mtime_ns)
            for p in Path(root).rglob('*') if p.is_file()}


class ScoreCacheTests(unittest.TestCase):
    def test_one_and_two_spawned_workers_have_identical_full_scores(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
            root = Path(temporary)
            cache, pool = disk_fixture(root)
            before = snapshot(cache['folder'])
            legacy_matching = {**scoring.MATCHING, 'sigma_pos': .35, 'sigma_scale': .5}
            original_matching = dict(legacy_matching)
            results = [pipeline.compute_scores(root/name, cache, pool, legacy_matching,
                         'synthetic_context', scoring.grid(), workers=workers)
                       for name, workers in (('one', 1), ('two', 2))]
            direct = scoring.Provider(cache, pool)
            for name, spec in scoring.grid().items():
                left, right = (result[name] for result in results)
                self.assertEqual(left.matrix.tobytes(), right.matrix.tobytes())
                self.assertEqual(left.matrix.dtype, np.float64)
                np.testing.assert_array_equal(left.query_ids, list(pool))
                np.testing.assert_array_equal(left.gallery_ids, pool.gallery)
                self.assertEqual(left.signature, right.signature)
                for row, query in enumerate(pool):
                    ids, expected = direct.score(query, spec)
                    np.testing.assert_array_equal(left.scores(query, ids), expected)
                    own = int(np.flatnonzero(left.gallery_ids == query)[0])
                    self.assertEqual(left.matrix[row, own], 0.)
                    observed_ids, observed_scores = left.full_row(query)
                    np.testing.assert_array_equal(observed_ids, ids)
                    np.testing.assert_array_equal(observed_scores, expected)
                metadata = read_json(root/'two'/'score_cache'/'valid'/name/'metadata.json')
                self.assertEqual(metadata['source_cache_sha256'], file_hash(cache['folder']/'completed.json'))
            self.assertEqual(before, snapshot(cache['folder']))
            self.assertEqual(legacy_matching, original_matching)

    def test_spawn_uses_explicit_rotation_features_instead_of_geometry_h512(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
            root = Path(temporary)
            cache, pool = disk_fixture(root)
            old_features = np.array(cache['features'], copy=True)
            rotation = np.zeros_like(old_features)
            rotation[:, 0] = 1.
            path = root/'rotation.npy'
            np.save(path, rotation)
            cache = pipeline.read_scoring_cache(cache['folder'], path)
            result = pipeline.compute_scores(root/'output', cache, pool, scoring.MATCHING,
                                             'rotation_context', scoring.grid(), workers=2)
            direct = scoring.Provider(cache, pool)
            for query in pool:
                ids, expected = direct.score(query, scoring.grid()['primary'])
                np.testing.assert_array_equal(result['primary'].scores(query, ids), expected)
            original = pipeline.read_scoring_cache(cache['folder'])
            self.assertFalse(np.array_equal(original['features'], cache['features']))
            np.testing.assert_array_equal(original['features'], old_features)
            meta = read_json(root/'output/score_cache/valid/primary/metadata.json')
            self.assertEqual(meta['h512_path'], str(path.resolve()))
            self.assertEqual(meta['h512_sha256'], file_hash(path))
            rotation[:, 0] = 0.
            np.save(path, rotation)
            with self.assertRaisesRegex(ValueError, 'source, specification or protocol'):
                pipeline.compute_scores(root/'output', cache, pool, scoring.MATCHING,
                                        'rotation_context', scoring.grid(), workers=2)

    def test_finished_cache_reuse_is_read_only_and_never_rescores(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
            root = Path(temporary)
            cache, pool = disk_fixture(root)
            output = root/'output'
            old = pipeline.compute_scores(output, cache, pool, scoring.MATCHING,
                                          'unchanged_context', scoring.grid(), workers=1)
            before = snapshot(output)
            with mock.patch.object(pipeline, '_init_scores', side_effect=AssertionError('unexpected scoring')):
                new = pipeline.compute_scores(output, cache, pool, scoring.MATCHING,
                                              'unchanged_context', scoring.grid(), workers=2)
            self.assertEqual(before, snapshot(output))
            for name in scoring.grid():
                self.assertEqual(old[name].matrix.tobytes(), new[name].matrix.tobytes())

    def test_rejects_changed_context_and_tampered_sealed_matrix(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
            root = Path(temporary)
            cache, pool = disk_fixture(root)
            output = root/'output'
            providers = pipeline.compute_scores(output, cache, pool, scoring.MATCHING,
                                                 'initial_context', scoring.grid(), workers=1)
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                pipeline.compute_scores(output, cache, pool, scoring.MATCHING,
                                        'different_context', scoring.grid(), workers=1)
            path = providers['primary'].marker.parent/'scores.npy'
            altered = np.load(path, mmap_mode='r+')
            altered[0, 0] = .5
            altered.flush()
            del altered
            with self.assertRaisesRegex(ValueError, 'Changed or missing'):
                pipeline.compute_scores(output, cache, pool, scoring.MATCHING,
                                        'initial_context', scoring.grid(), workers=1)

    def test_incomplete_cache_is_preserved_and_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, pool = disk_fixture(root)
            output = root/'output'
            folder = output/'score_cache'/'valid'/'primary'
            folder.mkdir(parents=True)
            (folder/'partial.bin').write_bytes(b'preserve failed attempt')
            before = snapshot(output)
            with self.assertRaisesRegex(FileExistsError, 'Incomplete score cache'):
                pipeline.compute_scores(output, cache, pool, scoring.MATCHING,
                                        'synthetic_context', scoring.grid(), workers=1)
            self.assertEqual(before, snapshot(output))

    def test_all_empty_region_caches_remain_finite_after_spawn(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
            root = Path(temporary)
            cache, pool = disk_fixture(root, split='test', all_empty=True)
            providers = pipeline.compute_scores(root/'output', cache, pool, scoring.MATCHING,
                                                 'empty_context', scoring.grid(), workers=2)
            for provider in providers.values():
                np.testing.assert_array_equal(provider.matrix, np.zeros((len(pool), len(pool.gallery))))

    def test_primary_and_no_a_replay_real_historical_format(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
            root = Path(temporary)
            cache, pool = disk_fixture(root)
            output = root/'output'
            providers = pipeline.compute_scores(output, cache, pool, scoring.MATCHING,
                                                 'replay_context', scoring.grid(), workers=1)
            old = HistoricalProvider(cache, pool, scoring.MATCHING)
            references = {}
            for name, previous in (('primary', 'add_a0p4_b0p15'), ('no_a', 'no_area_a0p4')):
                folder = root/'historical'/name
                folder.mkdir(parents=True)
                path = folder/'rankings.csv'
                with path.open('w', newline='') as handle:
                    writer = csv.writer(handle)
                    writer.writerow(pipeline.FIELDS)
                    for q in pool:
                        ids, scores = old.score(q, historical_grid()[previous])
                        order = order_scores(ids, scores, int(q), 2026)[:3]
                        writer.writerows((int(q), rank, int(ids[i]), float(scores[i]))
                                         for rank, i in enumerate(order, 1))
                seal(folder, 'synthetic_historical_control')
                references[name] = {'folder': str(folder), 'ranking_sha256': file_hash(path),
                                    'key': 'old_'+previous,
                                    'completed_sha256': file_hash(folder/'completed.json')}
            with mock.patch.object(pipeline, 'KS', [1, 3]):
                report = pipeline.replay_controls(output, 'valid', references, providers, pool)
            self.assertEqual(set(report['controls']), {'primary', 'no_a'})
            for control in report['controls'].values():
                self.assertTrue(control['top500_identical'])
                self.assertEqual(control['ranking_rows'], len(pool)*3)
                self.assertEqual(control['max_score_error'], 0.)


if __name__ == '__main__':
    unittest.main()
