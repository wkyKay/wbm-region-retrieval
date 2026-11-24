"""Synthetic deletion controls and complete spawned diagnostic output checks."""
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from paper_d_rotation import perturb


class DeletionLogicTests(unittest.TestCase):
    def test_rematching_and_original_denominator(self):
        # Greedy assigns q0->c0, q1->c1. Removing q0 frees c0 for q1;
        # the result is therefore not baseline minus the old q0 contribution.
        pair = np.array([[.9, .8], [.85, .1]])
        baseline, parts = perturb.scalar_match(pair, [9., 16.])
        masked = pair.copy()
        masked[0] = 0
        changed, contribution = perturb.scalar_match(masked, [9., 16.])
        self.assertAlmostEqual(baseline, (3*.9+4*.1)/7)
        self.assertAlmostEqual(changed, 4*.85/7)
        self.assertNotAlmostEqual(changed, baseline-parts[0])
        self.assertEqual(contribution[0], 0.)
        self.assertGreater(changed, baseline)  # No monotonicity assumption.

    def test_area_choice_is_before_contribution_and_ties_use_index(self):
        self.assertEqual(perturb.area_matched_pair([25, 36, 49, 64]), (2, 3))
        self.assertEqual(perturb.area_matched_pair([25, 25, 25]), (0, 1))
        self.assertIsNone(perturb.area_matched_pair([1, 4]))
        self.assertIsNone(perturb.area_matched_pair([5]))
        self.assertEqual(perturb.high_low([0., 0., 0.]), (0, 0))
        self.assertEqual(perturb.high_low([.2, .5, .5], [1, 2]), (1, 1))

    def test_random_is_mean_of_deletion_metrics_and_preserves_none(self):
        common = {'query_id': 7}
        original = dict(zip(perturb.METRICS, [.8, None, .9, 1, 1.]))
        deletions = [dict(zip(perturb.METRICS, [1., None, .5, 11, 0.])),
                     dict(zip(perturb.METRICS, [.2, None, .6, 1, 1.]))]
        rows = perturb.strategy_rows(common, original, deletions, [.2, .7], [25, 25])
        random = next(r for r in rows if r['subset']=='main' and r['strategy']=='random')
        self.assertEqual(random['Precision@10'], .6)
        self.assertEqual(random['reference_rank'], 6.)
        self.assertEqual(random['reference_top10'], .5)
        self.assertIsNone(random['Coverage-nDCG@10'])
        high = next(r for r in rows if r['subset']=='main' and r['strategy']=='high')
        self.assertEqual(high['selected_regions'], '1')

    def test_linear_gain_full_idcg_and_frozen_ties(self):
        from experiments.proposed.f_segmentation_geometry.coverage_metrics import query_metrics
        from wbm_retrieval.exact_metrics import order_scores
        labels = np.array([[1, 1], [1, 0], [0, 1], [1, 1], [1, 0]])
        ids = np.array([1, 2, 3, 4])
        scores = np.array([.7, .7, .1, .2])
        actual, order, ref = perturb.metric_values(ids, scores, 0, labels)
        expected_order = order_scores(ids, scores, 0, 2026)
        np.testing.assert_array_equal(order, expected_order)
        expected = query_metrics(labels[0], labels[ids], labels[ids[order]], [10])
        self.assertAlmostEqual(actual['Coverage-nDCG@10'], expected['Coverage-nDCG@10'])
        self.assertEqual(actual['Precision@10'], .1)
        self.assertEqual(actual['reference_rank'], 1)
        self.assertEqual(ref, ids[order[0]])
        no_overlap = np.array([[1, 0], [0, 1], [0, 1]])
        empty, _, _ = perturb.metric_values([1, 2], [.2, .3], 0, no_overlap, cross=True)
        self.assertIsNone(empty['Coverage-nDCG@10'])
        self.assertIsNone(empty['Precision@10'])


def disk_data(root):
    from paper_d_rotation.scoring import Provider
    from f_region_structure.state import seal
    from paper_c_rotation.pipeline import read_scoring_cache
    from paper_c_rotation.tests.test_scoring import fixture
    cache, pool = fixture(split='test')
    folder = root/'source'
    folder.mkdir()
    # The old-cache features intentionally differ. Spawned workers must reopen
    # the explicit rotation path, never silently use the legacy h512.npy.
    np.save(folder/'h512.npy', np.zeros_like(cache['features']))
    feature_path = root/'rotation_original.npy'
    np.save(feature_path, cache['features'].astype(np.float64))
    np.savez(folder/'regions.npz', **{k:v for k,v in cache.items() if k!='features'})
    seal(folder, 'synthetic_regions')
    cache = read_scoring_cache(folder, str(feature_path))
    cache['feature_path'] = str(feature_path)
    provider = Provider(cache, pool)
    base = root/'baseline'
    base.mkdir()
    matrix = np.zeros((len(pool),len(pool.gallery)))
    for row, q in enumerate(pool):
        ids, scores = provider.score(q)
        for i, c in enumerate(ids):
            matrix[row, np.flatnonzero(pool.gallery == c)[0]] = scores[i]
    np.save(base/'scores.npy', matrix)
    np.savez(base/'identities.npz',queries=np.array(list(pool)),gallery=pool.gallery)
    (base/'metadata.json').write_text(json.dumps({'signature':'synthetic_scores',
        'retrieval_protocol_signature':pool.protocol_signature}))
    seal(base, 'synthetic_scores')
    labels = np.zeros((max(pool.gallery)+1, 8),dtype=np.uint8)
    classes = np.full(len(labels),-1,dtype=np.int64)
    for sid, lab, cls in zip(pool.gallery, ([1,0,0],[1,1,0],[0,1,0],[1,0,0],[0,0,1]), (0,1,2,0,3)):
        labels[sid,:3] = lab
        classes[sid] = cls
    data = {'cache':cache,'pool':pool,'bundle':SimpleNamespace(labels=labels,class_ids=classes)}
    return data, base


class SpawnedPipelineTests(unittest.TestCase):
    def test_spawned_diagnostic_requires_explicit_rotation_feature_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, baseline = disk_data(root)
            data['cache'].pop('feature_path')
            with self.assertRaisesRegex(ValueError, 'rotation feature path'):
                perturb.run(root/'run', data, 1, baseline)

    def test_single_and_two_workers_identical_complete_diagnostic(self):
        from wbm_retrieval.common import file_hash
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, baseline = disk_data(root)
            source_hashes = {str(p):file_hash(p) for p in root.rglob('*') if p.is_file()}
            for name, workers in (('one',1),('two',2)):
                result = perturb.run(root/name,data,workers,baseline)
                check = result['verification']
                self.assertEqual(check['queries'], 5)
                self.assertEqual(check['eligible_queries'], 2)
                self.assertEqual(check['single_region_queries'], 2)
                self.assertEqual(check['zero_region_queries'], 1)
                self.assertEqual(check['masked_score_rows'], 12)
                self.assertEqual(check['area_matched_queries'], 1)
                self.assertEqual(check['baseline_full_scores_byte_exact'], True)
                self.assertLessEqual(check['max_scalar_match_error'], 1e-12)
            for filename in ('baseline.csv','deletions.csv','strategies.csv','summary.csv','summary.json'):
                self.assertEqual((root/'one/d4'/filename).read_bytes(),
                                 (root/'two/d4'/filename).read_bytes())
            self.assertEqual(source_hashes,{str(p):file_hash(Path(p)) for p in source_hashes})
            with (root/'one/d4/summary.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            main = [r for r in rows if r['stage']=='test' and r['subset']=='main']
            self.assertTrue(all(r['num_queries']=='2' for r in main))
            with self.assertRaises(FileExistsError):
                perturb.run(root/'one',data,1,baseline)


if __name__ == '__main__':
    unittest.main()
