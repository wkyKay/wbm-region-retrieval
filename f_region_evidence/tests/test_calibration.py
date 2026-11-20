import copy
import json
import unittest
from unittest import mock

import numpy as np

from f_region_area_fusion.scoring import independent_components, independent_score_components
from f_region_evidence import calibration, scoring
from f_region_evidence.tests.test_scoring import fixture


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.train, _ = fixture((1, 2, 4, 0, 3, 1, 2, 4, 2, 1, 3, 4), start=100)
        self.query, _ = fixture((3, 8, 0, 1), start=1000)
        self.reference = calibration.select_reference(self.train, size=8)

    def test_train_selection_deterministic_one_token_per_owner(self):
        again = calibration.select_reference(self.train, size=8)
        for key in ('token_indices', 'owner_ids', 'local_indices'):
            np.testing.assert_array_equal(again[key], self.reference[key])
        self.assertEqual(len(np.unique(self.reference['owner_ids'])), 8)
        self.assertNotIn(103, self.reference['owner_ids'])  # Empty train wafer.
        np.testing.assert_array_equal(self.train['owners'][self.reference['token_indices']],
                                      self.reference['owner_ids'])
        other = calibration.select_reference(self.train, size=8, seed=2027)
        self.assertNotEqual(other['reference_identity_sha256'], self.reference['reference_identity_sha256'])

    def test_selected_owner_and_local_token_are_independent_of_cache_map_order(self):
        order = np.arange(len(self.train['map_ids']))[::-1]
        token_order = np.concatenate([np.arange(*self.train['offsets'][row:row+2]) for row in order])
        altered = {name: value[token_order] if len(value) == len(self.train['features']) else value
                   for name, value in self.train.items()}
        altered['map_ids'] = self.train['map_ids'][order]
        altered['offsets'] = np.r_[0, np.cumsum(np.diff(self.train['offsets'])[order])]
        after = calibration.select_reference(altered, size=8)
        np.testing.assert_array_equal(after['owner_ids'], self.reference['owner_ids'])
        np.testing.assert_array_equal(after['local_indices'], self.reference['local_indices'])
        np.testing.assert_array_equal(altered['features'][after['token_indices']],
                                      self.train['features'][self.reference['token_indices']])

    def test_quantiles_match_independent_pair_formulas(self):
        actual, meta = calibration.calibrate(self.query, self.train, self.reference)
        selected = self.reference['token_indices']
        n = len(self.query['features'])
        h = np.concatenate((self.query['features'], self.train['features'][selected]))
        r = np.concatenate((self.query['relative_area'], self.train['relative_area'][selected]))
        p = np.concatenate((self.query['joint_all'], self.train['joint_all'][selected]))
        owners = np.concatenate((self.query['owners'], self.train['owners'][selected]))
        independent = independent_components(np.arange(n), h, r, p, owners,
                                             candidate_indices=np.arange(n, n+8), with_entropy=False)
        scores = independent_score_components(independent, scoring.SPEC)
        expected = np.quantile(scores, calibration.QUANTILES, axis=1, method='linear')
        for row, name in enumerate(('q75', 'q90', 'q95')):
            np.testing.assert_allclose(actual[name], expected[row], atol=1e-12, rtol=0.)
        self.assertFalse(meta['labels_used'])
        self.assertEqual(meta['region_reference_pairs'], n*8)
        self.assertTrue(np.all(actual['q75'] <= actual['q90']))
        self.assertTrue(np.all(actual['q90'] <= actual['q95']))

    def test_other_query_regions_do_not_fit_reference_or_thresholds(self):
        before, _ = calibration.calibrate(self.query, self.train, self.reference)
        altered = copy.deepcopy(self.query)
        altered['features'][3:] *= -1.
        altered['relative_area'][3:] = .9
        altered['joint_all'][3:] = np.eye(1, 128)[0]
        after, _ = calibration.calibrate(altered, self.train, self.reference)
        for name in before:
            np.testing.assert_array_equal(before[name][:3], after[name][:3])

    def test_reference_accepts_json_roundtrip_and_rejects_mutation(self):
        converted = json.loads(json.dumps(self.reference, default=lambda x: x.tolist()))
        actual, _ = calibration.calibrate(self.query, self.train, converted)
        expected, _ = calibration.calibrate(self.query, self.train, self.reference)
        for name in actual:
            np.testing.assert_array_equal(actual[name], expected[name])
        for key in ('token_indices', 'owner_ids', 'local_indices'):
            changed = copy.deepcopy(converted)
            changed[key][0] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                calibration.calibrate(self.query, self.train, changed)
        changed = copy.deepcopy(converted)
        changed['rule'] = 'any_region'
        with self.assertRaises(ValueError):
            calibration.calibrate(self.query, self.train, changed)

    def test_never_reads_labels_or_mutates_sources(self):
        class NoLabels(dict):
            def __getitem__(self, key):
                if 'label' in key or 'class' in key:
                    raise AssertionError('Calibration must not inspect labels')
                return super().__getitem__(key)
        train = NoLabels(self.train)
        query = NoLabels(self.query)
        before = {key: value.tobytes() for key, value in train.items()}
        calibration.calibrate(query, train, calibration.select_reference(train, size=8))
        self.assertEqual(before, {key: value.tobytes() for key, value in train.items()})

    def test_train_query_overlap_insufficient_reference_and_changed_grid_fail(self):
        with self.assertRaises(ValueError):
            calibration.calibrate(self.train, self.train, self.reference)
        for size in (0, 12, True, 2.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                calibration.select_reference(self.train, size=size)
        with self.assertRaises(ValueError):
            calibration.calibrate(self.query, self.train, self.reference, quantiles=(.8, .9, .95))

    def test_exact_one_background_is_recorded_with_finite_thresholds(self):
        train = copy.deepcopy(self.train)
        query = copy.deepcopy(self.query)
        for cache in (train, query):
            cache['features'][:] = 0.
            cache['features'][:, 0] = 1.
            cache['joint_all'][:] = 0.
            cache['joint_all'][:, 0] = 1.
            cache['relative_area'][:] = .1
        result, meta = calibration.calibrate(query, train, calibration.select_reference(train, size=8))
        for name, value in result.items():
            np.testing.assert_array_equal(value, np.ones(len(query['features'])))
            self.assertEqual(meta['tau_equal_one_counts'][name], len(query['features']))
        self.assertEqual(meta['tau_equal_one_policy'], scoring.TAU_ONE_POLICY)


if __name__ == '__main__':
    unittest.main()
