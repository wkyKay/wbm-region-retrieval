from copy import deepcopy
from datetime import datetime, timezone, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from f_region_wbbs import selection, state
from wbm_retrieval.common import read_json, write_json


def results():
    return {name: {'coverage_metrics': {'Coverage-nDCG@10': .75}}
            for name in state.evaluation_specs('cross_valid')}


class SelectionTests(unittest.TestCase):
    def test_global_benchmark_cannot_replace_explainable_primary(self):
        rows = results()
        rows[state.GLOBAL]['coverage_metrics']['Coverage-nDCG@10'] = 1.
        chosen = selection.select(rows)
        self.assertEqual(chosen['winner'], state.BASELINE)
        self.assertEqual(chosen['best_regional_wbbs'], 'all_bb_a04')
        self.assertEqual(len(chosen['selected_variants']), 4)

    def test_cross_metric_selects_and_test_retains_four_fixed_controls(self):
        rows = results()
        rows['denoised_wbbs_a06']['coverage_metrics']['Coverage-nDCG@10'] = .76
        chosen = selection.select(rows)
        self.assertEqual(chosen['winner'], 'denoised_wbbs_a06')
        self.assertEqual(chosen['best_regional_wbbs'], 'denoised_wbbs_a06')
        self.assertEqual(len(chosen['selected_specs']), 5)
        self.assertEqual(set(chosen['fixed_controls']), {'all_bb_a04','all_wbbs_a04','denoised_bb_a04','denoised_wbbs_a04'})
        self.assertFalse(chosen['test_reselection'])
        self.assertEqual(selection.select(dict(reversed(list(rows.items())))), chosen)

    def test_other_metrics_do_not_select_or_break_ties(self):
        rows = results()
        rows['denoised_wbbs_a06']['metrics'] = {'Precision@10': 1., 'mAP@100': 1.}
        rows['denoised_wbbs_a06']['coverage_metrics']['FullCoverage-Precision@10'] = 1.
        rows['denoised_wbbs_a06']['coverage_metrics']['Coverage-nDCG@10'] += 5e-13
        self.assertEqual(selection.select(rows)['winner'], state.BASELINE)

    def test_missing_extra_and_nonfinite_metrics_rejected(self):
        for name in state.evaluation_specs('cross_valid'):
            rows = results(); rows.pop(name)
            with self.assertRaises(ValueError): selection.select(rows)
        rows = results(); rows['unexpected'] = deepcopy(rows[state.BASELINE])
        with self.assertRaises(ValueError): selection.select(rows)
        for bad in (True, None, '0.75', -1., 1.1, float('nan'), float('inf')):
            rows = results(); rows[state.BASELINE]['coverage_metrics']['Coverage-nDCG@10'] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError): selection.select(rows)

    def test_actual_time_identity_immutable_and_split_protocol_aware(self):
        stamp = datetime(2026,11,21,14,0,tzinfo=timezone(timedelta(hours=8)))
        with tempfile.TemporaryDirectory() as temp, patch.object(state,'now',return_value=stamp):
            for stage in ('valid','cross_valid','test','cross_test'):
                ident = state.identity(Path(temp),stage,'all_bb_a04','signature')
                self.assertEqual(ident['started_at'],stamp.isoformat())
                self.assertEqual(ident['split'],'valid' if stage.endswith('valid') else 'test')
                self.assertEqual(ident['protocol'],'coverage_cross_combination_v1' if stage.startswith('cross_') else 'retrieval_v1')
                self.assertEqual(state.identity(Path(temp),stage,'all_bb_a04','signature'),ident)
                with self.assertRaises(ValueError): state.identity(Path(temp),stage,'all_bb_a04','changed')

    def test_test_cannot_run_without_frozen_selection(self):
        with self.assertRaises(ValueError): state.evaluation_specs('test')
        with self.assertRaises(ValueError): state.evaluation_specs('cross_test')
        self.assertEqual(len(state.evaluation_specs('valid')),13)
        self.assertEqual(len(state.evaluation_specs('cross_valid')),14)


if __name__=='__main__': unittest.main()
