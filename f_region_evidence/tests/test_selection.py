import copy
from datetime import datetime, timezone, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from f_region_evidence import selection, state, run
from wbm_retrieval.common import file_hash, read_json, write_json


def results(hits=2000):
    return {name: {'top10_hits': hits, 'metrics': {'Precision@10': hits/5000},
                   'coverage_metrics': {'Coverage-nDCG@10': .8}}
            for name in state.grid()}


def set_hits(rows, name, hits):
    rows[name]['top10_hits'] = hits
    rows[name]['metrics']['Precision@10'] = hits/5000


class SelectionTests(unittest.TestCase):
    def test_fixed_twelve_and_family_partition(self):
        grid = state.grid()
        self.assertEqual(len(grid), 12)
        self.assertEqual(set(grid), {p+'_'+r for p in ('base4', 'top8', 'residual8')
                                    for r in ('none', 'q75', 'q90', 'q95')})
        self.assertEqual({f: sum(state.family(s) == f for s in grid.values())
                          for f in ('baseline', 'rejection', 'completion', 'combined')},
                         {'baseline': 1, 'rejection': 3, 'completion': 2, 'combined': 6})
        grid['base4_none']['proposal'] = 'changed'
        self.assertEqual(state.grid()['base4_none']['proposal'], 'base4')

    def test_complete_tie_prefers_baseline_then_complexity_then_name(self):
        chosen = selection.select(results())
        self.assertEqual(chosen['winner'], 'base4_none')
        self.assertEqual(chosen['valid_order'], [
            'base4_none', 'base4_q75', 'base4_q90', 'base4_q95',
            'residual8_none', 'top8_none', 'residual8_q75', 'top8_q75',
            'residual8_q90', 'top8_q90', 'residual8_q95', 'top8_q95'])
        self.assertEqual(chosen['selected_variants'], ['base4_q75', 'residual8_none', 'residual8_q75'])
        self.assertNotIn('base4_none', chosen['selected_variants'])

    def test_hits_take_priority_and_one_winner_per_nonbaseline_family(self):
        rows = results()
        set_hits(rows, 'base4_q95', 2200)
        set_hits(rows, 'top8_none', 2300)
        set_hits(rows, 'top8_q90', 2400)
        chosen = selection.select(rows)
        self.assertEqual(chosen['winner'], 'top8_q90')
        self.assertEqual(chosen['selected_variants'], ['base4_q95', 'top8_none', 'top8_q90'])
        self.assertEqual(set(chosen['selected_specs']), set(chosen['selected_variants']))
        self.assertEqual(chosen['family_winners']['baseline'], state.BASELINE)
        self.assertEqual(selection.select(dict(reversed(list(rows.items())))), chosen)

    def test_secondary_metrics_cannot_break_precision_ties(self):
        rows = results()
        rows['top8_q95']['coverage_metrics']['Coverage-nDCG@10'] = 1.
        rows['top8_q95']['metrics']['mAP@100'] = 1.
        self.assertEqual(selection.select(rows)['winner'], state.BASELINE)

    def test_missing_extra_or_malformed_valid_rows_fail_closed(self):
        for change in ('missing', 'extra'):
            rows = results()
            if change == 'missing':
                rows.pop('top8_q95')
            else:
                rows['unfrozen'] = rows['base4_none']
            with self.subTest(change=change), self.assertRaises(ValueError):
                selection.select(rows)
        for bad in (None, [], {'metrics': {}}, {'top10_hits': 2000}):
            rows = results()
            rows[state.BASELINE] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                selection.select(rows)

    def test_noninteger_hits_or_nonfinite_precision_cannot_select(self):
        for bad in (True, 2000., -1, 5001, None):
            rows = results()
            rows[state.BASELINE]['top10_hits'] = bad
            with self.subTest(hits=bad), self.assertRaises(ValueError):
                selection.select(rows)
        for bad in (float('nan'), float('inf'), -float('inf'), True, '0.4', None, -.1, 1.1, .4001):
            rows = results()
            rows[state.BASELINE]['metrics']['Precision@10'] = bad
            with self.subTest(precision=bad), self.assertRaises(ValueError):
                selection.select(rows)

    def test_freeze_is_idempotent_and_test_guard_requires_unchanged_complete_valid(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            rows = results()
            for name in state.grid():
                folder = output/'valid'/name
                folder.mkdir(parents=True)
                write_json(folder/'metrics.json', {'metrics': rows[name]['metrics']})
                state.seal(folder, state.evaluation_signature('context', 'valid', name))
                rows[name]['completed_sha256'] = file_hash(folder/'completed.json')
            write_json(output/'evaluation_index_valid.json', {'context_signature': 'context',
                       'stage': 'valid', 'split': 'valid', 'specs': state.grid(), 'results': rows})
            first = run.freeze(output, 'context', rows)
            later = datetime(2030, 1, 1, tzinfo=timezone(timedelta(hours=8)))
            with mock.patch.object(run, 'now', return_value=later):
                self.assertEqual(run.freeze(output, 'context', rows), first)
            self.assertEqual(run.frozen_selection(output, 'context'), first)
            with self.assertRaises(ValueError):
                run.frozen_selection(output, 'other_context')
            write_json(output/'valid'/state.BASELINE/'metrics.json', {'tampered': True})
            with self.assertRaises(ValueError):
                run.frozen_selection(output, 'context')

    def test_freeze_refuses_changed_selection_after_partial_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            rows = results()
            for name in rows:
                folder = output/'valid'/name
                folder.mkdir(parents=True)
                write_json(folder/'completed.json', {'fixture': name})
            write_json(output/'evaluation_index_valid.json', {'fixture': 'fixed'})
            run.freeze(output, 'context', rows)
            (output/'valid_completed.json').unlink()
            set_hits(rows, 'top8_q95', 2001)
            with self.assertRaises(ValueError):
                run.freeze(output, 'context', rows)


if __name__ == '__main__':
    unittest.main()
