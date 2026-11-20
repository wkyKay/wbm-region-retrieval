"""No overwrite, no test-first execution, and stable valid selection."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wbm_retrieval.common import read_json, write_json
from f_region_area_fusion import inputs, run, state, scoring


class Guards(unittest.TestCase):
    def test_dry_run_performs_no_input_io_or_writes(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(run, 'ROOT', Path(tmp)), \
                patch.object(inputs, 'load_valid', side_effect=AssertionError('I/O')), \
                patch.object(inputs, 'load_test', side_effect=AssertionError('I/O')), redirect_stdout(StringIO()) as capture:
            self.assertEqual(run.main(['--dry-run']), 0)
            result = json.loads(capture.getvalue())
            self.assertEqual(len(result['full_grid']), 29)
            self.assertFalse(result['training'])
            self.assertFalse(result['test_parameter_search'])
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_test_requires_explicit_valid_source_and_failed_gate_opens_no_cache(self):
        with self.assertRaisesRegex(ValueError, 'Test requires --source-run'):
            run.main(['--stage', 'test', '--dry-run'])
        with patch.object(inputs, 'frozen_choice', side_effect=ValueError('invalid selection')), \
                patch.object(inputs, 'load_valid', side_effect=AssertionError('cache access')):
            with self.assertRaisesRegex(ValueError, 'invalid selection'):
                inputs.load_test('/tmp/unsealed-valid')

    def test_existing_nested_external_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(run, 'ROOT', Path(tmp)):
            root = Path(tmp)
            old = root/'artificial/old'
            old.mkdir(parents=True)
            write_json(old/'run_manifest.json', {'fixture': True})
            for target in ['artificial/old', 'artificial/old/nested', 'artificial', 'artifacts/new']:
                with self.assertRaises(ValueError):
                    run.main(['--dry-run', '--output-root', target])
            for workers in ['0', '9']:
                with self.assertRaises(ValueError):
                    run.main(['--dry-run', '--workers', workers])

    def test_identity_is_stable_and_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = state.identity(root, 'valid', 'entropy', 'sig')
            self.assertEqual(state.identity(root, 'valid', 'entropy', 'sig'), first)
            second = state.identity(root, 'test', 'entropy', 'sig2')
            self.assertNotEqual(first['run_id'], second['run_id'])
            with self.assertRaises(ValueError):
                state.identity(root, 'valid', 'entropy', 'changed')
            p = root/'identities/valid__entropy.json'
            wrong = read_json(p)
            wrong['started_at'] = wrong['started_at'].replace('+08:00', '+00:00')
            write_json(p, wrong)
            with self.assertRaises(ValueError):
                state.identity(root, 'valid', 'entropy', 'sig')

    def test_freeze_keeps_first_timestamp_and_rejects_changed_valid_ranking(self):
        rows = {n: {'top10_hits': i*10, 'num_queries': 500} for i, n in enumerate(scoring.frozen_grid())}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root/'evaluation_index_valid.json', {'fixture': True})
            first = run.freeze_selection(root, 'sig', rows)
            self.assertEqual(len(first['selected_variants']), 10)
            self.assertEqual(run.freeze_selection(root, 'sig', rows), first)
            rows['entropy']['top10_hits'] = 0
            with self.assertRaisesRegex(ValueError, 'selection changed'):
                run.freeze_selection(root, 'sig', rows)


if __name__ == '__main__':
    unittest.main()
