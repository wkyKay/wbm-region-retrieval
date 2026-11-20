"""Valid-only/no-overwrite guards and immutable execution identities."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wbm_retrieval.common import read_json, write_json
from f_region_score_search import run, state


class CommandGuardTests(unittest.TestCase):
    def test_dry_run_is_valid_only_and_performs_no_source_io(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(run, 'ROOT', Path(tmp)), \
                patch.object(run.inputs, 'load', side_effect=AssertionError('source I/O')), redirect_stdout(StringIO()) as capture:
            self.assertEqual(run.main(['--dry-run', '--output-root', 'artificial/fresh']), 0)
            value = json.loads(capture.getvalue())
            self.assertEqual(value['split'], 'valid')
            self.assertFalse(value['test_evaluated'])
            self.assertFalse(value['training'])
            self.assertEqual(len(value['grid']), 15)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_test_and_all_stage_arguments_are_unavailable(self):
        with patch.object(run.inputs, 'load', side_effect=AssertionError('source I/O')), redirect_stderr(StringIO()):
            for stage in ('test', 'all'):
                with self.assertRaises(SystemExit):
                    run.main(['--stage', stage])

    def test_existing_nested_external_and_invalid_workers_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(run, 'ROOT', Path(tmp)):
            root = Path(tmp)
            old = root/'artificial/old'
            old.mkdir(parents=True)
            write_json(old/'run_manifest.json', {'fixture': True})
            for name in ('artificial/old', 'artificial/old/nested', 'artifacts/new', 'artificial'):
                with self.assertRaises(ValueError):
                    run.main(['--dry-run', '--output-root', name])
            for workers in ('0', '9'):
                with self.assertRaises(ValueError):
                    run.main(['--dry-run', '--workers', workers])
            with self.assertRaises(ValueError):
                run.main(['--dry-run', '--resume-root', 'artificial/missing'])

    def test_persisted_identity_cannot_be_reassigned_or_malformed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            first = state.identity(p, 'product', 'sig')
            self.assertEqual(state.identity(p, 'product', 'sig'), first)
            self.assertTrue(first['key'].startswith('f_region_score_search_'))
            with self.assertRaises(ValueError):
                state.identity(p, 'product', 'different')
            path = p/'identities/product.json'
            bad = read_json(path)
            bad['started_at'] = bad['started_at'].replace('+08:00', '+00:00')
            write_json(path, bad)
            with self.assertRaises(ValueError):
                state.identity(p, 'product', 'sig')


if __name__ == '__main__':
    unittest.main()
