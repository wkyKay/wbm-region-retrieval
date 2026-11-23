from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

from cnn_refinement import followthrough as ft, register, run, state
from f_region_structure.state import seal
from wbm_retrieval.common import digest, file_hash, read_json, write_json


class FollowthroughTests(unittest.TestCase):
    def test_launch_whitelist_can_never_train_a_third_model(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(ft.subprocess, 'Popen') as popen:
            for stage in ('train', 'teacher', 'init'):
                with self.assertRaisesRegex(ValueError, 'never training'):
                    ft.launch(Path(temp), stage, stage)
            with self.assertRaisesRegex(ValueError, 'never training'):
                ft.launch(Path(temp), 'evaluate_baseline', 'evaluate', ('--models', 'baseline'))
            popen.assert_not_called()

    def test_receipt_is_durable_before_child_exec_and_duplicate_log_blocks_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            process = SimpleNamespace(pid=1234)
            def release(fd, token):
                receipt = read_json(output/'launch_finalize.json')
                self.assertEqual(receipt['pid'], 1234)
                self.assertIn('finalize', receipt['command'])
                self.assertEqual(token, b'1')
                return 1
            with patch.object(ft.subprocess, 'Popen', return_value=process) as popen, \
                    patch.object(ft.os, 'write', side_effect=release):
                self.assertIs(ft.launch(output, 'finalize', 'finalize'), process)
            self.assertIn('os.execv', popen.call_args.args[0][2])
            with patch.object(ft.subprocess, 'Popen') as duplicate:
                with self.assertRaisesRegex(ValueError, 'already launched'):
                    ft.launch(output, 'finalize', 'finalize')
                (output/'launch_finalize.json').unlink()
                with self.assertRaises(FileExistsError):
                    ft.launch(output, 'finalize', 'finalize')
                duplicate.assert_not_called()

    def test_supervisor_lock_excludes_concurrent_driver(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            with ft.supervisor_lock(output):
                with self.assertRaisesRegex(RuntimeError, 'Another followthrough'):
                    with ft.supervisor_lock(output):
                        self.fail('Concurrent supervisor entered')
            with ft.supervisor_lock(output):
                pass

    def test_real_exec_gate_runs_only_after_receipt_exists_using_a_toy_module(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root/'output'
            output.mkdir()
            package = root/'toy_repository/cnn_refinement'
            package.mkdir(parents=True)
            (package/'__init__.py').write_text('')
            (package/'run.py').write_text(
                'import json,sys\nfrom pathlib import Path\n'
                'output=Path(sys.argv[sys.argv.index("--output-root")+1])\n'
                'receipt=json.loads((output/"launch_finalize.json").read_text())\n'
                'assert receipt["command"][2:4]==["-m","cnn_refinement.run"]\n'
                '(output/"logs/toy_success.txt").write_text("receipt existed before exec")\n')
            with patch.object(ft, 'ROOT', package.parent):
                process = ft.launch(output, 'finalize', 'finalize')
                self.assertEqual(process.wait(timeout=10), 0)
            self.assertEqual((output/'logs/toy_success.txt').read_text(), 'receipt existed before exec')

    def test_process_identity_matches_full_recorded_command(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            command = ['/python', '-m', 'cnn_refinement.run', '--output-root', str(output)]
            write_json(output/'launch_context_dropout.json', {'pid': 123, 'command': command})
            with patch.object(ft, 'alive', return_value=False) as alive:
                with self.assertRaisesRegex(RuntimeError, 'exited'):
                    ft.require_running(output, 'context_dropout')
                alive.assert_called_once_with(123, command)

    def test_evaluation_index_does_not_hide_nonzero_exit_or_skip_full_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            write_json(output/'evaluation/context_dropout/evaluation_index_test.json', {'partial': True})
            failed = SimpleNamespace(poll=lambda: 1)
            with patch.object(register, 'validate_model_split') as validate:
                with self.assertRaisesRegex(RuntimeError, 'failed with exit 1'):
                    ft.evaluation_complete(output, 'context_dropout', failed)
                self.assertFalse(ft.evaluation_complete(output, 'context_dropout', SimpleNamespace(poll=lambda: None)))
                validate.assert_not_called()
                self.assertTrue(ft.evaluation_complete(output, 'context_dropout', SimpleNamespace(poll=lambda: 0)))
                self.assertEqual([call.args[2] for call in validate.call_args_list], ['valid', 'test'])
            with patch.object(register, 'validate_model_split', side_effect=ValueError('unsealed')):
                with self.assertRaisesRegex(ValueError, 'unsealed'):
                    ft.evaluation_complete(output, 'context_dropout', SimpleNamespace(poll=lambda: 0))

    def test_dead_incomplete_stage_cannot_be_restarted_by_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            write_json(output/'launch_report.json', {'pid': 123, 'command': ['python', 'failed']})
            with patch.object(ft, 'alive', return_value=False), patch.object(ft, 'launch') as launch:
                with self.assertRaisesRegex(RuntimeError, 'exited'):
                    ft.synchronous(output, 'report', 'report', resume=True, manifest={'signature': 'context'})
                launch.assert_not_called()

    def test_sealed_stage_resume_is_read_only_and_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            folder = output/'report'
            science = {'context_signature': 'context'}
            signature = digest(science)
            write_json(folder/'manifest.json', {**science, 'signature': signature})
            write_json(folder/'summary.json', {'signature': signature})
            seal(folder, signature)
            before = {p: p.read_bytes() for p in folder.rglob('*') if p.is_file()}
            with patch.object(ft, 'launch') as launch:
                ft.synchronous(output, 'report', 'report', resume=True, manifest={'signature': 'context'})
                launch.assert_not_called()
            self.assertTrue(all(p.read_bytes() == content for p, content in before.items()))
            write_json(folder/'summary.json', {'signature': signature, 'changed': True})
            with self.assertRaisesRegex(ValueError, 'Changed or missing'):
                ft.synchronous(output, 'report', 'report', resume=True, manifest={'signature': 'context'})

    def test_success_path_launches_exactly_two_evaluations_and_never_training(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            events = []
            def synchronous(output, name, stage, **kwargs):
                events.append(stage)
            def launch(output, name, stage, extra, **kwargs):
                events.append(name)
                self.assertEqual(stage, 'evaluate')
                return Mock()
            with patch.object(ft, 'synchronous', side_effect=synchronous), \
                    patch.object(ft, 'training_complete', return_value=True), \
                    patch.object(ft, 'evaluation_complete', return_value=True), \
                    patch.object(ft, 'launch', side_effect=launch) as launched:
                ft.supervise(output, {'signature': 'context'})
            self.assertEqual(launched.call_count, 2)
            self.assertEqual(events, ['negative-diagnostics', 'evaluate_context_dropout',
                'evaluate_filtered_negatives', 'diagnose', 'report', 'finalize'])

    def test_training_failure_stops_before_any_evaluation_or_final_seal(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            with patch.object(ft, 'synchronous') as sync, \
                    patch.object(ft, 'training_complete', return_value=False), \
                    patch.object(ft, 'require_running', side_effect=RuntimeError('training failed')), \
                    patch.object(ft, 'launch') as launch:
                with self.assertRaisesRegex(RuntimeError, 'training failed'):
                    ft.supervise(output, {'signature': 'context'})
                self.assertEqual(sync.call_count, 1)
                self.assertEqual(sync.call_args.args[2], 'negative-diagnostics')
                launch.assert_not_called()

    def test_finalize_preverification_failure_never_writes_completed(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            argv = ['cnn_refinement.run', '--output-root', str(output), '--stage', 'finalize']
            with patch.object(run.sys, 'argv', argv), patch.object(run, 'run_context', return_value={'signature': 'c'}), \
                    patch('cnn_refinement.inputs.load', return_value={'protection': {}}), \
                    patch('f_representation_audit.inputs.verify_protection'), \
                    patch('f_region_structure.state.protection'), \
                    patch.object(register, 'verify_run', side_effect=ValueError('report incomplete')) as verify:
                with self.assertRaisesRegex(ValueError, 'report incomplete'):
                    run.main()
                self.assertFalse((output/'completed.json').exists())
                self.assertFalse((output/'status.json').exists())
                self.assertEqual(verify.call_args.kwargs, {'require_root_seal': False})

    def test_finalization_excludes_only_logs_and_later_catalog_artifacts_remain_compatible(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp).resolve()
            write_json(output/'scientific.json', {'data': 'fixed'})
            (output/'logs').mkdir()
            (output/'logs/driver.log').write_text('running\n')
            manifest = {'signature': 'c', 'run_id': 'actual_run'}
            argv = ['cnn_refinement.run', '--output-root', str(output), '--stage', 'finalize']
            calls = []
            def verify(path, **kwargs):
                calls.append(kwargs)
                if not kwargs:
                    register._root_seal(path, 'c', ('scientific.json', 'status.json'))
            with patch.object(run.sys, 'argv', argv), patch.object(run, 'run_context', return_value=manifest), \
                    patch('cnn_refinement.inputs.load', return_value={'protection': {}}), \
                    patch('f_representation_audit.inputs.verify_protection'), \
                    patch('f_region_structure.state.protection'), patch.object(register, 'verify_run', side_effect=verify):
                run.main()
            self.assertEqual(calls, [{'require_root_seal': False}, {}])
            marker = read_json(output/'completed.json')
            self.assertEqual(marker['excluded'], ['logs/'])
            self.assertNotIn('logs/driver.log', marker['files'])
            (output/'logs/driver.log').write_text('running\nfinished\n')
            write_json(output/'catalog_sources.json', {'registered': True})
            write_json(output/'catalog_backup/transaction/manifest.json', {'backup': True})
            register._root_seal(output, 'c', ('scientific.json', 'status.json'))
            write_json(output/'unsealed_science.json', {'unexpected': True})
            with self.assertRaisesRegex(ValueError, 'omits scientific'):
                register._root_seal(output, 'c', ('scientific.json', 'status.json'))

    def test_auxiliary_missing_negative_or_report_cannot_be_registered(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                register._auxiliary(Path(temp), 'context', 'graph')


if __name__ == '__main__':
    unittest.main()
