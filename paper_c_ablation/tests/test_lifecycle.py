"""Regression tests for pre-result freezing, test gates and truthful completion."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from paper_c_ablation import inputs, register, run, state
from wbm_retrieval.common import digest, read_json, write_json


def data_fixture():
    return {'source_binding': {'/frozen/source': 'a'*64}, 'source_compatibility': [],
            'source_metadata': {'region_source': '/frozen/regions'}, 'replay_references': {},
            'config': {'matching': {'unchanged': True}}}


class LifecycleTests(unittest.TestCase):
    def test_output_refuses_historical_and_nested_roots(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(run, 'ROOT', Path(temp)):
            args = run.parser().parse_args(['--output-root', 'artificial/unsafe'])
            with self.assertRaises(ValueError):
                run.output_path(args)
            old = Path(temp)/'results/old'
            old.mkdir(parents=True)
            write_json(old/'run_manifest.json', {})
            args = run.parser().parse_args(['--output-root', 'results/old/new'])
            with self.assertRaises(ValueError):
                run.output_path(args)

    def test_plan_is_persisted_before_any_evaluation_and_cannot_be_rewritten(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(run, 'code_identity', return_value={'science': 'a'*64}):
            output = Path(temp)/'new'
            data = data_fixture()
            context = run.make_context(data)
            signature = run.prepare_root(output, context, data)
            frozen = state.read_signed(output/'frozen_plan.json')
            self.assertEqual(frozen['specs'], state.grid())
            self.assertEqual(frozen['variants'], list(state.grid()))
            self.assertFalse((output/'valid').exists())
            self.assertEqual(frozen['context_signature'], signature)
            self.assertEqual(run.prepare_root(output, context, data, resume=True), signature)
            altered = dict(context, ks=[1, 10, 100])
            with self.assertRaises(ValueError):
                run.prepare_root(output, altered, data, resume=True)
            bad = {**frozen, 'variants': ['primary']}
            write_json(output/'frozen_plan.json', state.signed({k:v for k,v in bad.items() if k != 'signature'}))
            with self.assertRaises(ValueError):
                run.prepare_root(output, context, data, resume=True)

    def test_identity_is_stable_and_rejects_changed_science_or_timestamp(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            signature = state.evaluation_signature('context', 'valid', 'no_a')
            first = state.identity(output, 'valid', 'no_a', signature)
            self.assertEqual(first, state.identity(output, 'valid', 'no_a', signature))
            with self.assertRaises(ValueError):
                state.identity(output, 'valid', 'no_a', 'new-science')
            path = output/'identities/valid__no_a.json'
            write_json(path, {**first, 'started_at': '2026-11-21T11:00:00Z'})
            with self.assertRaises(ValueError):
                state.identity(output, 'valid', 'no_a', signature)

    def test_absent_gate_never_invokes_stage_validator(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(register, 'validate_stage') as validator:
            with self.assertRaisesRegex(ValueError, 'complete independently verified'):
                run.require_valid_gate(Path(temp), 'context')
            validator.assert_not_called()

    def _pipeline(self, output, data, stage, context, workers):
        frozen = state.read_signed(output/'frozen_plan.json')
        self.assertEqual(set(frozen['variants']), set(state.grid()))
        if stage != 'valid':
            self.assertTrue((output/'valid_completed.json').exists())
        index = {'context_signature': context, 'stage': stage,
                 'split': 'valid' if stage == 'valid' else 'test', 'specs': state.grid(),
                 'results': {n: {'completed_sha256': 'b'*64} for n in state.grid()}}
        write_json(output/f'evaluation_index_{stage}.json', index)
        return index

    def test_failed_valid_verification_blocks_all_test_input_reads(self):
        from paper_c_ablation import pipeline
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(run, 'ROOT', Path(temp)))
            stack.enter_context(patch.object(run, 'code_identity', return_value={'science': 'a'*64}))
            loader = stack.enter_context(patch.object(inputs, 'load', return_value=data_fixture()))
            stack.enter_context(patch.object(run, 'protection'))
            stack.enter_context(patch.object(run, 'check_sources'))
            stack.enter_context(patch.object(pipeline, 'evaluate', side_effect=self._pipeline))
            stack.enter_context(patch.object(register, 'validate_stage', side_effect=ValueError('bad valid evidence')))
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'bad valid'):
                run.main(['--stage', 'all', '--output-root', 'results/new'])
            self.assertEqual(loader.call_args_list[0].args, ('valid',))
            self.assertEqual(loader.call_count, 1)
            self.assertFalse((Path(temp)/'results/new/valid_completed.json').exists())
            self.assertFalse((Path(temp)/'results/new/completed.json').exists())

    def test_failed_final_validation_removes_success_marker_and_preserves_evidence(self):
        from paper_c_ablation import pipeline
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(run, 'ROOT', Path(temp)))
            stack.enter_context(patch.object(run, 'code_identity', return_value={'science': 'a'*64}))
            output = Path(temp)/'results/new'
            def load(split):
                if split == 'test':
                    self.assertTrue((output/'valid_completed.json').exists())
                return data_fixture()
            stack.enter_context(patch.object(inputs, 'load', side_effect=load))
            stack.enter_context(patch.object(run, 'protection'))
            stack.enter_context(patch.object(run, 'check_sources'))
            stack.enter_context(patch.object(pipeline, 'evaluate', side_effect=self._pipeline))
            stack.enter_context(patch.object(register, 'validate_stage',
                side_effect=lambda out,stage,ctx: read_json(out/f'evaluation_index_{stage}.json')))
            stack.enter_context(patch.object(register, 'verify_run', side_effect=ValueError('rejected seal')))
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'rejected seal'):
                run.main(['--stage', 'all', '--output-root', 'results/new'])
            self.assertFalse((output/'completed.json').exists())
            self.assertEqual(read_json(output/'status.json')['status'], 'failed')
            failed = list((output/'failed_completion_checks').glob('*/completed.json'))
            self.assertEqual(len(failed), 1)
            self.assertEqual(read_json(failed[0])['status'], 'completed')
            self.assertTrue((output/'valid_completed.json').exists())


if __name__ == '__main__':
    unittest.main()
