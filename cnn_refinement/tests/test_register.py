import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from cnn_refinement import evaluation, register, state
from f_region_structure.state import seal
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from wbm_retrieval.training import parameter_hash


class RegistrationTests(unittest.TestCase):
    def root_fixture(self, output):
        stamp = '2026-11-23T12:00:00+08:00'
        plan = state.signed({'plan': state.PLAN, 'frozen_at': stamp})
        write_json(output/'frozen_plan.json', plan)
        manifest = state.signed({'version': state.VERSION, 'started_at': stamp,
            'run_id': '20261123T120000+0800__cnn_refinement', 'plan_signature': plan['signature'],
            'training_labels': False, 'test_parameter_search': False, 'authorized_models': list(state.VARIANTS),
            'split_counts': {'train': 25193, 'valid': 3599, 'test': 7199},
            'region_counts': {'train': 99786, 'valid': 14240}, 'source_binding': {},
            'baseline': {'training_summary': {'completed': True, 'epochs_completed': 100,
                'best_nce_epoch': 100, 'optimizer_steps': 39000, 'initial_parameter_sha256': register.INITIAL_HASH}}})
        write_json(output/'run_manifest.json', manifest)
        return manifest

    def evaluation_grid(self, output, cross_valid_reference=True):
        for model in state.MODELS:
            for split in ('valid', 'test'):
                rows = {}
                for stage in (split, 'cross_'+split):
                    for style in evaluation.STYLES:
                        replay = model == 'baseline' and style == 'fusion' and (
                            stage != 'cross_valid' or cross_valid_reference)
                        folder = output/'evaluation'/model/stage/style
                        signature = digest({'model': model, 'stage': stage, 'style': style})
                        ident = evaluation._identity(output/'evaluation'/model, stage, style, signature)
                        write_json(folder/'run_identity.json', ident)
                        write_json(folder/'evaluation_manifest.json', {**ident, 'signature': signature})
                        seal(folder, signature)
                        rows[stage+'/'+style] = {'path': str(folder.resolve()), 'register': not replay,
                            'historical_reference': replay, 'stage': stage, 'style': style,
                            'model': model, 'split': split}
                write_json(output/'evaluation'/model/f'evaluation_index_{split}.json', {'evaluations': rows})

    def test_twenty_new_identities_leave_all_four_fusion_baselines_unregistered(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root/'results/run'
            self.evaluation_grid(output)
            with patch.object(register, 'verify_run'):
                result = register.proposal(output, root=root)
            self.assertEqual(len(result['runs']), 20)
            self.assertEqual(len({row['key'] for row in result['runs']}), 20)
            self.assertTrue(all(row['variant'] != 'CNNRefinement/baseline/fusion' for row in result['runs']))
            self.assertEqual(sum(row['variant'] == 'CNNRefinement/baseline/cnn_only' for row in result['runs']), 4)
            self.assertTrue(all(row['identity_schema_version'] == 1 and row['selection'] is None for row in result['runs']))

    def test_missing_crossvalid_reference_allows_exactly_one_additional_baseline_role(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root/'results/run'
            self.evaluation_grid(output, cross_valid_reference=False)
            with patch.object(register, 'verify_run'):
                result = register.proposal(output, root=root)
            self.assertEqual(len(result['runs']), 21)
            baseline = [row for row in result['runs'] if row['variant'] == 'CNNRefinement/baseline/fusion']
            self.assertEqual([(row['split'], row['protocol']) for row in baseline], [('valid', register.PROTOCOL)])

    def test_plan_budget_timestamp_and_missing_root_seal_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root/'results/run'
            manifest = self.root_fixture(output)
            self.assertEqual(register._root_context(output), manifest)
            with self.assertRaisesRegex(ValueError, 'completed sealed'):
                register.verify_run(output, root=root)
            altered = {**manifest, 'region_counts': {'train': 99786, 'valid': 2000}}
            write_json(output/'run_manifest.json', state.signed({k: v for k, v in altered.items() if k != 'signature'}))
            with self.assertRaisesRegex(ValueError, 'deduplicated'):
                register._root_context(output)
            self.root_fixture(output)
            plan = read_json(output/'frozen_plan.json')
            plan['plan']['common_training']['epochs'] = 15
            write_json(output/'frozen_plan.json', state.signed({k: v for k, v in plan.items() if k != 'signature'}))
            with self.assertRaisesRegex(ValueError, 'plan'):
                register._root_context(output)
            with self.assertRaisesRegex(ValueError, 'start time'):
                register._stamp('2026-11-23T04:00:00Z')

    def test_training_cannot_use_bestloss_or_99_epoch_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            root = self.root_fixture(output)
            variant = 'context_dropout'
            folder = output/'training'/variant
            folder.mkdir(parents=True)
            local = {'epochs': 100, 'learning_rate': .01, 'momentum': .9, 'weight_decay': .001,
                     'projector_dim': 128, 'temperature': .07, 'num_negatives': 5000, 'memory_momentum': .5,
                     'optimizer': 'sgd', 'scheduler': 'wapirl_cosine', 'projector_type': 'linear',
                     'negative_sampling': 'with_replacement'}
            science = {'variant': variant, 'seed': 2026, 'batch_size': 256,
                'initial_parameter_sha256': register.INITIAL_HASH, 'local': local, 'input': {'test': True},
                'refinement_context_signature': root['signature'], 'plan_signature': root['plan_signature'],
                'source_protection': {}, 'selection': 'fixed_epoch_100_no_best_loss_or_test_selection',
                'neighbor_graph': None, 'code': {'cnn_refinement/training.py': file_hash(register.ROOT/'cnn_refinement/training.py')}}
            signature = digest(science)
            run_id = '20261123T120001+0800__cnn_context_dropout_seed2026_epoch100'
            manifest = state.signed({'science': science, 'training_signature': signature,
                'run_id': run_id, 'started_at': '2026-11-23T12:00:01+08:00'})
            write_json(folder/'run_manifest.json', manifest)
            history = [{'epoch': i, 'optimizer_steps': i*390, 'train_loss': .1, 'valid_nce': .2} for i in range(1, 101)]
            write_json(folder/'history.json', history)
            model = torch.nn.Linear(2, 2).state_dict()
            summary = {'completed': True, 'epochs_completed': 100, 'max_epochs': 100, 'optimizer_steps': 39000,
                'initial_parameter_sha256': register.INITIAL_HASH, 'checkpoint_selection': 'fixed_epoch_100',
                'run_id': run_id, 'started_at': manifest['started_at'], 'final_parameter_sha256': parameter_hash(model)}
            write_json(folder/'training_summary.json', summary)
            checkpoint = {'signature': signature, 'epoch': 100, 'variant': variant, 'optimizer_steps': 39000,
                'initial_parameter_sha256': register.INITIAL_HASH, 'input_spec': science['input'],
                'selection': 'fixed_epoch_100', 'model': model}
            torch.save(checkpoint, folder/'checkpoint.pt')
            # Completion required-file validation is separately covered by _complete;
            # this fixture isolates checkpoint-budget and model-byte checks.
            with patch.object(register, '_complete'):
                self.assertEqual(register._training(output, variant, root, 'graph'), folder/'checkpoint.pt')
                checkpoint['epoch'] = 99
                torch.save(checkpoint, folder/'checkpoint.pt')
                with self.assertRaisesRegex(ValueError, 'epoch100'):
                    register._training(output, variant, root, 'graph')
                checkpoint['epoch'] = 100
                checkpoint['selection'] = 'best_nce'
                torch.save(checkpoint, folder/'checkpoint.pt')
                with self.assertRaisesRegex(ValueError, 'epoch100'):
                    register._training(output, variant, root, 'graph')

    def test_replay_must_have_zero_score_error_and_the_old_key(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp).resolve()
            source = output/'old'
            source.mkdir()
            (source/'rankings.csv').write_text('old immutable rankings\n')
            write_json(source/'completed.json', {'fixture': True})
            write_json(source/'metrics.json', {'metrics': {'Precision@10': .2}})
            write_json(source/'coverage_metrics.json', {'metrics': {'MeanCoverage@10': .3}})
            reference = {'folder': str(source), 'key': 'old_registered_key',
                'ranking_sha256': file_hash(source/'rankings.csv'), 'completed_sha256': file_hash(source/'completed.json')}
            inputs = {'replay_references': {'valid': {'primary': reference}}, 'source_binding': {
                str(source/'rankings.csv'): reference['ranking_sha256'],
                str(source/'completed.json'): reference['completed_sha256']}}
            cache = output/'cache_completed.json'
            write_json(cache, {'fixture': True})
            receipt = {'status': 'verified', 'stage': 'valid', 'key': reference['key'],
                'historical_reference': True, 'register': False, 'source': str(source),
                'source_completed_sha256': reference['completed_sha256'], 'ranking_sha256': reference['ranking_sha256'],
                'score_cache_sha256': file_hash(cache), 'num_queries': 500, 'ranking_rows': 250000,
                'topk_identical': True, 'max_score_error': 0.}
            row = {**receipt, 'style': 'fusion', 'path': str(source), 'metrics': {'Precision@10': .2},
                   'coverage_metrics': {'MeanCoverage@10': .3}}
            path = output/'evaluation/baseline/baseline_replay_valid.json'
            write_json(path, receipt)
            register._replay(output, 'baseline', 'valid', 'valid', row, inputs, cache)
            receipt['max_score_error'] = row['max_score_error'] = 1e-15
            write_json(path, receipt)
            with self.assertRaisesRegex(ValueError, 'replay identity'):
                register._replay(output, 'baseline', 'valid', 'valid', row, inputs, cache)

    def catalog_fixture(self, temp):
        root = Path(temp)/'wbm-region-retrieval'
        output = root/'results/run'
        output.mkdir(parents=True)
        folder = root/'results_catalog'
        sources = {'schema_version': 1, 'runs': [{'key': 'old', 'path': 'old_path'}]}
        old = {'key': 'old', 'run_id': 'historical_id', 'id_time': {'basis': 'completion_marker_mtime'}}
        previous = {'source_spec_sha256': hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
                    'runs': [old]}
        write_json(folder/'sources.json', sources)
        write_json(folder/'snapshot.json', previous)
        (folder/'table.csv').write_text('1\n')
        proposed = {'schema_version': 1, 'runs': [{'key': 'new', 'path': 'new_path'}]}
        def render(snapshot):
            return {'table.csv': str(len(snapshot['runs']))+'\n'}
        def collect(workspace, merged, old_snapshot):
            return {'source_spec_sha256': hashlib.sha256(json.dumps(merged, sort_keys=True).encode()).hexdigest(),
                    'runs': [*old_snapshot['runs'], {'key': 'new', 'run_id': 'actual_start_id'}]}
        return root, output, folder, proposed, old, render, collect

    def test_catalog_addition_preserves_every_old_identity_and_repeated_apply_is_noop(self):
        with tempfile.TemporaryDirectory() as temp:
            root, output, folder, proposed, old, render, collect = self.catalog_fixture(temp)
            with patch.object(register, 'proposal', return_value=proposed), \
                    patch.object(register.catalog, 'render', side_effect=render), \
                    patch.object(register.catalog, 'collect', side_effect=collect):
                result = register.apply_proposal(output, root=root)
                self.assertEqual(result['new_results'], 1)
                self.assertEqual(read_json(folder/'snapshot.json')['runs'][0], old)
                self.assertEqual(read_json(folder/'sources.json')['runs'][0], {'key': 'old', 'path': 'old_path'})
                self.assertEqual(register.apply_proposal(output, root=root)['status'], 'already_registered')

    def test_catalog_refuses_collect_that_relabels_any_historical_identity_without_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root, output, folder, proposed, old, render, collect = self.catalog_fixture(temp)
            before = {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
            def bad_collect(workspace, merged, previous):
                result = collect(workspace, merged, previous)
                result['runs'][0] = {**old, 'run_id': 'wrong_new_time'}
                return result
            with patch.object(register, 'proposal', return_value=proposed), \
                    patch.object(register.catalog, 'render', side_effect=render), \
                    patch.object(register.catalog, 'collect', side_effect=bad_collect):
                with self.assertRaisesRegex(ValueError, 'earlier result'):
                    register.apply_proposal(output, root=root)
            self.assertTrue(all((folder/name).read_bytes() == content for name, content in before.items()))


if __name__ == '__main__':
    unittest.main()
