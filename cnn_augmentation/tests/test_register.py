import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from cnn_augmentation import evaluation, register, state
from cnn_augmentation.augmentation import AUGMENTATION_SPECS
from cnn_augmentation.training import AUDIT_COUNTERS, audit_history
from f_region_structure.state import seal
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from wbm_retrieval.training import parameter_hash


class RegistrationTests(unittest.TestCase):
    def root_fixture(self, output):
        stamp = '2026-11-23T12:00:00+08:00'
        write_json(output/'paper_source.json',read_json(register.ROOT/'cnn_augmentation/paper_source.json'))
        plan = state.signed({'plan': state.PLAN, 'frozen_at': stamp})
        write_json(output/'frozen_plan.json', plan)
        manifest = state.signed({'version': state.VERSION, 'started_at': stamp,
            'run_id': '20261123T120000+0800__cnn_augmentation', 'plan_signature': plan['signature'],
            'training_labels': False, 'test_parameter_search': False, 'authorized_models': list(state.VARIANTS),
            'paper_source_sha256':file_hash(output/'paper_source.json'),
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
                        replay = model == 'baseline'
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

    def test_forty_new_identities_leave_all_eight_baselines_unregistered(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);output=root/'results/run';self.evaluation_grid(output)
            with patch.object(register,'verify_run'):
                result=register.proposal(output,root=root)
            self.assertEqual(len(result['runs']),40)
            self.assertEqual(len({row['key'] for row in result['runs']}),40)
            self.assertTrue(all('/baseline/' not in row['variant'] for row in result['runs']))
            self.assertTrue(all(row['identity_schema_version']==1 and row['selection'] is None for row in result['runs']))

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
            variant = 'crop'
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
                'augmentation_spec': AUGMENTATION_SPECS[variant], 'code': {'cnn_augmentation/training.py': file_hash(register.ROOT/'cnn_augmentation/training.py')}}
            signature = digest(science)
            run_id = '20261123T120001+0800__cnn_aug_crop_seed2026_epoch100'
            manifest = state.signed({'science': science, 'training_signature': signature,
                'run_id': run_id, 'started_at': '2026-11-23T12:00:01+08:00'})
            write_json(folder/'run_manifest.json', manifest)
            history = [{'epoch': i, 'optimizer_steps': i*390, 'train_loss': .1, 'valid_nce': .2} for i in range(1, 101)]
            for row in history:
                row.update({key:0 for key in AUDIT_COUNTERS});row.update(sample_count=99786,empty_target_before_count=99786,empty_target_after_count=99786)
            audit=audit_history(history,variant)
            write_json(folder/'augmentation_audit.json',audit)
            write_json(folder/'history.json', history)
            model = torch.nn.Linear(2, 2).state_dict()
            summary = {'completed': True, 'epochs_completed': 100, 'max_epochs': 100, 'optimizer_steps': 39000,
                'initial_parameter_sha256': register.INITIAL_HASH, 'checkpoint_selection': 'fixed_epoch_100',
                'run_id': run_id, 'started_at': manifest['started_at'], 'final_parameter_sha256': parameter_hash(model), 'augmentation_audit':audit}
            write_json(folder/'training_summary.json', summary)
            checkpoint = {'signature': signature, 'epoch': 100, 'variant': variant, 'optimizer_steps': 39000,
                'initial_parameter_sha256': register.INITIAL_HASH, 'input_spec': science['input'],
                'selection': 'fixed_epoch_100', 'model': model}
            torch.save(checkpoint, folder/'checkpoint.pt')
            # Completion required-file validation is separately covered by _complete;
            # this fixture isolates checkpoint-budget and model-byte checks.
            with patch.object(register, '_complete'):
                self.assertEqual(register._training(output, variant, root), folder/'checkpoint.pt')
                checkpoint['epoch'] = 99
                torch.save(checkpoint, folder/'checkpoint.pt')
                with self.assertRaisesRegex(ValueError, 'epoch100'):
                    register._training(output, variant, root)
                checkpoint['epoch'] = 100
                checkpoint['selection'] = 'best_nce'
                torch.save(checkpoint, folder/'checkpoint.pt')
                with self.assertRaisesRegex(ValueError, 'epoch100'):
                    register._training(output, variant, root)

    def test_aug_audit_rejects_tampered_retention_rates(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            history=[]
            for epoch in range(1,101):
                row={key:0 for key in AUDIT_COUNTERS}
                row.update(epoch=epoch,sample_count=99786,empty_target_before_count=99786,empty_target_after_count=99786)
                history.append(row)
            audit=audit_history(history,'crop')
            write_json(p/'history.json',history);write_json(p/'augmentation_audit.json',audit)
            write_json(p/'training_summary.json',{'augmentation_audit':audit})
            self.assertEqual(register._augmentation_audit(p,'crop'),audit)
            audit['rates']['empty_target_after_fraction']=.5
            write_json(p/'augmentation_audit.json',audit)
            write_json(p/'training_summary.json',{'augmentation_audit':audit})
            with self.assertRaisesRegex(ValueError,'rates'):
                register._augmentation_audit(p,'crop')

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
