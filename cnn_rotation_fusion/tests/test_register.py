import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cnn_rotation_fusion import register
from f_region_structure.state import seal, signed
from wbm_retrieval.common import digest, file_hash, read_json, write_json


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class RegistrationTests(unittest.TestCase):
    def evaluation(self, output, name, stage='valid', model='rotation'):
        split = stage.removeprefix('cross_')
        folder = output/'evaluations'/model/stage/name
        signature = digest({'model': model, 'stage': stage, 'variant': name})
        key = 'cnn_rotation_fusion_fixture_'+model+'_'+stage+'_'+name
        identity = {'schema_version': 1, 'key': key, 'split': split, 'stage': stage,
                    'variant': name, 'started_at': '2026-11-24T08:00:01+08:00',
                    'run_id': '20261124T080001+0800__'+key, 'evaluation_signature': signature}
        spec = {'family': 'add', 'alpha': .4 if name == 'theta0' else .2, 'beta': .25}
        manifest = {**identity, 'signature': signature, 'model': model, 'spec': spec,
                    'ks': register.KS, 'training_required': False, 'test_parameter_search': False}
        write_json(folder/'run_identity.json', identity)
        write_json(folder/'evaluation_manifest.json', manifest)
        write_json(folder/'verification.json', {'status': 'verified', 'signature': signature})
        for file in ('rankings.csv', 'coverage_per_query.csv', 'coverage_per_class.csv'):
            (folder/file).write_text('fixture\n')
        write_json(folder/'coverage_metrics.json', {'fixture': True})
        if not stage.startswith('cross_'):
            write_json(folder/'metrics.json', {'fixture': True})
            (folder/'per_query.csv').write_text('fixture\n')
            (folder/'per_class.csv').write_text('fixture\n')
        seal(folder, signature)
        return {'model': model, 'stage': stage, 'split': split, 'variant': name, 'spec': spec,
                'path': str(folder), 'key': key, 'run_id': identity['run_id'], 'signature': signature,
                'register': True, 'historical_reference': False,
                'completed_sha256': file_hash(folder/'completed.json')}

    def fixture(self, temp, *, extra=False):
        root = Path(temp).resolve()/'wbm-region-retrieval'
        output = root/'results/20261124T080000+0800__rotation_weights'
        output.mkdir(parents=True)
        manifest = signed({'version': 'fixture', 'started_at': '2026-11-24T08:00:00+08:00'})
        write_json(output/'run_manifest.json', manifest)
        for name in ('frozen_plan.json', 'selection.json', 'source_protection_after.json',
                     'inputs_valid.json', 'inputs_test.json', 'report/manifest.json', 'report/completed.json'):
            write_json(output/name, {'fixture': name})
        ordinary = self.evaluation(output, 'winner')
        cross = self.evaluation(output, 'winner', 'cross_valid', 'baseline')
        test = self.evaluation(output, 'winner', 'test')
        historical = {'key': 'old_theta0', 'register': False, 'historical_reference': True,
                      'path': '/historical/read_only', 'model': 'baseline', 'variant': 'theta0'}
        grid = ({'a': ordinary, 'theta0': historical}, {'a': cross}, {'a': test, 'alias': test})
        if extra:
            grid[2]['extra'] = self.evaluation(output, 'winner', 'cross_test')
        for name, rows in zip(register.INDICES, grid):
            write_json(output/name, {'context_signature': manifest['signature'], 'signature': digest(rows),
                                    'evaluations': rows})
        seal(output, manifest['signature'])
        return root, output, ordinary

    def test_dynamic_unique_new_roles_do_not_register_historical_theta0(self):
        for extra in (False, True):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as temp:
                root, output, _ = self.fixture(temp, extra=extra)
                with patch.object(register, 'verify_run') as verify:
                    proposed = register.proposal(output, root=root)
                verify.assert_called_once_with(output, check_source=True, require_root_seal=True)
                self.assertEqual(len(proposed['runs']), 3+int(extra))
                self.assertEqual(len({r['key'] for r in proposed['runs']}), len(proposed['runs']))
                self.assertNotIn('old_theta0', {r['key'] for r in proposed['runs']})
                self.assertTrue(all(r['selection'].endswith('/selection.json') for r in proposed['runs']))
                self.assertEqual({r['category'] for r in proposed['runs']}, {'fusion_ablation', 'cross_combination'})
                self.assertTrue(all(r['supervision'] == 'self_supervised' and r['posthoc'] is True for r in proposed['runs']))

    def test_core_selection_or_source_failure_prevents_proposal_write(self):
        with tempfile.TemporaryDirectory() as temp:
            root, output, _ = self.fixture(temp)
            with patch.object(register, 'verify_run', side_effect=ValueError('selection/source invalid')):
                with self.assertRaisesRegex(ValueError, 'selection/source'):
                    register.write_proposal(output, root=root)
            self.assertFalse((output/'catalog_sources.json').exists())

    def test_missing_root_seal_and_changed_finalized_artifact_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root, output, row = self.fixture(temp)
            with patch.object(register, 'verify_run'):
                (Path(row['path'])/'rankings.csv').write_text('tampered\n')
                with self.assertRaisesRegex(ValueError, 'Changed or missing'):
                    register.proposal(output, root=root)
                (output/'completed.json').unlink()
                with self.assertRaisesRegex(ValueError, 'completed sealed'):
                    register.proposal(output, root=root)

    def test_conflicting_alias_and_missing_no_training_declaration_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root, output, _ = self.fixture(temp)
            index = read_json(output/register.INDICES[2])
            index['evaluations']['alias'] = {**index['evaluations']['alias'], 'variant': 'another_weight'}
            write_json(output/register.INDICES[2], index)
            (output/'completed.json').unlink()
            seal(output, read_json(output/'run_manifest.json')['signature'])
            with patch.object(register, 'verify_run'):
                with self.assertRaisesRegex(ValueError, 'Conflicting repeated'):
                    register.proposal(output, root=root)
        with tempfile.TemporaryDirectory() as temp:
            root, output, row = self.fixture(temp)
            folder = Path(row['path']);manifest = read_json(folder/'evaluation_manifest.json')
            manifest['training_required'] = True
            write_json(folder/'evaluation_manifest.json', manifest)
            (folder/'completed.json').unlink();seal(folder, row['signature'])
            index = read_json(output/register.INDICES[0]);index['evaluations']['a']['completed_sha256'] = file_hash(folder/'completed.json')
            write_json(output/register.INDICES[0], index)
            (output/'completed.json').unlink();seal(output, read_json(output/'run_manifest.json')['signature'])
            with patch.object(register, 'verify_run'):
                with self.assertRaisesRegex(ValueError, 'no-training declaration'):
                    register.proposal(output, root=root)

    def catalog_fixture(self, temp):
        root = Path(temp).resolve()/'wbm-region-retrieval';output = root/'results/run';output.mkdir(parents=True)
        folder = root/'results_catalog'
        sources = {'schema_version': 1, 'runs': [{'key': f'old_{i}', 'path': f'old_path_{i}'} for i in range(551)]}
        old = [{'key': r['key'], 'run_id': 'historical_'+r['key'],
                'id_time': {'basis': 'completion_marker_mtime'}, 'nested': {'metrics': [.2, None, .3]}}
               for r in sources['runs']]
        previous = {'source_spec_sha256': fingerprint(sources), 'runs': old}
        write_json(folder/'sources.json', sources);write_json(folder/'snapshot.json', previous)
        (folder/'table.csv').write_text('551\n')
        proposed = {'schema_version': 1, 'runs': [{'key': 'new', 'path': 'new_path'}]}
        def render(snapshot):
            return {'table.csv': str(len(snapshot['runs']))+'\n'}
        def collect(workspace, merged, old_snapshot):
            return {'source_spec_sha256': fingerprint(merged),
                    'runs': [*copy.deepcopy(old_snapshot['runs']), {'key': 'new', 'run_id': 'actual_start_id'}]}
        return root, output, folder, proposed, sources, old, render, collect

    def test_transaction_preserves_551_objects_and_repeat_is_noop(self):
        with tempfile.TemporaryDirectory() as temp:
            root, output, folder, proposed, sources, old, render, collect = self.catalog_fixture(temp)
            with patch.object(register, 'proposal', return_value=proposed), \
                    patch.object(register.catalog, 'render', side_effect=render), \
                    patch.object(register.catalog, 'collect', side_effect=collect):
                result = register.apply_proposal(output, root=root)
                self.assertEqual(result['new_results'], 1)
                self.assertEqual(read_json(folder/'snapshot.json')['runs'][:551], old)
                self.assertEqual(read_json(folder/'sources.json')['runs'][:551], sources['runs'])
                self.assertEqual(read_json(Path(result['backup'])/'manifest.json')['status'], 'committed')
                self.assertEqual(register.apply_proposal(output, root=root)['status'], 'already_registered')

    def test_changed_old_result_or_stale_generated_catalogue_refused_before_writes(self):
        for stale in (False, True):
            with self.subTest(stale=stale), tempfile.TemporaryDirectory() as temp:
                root, output, folder, proposed, _, _, render, collect = self.catalog_fixture(temp)
                if stale:
                    (folder/'table.csv').write_text('stale\n')
                before = {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
                def bad_collect(workspace, merged, previous):
                    result = collect(workspace, merged, previous)
                    result['runs'][200]['nested']['metrics'][0] = .9
                    return result
                with patch.object(register, 'proposal', return_value=proposed), \
                        patch.object(register.catalog, 'render', side_effect=render), \
                        patch.object(register.catalog, 'collect', side_effect=bad_collect):
                    with self.assertRaisesRegex(ValueError, 'stale|earlier result'):
                        register.apply_proposal(output, root=root)
                self.assertTrue(all((folder/name).read_bytes() == value for name, value in before.items()))
                self.assertFalse((folder/'.registration.lock').exists())

    def test_partial_write_failure_restores_original_bytes_and_records_rollback(self):
        with tempfile.TemporaryDirectory() as temp:
            root, output, folder, proposed, _, _, render, collect = self.catalog_fixture(temp)
            before = {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
            real_write = register._atomic_write
            failed = False
            def interrupted(path, data):
                nonlocal failed
                if Path(path) == folder/'sources.json' and not failed:
                    failed = True
                    raise OSError('simulated disk failure')
                return real_write(path, data)
            with patch.object(register, 'proposal', return_value=proposed), \
                    patch.object(register.catalog, 'render', side_effect=render), \
                    patch.object(register.catalog, 'collect', side_effect=collect), \
                    patch.object(register, '_atomic_write', side_effect=interrupted):
                with self.assertRaisesRegex(OSError, 'simulated disk'):
                    register.apply_proposal(output, root=root)
            self.assertTrue(all((folder/name).read_bytes() == value for name, value in before.items()))
            backups = list((output/'catalog_backup').glob('transaction_*/manifest.json'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(read_json(backups[0])['status'], 'rolled_back')
            self.assertFalse((folder/'.registration.lock').exists())


if __name__ == '__main__':
    unittest.main()
