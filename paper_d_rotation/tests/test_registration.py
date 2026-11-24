"""Administrative support attachment preserves scientific catalogue records."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from paper_d_rotation import register


def fixture():
    rows = []
    for stage, key in register.TARGETS.items():
        folder = register.PRIMARY_ROOT+'/evaluation/rotation/'+stage+'/add_a0p4_b0p15'
        rows.append({'key': key, 'repository': 'wbm-region-retrieval', 'path': folder,
            'variant': 'CNNRotationFusion/rotation/add_a0p4_b0p15', 'split': 'test',
            'manifest': folder+'/evaluation_manifest.json', 'marker': folder+'/completed.json',
            'identity': folder+'/run_identity.json', 'identity_schema_version': 1,
            'protocol': 'coverage_cross_combination_v1' if stage == 'cross_test' else 'retrieval_v1',
            'required_families': ['coverage'] if stage == 'cross_test' else ['exact', 'coverage'],
            'supporting_manifests': ['historical/'+stage+'.json'], 'selection': 'fixed/selection.json'})
    rows.append({'key': 'unrelated', 'path': 'original/result', 'supporting_manifests': ['old/evidence.json']})
    sources = {'schema_version': 1, 'runs': rows}
    root = 'result1124/20261124T170000+0800__paper_d_rotation'
    additions = [root+'/'+name for name in register.SUPPORT_FILES]
    proposal = {'schema_version': 1, 'operation': 'append_supporting_manifests',
        'analysis_root': root, 'target_keys': list(register.TARGETS.values()),
        'new_formal_model_evaluations': 0, 'supporting_manifests': additions,
        'file_sha256': {name: hashlib.sha256(name.encode()).hexdigest() for name in additions}}
    runs = []
    for i, row in enumerate(rows):
        runs.append({**deepcopy(row), 'run_id': 'original_'+str(i), 'id_time': {'value': 'original'},
            'completion': {'sha256': 'immutable'}, 'reports': {'exact': {'data': {'metrics': {'P@10': .6}}}},
            'provenance': {'identity_data': {'run_id': 'original_'+str(i)},
                'supporting_manifests': [{'source': {'path': name, 'sha256': 'old'}, 'data': {'original': True}}
                                        for name in row['supporting_manifests']]}})
    previous = {'runs': runs, 'source_spec_sha256': hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()}
    return sources, previous, proposal


def collected(previous, merged, proposal):
    updated = deepcopy(previous)
    for row, spec in zip(updated['runs'], merged['runs']):
        old = row['supporting_manifests']
        extra = [name for name in spec['supporting_manifests'] if name not in old]
        row['supporting_manifests'] = list(spec['supporting_manifests'])
        row['provenance']['supporting_manifests'] += [
            {'source': {'path': name, 'sha256': proposal['file_sha256'][name]}, 'data': {'D': True}}
            for name in extra]
    updated['source_spec_sha256'] = hashlib.sha256(json.dumps(merged, sort_keys=True).encode()).hexdigest()
    return updated


class SupportMergeTests(unittest.TestCase):
    def test_append_only_exact_targets_and_idempotent(self):
        sources, previous, proposal = fixture()
        original = deepcopy(sources)
        merged, changed = register.merge_sources(sources, proposal)
        self.assertEqual(sources, original)
        self.assertEqual(changed, list(register.TARGETS.values()))
        self.assertEqual(merged['runs'][2], sources['runs'][2])
        for old, new in zip(sources['runs'][:2], merged['runs'][:2]):
            self.assertEqual(new['supporting_manifests'], old['supporting_manifests']+proposal['supporting_manifests'])
        twice, changed = register.merge_sources(merged, proposal)
        self.assertEqual(twice, merged)
        self.assertEqual(changed, [])
        register.validate_snapshot(previous, collected(previous, merged, proposal), merged, proposal)

    def test_missing_extra_or_wrong_primary_identity_rejected(self):
        for mode in ('extra_target', 'missing_target', 'wrong_model', 'duplicate_key'):
            sources, _, proposal = fixture()
            if mode == 'extra_target': proposal['target_keys'].append('unrelated')
            elif mode == 'missing_target': sources['runs'].pop(0)
            elif mode == 'wrong_model': sources['runs'][0]['path'] = 'old_cnn/result'
            else: sources['runs'].append(deepcopy(sources['runs'][0]))
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                register.merge_sources(sources, proposal)

    def test_existing_support_is_not_replaced_or_duplicated(self):
        sources, _, proposal = fixture()
        sources['runs'][0]['supporting_manifests'].append(proposal['supporting_manifests'][0])
        merged, _ = register.merge_sources(sources, proposal)
        self.assertEqual(merged['runs'][0]['supporting_manifests'][:2], sources['runs'][0]['supporting_manifests'])
        self.assertEqual(len(merged['runs'][0]['supporting_manifests']), 8)
        sources['runs'][0]['supporting_manifests'].append(proposal['supporting_manifests'][0])
        with self.assertRaisesRegex(ValueError, 'duplicates'):
            register.merge_sources(sources, proposal)

    def test_any_scientific_or_old_support_mutation_rejected(self):
        for mode in ('report', 'identity', 'unrelated', 'old_support', 'new_support_hash', 'new_result'):
            sources, previous, proposal = fixture()
            merged, _ = register.merge_sources(sources, proposal)
            updated = collected(previous, merged, proposal)
            if mode == 'report': updated['runs'][0]['reports']['exact']['data']['metrics']['P@10'] = .9
            elif mode == 'identity': updated['runs'][0]['run_id'] = 'new_identity'
            elif mode == 'unrelated': updated['runs'][2]['selection'] = 'new'
            elif mode == 'old_support': updated['runs'][0]['provenance']['supporting_manifests'][0]['data']['original'] = False
            elif mode == 'new_support_hash': updated['runs'][0]['provenance']['supporting_manifests'][-1]['source']['sha256'] = 'wrong'
            else: updated['runs'].append({'key': 'new'})
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                register.validate_snapshot(previous, updated, merged, proposal)


class SupportTransactionTests(unittest.TestCase):
    def setup_catalog(self, directory):
        root = Path(directory)/'wbm-region-retrieval'
        folder = root/'results_catalog'; folder.mkdir(parents=True)
        sources, previous, proposal = fixture()
        (folder/'sources.json').write_bytes(register._json_bytes(sources))
        (folder/'snapshot.json').write_bytes(register._json_bytes(previous))
        (folder/'report.txt').write_text('stable reports\n')
        output = root/proposal['analysis_root']; output.mkdir(parents=True)
        path = output.parent/(output.name+'__catalog_support.json')
        path.write_bytes(register._json_bytes(proposal))
        return root, folder, output, path, proposal

    def patches(self, path, proposal):
        return (mock.patch.object(register, 'write_proposal', return_value=path),
                mock.patch.object(register.catalog, 'render', return_value={'report.txt': 'stable reports\n'}),
                mock.patch.object(register.catalog, 'collect', side_effect=lambda workspace, merged, previous:
                                  collected(previous, merged, proposal)))

    def test_transaction_attaches_then_becomes_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root, folder, output, path, proposal = self.setup_catalog(directory)
            a, b, c = self.patches(path, proposal)
            with a, b, c:
                result = register.apply_proposal(output, root=root)
                self.assertEqual(result['new_results'], 0)
                self.assertEqual(result['status'], 'attached')
                self.assertEqual(len(result['updated_keys']), 2)
                self.assertEqual(register.apply_proposal(output, root=root)['status'], 'already_attached')
            self.assertFalse(any(output.iterdir()))  # No mutation of sealed D files.

    def test_stale_sources_snapshot_rejected_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root, folder, output, path, proposal = self.setup_catalog(directory)
            snapshot = json.loads((folder/'snapshot.json').read_text())
            snapshot['source_spec_sha256'] = 'wrong'
            (folder/'snapshot.json').write_bytes(register._json_bytes(snapshot))
            before = {p.name: p.read_bytes() for p in folder.iterdir()}
            with mock.patch.object(register, 'write_proposal', return_value=path), self.assertRaisesRegex(ValueError, 'fingerprints'):
                register.apply_proposal(output, root=root)
            self.assertEqual(before, {p.name: p.read_bytes() for p in folder.iterdir()})

    def test_write_failure_rolls_back_original_catalogue(self):
        with tempfile.TemporaryDirectory() as directory:
            root, folder, output, path, proposal = self.setup_catalog(directory)
            before = {p.name: p.read_bytes() for p in folder.iterdir()}
            atomic = register._atomic_write
            failed = []
            def fail_once(target, value):
                if Path(target).name == 'snapshot.json' and not failed:
                    failed.append(True)
                    raise OSError('simulated disk failure')
                return atomic(target, value)
            a, b, c = self.patches(path, proposal)
            with a, b, c, mock.patch.object(register, '_atomic_write', side_effect=fail_once), self.assertRaisesRegex(OSError, 'simulated'):
                register.apply_proposal(output, root=root)
            self.assertEqual(before, {p.name: p.read_bytes() for p in folder.iterdir()})
            receipt = next(output.parent.glob(output.name+'__catalog_support_backup/transaction_*/manifest.json'))
            self.assertEqual(json.loads(receipt.read_text())['status'], 'rolled_back')


if __name__ == '__main__':
    unittest.main()
