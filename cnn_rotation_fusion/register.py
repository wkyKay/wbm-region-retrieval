"""Register sealed rotation-weight evaluations without duplicating historical roles."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from b1_structure_fusion.register import _atomic_write, _json_bytes, _lock
from f_region_structure.register import _complete, _relative
from f_region_structure.state import read_signed
from f_region_wbbs.protocol import PROTOCOL
from results_catalog import build as catalog
from wbm_retrieval.common import file_hash, read_json

ROOT = Path(__file__).resolve().parents[1]
INDICES = ('evaluation_index_valid.json', 'controls_index_valid.json', 'evaluation_index_test.json')
STAGES = ('valid', 'cross_valid', 'test', 'cross_test')
KS = [1, 5, 10, 20, 50, 100, 500]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_run(output, *, check_source=True, require_root_seal=True):
    """Delegate scientific/source/selection/report validation to the isolated core."""
    from .run import verify_run as verify_core
    return verify_core(output, check_sources=check_source, require_root_seal=require_root_seal)


def _entry(output, root, row, index_name, context_signature):
    model, stage = row.get('model'), row.get('stage')
    _require(model in ('baseline', 'rotation') and stage in STAGES, 'Unknown model or evaluation stage')
    split = stage.removeprefix('cross_')
    _require(row.get('split') == split, 'Indexed stage/split mismatch')
    folder = Path(row['path']).resolve()
    _require(folder.is_relative_to(output) and folder != output, 'New evaluation must be inside this run')
    cross = stage.startswith('cross_')
    required = ['run_identity.json', 'evaluation_manifest.json', 'verification.json', 'rankings.csv',
                'coverage_metrics.json', 'coverage_per_query.csv', 'coverage_per_class.csv']
    if not cross:
        required += ['metrics.json', 'per_query.csv', 'per_class.csv']
    _complete(folder, row['signature'], required)
    _require(file_hash(folder/'completed.json') == row.get('completed_sha256'), 'Indexed completion hash changed')
    identity = read_json(folder/'run_identity.json')
    manifest = read_json(folder/'evaluation_manifest.json')
    _require(identity.get('key') == row.get('key') and identity.get('run_id') == row.get('run_id')
             and identity.get('evaluation_signature') == row.get('signature')
             and manifest.get('signature') == row.get('signature')
             and manifest.get('model') == model and manifest.get('stage') == stage
             and manifest.get('variant') == row.get('variant') and manifest.get('spec') == row.get('spec')
             and manifest.get('split') == split and manifest.get('ks') == KS
             and manifest.get('training_required') is False
             and manifest.get('test_parameter_search') is False
             and all(manifest.get(k) == v for k, v in identity.items()),
             'New evaluation identity, formula or no-training declaration changed')
    checked = read_json(folder/'verification.json')
    _require(checked.get('status') == 'verified' and checked.get('signature') == row['signature'],
             'Independent evaluation verification missing')
    if cross:
        _require(not any((folder/name).exists() for name in ('metrics.json', 'per_query.csv', 'per_class.csv')),
                 'Cross-combination evaluation must not contain exact reports')
    supporting = ['run_manifest.json', 'completed.json', 'frozen_plan.json', 'selection.json',
                  'source_protection_after.json', 'report/manifest.json', index_name,
                  f'inputs_{split}.json', str((folder/'verification.json').relative_to(output))]
    _require(all((output/name).is_file() for name in supporting), 'Missing registration supporting evidence')
    title = '旋转编码器' if model == 'rotation' else '原直角旋转编码器'
    item = {'key': identity['key'], 'repository': 'wbm-region-retrieval', 'path': _relative(folder, root),
            'category': 'cross_combination' if cross else 'fusion_ablation', 'method': 'F',
            'variant': 'CNNRotationFusion/'+model+'/'+row['variant'], 'split': split,
            'manifest': _relative(folder/'evaluation_manifest.json', root),
            'marker': _relative(folder/'completed.json', root), 'identity_schema_version': 1,
            'identity': _relative(folder/'run_identity.json', root),
            'selection': _relative(output/'selection.json', root), 'coverage': None,
            'supervision': 'self_supervised',
            'display': title+'区域加法权重 '+row['variant']+('（跨组合）' if cross else ''),
            'required_families': ['coverage'] if cross else ['exact', 'coverage'],
            'protocol': PROTOCOL if cross else 'retrieval_v1', 'posthoc': True,
            'supporting_manifests': [_relative(output/name, root) for name in supporting],
            'design_note': 'Twelve prespecified regional additive weight combinations with frozen seed2026 '
                'epoch100 arbitrary-angle rotation encoder and unchanged original right-angle encoder controls. '
                'No training or feature re-extraction. Only rotation valid exact P@10 chooses one frozen '
                'weight combination; theta0 alpha=.4 beta=.25 remains the original-weight control. '
                'Both encoders are evaluated at unique(theta0,winner) on test and cross-combination pools; '
                'test never selects weights or checkpoints. Regional C/T/A fusion precedes unchanged greedy '
                'matching and query sqrt(area) aggregation. WBBS and existing identical model/formula/split '
                'roles are historical references, not duplicate registrations. This is a posthoc single-seed '
                'extension of prior test observations, not an independent confirmation of model selection.'}
    catalog.execution_identity(root, item, folder/'completed.json', manifest)
    return item


def proposal(output, *, root=ROOT):
    """Validate first, then emit each new canonical identity exactly once."""
    output, root = Path(output).resolve(), Path(root).resolve()
    _relative(output, root/'results')
    _require(output != root/'results', 'An isolated time-identified output root is required')
    verify_run(output, check_source=True, require_root_seal=True)
    manifest = read_signed(output/'run_manifest.json')
    context = manifest['signature']
    _complete(output, context, ('run_manifest.json', 'frozen_plan.json', 'selection.json',
        'source_protection_after.json', 'report/manifest.json', 'report/completed.json', *INDICES))
    seen, items = {}, []
    for index_name in INDICES:
        index = read_json(output/index_name)
        _require(index.get('context_signature') == context and isinstance(index.get('signature'), str)
                 and bool(index['signature']) and isinstance(index.get('evaluations'), dict),
                 'Evaluation index detached from the frozen run')
        for row in index['evaluations'].values():
            _require(type(row.get('register')) is bool and type(row.get('historical_reference')) is bool
                     and row['register'] != row['historical_reference'], 'Ambiguous registration/reference role')
            key = row.get('key')
            _require(isinstance(key, str) and bool(key), 'Missing canonical evaluation key')
            if key in seen:
                _require(seen[key] == row, 'Conflicting repeated canonical identity: '+key)
                continue
            seen[key] = row
            if row['register']:
                items.append(_entry(output, root, row, index_name, context))
    _require(items and len({x['key'] for x in items}) == len(items), 'No unique new completed evaluations')
    return {'schema_version': 1, 'runs': items}


def write_proposal(output, *, root=ROOT):
    value = proposal(output, root=root)
    path = Path(output)/'catalog_sources.json'
    if path.exists():
        _require(read_json(path) == value, 'Existing registration proposal changed')
    else:
        _atomic_write(path, _json_bytes(value))
    return path


def apply_proposal(output, *, workspace=None, root=ROOT):
    """Collect before mutation, preserve old objects, and roll back failed writes."""
    output, root = Path(output).resolve(), Path(root).resolve()
    workspace = root.parent if workspace is None else Path(workspace).resolve()
    _require((workspace/'wbm-region-retrieval').resolve() == root, 'Workspace/repository mismatch')
    proposed = read_json(write_proposal(output, root=root))
    folder = root/'results_catalog'
    with _lock(folder):
        original = {name: (folder/name).read_bytes() for name in ('sources.json', 'snapshot.json')}
        sources, previous = json.loads(original['sources.json']), json.loads(original['snapshot.json'])
        fingerprint = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
        _require(previous.get('source_spec_sha256') == fingerprint, 'Existing sources/snapshot fingerprints differ')
        for name, content in catalog.render(previous).items():
            _require((folder/name).is_file() and (folder/name).read_text() == content, 'Existing catalogue is stale: '+name)
            original[name] = (folder/name).read_bytes()
        entries = {item['key']: item for item in sources['runs']}
        old_runs = {item['key']: item for item in previous['runs']}
        _require(len(entries) == len(sources['runs']) and len(old_runs) == len(previous['runs'])
                 and set(entries) == set(old_runs), 'Existing catalogue identities are inconsistent')
        additions = []
        for item in proposed['runs']:
            if item['key'] in entries:
                _require(entries[item['key']] == item, 'Conflicting catalogue identity: '+item['key'])
            else:
                additions.append(item)
        if not additions:
            return {'status': 'already_registered', 'new_results': 0}
        merged = {**sources, 'runs': [*sources['runs'], *additions]}
        updated = catalog.collect(workspace, merged, previous)
        new_runs = {item['key']: item for item in updated['runs']}
        _require(len(new_runs) == len(updated['runs'])
                 and set(new_runs) == set(old_runs)|{item['key'] for item in additions}
                 and all(new_runs.get(k) == v for k, v in old_runs.items()),
                 'Collection modified an earlier result or introduced unexpected identities')
        contents = {'snapshot.json': _json_bytes(updated), 'sources.json': _json_bytes(merged),
                    **{name: text.encode() for name, text in catalog.render(updated).items()}}
        _require(all((folder/name).read_bytes() == data for name, data in original.items()), 'Catalogue changed concurrently')
        before = {name: original.get(name) for name in contents}
        _require(all(data is not None or not (folder/name).exists() for name, data in before.items()),
                 'A generated output would overwrite untracked data')
        backup_parent = output/'catalog_backup'
        backup_parent.mkdir(exist_ok=True)
        backup = Path(tempfile.mkdtemp(prefix='transaction_', dir=backup_parent))
        for name, data in before.items():
            if data is not None:
                target = backup/'files'/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        audit = {'status': 'prepared', 'new_keys': [item['key'] for item in additions],
                 'before_sha256': {name: hashlib.sha256(data).hexdigest() if data is not None else None for name, data in before.items()},
                 'after_sha256': {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}}
        (backup/'manifest.json').write_bytes(_json_bytes(audit))
        written = []
        try:
            for name, content in contents.items():
                _require(before[name] == ((folder/name).read_bytes() if (folder/name).exists() else None),
                         'Concurrent catalogue write: '+name)
                written.append(name)
                _atomic_write(folder/name, content)
            _require(all((folder/name).read_bytes() == data for name, data in contents.items()), 'Catalogue write verification failed')
        except BaseException as error:
            failures = []
            for name in reversed(written):
                try:
                    if before[name] is None:
                        (folder/name).unlink(missing_ok=True)
                    else:
                        _atomic_write(folder/name, before[name])
                except BaseException as restore_error:
                    failures.append(str(restore_error))
            audit.update(status='rollback_failed' if failures else 'rolled_back', error=str(error), rollback_errors=failures)
            (backup/'manifest.json').write_bytes(_json_bytes(audit))
            if failures:
                raise RuntimeError('Catalogue rollback failed; backup: '+str(backup)) from error
            raise
        audit['status'] = 'committed'
        (backup/'manifest.json').write_bytes(_json_bytes(audit))
        return {'status': 'registered', 'new_results': len(additions), 'backup': str(backup)}


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--workspace', type=Path, default=ROOT.parent)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    if args.apply:
        result = apply_proposal(args.run_root, workspace=args.workspace)
    else:
        result = {'proposal': str(write_proposal(args.run_root))}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
