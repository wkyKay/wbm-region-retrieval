"""Attach sealed rotation explanations to two existing primary catalogue entries.

This administrative module is deliberately outside the D scientific signature.
It adds supporting provenance only; result identities and metric reports remain
byte-for-byte equivalent as parsed objects. Administrative receipts stay outside
the sealed D directory.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile

from b1_structure_fusion.register import _atomic_write, _json_bytes, _lock
from f_region_structure.state import read_signed, verified
from results_catalog import build as catalog
from wbm_retrieval.common import file_hash, read_json

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_ROOT = 'results/20261124T093446+0800__cnn_rotation_fusion'
TARGETS = {stage: 'cnn_rotation_fusion_20261124T093446_0800__cnn_rotation_fusion_rotation_'
           + stage + '_add_a0p4_b0p15' for stage in ('test', 'cross_test')}
SUPPORT_FILES = ('run_manifest.json', 'frozen_plan.json', 'completed.json', 'verification.json',
                 'summary.json', 'figures/figures_manifest.json', 'figure_review.json')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _relative(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    _require(path != root and path.is_relative_to(root), 'Path must stay inside repository')
    return str(path.relative_to(root))


def verify_run(output, *, root=ROOT):
    """Require the sealed D analysis and exact current-primary C replay chain."""
    from .run import ensure_code
    from .verify import verify_output
    output, root = Path(output).resolve(), Path(root).resolve()
    _require(output.parent == root/'result1124', 'D evidence must be an isolated result1124 run')
    manifest = read_signed(output/'run_manifest.json')
    _require(manifest.get('version') == 'paper_d_rotation_v1'
             and verified(output, manifest['signature']), 'D run is not completed and hash verified')
    ensure_code(manifest)
    evidence = verify_output(output)
    stored = read_json(output/'verification.json')
    _require(all(stored.get(k) == v for k, v in evidence.items()), 'Stored D acceptance differs from independent verification')
    _require(stored.get('source_protection') == 'unchanged', 'Source protection did not pass')
    context = manifest['context']
    binding = context['source_binding']
    for name, expected in binding.items():
        _require(Path(name).is_file() and file_hash(name) == expected, 'Frozen source changed: '+name)
    source = Path(context['source_root']).resolve()
    _require(source.parent == root/'result1124', 'Source must be the current isolated rotation C run')
    c_manifest = read_signed(source/'run_manifest.json')
    _require(c_manifest.get('version') == 'paper_c_rotation_v1'
             and binding.get(str(source/'run_manifest.json')) == file_hash(source/'run_manifest.json')
             and binding.get(str(source/'completed.json')) == file_hash(source/'completed.json'),
             'D evidence lacks the sealed rotation C source binding')
    inputs = read_json(source/'inputs_test.json')
    _require(binding.get(str(source/'inputs_test.json')) == file_hash(source/'inputs_test.json'),
             'Rotation C test inputs are not hash-bound')
    example_sources = read_json(output/'explanations/examples.json')['sources']
    for stage, key in TARGETS.items():
        reference = inputs['replay_references'][stage]['primary']
        expected_folder = root/PRIMARY_ROOT/'evaluation/rotation'/stage/'add_a0p4_b0p15'
        _require(reference.get('key') == key and Path(reference['folder']).resolve() == expected_folder,
                 'C replay does not refer to the exact current primary role: '+stage)
        control = read_json(source/('replay_'+stage+'.json'))['controls']['primary']
        _require(control.get('source_key') == key and control.get('top500_identical') is True
                 and control.get('full_score_matrix_identical') is True,
                 'Current primary replay proof is missing: '+stage)
        ranking = source/stage/'primary/rankings.csv'
        _require(example_sources[stage].get('path') == str(ranking)
                 and example_sources[stage].get('sha256') == binding.get(str(ranking))
                 == file_hash(ranking), 'D explanation ranking is detached from current C primary')
    baseline = source/'score_cache/test/primary'
    d4 = read_json(output/'d4/verification.json')
    _require(d4.get('baseline_source') == str(baseline)
             and d4.get('baseline_completion_sha256') == binding.get(str(baseline/'completed.json'))
             == file_hash(baseline/'completed.json')
             and d4.get('baseline_full_scores_byte_exact') is True,
             'D intervention baseline is detached from current C primary')
    return manifest


def proposal(output, *, root=ROOT):
    output, root = Path(output).resolve(), Path(root).resolve()
    manifest = verify_run(output, root=root)
    paths = [_relative(output/name, root) for name in SUPPORT_FILES]
    return {'schema_version': 1, 'operation': 'append_supporting_manifests',
            'analysis_run_id': manifest['run_id'], 'analysis_root': _relative(output, root),
            'analysis_completed_sha256': file_hash(output/'completed.json'),
            'target_keys': list(TARGETS.values()), 'new_formal_model_evaluations': 0,
            'supporting_manifests': paths,
            'file_sha256': {name: file_hash(root/name) for name in paths}}


def write_proposal(output, *, root=ROOT):
    output = Path(output).resolve()
    value = proposal(output, root=root)
    path = output.parent/(output.name+'__catalog_support.json')
    if path.exists():
        _require(read_json(path) == value, 'Existing supporting-evidence proposal changed')
    else:
        _atomic_write(path, _json_bytes(value))
    return path


def _target_entry(entry, stage):
    folder = PRIMARY_ROOT+'/evaluation/rotation/'+stage+'/add_a0p4_b0p15'
    expected = {'key': TARGETS[stage], 'repository': 'wbm-region-retrieval', 'path': folder,
                'variant': 'CNNRotationFusion/rotation/add_a0p4_b0p15', 'split': 'test',
                'manifest': folder+'/evaluation_manifest.json', 'marker': folder+'/completed.json',
                'identity': folder+'/run_identity.json', 'identity_schema_version': 1,
                'protocol': 'coverage_cross_combination_v1' if stage == 'cross_test' else 'retrieval_v1',
                'required_families': ['coverage'] if stage == 'cross_test' else ['exact', 'coverage']}
    _require(all(entry.get(k) == v for k, v in expected.items()), 'Canonical primary identity changed: '+stage)


def merge_sources(sources, proposed):
    """Pure append-only merge; rejects duplicates, extra targets and partial schema."""
    _require(proposed.get('operation') == 'append_supporting_manifests'
             and proposed.get('target_keys') == list(TARGETS.values())
             and proposed.get('new_formal_model_evaluations') == 0, 'Proposal must target exactly two current primary roles')
    additions = proposed.get('supporting_manifests')
    _require(isinstance(additions, list) and len(additions) == len(SUPPORT_FILES)
             and len(set(additions)) == len(additions)
             and additions == [proposed['analysis_root']+'/'+name for name in SUPPORT_FILES]
             and set(proposed.get('file_sha256', {})) == set(additions), 'Support inventory must contain exactly seven sealed documents')
    paths = [Path(p) for p in additions]
    _require(all(not p.is_absolute() and '..' not in p.parts for p in paths), 'Support paths must remain repository-relative')
    merged = deepcopy(sources)
    entries = {item['key']: item for item in merged['runs']}
    _require(len(entries) == len(merged['runs']) and set(TARGETS.values()) <= set(entries), 'Missing or duplicate existing canonical targets')
    changed = []
    for stage, key in TARGETS.items():
        entry = entries[key]
        _target_entry(entry, stage)
        previous = entry.get('supporting_manifests', [])
        _require(isinstance(previous, list) and len(set(previous)) == len(previous), 'Existing support inventory contains duplicates')
        missing = [name for name in additions if name not in previous]
        if missing:
            entry['supporting_manifests'] = [*previous, *missing]
            changed.append(key)
    return merged, changed


def validate_snapshot(previous, updated, merged, proposed):
    """Allow only support-list/provenance append on the two named old records."""
    before, after = previous['runs'], updated['runs']
    _require([x['key'] for x in before] == [x['key'] for x in after], 'Support registration must not add, remove or reorder results')
    specs = {x['key']: x for x in merged['runs']}
    targets = set(TARGETS.values())
    for old, new in zip(before, after):
        key = old['key']
        if key not in targets:
            _require(old == new, 'Untargeted result changed: '+key)
            continue
        _require(new['supporting_manifests'] == specs[key]['supporting_manifests'], 'Collected support list differs from proposal')
        old_support = old.get('provenance', {}).get('supporting_manifests', [])
        new_support = new.get('provenance', {}).get('supporting_manifests', [])
        old_names = old.get('supporting_manifests', [])
        _require([x['source']['path'] for x in old_support] == old_names,
                 'Old supporting provenance does not match old source inventory')
        _require(new_support[:len(old_support)] == old_support, 'An existing supporting document changed')
        expected_names = [name for name in proposed['supporting_manifests'] if name not in old_names]
        extra = new_support[len(old_support):]
        _require([x['source']['path'] for x in extra] == expected_names
                 and all(x['source']['sha256'] == proposed['file_sha256'][x['source']['path']] for x in extra),
                 'Unexpected supporting provenance addition')
        old_core, new_core = deepcopy(old), deepcopy(new)
        old_core.pop('supporting_manifests', None); new_core.pop('supporting_manifests', None)
        old_core.get('provenance', {}).pop('supporting_manifests', None)
        new_core.get('provenance', {}).pop('supporting_manifests', None)
        _require(old_core == new_core, 'Scientific identity, report or existing provenance changed: '+key)


def apply_proposal(output, *, workspace=None, root=ROOT):
    """Collect first, then atomically update catalogue with rollback receipts."""
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
        old_rows = {row['key']: row for row in previous['runs']}
        _require(len(old_rows) == len(previous['runs']) == len(sources['runs'])
                 and set(old_rows) == {row['key'] for row in sources['runs']}
                 and all(all(old_rows[row['key']].get(k) == v for k, v in row.items()) for row in sources['runs']),
                 'Existing source/snapshot identities are inconsistent')
        for name, content in catalog.render(previous).items():
            _require((folder/name).is_file() and (folder/name).read_text() == content, 'Existing catalogue is stale: '+name)
            original[name] = (folder/name).read_bytes()
        merged, changed = merge_sources(sources, proposed)
        if not changed:
            return {'status': 'already_attached', 'new_results': 0, 'updated_keys': []}
        updated = catalog.collect(workspace, merged, previous)
        validate_snapshot(previous, updated, merged, proposed)
        contents = {'sources.json': _json_bytes(merged), 'snapshot.json': _json_bytes(updated),
                    **{name: content.encode() for name, content in catalog.render(updated).items()}}
        _require(all((folder/name).read_bytes() == data for name, data in original.items()), 'Catalogue changed concurrently')
        before = {name: original.get(name) for name in contents}
        _require(all(data is not None or not (folder/name).exists() for name, data in before.items()),
                 'Generated output would overwrite an untracked file')
        backup_parent = output.parent/(output.name+'__catalog_support_backup')
        backup_parent.mkdir(exist_ok=True)
        backup = Path(tempfile.mkdtemp(prefix='transaction_', dir=backup_parent))
        for name, data in before.items():
            if data is not None:
                target = backup/'files'/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        audit = {'status': 'prepared', 'new_results': 0, 'updated_keys': changed,
                 'before_sha256': {name: hashlib.sha256(data).hexdigest() if data is not None else None for name, data in before.items()},
                 'after_sha256': {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}}
        (backup/'manifest.json').write_bytes(_json_bytes(audit))
        written = []
        try:
            for name, content in contents.items():
                _require(before[name] == ((folder/name).read_bytes() if (folder/name).exists() else None), 'Concurrent catalogue write: '+name)
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
        return {'status': 'attached', 'new_results': 0, 'updated_keys': changed,
                'documents_per_target': len(SUPPORT_FILES), 'backup': str(backup)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    output = (ROOT/args.run_root).resolve()
    result = apply_proposal(output) if args.apply else {'proposal': str(write_proposal(output))}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
