"""Validate eighteen reports and register fourteen new results, excluding replay controls."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile

from b1_structure_fusion.register import _atomic_write, _json_bytes, _lock
from f_region_structure.register import _relative, _metric_values
from f_region_wbbs.protocol import BASE_PROTOCOL_SIGNATURE, FROZEN, PROTOCOL, RULE
from results_catalog import build as catalog
from wbm_retrieval.common import digest, file_hash, read_json
from .state import (ROOT, TZ, VERSION, KS, SETTINGS, STAGES, REPLAY_KEYS, grid,
                    plan, read_signed, verified, evaluation_signature)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _error(value, tolerance, label):
    _require(type(value) in (float, int) and math.isfinite(value) and 0 <= value <= tolerance,
             'Invalid independent audit error: '+label)


def _bound_files(binding):
    _require(isinstance(binding, dict) and bool(binding), 'Nonempty source-file bindings required')
    for path, expected in binding.items():
        _require(isinstance(path, str) and Path(path).is_absolute()
                 and isinstance(expected, str) and re.fullmatch(r'[0-9a-f]{64}', expected),
                 'Canonical absolute source path and SHA256 required')
        _require(Path(path).is_file() and file_hash(path) == expected, 'Frozen source changed: '+path)


def _stamp(value):
    _require(isinstance(value, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00', value),
             'Real Asia/Shanghai execution timestamp required')
    return datetime.fromisoformat(value)


def _context(output, expected=None):
    output = Path(output)
    manifest = read_signed(output/'run_manifest.json')
    context = manifest['context']
    signature = digest(context)
    _require(manifest.get('version') == VERSION and context.get('version') == VERSION
             and context.get('grid') == grid() and context.get('stages') == list(STAGES)
             and context.get('settings') == SETTINGS and context.get('ks') == KS
             and context.get('training_required') is False and context.get('test_parameter_search') is False,
             'Scientific context does not match the frozen six-way design')
    _require(expected is None or expected == signature, 'Context signature changed')
    started = _stamp(manifest.get('started_at'))
    _require(manifest.get('run_id') == started.strftime('%Y%m%dT%H%M%S%z')+'__paper_c_ablation',
             'Root run_id is not the persisted actual start time')
    _require(read_signed(output/'frozen_plan.json') == plan(signature, manifest['started_at']),
             'Prespecified plan was changed or selected after observing performance')
    return manifest, signature


def _replay(output, stage, context_signature, binding):
    expected = REPLAY_KEYS[stage]
    receipt = read_json(output/f'replay_{stage}.json')
    _require(receipt.get('status') == 'verified' and receipt.get('stage') == stage
             and set(receipt.get('controls', {})) == set(expected), 'Missing or changed replay-control receipt')
    split = 'valid' if stage == 'valid' else 'test'
    inputs = read_json(output/f'inputs_{split}.json')
    references = inputs.get('replay_references', {}).get(stage, {})
    _require(set(references) == set(expected), 'Replay source set differs from the frozen plan')
    for name, old_key in expected.items():
        item, source = receipt['controls'][name], references[name]
        source_file = Path(source['folder'])/'rankings.csv'
        _require(source['key'] == old_key and item.get('status') == 'verified'
                 and item.get('source_key') == old_key and item.get('source') == str(source_file)
                 and item.get('source_sha256') == source['ranking_sha256']
                 and item.get('source_completed_sha256') == source['completed_sha256']
                 and binding.get(str(source_file)) == source['ranking_sha256']
                 and binding.get(str(Path(source['folder'])/'completed.json')) == source['completed_sha256']
                 and item.get('score_cache_sha256') == file_hash(output/'score_cache'/split/name/'completed.json')
                 and item.get('num_queries') == (500 if stage == 'valid' else 7199)
                 and item.get('top500_identical') is True
                 and item.get('ranking_rows') == (500 if stage == 'valid' else 7199)*max(KS),
                 'Historical control did not replay every saved identity, rank and score')
        _error(item.get('max_score_error'), 0., 'historical replay')


def _formula(output, split):
    result = read_json(output/f'formula_audit_{split}.json')
    _require(result.get('status') == 'verified' and set(result.get('variants', {})) == set(grid())
             and result.get('candidate_pairs') == 740*len(grid()), 'Sampled regional formula audit incomplete')
    for row in result['variants'].values():
        _require(len(set(row.get('query_ids', []))) == 37 and row.get('candidate_pairs') == 740,
                 'Formula audit must include all 37 combinations and 740 pairs per variant')
        _error(row.get('max_raw_formula_absolute_error'), 2e-6, 'sampled raw formula')
        _error(row.get('max_fusion_absolute_error'), 2e-6, 'sampled fusion formula')
        _error(row.get('max_matching_absolute_error'), 1e-12, 'sampled matching')
    _error(result.get('max_raw_formula_absolute_error'), 2e-6, 'maximum sampled raw formula')
    _error(result.get('max_fusion_absolute_error'), 2e-6, 'maximum sampled fusion formula')
    _error(result.get('max_matching_absolute_error'), 1e-12, 'maximum sampled matching')


def validate_stage(output, stage, context_signature=None):
    """Complete read-only stage gate, usable before the final root seal exists."""
    output = Path(output).resolve()
    _require(stage in STAGES, 'Unknown formal stage')
    manifest, signature = _context(output, context_signature)
    context = manifest['context']
    split = 'valid' if stage == 'valid' else 'test'
    cross = stage == 'cross_test'
    nq, ng = (500, 3599) if split == 'valid' else (7199, 7199)
    protocol = PROTOCOL if cross else 'retrieval_v1'
    rule = RULE if cross else 'all_gallery_except_query'
    protocol_signature = FROZEN['test']['derived_signature'] if cross else BASE_PROTOCOL_SIGNATURE
    pair_count = FROZEN['test']['candidate_pairs'] if cross else nq*(ng-1)
    input_record = read_json(output/f'inputs_{split}.json')
    binding = input_record['source_binding']
    _require(input_record.get('context_signature') == signature, 'Stage input context changed')
    _require(split != 'valid' or binding == context['source_binding'], 'Initial source binding changed')
    if split == 'test':
        _require(input_record.get('valid_completed_sha256') == file_hash(output/'valid_completed.json')
                 and input_record.get('frozen_plan_sha256') == file_hash(output/'frozen_plan.json'),
                 'Test inputs are not linked to the unchanged verified valid gate')
    index = read_json(output/f'evaluation_index_{stage}.json')
    _require(index.get('context_signature') == signature and index.get('stage') == stage
             and index.get('split') == split and index.get('specs') == grid()
             and set(index.get('results', {})) == set(grid()), 'Complete six-variant stage index required')
    _require({p.name for p in (output/stage).iterdir() if not p.name.startswith('.')} == set(grid()),
             'Stage must contain exactly the six prespecified evaluations')
    _formula(output, split)
    frozen_at = _stamp(read_signed(output/'frozen_plan.json')['frozen_at'])
    for name, spec in grid().items():
        folder = output/stage/name
        esig = evaluation_signature(signature, stage, name)
        _require(verified(folder, esig), 'Unsealed evaluation: '+stage+'/'+name)
        row = index['results'][name]
        _require(row.get('completed_sha256') == file_hash(folder/'completed.json'), 'Index completion hash differs')
        identity = read_json(folder/'run_identity.json')
        persistent = read_json(output/'identities'/(stage+'__'+name+'.json'))
        meta = read_json(folder/'evaluation_manifest.json')
        old_key = REPLAY_KEYS[stage].get(name)
        expected = {'signature': esig, 'context_signature': signature, 'stage': stage, 'split': split,
            'variant': name, 'spec': spec, 'ks': KS, 'tie_seed': 2026, 'num_queries': nq, 'gallery_count': ng,
            'protocol': protocol, 'candidate_rule': rule, 'retrieval_protocol_signature': protocol_signature,
            'base_protocol_signature': BASE_PROTOCOL_SIGNATURE, 'source_binding_sha256': digest(binding),
            'formal_protocol': True, 'training_required': False, 'test_parameter_search': False,
            'frozen_plan_sha256': file_hash(output/'frozen_plan.json'),
            'replay_control': old_key is not None, 'source_evaluation_key': old_key}
        _require(identity == persistent and identity.get('evaluation_signature') == esig
                 and all(meta.get(k) == v for k, v in {**identity, **expected}.items())
                 and _stamp(identity.get('started_at')) >= frozen_at,
                 'Evaluation identity, original plan time or scientific metadata differs')
        if cross:
            derived = meta.get('derived_protocol', {})
            _require(derived.get('candidate_rule') == RULE
                     and derived.get('candidate_identity_sha256') == FROZEN['test']['candidate_identity_sha256'],
                     'Cross candidate identities differ')
        cache = output/'score_cache'/split/name
        csig = digest({'context': signature, 'split': split, 'variant': name, 'spec': spec, 'score_cache_version': 1})
        _require(verified(cache, csig), 'Full candidate score cache is unsealed')
        cache_meta = read_json(cache/'metadata.json')
        source_cache = str(Path(input_record['source_metadata']['region_source'])/'cache'/split/'completed.json')
        _require(all(cache_meta.get(k) == v for k, v in {'signature': csig, 'context_signature': signature,
            'split': split, 'variant': name, 'spec': spec, 'retrieval_protocol_signature': BASE_PROTOCOL_SIGNATURE,
            'num_queries': nq, 'gallery_count': ng, 'dtype': 'float64', 'self_scores': 'zero; never ranked',
            'source_cache_sha256': binding.get(source_cache)}.items()) and binding.get(source_cache),
            'Score cache settings or historical region source binding differs')
        provider = {'signature': csig, 'completed_sha256': file_hash(cache/'completed.json')}
        _require(meta.get('provider_source') == provider, 'Report detached from complete score cache')
        receipt = read_json(folder/'verification.json')
        receipt_expected = {'status': 'verified', 'signature': esig, 'context_signature': signature,
            'stage': stage, 'split': split, 'variant': name, 'spec': spec, 'ks': KS, 'tie_seed': 2026,
            'provider_source': provider, 'source_binding_sha256': digest(binding), 'source_file_count': len(binding),
            'num_queries': nq, 'ranking_rows': nq*max(KS), 'full_candidates_checked': pair_count,
            'all_queries_full_candidate_topk_exact': True, 'exact_reports': not cross,
            'candidate_count_min': 6999 if cross else ng-1, 'candidate_count_max': 7170 if cross else ng-1,
            'ranking_sha256': file_hash(folder/'rankings.csv')}
        _require(all(receipt.get(k) == v for k, v in receipt_expected.items()), 'Full-candidate independent verification differs')
        _error(receipt.get('max_metric_absolute_error'), 1e-12, 'metrics')
        _error(receipt.get('max_saved_score_absolute_error'), 1e-12, 'saved scores')
        files = {'run_identity.json', 'evaluation_manifest.json', 'rankings.csv', 'coverage_metrics.json',
                 'coverage_per_query.csv', 'coverage_per_class.csv'}
        if not cross:
            files.update(('metrics.json', 'per_query.csv', 'per_class.csv'))
        _require(set(receipt.get('files', {})) == files, 'Independent verification did not bind every report')
        for filename, expected_hash in receipt['files'].items():
            _require(file_hash(folder/filename) == expected_hash, 'Verified report changed: '+filename)
        if cross:
            _require(not any((folder/p).exists() for p in ('metrics.json', 'per_query.csv', 'per_class.csv')),
                     'Cross evaluation must not invent exact positives')
            _require(receipt.get('candidate_identity_sha256') == FROZEN['test']['candidate_identity_sha256'],
                     'Cross complete candidate identity changed')
        for family in (('coverage',) if cross else ('exact', 'coverage')):
            filename = 'metrics.json' if family == 'exact' else 'coverage_metrics.json'
            report = read_json(folder/filename)
            _metric_values(report.get('metrics'), family)
            _require(report.get('split') == split and report.get('num_queries') == nq
                     and report.get('ks') == KS and report.get('metric_version') == family+'_v1'
                     and all(report.get('metadata', {}).get(k) == v for k, v in meta.items())
                     and not report.get('metadata', {}).get('smoke')
                     and row.get('metrics' if family == 'exact' else 'coverage_metrics') == report['metrics'],
                     'Metric report protocol, metadata or indexed result differs')
            if family == 'coverage' and cross:
                counts = {m+'@'+str(k): (7000 if m == 'Coverage-nDCG' else
                          6200 if m in ('FullCoverageRecall', 'FullCoverage_mAP') else 7199)
                          for m in ('Coverage-nDCG', 'MeanCoverage', 'FullCoverageHit',
                                    'FullCoveragePrecision', 'FullCoverageRecall', 'FullCoverage_mAP') for k in KS}
                _require(report.get('metric_query_counts') == counts
                         and report.get('queries_without_full_coverage_candidates') == 999,
                         'Cross metric validity denominators changed')
        if not cross:
            hits = receipt.get('top10_hits')
            _require(type(hits) is int and 0 <= hits <= nq*10 and row.get('top10_hits') == hits
                     and abs(row['metrics']['Precision@10']-hits/(nq*10)) <= 1e-12,
                     'Integer exact Top-10 hits differ from report')
    _replay(output, stage, signature, binding)
    return index


def verify_run(output, *, check_source=True):
    output = Path(output).resolve()
    _require(output != (ROOT/'results').resolve() and output.is_relative_to((ROOT/'results').resolve()),
             'New paper artifacts must remain under an isolated results/ root')
    manifest, signature = _context(output)
    _require(verified(output, signature), 'Full eighteen-report root is not sealed')
    from .run import code_identity, require_valid_gate
    _require(manifest['context']['code'] == code_identity(), 'Scientific or validation source code changed')
    require_valid_gate(output, signature)
    for stage in STAGES:
        validate_stage(output, stage, signature)
    for split in ('valid', 'test'):
        binding = read_json(output/f'inputs_{split}.json')['source_binding']
        if check_source:
            _bound_files(binding)
    tc = read_signed(output/'test_context.json')
    _require(tc.get('context_signature') == signature and tc.get('specs') == grid()
             and tc.get('valid_completed_sha256') == file_hash(output/'valid_completed.json')
             and tc.get('frozen_plan_sha256') == file_hash(output/'frozen_plan.json')
             and tc.get('source_binding_sha256') == digest(read_json(output/'inputs_test.json')['source_binding']),
             'Test stage is detached from valid verification or sources')
    before, after = (read_json(output/('source_protection_'+p+'.json')) for p in ('before', 'after'))
    _require(after.get('status') == 'unchanged' and after.get('files') == len(before), 'Historical source protection failed')
    return manifest


def proposal(output, *, root=ROOT):
    output, root = Path(output).resolve(), Path(root).resolve()
    verify_run(output)
    items = []
    for stage in STAGES:
        for name in grid():
            if name in REPLAY_KEYS[stage]:
                continue
            folder = output/stage/name
            ident = read_json(folder/'run_identity.json')
            cross = stage == 'cross_test'
            item = {'key': ident['key'], 'repository': 'wbm-region-retrieval', 'path': _relative(folder, root),
                'category': 'cross_combination' if cross else 'fusion_ablation', 'method': 'F',
                'variant': 'PaperC/'+name, 'split': ident['split'],
                'manifest': _relative(folder/'evaluation_manifest.json', root),
                'marker': _relative(folder/'completed.json', root), 'identity_schema_version': 1,
                'identity': _relative(folder/'run_identity.json', root), 'selection': None, 'coverage': None,
                'supervision': 'self_supervised', 'display': '区域J＋F固定消融 '+name+('（跨组合）' if cross else ''),
                'required_families': ['coverage'] if cross else ['exact', 'coverage'],
                'protocol': PROTOCOL if cross else 'retrieval_v1', 'posthoc': True,
                'supporting_manifests': [_relative(output/p, root) for p in (
                    'run_manifest.json', 'completed.json', 'frozen_plan.json', 'valid_completed.json',
                    'source_protection_after.json', f'inputs_{ident["split"]}.json',
                    f'evaluation_index_{stage}.json', f'formula_audit_{ident["split"]}.json',
                    f'replay_{stage}.json', f'{stage}/{name}/verification.json')],
                'design_note': 'Six prespecified fixed-recipe component ablations. All six valid results pass '
                    'implementation/protocol verification before all six ordinary and all six cross-test '
                    'evaluations; no performance-based screening, retuning, retraining or checkpoint selection. '
                    'Reuses frozen cc8 Top-4 h512 best_nce seed2026 and original joint region descriptors. '
                    'Four historical endpoints are exact replay controls and are not registered again. '
                    'Removal of the pairwise area score retains sqrt(area) query aggregation. Conditional '
                    'posthoc component analysis; no claim of independent component contributions or seed stability.'}
            catalog.execution_identity(root, item, folder/'completed.json', read_json(folder/'evaluation_manifest.json'))
            items.append(item)
    _require(len(items) == 14 and len({x['key'] for x in items}) == 14, 'Exactly fourteen new identities required')
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
    """Collect and compare in memory; atomically preserve every earlier catalogue identity."""
    output, root = Path(output).resolve(), Path(root).resolve()
    workspace = root.parent if workspace is None else Path(workspace).resolve()
    _require((workspace/'wbm-region-retrieval').resolve() == root, 'Workspace/repository mismatch')
    proposed = read_json(write_proposal(output, root=root))
    folder = root/'results_catalog'
    with _lock(folder):
        original = {name: (folder/name).read_bytes() for name in ('sources.json', 'snapshot.json')}
        sources, previous = json.loads(original['sources.json']), json.loads(original['snapshot.json'])
        fingerprint = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
        _require(previous.get('source_spec_sha256') == fingerprint, 'Existing source/snapshot fingerprints differ')
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
                 and set(new_runs) == set(old_runs)|{x['key'] for x in additions}
                 and all(new_runs.get(k) == v for k, v in old_runs.items()),
                 'Collection modified an earlier result or introduced unexpected identities')
        contents = {'snapshot.json': _json_bytes(updated), 'sources.json': _json_bytes(merged),
                    **{name: text.encode() for name, text in catalog.render(updated).items()}}
        _require(all((folder/n).read_bytes() == data for n, data in original.items()), 'Catalogue changed concurrently')
        before = {name: original.get(name) for name in contents}
        _require(all(data is not None or not (folder/name).exists() for name, data in before.items()),
                 'A new generated output would overwrite untracked data')
        backup_parent = output/'catalog_backup'
        backup_parent.mkdir(exist_ok=True)
        backup = Path(tempfile.mkdtemp(prefix=datetime.now(TZ).strftime('%Y%m%dT%H%M%S%z')+'_', dir=backup_parent))
        for name, data in before.items():
            if data is not None:
                target = backup/'files'/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        audit = {'status': 'prepared', 'new_keys': [x['key'] for x in additions],
            'before_sha256': {n: hashlib.sha256(v).hexdigest() if v is not None else None for n,v in before.items()},
            'after_sha256': {n: hashlib.sha256(v).hexdigest() for n,v in contents.items()}}
        (backup/'manifest.json').write_bytes(_json_bytes(audit))
        written = []
        try:
            for name, content in contents.items():
                _require(before[name] == ((folder/name).read_bytes() if (folder/name).exists() else None),
                         'Concurrent catalogue write: '+name)
                written.append(name)
                _atomic_write(folder/name, content)
            _require(all((folder/n).read_bytes() == v for n,v in contents.items()), 'Catalogue write verification failed')
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    output = (ROOT/args.run_root).resolve()
    print(json.dumps(apply_proposal(output) if args.apply else {'proposal': str(write_proposal(output))}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
