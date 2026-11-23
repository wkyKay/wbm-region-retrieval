"""Reject detached validation evidence and altered frozen inference inputs."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from cnn_refinement import inputs as input_module
from cnn_refinement import register, run, state
from cnn_refinement.evaluation import code_identity, specifications
from f_region_structure.state import seal
from wbm_retrieval.common import digest, file_hash, read_json, write_json


def _reseal(folder, signature):
    (folder / 'completed.json').unlink(missing_ok=True)
    seal(folder, signature)


def formal_validation_fixture(output):
    """Synthetic signed evidence, no real model training or scientific results."""
    model, split = 'context_dropout', 'valid'
    start = '2026-11-20T12:00:00+08:00'
    plan = state.signed({'plan': state.PLAN, 'frozen_at': start})
    write_json(output / 'frozen_plan.json', plan)
    root = state.signed({'version': state.VERSION, 'started_at': start,
        'run_id': '20261120T120000+0800__cnn_refinement', 'plan_signature': plan['signature'],
        'training_labels': False, 'test_parameter_search': False, 'authorized_models': list(state.VARIANTS),
        'split_counts': {'train': 25193, 'valid': 3599, 'test': 7199},
        'region_counts': {'train': 99786, 'valid': 14240}, 'source_binding': {},
        'baseline': {'training_summary': {'completed': True, 'epochs_completed': 100,
            'best_nce_epoch': 100, 'optimizer_steps': 39000, 'initial_parameter_sha256': register.INITIAL_HASH}}})
    write_json(output / 'run_manifest.json', root)
    source = output / 'synthetic_source'
    source_cache_marker = source / 'cache/valid/completed.json'
    write_json(source_cache_marker, {'synthetic': True})
    binding = {str(source_cache_marker): file_hash(source_cache_marker)}
    inputs = state.signed({'context_signature': root['signature'], 'source_binding': binding,
        'source_metadata': {'region_source': str(source)}, 'replay_references': {}})
    write_json(output / 'inputs_valid.json', inputs)
    checkpoint = output / 'training' / model / 'checkpoint.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'synthetic sealed-evidence fixture only')
    feature = output / 'features' / model / split
    feature.mkdir(parents=True)
    np.save(feature / 'original.npy', np.ones((1, 512), dtype=np.float64))
    feature_signature = 'synthetic_feature_signature'
    fm = state.signed({'feature_signature': feature_signature, 'parent_context': root['signature'],
        'model': model, 'split': split, 'views': ['original'], 'checkpoint_sha256': file_hash(checkpoint),
        'checkpoint': {'sha256': file_hash(checkpoint)}, 'checkpoint_path': str(checkpoint), 'token_layout': {}})
    write_json(feature / 'manifest.json', fm)
    write_json(feature / 'verification.json', {'synthetic': True})
    seal(feature, feature_signature)
    context = {'parent_context': root['signature'], 'model': model, 'split': split,
        'source_binding_sha256': digest(binding), 'code': code_identity(), 'specifications': specifications(),
        'token_layout': {}, 'h512_sha256': file_hash(feature / 'original.npy')}
    context['signature'] = digest(context)
    base = output / 'evaluation' / model
    write_json(base / 'model_context_valid.json', context)
    formula = {'status': 'verified', 'num_classes': 37, 'query_ids': list(range(37)),
        'variants': {style: {'candidate_pairs': 740, 'max_raw_formula_absolute_error': 0.,
                            'max_matching_absolute_error': 0.} for style in register.STYLES}}
    write_json(base / 'formula_audit_valid.json', formula)
    score_cache = {}
    for style in register.STYLES:
        folder = base / 'score_cache' / split / style
        metadata = {'context_signature': context['signature'], 'h512_sha256': context['h512_sha256'],
            'token_layout': {}, 'spec': specifications()[style], 'dtype': 'float64', 'num_queries': 500,
            'gallery_count': 3599, 'source_cache_sha256': file_hash(source_cache_marker),
            'retrieval_protocol_signature': register.BASE_PROTOCOL_SIGNATURE}
        signature = digest(metadata)
        write_json(folder / 'metadata.json', {**metadata, 'signature': signature})
        np.save(folder / 'scores.npy', np.zeros((1, 1), dtype=np.float64))
        np.savez(folder / 'identities.npz', synthetic=np.array([1]))
        seal(folder, signature)
        score_cache[style] = {'path': str(folder), 'signature': signature,
                              'completed_sha256': file_hash(folder / 'completed.json')}
    index = {'signature': context['signature'], 'context_signature': root['signature'], 'model': model,
             'split': split, 'formula_audit': str(base / 'formula_audit_valid.json'),
             'score_cache': score_cache, 'new_evaluations': 4, 'evaluations': {}}
    fixed = register.FROZEN[split]
    for stage in ('valid', 'cross_valid'):
        cross = stage.startswith('cross_')
        for style in register.STYLES:
            folder = base / stage / style
            signature = digest({'context': context['signature'], 'stage': stage,
                                'variant': style, 'spec': specifications()[style]})
            ident = {'key': 'synthetic_' + stage + '_' + style, 'run_id': 'synthetic_stage',
                     'started_at': '2026-11-20T12:00:01+08:00', 'evaluation_signature': signature}
            write_json(folder / 'run_identity.json', ident)
            write_json(base / 'identities' / (stage + '__' + style + '.json'), ident)
            provider = {key: score_cache[style][key] for key in ('signature', 'completed_sha256')}
            manifest = {**ident, 'signature': signature, 'context_signature': context['signature'], 'model': model,
                'stage': stage, 'split': split, 'variant': style, 'spec': specifications()[style], 'ks': register.KS,
                'tie_seed': 2026, 'num_queries': 500, 'gallery_count': 3599,
                'protocol': register.PROTOCOL if cross else 'retrieval_v1',
                'candidate_rule': register.RULE if cross else 'all_gallery_except_query',
                'retrieval_protocol_signature': fixed['derived_signature'] if cross else register.BASE_PROTOCOL_SIGNATURE,
                'base_protocol_signature': register.BASE_PROTOCOL_SIGNATURE, 'provider_source': provider,
                'source_binding_sha256': digest(binding), 'formal_protocol': True,
                'training_required': True, 'test_parameter_search': False, 'single_seed': 2026}
            if cross:
                manifest['derived_protocol'] = {'candidate_identity_sha256': fixed['candidate_identity_sha256']}
            write_json(folder / 'evaluation_manifest.json', manifest)
            (folder / 'rankings.csv').write_text('query_id,rank,candidate_id,similarity_score\n')
            row = {**ident, 'path': str(folder), 'signature': signature, 'model': model, 'style': style,
                   'stage': stage, 'split': split, 'register': True, 'historical_reference': False,
                   'metric_query_counts': {}}
            report_names = {'run_identity.json', 'evaluation_manifest.json', 'rankings.csv'}
            for family in (('coverage',) if cross else ('exact', 'coverage')):
                names = ('Hit', 'Precision', 'Recall', 'mAP') if family == 'exact' else (
                    'Coverage-nDCG', 'MeanCoverage', 'FullCoverageHit', 'FullCoveragePrecision',
                    'FullCoverageRecall', 'FullCoverage_mAP')
                metrics = {f'{name}@{k}': 0. for name in names for k in register.KS}
                counts = {key: 500 for key in metrics}
                document = {'metric_version': family + '_v1', 'ks': register.KS, 'split': split,
                            'num_queries': 500, 'metadata': manifest, 'metrics': metrics, 'metric_query_counts': counts}
                prefix = '' if family == 'exact' else 'coverage_'
                write_json(folder / (prefix + 'metrics.json'), document)
                for filename in ('per_query.csv', 'per_class.csv'):
                    (folder / (prefix + filename)).write_text('synthetic header\n')
                report_names.update(prefix + name for name in ('metrics.json', 'per_query.csv', 'per_class.csv'))
                row['metrics' if family == 'exact' else 'coverage_metrics'] = metrics
                row['metric_query_counts'][family] = counts
            pairs = fixed['candidate_pairs'] if cross else 500 * 3598
            candidate = {key: manifest[key] for key in ('split', 'ks', 'num_queries', 'gallery_count', 'protocol',
                'candidate_rule', 'retrieval_protocol_signature', 'base_protocol_signature')}
            candidate.update(status='verified', full_candidates_checked=pairs, query_ids_sha256=fixed['query_ids_sha256'],
                gallery_ids_sha256=fixed['gallery_ids_sha256'], metric_query_counts=row['metric_query_counts']['coverage'])
            if cross:
                candidate.update({key: fixed[key] for key in ('candidate_identity_sha256', 'candidate_count_min', 'candidate_count_max')})
            audit = {key: manifest[key] for key in ('signature', 'context_signature', 'stage', 'split', 'variant',
                'spec', 'ks', 'provider_source', 'source_binding_sha256')}
            audit.update(status='verified', exact_reports=not cross, num_queries=500, ranking_rows=250000,
                full_candidates_checked=pairs, all_queries_full_candidate_topk_exact=True,
                ranking_sha256=file_hash(folder / 'rankings.csv'), candidate_audit=candidate,
                files={name: file_hash(folder / name) for name in report_names}, metric_query_counts=row['metric_query_counts'],
                aggregation_values_checked=1, max_metric_absolute_error=0., max_saved_score_absolute_error=0.)
            if cross:
                audit['candidate_identity_sha256'] = fixed['candidate_identity_sha256']
            else:
                audit['top10_hits'] = 0
            write_json(folder / 'verification.json', audit)
            seal(folder, signature)
            row['completed_sha256'] = file_hash(folder / 'completed.json')
            index['evaluations'][stage + '/' + style] = row
    write_json(base / 'evaluation_index_valid.json', index)
    return model, root, index


def test_valid_gate_accepts_complete_evidence_without_root_seal_and_reuses_receipt(tmp_path):
    model, root, _ = formal_validation_fixture(tmp_path)
    assert not (tmp_path / 'completed.json').exists()
    gate = run.require_valid_gate(tmp_path, model, root['signature'])
    assert gate['status'] == 'verified'
    receipt = tmp_path / ('valid_gate_' + model + '.json')
    before = (receipt.read_bytes(), receipt.stat().st_mtime_ns)
    assert run.require_valid_gate(tmp_path, model, root['signature']) == gate
    assert before == (receipt.read_bytes(), receipt.stat().st_mtime_ns)


@pytest.mark.parametrize('attack', ['root_context', 'feature_context', 'empty_score_cache', 'missing_seal',
    'unsealed_verification', 'false_topk', 'wrong_full_pool', 'wrong_stage', 'wrong_rank_hash', 'nonzero_metric_error'])
def test_valid_gate_rejects_detached_or_forged_verification(tmp_path, attack):
    model, root, index = formal_validation_fixture(tmp_path)
    index_path = tmp_path / 'evaluation' / model / 'evaluation_index_valid.json'
    row = index['evaluations']['valid/cnn_only']
    folder = Path(row['path'])
    if attack == 'root_context':
        index['context_signature'] = 'another_experiment'
    elif attack == 'feature_context':
        path = tmp_path / 'features' / model / 'valid/manifest.json'
        feature = read_json(path)
        feature['parent_context'] = 'another_experiment'
        write_json(path, state.signed({k: v for k, v in feature.items() if k != 'signature'}))
        _reseal(path.parent, feature['feature_signature'])
    elif attack == 'empty_score_cache':
        index['score_cache'] = {}
    elif attack == 'missing_seal':
        (folder / 'completed.json').unlink()
    else:
        audit = read_json(folder / 'verification.json')
        if attack == 'unsealed_verification':
            marker = read_json(folder / 'completed.json')
            marker['files'].pop('verification.json')
            write_json(folder / 'completed.json', marker)
        else:
            changes = {'false_topk': ('all_queries_full_candidate_topk_exact', False),
                       'wrong_full_pool': ('full_candidates_checked', 500 * 1000),
                       'wrong_stage': ('stage', 'test'), 'wrong_rank_hash': ('ranking_sha256', 'forged'),
                       'nonzero_metric_error': ('max_metric_absolute_error', .01)}
            key, value = changes[attack]
            audit[key] = value
            write_json(folder / 'verification.json', audit)
            _reseal(folder, row['signature'])
        row['completed_sha256'] = file_hash(folder / 'completed.json')
    write_json(index_path, index)
    with pytest.raises(ValueError):
        run.require_valid_gate(tmp_path, model, root['signature'])
    assert not (tmp_path / ('valid_gate_' + model + '.json')).exists()


def evaluation_input_fixture(tmp_path, monkeypatch):
    import paper_c_ablation.inputs
    source = tmp_path / 'source'
    cache = source / 'cache'
    cache.mkdir(parents=True)
    arrays = {'map_ids': np.array([7, 9]), 'offsets': np.array([0, 1, 2]), 'area': np.array([5, 6])}
    np.save(cache / 'test_F.npy', np.ones((2, 3, 96, 96), dtype=np.uint8))
    np.savez(cache / 'test_regions.npz', **arrays)
    np.save(cache / 'test_region_labels.npy', np.ones((2, 52, 52), dtype=np.uint8))
    marker = {'signature': 'frozen_suite', 'files': {
        'cache/test_' + suffix: file_hash(cache / ('test_' + suffix))
        for suffix in ('F.npy', 'regions.npz', 'region_labels.npy')}}
    write_json(cache / 'test_prepared.json', marker)
    data = {'cache': arrays, 'source_binding': {}, 'source_metadata': {}, 'replay_references': {}}
    monkeypatch.setattr(paper_c_ablation.inputs, 'load', lambda split: deepcopy(data))
    monkeypatch.setattr(input_module, 'run_context', lambda *_: {'signature': 'current_context'})
    return source, cache, marker, {'source_root': source, 'source_manifest': {'signature': 'frozen_suite'}}


@pytest.mark.parametrize('attack', ['none', 'source_suite', 'patch_bytes', 'token_order', 'patch_dtype'])
def test_inference_inputs_bind_original_patch_bytes_token_order_and_dtype(tmp_path, monkeypatch, attack):
    source, cache, marker, inputs = evaluation_input_fixture(tmp_path, monkeypatch)
    if attack == 'source_suite':
        marker['signature'] = 'obsolete_suite'
    elif attack == 'patch_bytes':
        (cache / 'test_F.npy').write_bytes((cache / 'test_F.npy').read_bytes() + b'changed')
    elif attack == 'token_order':
        np.savez(cache / 'test_regions.npz', map_ids=[9, 7], offsets=[0, 1, 2], area=[5, 6])
        marker['files']['cache/test_regions.npz'] = file_hash(cache / 'test_regions.npz')
    elif attack == 'patch_dtype':
        np.save(cache / 'test_F.npy', np.ones((2, 3, 96, 96), dtype=np.float32))
        marker['files']['cache/test_F.npy'] = file_hash(cache / 'test_F.npy')
    write_json(cache / 'test_prepared.json', marker)
    output = tmp_path / 'new_run'
    if attack == 'none':
        data = input_module.evaluation_data('test', inputs, output)
        assert isinstance(data['refinement_patches'], np.memmap)
        assert len(data['source_binding']) == 4
        receipt = state.read_signed(output / 'inputs_test.json')
        assert receipt['context_signature'] == 'current_context'
    else:
        with pytest.raises(ValueError):
            input_module.evaluation_data('test', inputs, output)
        assert not (output / 'inputs_test.json').exists()


@pytest.mark.parametrize('attack', ['none', 'unsealed_verification', 'foreign_signature', 'foreign_stage',
                                   'false_topk', 'wrong_candidate_identity'])
def test_historical_crossvalid_baseline_must_bind_the_exact_independent_audit(tmp_path, monkeypatch, attack):
    import paper_c_ablation.inputs
    from f_region_wbbs.protocol import BASE_PROTOCOL_SIGNATURE, FROZEN, PROTOCOL, RULE
    source, cache, marker, inputs = evaluation_input_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(input_module, 'ROOT', tmp_path)
    # Reuse the synthetic cache contents under valid names, keeping all byte bindings explicit.
    for suffix in ('F.npy', 'regions.npz', 'region_labels.npy'):
        (cache / ('valid_' + suffix)).write_bytes((cache / ('test_' + suffix)).read_bytes())
    write_json(cache / 'valid_prepared.json', {'signature': 'frozen_suite', 'files': {
        'cache/valid_' + suffix: file_hash(cache / ('valid_' + suffix))
        for suffix in ('F.npy', 'regions.npz', 'region_labels.npy')}})
    folder = tmp_path / 'artificial/f_region_wbbs_20261121T143545+0800/cross_valid/region_j'
    meta = {'key': 'f_region_wbbs_f_region_wbbs_20261121T143545_0800_cross_valid_region_j',
        'signature': 'historical_signature', 'context_signature': 'historical_context',
        'stage': 'cross_valid', 'split': 'valid', 'variant': 'region_j', 'spec': {'family': 'j_baseline'},
        'protocol': PROTOCOL, 'candidate_rule': RULE, 'retrieval_protocol_signature': FROZEN['valid']['derived_signature'],
        'base_protocol_signature': BASE_PROTOCOL_SIGNATURE, 'num_queries': 500, 'gallery_count': 3599,
        'ks': [1, 5, 10, 20, 50, 100, 500]}
    write_json(folder / 'evaluation_manifest.json', meta)
    (folder / 'rankings.csv').write_text('synthetic historical ranking\n')
    audit = {key: meta[key] for key in ('signature', 'context_signature', 'stage', 'split', 'variant', 'spec', 'ks')}
    audit.update(status='verified', num_queries=500, ranking_rows=250000, all_queries_full_candidate_topk_exact=True,
        full_candidates_checked=FROZEN['valid']['candidate_pairs'],
        candidate_identity_sha256=FROZEN['valid']['candidate_identity_sha256'],
        ranking_sha256=file_hash(folder / 'rankings.csv'))
    if attack == 'foreign_signature':
        audit['signature'] = 'foreign'
    elif attack == 'foreign_stage':
        audit['stage'] = 'cross_test'
    elif attack == 'false_topk':
        audit['all_queries_full_candidate_topk_exact'] = False
    elif attack == 'wrong_candidate_identity':
        audit['candidate_identity_sha256'] = 'obsolete_pool'
    write_json(folder / 'verification.json', audit)
    for filename in ('coverage_metrics.json', 'coverage_per_query.csv', 'coverage_per_class.csv'):
        (folder / filename).write_text('synthetic historical evidence\n')
    seal(folder, meta['signature'])
    if attack == 'unsealed_verification':
        done = read_json(folder / 'completed.json')
        done['files'].pop('verification.json')
        write_json(folder / 'completed.json', done)
    output = tmp_path / 'new_run'
    if attack == 'none':
        data = input_module.evaluation_data('valid', inputs, output)
        assert str(folder / 'verification.json') in data['source_binding']
        assert data['replay_references']['cross_valid']['primary']['folder'] == str(folder)
    else:
        with pytest.raises(ValueError, match='Historical cross-valid'):
            input_module.evaluation_data('valid', inputs, output)
        assert not (output / 'inputs_valid.json').exists()
