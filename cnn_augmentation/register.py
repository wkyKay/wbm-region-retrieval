"""Register only sealed, independently verified CNN experiments; preserve old identities."""
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
from f_region_structure.register import _complete, _metric_values, _relative
from f_region_wbbs.protocol import BASE_PROTOCOL_SIGNATURE, FROZEN, PROTOCOL, RULE
from paper_c_ablation.register import _bound_files
from results_catalog import build as catalog
from wbm_retrieval.common import digest, file_hash, read_json
from .evaluation import STYLES, code_identity, specifications
from .state import ROOT, VERSION, PLAN, MODELS, VARIANTS, KS, read_signed, run_context

INITIAL_HASH = 'df8cfa59821feeff548ef6e963339b770ecc058fad7c9b4ac0e2727a50a57260'
STAGES = ('valid', 'cross_valid', 'test', 'cross_test')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _stamp(value):
    _require(isinstance(value, str) and re.fullmatch(
        r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00', value), 'Real Asia/Shanghai start time required')
    return datetime.fromisoformat(value)


def _error(value, tolerance, name):
    _require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= tolerance,
             'Invalid independent verification error: '+name)


def _validate_augmentation_plan():
    from .augmentation import AUGMENTATION_SPECS
    p,a=PLAN['single_augmentations'],AUGMENTATION_SPECS
    _require(set(p)==set(a)==set(VARIANTS), 'Frozen plan and implemented augmentations differ')
    checks=(p['crop']['area_scale']==a['crop']['area_scale'],
        p['crop']['aspect_ratio']==a['crop']['aspect_ratio'],
        p['crop']['probability']==a['crop']['probability'],
        p['cutout']['holes']==a['cutout']['holes'],
        p['cutout']['max_height']==p['cutout']['max_width']==a['cutout']['side'],
        p['cutout']['probability']==a['cutout']['probability'],
        p['cutout']['fill']==a['cutout']['fill']==0,
        p['noise']['selection_probability']==a['noise']['probability'],
        p['noise']['replacement_defect_probability']==.5,
        a['noise']['replacement']=='uniform normal/defect, including possible unchanged values',
        p['rotation']['angle_degrees']==a['rotation']['angle_degrees'],
        p['rotation']['probability']==a['rotation']['probability'],
        p['shift']['x_fraction']==a['shift']['fraction_x'],
        p['shift']['y_fraction']==a['shift']['fraction_y'],
        p['shift']['probability']==a['shift']['probability'])
    _require(all(checks), 'Implemented augmentation parameters differ from the frozen plan')


def _root_context(output):
    manifest = run_context(output)
    _validate_augmentation_plan()
    evidence=read_json(output/'paper_source.json')
    _require(file_hash(output/'paper_source.json') == manifest.get('paper_source_sha256')
             and evidence == read_json(ROOT/'cnn_augmentation/paper_source.json')
             and all(evidence.get('author_implementation',{}).get(k)==v for k,v in PLAN['paper'].items()),
             'Paper/operator source evidence differs from the frozen run')
    plan = read_signed(output/'frozen_plan.json')
    started = _stamp(manifest.get('started_at'))
    _require(manifest.get('run_id') == started.strftime('%Y%m%dT%H%M%S%z')+'__cnn_augmentation'
             and plan.get('frozen_at') == manifest['started_at'] and plan.get('plan') == PLAN
             and manifest.get('version') == VERSION and manifest.get('training_labels') is False
             and manifest.get('test_parameter_search') is False
             and manifest.get('authorized_models') == list(VARIANTS)
             and manifest.get('split_counts') == {'train': 25193, 'valid': 3599, 'test': 7199}
             and manifest.get('region_counts') == {'train': 99786, 'valid': 14240},
             'Root identity, frozen plan or deduplicated data changed')
    baseline = manifest.get('baseline', {}).get('training_summary', {})
    _require(baseline.get('completed') is True and baseline.get('epochs_completed') == 100
             and baseline.get('best_nce_epoch') == 100 and baseline.get('optimizer_steps') == 39000
             and baseline.get('initial_parameter_sha256') == INITIAL_HASH,
             'Baseline must be the original completed seed2026 epoch100 reference')
    return manifest


def _training(output, variant, root_manifest):
    from .augmentation import AUGMENTATION_SPECS
    folder = output/'training'/variant
    manifest = read_signed(folder/'run_manifest.json')
    science = manifest.get('science', {})
    signature = manifest.get('training_signature')
    _require(signature == digest(science), 'Training scientific signature changed')
    _require(science.get('code') and all(file_hash(ROOT/name) == sha for name, sha in science['code'].items()),
             'Training implementation changed after the frozen run')
    required = ['run_manifest.json', 'training_summary.json', 'history.json', 'history.csv',
                'checkpoint.pt', 'last.pt', 'status.json', 'augmentation_audit.json']
    required += [f'checkpoints/epoch_{epoch:03d}.pt' for epoch in (15, 25, 50, 75, 100)]
    _complete(folder, signature, required)
    _require(science.get('variant') == variant and science.get('seed') == 2026
             and science.get('batch_size') == 256 and science.get('initial_parameter_sha256') == INITIAL_HASH
             and science.get('refinement_context_signature') == root_manifest['signature']
             and science.get('plan_signature') == root_manifest['plan_signature']
             and science.get('source_protection') == root_manifest['source_binding']
             and science.get('selection') == 'fixed_epoch_100_no_best_loss_or_test_selection'
             and science.get('augmentation_spec') == AUGMENTATION_SPECS[variant],
             'Training is detached from the frozen experiment, baseline or augmentation')
    settings = science.get('local', {})
    recipe = {'epochs': 100, 'learning_rate': .01, 'momentum': .9, 'weight_decay': .001,
              'projector_dim': 128, 'temperature': .07, 'num_negatives': 5000,
              'memory_momentum': .5, 'optimizer': 'sgd', 'scheduler': 'wapirl_cosine',
              'projector_type': 'linear', 'negative_sampling': 'with_replacement'}
    _require(all(settings.get(k) == v for k, v in recipe.items())
             and not settings.get('early_stopping', {}).get('enabled')
             and not settings.get('opening', {}).get('enabled'), 'Training budget or objective changed')
    _require(_stamp(manifest.get('started_at')) >= _stamp(root_manifest['started_at'])
             and manifest.get('run_id') == _stamp(manifest['started_at']).strftime('%Y%m%dT%H%M%S%z')
                 + '__cnn_aug_'+variant+'_seed2026_epoch100', 'Training start identity changed')
    summary = read_json(folder/'training_summary.json')
    history = read_json(folder/'history.json')
    _require(summary.get('completed') is True and summary.get('epochs_completed') == 100
             and summary.get('max_epochs') == 100 and summary.get('optimizer_steps') == 39000
             and summary.get('initial_parameter_sha256') == INITIAL_HASH
             and summary.get('checkpoint_selection') == 'fixed_epoch_100'
             and summary.get('run_id') == manifest['run_id'] and summary.get('started_at') == manifest['started_at']
             and len(history) == 100 and all(row.get('epoch') == i and row.get('optimizer_steps') == i*390
                 and all(type(row.get(k)) in (int, float) and math.isfinite(row[k])
                         for k in ('train_loss', 'valid_nce')) for i, row in enumerate(history, 1)),
             'Training did not complete exactly 100 epochs and 39,000 steps')
    from wbm_retrieval.training import load_checkpoint, parameter_hash
    checkpoint = load_checkpoint(folder/'checkpoint.pt')
    _require(checkpoint.get('signature') == signature and checkpoint.get('epoch') == 100
             and checkpoint.get('variant') == variant and checkpoint.get('optimizer_steps') == 39000
             and checkpoint.get('initial_parameter_sha256') == INITIAL_HASH
             and checkpoint.get('input_spec') == science.get('input')
             and checkpoint.get('selection') == 'fixed_epoch_100'
             and parameter_hash(checkpoint['model']) == summary.get('final_parameter_sha256'),
             'Final checkpoint is not the bound completed epoch100 model')
    _augmentation_audit(folder, variant, history, summary)
    return folder/'checkpoint.pt'


def _formula(output, model, split, index):
    path = output/'evaluation'/model/f'formula_audit_{split}.json'
    _require(Path(index['formula_audit']).resolve() == path, 'Formula audit path escaped its model stage')
    audit = read_json(path)
    _require(audit.get('status') == 'verified' and audit.get('num_classes') == 37
             and len(set(audit.get('query_ids', []))) == 37
             and set(audit.get('variants', {})) == set(STYLES), 'Both scoring formulas need all 37 classes')
    for detail in audit['variants'].values():
        _require(detail.get('candidate_pairs') == 740, 'Formula audit did not cover 740 pairs per style')
        for name in ('max_raw_formula_absolute_error', 'max_matching_absolute_error'):
            _error(detail.get(name), 2e-6, name)


def _baseline_reference(output, split, root_manifest, inputs, index):
    """Revalidate the canonical historical baseline, without changing its identity."""
    from cnn_refinement.register import validate_model_split as validate_original
    source_root = Path(inputs['baseline_evaluation_root']).resolve()
    source_index = source_root/'evaluation/baseline'/f'evaluation_index_{split}.json'
    science = index.get('reference_science', {})
    original_context=read_json(source_root/'evaluation/baseline'/f'model_context_{split}.json')
    expected = {'parent_context': root_manifest['signature'], 'model': 'baseline', 'split': split,
        'source_binding_sha256': digest(inputs['source_binding']), 'code': code_identity(),
        'historical_index_path': str(source_index), 'historical_index_sha256': file_hash(source_index),
        'historical_root': str(source_root), 'specifications': specifications(),
        'baseline_feature_sha256':original_context['h512_sha256'], 'baseline_token_layout':original_context['token_layout'],
        'historical_validation': 'cnn_refinement.register.validate_model_split; all source seals and complete-pool proof checked'}
    _require(science == expected and index.get('signature') == digest(science)
             and inputs.get('context_signature') == root_manifest['signature']
             and index.get('context_signature') == root_manifest['signature']
             and index.get('model') == 'baseline' and index.get('split') == split
             and inputs['source_binding'].get(str(source_index)) == expected['historical_index_sha256']
             and index.get('new_evaluations') == 0 and index.get('historical_references') == 4,
             'Historical baseline reference context changed')
    original = validate_original(source_root, 'baseline', split)
    _require(set(index.get('evaluations', {})) == set(original['evaluations'])
             == {stage+'/'+style for stage in (split, 'cross_'+split) for style in STYLES},
             'Baseline reference must include all four existing roles')
    for key, prior in original['evaluations'].items():
        folder = Path(prior['path']).resolve()
        marker_hash = prior.get('completed_sha256', prior.get('source_completed_sha256'))
        row = {**prior, 'register': False, 'historical_reference': True,
               'source_completed_sha256': marker_hash,
               'reference_reason': 'canonical baseline role already formally registered'}
        _require(index['evaluations'][key] == row
                 and inputs['source_binding'].get(str(folder/'completed.json')) == marker_hash
                 and file_hash(folder/'completed.json') == marker_hash,
                 'An existing baseline role was changed or assigned a new identity')
    return index


def _augmentation_audit(folder, variant, history=None, summary=None):
    """Recompute intervention exposure from all epoch counters, with no outcome gate."""
    folder = Path(folder)
    history = read_json(folder/'history.json') if history is None else history
    summary = read_json(folder/'training_summary.json') if summary is None else summary
    audit = read_json(folder/'augmentation_audit.json')
    counters = ('augmentation_applied_count', 'augmentation_no_op_count', 'empty_target_before_count',
        'empty_target_after_count', 'newly_empty_target_count', 'target_before_pixels', 'target_after_pixels',
        'target_retained_source_pixels', 'target_retention_eligible_count', 'target_retention_fraction_sum',
        'changed_positions', 'support_before_pixels', 'support_after_pixels', 'support_retained_source_pixels',
        'support_retention_fraction_sum', 'support_retention_eligible_count', 'noise_selected_pixels',
        'noise_added_defects', 'noise_removed_defects', 'noise_changed_pixels')
    _require(len(history) == 100 and all(row.get('sample_count') == 99786 for row in history),
             'Augmentation audit must cover every training token in every epoch')
    for row in history:
        _require(all(type(row.get(k)) in (int, float) and math.isfinite(row[k]) and row[k] >= 0 for k in counters),
                 'Augmentation audit contains missing or invalid counts')
        n = row['sample_count']
        _require(all(row[k] <= n for k in counters if k.endswith('_count'))
                 and row['newly_empty_target_count'] <= row['empty_target_after_count']
                 and row['target_retention_eligible_count'] == n-row['empty_target_before_count']
                 and row['target_retention_fraction_sum'] <= row['target_retention_eligible_count']+1e-8
                 and row['target_retained_source_pixels'] <= row['target_before_pixels']
                 and row['support_retained_source_pixels'] <= row['support_before_pixels']
                 and row['support_retention_fraction_sum'] <= row['support_retention_eligible_count']+1e-8
                 and row['noise_changed_pixels'] == row['noise_added_defects']+row['noise_removed_defects']
                 and row['noise_changed_pixels'] <= row['noise_selected_pixels'],
                 'Augmentation target retention/exposure counters are inconsistent')
    totals = {key: sum(row[key] for row in history) for key in counters}
    samples = 99786*100
    rates = {'applied_fraction':totals['augmentation_applied_count']/samples,
        'no_op_fraction':totals['augmentation_no_op_count']/samples,
        'empty_target_after_fraction':totals['empty_target_after_count']/samples,
        'newly_empty_target_fraction':totals['newly_empty_target_count']/samples,
        'mean_target_retention_fraction':(totals['target_retention_fraction_sum']/totals['target_retention_eligible_count']
            if totals['target_retention_eligible_count'] else None),
        'mean_support_retention_fraction':(totals['support_retention_fraction_sum']/totals['support_retention_eligible_count']
            if totals['support_retention_eligible_count'] else None)}
    _require(audit.get('variant') == variant and audit.get('epochs') == 100
             and audit.get('samples') == samples and audit.get('definitions')
             and audit.get('totals') == totals and summary.get('augmentation_audit') == audit,
             'Augmentation audit is detached from its complete training history')
    _require(set(audit.get('rates', {})) == set(rates)
             and all((audit['rates'][k] is None and v is None)
                     or (v is not None and audit['rates'][k] is not None and abs(audit['rates'][k]-v) <= 1e-12)
                     for k,v in rates.items()), 'Augmentation rates do not match actual counter totals')
    return audit


def _evaluation(output, model, split, stage, style, row, index, inputs, model_context):
    cross = stage.startswith('cross_')
    folder = output/'evaluation'/model/stage/style
    signature = digest({'context': index['signature'], 'stage': stage,
                        'variant': style, 'spec': specifications()[style]})
    reports = {'run_identity.json', 'evaluation_manifest.json', 'rankings.csv',
               'coverage_metrics.json', 'coverage_per_query.csv', 'coverage_per_class.csv'}
    if not cross:
        reports.update(('metrics.json', 'per_query.csv', 'per_class.csv'))
    _complete(folder, signature, reports | {'verification.json'})
    _require(Path(row['path']).resolve() == folder and row.get('signature') == signature
             and row.get('completed_sha256') == file_hash(folder/'completed.json')
             and row.get('register') is True and row.get('historical_reference') is False,
             'Evaluation index is detached from its completed reports')
    identity = read_json(folder/'run_identity.json')
    persistent = read_json(output/'evaluation'/model/'identities'/f'{stage}__{style}.json')
    manifest = read_json(folder/'evaluation_manifest.json')
    fixed = FROZEN[split]
    provider = index['score_cache'][style]
    source = {'signature': provider['signature'], 'completed_sha256': provider['completed_sha256']}
    expected = {'signature': signature, 'context_signature': index['signature'], 'model': model,
        'stage': stage, 'split': split, 'variant': style, 'spec': specifications()[style], 'ks': KS,
        'tie_seed': 2026, 'num_queries': fixed['num_queries'], 'gallery_count': fixed['gallery_count'],
        'protocol': PROTOCOL if cross else 'retrieval_v1',
        'candidate_rule': RULE if cross else 'all_gallery_except_query',
        'retrieval_protocol_signature': fixed['derived_signature'] if cross else BASE_PROTOCOL_SIGNATURE,
        'base_protocol_signature': BASE_PROTOCOL_SIGNATURE, 'provider_source': source,
        'source_binding_sha256': digest(inputs['source_binding']), 'formal_protocol': True,
        'training_required': model != 'baseline', 'test_parameter_search': False, 'single_seed': 2026}
    _require(identity == persistent and identity.get('evaluation_signature') == signature
             and row.get('key') == identity.get('key') and row.get('run_id') == identity.get('run_id')
             and all(manifest.get(k) == v for k, v in {**identity, **expected}.items()),
             'Evaluation identity, checkpoint context or fixed scoring/protocol changed')
    checked = read_json(folder/'verification.json')
    candidate_audit = checked.get('candidate_audit', {})
    pairs = fixed['candidate_pairs'] if cross else fixed['num_queries']*(fixed['gallery_count']-1)
    audit_expected = {k: v for k, v in expected.items() if k in ('split', 'ks', 'num_queries', 'gallery_count',
                       'protocol', 'candidate_rule', 'retrieval_protocol_signature', 'base_protocol_signature')}
    audit_expected.update(status='verified', full_candidates_checked=pairs,
                          query_ids_sha256=fixed['query_ids_sha256'], gallery_ids_sha256=fixed['gallery_ids_sha256'])
    _require(all(candidate_audit.get(k) == v for k, v in audit_expected.items())
             and all(checked.get(k) == v for k, v in {'status': 'verified', 'signature': signature,
                'context_signature': index['signature'], 'stage': stage, 'split': split, 'variant': style,
                'spec': specifications()[style], 'ks': KS, 'provider_source': source,
                'source_binding_sha256': digest(inputs['source_binding']), 'exact_reports': not cross,
                'num_queries': fixed['num_queries'], 'ranking_rows': fixed['num_queries']*500,
                'full_candidates_checked': pairs, 'all_queries_full_candidate_topk_exact': True,
                'ranking_sha256': file_hash(folder/'rankings.csv')}.items())
             and set(checked.get('files', {})) == reports
             and row.get('metric_query_counts') == checked.get('metric_query_counts')
             and all(file_hash(folder/name) == sha for name, sha in checked['files'].items())
             and type(checked.get('aggregation_values_checked')) is int and checked['aggregation_values_checked'] > 0,
             'Independent full-candidate/metric verification is absent, incomplete or stale')
    for name in ('max_metric_absolute_error', 'max_saved_score_absolute_error'):
        _error(checked.get(name), 1e-12, name)
    if cross:
        _require(candidate_audit.get('candidate_identity_sha256') == fixed['candidate_identity_sha256']
                 and checked.get('candidate_identity_sha256') == fixed['candidate_identity_sha256']
                 and manifest.get('derived_protocol', {}).get('candidate_identity_sha256') == fixed['candidate_identity_sha256']
                 and candidate_audit.get('candidate_count_min') == fixed['candidate_count_min']
                 and candidate_audit.get('candidate_count_max') == fixed['candidate_count_max']
                 and not any((folder/n).exists() for n in ('metrics.json', 'per_query.csv', 'per_class.csv')),
                 'Cross-combination identities, complete denominator or report families changed')
    for family in (('coverage',) if cross else ('exact', 'coverage')):
        filename = 'metrics.json' if family == 'exact' else 'coverage_metrics.json'
        report = read_json(folder/filename)
        _metric_values(report.get('metrics'), family)
        _require(report.get('metric_version') == family+'_v1' and report.get('ks') == KS
                 and report.get('split') == split and report.get('num_queries') == fixed['num_queries']
                 and all(report.get('metadata', {}).get(k) == v for k, v in manifest.items())
                 and not report.get('metadata', {}).get('smoke')
                 and row.get('metrics' if family == 'exact' else 'coverage_metrics') == report['metrics']
                 and checked.get('metric_query_counts', {}).get(family) == report.get('metric_query_counts'),
                 'Verified report and indexed metrics/query counts disagree')
        if family == 'coverage':
            _require(report['metric_query_counts'] == candidate_audit['metric_query_counts'],
                     'Coverage denominators differ from full candidate audit')
    if not cross:
        hits = checked.get('top10_hits')
        _require(type(hits) is int and 0 <= hits <= fixed['num_queries']*10
                 and abs(row['metrics']['Precision@10']-hits/(fixed['num_queries']*10)) <= 1e-12,
                 'Precision@10 differs from independently verified integer hit count')
    _require(model_context['signature'] == index['signature'], 'Wrong feature/model context')
    return manifest


def validate_model_split(output, model, split, root_manifest=None):
    """Read-only gate usable before root completion, including before test access."""
    output = Path(output).resolve()
    _require(model in MODELS and split in ('valid', 'test'), 'Unknown model/split')
    root_manifest = _root_context(output) if root_manifest is None else root_manifest
    base = output/'evaluation'/model
    inputs = read_signed(output/f'inputs_{split}.json')
    index = read_json(base/f'evaluation_index_{split}.json')
    if model == 'baseline':
        return _baseline_reference(output, split, root_manifest, inputs, index)
    context = read_json(base/f'model_context_{split}.json')
    _require(inputs.get('context_signature') == root_manifest['signature']
             and index.get('context_signature') == root_manifest['signature']
             and index.get('model') == model and index.get('split') == split
             and context.get('signature') == digest({k: v for k, v in context.items() if k != 'signature'})
             and context.get('signature') == index.get('signature')
             and context.get('parent_context') == root_manifest['signature']
             and context.get('model') == model and context.get('split') == split
             and context.get('source_binding_sha256') == digest(inputs['source_binding'])
             and context.get('code') == code_identity()
             and context.get('specifications') == specifications(), 'Model/split index context changed')
    if model != 'baseline':
        features = output/'features'/model/split
        fm = read_signed(features/'manifest.json')
        _complete(features, fm['feature_signature'], ('manifest.json', 'original.npy', 'verification.json'))
        _require(fm.get('parent_context') == root_manifest['signature']
                 and fm.get('model') == model and fm.get('split') == split and fm.get('views') == ['original']
                 and fm.get('checkpoint_sha256') == file_hash(output/'training'/model/'checkpoint.pt')
                 and fm.get('checkpoint', {}).get('sha256') == fm['checkpoint_sha256']
                 and Path(fm['checkpoint_path']).resolve() == output/'training'/model/'checkpoint.pt'
                 and fm.get('token_layout') == context['token_layout']
                 and file_hash(features/'original.npy') == context['h512_sha256'],
                 'h512 features detached from the epoch100 checkpoint or token order')
    else:
        source = Path(inputs['source_metadata']['region_source'])/'cache'/split/'h512.npy'
        _require(inputs['source_binding'].get(str(source)) == context['h512_sha256'],
                 'Baseline h512 is not the unchanged bound historical cache')
    expected_rows = {stage+'/'+style for stage in (split, 'cross_'+split) for style in STYLES}
    _require(set(index.get('evaluations', {})) == expected_rows
             and set(index.get('score_cache', {})) == set(STYLES), 'Complete four-cell model/split grid required')
    _formula(output, model, split, index)
    for style in STYLES:
        cache = base/'score_cache'/split/style
        entry = index['score_cache'][style]
        metadata = read_json(cache/'metadata.json')
        _complete(cache, entry['signature'], ('metadata.json', 'scores.npy', 'identities.npz'))
        _require(Path(entry['path']).resolve() == cache and entry['completed_sha256'] == file_hash(cache/'completed.json')
                 and metadata.get('signature') == entry['signature']
                 and entry['signature'] == digest({k: v for k, v in metadata.items() if k != 'signature'})
                 and metadata.get('context_signature') == index['signature']
                 and metadata.get('h512_sha256') == context['h512_sha256']
                 and metadata.get('token_layout') == context['token_layout']
                 and metadata.get('spec') == specifications()[style]
                 and metadata.get('dtype') == 'float64' and metadata.get('num_queries') == FROZEN[split]['num_queries']
                 and metadata.get('gallery_count') == FROZEN[split]['gallery_count']
                 and metadata.get('source_cache_sha256') == inputs['source_binding'].get(
                     str(Path(inputs['source_metadata']['region_source'])/'cache'/split/'completed.json'))
                 and metadata.get('retrieval_protocol_signature') == BASE_PROTOCOL_SIGNATURE,
                 'Complete score cache/source feature binding changed')
    new_count = 0
    for key, row in index['evaluations'].items():
        stage, style = key.split('/')
        _require(row.get('model') == model and row.get('style') == style and row.get('stage') == stage
                 and row.get('split') == split, 'Indexed evaluation role changed')
        manifest = _evaluation(output, model, split, stage, style, row, index, inputs, context)
        _require(_stamp(manifest['started_at']) >= _stamp(root_manifest['started_at']),
                 'Evaluation start predates the prespecified plan')
        new_count += 1
    _require(index.get('new_evaluations') == new_count, 'New-result count includes historical replay duplicates')
    return index


def _valid_gate(output, model, context_signature):
    path = output/'evaluation'/model/'evaluation_index_valid.json'
    index = read_json(path)
    gate = read_signed(output/f'valid_gate_{model}.json')
    expected = {'status': 'verified', 'model': model, 'context_signature': context_signature,
        'evaluation_index_sha256': file_hash(path),
        'completed_reports': {key: file_hash(Path(row['path'])/'completed.json')
                              for key, row in index['evaluations'].items()},
        'frozen_plan_sha256': file_hash(output/'frozen_plan.json'),
        'gate': 'implementation and protocol only; no performance threshold or selection'}
    _require(all(gate.get(k) == v for k, v in expected.items()),
             'Test gate is detached from all four verified validation roles')


def _auxiliary(output, context_signature):
    """Require all requested comparisons, actual augmentation exposure and figures."""
    folder = output/'report'
    meta = read_json(folder/'manifest.json')
    signature = meta.get('signature')
    _require(signature == digest({k:v for k,v in meta.items() if k != 'signature'})
             and meta.get('context_signature') == context_signature
             and meta.get('decision_rules') == PLAN['decision_rules']
             and meta.get('bootstrap') == 2000 and meta.get('seed') == 2026,
             'Final report is detached from prespecified interpretation rules')
    required = ('manifest.json', 'summary.json', 'paper_display.json', 'all_metrics.csv',
        'paper_main_metrics.csv', 'paired_retrieval_differences.csv', 'training_history.csv',
        'augmentation_audit.csv')
    marker = _complete(folder, signature, required)
    summary = read_json(folder/'summary.json')
    _require(summary.get('signature') == signature and summary.get('context_signature') == context_signature
             and set(summary.get('evaluations', {})) == {
                 '/'.join((model,stage,style)) for model in MODELS for stage in STAGES for style in STYLES}
             and summary.get('new_evaluations') == meta.get('new_evaluations') == 40
             and summary.get('historical_references') == meta.get('historical_references') == 8
             and summary.get('paper_display') == read_json(folder/'paper_display.json')
             and set(summary.get('augmentation_audits', {})) == set(VARIANTS),
             'Final report omits evaluation roles, original references or augmentation audit')
    for variant in VARIANTS:
        _require(summary['augmentation_audits'][variant] == _augmentation_audit(output/'training'/variant, variant),
                 'Reported augmentation exposure differs from training')
    figures = summary.get('figures', {})
    _require(set(figures) == {'retrieval_test', 'retrieval_valid', 'augmentation_exposure', 'training_nce'},
             'Requested figure set is incomplete')
    for outputs in figures.values():
        _require(isinstance(outputs,list) and len(outputs) == 3
                 and {Path(path).suffix for path in outputs} == {'.png','.pdf','.svg'},
                 'Each figure needs PNG, PDF and SVG')
        for path in outputs:
            _require(_relative(Path(path), folder) in marker['files'], 'Figure is not bound by completion seal')
    for binding in (meta.get('source_binding',{}), meta.get('code',{})):
        _bound_files(binding)


def _root_seal(output, signature, required):
    marker = _complete(output, signature, required)
    _require(marker.get('excluded', []) in ([], ['logs/']), 'Root seal can exclude only operational logs/')
    # Catalogue proposals/backups are administrative products created AFTER
    # this seal; all scientific files must already be bound by it.
    scientific = {str(path.relative_to(output)) for path in output.rglob('*') if path.is_file()
        and path != output/'completed.json' and path != output/'catalog_sources.json'
        and not path.is_relative_to(output/'catalog_backup')
        and not (marker.get('excluded') == ['logs/'] and path.is_relative_to(output/'logs'))}
    _require(scientific <= set(marker['files']), 'Root completion omits scientific artifacts')
    return marker


def verify_run(output, *, root=ROOT, check_source=True, require_root_seal=True):
    output, root = Path(output).resolve(), Path(root).resolve()
    _require(output != root/'results' and output.is_relative_to(root/'results'),
             'New experiments require an isolated results/ root')
    manifest = _root_context(output)
    required = {'run_manifest.json', 'frozen_plan.json', 'paper_source.json', 'source_protection_before.json',
        'source_protection_after.json', 'inputs_valid.json', 'inputs_test.json', 'report/completed.json',
        'augmentation_audit/completed.json'}
    required.update(f'evaluation/{model}/evaluation_index_{split}.json' for model in MODELS for split in ('valid','test'))
    required.update(f'valid_gate_{model}.json' for model in MODELS)
    if require_root_seal:
        _root_seal(output, manifest['signature'], required)
    else:
        _require(all((output/name).is_file() for name in required), 'Complete scientific artifacts required before root sealing')
    for model in VARIANTS:
        _training(output, model, manifest)
    for model in MODELS:
        for split in ('valid','test'):
            validate_model_split(output, model, split, manifest)
        _valid_gate(output, model, manifest['signature'])
    audit=read_signed(output/'augmentation_audit/manifest.json')
    _complete(output/'augmentation_audit',audit['signature'],('manifest.json','summary.json','summary.csv','cohort.csv','per_token.csv'))
    checked=read_json(output/'augmentation_audit/summary.json')
    _require(checked.get('signature')==audit['signature'] and checked.get('context_signature')==manifest['signature']
             and set(checked.get('models',{}))==set(VARIANTS), 'Input augmentation audit is incomplete or detached')
    _auxiliary(output, manifest['signature'])
    before = read_json(output/'source_protection_before.json')
    after = read_json(output/'source_protection_after.json')
    _require(after.get('status') == 'unchanged' and after.get('files') == len(before), 'Historical source protection failed')
    if check_source:
        binding = dict(manifest['source_binding'])
        for split in ('valid','test'):
            for path,sha in read_signed(output/f'inputs_{split}.json')['source_binding'].items():
                _require(path not in binding or binding[path] == sha, 'Inconsistent historical source bindings')
                binding[path] = sha
        _bound_files(binding)
    return manifest


def proposal(output, *, root=ROOT):
    output, root = Path(output).resolve(), Path(root).resolve()
    verify_run(output, root=root)
    items = []
    for model in MODELS:
        for split in ('valid', 'test'):
            index = read_json(output/'evaluation'/model/f'evaluation_index_{split}.json')
            for row in index['evaluations'].values():
                if row['register'] is False:
                    continue
                stage, style = row['stage'], row['style']
                folder = Path(row['path'])
                identity = read_json(folder/'run_identity.json')
                cross = stage.startswith('cross_')
                supporting = ['run_manifest.json', 'completed.json', 'frozen_plan.json',
                    'source_protection_after.json', f'inputs_{split}.json', 'report/manifest.json',
                    f'evaluation/{model}/evaluation_index_{split}.json',
                    f'evaluation/{model}/model_context_{split}.json', f'evaluation/{model}/formula_audit_{split}.json',
                    f'evaluation/{model}/{stage}/{style}/verification.json']
                if model != 'baseline':
                    supporting += [f'training/{model}/run_manifest.json', f'training/{model}/completed.json',
                                   f'training/{model}/training_summary.json', f'features/{model}/{split}/manifest.json']
                item = {'key': identity['key'], 'repository': 'wbm-region-retrieval', 'path': _relative(folder, root),
                    'category': 'cross_combination' if cross else 'fusion_ablation', 'method': 'F',
                    'variant': 'CNNAugmentation/'+model+'/'+style, 'split': split,
                    'manifest': _relative(folder/'evaluation_manifest.json', root),
                    'marker': _relative(folder/'completed.json', root), 'identity_schema_version': 1,
                    'identity': _relative(folder/'run_identity.json', root), 'selection': None, 'coverage': None,
                    'supervision': 'self_supervised', 'display': 'CNN单项增强 '+model+' '+style+('（跨组合）' if cross else ''),
                    'required_families': ['coverage'] if cross else ['exact', 'coverage'],
                    'protocol': PROTOCOL if cross else 'retrieval_v1', 'posthoc': True,
                    'supporting_manifests': [_relative(output/name, root) for name in supporting],
                    'design_note': 'Five prespecified single augmentation modes from Kahng and Kim2021 author implementation, '
                        'adapted to the unchanged three-channel fixed regional input. Each starts from original seed2026 '
                        'initial parameters, runs exactly100epochs/39000steps and uses fixed epoch100; baseline90degree '
                        'rotation is reused without retraining. No labels, negative filtering, best-loss checkpoint selection '
                        'or test tuning. CNN-only and unchanged regional fusion are both evaluated on full ordinary/cross pools. '
                        'All baseline roles retain previously registered identities. Target erasure/retention exposure is reported; '
                        'classification findings are hypotheses for region retrieval, not a reproduced classification claim. '
                        'Posthoc single-seed evidence with conditional query intervals only.'}
                catalog.execution_identity(root, item, folder/'completed.json', read_json(folder/'evaluation_manifest.json'))
                items.append(item)
    _require(len(items) == 40 and len({item['key'] for item in items}) == len(items),
             'Expected 40 new roles and eight canonical baseline references')
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
    """Collect before mutation; retain every pre-existing catalogue object verbatim."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    output = (ROOT/args.run_root).resolve()
    result = apply_proposal(output) if args.apply else {'proposal': str(write_proposal(output))}
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
