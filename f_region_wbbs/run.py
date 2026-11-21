"""Freeze cross-valid regional WBBS selection, then evaluate the untouched test pool."""
from __future__ import annotations
import argparse
import csv
from itertools import groupby
import json
import os
from pathlib import Path
import sys

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
import numpy as np
import scipy
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from . import inputs, prepare, counts, scoring, selection, pipeline
from .protocol import make_cross_pool, audit_pool
from .state import (ROOT, VERSION, KS, SETTINGS, SELECTION_RULE, BASELINE, GLOBAL, grid,
    evaluation_specs, evaluation_signature, now, record, signed, read_signed, seal, verified, protection)

CROSS_J = ROOT/'artificial/f_region_j_cross_test_20261120T202638+0800/test/add_a0p4_b0p25'
GLOBAL_TEST = ROOT/'artificial/fgj_cross_test_20261120T193642+0800/test/WBBS'


def code_identity():
    from .compat import ARCHIVED_PATH
    paths = set(Path(__file__).parent.glob('*.py'))
    paths.discard(Path(__file__).with_name('register.py'))
    paths.add(ROOT/ARCHIVED_PATH)
    for folder in ('f_region_evidence', 'f_region_structure', 'f_region_score_search',
                   'f_region_score_test', 'f_region_area_fusion', 'f_region_j_cross_test',
                   'fgj_cross_test', 'fgw_grid_search', 'fgw_cross_test', 'fgw_fusion',
                   'wbm_retrieval', 'experiments/proposed/_shared',
                   'experiments/proposed/f_segmentation', 'experiments/proposed/f_segmentation_geometry',
                   'experiments/proposed/f_global_fusion', 'b1_structure_fusion'):
        paths.update((ROOT/folder).glob('*.py'))
    paths.add(ROOT/'b1_structure_fusion/config.json')
    for folder in ('methods/wbbs', 'shared'):
        paths.update((ROOT.parent/'compared_methods'/folder).glob('*.py'))
    return {os.path.relpath(p, ROOT): file_hash(p) for p in sorted(paths)}


def first_per_class(pool, bundle):
    selected = {}
    for q in pool:
        selected.setdefault(int(bundle.class_ids[q]), int(q))
    return list(selected.values())


def prepare_stage(output, data, split, ctx, specs, args):
    from .verify import audit_formulas
    pool = getattr(data['protocol'].retrieval, split)
    record(output/f'protocol_audit_{split}.json', audit_pool(pool, data['bundle'].labels))
    cross = make_cross_pool(pool, data['bundle'].labels)
    record(output/f'protocol_audit_cross_{split}.json', audit_pool(cross, data['bundle'].labels))
    cache = prepare.prepare(output, data['maps'], data['protocol'], data['cache'], data['old_labels'],
        ctx, data['source_binding'], workers=args.prepare_workers, split=split,
        audit_ids=first_per_class(pool, data['bundle']))
    phase(output, split+'_counts')
    matrices, tokens = counts.create(output, cache, pool, ctx, workers=args.workers)
    provider = scoring.Provider(cache, pool, matrices, tokens)
    record(output/f'formula_audit_{split}.json', audit_formulas(provider, data['bundle'], specs))
    phase(output, split+'_full_scores')
    scores = pipeline.compute_scores(output, cache, pool, ctx, specs, args.score_workers)
    pipeline.replay_baseline(output, scores[BASELINE], data['baseline_folder'], split)
    return pool, cross, provider, scores


def freeze(output, ctx, results):
    chosen = selection.select(results)
    payload = {**chosen, 'context_signature': ctx,
        'valid_index_sha256': file_hash(output/'evaluation_index_valid.json'),
        'cross_valid_index_sha256': file_hash(output/'evaluation_index_cross_valid.json')}
    if (output/'selection.json').exists():
        value = read_signed(output/'selection.json')
        if any(value.get(k) != v for k, v in payload.items()):
            raise ValueError('Frozen cross-valid selection changed')
    else:
        value = signed({**payload, 'frozen_at': now().isoformat()})
    record(output/'selection.json', value)
    record(output/'valid_completed.json', {'status': 'completed', 'context_signature': ctx,
        'selection_sha256': file_hash(output/'selection.json'),
        'indexes': {stage: file_hash(output/f'evaluation_index_{stage}.json') for stage in ('valid', 'cross_valid')},
        'evaluations': {stage: {name: file_hash(output/stage/name/'completed.json')
            for name in evaluation_specs(stage)} for stage in ('valid', 'cross_valid')}})
    return value


def frozen_selection(output, ctx):
    marker = read_json(output/'valid_completed.json')
    chosen = read_signed(output/'selection.json')
    if (marker.get('status') != 'completed' or marker.get('context_signature') != ctx
            or chosen.get('context_signature') != ctx
            or marker.get('selection_sha256') != file_hash(output/'selection.json')):
        raise ValueError('No unchanged complete frozen valid selection')
    for stage in ('valid', 'cross_valid'):
        path = output/f'evaluation_index_{stage}.json'
        index = read_json(path)
        if (index.get('context_signature') != ctx or index.get('stage') != stage
                or index.get('specs') != evaluation_specs(stage)
                or marker['indexes'][stage] != file_hash(path)
                or chosen[stage+'_index_sha256'] != file_hash(path)
                or set(index['results']) != set(evaluation_specs(stage))):
            raise ValueError('Incomplete or changed valid comparison index')
        for name in evaluation_specs(stage):
            folder = output/stage/name
            if (not verified(folder, evaluation_signature(ctx, stage, name))
                    or marker['evaluations'][stage][name] != file_hash(folder/'completed.json')
                    or index['results'][name]['completed_sha256'] != file_hash(folder/'completed.json')):
                raise ValueError('Changed valid evaluation used for selection')
    expected = selection.select(read_json(output/'evaluation_index_cross_valid.json')['results'])
    if any(chosen.get(k) != v for k, v in expected.items()):
        raise ValueError('Selection did not follow the prespecified cross-valid rule')
    return chosen


def summarize(output, stage, results):
    names = list(results)
    exact = list(results[names[0]].get('metrics', {}))
    cov = list(results[names[0]]['coverage_metrics'])
    with (output/f'comparison_{stage}.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['variant', 'stage', *exact, *cov])
        for name, row in results.items():
            writer.writerow([name, stage, *[row['metrics'][k] for k in exact], *[row['coverage_metrics'][k] for k in cov]])


def explanations(output, stage, provider, bundle, specs):
    wanted = set(first_per_class(provider.pool, bundle))
    for name, spec in specs.items():
        examples = []
        with (output/stage/name/'rankings.csv').open(newline='') as stream:
            for q, records in groupby(csv.DictReader(stream), key=lambda r: int(r['query_id'])):
                if q not in wanted:
                    continue
                rows = list(records)
                parts = provider.parts(q)
                for row in (rows[0], rows[9], rows[99]):
                    item = provider.explain(q, int(row['candidate_id']), spec, parts)
                    if abs(item['score']-float(row['similarity_score'])) > 1e-12:
                        raise ValueError('Regional explanation did not reconstruct the saved score')
                    examples.append({**item, 'rank': int(row['rank']),
                        'query_labels': bundle.labels[q].tolist(),
                        'candidate_labels': bundle.labels[int(row['candidate_id'])].tolist()})
        record(output/'explanations'/stage/name/'examples.json', {
            'selection': 'first fixed query per complete combination; ranks 1,10,100',
            'labels_only_used_for_reporting': True, 'region_masks': str(provider.cache['folder']/'region_labels.npy'),
            'region_trace': str(provider.cache['folder']/'region_trace.jsonl'), 'examples': examples})


def phase(output, name, **extra):
    write_json(output/'status.json', {'status': 'running', 'phase': name, 'at': now().isoformat(), **extra})
    print('PHASE '+name, flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('valid', 'test', 'all'), default='all')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--output-root')
    group.add_argument('--resume-root')
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--score-workers', type=int, default=16)
    p.add_argument('--eval-workers', type=int, default=8)
    p.add_argument('--prepare-workers', type=int, default=4)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--check-inputs', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if (not 1 <= args.workers <= 48 or not 1 <= args.score_workers <= 32
            or not 1 <= args.eval_workers <= 16 or not 0 <= args.prepare_workers <= 16):
        raise ValueError('Invalid worker count')
    output = (ROOT/(args.resume_root or args.output_root or 'artificial/f_region_wbbs_'+now().strftime('%Y%m%dT%H%M%S%z'))).resolve()
    if not output.is_relative_to(ROOT/'artificial') or output == ROOT/'artificial':
        raise ValueError('Use a new artificial/ subdirectory')
    for parent in output.parents:
        if parent == ROOT/'artificial':
            break
        if (parent/'run_manifest.json').exists():
            raise ValueError('Do not nest a run inside a historical experiment')
    if args.stage == 'test' and not args.resume_root:
        raise ValueError('Test requires --resume-root with completed frozen valid selection')
    if args.dry_run:
        print(json.dumps({'output': str(output), 'grid': grid(), 'settings': SETTINGS,
            'selection_rule': SELECTION_RULE, 'training_required': False, 'posthoc_research': True}, indent=2))
        return 0
    if any(os.environ.get(k) != '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('Set OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1 for original numeric replay')
    print('Loading frozen valid Top-4/J/CNN/protocol and global WBBS cache', flush=True)
    data = inputs.load('valid')
    source_binding = data['source_binding']
    context = {'version': VERSION, 'grid': grid(), 'settings': SETTINGS, 'selection_rule': SELECTION_RULE,
        'source_binding': source_binding, 'source_compatibility': data['source_compatibility'],
        'code': code_identity(), 'numpy': np.__version__, 'scipy': scipy.__version__,
        'ks': KS, 'cnn_training': False, 'cnn_forward_passes': 0, 'posthoc_research': True,
        'test_parameter_search': False, 'matching': scoring.MATCHING, 'score_dtype': 'float64; unchanged source feature arithmetic',
        'seed': 2026, 'region_count': 'unchanged Top-4', 'structure': 'WBBS original full-wafer TV and radial normalization, then frozen region masks'}
    ctx = digest(context)
    if args.check_inputs:
        print(json.dumps({'status': 'verified', 'source_files': len(source_binding), 'context_signature': ctx}, indent=2))
        return 0
    if args.resume_root:
        manifest = read_signed(output/'run_manifest.json')
        if manifest['context'] != context:
            raise ValueError('Resume source/scientific code/environment changed')
    else:
        if output.exists():
            raise FileExistsError('Refuse existing output; explicit unchanged signed resume only')
        output.mkdir(parents=True)
        stamp = now()
        record(output/'run_manifest.json', signed({'version': VERSION, 'context': context,
            'started_at': stamp.isoformat(), 'run_id': stamp.strftime('%Y%m%dT%H%M%S%z')+'__f_region_wbbs'}))
        record(output/'settings.json', SETTINGS)
        record(output/'inputs_valid.json', {'context_signature': ctx, 'source_binding': source_binding,
            'source_compatibility': data['source_compatibility']})
        record(output/'source_compatibility_valid.json', {'context_signature': ctx,
            'source_compatibility': data['source_compatibility']})
    if verified(output, ctx):
        print('Run already complete and verified', flush=True)
        return 0
    protection(output)
    try:
        if not (output/'valid_completed.json').exists():
            if args.stage == 'test':
                raise ValueError('No frozen complete valid selection')
            phase(output, 'valid_prepare')
            pool, cross, provider, scores = prepare_stage(output, data, 'valid', ctx, grid(), args)
            inputs.export_global(output, data['global_provider'], pool, ctx)
            for stage, candidates in [('valid', pool), ('cross_valid', cross)]:
                phase(output, stage+'_evaluation')
                results = pipeline.evaluate(output, stage, evaluation_specs(stage), candidates, data['bundle'], ctx,
                    source_binding, workers=args.eval_workers)
                summarize(output, stage, results)
            chosen = selection.select(results)
            for stage in ('valid', 'cross_valid'):
                explanations(output, stage, provider, data['bundle'], {BASELINE: grid()[BASELINE], **chosen['selected_specs']})
            chosen = freeze(output, ctx, results)
            print('VALID FROZEN: '+json.dumps(chosen, ensure_ascii=False), flush=True)
        chosen = frozen_selection(output, ctx)
        if args.stage == 'valid':
            write_json(output/'status.json', {'status': 'valid_completed', 'winner': chosen['winner'], 'test_pending': True})
            return 0
        # No test features, rankings, or metrics are read before the frozen valid gate above.
        phase(output, 'test_inputs', winner=chosen['winner'])
        data = inputs.load('test')
        binding = data['source_binding']
        for folder in (CROSS_J, GLOBAL_TEST):
            inputs.bind_sealed(folder, binding)
        record(output/'inputs_test.json', {'context_signature': ctx, 'source_binding': binding,
            'source_compatibility': data['source_compatibility'],
            'selection_sha256': file_hash(output/'selection.json')})
        record(output/'source_compatibility_test.json', {'context_signature': ctx,
            'source_compatibility': data['source_compatibility'],
            'selection_sha256': file_hash(output/'selection.json')})
        record(output/'test_context.json', signed({'context_signature': ctx, 'selection_sha256': file_hash(output/'selection.json'),
            'source_binding_sha256': digest(binding), 'specs': chosen['selected_specs']}))
        specs = {BASELINE: grid()[BASELINE], **chosen['selected_specs']}
        pool, cross, provider, scores = prepare_stage(output, data, 'test', ctx, specs, args)
        pipeline.replay_baseline(output, scores[BASELINE], CROSS_J, 'cross_test', pool=cross)
        for stage, candidates in [('test', pool), ('cross_test', cross)]:
            phase(output, stage+'_evaluation')
            results = pipeline.evaluate(output, stage, chosen['selected_specs'], candidates, data['bundle'], ctx,
                binding, chosen, args.eval_workers)
            summarize(output, stage, results)
            explanations(output, stage, provider, data['bundle'], chosen['selected_specs'])
        record(output/'historical_test_controls.json', {'regional_j': str(data['baseline_folder']),
            'regional_j_cross': str(CROSS_J), 'global_wbbs_cross': str(GLOBAL_TEST),
            'comparison_scope': 'unchanged frozen full-gallery and cross-combination protocols; original sealed controls reused',
            'regional_j_replays': ['baseline_replay_test.json', 'baseline_replay_cross_test.json']})
        protection(output, after=True)
        if code_identity() != context['code']:
            raise ValueError('Scientific code changed while this run was active')
        for path, sha in {**source_binding, **binding}.items():
            if file_hash(path) != sha:
                raise ValueError('Frozen source changed: '+path)
        write_json(output/'status.json', {'status': 'completed', 'winner': chosen['winner'],
            'best_regional_wbbs': chosen['best_regional_wbbs'], 'completed_at': now().isoformat()})
        seal(output, ctx)
        if not verified(output, ctx):
            raise ValueError('Root completion failed verification')
        print('COMPLETED '+str(output), flush=True)
        return 0
    except Exception:
        marker = output/'completed.json'
        if marker.exists():
            marker.rename(output/('failed_completed_'+now().strftime('%Y%m%dT%H%M%S%z')+'.json'))
        write_json(output/'status.json', {'status': 'failed', 'at': now().isoformat()})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
