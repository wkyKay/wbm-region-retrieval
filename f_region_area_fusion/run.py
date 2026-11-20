#!/usr/bin/env python3
"""Evaluate 29 valid recipes, then only their ten frozen selections on test."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import sys

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from f_region_score_test.run import code_identity as upstream_code_identity
from f_region_score_search.run import check_sources, run_lock
from wbm_retrieval.common import digest, file_hash, write_json
from . import inputs, pipeline, scoring, selection
from .state import now, protection, read_signed, record, seal, signed, verified

VERSION = 'f_region_area_fusion_v1'


def code_identity():
    result = upstream_code_identity()
    # The registrar is also the complete-grid pre-test validation gate.
    for path in sorted(Path(__file__).parent.glob('*.py')):
        result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('valid', 'test'), default='valid')
    p.add_argument('--source-run', help='Valid: old descriptor run. Test: newly sealed 29-way valid run.')
    out = p.add_mutually_exclusive_group()
    out.add_argument('--output-root')
    out.add_argument('--resume-root')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--check-inputs', action='store_true')
    return p


def output_path(args):
    output = (ROOT/(args.resume_root or args.output_root or
              ('artificial/f_region_area_fusion_'+args.stage+'_'+now().strftime('%Y%m%dT%H%M%S%z')))).resolve()
    parent = (ROOT/'artificial').resolve()
    if not output.is_relative_to(parent) or output == parent:
        raise ValueError('Output must be an isolated new artificial/ directory')
    for ancestor in output.parents:
        if ancestor == parent:
            break
        if (ancestor/'run_manifest.json').exists():
            raise ValueError('Do not nest inside an old experiment')
    if args.resume_root:
        old = read_signed(output/'run_manifest.json')
        if old.get('version') != VERSION or old.get('requested_stage') != args.stage:
            raise ValueError('Resume requires the same area-fusion stage and signed manifest')
    elif output.exists():
        raise ValueError('Refuse an existing output; explicit --resume-root is required')
    return output


def baseline_replay(output, split, old_folder):
    old = Path(old_folder)/'rankings.csv'
    new = Path(output)/split/inputs.BASELINE/'rankings.csv'
    if file_hash(new) != file_hash(old):
        raise ValueError('Baseline must reproduce the old equal-weight rankings byte for byte')
    with new.open(newline='') as handle:
        rows = sum(1 for _ in csv.DictReader(handle))
    if rows != (500 if split == 'valid' else 7199)*500:
        raise ValueError('Baseline must cover every query and Top-500')
    record(Path(output)/'baseline_replay.json', {'status': 'verified', 'split': split,
        'variant': inputs.BASELINE, 'rows': rows, 'max_score_error': 0., 'source': str(old),
        'source_sha256': file_hash(old), 'ranking_byte_identical': True})


def freeze_selection(output, signature, results):
    output = Path(output)
    value = selection.select_top10(results)
    payload = {**value, 'context_signature': signature,
               'valid_index_sha256': file_hash(output/'evaluation_index_valid.json')}
    path = output/'selection.json'
    if path.exists():
        previous = read_signed(path)
        if any(previous.get(k) != v for k, v in payload.items()):
            raise ValueError('Existing valid Top-10 selection changed')
        return previous
    value = signed({**payload, 'frozen_at': now().isoformat()})
    record(path, value)
    return value


def summarize(output, split, results, selected):
    ranks = {row['variant']: row['rank'] for row in selected['ranking']}
    with (Path(output)/f'comparison_{split}.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['variant', 'split', 'valid_rank', 'selected_for_test', 'top10_hits',
                         'Precision@10', 'mAP@100', 'Coverage-nDCG@10'])
        for name in sorted(results, key=ranks.__getitem__):
            row = results[name]
            writer.writerow([name, split, ranks[name], name in selected['selected_variants'],
                row['top10_hits'], row['metrics']['Precision@10'], row['metrics']['mAP@100'],
                row['coverage_metrics']['Coverage-nDCG@10']])


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1 <= args.workers <= 8:
        raise ValueError('workers must be between 1 and 8')
    output = output_path(args)
    source_arg = args.source_run
    if not source_arg and args.resume_root:
        source_arg = read_signed(output/'run_manifest.json')['source_run']
    if not source_arg and args.stage == 'test':
        raise ValueError('Test requires --source-run pointing to the complete new valid run')
    source = (ROOT/(source_arg or inputs.DEFAULT_SOURCE)).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Source and output must be isolated')
    if args.dry_run:
        print(json.dumps({'output': str(output), 'source_run': str(source), 'split': args.stage,
            'full_grid': scoring.frozen_grid(), 'selection_rule': selection.RULE, 'training': False,
            'test_parameter_search': False, 'workers': args.workers,
            'test_policy': 'Test reads exactly ten variants from the sealed valid selection.'}, indent=2))
        return 0
    data = inputs.load_valid(source) if args.stage == 'valid' else inputs.load_test(source)
    pool = getattr(data['protocol'].retrieval, args.stage)
    frozen = data.get('frozen_selection')
    # JSON canonicalization may reorder mapping keys; the signed list defines order.
    specs = (scoring.frozen_grid() if args.stage == 'valid' else
             {name: frozen['selected_specs'][name] for name in frozen['selected_variants']})
    if args.check_inputs:
        print(json.dumps({'status': 'verified_inputs', 'split': args.stage, 'num_queries': len(pool),
            'gallery': len(pool.gallery), 'source_files': len(data['source_binding']), 'variants': list(specs),
            'training': False, 'test_parameter_search': False}))
        return 0
    context = {'version': VERSION, 'split': args.stage, 'grid': specs, 'full_grid': scoring.frozen_grid(),
        'selection_rule': selection.RULE, 'source_run': str(source),
        'cache_source_run': str(data['cache_source_run']), 'source_binding': data['source_binding'],
        'code': code_identity(), 'matching': data['config']['matching'], 'numpy': np.__version__,
        'torch': torch.__version__, 'posthoc_design': True, 'training': False,
        'test_parameter_search': False, 'frozen_selection_signature': frozen['signature'] if frozen else None,
        'literature_basis': {'entropy': '10.1016/j.cie.2024.110486 section 3.5; regional adaptation',
                            'weighted_fusion': 'literature-guided design; numeric grid is project-specific'},
        'scale_policy': 'unchanged C, T=exp(-0.5H), A=exp(-Darea/1.5); no fitted calibration',
        'evaluation_policy': 'All 29 valid results select ten; test ranks never select or tune a recipe.'}
    signature = digest(context)
    if args.resume_root:
        previous = read_signed(output/'run_manifest.json')
        if previous.get('context') != context:
            raise ValueError('Settings, sources, scientific code or numerical environment changed; use a new run')
    else:
        output.mkdir(parents=True, exist_ok=False)
    with run_lock(output):
        if args.resume_root and verified(output, signature):
            print('Already completed and verified: '+str(output), flush=True)
            return 0
        if not args.resume_root:
            time = now()
            record(output/'run_manifest.json', signed({'version': VERSION, 'context': context,
                'started_at': time.isoformat(), 'run_id': time.strftime('%Y%m%dT%H%M%S%z')+'__f_region_area_fusion_'+args.stage,
                'source_run': str(source), 'workers': args.workers, 'requested_stage': args.stage}))
        record(output/'inputs.json', {'source_binding': data['source_binding'], 'context_signature': signature,
            'source_run': str(source), 'cache_source_run': str(data['cache_source_run']),
            'baseline_folder': str(data['baseline_folder']) if data['baseline_folder'] else None})
        if frozen:
            record(output/'frozen_selection.json', frozen)
        protection(output)
        print('OUTPUT_ROOT='+str(output), flush=True)
        results = pipeline.evaluate(output, data['cache'], pool, data['bundle'], data['config']['matching'],
                                    signature, specs, args.workers, frozen_selection=frozen)
        if data['baseline_folder']:
            baseline_replay(output, args.stage, data['baseline_folder'])
        selected = freeze_selection(output, signature, results) if args.stage == 'valid' else frozen['selection']
        summarize(output, args.stage, results, selected)
        check_sources(data['source_binding'])
        for name, expected in context['code'].items():
            if file_hash(ROOT/name) != expected:
                raise ValueError('Scientific code changed during execution: '+name)
        protection(output, after=True)
        write_json(output/'status.json', {'status': 'completed', 'split': args.stage,
            'test_evaluated': args.stage == 'test', 'winner': selected['winner'],
            'selected_variants': selected['selected_variants'], 'context_signature': signature})
        seal(output, signature)
        print('VALID TOP TEN: '+', '.join(selected['selected_variants']), flush=True)
        print('COMPLETED: '+str(output), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
