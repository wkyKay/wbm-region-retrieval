#!/usr/bin/env python3
"""Frozen regional scores, a finite valid-only search, and no test execution."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import csv
import fcntl
import json
import os
from pathlib import Path
import sys

# Fix numerical thread counts before importing NumPy or spawned workers.
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from f_region_structure.run import code_identity as original_code_identity
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from . import inputs, pipeline, scoring, selection
from .state import now, protection, read_signed, record, seal, signed, verified

VERSION = 'f_region_score_search_v1'
DEFAULT_SOURCE = 'artificial/f_region_structure_20261120T120331+0800'


def code_identity():
    result = original_code_identity()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        if path.name != 'register.py':
            result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', default=DEFAULT_SOURCE)
    out = p.add_mutually_exclusive_group()
    out.add_argument('--output-root')
    out.add_argument('--resume-root')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--check-inputs', action='store_true')
    return p


def output_path(args):
    output = (ROOT/(args.resume_root or args.output_root or
              ('artificial/f_region_score_search_'+now().strftime('%Y%m%dT%H%M%S%z')))).resolve()
    parent = (ROOT/'artificial').resolve()
    if not output.is_relative_to(parent) or output == parent:
        raise ValueError('Output must be an isolated new artificial/ directory')
    for ancestor in output.parents:
        if ancestor == parent:
            break
        if (ancestor/'run_manifest.json').exists():
            raise ValueError('Do not nest inside an old experiment')
    if args.resume_root:
        if not (output/'run_manifest.json').is_file():
            raise ValueError('Resume requires a signed run manifest')
    elif output.exists():
        raise ValueError('Refuse an existing output; explicit --resume-root is required')
    return output


@contextmanager
def run_lock(output):
    with (Path(output)/'.run.lock').open('a+b') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another process owns this run') from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def check_sources(binding):
    for name, expected in binding.items():
        if file_hash(Path(name)) != expected:
            raise ValueError('Frozen source changed: '+name)


def baseline_replay(output, old_folder):
    old = Path(old_folder)/'rankings.csv'
    new = Path(output)/'valid/product/rankings.csv'
    if file_hash(new) != file_hash(old):
        raise ValueError('Current product baseline must reproduce old rankings byte for byte')
    with new.open(newline='') as handle:
        reader = csv.DictReader(handle)
        rows = sum(1 for _ in reader)
    if rows != 250000:
        raise ValueError('Baseline must cover 500 fixed queries and Top-500')
    record(Path(output)/'baseline_replay.json', {'status': 'verified', 'rows': rows,
        'max_score_error': 0., 'source': str(old), 'source_sha256': file_hash(old),
        'ranking_byte_identical': True})


def summarize(output, results):
    with (Path(output)/'comparison_valid.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['variant', 'split', 'top10_hits', 'Precision@10', 'mAP@100', 'Coverage-nDCG@10'])
        for name, row in results.items():
            writer.writerow([name, 'valid', row['top10_hits'], row['metrics']['Precision@10'],
                             row['metrics']['mAP@100'], row['coverage_metrics']['Coverage-nDCG@10']])


def freeze_selection(output, context_signature, results):
    output = Path(output)
    selected = selection.select_best(results)
    payload = {**selected, 'context_signature': context_signature,
               'valid_index_sha256': file_hash(output/'evaluation_index_valid.json')}
    path = output/'selection.json'
    if path.exists():
        existing = read_signed(path)
        if any(existing.get(k) != v for k, v in payload.items()):
            raise ValueError('Existing valid selection changed')
        return existing
    value = signed({**payload, 'frozen_at': now().isoformat()})
    record(path, value)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1 <= args.workers <= 8:
        raise ValueError('workers must be between 1 and 8')
    output = output_path(args)
    source = (ROOT/args.source_run).resolve()
    if source == output or source in output.parents:
        raise ValueError('Output cannot overwrite or nest inside source run')
    if args.dry_run:
        print(json.dumps({'output': str(output), 'source_run': str(source), 'split': 'valid',
              'grid': scoring.frozen_grid(), 'selection_rule': selection.RULE,
              'training': False, 'test_evaluated': False, 'workers': args.workers}, indent=2))
        return 0
    data = inputs.load(source)
    if args.check_inputs:
        print(json.dumps({'status': 'verified_inputs', 'valid_queries': len(data['protocol'].retrieval.valid),
              'valid_gallery': len(data['cache']['map_ids']), 'source_files': len(data['source_binding']),
              'baseline_folder': str(data['baseline_folder']), 'test_evaluated': False}))
        return 0
    context = {'version': VERSION, 'split': 'valid', 'grid': scoring.frozen_grid(),
               'selection_rule': selection.RULE, 'source_binding': data['source_binding'],
               'code': code_identity(), 'matching': data['config']['matching'],
               'numpy': np.__version__, 'torch': torch.__version__,
               'posthoc_design': True, 'test_evaluated': False, 'training': False}
    signature = digest(context)
    if args.resume_root:
        old = read_signed(output/'run_manifest.json')
        if old.get('context') != context:
            raise ValueError('Source, settings, code or numerical environment changed; use a new run')
    else:
        output.mkdir(parents=True, exist_ok=False)
    with run_lock(output):
        if args.resume_root and verified(output, signature):
            print('Already completed and verified: '+str(output), flush=True)
            return 0
        if not args.resume_root:
            time = now()
            record(output/'run_manifest.json', signed({'version': VERSION, 'context': context,
                'started_at': time.isoformat(), 'run_id': time.strftime('%Y%m%dT%H%M%S%z')+'__f_region_score_search',
                'source_run': str(source), 'workers': args.workers, 'requested_stage': 'valid'}))
        record(output/'inputs.json', {'source_binding': data['source_binding'], 'context_signature': signature,
               'source_run': str(source), 'baseline_folder': str(data['baseline_folder'])})
        protection(output)
        print('OUTPUT_ROOT='+str(output), flush=True)
        results = pipeline.evaluate(output, data['cache'], data['protocol'].retrieval.valid,
                    data['bundle'], data['config']['matching'], signature, args.workers)
        baseline_replay(output, data['baseline_folder'])
        summarize(output, results)
        selected = freeze_selection(output, signature, results)
        check_sources(data['source_binding'])
        for name, expected in context['code'].items():
            if file_hash(ROOT/name) != expected:
                raise ValueError('Scientific code changed during execution: '+name)
        protection(output, after=True)
        write_json(output/'status.json', {'status': 'completed', 'split': 'valid',
                   'test_evaluated': False, 'winner': selected['winner'], 'context_signature': signature})
        seal(output, signature)
        print('VALID WINNER: '+selected['winner'], flush=True)
        print('COMPLETED: '+str(output), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
