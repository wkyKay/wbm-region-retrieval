"""Evaluate the frozen regional-J valid winner under the existing cross protocol."""
import argparse
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from f_region_area_fusion.pipeline import audit_formulas
from f_region_score_search.run import check_sources, run_lock
from wbm_retrieval.common import digest, file_hash, write_json
from . import inputs, pipeline
from .cache import AuditProvider, build_cache
from .state import (VERSION, VARIANT, SPEC, SETTINGS, PROTOCOL, CANDIDATE_RULE,
                    fixed_variants, now, record, signed, read_signed, protection, seal,
                    verified, verify_run)


def code_identity():
    from f_region_area_fusion.run import code_identity as source_code
    result = source_code()
    for name in ('__init__.py', 'run.py', 'cache.py', 'pipeline.py', 'inputs.py', 'state.py', 'verify.py'):
        path = Path(__file__).parent / name
        result[str(path.relative_to(ROOT))] = file_hash(path)
    for folder, names in {'fgw_cross_test': ('state.py', 'verify.py'),
                          'fgj_cross_test': ('state.py', 'verify.py')}.items():
        for name in names:
            path = ROOT / folder / name
            result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


def seal_and_verify(output, signature):
    """A failed final semantic check must never leave a completed root marker."""
    output = Path(output)
    seal(output, signature)
    try:
        verify_run(output, check_source=False)
    except BaseException as error:
        parent = output / 'failed_completion_checks'
        parent.mkdir(exist_ok=True)
        saved = Path(tempfile.mkdtemp(prefix=now().strftime('%Y%m%dT%H%M%S%z')+'_', dir=parent))
        shutil.copy2(output / 'status.json', saved / 'status.json')
        (output / 'completed.json').rename(saved / 'completed.json')
        write_json(output / 'status.json', {'status': 'failed', 'phase': 'final_semantic_verification',
                   'error_type': type(error).__name__, 'error': str(error), 'retained_evidence': str(saved)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--valid-root', default=str(inputs.DEFAULT_VALID))
    parser.add_argument('--test-root', default=str(inputs.DEFAULT_TEST))
    parser.add_argument('--reference-root', default=str(inputs.DEFAULT_REFERENCE))
    parser.add_argument('--workers', type=int, default=8)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument('--output-root')
    choice.add_argument('--resume-root')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--check-inputs', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8:
        raise ValueError('workers must be between one and eight')
    started = now()
    output = (ROOT / (args.resume_root or args.output_root or
              'artificial/f_region_j_cross_test_'+started.strftime('%Y%m%dT%H%M%S%z'))).resolve()
    allowed = (ROOT / 'artificial').resolve()
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError('An isolated new artificial/ output is required')
    for parent in output.parents:
        if parent == allowed:
            break
        if (parent / 'run_manifest.json').exists():
            raise ValueError('Do not nest inside another run')
    if output.exists() and not args.resume_root:
        raise FileExistsError('Use a new output or explicit matching resume')
    if args.dry_run:
        print({'output': str(output), 'variant': VARIANT, 'spec': SPEC, 'protocol': PROTOCOL,
               'workers': args.workers, 'device': 'original CPU scorer; parallel query shards', 'training': False})
        return 0
    data = inputs.load(args.valid_root, args.test_root, args.reference_root)
    if args.check_inputs:
        print({'status': 'verified_inputs', 'variant': VARIANT, 'queries': len(data['pool']),
               'source_files': len(data['source_binding'])})
        return 0
    import numpy as np
    import torch
    torch.set_num_threads(1)
    data['source_binding_signature'] = digest(data['source_binding'])
    frozen = fixed_variants()
    context = {'version': VERSION, 'split': 'test', 'variant': VARIANT, 'spec': SPEC, 'settings': SETTINGS,
        'code': code_identity(), 'source_binding': data['source_binding'], 'source_metadata': data['source_metadata'],
        'protocol': PROTOCOL, 'candidate_rule': CANDIDATE_RULE, 'derived_protocol': data['pool'].metadata,
        'base_protocol_signature': data['base_pool'].protocol_signature,
        'retrieval_protocol_signature': data['pool'].protocol_signature,
        'f_version': 'regional_J_area_greedy', 'matching': 'original_F_greedy_sqrt_area',
        'matching_config': data['matching'], 'structure_strength': .5,
        'origin_valid_winner': True, 'valid_selection_rank': 1,
        'numpy': np.__version__, 'torch': torch.__version__, 'device': 'cpu',
        'workers': args.workers, 'blas_threads_per_worker': 1,
        'training_required': False, 'posthoc_research': True, 'test_parameter_search': False,
        'selection_rule': 'user_requested_cross_evaluation_of_frozen_valid_winner',
        'primary_variant': None, 'normalization_scope': 'none', 'formal_protocol': True,
        'frozen_variants_signature': frozen['signature'],
        'verification_scope': 'new complete score cache; independent ranking/metrics; regional formula audit separately'}
    signature = digest(context)
    if args.resume_root:
        previous = read_signed(output / 'run_manifest.json')
        if previous.get('version') != VERSION or previous.get('context') != context:
            raise ValueError('Sources or scientific settings changed; use a new run')
    else:
        output.mkdir(parents=True, exist_ok=False)
    with run_lock(output):
        if args.resume_root and verified(output, signature):
            verify_run(output)
            print('Already complete: '+str(output), flush=True)
            return 0
        if not args.resume_root:
            record(output / 'run_manifest.json', signed({'version': VERSION, 'split': 'test', 'stage': 'test',
                'context': context, 'started_at': started.isoformat(),
                'run_id': started.strftime('%Y%m%dT%H%M%S%z')+'__f_region_j_cross_test',
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}))
        record(output / 'frozen_variants.json', frozen)
        record(output / 'frozen_selection.json', data['frozen_selection'])
        record(output / 'settings.json', SETTINGS)
        record(output / 'candidate_protocol.json', data['pool'].metadata)
        record(output / 'inputs.json', {'source_binding': data['source_binding'],
               'source_metadata': data['source_metadata'], 'context_signature': signature})
        path = output / 'protocol/cross_combination_candidates.csv'
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(data['pool'].audit[0]))
                writer.writeheader()
                writer.writerows(data['pool'].audit)
        else:
            with path.open(newline='') as stream:
                actual = [{k: int(v) for k, v in row.items()} for row in csv.DictReader(stream)]
            if actual != data['pool'].audit:
                raise ValueError('Candidate audit changed')
        protection(output)
        print('OUTPUT_ROOT='+str(output), flush=True)
        try:
            begin = time.perf_counter()
            audit_formulas(AuditProvider(data), data['bundle'], output, {VARIANT: SPEC}, signature)
            formula_seconds = time.perf_counter()-begin
            provider, cache_timing = build_cache(output, data, signature, args.workers)
            print('Replaying the original ordinary regional test...', flush=True)
            pipeline.historical_replay(output, provider, data, signature)
            result = pipeline.evaluate(output, data, signature, provider)
            pipeline.summarize(output, data, result, signature)
            if not (output / 'preparation_timing.json').exists():
                record(output / 'preparation_timing.json', {'formula_audit_wall_seconds': formula_seconds,
                                                           'score_cache': cache_timing})
            check_sources(data['source_binding'])
            if context['code'] != code_identity():
                raise ValueError('Scientific code changed during the run')
            protection(output, after=True)
            write_json(output / 'status.json', {'status': 'completed', 'split': 'test', 'stage': 'test',
                'test_evaluated': True, 'new_evaluations': 1, 'context_signature': signature,
                'finished_at': now().isoformat()})
            seal_and_verify(output, signature)
            print('COMPLETED: '+str(output), flush=True)
        except BaseException as error:
            if not (output / 'completed.json').exists():
                write_json(output / 'status.json', {'status': 'failed', 'error_type': type(error).__name__,
                                                   'error': str(error)})
            raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
