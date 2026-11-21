"""Evaluate all six frozen C1-C5 controls; valid is a verification gate, not a selector."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

import numpy as np
import scipy
import torch
from f_region_score_search.run import check_sources, run_lock
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from . import inputs
from .state import (ROOT, VERSION, KS, SETTINGS, STAGES, REPLAY_KEYS, grid, now, plan,
                    record, signed, read_signed, verified, seal, protection)


def code_identity():
    """Freeze all new code and imported scientific/validation dependencies."""
    paths = set()
    for folder in ('paper_c_ablation', 'f_region_wbbs', 'f_region_evidence', 'f_region_structure',
                   'f_region_score_search', 'f_region_score_test', 'f_region_area_fusion',
                   'f_region_j_cross_test', 'fgj_cross_test', 'fgw_grid_search', 'fgw_cross_test',
                   'fgw_fusion', 'b1_structure_fusion', 'wbm_retrieval',
                   'experiments/proposed/_shared', 'experiments/proposed/f_segmentation',
                   'experiments/proposed/f_segmentation_geometry', 'experiments/proposed/f_global_fusion'):
        paths.update((ROOT/folder).glob('*.py'))
    paths.add(ROOT/'f_region_wbbs/provenance/results_catalog_build_13d29454.py.txt')
    paths.add(ROOT/'results_catalog/build.py')
    paths.add(ROOT/'b1_structure_fusion/config.json')
    paths.update((ROOT.parent/'compared_methods/shared').glob('*.py'))
    return {os.path.relpath(path, ROOT): file_hash(path) for path in sorted(paths)}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--stage', choices=('valid', 'test', 'all'), default='all')
    roots = result.add_mutually_exclusive_group()
    roots.add_argument('--output-root')
    roots.add_argument('--resume-root')
    result.add_argument('--workers', type=int, choices=range(1, 9), default=8)
    result.add_argument('--dry-run', action='store_true')
    result.add_argument('--check-inputs', action='store_true')
    return result


def output_path(args):
    output = (ROOT/(args.resume_root or args.output_root or
              ('results/'+now().strftime('%Y%m%dT%H%M%S%z')+'__paper_c_ablation'))).resolve()
    parent = (ROOT/'results').resolve()
    if output == parent or not output.is_relative_to(parent):
        raise ValueError('All new outputs must be isolated inside results/')
    for ancestor in output.parents:
        if ancestor == parent:
            break
        if (ancestor/'run_manifest.json').exists():
            raise ValueError('Do not nest output inside any earlier experiment')
    if args.resume_root:
        manifest = read_signed(output/'run_manifest.json')
        if manifest.get('version') != VERSION:
            raise ValueError('Resume requires an existing signed paper C ablation run')
    elif output.exists():
        raise FileExistsError('Existing output requires explicit unchanged --resume-root')
    if args.stage == 'test' and not args.resume_root:
        raise ValueError('Test requires --resume-root with verified complete valid evaluations')
    return output


def make_context(data):
    return {'version': VERSION, 'grid': grid(), 'stages': list(STAGES), 'settings': SETTINGS,
        'source_binding': data['source_binding'], 'source_compatibility': data['source_compatibility'],
        'source_metadata': data['source_metadata'], 'code': code_identity(),
        'matching': data['config']['matching'], 'numpy': np.__version__, 'scipy': scipy.__version__,
        'torch': torch.__version__, 'score_dtype': 'float64; preserve original feature arithmetic',
        'ks': KS, 'training_required': False, 'cnn_forward_passes': 0,
        'posthoc_research': True, 'test_parameter_search': False,
        'selection_rule': 'none; all six prespecified variants in valid, test and cross_test',
        'registration_policy': {'replay_controls': REPLAY_KEYS, 'evaluations': 18, 'new_results': 14}}


def prepare_root(output, context, data, *, resume=False, workers=8):
    """Commit the full pre-result plan together with a real start-time manifest."""
    output = Path(output)
    signature = digest(context)
    if resume:
        manifest = read_signed(output/'run_manifest.json')
        if manifest.get('version') != VERSION or manifest.get('context') != context:
            raise ValueError('Source, scientific code, settings or numeric environment changed; use a new run')
        frozen = read_signed(output/'frozen_plan.json')
        if frozen != plan(signature, manifest['started_at']):
            raise ValueError('The original pre-result ablation plan changed')
    else:
        output.mkdir(parents=True, exist_ok=False)
        started = now()
        manifest = signed({'version': VERSION, 'context': context, 'started_at': started.isoformat(),
            'run_id': started.strftime('%Y%m%dT%H%M%S%z')+'__paper_c_ablation', 'workers': workers})
        record(output/'run_manifest.json', manifest)
        record(output/'frozen_plan.json', plan(signature, started.isoformat()))
        record(output/'inputs_valid.json', {'context_signature': signature,
            'source_binding': data['source_binding'], 'source_compatibility': data['source_compatibility'],
            'source_metadata': data['source_metadata'], 'replay_references': data['replay_references']})
    return signature


def freeze_valid_gate(output, context_signature):
    from .register import validate_stage
    output = Path(output)
    index = validate_stage(output, 'valid', context_signature)
    payload = {'status': 'completed', 'context_signature': context_signature,
        'frozen_plan_sha256': file_hash(output/'frozen_plan.json'),
        'valid_index_sha256': file_hash(output/'evaluation_index_valid.json'),
        'evaluations': {name: index['results'][name]['completed_sha256'] for name in grid()},
        'test_variants': list(grid()), 'performance_screening': False}
    path = output/'valid_completed.json'
    if path.exists():
        existing = read_signed(path)
        if any(existing.get(k) != v for k, v in payload.items()):
            raise ValueError('Verified valid gate changed')
        return existing
    value = signed({**payload, 'verified_at': now().isoformat()})
    record(path, value)
    return value


def require_valid_gate(output, context_signature):
    """Gate new test-region cache loading/scoring on every prespecified valid result."""
    output = Path(output)
    if not (output/'valid_completed.json').is_file():
        raise ValueError('Test requires the complete independently verified valid gate')
    return freeze_valid_gate(output, context_signature)


def phase(output, label, **extra):
    write_json(Path(output)/'status.json', {'status': 'running', 'phase': label,
                                         'at': now().isoformat(), **extra})
    print('PHASE '+label, flush=True)


def main(argv=None):
    args = parser().parse_args(argv)
    output = output_path(args)
    if args.dry_run:
        print(json.dumps({'output': str(output), 'stage': args.stage, 'grid': grid(), 'settings': SETTINGS,
            'replay_controls': REPLAY_KEYS, 'evaluation_count': 18, 'new_registration_count': 14,
            'training_required': False, 'test_parameter_search': False, 'workers': args.workers}, indent=2))
        return 0
    print('Loading read-only frozen valid model/protocol/region-J cache', flush=True)
    data = inputs.load('valid')
    context = make_context(data)
    signature = digest(context)
    if args.check_inputs:
        if args.resume_root:
            if read_signed(output/'run_manifest.json')['context'] != context:
                raise ValueError('Existing run scientific context differs')
        if args.stage == 'test':
            require_valid_gate(output, signature)
            data = inputs.load('test')
        print(json.dumps({'status': 'verified_inputs', 'split': 'test' if args.stage == 'test' else 'valid',
            'source_files': len(data['source_binding']), 'variants': list(grid()),
            'context_signature': signature, 'training_required': False}))
        return 0
    prepare_root(output, context, data, resume=bool(args.resume_root), workers=args.workers)
    with run_lock(output):
        if verified(output, signature):
            from .register import verify_run
            verify_run(output)
            print('Run already completed and verified: '+str(output), flush=True)
            return 0
        protection(output)
        print('OUTPUT_ROOT='+str(output), flush=True)
        completion_written = False
        try:
            from . import pipeline
            if not (output/'valid_completed.json').exists():
                if args.stage == 'test':
                    raise ValueError('Test cannot bypass the complete valid stage')
                phase(output, 'valid')
                pipeline.evaluate(output, data, 'valid', signature, args.workers)
                check_sources(data['source_binding'])
                freeze_valid_gate(output, signature)
            gate = require_valid_gate(output, signature)
            if args.stage == 'valid':
                write_json(output/'status.json', {'status': 'valid_completed', 'at': now().isoformat(),
                    'test_pending': True, 'screening': False, 'all_test_variants': list(grid())})
                return 0
            # The new test-region loader/scoring stage starts only after this gate.
            # Frozen protocol metadata and protection hashes may mention old test files earlier.
            phase(output, 'test_inputs')
            test = inputs.load('test')
            record(output/'inputs_test.json', {'context_signature': signature,
                'source_binding': test['source_binding'], 'source_compatibility': test['source_compatibility'],
                'source_metadata': test['source_metadata'], 'replay_references': test['replay_references'],
                'valid_completed_sha256': file_hash(output/'valid_completed.json'),
                'frozen_plan_sha256': file_hash(output/'frozen_plan.json')})
            record(output/'test_context.json', signed({'context_signature': signature,
                'valid_completed_sha256': file_hash(output/'valid_completed.json'),
                'frozen_plan_sha256': file_hash(output/'frozen_plan.json'),
                'source_binding_sha256': digest(test['source_binding']), 'specs': grid()}))
            from .register import validate_stage
            for stage in ('test', 'cross_test'):
                phase(output, stage)
                pipeline.evaluate(output, test, stage, signature, args.workers)
                validate_stage(output, stage, signature)
            check_sources(data['source_binding'])
            check_sources(test['source_binding'])
            protection(output, after=True)
            write_json(output/'status.json', {'status': 'completed', 'finished_at': now().isoformat(),
                'evaluations': 18, 'new_registration_results': 14, 'replay_controls': 4,
                'test_parameter_search': False})
            seal(output, signature)
            completion_written = True
            from .register import verify_run, write_proposal
            verify_run(output)
            print('CATALOG_PROPOSAL='+str(write_proposal(output)), flush=True)
        except BaseException as error:
            if completion_written and (output/'completed.json').exists():
                parent = output/'failed_completion_checks'
                parent.mkdir(exist_ok=True)
                failed = Path(tempfile.mkdtemp(prefix=now().strftime('%Y%m%dT%H%M%S%z')+'_', dir=parent))
                (output/'completed.json').rename(failed/'completed.json')
                write_json(failed/'failure.json', {'status': 'rejected_completion',
                    'at': now().isoformat(), 'error_type': type(error).__name__, 'error': str(error)})
            if not (output/'completed.json').exists():
                write_json(output/'status.json', {'status': 'failed', 'at': now().isoformat(),
                    'error_type': type(error).__name__, 'error': str(error)})
            raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
