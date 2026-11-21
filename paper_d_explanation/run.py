"""Frozen D1-D4 diagnostics; scientific computation and visual sealing are separate."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import numpy as np
import torch

from f_region_structure.state import now, record, signed, read_signed, seal, verified
from f_region_score_search.run import check_sources, run_lock
from paper_c_ablation.state import protection
from wbm_retrieval.common import digest, file_hash, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/20261121T175924+0800__paper_c_ablation'
VERSION = 'paper_d_explanation_v1'


def plan():
    return {
        'version': VERSION, 'posthoc_research': True, 'new_formal_model_evaluations': 0,
        'training_required': False, 'test_parameter_search': False,
        'source_variant': 'add_a0p4_b0p25',
        'score': '0.75*(0.4*C+0.6*T)+0.25*A; preserve historical operation order',
        'regions': 'frozen cc8 min_area5 Top4 h512 best_nce seed2026',
        'matching': 'original unidirectional greedy one-to-one, complete query sqrt(area) denominator',
        'protocols': ['retrieval_v1', 'coverage_cross_combination_v1'],
        'examples': {'selection': 'first query in fixed protocol order per complete label combination',
                     'ranks': [1, 10, 100], 'per_protocol': 111, 'total': 222},
        'figure_selection': {'categories': ['ordinary_exact', 'cross_full', 'cross_partial', 'cross_none'],
            'order': 'query_id, rank, candidate_id',
            'failure_eligibility': 'no-overlap display case must have at least one eligible cross-gallery candidate with positive label coverage',
            'fallback': 'if category absent in fixed 111 examples, use all test rank1 of that protocol'},
        'intervention': {'queries': 'all frozen 7199 test queries; compare strategies only where n_regions>=2',
            'expected_eligible': 7161, 'expected_single': 38,
            'reference': 'each protocol own original rank1, frozen before intervention; labels not used',
            'contribution': 'actual weighted accepted-match contribution to reference; unmatched zero',
            'contribution_ties': 'ascending original local region index',
            'budget': 'one whole query-region score row zeroed for every candidate',
            'recompute': 'greedy matching and complete eligible-gallery ranking; unchanged original denominator',
            'strategies': ['high', 'low', 'uniform_random'],
            'random': 'enumerate all single-region deletions, average outcome metrics, never average scores then rank',
            'area_matched': {'weight': 'sqrt(area)/sum(sqrt(area))', 'max_weight_ratio': 1.2,
                'selection': 'closest absolute log weight ratio, then ascending region-index pair; before contributions',
                'random': 'mean of the selected pair deletion outcomes',
                'missing_pair': 'exclude from area-matched subset and report denominator; do not adjust threshold'},
            'metrics': ['ordinary Precision@10', 'Coverage-nDCG@10 for both protocols',
                        'reference score change and reranked position'],
            'undefined_ndcg': 'exclude IDCG=0 from that metric only, with valid query count',
            'baseline': 'unperturbed metrics on each identical diagnostic subset'},
        'claims': 'traceable score decomposition and score-evidence sensitivity; no regional semantic truth or physical causality',
        'numeric_check': 'same-components accepted contribution tolerance 1e-12; algebraic C/T/A float32 rounding tolerance 2e-6',
        'display_code': 'renderer identity recorded separately after visual QA; scientific source frozen before computation',
    }


def code_identity():
    from paper_c_ablation.run import code_identity as base_identity
    result = base_identity()
    for name in ('__init__.py', 'run.py', 'examples.py', 'perturb.py', 'verify.py'):
        path = ROOT/'paper_d_explanation'/name
        result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


def bind_sources(data):
    """Bind the already sealed primary C replay, without touching other variants."""
    binding = data['source_binding']
    root = read_json(SOURCE/'completed.json')
    if root.get('status') != 'completed':
        raise ValueError('Source C experiment is not complete')
    binding[str(SOURCE/'completed.json')] = file_hash(SOURCE/'completed.json')
    names = ['run_manifest.json', 'frozen_plan.json', 'replay_test.json', 'replay_cross_test.json']
    for name in names:
        path = SOURCE/name
        if file_hash(path) != root['files'][name]:
            raise ValueError('Changed source C root file: '+name)
        binding[str(path)] = root['files'][name]
    for relative in ('test/primary', 'cross_test/primary', 'score_cache/test/primary'):
        folder = SOURCE/relative
        marker_path = folder/'completed.json'
        if file_hash(marker_path) != root['files'][relative+'/completed.json'] or not verified(folder):
            raise ValueError('Changed primary source: '+relative)
        binding[str(marker_path)] = file_hash(marker_path)
        for name, value in read_json(marker_path)['files'].items():
            binding[str(folder/name)] = value
    return data


def load_data():
    from paper_c_ablation.inputs import load
    data = bind_sources(load('test'))
    data['pool'] = data['protocol'].retrieval.test
    candidates = [Path(p) for p in data['source_binding'] if p.endswith('/test/region_labels.npy')]
    matches = []
    for path in candidates:
        with np.load(path.parent/'regions.npz', allow_pickle=False) as archive:
            if (np.array_equal(archive['map_ids'], data['cache']['map_ids'])
                    and np.array_equal(archive['offsets'], data['cache']['offsets'])
                    and np.array_equal(archive['area'], data['cache']['area'])):
                matches.append(path)
    if not matches:
        raise ValueError('No hash-bound original mask source matches regional cache identities and area')
    path = sorted(matches)[0]
    data['region_labels'] = np.load(path, mmap_mode='r')
    data['mask_map_ids'] = data['cache']['map_ids']
    data['mask_source'] = str(path)
    with np.load(data['config']['data_file'], allow_pickle=False) as archive:
        data['maps'] = archive['arr_0']
    if len(data['pool']) != 7199 or len(data['pool'].gallery) != 7199:
        raise ValueError('Frozen test identities changed')
    return data


def ensure_code(manifest):
    for name, value in manifest['context']['code'].items():
        if file_hash(ROOT/name) != value:
            raise ValueError('Frozen scientific dependency changed: '+name)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('compute', 'render', 'seal', 'verify'), default='compute')
    p.add_argument('--output-root')
    p.add_argument('--workers', type=int, choices=range(1, 9), default=8)
    p.add_argument('--check-inputs', action='store_true')
    args = p.parse_args(argv)
    if args.stage == 'compute':
        print('Loading and checking read-only frozen sources', flush=True)
        data = load_data()
        if args.check_inputs:
            print('Verified inputs: '+str(len(data['source_binding']))+' source files; mask '+data['mask_source'], flush=True)
            return 0
        started = now()
        output = (ROOT/(args.output_root or ('results/'+started.strftime('%Y%m%dT%H%M%S%z')+'__paper_d_explanation'))).resolve()
        if output.exists() or output.parent != ROOT/'results':
            raise ValueError('Use a fresh immediate child of results/')
        output.mkdir(parents=True)
        context = {'plan': plan(), 'code': code_identity(), 'source_binding': data['source_binding'],
            'source_compatibility': data['source_compatibility'], 'mask_source': data['mask_source'],
            'numpy': np.__version__, 'torch': torch.__version__, 'workers': args.workers}
        manifest = signed({'version': VERSION, 'run_id': output.name, 'started_at': started.isoformat(), 'context': context})
        record(output/'run_manifest.json', manifest)
        record(output/'frozen_plan.json', signed({'started_at': started.isoformat(), 'plan': plan()}))
        print('OUTPUT_ROOT='+str(output), flush=True)
        with run_lock(output):
            protection(output)
            try:
                from . import examples, perturb
                write_json(output/'status.json', {'status': 'running', 'phase': 'D1_D2_examples'})
                examples.run(output, data, source_root=SOURCE)
                write_json(output/'status.json', {'status': 'running', 'phase': 'D4_interventions'})
                perturb.run(output, data, workers=args.workers, baseline_root=SOURCE/'score_cache/test/primary')
                check_sources(data['source_binding'])
                ensure_code(manifest)
                write_json(output/'status.json', {'status': 'computed', 'phase': 'awaiting_figures_and_visual_review', 'at': now().isoformat()})
            except BaseException as error:
                write_json(output/'status.json', {'status': 'failed', 'at': now().isoformat(), 'error': str(error)})
                raise
        return 0
    if not args.output_root:
        raise ValueError('Existing output required for render/seal/verify')
    output = (ROOT/args.output_root).resolve()
    if output.parent != ROOT/'results':
        raise ValueError('Output must be inside results/')
    manifest = read_signed(output/'run_manifest.json')
    ensure_code(manifest)
    if args.stage == 'verify':
        if not verified(output, manifest['signature']):
            raise ValueError('Run is not sealed')
        from .verify import verify_output
        verify_output(output)
        print('VERIFIED '+str(output), flush=True)
        return 0
    if (output/'completed.json').exists():
        raise ValueError('Completed output is immutable')
    if args.stage == 'render':
        from .figures import render
        render(output)
        print('FIGURES_READY '+str(output), flush=True)
        return 0
    from .verify import verify_output
    review = read_json(output/'figure_review.json')
    if review.get('status') != 'verified' or not review.get('reviewed_images'):
        raise ValueError('Visual review must be recorded before final sealing')
    for relative, value in review['reviewed_images'].items():
        if file_hash(output/relative) != value:
            raise ValueError('Visually reviewed image changed')
    evidence = verify_output(output)
    check_sources(manifest['context']['source_binding'])
    protection(output, after=True)
    record(output/'verification.json', {**evidence, 'source_protection': 'unchanged',
        'figure_review_sha256': file_hash(output/'figure_review.json'), 'verified_at': now().isoformat()})
    record(output/'summary.json', {'version': VERSION, 'run_id': manifest['run_id'],
        'started_at': manifest['started_at'], 'analysis_only': True, 'new_formal_model_evaluations': 0,
        'explanations': read_json(output/'explanations/verification.json'),
        'interventions': read_json(output/'d4/summary.json'),
        'display': 'four real examples plus intervention summary and area-matched control; pending manuscript integration'})
    write_json(output/'status.json', {'status': 'completed', 'at': now().isoformat()})
    seal(output, manifest['signature'])
    if not verified(output, manifest['signature']):
        raise ValueError('Final artifact sealing failed')
    print('COMPLETED '+str(output), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
