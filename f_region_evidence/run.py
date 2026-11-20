"""Validate region completion and calibrated rejection; freeze family winners before test."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import groupby
import csv
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
import numpy as np
import torch

from fgw_cross_test.state import cross_pool, CROSS_SOURCE
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from . import inputs, calibration, prepare, proposals, scoring, selection, pipeline
from .state import ROOT, VERSION, KS, REFERENCE, BASELINE, grid, now, record, signed, read_signed, seal, verified, protection, evaluation_signature


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def code_identity():
    paths = set(Path(__file__).parent.glob('*.py'))
    paths.discard(Path(__file__).with_name('register.py'))
    for folder in ('f_region_structure', 'f_region_score_search', 'f_region_score_test',
                   'f_region_area_fusion', 'f_region_j_cross_test', 'fgj_cross_test',
                   'fgw_grid_search', 'fgw_cross_test', 'wbm_retrieval',
                   'experiments/proposed/_shared', 'experiments/proposed/f_segmentation',
                   'experiments/proposed/f_segmentation_geometry', 'experiments/proposed/f_global_fusion',
                   'b1_structure_fusion'):
        paths.update((ROOT/folder).glob('*.py'))
    paths.add(ROOT/'b1_structure_fusion/config.json')
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)}


def _calibrate_one(task):
    output, name, cache_folder, train_folder, reference, split, context_signature = task
    folder = Path(output)/'calibration'/split/name
    signature = digest({'context': context_signature, 'split': split, 'proposal': name,
                        'reference': reference['reference_identity_sha256'], 'version': 1})
    if not verified(folder, signature):
        if folder.exists():
            raise FileExistsError('Incomplete calibration requires a fresh run')
        folder.mkdir(parents=True)
        started = time.perf_counter()
        cache = pipeline.read_scoring_cache(cache_folder)
        train = pipeline.read_scoring_cache(train_folder)
        thresholds, metadata = calibration.calibrate(cache, train, reference)
        np.savez(folder/'thresholds.npz', **thresholds)
        record(folder/'metadata.json', {**jsonable(metadata), 'signature': signature,
            'context_signature': context_signature, 'split': split, 'proposal': name,
            'cache_sha256': file_hash(cache['folder']/'completed.json'),
            'wall_seconds': time.perf_counter()-started})
        seal(folder, signature)
        print(f'Calibrated {split}/{name}: {len(cache["features"])} regions against {reference["size"]} train references', flush=True)
    with np.load(folder/'thresholds.npz', allow_pickle=False) as archive:
        return name, dict(archive)


def prepare_stage(output, data, split, context_signature, reference, args):
    caches = prepare.prepare(output, data['maps'], data['protocol'], data['cache'], data['old_labels'],
        data['config'], data['suite'], data['checkpoint'], context_signature, data['source_binding'],
        workers=args.prepare_workers, device=args.device, split=split)
    from .verify import audit_preserved_regions
    preserved = {name: audit_preserved_regions(data['cache'], cache) for name, cache in caches.items()}
    record(output/f'preserved_regions_{split}.json', {'status': 'verified', 'proposals': preserved})
    tasks = [(str(output), name, str(cache['folder']), str(data['train_cache']['folder']),
              reference, split, context_signature)
             for name, cache in caches.items()]
    thresholds = {}
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn')) as executor:
        for future in as_completed([executor.submit(_calibrate_one, task) for task in tasks]):
            name, values = future.result()
            thresholds[name] = values
    return caches, thresholds


def audit_stage(output, data, caches, thresholds, split, specs, reference):
    from .verify import audit_formulas, audit_calibration
    pool = getattr(data['protocol'].retrieval, split)
    providers = {name: scoring.Provider(cache, pool, thresholds=thresholds[name]) for name, cache in caches.items()}
    formula = audit_formulas(providers, data['bundle'], specs)
    record(output/f'formula_audit_{split}.json', formula)
    calibration_audit = audit_calibration(providers, data['bundle'], data['train_cache'], reference)
    record(output/f'calibration_audit_{split}.json', calibration_audit)
    return providers


def freeze(output, context_signature, results):
    chosen = selection.select(results)
    payload = {**chosen, 'context_signature': context_signature,
               'valid_index_sha256': file_hash(output/'evaluation_index_valid.json')}
    if (output/'selection.json').exists():
        value = read_signed(output/'selection.json')
        if any(value.get(key) != expected for key, expected in payload.items()):
            raise ValueError('An existing frozen valid selection changed')
    else:
        value = signed({**payload, 'frozen_at': now().isoformat()})
    record(output/'selection.json', value)
    record(output/'valid_completed.json', {'status': 'completed', 'context_signature': context_signature,
        'selection_sha256': file_hash(output/'selection.json'),
        'evaluation_index_sha256': file_hash(output/'evaluation_index_valid.json'),
        'evaluation_completions': {n: file_hash(output/'valid'/n/'completed.json') for n in grid()}})
    return value


def frozen_selection(output, context_signature):
    marker = read_json(output/'valid_completed.json')
    chosen = read_signed(output/'selection.json')
    index = read_json(output/'evaluation_index_valid.json')
    if (marker.get('status') != 'completed' or marker.get('context_signature') != context_signature
            or chosen.get('context_signature') != context_signature or index['context_signature'] != context_signature
            or marker['selection_sha256'] != file_hash(output/'selection.json')
            or chosen['valid_index_sha256'] != file_hash(output/'evaluation_index_valid.json')
            or marker['evaluation_index_sha256'] != chosen['valid_index_sha256']
            or index['specs'] != grid()):
        raise ValueError('Invalid frozen complete-valid selection')
    for name in grid():
        folder = output/'valid'/name
        if (not verified(folder, evaluation_signature(context_signature, 'valid', name))
                or marker['evaluation_completions'][name] != file_hash(folder/'completed.json')
                or index['results'][name]['completed_sha256'] != file_hash(folder/'completed.json')):
            raise ValueError('Valid result differs from its frozen selection')
    expected = selection.select(index['results'])
    if any(chosen[k] != v for k, v in expected.items()):
        raise ValueError('Selection differs from fixed integer-hit rule')
    return chosen


def summarize(output, stage, results):
    path = output/f'comparison_{stage}.csv'
    names = list(results)
    metrics = list(results[names[0]].get('metrics', {}))
    coverage = list(results[names[0]]['coverage_metrics'])
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['variant', 'stage', 'top10_hits', *metrics, *coverage])
        for name, row in results.items():
            writer.writerow([name, stage, row.get('top10_hits', ''),
                             *[row['metrics'][k] for k in metrics], *[row['coverage_metrics'][k] for k in coverage]])


def explanations(output, stage, providers, bundle, specs):
    """Fixed one-query-per-combination examples; no best-case selection."""
    for name, spec in specs.items():
        folder = output/'explanations'/stage/name
        if (folder/'examples.json').exists():
            read_json(folder/'examples.json')
            continue
        folder.mkdir(parents=True, exist_ok=True)
        selected = {}
        provider = providers[spec['proposal']]
        for q in provider.pool:
            selected.setdefault(int(bundle.class_ids[q]), int(q))
        examples = []
        with (output/stage/name/'rankings.csv').open(newline='') as f:
            for q, rows in groupby(csv.DictReader(f), key=lambda r: int(r['query_id'])):
                records = list(rows)
                if q not in selected.values():
                    continue
                parts = provider.parts(q)
                for row in (records[0], records[9], records[99]):
                    e = provider.explain(q, int(row['candidate_id']), spec['rejection'], parts)
                    if abs(e['score']-float(row['similarity_score'])) > 1e-12:
                        raise ValueError('Region contribution explanation disagrees with ranking')
                    examples.append({**e, 'rank': int(row['rank']), 'proposal': spec['proposal'],
                        'query_labels': bundle.labels[q].tolist(),
                        'candidate_labels': bundle.labels[e['candidate_id']].tolist()})
        record(folder/'examples.json', {'selection': 'first fixed query in each complete combination; ranks1,10,100',
            'labels_only_used_for_reporting': True, 'region_trace': str(provider.cache['folder']/'region_trace.jsonl'),
            'region_masks': str(provider.cache['folder']/'region_labels.npy'), 'examples': examples})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('valid', 'test', 'all'), default='all')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--output-root')
    group.add_argument('--resume-root')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--prepare-workers', type=int, default=4)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--check-inputs', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1 <= args.workers <= 8 or not 0 <= args.prepare_workers <= 8:
        raise ValueError('Invalid worker count')
    output = (ROOT/(args.resume_root or args.output_root or 'artificial/f_region_evidence_'+now().strftime('%Y%m%dT%H%M%S%z'))).resolve()
    if not output.is_relative_to(ROOT/'artificial') or output == ROOT/'artificial':
        raise ValueError('Use a new artificial/ subdirectory')
    for parent in output.parents:
        if parent == ROOT/'artificial':
            break
        if (parent/'run_manifest.json').exists():
            raise ValueError('Do not nest a new experiment inside a historical run')
    if args.stage == 'test' and not args.resume_root:
        raise ValueError('Test requires the completed valid run via --resume-root')
    if args.dry_run:
        print(json.dumps({'output': str(output), 'grid': grid(), 'proposals': proposals.PROPOSAL_SPECS,
                         'reference': REFERENCE, 'selection_rule': selection.RULE,
                         'no_training': True, 'posthoc_research': True}, ensure_ascii=False, indent=2))
        return 0
    if any(os.environ.get(k) != '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('Set OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1 to preserve numerical replay')
    torch.set_num_threads(1)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA required for frozen-encoder extraction')
    print('Loading and binding frozen valid/model/train-reference inputs', flush=True)
    data = inputs.load('valid')
    reference = jsonable(calibration.select_reference(data['train_cache']))
    source_binding = {**data['source_binding'], str(CROSS_SOURCE.resolve()): file_hash(CROSS_SOURCE)}
    context = {'version': VERSION, 'grid': grid(), 'proposals': proposals.PROPOSAL_SPECS,
        'reference_design': REFERENCE, 'reference': reference, 'selection_rule': selection.RULE,
        'source_binding': source_binding, 'code': code_identity(),
        'numpy': np.__version__, 'torch': torch.__version__, 'ks': KS,
        'cnn_training': False, 'posthoc_research': True, 'test_parameter_search': False,
        'matching': scoring.MATCHING, 'feature_device': args.device,
        'score_dtype': 'float64; original source feature arithmetic', 'seed': 2026}
    context_signature = digest(context)
    if args.check_inputs:
        print(json.dumps({'status': 'verified', 'source_files': len(source_binding),
            'valid_queries': 500, 'gallery': 3599, 'reference_regions': reference['size'],
            'context_signature': context_signature}, indent=2))
        return 0
    if args.resume_root:
        manifest = read_signed(output/'run_manifest.json')
        if manifest['context'] != context:
            raise ValueError('Resume scientific/code/source environment changed')
    else:
        if output.exists():
            raise FileExistsError('Refuse existing output; explicit signed resume only')
        output.mkdir(parents=True)
        stamp = now()
        record(output/'run_manifest.json', signed({'version': VERSION, 'context': context,
            'started_at': stamp.isoformat(), 'run_id': stamp.strftime('%Y%m%dT%H%M%S%z')+'__f_region_evidence',
            'source_run': str(inputs.REGION_SOURCE)}))
        record(output/'inputs_valid.json', {'context_signature': context_signature, 'source_binding': source_binding})
        record(output/'reference.json', reference)
    if verified(output, context_signature):
        print('Run already completed and verified', flush=True)
        return 0
    protection(output)
    try:
        if not (output/'valid_completed.json').exists():
            if args.stage == 'test':
                raise ValueError('No frozen valid selection')
            write_json(output/'status.json', {'status': 'running', 'phase': 'valid_prepare'})
            caches, thresholds = prepare_stage(output, data, 'valid', context_signature, reference, args)
            providers = audit_stage(output, data, caches, thresholds, 'valid', grid(), reference)
            pool = data['protocol'].retrieval.valid
            score_providers = pipeline.compute_scores(output, caches, pool, thresholds, context_signature, grid(), args.workers)
            pipeline.replay_baseline(output, score_providers[BASELINE], data['baseline_folder'], 'valid')
            results = pipeline.evaluate(output, 'valid', grid(), pool, data['bundle'], context_signature, source_binding, workers=args.workers)
            summarize(output, 'valid', results)
            chosen = selection.select(results)
            examples = {BASELINE: grid()[BASELINE], **chosen['selected_specs']}
            explanations(output, 'valid', providers, data['bundle'], examples)
            chosen = freeze(output, context_signature, results)
            print('VALID FROZEN: '+json.dumps(chosen, ensure_ascii=False), flush=True)
        chosen = frozen_selection(output, context_signature)
        if args.stage == 'valid':
            write_json(output/'status.json', {'status': 'valid_completed', 'winner': chosen['winner'], 'test_pending': True})
            return 0
        # This is the first place any held-out test features or results are opened.
        write_json(output/'status.json', {'status': 'running', 'phase': 'test_inputs', 'winner': chosen['winner']})
        data = inputs.load('test')
        test_binding = {**data['source_binding'], str(CROSS_SOURCE.resolve()): file_hash(CROSS_SOURCE)}
        record(output/'inputs_test.json', {'context_signature': context_signature, 'source_binding': test_binding,
            'selection_sha256': file_hash(output/'selection.json')})
        caches, thresholds = prepare_stage(output, data, 'test', context_signature, reference, args)
        specs = chosen['selected_specs']
        replay_specs = {BASELINE: grid()[BASELINE], **specs}
        providers = audit_stage(output, data, caches, thresholds, 'test', replay_specs, reference)
        base = data['protocol'].retrieval.test
        score_providers = pipeline.compute_scores(output, caches, base, thresholds, context_signature, replay_specs, args.workers)
        pipeline.replay_baseline(output, score_providers[BASELINE], data['baseline_folder'], 'test')
        for stage, pool in [('test', base), ('cross_test', cross_pool(base, data['bundle'].labels))]:
            results = pipeline.evaluate(output, stage, specs, pool, data['bundle'], context_signature, test_binding, chosen, args.workers)
            summarize(output, stage, results)
            if stage == 'test':
                explanations(output, stage, providers, data['bundle'], specs)
        protection(output, after=True)
        # Recheck scientific code and all source bytes before publishing completion.
        if code_identity() != context['code']:
            raise ValueError('Scientific code changed during the experiment')
        for path, expected in {**source_binding, **test_binding}.items():
            if file_hash(path) != expected:
                raise ValueError('Frozen source changed during the experiment: '+path)
        write_json(output/'status.json', {'status': 'completed', 'winner': chosen['winner'],
                   'family_winners': chosen['family_winners'], 'completed_at': now().isoformat()})
        seal(output, context_signature)
        if not verified(output, context_signature):
            raise ValueError('Final completion marker failed verification')
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
