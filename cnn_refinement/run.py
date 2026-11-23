"""Stage-oriented runner; each stage binds inputs and has its own actual start time."""
from pathlib import Path
import argparse
import json
import sys

from wbm_retrieval.common import digest, file_hash, read_json, write_json
from .state import ROOT, VERSION, PLAN, VARIANTS, MODELS, now, record, signed, run_context, verified


def require_valid_gate(output, model_name, context_signature):
    # Use the same complete identity/protocol checks as catalogue registration,
    # without requiring root completion or opening any test arrays.
    from .register import validate_model_split
    index = validate_model_split(output, model_name, 'valid')
    folder = Path(output)/'evaluation'/model_name
    index_path = folder/'evaluation_index_valid.json'
    expected = {stage+'/'+style for stage in ('valid', 'cross_valid') for style in ('cnn_only', 'fusion')}
    if (index.get('context_signature') != context_signature or index.get('model') != model_name
            or index.get('split') != 'valid' or set(index.get('evaluations', {})) != expected):
        raise ValueError('Validation index is detached from model/plan or incomplete')
    checks = {}
    for key, result in index['evaluations'].items():
        path = Path(result['path'])
        if not verified(path, result['signature']):
            raise ValueError('Unsealed validation result: '+key)
        if result.get('historical_reference'):
            replay = read_json(folder/('baseline_replay_'+result['stage']+'.json'))
            if (replay.get('status') != 'verified' or replay.get('topk_identical') is not True
                    or replay.get('max_score_error') != 0.):
                raise ValueError('Baseline replay did not verify complete original Top500')
        else:
            audit = read_json(path/'verification.json')
            if (audit.get('status') != 'verified' or audit.get('signature') != result['signature']
                    or audit.get('num_queries') != 500
                    or audit.get('all_queries_full_candidate_topk_exact') is not True
                    or audit.get('ranking_sha256') != file_hash(path/'rankings.csv')):
                raise ValueError('Validation lacks full-gallery independent verification')
        checks[key] = file_hash(path/'completed.json')
    for cache in index['score_cache'].values():
        if not verified(cache['path'], cache['signature']):
            raise ValueError('Validation score cache is unsealed')
    gate = signed({'status': 'verified', 'model': model_name, 'context_signature': context_signature,
        'evaluation_index_sha256': file_hash(index_path), 'completed_reports': checks,
        'frozen_plan_sha256': file_hash(Path(output)/'frozen_plan.json'),
        'gate': 'implementation and protocol only; no performance threshold or selection'})
    record(Path(output)/('valid_gate_'+model_name+'.json'), gate)
    return gate


def initialize(output):
    from .inputs import load
    from f_region_structure.state import protection
    if (output/'run_manifest.json').exists():
        return run_context(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('A new experiment requires an empty isolated output root')
    time = now()
    output.mkdir(parents=True, exist_ok=True)
    inputs = load()
    plan = signed({'plan': PLAN, 'frozen_at': time.isoformat()})
    record(output/'frozen_plan.json', plan)
    manifest = signed({'version': VERSION,
        'run_id': time.strftime('%Y%m%dT%H%M%S%z')+'__cnn_refinement',
        'started_at': time.isoformat(), 'plan_signature': plan['signature'],
        'source_binding': inputs['protection'],
        'baseline': {'checkpoint': str(inputs['checkpoint_paths']['best_nce']),
                     'training_summary': inputs['run_summary']},
        'split_counts': {k: len(v) for k, v in inputs['split_ids'].items()},
        'region_counts': {k: len(v['patches']) for k, v in inputs['datasets'].items()},
        'training_labels': False, 'test_parameter_search': False,
        'authorized_models': list(VARIANTS)})
    record(output/'run_manifest.json', manifest)
    protection(output)
    write_json(output/'status.json', {'status': 'initialized', 'started_at': time.isoformat(),
                                    'run_id': manifest['run_id']})
    print(json.dumps({'output_root': str(output), 'run_id': manifest['run_id'],
                      'context_signature': manifest['signature']}, ensure_ascii=False), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--stage', required=True, choices=['init', 'train', 'teacher', 'evaluate',
                        'diagnose', 'negative-diagnostics', 'report', 'finalize'])
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--splits', nargs='+', choices=['valid', 'test'], default=['valid', 'test'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.output_root is None and args.stage != 'init':
        parser.error('Specify the persisted --output-root for every continuation stage')
    output = (args.output_root or ROOT/'results'/(now().strftime('%Y%m%dT%H%M%S%z')+'__cnn_refinement')).resolve()
    if args.stage == 'init':
        initialize(output)
        return
    manifest = run_context(output)
    from .inputs import load
    inputs = load(output)
    if args.stage == 'train':
        if args.variant is None:
            parser.error('Training requires exactly one --variant')
        from .training import train_variant
        graph = None
        if args.variant == 'filtered_negatives':
            from .neighbors import load_graph
            graph = load_graph(output/'neighbors')
        print(train_variant(output, inputs, args.variant, device=args.device, workers=args.workers,
                            neighbor_graph=graph, resume=args.resume), flush=True)
    elif args.stage == 'teacher':
        from .features import extract, array_identity
        from .neighbors import build_neighbor_graph
        ds = inputs['datasets']['train']
        layout = {k: array_identity(ds[k]) for k in ('map_ids', 'offsets', 'area', 'token_map_ids', 'token_region_ids')}
        features = extract(output, inputs, 'baseline', ds['patches'], layout, 'train',
                            device=args.device, batch_size=args.batch_size, views=('original', 'rot90'))
        graph = build_neighbor_graph(output/'neighbors', features['original'], features['rot90'],
            ds['token_map_ids'], k=500, device=args.device,
            binding={'context_signature': manifest['signature'],
                     'teacher_completed_sha256': file_hash(output/'features/baseline/train/completed.json'),
                     'train_prepared': ds['prepared_marker'], 'labels_used': False})
        print('Frozen neighbor graph:', graph['signature'], flush=True)
    elif args.stage == 'evaluate':
        from .inputs import evaluation_data
        from .features import extract
        from .evaluation import evaluate_model, token_layout
        for split in args.splits:
            if split == 'test':
                for name in args.models:
                    require_valid_gate(output, name, manifest['signature'])
            data = evaluation_data(split, inputs, output)
            for name in args.models:
                features = data['cache']['folder']/'h512.npy' if name == 'baseline' else extract(
                    output, inputs, name, data['refinement_patches'], token_layout(data['cache']), split,
                    device=args.device, batch_size=args.batch_size)['original']
                index = evaluate_model(output, data, name, features, manifest['signature'], workers=args.workers)
                print(json.dumps({'model': name, 'split': split, 'signature': index['signature']}), flush=True)
    elif args.stage == 'diagnose':
        from .diagnostics import run_diagnostics
        from .features import checkpoint_path
        paths = {name: checkpoint_path(output, inputs, name) for name in MODELS}
        run_diagnostics(output, inputs, paths, manifest['signature'],
                        device=args.device, batch_size=args.batch_size)
    elif args.stage == 'negative-diagnostics':
        from .neighbors import load_graph
        from .negative_diagnostics import audit_negatives
        audit_negatives(output, inputs, load_graph(output/'neighbors'))
    elif args.stage == 'report':
        from .reporting import build_report
        build_report(output, bootstrap=2000)
    elif args.stage == 'finalize':
        from .register import verify_run
        from f_representation_audit.inputs import verify_protection
        from f_region_structure.state import protection
        if (output/'completed.json').exists():
            verify_run(output)
            return
        verify_protection(inputs['protection'])
        protection(output, after=True)
        verify_run(output, require_root_seal=False)
        write_json(output/'status.json', {'status': 'completed', 'run_id': manifest['run_id'],
                                         'completed_at': now().isoformat()})
        # A live supervisor's execution log is not scientific evidence; all
        # immutable scientific outputs and source receipts are sealed here.
        files = {str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob('*'))
                 if p.is_file() and p != output/'completed.json'
                 and p.relative_to(output).parts[0] != 'logs'}
        record(output/'completed.json', {'status': 'completed', 'signature': manifest['signature'],
            'files': files, 'excluded': ['logs/'], 'exclusion_reason': 'live execution logs only'})
        verify_run(output)


if __name__ == '__main__':
    main()
