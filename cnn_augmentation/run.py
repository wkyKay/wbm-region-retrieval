"""Isolated five-augmentation experiment stages; immutable source and real identities."""
from pathlib import Path
import argparse
import json

from wbm_retrieval.common import file_hash, read_json, write_json
from .state import ROOT, VERSION, PLAN, VARIANTS, MODELS, now, record, signed, run_context, verified


def require_valid_gate(output, model_name, context_signature):
    from .register import validate_model_split
    index = validate_model_split(output, model_name, 'valid')
    index_path = Path(output)/'evaluation'/model_name/'evaluation_index_valid.json'
    expected = {stage+'/'+style for stage in ('valid', 'cross_valid') for style in ('cnn_only', 'fusion')}
    if (index.get('context_signature') != context_signature or index.get('model') != model_name
            or index.get('split') != 'valid' or set(index.get('evaluations', {})) != expected):
        raise ValueError('Validation index is detached from model/plan or incomplete')
    gate = signed({'status': 'verified', 'model': model_name, 'context_signature': context_signature,
        'evaluation_index_sha256': file_hash(index_path),
        'completed_reports': {k:file_hash(Path(row['path'])/'completed.json') for k,row in index['evaluations'].items()},
        'frozen_plan_sha256': file_hash(Path(output)/'frozen_plan.json'),
        'gate': 'implementation and protocol only; no performance threshold or selection'})
    record(Path(output)/('valid_gate_'+model_name+'.json'), gate)
    return gate


def initialize(output):
    from .inputs import load
    from f_region_structure.state import protection
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT/'results') or output == ROOT/'results':
        raise ValueError('Use an isolated results/ run root')
    if (output/'run_manifest.json').exists():
        manifest = run_context(output)
        required = ['source_protection_before.json', 'paper_source.json', 'status.json']
        if not all((output/name).exists() for name in required):
            # Initialization may be completed only before training/evaluation starts.
            if (output/'training').exists() or (output/'evaluation').exists():
                raise ValueError('Incomplete initialization after scientific stages started; preserve this run')
            from f_representation_audit.inputs import verify_protection
            verify_protection(manifest['source_binding'])
            if not (output/'paper_source.json').exists():
                source = ROOT/'cnn_augmentation/paper_source.json'
                if file_hash(source) != manifest['paper_source_sha256']:
                    raise ValueError('Original paper source receipt changed during incomplete initialization')
                record(output/'paper_source.json', read_json(source))
            protection(output)
            if not (output/'status.json').exists():
                write_json(output/'status.json', {'status':'initialized',
                    'started_at':manifest['started_at'], 'run_id':manifest['run_id']})
        return manifest
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('A new experiment requires an empty isolated output root')
    time = now()
    output.mkdir(parents=True, exist_ok=True)
    inputs = load()
    plan = signed({'plan': PLAN, 'frozen_at': time.isoformat()})
    record(output/'frozen_plan.json', plan)
    evidence = read_json(ROOT/'cnn_augmentation/paper_source.json')
    record(output/'paper_source.json', evidence)
    manifest = signed({'version': VERSION,
        'run_id': time.strftime('%Y%m%dT%H%M%S%z')+'__cnn_augmentation',
        'started_at': time.isoformat(), 'plan_signature': plan['signature'],
        'paper_source_sha256': file_hash(output/'paper_source.json'),
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


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--stage', required=True, choices=['init','audit','train','evaluate','report','finalize'])
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--splits', nargs='+', choices=['valid','test'], default=['valid','test'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.output_root is None and args.stage != 'init':
        parser.error('Every continuation stage requires persisted --output-root')
    output = (args.output_root or ROOT/'results'/(now().strftime('%Y%m%dT%H%M%S%z')+'__cnn_augmentation')).resolve()
    if args.stage == 'init':
        initialize(output)
        return
    manifest = run_context(output)
    if not all((output/name).is_file() for name in ('source_protection_before.json','paper_source.json','status.json')):
        raise ValueError('Initialization incomplete; finish --stage init before scientific stages')
    if args.stage == 'report':
        from .reporting import build_report
        build_report(output, bootstrap=2000)
        return
    from .inputs import load
    inputs = load(output)
    if args.stage == 'audit':
        from .audit import run_audit
        run_audit(output, inputs)
    elif args.stage == 'train':
        if args.variant is None:
            parser.error('Training requires one --variant')
        audit = read_json(output/'augmentation_audit/manifest.json')
        if audit.get('context_signature') != manifest['signature'] or not verified(output/'augmentation_audit', audit['signature']):
            raise ValueError('Train-only augmentation audit must be sealed before training')
        from .training import train_variant
        print(train_variant(output, inputs, args.variant, device=args.device, workers=args.workers,
                            resume=args.resume), flush=True)
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
                feature = data['cache']['folder']/'h512.npy' if name == 'baseline' else extract(
                    output, inputs, name, data['refinement_patches'], token_layout(data['cache']), split,
                    device=args.device, batch_size=args.batch_size)['original']
                index = evaluate_model(output, data, name, feature, manifest['signature'], workers=args.workers)
                print(json.dumps({'model': name, 'split': split, 'signature': index['signature']}), flush=True)
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
        write_json(output/'status.json', {'status':'completed','run_id':manifest['run_id'],
                                        'completed_at':now().isoformat()})
        files = {str(p.relative_to(output)):file_hash(p) for p in sorted(output.rglob('*'))
                 if p.is_file() and p != output/'completed.json' and p.relative_to(output).parts[0] != 'logs'}
        record(output/'completed.json', {'status':'completed','signature':manifest['signature'],
            'files':files,'excluded':['logs/'],'exclusion_reason':'live execution logs only'})
        verify_run(output)


if __name__ == '__main__':
    main()
