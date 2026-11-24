"""Limited validation search, frozen controls and immutable result sealing."""
from pathlib import Path
import argparse
import csv
from datetime import datetime
import json
import math
from .state import (ROOT, VERSION, PLAN, GRID, MODELS, THETA0, TIE_RULE, now, record,
    signed, read_signed, read_json, sha, digest, code_binding, check_binding, run_context, verified)
INDEX_NAMES = ('evaluation_index_valid.json','controls_index_valid.json','evaluation_index_test.json')
EXCLUDED = ('logs','catalog_backup','registration_proposal.json','registration_receipt.json','catalog_sources.json')


def initialize(output, *, started_at=None):
    from .inputs import load, persist_inputs
    from .report import bind_wbbs_references
    from f_region_structure.state import protection
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT/'results') or output == ROOT/'results':
        raise ValueError('Use a new isolated timestamped results/ root')
    if (output/'run_manifest.json').exists():
        manifest = run_context(output)
        check_binding(manifest['source_binding'])
        bind_wbbs_references(output)
        protection(output)
        return manifest
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('A new scientific run requires an empty output root')
    time = started_at or now()
    data = load('valid')
    output.mkdir(parents=True,exist_ok=True)
    plan = signed({'plan':PLAN,'frozen_at':time.isoformat()})
    record(output/'frozen_plan.json',plan)
    manifest = signed({'version':VERSION, 'run_id':time.strftime('%Y%m%dT%H%M%S%z')+'__cnn_rotation_fusion',
        'started_at':time.isoformat(), 'plan_signature':plan['signature'], 'code':code_binding(),
        'source_binding':data['source_binding'], 'training_required':False, 'test_parameter_search':False,
        'selection_split':'valid','models':list(MODELS),'grid':GRID})
    record(output/'run_manifest.json',manifest)
    persist_inputs(output,data,'valid',manifest['signature'])
    bind_wbbs_references(output)
    protection(output)
    return manifest


def write_index(output, name, rows, context):
    index = signed({'context_signature':context,'evaluations':dict(sorted(rows.items())),
        'new_evaluations':sum(row['register'] for row in rows.values()),
        'historical_references':sum(row['historical_reference'] for row in rows.values())})
    record(Path(output)/name,index)
    return index


def valid(output, workers=4):
    from .inputs import load, persist_inputs
    from .evaluation import evaluate_variants
    manifest = run_context(output)
    data = load('valid')
    persist_inputs(output,data,'valid',manifest['signature'])
    rows = {}
    for model in MODELS:
        rows.update(evaluate_variants(output,data,model,'valid',list(GRID),['valid'],manifest['signature'],
                                     workers,replay_theta0=model=='rotation'))
    return write_index(output,'evaluation_index_valid.json',rows,manifest['signature'])


def selection_data(index, *, formal=True):
    """Use integer hits from ordinary-valid per-query rows, never test metrics."""
    expected = {model+'/valid/'+name for model in MODELS for name in GRID}
    if set(index['evaluations']) != expected:
        raise ValueError('Selection requires the complete twelve-point grid for both CNNs')
    ranking = []
    identities = None
    for name in GRID:
        row = index['evaluations']['rotation/valid/'+name]
        if row['model'] != 'rotation' or row['stage'] != 'valid' or row['spec'] != GRID[name]:
            raise ValueError('Selection row is not the declared rotation ordinary-valid endpoint')
        report = read_json(row['metrics'])
        with (Path(row['metrics']).parent/'per_query.csv').open(newline='') as stream:
            queries = list(csv.DictReader(stream))
        ids = [int(q['query_id']) for q in queries]
        if len(ids) != len(set(ids)) or (formal and len(ids) != 500) or not ids:
            raise ValueError('Selection requires all 500 distinct frozen validation queries')
        if identities is not None and ids != identities:
            raise ValueError('Validation weight queries/order differ')
        identities = ids
        hits = 0
        for q in queries:
            value = float(q['Precision@10'])*10
            if not math.isfinite(value) or abs(value-round(value)) > 1e-8 or not 0 <= value <= 10:
                raise ValueError('Top10 exact hits must be integers from zero to ten')
            hits += round(value)
        if (report['metric_query_counts']['Precision@10'] != len(ids)
                or abs(report['metrics']['Precision@10']-hits/(10*len(ids))) > 1e-12):
            raise ValueError('Integer selection hits disagree with exact report')
        ranking.append({'variant':name,'top10_hits':hits,'num_queries':len(ids),'Precision@10':hits/(10*len(ids))})
    ranking.sort(key=lambda row:(-row['top10_hits'],row['variant']!=THETA0,row['variant']))
    for rank,row in enumerate(ranking,1): row['rank'] = rank
    winner = ranking[0]['variant']
    return {'theta0':THETA0,'winner':winner,'spec':GRID[winner], 'ranking':ranking,'tie_rule':TIE_RULE,
        'selected_variants':list(dict.fromkeys([THETA0,winner])), 'winner_is_theta0':winner==THETA0,
        'selection_split':'valid','test_used':False, 'query_ids_sha256':digest(identities)}


def verify_selection(output, *, check_sources=True):
    from .evaluation import verify_index
    output = Path(output)
    manifest = run_context(output)
    index = verify_index(output/'evaluation_index_valid.json',context_signature=manifest['signature'],check_sources=check_sources)
    selection = read_signed(output/'selection.json')
    stamp = selection.get('frozen_at')
    if not isinstance(stamp,str) or not stamp.endswith('+08:00') or datetime.fromisoformat(stamp) < datetime.fromisoformat(manifest['started_at']):
        raise ValueError('Invalid actual selection freeze timestamp')
    expected = {**selection_data(index),'frozen_at':stamp,'context_signature':manifest['signature'],
                'evaluation_index_valid_sha256':sha(output/'evaluation_index_valid.json')}
    if selection != signed(expected):
        raise ValueError('Selection is detached from the full validation-only frozen rule')
    return selection


def select(output, workers=4):
    from .evaluation import verify_index, evaluate_variants
    from .inputs import load, persist_inputs
    output = Path(output)
    manifest = run_context(output)
    index = verify_index(output/'evaluation_index_valid.json',context_signature=manifest['signature'])
    freeze_stamp = read_signed(output/'selection.json')['frozen_at'] if (output/'selection.json').exists() else now().isoformat()
    selection = signed({**selection_data(index),'frozen_at':freeze_stamp,'context_signature':manifest['signature'],
        'evaluation_index_valid_sha256':sha(output/'evaluation_index_valid.json')})
    record(output/'selection.json',selection)
    data = load('valid')
    persist_inputs(output,data,'valid',manifest['signature'])
    rows = {}
    for model in MODELS:
        rows.update(evaluate_variants(output,data,model,'valid',selection['selected_variants'],['cross_valid'],
                                    manifest['signature'],workers))
    write_index(output,'controls_index_valid.json',rows,manifest['signature'])
    return selection


def test(output, workers=4):
    from .inputs import load, persist_inputs
    from .evaluation import verify_index, evaluate_variants
    output = Path(output)
    manifest = run_context(output)
    selection = verify_selection(output)
    controls = verify_index(output/'controls_index_valid.json',context_signature=manifest['signature'])
    expected = {model+'/cross_valid/'+name for model in MODELS for name in selection['selected_variants']}
    if set(controls['evaluations']) != expected:
        raise ValueError('Selected cross-validation controls must finish before test')
    # Selected-model test scoring begins only after this gate. Historical WBBS
    # references are bound at init and never enter the selection calculation.
    data = load('test')
    persist_inputs(output,data,'test',manifest['signature'])
    rows = {}
    for model in MODELS:
        rows.update(evaluate_variants(output,data,model,'test',selection['selected_variants'],['test','cross_test'],
                                     manifest['signature'],workers))
    return write_index(output,'evaluation_index_test.json',rows,manifest['signature'])


def verify_run(output, *, check_sources=True, require_root_seal=True):
    """Verify sealed scientific evidence without rescoring or changing any artifact."""
    from .evaluation import verify_index
    from .report import verify_report
    output = Path(output).resolve()
    manifest = run_context(output)
    if check_sources: check_binding(manifest['source_binding'])
    selection = verify_selection(output,check_sources=check_sources)
    all_rows = {}
    for name in INDEX_NAMES:
        index = verify_index(output/name,context_signature=manifest['signature'],check_sources=check_sources)
        if name == 'controls_index_valid.json':
            expected = {m+'/cross_valid/'+v for m in MODELS for v in selection['selected_variants']}
        elif name == 'evaluation_index_test.json':
            expected = {m+'/'+s+'/'+v for m in MODELS for s in ('test','cross_test') for v in selection['selected_variants']}
        else:
            expected = {m+'/valid/'+v for m in MODELS for v in GRID}
        if set(index['evaluations']) != expected:
            raise ValueError('Missing or unauthorized evaluation roles: '+name)
        all_rows.update(index['evaluations'])
    for split in ('valid','test'):
        inputs = read_signed(output/('inputs_'+split+'.json'))
        if inputs['context_signature'] != manifest['signature'] or inputs['split'] != split:
            raise ValueError('Inputs detached from experiment context')
        if check_sources: check_binding(inputs['source_binding'])
        for role,row in all_rows.items():
            if row['split'] != split: continue
            if row['historical_reference'] and inputs['references'].get(role) != row:
                raise ValueError('Historical role differs from bound canonical source')
            if row['register'] and (role in inputs['references'] or not Path(row['path']).resolve().is_relative_to(output)):
                raise ValueError('New evaluation duplicates a canonical role or escapes its run')
            if row['register'] and Path(row['model_context_path']).resolve() != output/'evaluation'/row['model']/f'model_context_{split}.json':
                raise ValueError('New evaluation uses a different model context file')
        for model in MODELS:
            context = read_signed(output/'evaluation'/model/f'model_context_{split}.json')
            if (context['parent_context'] != manifest['signature'] or context['model'] != model or context['split'] != split
                    or context['source_binding_sha256'] != digest(inputs['source_binding'])
                    or context['h512_path'] != inputs['features'][model] or context['token_layout'] != inputs['token_layout']
                    or context['h512_sha256'] != inputs['source_binding'][inputs['features'][model]]):
                raise ValueError('Model context detached from original feature or geometry provenance')
    replay = read_signed(output/'theta0_replay.json')
    if (replay['context_signature'] != manifest['signature'] or replay['status'] != 'verified'
            or replay['verification']['max_saved_score_absolute_error'] != 0
            or replay['verification']['all_queries_full_candidate_topk_exact'] is not True):
        raise ValueError('Exact theta0 compatibility replay required')
    if read_json(output/'source_protection_after.json').get('status') != 'unchanged':
        raise ValueError('Historical source preservation check required')
    verify_report(output)
    if require_root_seal:
        if not verified(output,manifest['signature']):
            raise ValueError('Scientific root is not sealed')
        marker = read_json(output/'completed.json')
        actual = {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()
                  and p != output/'completed.json' and p.relative_to(output).parts[0] not in EXCLUDED}
        if actual != set(marker['files']):
            raise ValueError('Unsealed or missing scientific files at root')
    keys = [r['key'] for r in all_rows.values() if r['register']]
    if len(keys) != len(set(keys)):
        raise ValueError('New evaluation identity duplicated')
    return {'manifest':manifest,'selection':selection,'evaluation_count':len(all_rows),
            'new_evaluation_count':len(keys),'historical_reference_count':len(all_rows)-len(keys)}


def finalize(output):
    from .report import run_report
    from f_region_structure.state import protection
    output = Path(output)
    if (output/'completed.json').exists(): return verify_run(output)
    manifest = run_context(output)
    run_report(output)
    protection(output,after=True)
    verify_run(output,require_root_seal=False)
    files = {str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()
             and p != output/'completed.json' and p.relative_to(output).parts[0] not in EXCLUDED}
    record(output/'completed.json',{'status':'completed','signature':manifest['signature'],
        'completed_at':now().isoformat(),'files':files,'excluded':list(EXCLUDED)})
    return verify_run(output)


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--stage',required=True,choices=['init','valid','select','test','report','finalize','all'])
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args(argv)
    if args.output_root is None and args.stage not in ('init','all'):
        parser.error('Continuation requires --output-root')
    started_at = now()
    output = (args.output_root or ROOT/'results'/(started_at.strftime('%Y%m%dT%H%M%S%z')+'__cnn_rotation_fusion')).resolve()
    if args.stage in ('init','all'): initialize(output,started_at=started_at)
    if args.stage in ('valid','all'): valid(output,args.workers)
    if args.stage in ('select','all'): select(output,args.workers)
    if args.stage in ('test','all'): test(output,args.workers)
    if args.stage == 'report':
        from .report import run_report
        run_report(output)
    if args.stage in ('finalize','all'): finalize(output)
    print(json.dumps({'output_root':str(output),'stage':args.stage,'status':'completed'}),flush=True)


if __name__ == '__main__': main()
