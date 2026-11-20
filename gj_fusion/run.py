"""Eight frozen G/J recipes on valid, followed by deduplicated frozen test roles."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import sys

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from b1_ablation.run import identity as upstream_identity
from f_region_score_search.run import run_lock
from experiments.proposed.f_segmentation_geometry import pipeline as geo
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from . import inputs, pipeline, scoring, selection
from .state import now, protection, read_signed, record, seal, signed, verified

VERSION = 'gj_fusion_v1'
SCIENCE_FILES = ('__init__.py','inputs.py','pipeline.py','run.py','scoring.py','selection.py','state.py','verify.py')


def code_identity():
    result = upstream_identity()
    # Registration and subsequent statistical reports do not change scientific scoring.
    for name in SCIENCE_FILES:
        path = Path(__file__).parent/name
        result[str(path.relative_to(ROOT))] = file_hash(path)
    for name in ('f_region_structure/verify.py','f_region_structure/state.py',
                 'f_region_score_search/run.py'):
        result[name]=file_hash(ROOT/name)
    return result


def check_binding(binding):
    for filename, expected in binding.items():
        if not Path(filename).is_file() or file_hash(filename) != expected:
            raise ValueError('Frozen source changed: '+filename)


def read_frozen(valid_root):
    valid_root = Path(valid_root).resolve()
    manifest = read_signed(valid_root/'run_manifest.json')
    if manifest.get('version') != VERSION or manifest.get('split') != 'valid':
        raise ValueError('Test requires this study\'s completed valid run')
    signature = digest(manifest['context'])
    if not verified(valid_root,signature):
        raise ValueError('Valid must be sealed before test')
    if manifest['context']['code'] != code_identity():
        raise ValueError('Scientific code differs from frozen valid')
    index = read_json(valid_root/'evaluation_index_valid.json')
    if index.get('context_signature') != signature or index.get('variants') != scoring.frozen_grid():
        raise ValueError('Valid index does not cover the frozen grid')
    for name, result in index['results'].items():
        folder = valid_root/'valid'/name
        sig = pipeline.variant_signature(signature,'valid',name,scoring.frozen_grid()[name])
        if not geo.completed(folder,sig) or result['completed_sha256'] != file_hash(folder/'completed.json'):
            raise ValueError('Valid evaluation completion changed')
        report = read_json(folder/'metrics.json')
        checked = read_json(folder/'verification.json')
        if (result['top10_hits'] != checked['top10_hits'] or result['metrics'] != report['metrics']
                or abs(result['top10_hits']/5000-report['metrics']['Precision@10']) > 1e-12):
            raise ValueError('Selection values differ from independently verified valid metrics')
    expected = selection.select_candidates(index['results'])
    frozen = read_signed(valid_root/'selection.json')
    if (frozen.get('context_signature') != signature
            or frozen.get('valid_index_sha256') != file_hash(valid_root/'evaluation_index_valid.json')
            or any(frozen.get(k) != v for k,v in expected.items())):
        raise ValueError('Selection differs from the complete valid-only rule')
    return frozen, manifest


def baseline_replay(output,data,specs):
    """Re-evaluated controls are audit roles, not independent scientific repetitions."""
    paths = data['source_metadata'].get('control_folders',{})
    if not paths:
        raise ValueError('Source loader must bind original G/J/product controls')
    from itertools import groupby
    results = {}
    def groups(path):
        with Path(path).open(newline='') as handle:
            for q,rows in groupby(csv.DictReader(handle),key=lambda r:int(r['query_id'])):
                yield q,list(rows)
    from itertools import zip_longest
    for name in ('J_only','G_only','product'):
        if name not in specs:
            continue
        old = Path(paths[name])/'rankings.csv'
        new = Path(output)/data['pool'].split_name/name/'rankings.csv'
        count = 0
        max_error = 0.
        for left,right in zip_longest(groups(old),groups(new)):
            if left is None or right is None or left[0] != right[0]:
                raise ValueError('Control replay query identities changed: '+name)
            old_ids = [int(r['candidate_id']) for r in left[1]][:500]
            new_ids = [int(r['candidate_id']) for r in right[1]]
            if old_ids != new_ids:
                raise ValueError(f'Control ranking replay changed for {name}, query {left[0]}')
            if name != 'J_only':
                for a,b in zip(left[1],right[1]):
                    old_score = float(a.get('similarity_score',a.get('score','nan')))
                    error = abs(old_score-float(b['similarity_score']))
                    if not np.isfinite(error) or error > 2e-7:
                        raise ValueError('Control score replay changed: '+name)
                    max_error = max(max_error,error)
            count += 1
        if count != len(data['pool']):
            raise ValueError('Control replay does not cover every query')
        results[name] = dict(num_queries=count,top500_identical=True,source=str(old),
            source_sha256=file_hash(old),new_sha256=file_hash(new),
            max_score_error=max_error if name!='J_only' else None,
            note='J source uses exp(-D), new endpoint exp(-2D); rank equivalence checked, not equal scores.'
                 if name=='J_only' else 'Frozen scores and ranks replayed; not an independent repetition.')
    record(Path(output)/'baseline_replay.json',dict(status='verified',split=data['pool'].split_name,controls=results))


def summarize(output,split,results,selected):
    ranks = {r['variant']:r['rank'] for r in selected['ranking']}
    with (Path(output)/f'comparison_{split}.csv').open('w',newline='') as handle:
        w=csv.writer(handle)
        w.writerow(['variant','split','valid_rank','global_winner','additive_winner','geometric_winner',
                    'top10_hits','Precision@10','mAP@100','Coverage-nDCG@10'])
        for name in sorted(results,key=ranks.__getitem__):
            row=results[name]
            w.writerow([name,split,ranks[name],name==selected['winner'],
                name==selected['family_winners']['additive'],name==selected['family_winners']['geometric'],
                row['top10_hits'],row['metrics']['Precision@10'],row['metrics']['mAP@100'],
                row['coverage_metrics']['Coverage-nDCG@10']])


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('valid','test','all'),default='valid')
    p.add_argument('--source-run',default=inputs.DEFAULT_SOURCE)
    p.add_argument('--valid-run',help='Completed G/J valid stage; required for test')
    out=p.add_mutually_exclusive_group()
    out.add_argument('--output-root');out.add_argument('--resume-root')
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--check-inputs',action='store_true')
    return p


def output_path(args,split):
    output=(ROOT/(args.resume_root or args.output_root or
                 ('artificial/gj_fusion_'+split+'_'+now().strftime('%Y%m%dT%H%M%S%z')))).resolve()
    allowed=(ROOT/'artificial').resolve()
    if not output.is_relative_to(allowed) or output==allowed:
        raise ValueError('Output must be an isolated artificial/ run')
    for parent in output.parents:
        if parent==allowed:break
        if (parent/'run_manifest.json').exists():raise ValueError('Do not nest in existing runs')
    if args.resume_root:
        old=read_signed(output/'run_manifest.json')
        if old.get('version')!=VERSION or old.get('split')!=split:
            raise ValueError('Resume requires the same G/J stage')
    elif output.exists():
        raise ValueError('Refuse existing output; explicit resume is required')
    return output


def run_stage(args,split):
    output=output_path(args,split)
    source=(ROOT/args.source_run).resolve()
    if source==output or source in output.parents or output in source.parents:
        raise ValueError('Source and output must be isolated')
    frozen=valid_manifest=None
    if split=='test':
        if not args.valid_run:raise ValueError('Test requires --valid-run')
        valid_root=(ROOT/args.valid_run).resolve()
        frozen,valid_manifest=read_frozen(valid_root)
        if str(source)!=valid_manifest['source_run'] or args.device!=valid_manifest['context']['device']:
            raise ValueError('Test source/backend must match frozen valid')
    specs=scoring.frozen_grid() if frozen is None else frozen['selected_specs']
    if args.dry_run:
        print(json.dumps(dict(output=str(output),source=str(source),split=split,variants=specs,
            selection_rule=selection.RULE,training=False,angle=False,test_parameter_search=False),indent=2))
        return output
    started=now()
    data=inputs.load_inputs(source,split,args.device)
    if args.check_inputs:
        print(json.dumps(dict(status='verified_inputs',split=split,queries=len(data['pool']),
            gallery=len(data['pool'].gallery),source_files=len(data['source_binding']),variants=list(specs))))
        return output
    binding=dict(data['source_binding'])
    if frozen is not None:
        for name in ('completed.json','selection.json','run_manifest.json','evaluation_index_valid.json'):
            p=valid_root/name;binding[str(p)]=file_hash(p)
    context=dict(version=VERSION,split=split,source_run=str(source),source_binding=binding,
        code=code_identity(),grid=specs,full_grid=scoring.frozen_grid(),selection_rule=selection.RULE,
        device=args.device,numpy=np.__version__,torch=torch.__version__,structure_strength=2.,
        angular_component=False,local_component=False,training_required=False,posthoc_research=True,
        test_parameter_search=False,score_scale='original; no calibration',
        frozen_selection_signature=frozen['signature'] if frozen else None)
    signature=digest(context)
    if args.resume_root:
        old=read_signed(output/'run_manifest.json')
        if old['context']!=context:raise ValueError('Sources/settings/code changed; create a new run')
    else:
        output.mkdir(parents=True,exist_ok=False)
    with run_lock(output):
        if args.resume_root and verified(output,signature):
            print('Already completed: '+str(output),flush=True);return output
        if not args.resume_root:
            record(output/'run_manifest.json',signed(dict(version=VERSION,split=split,source_run=str(source),
                context=context,started_at=started.isoformat(),
                run_id=started.strftime('%Y%m%dT%H%M%S%z')+'__gj_fusion_'+split,
                valid_run=str(valid_root) if frozen else None)))
        record(output/'inputs.json',dict(source_binding=binding,source_metadata=data['source_metadata'],
               context_signature=signature,source_run=str(source)))
        if frozen:record(output/'frozen_selection.json',frozen)
        protection(output)
        print('OUTPUT_ROOT='+str(output),flush=True)
        results=pipeline.evaluate(output,data,signature,specs,args.device)
        baseline_replay(output,data,specs)
        if split=='valid':
            value=selection.select_candidates(results)
            path=output/'selection.json'
            if path.exists():
                selected=read_signed(path)
                if any(selected.get(k)!=v for k,v in value.items()):raise ValueError('Existing selection changed')
            else:
                selected=signed(dict(**value,context_signature=signature,
                    valid_index_sha256=file_hash(output/'evaluation_index_valid.json'),frozen_at=now().isoformat()))
                record(path,selected)
        else:selected=frozen
        summarize(output,split,results,selected)
        check_binding(binding)
        if context['code']!=code_identity():raise ValueError('Scientific code changed while running')
        protection(output,after=True)
        write_json(output/'status.json',dict(status='completed',split=split,test_evaluated=split=='test',
            winner=selected['winner'],family_winners=selected['family_winners'],context_signature=signature))
        seal(output,signature)
        print('FROZEN CHOICES: '+json.dumps(selected['family_winners']),flush=True)
        print('COMPLETED: '+str(output),flush=True)
    return output


def main(argv=None):
    args=parser().parse_args(argv)
    if args.stage=='all':
        if args.output_root or args.resume_root or args.valid_run or args.check_inputs:
            raise ValueError('all uses independent actual-time valid/test roots; resume stages separately')
        valid=run_stage(args,'valid')
        if not args.dry_run:
            args.valid_run=str(valid)
            run_stage(args,'test')
    else:run_stage(args,args.stage)
    return 0


if __name__=='__main__':raise SystemExit(main())
