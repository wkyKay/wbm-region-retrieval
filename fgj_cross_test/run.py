"""Re-evaluate six fixed F/G/J/WBBS formulas on the complete exact-excluded pool."""
import argparse
import csv
from itertools import groupby, zip_longest
import os
from pathlib import Path
import subprocess
import sys

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from wbm_retrieval.common import digest,file_hash,read_json,write_json
from f_region_score_search.run import run_lock
from . import inputs,pipeline
from .state import (VERSION,VARIANTS,SPECS,SETTINGS,PROTOCOL,CANDIDATE_RULE,
    fixed_variants,now,record,signed,read_signed,protection,seal,verified,verify_run)

SCIENCE_FILES=('__init__.py','scoring.py','inputs.py','pipeline.py','run.py','state.py','verify.py')


def code_identity():
    from gj_fusion.run import code_identity as gj_code
    from fgw_grid_search.run import code_identity as fgw_code
    result={**gj_code(),**fgw_code()}
    for name in SCIENCE_FILES:
        p=Path(__file__).parent/name; result[str(p.relative_to(ROOT))]=file_hash(p)
    for folder in ('fgw_cross_test','fw_cross_test'):
        for p in sorted((ROOT/folder).glob('*.py')):
            result[str(p.relative_to(ROOT))]=file_hash(p)
    return result


def check_binding(binding):
    for name,expected in binding.items():
        p=Path(name)
        if not p.is_absolute() or not p.is_file() or file_hash(p)!=expected:
            raise ValueError('Frozen source changed: '+name)


def baseline_replay(output,data):
    controls=data['source_metadata']['controls']
    if set(controls)!={'F_ONLY','WBBS'}:
        raise ValueError('Both historical cross-combination controls are required')
    def groups(path):
        with Path(path).open(newline='') as handle:
            for q,rows in groupby(csv.DictReader(handle),key=lambda x:int(x['query_id'])):
                yield q,list(rows)
    results={}
    for name,folder in controls.items():
        old=Path(folder)/'rankings.csv'; new=Path(output)/'test'/name/'rankings.csv'
        maximum=0.; count=0
        for left,right in zip_longest(groups(old),groups(new)):
            if left is None or right is None or left[0]!=right[0]:
                raise ValueError('Historical control query identities differ: '+name)
            if [int(x['candidate_id']) for x in left[1]][:500] != [int(x['candidate_id']) for x in right[1]]:
                raise ValueError('Historical control Top-500 differs: '+name)
            for a,b in zip(left[1],right[1]):
                error=abs(float(a.get('similarity_score',a.get('score','nan')))-float(b['similarity_score']))
                if not 0<=error<=1e-12:raise ValueError('Historical control score differs: '+name)
                maximum=max(maximum,error)
            count+=1
        if count!=7199:raise ValueError('Incomplete historical control replay')
        results[name]=dict(num_queries=count,top500_identical=True,source=str(old),
            source_sha256=file_hash(old),new_sha256=file_hash(new),max_score_error=maximum)
    context_signature=digest(read_signed(Path(output)/'run_manifest.json')['context'])
    record(Path(output)/'baseline_replay.json',dict(status='verified',controls=results,
        context_signature=context_signature))


def summarize(output,results):
    metrics=list(next(iter(results.values()))['coverage_metrics'])
    with (Path(output)/'comparison.csv').open('w',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['variant','formula',*metrics])
        for name in VARIANTS:writer.writerow([name,SPECS[name]['formula'],
            *[results[name]['coverage_metrics'][k] for k in metrics]])
    with (Path(output)/'metric_orderings.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['metric','rank','variant','value','direction'])
        for metric in metrics:
            values={n:results[n]['coverage_metrics'][metric] for n in VARIANTS}
            names=sorted(VARIANTS,key=lambda n:(-(values[n] if values[n] is not None else float('-inf')),n))
            for name in names:
                value=values[name]
                rank=1+sum(v>value for v in values.values() if v is not None) if value is not None else ''
                writer.writerow([metric,rank,name,value,'higher_is_better'])
    record(Path(output)/'comparison_design.json',dict(primary_display_metric='Coverage-nDCG@10',
        ranking_is_descriptive_only=True,test_selects_parameters=False,primary_variant=None,
        display_ties='equal values share competition rank; name only orders display'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fgw-root',default=inputs.DEFAULT_FGW)
    parser.add_argument('--gj-root',default=inputs.DEFAULT_GJ)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    choice=parser.add_mutually_exclusive_group()
    choice.add_argument('--output-root');choice.add_argument('--resume-root')
    parser.add_argument('--check-inputs',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    started=now()
    output=(ROOT/(args.resume_root or args.output_root or
        ('artificial/fgj_cross_test_'+started.strftime('%Y%m%dT%H%M%S%z')))).resolve()
    allowed=(ROOT/'artificial').resolve()
    if output==allowed or not output.is_relative_to(allowed):raise ValueError('Output must be an isolated artificial/ run')
    for parent in output.parents:
        if parent==allowed:break
        if (parent/'run_manifest.json').exists():raise ValueError('Do not nest runs')
    if output.exists() and not args.resume_root:raise FileExistsError('Use a new output directory')
    if args.dry_run:
        print(dict(output=str(output),variants=SPECS,protocol=PROTOCOL,settings=SETTINGS,training=False))
        return 0
    data=inputs.load_inputs(args.fgw_root,args.gj_root,args.device)
    if args.check_inputs:
        print(dict(status='verified_inputs',queries=len(data['pool']),
            source_files=len(data['source_binding']),derived_protocol=data['pool'].metadata))
        return 0
    data['source_binding_signature']=digest(data['source_binding'])
    import numpy as np
    import torch
    torch.set_num_threads(1)
    frozen=fixed_variants()
    context=dict(version=VERSION,split='test',variants=VARIANTS,specs=SPECS,settings=SETTINGS,
        code=code_identity(),source_binding=data['source_binding'],source_metadata=data['source_metadata'],
        protocol=PROTOCOL,candidate_rule=CANDIDATE_RULE,derived_protocol=data['pool'].metadata,
        retrieval_protocol_signature=data['pool'].protocol_signature,
        base_protocol_signature=data['base_pool'].protocol_signature,
        f_version='cc8_optimal_radial_area',structure_strength=2.0,
        device=args.device,numpy=np.__version__,torch=torch.__version__,
        primary_variant=None,selection_rule='user_specified_posthoc_components',
        test_parameter_search=False,formal_protocol=True,training_required=False,posthoc_research=True,
        frozen_variants_signature=frozen['signature'],normalization_scope='none')
    signature=digest(context)
    if args.resume_root:
        old=read_signed(output/'run_manifest.json')
        if old.get('version')!=VERSION or old.get('context')!=context:
            raise ValueError('Sources/settings/scientific code changed; create a new run')
    else: output.mkdir(parents=True,exist_ok=False)
    with run_lock(output):
        if args.resume_root and verified(output,signature):
            verify_run(output);print('Already complete: '+str(output),flush=True);return 0
        if not args.resume_root:
            record(output/'run_manifest.json',signed(dict(version=VERSION,stage='test',split='test',
                context=context,started_at=started.isoformat(),
                run_id=started.strftime('%Y%m%dT%H%M%S%z')+'__fgj_cross_test',
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())))
        record(output/'frozen_variants.json',frozen)
        record(output/'settings.json',SETTINGS)
        record(output/'candidate_protocol.json',data['pool'].metadata)
        record(output/'inputs.json',dict(source_binding=data['source_binding'],source_metadata=data['source_metadata']))
        audit_path=output/'protocol/cross_combination_candidates.csv'
        if not audit_path.exists():
            audit_path.parent.mkdir(parents=True,exist_ok=True)
            with audit_path.open('x',newline='') as handle:
                w=csv.DictWriter(handle,fieldnames=list(data['pool'].audit[0]));w.writeheader();w.writerows(data['pool'].audit)
        else:
            with audit_path.open(newline='') as handle:
                actual=[{k:int(v) for k,v in r.items()} for r in csv.DictReader(handle)]
            if actual!=data['pool'].audit:raise ValueError('Candidate audit changed')
        protection(output)
        print('OUTPUT_ROOT='+str(output),flush=True)
        try:
            provider=inputs.Components(data,args.device)
            record(output/'component_audit.json',dict(**inputs.audit_components(data,provider),
                context_signature=signature))
            results=pipeline.evaluate(output,data,signature,provider)
            print('Replaying historical cross F and WBBS...',flush=True)
            baseline_replay(output,data)
            summarize(output,results)
            check_binding(data['source_binding'])
            if context['code']!=code_identity():raise ValueError('Scientific code changed while running')
            protection(output,after=True)
            write_json(output/'status.json',dict(status='completed',stage='test',split='test',
                context_signature=signature,test_evaluated=True,new_evaluations=6,finished_at=now().isoformat()))
            seal(output,signature)
            verify_run(output,check_source=False)
            print('COMPLETED: '+str(output),flush=True)
        except BaseException as error:
            if not (output/'completed.json').exists():
                write_json(output/'status.json',dict(status='failed',error_type=type(error).__name__,error=str(error)))
            raise
    return 0


if __name__=='__main__':raise SystemExit(main())
