#!/usr/bin/env python3
"""Evaluate the one frozen valid-selected additive recipe on the full test pool."""
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
from f_region_score_search.run import code_identity as valid_code_identity, check_sources, run_lock
from wbm_retrieval.common import digest, file_hash, write_json
from . import inputs, pipeline
from .state import now, protection, read_signed, record, seal, verified

VERSION = 'f_region_score_test_v1'


def code_identity():
    result = valid_code_identity()
    paths = [ROOT/'f_region_score_search/register.py',
             ROOT/'f_region_structure/register.py', ROOT/'results_catalog/build.py']
    paths += [p for p in Path(__file__).parent.glob('*.py') if p.name != 'register.py']
    for path in sorted(paths):
        result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',default=str(inputs.DEFAULT_SOURCE))
    out = p.add_mutually_exclusive_group()
    out.add_argument('--output-root')
    out.add_argument('--resume-root')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--check-inputs',action='store_true')
    return p


def output_path(args):
    output = (ROOT/(args.resume_root or args.output_root or
                   ('artificial/f_region_score_test_'+now().strftime('%Y%m%dT%H%M%S%z')))).resolve()
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
        old = read_signed(output/'run_manifest.json')
        if (old.get('version') != VERSION or old.get('requested_stage') != 'test'
                or not (output/'frozen_selection.json').is_file()):
            raise ValueError('Resume is restricted to an existing isolated regional score test')
    elif output.exists():
        raise ValueError('Refuse an existing output; explicit --resume-root is required')
    return output


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1 <= args.workers <= 8:
        raise ValueError('workers must be between 1 and 8')
    output, source = output_path(args), (ROOT/args.source_run).resolve()
    if source == output or source in output.parents:
        raise ValueError('Output cannot overwrite or nest in its source')
    if args.dry_run:
        print(json.dumps({'output':str(output),'source_run':str(source),'split':'test',
              'variant':inputs.VARIANT,'spec':inputs.SPEC,'training':False,
              'test_parameter_search':False,'selection_policy':'frozen_valid_winner_only',
              'workers':args.workers},indent=2))
        return 0
    data = inputs.load(source)
    if args.check_inputs:
        print(json.dumps({'status':'verified_inputs','test_queries':len(data['protocol'].retrieval.test),
              'test_gallery':len(data['cache']['map_ids']),'source_files':len(data['source_binding']),
              'cache_source_run':str(data['cache_source_run']),'frozen_variant':inputs.VARIANT,
              'training':False,'test_parameter_search':False}))
        return 0
    if not args.resume_root:
        output.mkdir(parents=True,exist_ok=False)
    elif read_signed(output/'frozen_selection.json') != data['frozen_selection']:
        raise ValueError('Resume frozen valid selection differs')
    with run_lock(output):
        record(output/'frozen_selection.json',data['frozen_selection'])
        context = {'version':VERSION,'split':'test','frozen_variant':inputs.VARIANT,
                   'frozen_spec':inputs.SPEC,'selection_policy':'frozen_valid_winner_only',
                   'frozen_selection_sha256':file_hash(output/'frozen_selection.json'),
                   'source_run':str(source),'cache_source_run':str(data['cache_source_run']),
                   'source_binding':data['source_binding'],'code':code_identity(),
                   'matching':data['config']['matching'],'numpy':np.__version__,'torch':torch.__version__,
                   'posthoc_design':True,'test_parameter_search':False,'training':False}
        signature = digest(context)
        if args.resume_root:
            old = read_signed(output/'run_manifest.json')
            if old.get('context') != context:
                raise ValueError('Frozen context changed; use a new run')
            if verified(output,signature):
                print('Already completed and verified: '+str(output),flush=True)
                return 0
        else:
            from .state import signed
            time = now()
            record(output/'run_manifest.json',signed({'version':VERSION,'context':context,
                'started_at':time.isoformat(),'run_id':time.strftime('%Y%m%dT%H%M%S%z')+'__f_region_score_test',
                'source_run':str(source),'cache_source_run':str(data['cache_source_run']),
                'workers':args.workers,'requested_stage':'test'}))
        record(output/'inputs.json',{'source_binding':data['source_binding'],'context_signature':signature,
               'source_run':str(source),'cache_source_run':str(data['cache_source_run'])})
        protection(output)
        print('OUTPUT_ROOT='+str(output),flush=True)
        results = pipeline.evaluate(output,data['cache'],data['protocol'].retrieval.test,
                                    data['bundle'],data['config']['matching'],signature,args.workers)
        with (output/'comparison_test.csv').open('w',newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['variant','split','top10_hits','Precision@10','mAP@100','Coverage-nDCG@10'])
            row = results[inputs.VARIANT]
            writer.writerow([inputs.VARIANT,'test',row['top10_hits'],row['metrics']['Precision@10'],
                             row['metrics']['mAP@100'],row['coverage_metrics']['Coverage-nDCG@10']])
        check_sources(data['source_binding'])
        for name, expected in context['code'].items():
            if file_hash(ROOT/name) != expected:
                raise ValueError('Scientific code changed during evaluation: '+name)
        protection(output,after=True)
        write_json(output/'status.json',{'status':'completed','split':'test','test_evaluated':True,
                   'test_parameter_search':False,'winner':inputs.VARIANT,'context_signature':signature})
        seal(output,signature)
        print('COMPLETED: '+str(output),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
