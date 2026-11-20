#!/usr/bin/env python3
"""Train small region fusers, freeze a valid-only winner, then evaluate that one test recipe."""
from __future__ import annotations
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

import numpy as np
import torch
from experiments.proposed.f_segmentation_geometry import pipeline as geo
from wbm_retrieval.common import digest,file_hash,read_json,write_json
from . import learning,pipeline,scoring,selection,sources
from .state import now,protection,read_signed,record,seal,signed,verified

VERSION = 'f_region_structure_v1'
FAMILIES = ('plain_mlp','gated_mlp')


def code_identity():
    paths = set(Path(__file__).parent.glob('*.py'))
    paths.discard(Path(__file__).with_name('register.py'))
    for folder in ('wbm_retrieval','experiments/proposed/_shared','experiments/proposed/f_segmentation',
                   'experiments/proposed/f_segmentation_geometry','experiments/proposed/f_global_fusion',
                   'b1_structure_fusion','fgw_grid_search'):
        paths.update((ROOT/folder).glob('*.py'))
    paths.add(ROOT/'b1_structure_fusion/config.json')
    return {str(p.relative_to(ROOT)):file_hash(p) for p in sorted(paths)}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('valid','test','all'),default='all')
    out = p.add_mutually_exclusive_group()
    out.add_argument('--output-root')
    out.add_argument('--resume-root')
    p.add_argument('--source-root',default='artifacts/f_segmentation_v1')
    p.add_argument('--feature-roots',nargs='+',default=['artifacts/f_segmentation_radial_area_v1/cc8/features',
                   'artifacts/f_global_fusion_v1/f_cache/features','artifacts/f_fusion_ablation_test_v1/f_cache/features'])
    p.add_argument('--data-file')
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--prepare-workers',type=int,default=4)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--check-inputs',action='store_true')
    return p


def output_path(args):
    output = (ROOT/(args.resume_root or args.output_root or
              ('artificial/f_region_structure_'+now().strftime('%Y%m%dT%H%M%S%z')))).resolve()
    parent = (ROOT/'artificial').resolve()
    if not output.is_relative_to(parent) or output==parent:
        raise ValueError('New experiment output must be under artificial/')
    for p in output.parents:
        if p==parent:
            break
        if (p/'run_manifest.json').exists():
            raise ValueError('Do not nest an experiment inside an existing run')
    if args.resume_root:
        if not (output/'run_manifest.json').exists():
            raise ValueError('Resume requires an existing signed experiment')
    elif output.exists():
        raise ValueError('Refuse an existing output directory; use explicit --resume-root')
    return output


def train_fusers(output,cache,protocol,device):
    if set(map(int,cache['map_ids'])) != set(map(int,protocol.ids['train'])):
        raise ValueError('Fusion training must use exactly the frozen train maps')
    if not np.isin(cache['owners'],protocol.ids['train']).all():
        raise ValueError('Non-train region entered fusion training')
    binding = {'train_cache_sha256':file_hash(cache['folder']/'completed.json'),
               'map_ids_sha256':digest(cache['map_ids'].tolist()),'labels_used':False,
               'fixed_epochs':20,'checkpoint_rule':'final_epoch','seed':2026}
    record(output/'training_inputs.json',binding)
    for family in FAMILIES:
        folder = output/'training'/family
        if (folder/'completed.json').exists():
            learning.load_model(folder/'final.pt',device='cpu')
            continue
        if folder.exists():
            # Preserve partial optimization rather than overwrite it or use an incomplete model.
            folder.rename(folder.with_name(family+'_interrupted_'+now().strftime('%Y%m%dT%H%M%S%z')))
        print('Training new frozen-CNN fusion head:',family,flush=True)
        learning.fit(cache['h_views'],cache['joint_all'],cache['relative_area'],cache['owners'],folder,
                     family=family,device=device,epochs=20,batch_size=512,seed=2026,split='train')
    return {family:file_hash(output/'training'/family/'final.pt') for family in FAMILIES}


def learned_features(output,cache,specs,device):
    result = {}
    for family in FAMILIES:
        if any(spec['family']==family for spec in specs.values()):
            result[family] = learning.encode(output/'training'/family/'final.pt',cache['features'],
                                             cache['joint_all'],cache['relative_area'],device=device)
    return result


def frozen_selection(output,context_signature):
    selected = read_signed(output/'selection.json')
    if selected.get('context_signature')!=context_signature or selected.get('synthetic') is not False:
        raise ValueError('Selection has a different scientific context or is synthetic')
    if selected['valid_index_sha256']!=file_hash(output/'evaluation_index_valid.json'):
        raise ValueError('Frozen valid results changed')
    index = read_json(output/'evaluation_index_valid.json')
    if index['specs']!=scoring.frozen_grid() or index['context_signature']!=context_signature:
        raise ValueError('Selection did not cover the complete predefined valid grid')
    for name,result in index['results'].items():
        folder = output/'valid'/name
        expected_signature = digest({'context':context_signature,'split':'valid','variant':name,'spec':index['specs'][name]})
        if not verified(folder,expected_signature) or result['completed_sha256']!=file_hash(folder/'completed.json'):
            raise ValueError('Frozen valid evaluation is incomplete or changed')
    expected = selection.select_best(index['results'],split='valid')
    if any(selected.get(k)!=v for k,v in expected.items()):
        raise ValueError('Selection disagrees with the predefined integer objective and tie rules')
    return selected


def baseline_replay(output,feature_roots):
    new = output/'valid/baseline/rankings.csv'
    old = next((p.parent.parent/'valid/best_nce/rankings.csv' for p in feature_roots
                if (p.parent.parent/'valid/best_nce/rankings.csv').exists()),None)
    # The standard geometry feature root is cc8/features; its sibling is cc8/valid.
    if old is None:
        old = next((p.parent/'valid/best_nce/rankings.csv' for p in feature_roots
                    if (p.parent/'valid/best_nce/rankings.csv').exists()),None)
    if old is None:
        raise FileNotFoundError('Original cc8 F valid ranking is needed for baseline replay')
    count,error = 0,0.
    with new.open(newline='') as a, old.open(newline='') as b:
        left,right = csv.DictReader(a),csv.DictReader(b)
        for x,y in __import__('itertools').zip_longest(left,right):
            if x is None or y is None or any(x[k]!=y[k] for k in ('query_id','rank','candidate_id')):
                raise ValueError('Original F baseline identities/order were not reproduced')
            error=max(error,abs(float(x['similarity_score'])-float(y['similarity_score'])))
            count+=1
    if error>1e-12:
        raise ValueError('Original F baseline score replay mismatch')
    record(output/'baseline_replay.json',{'status':'verified','rows':count,'max_score_error':error,
                                          'source':str(old),'source_sha256':file_hash(old)})


def summary(output,split,results):
    path = output/f'comparison_{split}.csv'
    with path.open('w',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['variant','split','top10_hits','Precision@10','mAP@100','Coverage-nDCG@10'])
        for name,row in results.items():
            writer.writerow([name,split,row['top10_hits'],row['metrics']['Precision@10'],
                             row['metrics']['mAP@100'],row['coverage_metrics']['Coverage-nDCG@10']])


def main(argv=None):
    args=parser().parse_args(argv)
    if args.prepare_workers<0 or args.prepare_workers>16:
        raise ValueError('prepare-workers must be in [0,16]')
    if args.stage=='test' and not args.resume_root:
        raise ValueError('Test requires --resume-root of a completed valid search')
    output=output_path(args)
    if args.dry_run:
        print(__import__('json').dumps({'output':str(output),'stage':args.stage,'grid':scoring.frozen_grid(),
              'selection_rule':selection.RULE,'training':'two fixed 20-epoch heads; train only; frozen CNN',
              'test':'one valid-selected recipe only'},ensure_ascii=False,indent=2))
        return 0
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('Activate the existing remote PyTorch CUDA environment')
    source_root=(ROOT/args.source_root).resolve()
    feature_roots=[(ROOT/p).resolve() for p in args.feature_roots]
    data_file=(ROOT/args.data_file).resolve() if args.data_file else None
    with geo.source_lock(source_root):
        config,suite,checkpoint,protocol,source_identity=sources.load(source_root,data_file)
        for split in ('train','valid'):
            sources.feature_source(split,config,source_identity['geometry'],feature_roots)
        if args.check_inputs:
            print('Frozen model, train regions, valid features and protocol verified',flush=True)
            return 0
        context={'version':VERSION,'source':source_identity,'code':code_identity(),'grid':scoring.frozen_grid(),
                 'selection_rule':selection.RULE,'device':args.device,'numpy':np.__version__,'torch':torch.__version__,
                 'training':{'epochs':20,'batch_size':512,'seed':2026,'checkpoint_rule':'final_epoch'},
                 'matching':config['matching'],'posthoc_design':True,'test_parameter_search':False}
        context_signature=digest(context)
        if args.resume_root:
            manifest=read_signed(output/'run_manifest.json')
            if manifest['context']!=context:
                raise ValueError('Source/code/environment changed; resume cannot mix scientific settings')
            if verified(output,context_signature):
                print('Experiment already completed and verified:',output,flush=True)
                return 0
        else:
            output.mkdir(parents=True,exist_ok=False)
            time=now()
            record(output/'run_manifest.json',signed({'version':VERSION,'context':context,'started_at':time.isoformat(),
                   'run_id':time.strftime('%Y%m%dT%H%M%S%z')+'__f_region_structure','requested_stage':args.stage,
                   'source_root':str(source_root),'feature_roots':list(map(str,feature_roots))}))
        print('OUTPUT_ROOT='+str(output),flush=True)
        protection(output)
        with np.load(config['data_file'],allow_pickle=False) as archive:
            maps,labels=archive['arr_0'],archive['arr_1']
        bundle=pipeline.bundle_for(labels,protocol)
        if args.stage!='test':
            train=sources.prepare(output,'train',maps,config,suite,checkpoint,protocol,source_identity,
                                  feature_roots,context_signature,args.prepare_workers,args.device)
            models=train_fusers(output,train,protocol,args.device)
            record(output/'model_binding.json',models)
            del train
            valid=sources.prepare(output,'valid',maps,config,suite,checkpoint,protocol,source_identity,
                                  feature_roots,context_signature,args.prepare_workers,args.device)
            specs=scoring.frozen_grid()
            learned=learned_features(output,valid,specs,args.device)
            results=pipeline.evaluate(output,valid,protocol.retrieval.valid,bundle,config['matching'],
                                      context_signature,specs,learned)
            baseline_replay(output,feature_roots)
            summary(output,'valid',results)
            selected=selection.select_best(results,split='valid')
            record(output/'selection.json',signed({**selected,'context_signature':context_signature,
                  'valid_index_sha256':file_hash(output/'evaluation_index_valid.json'),
                  'models':models,'frozen_at':now().isoformat()})) if not (output/'selection.json').exists() else None
            del valid,learned
        selected=frozen_selection(output,context_signature)
        if any(file_hash(output/'training'/k/'final.pt')!=h for k,h in selected['models'].items()):
            raise ValueError('Fusion models changed after valid selection')
        print('VALID WINNER: '+selected['winner'],flush=True)
        if args.stage in ('all','test'):
            test=sources.prepare(output,'test',maps,config,suite,checkpoint,protocol,source_identity,
                                 feature_roots,context_signature,args.prepare_workers,args.device)
            specs={selected['winner']:selected['selected_spec']}
            learned=learned_features(output,test,specs,args.device)
            results=pipeline.evaluate(output,test,protocol.retrieval.test,bundle,config['matching'],
                                      context_signature,specs,learned)
            summary(output,'test',results)
        for name,expected in context['code'].items():
            if file_hash(ROOT/name)!=expected:
                raise ValueError('Scientific source code changed during execution')
        fresh=sources.load(source_root,data_file)[-1]
        if fresh!=source_identity:
            raise ValueError('Frozen model/data/protocol changed during execution')
        protection(output,after=True)
        write_json(output/'status.json',{'status':'completed','stage':args.stage,'winner':selected['winner'],
                   'test_evaluated':args.stage in ('all','test'),'context_signature':context_signature})
        if args.stage in ('all','test'):
            seal(output,context_signature)
        else:
            record(output/'valid_completed.json',{'status':'completed','selection_sha256':file_hash(output/'selection.json'),
                                                  'context_signature':context_signature})
        print('COMPLETED: '+str(output),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
