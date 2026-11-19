#!/usr/bin/env python3
"""Frozen F/G/WBBS fusion and two branch deletions, isolated and resumable."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import scipy
import torch
from b1_ablation import run as oldrun,register as oldregister
from b1_ablation.analysis import postprocess
from experiments.proposed._shared.ef_suite import exclusive
from experiments.proposed.f_global_fusion import pipeline as fusion
from experiments.proposed.f_global_fusion.run import separate_output
from wbm_retrieval.common import digest,file_hash,read_json,write_json
from fgw_fusion import scoring,pipeline,sources
from fgw_fusion.cache import build_cache

TZ=ZoneInfo('Asia/Shanghai')
VERSION='fgw_fusion_v1'

def signed(value):return {**value,'signature':digest(value)}
def identity():
    result=oldrun.identity()
    for f in sorted(Path(__file__).parent.glob('*.py')):result[str(f.relative_to(ROOT))]=file_hash(f)
    return result

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    out=p.add_mutually_exclusive_group();out.add_argument('--output-root');out.add_argument('--resume-root')
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--check-inputs',action='store_true')
    return p

def choose_output(args):
    output=(ROOT/Path(args.resume_root or args.output_root or ('artificial/fgw_fusion_'+datetime.now(TZ).strftime('%Y%m%dT%H%M%S%z')))).resolve()
    if ROOT/'artificial' not in output.parents or not output.name.startswith('fgw_fusion_'):
        raise ValueError('Use isolated artificial/fgw_fusion_<actual-time> output')
    for parent in output.parents:
        if parent==ROOT/'artificial':break
        if (parent/'run_manifest.json').exists():raise ValueError('Cannot nest inside another experiment')
    if args.resume_root:
        if not (output/'run_manifest.json').is_file():raise ValueError('Resume requires original signed manifest')
    elif output.exists():raise FileExistsError('New output already exists; use explicit signed resume')
    separate_output(output,[sources.SOURCE,sources.REFERENCE,*sources.PROTECTED,*sources.WBBS_RUNS.values(),ROOT/'protocol',ROOT/'results_catalog',ROOT/'artifacts'])
    return output

def analysis_sources(output,split,provenance):
    result={name:{'exact_dir':output/split/name} for name in scoring.specs()}
    result['FGJ']={'exact_dir':sources.REFERENCE/split/'no_angle'}
    reports=provenance['reference']['reports']
    if 'coverage' in reports:
        compared=ROOT.parent/'compared_methods'
        exact=compared/reports['exact']['path'];coverage=compared/reports['coverage']['path']
        result['WBBS']={'exact_dir':exact.parent,'coverage_dir':coverage.parent,'ranking_path':exact.parent/'rankings.csv'}
    return result

def descriptor_audit(pool,bundle,arrays):
    from fgw_fusion.wbbs_backend import original_module
    original=original_module();rows={int(q):i for i,q in enumerate(arrays['sample_ids'])};checked=[]
    for q in sources.selected_queries(pool,bundle.labels):
        i=rows[q];start,end=arrays['offsets'][i:i+2];actual=original.encode_wafer(bundle.maps[q],sources.WBBS_PARAMETERS)
        if (not np.array_equal(actual.points,arrays['points'][start:end]) or actual.original_count!=int(arrays['original_counts'][i])):
            raise ValueError('WBBS retained/raw point identity changed: '+str(q))
        np.testing.assert_array_equal(actual.descriptors,arrays['descriptors'][start:end])
        checked.append(q)
    return {'status':'verified','query_ids':checked,'source':'original raw map -> frozen original WBBS encoder -> exact cached points/descriptors'}

def main(argv=None):
    args=parser().parse_args(argv)
    if not 1<=args.workers<=32:raise ValueError('Use 1..32 bounded W workers')
    output=choose_output(args)
    plan={'version':VERSION,'mode':'all','splits':['valid','test'],'settings':scoring.SETTINGS,
        'specs':scoring.specs(),'source_root':str(sources.SOURCE),'reference_root':str(sources.REFERENCE),
        'wbbs_roots':{k:str(v) for k,v in sources.WBBS_RUNS.items()},'device':'cpu',
        'source_device':'cuda','parameter_search':False,'posthoc_design':True,'training_required':False,
        'fusion':'(.5 F_optimal + .5 G) * raw_WBBS; deletions G*W and F*W; no angular A',
        'full_candidate_rule':'all_gallery_except_query','W_backend':'exact_ckdtree_native_tie_fallback_v1',
        'native_W_audit':'minimum query ID in every class, all candidates; bit-for-bit original CPU equality',
        'source_replay':'all queries W Top500 identities/order/scores must reproduce frozen WBBS',
        'FG_audit':'37 class queries, independent geometry/subset-DP optimum, all candidates and three Top500',
        'bootstrap':{'replicates':10000,'seed':2026,'strata':'37 full label combinations','anchor':'FGW'}}
    if args.dry_run:print(json.dumps({**plan,'output':str(output)},ensure_ascii=False,indent=2));return 0
    torch.set_num_threads(1)
    bundle,contexts=oldrun.load_inputs(sources.SOURCE,'cuda')
    oldregister.proposal(sources.REFERENCE)
    original_manifest=fusion.signed_manifest(sources.SOURCE/'run_manifest.json')
    backends={};arrays={};w_provenance={}
    for split,(pool,*_) in contexts.items():
        backends[split],arrays[split],w_provenance[split]=sources.load_wbbs(split,pool,original_manifest['context'])
        print(f'Verified frozen WBBS {split} descriptors and source evaluation',flush=True)
    if args.check_inputs:print('All frozen sources checked; no run created.',flush=True);return 0
    code=identity()
    context={'plan':plan,'code':code,'source_manifest_sha256':file_hash(sources.SOURCE/'run_manifest.json'),
        'reference_manifest_sha256':file_hash(sources.REFERENCE/'run_manifest.json'),
        'inputs':{s:{**v[4],'wbbs':w_provenance[s]} for s,v in contexts.items()},
        'environment':{'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__,'python':sys.version}}
    with exclusive(output):
        if (output/'run_manifest.json').exists():
            manifest=fusion.signed_manifest(output/'run_manifest.json')
            if manifest['context']!=context:raise ValueError('Resume code/settings/source/environment changed')
            before=read_json(output/'source_protection_before.json')
            if (output/'completed.json').exists():
                from fgw_fusion.register import proposal
                proposal(output)
                print('Verified completed run; no files changed:',output,flush=True)
                return 0
        else:
            now=datetime.now(TZ).replace(microsecond=0)
            manifest=signed({'version':VERSION,'context':context,'batch_id':now.strftime('%Y%m%dT%H%M%S%z')+'__fgw_fusion',
                'started_at':now.isoformat(),'output':str(output),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'launch_command':sys.argv,'workers':args.workers})
            write_json(output/'run_manifest.json',manifest);write_json(output/'plan_before_evaluation.json',plan)
            print('Hashing protected historical source files',flush=True)
            before=sources.inventory();write_json(output/'source_protection_before.json',before)
        try:
            for split,(pool,scorer,descriptors,reference,provenance) in contexts.items():
                write_json(output/'status.json',{'status':'running','phase':'full_candidate_W_cache','split':split})
                inputs=signed({'parent_signature':manifest['signature'],'split':split,'provenance':context['inputs'][split],
                    'specs':scoring.specs(),'code':code})
                pipeline.record(output/f'inputs_{split}.json',inputs)
                cache_signature=digest({'inputs':inputs['signature'],'operation':'full_candidate_W_cache'})
                provider,cache_marker=build_cache(output/'wbbs_cache'/split,cache_signature,pool,backends[split],
                    workers=args.workers,audit_ids=sources.selected_queries(pool,bundle.labels))
                write_json(output/'status.json',{'status':'running','phase':'source_audit','split':split})
                audit_path=output/f'source_audit_{split}.json'
                if not audit_path.exists():
                    replay=sources.verify_wbbs_replay(pool,provider)
                    raw_audit=descriptor_audit(pool,bundle,arrays[split])
                    fg_audit=sources.audit_fg(pool,bundle,scorer,provider)
                    audit=signed({'status':'verified','input_context_signature':digest(context),'split':split,
                        'wbbs_cache_completed_sha256':file_hash(output/'wbbs_cache'/split/'completed.json'),
                        'wbbs_native_full_gallery':cache_marker['native_audits'],'wbbs_all_query_source_replay':replay,
                        'wbbs_raw_descriptor_replay':raw_audit,'fg_full_gallery':fg_audit})
                    write_json(audit_path,audit)
                audit=fusion.signed_manifest(audit_path)
                if (audit.get('status')!='verified' or audit.get('input_context_signature')!=digest(context)
                        or audit['wbbs_cache_completed_sha256']!=file_hash(output/'wbbs_cache'/split/'completed.json')):
                    raise ValueError('Source audit signature differs')
                pipeline.evaluate(output,inputs['signature'],scorer,pool,bundle,scoring.SETTINGS['ks'],2026,provider,scoring.SETTINGS)
                from fgw_fusion.verify import verify_evaluation
                index=read_json(output/f'evaluation_index_{split}.json')
                verification={}
                for name in scoring.specs():
                    write_json(output/'status.json',{'status':'running','phase':'independent_evaluation_verification','split':split,'variant':name})
                    verification[name]=verify_evaluation(output/split/name,pool,bundle,scoring.SETTINGS['ks'],index['variant_signatures'][name],2026)
                write_json(output/f'verification_{split}.json',verification)
                write_json(output/'status.json',{'status':'running','phase':'paired_analysis','split':split})
                postprocess(output/'analysis'/split,analysis_sources(output,split,provenance),bundle.labels,
                    list(map(int,pool)),list(map(int,pool.gallery)),split=split,ks=scoring.SETTINGS['ks'],
                    bootstrap_replicates=10000,seed=2026,anchor='FGW')
            fusion.checked_files(ROOT,code)
            after=sources.inventory()
            if before!=after:raise ValueError('Historical source files changed')
            write_json(output/'source_protection_after.json',{'status':'unchanged','files':len(after),'inventory_digest':digest(after)})
            write_json(output/'status.json',{'status':'completed','mode':'all','test_evaluated':True,'new_evaluations':6,'output':str(output)})
            files={}
            for p in sorted(output.rglob('*')):
                if not p.is_file() or p.name=='completed.json' and p.parent==output:continue
                rel=p.relative_to(output)
                if rel.parts[0] in ('shards','identities') or 'rows' in rel.parts or p.suffix in ('.npz','.npy','.csv','.log'):continue
                if p.name.startswith('.') or any(x.startswith('.') for x in rel.parts):continue
                files[str(rel)]=file_hash(p)
            write_json(output/'completed.json',{'status':'completed','signature':manifest['signature'],'batch_id':manifest['batch_id'],
                'started_at':manifest['started_at'],'new_evaluations':6,'files':files})
            from fgw_fusion.register import write_proposal
            write_proposal(output)
            print('Completed FGW fusion, six verified evaluations:',output,flush=True)
        except BaseException as exc:
            write_json(output/'status.json',{'status':'failed','error_type':type(exc).__name__,'error':str(exc)})
            raise
    return 0

if __name__=='__main__':raise SystemExit(main())
