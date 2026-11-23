"""Run only the five prespecified augmentations, then validate and seal every result."""
from pathlib import Path
from contextlib import contextmanager
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time

from .state import ROOT, MODELS, VARIANTS, now, record, run_context, read_signed, verified
from wbm_retrieval.common import read_json


def alive(receipt):
    p=Path('/proc')/str(receipt['pid'])
    try:
        state=(p/'stat').read_text().rsplit(')',1)[1].split()[0]
        argv=[os.fsdecode(x) for x in (p/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')]
    except (FileNotFoundError,ProcessLookupError):
        return False
    return state!='Z' and argv==receipt['command']


@contextmanager
def lock(output):
    (output/'logs').mkdir(exist_ok=True)
    with (output/'logs/followthrough.lock').open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def gpu_free(gpu):
    raw=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid',
                                  '--format=csv,noheader,nounits'],text=True).strip()
    if raw:
        raise RuntimeError('GPU '+str(gpu)+' has another compute process; no oversubscription: '+raw)


def launch(output,name,stage,*,model=None,gpu=None):
    if stage=='train' and model in VARIANTS and name=='train_'+model:
        extra=['--variant',model,'--workers','4']
    elif stage=='evaluate' and model in MODELS and name=='evaluate_'+model:
        extra=['--models',model,'--splits','valid','test','--workers','8']
    elif stage in ('report','finalize') and name==stage and model is None:
        extra=[]
    else:raise ValueError('Not an authorized fixed stage')
    receipt=output/('launch_'+name+'.json')
    if receipt.exists():raise ValueError('Refusing duplicate launch: '+name)
    if gpu is not None:gpu_free(gpu)
    command=[sys.executable,'-u','-m','cnn_augmentation.run','--output-root',str(output),
             '--stage',stage,'--device','cpu' if gpu is None else 'cuda:0',*extra]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
             OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    (output/'logs').mkdir(exist_ok=True)
    with (output/'logs'/(name+'.log')).open('x') as stream:
        rd,wr=os.pipe()
        bootstrap=('import os,sys; fd=int(sys.argv[1]); token=os.read(fd,1); os.close(fd); '
                   'sys.exit(73) if token!=b"1" else os.execv(sys.argv[2],sys.argv[2:])')
        try:
            process=subprocess.Popen([sys.executable,'-c',bootstrap,str(rd),*command],cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(rd,))
            os.close(rd);rd=None
            value={'pid':process.pid,'command':command,'physical_gpu':gpu,'started_at':now().isoformat()}
            record(receipt,value);os.write(wr,b'1')
        finally:
            if rd is not None:os.close(rd)
            os.close(wr)
    print(json.dumps({'launched':name,**value}),flush=True)
    return {'process':process,'receipt':value,'name':name,'stage':stage,'model':model,'gpu':gpu}


def training_complete(output,model,context):
    p=output/'training'/model
    if not (p/'completed.json').exists():return False
    m=read_signed(p/'run_manifest.json');s=read_json(p/'training_summary.json')
    if (m['science'].get('refinement_context_signature')!=context or not verified(p,m['training_signature'])
            or s.get('completed') is not True or s.get('epochs_completed')!=100 or s.get('optimizer_steps')!=39000):
        raise ValueError('Invalid completed training: '+model)
    return True


def evaluation_complete(output,model):
    if not (output/'evaluation'/model/'evaluation_index_test.json').exists():return False
    from .register import validate_model_split
    for split in ('valid','test'):validate_model_split(output,model,split)
    return True


def stage_finished(output,job,context):
    process=job['process']
    if process is not None:
        code=process.poll()
        if code is None:return False
        if code:raise RuntimeError(job['name']+' failed, exit '+str(code)+'; inspect retained log')
    elif alive(job['receipt']):return False
    if job['stage']=='train':complete=training_complete(output,job['model'],context)
    elif job['stage']=='evaluate':complete=evaluation_complete(output,job['model'])
    elif job['stage']=='report':
        p=output/'report';meta=read_json(p/'manifest.json')
        complete=meta.get('context_signature')==context and verified(p,meta['signature'])
    else:
        from .register import verify_run
        verify_run(output);complete=True
    if not complete:raise RuntimeError(job['name']+' exited without sealed validated results')
    return True


def adopt(output,name,stage,model=None):
    p=output/('launch_'+name+'.json')
    if not p.exists():return None
    receipt=read_json(p)
    if not alive(receipt):raise RuntimeError('Previous '+name+' stopped before verified completion; inspect, no automatic retry')
    return {'process':None,'receipt':receipt,'name':name,'stage':stage,'model':model,'gpu':receipt['physical_gpu']}


def supervise(output,gpus,*,resume=False):
    manifest=run_context(output);context=manifest['signature']
    if len(gpus)!=len(set(gpus)) or not gpus:raise ValueError('Distinct allowed GPUs required')
    queue=['crop','rotation','cutout','noise','shift']
    trained=set();evaluated=set();active=[]
    for model in MODELS:
        evaluation_receipt=output/('launch_evaluate_'+model+'.json')
        evaluation_running=bool(resume and evaluation_receipt.exists() and alive(read_json(evaluation_receipt)))
        if evaluation_running:
            active.append(adopt(output,'evaluate_'+model,'evaluate',model))
        elif evaluation_complete(output,model):evaluated.add(model)
        elif resume:
            job=adopt(output,'evaluate_'+model,'evaluate',model)
            if job:active.append(job)
        if model!='baseline':
            training_receipt=output/('launch_train_'+model+'.json')
            training_running=bool(resume and training_receipt.exists() and alive(read_json(training_receipt)))
            if training_running:
                active.append(adopt(output,'train_'+model,'train',model))
            elif training_complete(output,model,context):trained.add(model)
            elif resume:
                job=adopt(output,'train_'+model,'train',model)
                if job:active.append(job)
    if evaluated==set(MODELS):print('All model evaluations already validated',flush=True)
    if 'baseline' not in evaluated and not any(j['model']=='baseline' for j in active):
        active.append(launch(output,'evaluate_baseline','evaluate',model='baseline',gpu=None))
    last_update=0.
    while len(evaluated)<len(MODELS) or active:
        for job in active[:]:
            if stage_finished(output,job,context):
                active.remove(job)
                (trained if job['stage']=='train' else evaluated).add(job['model'])
                print(json.dumps({'completed':job['name'],'at':now().isoformat()}),flush=True)
        occupied={j['gpu'] for j in active if j['gpu'] is not None}
        working={j['model'] for j in active}
        for gpu in gpus:
            if gpu in occupied:continue
            pending=[m for m in queue if m not in trained and m not in working]
            if pending:
                model=pending[0];stage='train'
            else:
                pending=[m for m in queue if m in trained and m not in evaluated and m not in working]
                if not pending:continue
                model=pending[0];stage='evaluate'
            active.append(launch(output,stage+'_'+model,stage,model=model,gpu=gpu));working.add(model)
        if time.monotonic()-last_update>=60:
            progress={m:(read_json(output/'training'/m/'status.json') if (output/'training'/m/'status.json').exists()
                         else {'status':'queued'}) for m in VARIANTS}
            print(json.dumps({'at':now().isoformat(),'training':progress,'evaluated':sorted(evaluated)}),flush=True)
            last_update=time.monotonic()
        if active:time.sleep(30)
        elif len(evaluated)<len(MODELS):raise RuntimeError('Queue stalled without an active process')
    for stage in ('report','finalize'):
        if stage=='report' and (output/'report/completed.json').exists():
            if not resume:raise ValueError('Report already exists; use --resume to validate/reuse')
            job={'process':None,'receipt':{'pid':-1,'command':[]},'stage':stage,'name':stage,'model':None}
        elif stage=='finalize' and (output/'completed.json').exists():
            job={'process':None,'receipt':{'pid':-1,'command':[]},'stage':stage,'name':stage,'model':None}
        else:
            job=adopt(output,stage,stage) if resume else None
            job=job or launch(output,stage,stage)
        while not stage_finished(output,job,context):time.sleep(30)
    print('Scientific results sealed; catalogue registration remains a separate verified step.',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output-root',required=True,type=Path)
    p.add_argument('--gpus',nargs='+',default=['1','2','3']);p.add_argument('--resume',action='store_true')
    a=p.parse_args(argv);output=a.output_root.resolve()
    with lock(output):supervise(output,a.gpus,resume=a.resume)


if __name__=='__main__':main()
