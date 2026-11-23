"""Continue the already-started two runs through evaluation and sealed reports.

This supervisor never trains a third model, changes a recipe, or retries a failed
scientific stage. Failure is surfaced with its existing evidence retained.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time

from .state import ROOT, MODELS, VARIANTS, now, record, read_signed, run_context, verified
from wbm_retrieval.common import digest, read_json


def alive(pid, command=None):
    path = Path('/proc')/str(pid)
    try:
        state = (path/'stat').read_text().rsplit(')', 1)[1].split()[0]
        actual = [os.fsdecode(v) for v in (path/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')]
    except (FileNotFoundError, ProcessLookupError):
        return False
    return state != 'Z' and (actual == command if command is not None else 'cnn_refinement.run' in actual)


@contextmanager
def supervisor_lock(output):
    (output/'logs').mkdir(exist_ok=True)
    with (output/'logs/followthrough.lock').open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another followthrough supervisor already owns this run') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def launch(output, name, stage, extra=(), gpu='2'):
    allowed = {'negative_diagnostics': ('negative-diagnostics', ()),
        'representation_diagnostics': ('diagnose', ()), 'report': ('report', ()), 'finalize': ('finalize', ())}
    allowed.update({'evaluate_'+m: ('evaluate', ('--models', m, '--splits', 'valid', 'test')) for m in VARIANTS})
    if name not in allowed or (stage, tuple(extra)) != allowed[name]:
        raise ValueError('Supervisor may only launch fixed downstream stages, never training/teacher/init')
    receipt = output/('launch_'+name+'.json')
    if receipt.exists():
        raise ValueError('A stage was already launched; inspect its outcome before any retry: '+name)
    command = [sys.executable, '-u', '-m', 'cnn_refinement.run', '--output-root', str(output),
               '--stage', stage, '--device', 'cuda:0', '--workers', '8', *extra]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    (output/'logs').mkdir(exist_ok=True)
    # Exclusive log creation also blocks duplicate launch after a supervisor crash.
    # A small exec gate ensures finalization cannot start before its receipt exists.
    with (output/'logs'/(name+'.log')).open('x') as stream:
        read_fd, write_fd = os.pipe()
        bootstrap = ('import os,sys; fd=int(sys.argv[1]); token=os.read(fd,1); os.close(fd); '
                     'sys.exit(73) if token!=b"1" else os.execv(sys.argv[2],sys.argv[2:])')
        try:
            process = subprocess.Popen([sys.executable, '-c', bootstrap, str(read_fd), *command],
                cwd=ROOT, env=environment, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(read_fd,))
            os.close(read_fd)
            read_fd = None
            record(receipt, {'pid': process.pid, 'command': command, 'physical_gpu': gpu,
                             'started_at': now().isoformat()})
            os.write(write_fd, b'1')
        finally:
            if read_fd is not None:
                os.close(read_fd)
            os.close(write_fd)
    print(f'{now().isoformat()} launched {name} pid={process.pid}', flush=True)
    return process


def require_running(output, name):
    receipt = read_json(output/('launch_'+name+'.json'))
    if not alive(receipt['pid'], receipt['command']):
        raise RuntimeError('Stage exited before verified completion; inspect logs/'+name+'.log')


def stage_complete(output, stage, manifest):
    if stage == 'finalize':
        if not (output/'completed.json').exists():
            return False
        from .register import verify_run
        verify_run(output)
        return True
    folder = output/{'negative-diagnostics': 'negative_diagnostics', 'diagnose': 'diagnostics', 'report': 'report'}[stage]
    if not (folder/'completed.json').exists():
        return False
    if stage == 'negative-diagnostics':
        meta = read_signed(folder/'manifest.json')
        signature, science = meta['diagnostic_signature'], meta['science']
    else:
        meta = read_json(folder/'manifest.json')
        signature, science = meta['signature'], {k: v for k, v in meta.items() if k != 'signature'}
    if signature != digest(science) or science.get('context_signature') != manifest['signature']:
        raise ValueError('Completed downstream stage belongs to another context: '+stage)
    if not verified(folder, signature) or read_json(folder/'summary.json').get('signature') != signature:
        raise ValueError('Downstream stage/summary is not sealed: '+stage)
    return True


def synchronous(output, name, stage, *, gpu='2', resume=False, manifest=None):
    manifest = run_context(output) if manifest is None else manifest
    if stage_complete(output, stage, manifest):
        if not resume:
            raise ValueError('Stage already completed; use --resume for read-only reuse: '+name)
        return
    receipt_path = output/('launch_'+name+'.json')
    if receipt_path.exists():
        if not resume:
            raise ValueError('Stage was already launched; use --resume only to follow a live process: '+name)
        receipt = read_json(receipt_path)
        require_running(output, name)
        while alive(receipt['pid'], receipt['command']):
            time.sleep(30)
    else:
        process = launch(output, name, stage, gpu=gpu)
        while process.poll() is None:
            time.sleep(30)
        if process.returncode != 0:
            raise RuntimeError(f'{name} failed with exit {process.returncode}; original outputs retained')
    if not stage_complete(output, stage, manifest):
        raise RuntimeError(name+' exited without verified scientific completion; no automatic retry')


def training_complete(output, model, manifest):
    folder = output/'training'/model
    if not (folder/'completed.json').exists():
        return False
    meta = read_signed(folder/'run_manifest.json')
    summary = read_json(folder/'training_summary.json')
    if (meta['science'].get('refinement_context_signature') != manifest['signature']
            or not verified(folder, meta['training_signature']) or summary.get('completed') is not True
            or summary.get('epochs_completed') != 100 or summary.get('optimizer_steps') != 39000):
        raise ValueError('Training completion is unsealed, incomplete or from another context: '+model)
    return True


def evaluation_complete(output, model, process=None):
    if process is not None:
        status = process.poll()
        if status is not None and status != 0:
            raise RuntimeError(f'evaluate_{model} failed with exit {status}; original outputs retained')
        running = status is None
    else:
        receipt_path = output/('launch_evaluate_'+model+'.json')
        receipt = read_json(receipt_path) if receipt_path.exists() else None
        running = bool(receipt and alive(receipt['pid'], receipt['command']))
    path = output/'evaluation'/model/'evaluation_index_test.json'
    if not path.exists():
        if not running:
            raise RuntimeError('Evaluation exited before verified test completion: '+model)
        return False
    if running:
        return False
    from .register import validate_model_split
    for split in ('valid', 'test'):
        validate_model_split(output, model, split)
    return True


def supervise(output, manifest, *, resume=False):
    synchronous(output, 'negative_diagnostics', 'negative-diagnostics', resume=resume, manifest=manifest)
    launched, done, training_checked = {}, set(), set()
    last_update = 0.
    while True:
        for model in VARIANTS:
            if model in launched or model in done:
                continue
            if model not in training_checked:
                if not training_complete(output, model, manifest):
                    require_running(output, model)
                    continue
                training_checked.add(model)
            receipt = output/('launch_evaluate_'+model+'.json')
            index = output/'evaluation'/model/'evaluation_index_test.json'
            if receipt.exists() or index.exists():
                if not resume:
                    raise ValueError('Evaluation already exists; use --resume to reuse/follow it: '+model)
                launched[model] = None
            else:
                launched[model] = launch(output, 'evaluate_'+model, 'evaluate',
                    ('--models', model, '--splits', 'valid', 'test'), gpu='2' if model == 'context_dropout' else '3')
        for model in MODELS:
            if model in done or (model != 'baseline' and model not in launched):
                continue
            if evaluation_complete(output, model, launched.get(model)):
                done.add(model)
        if len(done) == len(MODELS):
            break
        if time.monotonic()-last_update >= 60:
            progress = {}
            for model in VARIANTS:
                path = output/'training'/model/'status.json'
                progress[model] = read_json(path) if path.exists() else {'status': 'preparing'}
            print(json.dumps({'at': now().isoformat(), 'training': progress,
                              'test_evaluation_complete': sorted(done)}), flush=True)
            last_update = time.monotonic()
        time.sleep(30)
    synchronous(output, 'representation_diagnostics', 'diagnose', resume=resume, manifest=manifest)
    synchronous(output, 'report', 'report', resume=resume, manifest=manifest)
    synchronous(output, 'finalize', 'finalize', resume=resume, manifest=manifest)
    print(f'{now().isoformat()} scientific run sealed; catalogue registration is a separate verified step', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='Reuse sealed stages or follow existing live launches; never retry failures')
    args = parser.parse_args(argv)
    output = args.output_root.resolve()
    manifest = run_context(output)
    if (output/'completed.json').exists():
        if not args.resume:
            raise ValueError('Scientific run already complete; use --resume for read-only verification')
        from .register import verify_run
        verify_run(output)
        return
    with supervisor_lock(output):
        supervise(output, manifest, resume=args.resume)


if __name__ == '__main__':
    main()
