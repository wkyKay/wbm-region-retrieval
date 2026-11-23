"""Frozen-state inference with explicit checkpoint and original token identities."""
from pathlib import Path
import hashlib
import numpy as np
import torch

from experiments.proposed._shared.ef_suite import build_model
from f_representation_audit.inputs import verify_checkpoint_metadata
from f_representation_audit.run import state_hash
from wbm_retrieval.common import digest, file_hash, seed_everything
from wbm_retrieval.diagnostics import normalized
from wbm_retrieval.training import parameter_hash
from .state import ROOT, code_binding, identity, record, signed, seal, verified, read_signed


def checkpoint_path(output, inputs, name):
    if name == 'baseline':
        return inputs['checkpoint_paths']['best_nce']
    if name not in ('context_dropout', 'filtered_negatives'):
        raise ValueError('Unknown model')
    folder = Path(output)/'training'/name
    manifest = read_signed(folder/'run_manifest.json')
    if not verified(folder, manifest['training_signature']):
        raise ValueError('New model has not completed its fixed training budget')
    return folder/'checkpoint.pt'


def load_model(output, inputs, name, device):
    path = checkpoint_path(output, inputs, name)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if name == 'baseline':
        verify_checkpoint_metadata(payload, 'best_nce', inputs)
    elif (payload.get('epoch') != 100 or payload.get('optimizer_steps') != 39000
          or payload.get('selection') != 'fixed_epoch_100' or payload.get('variant') != name
          or payload.get('initial_parameter_sha256') != inputs['run_summary']['initial_parameter_sha256']
          or payload.get('input_spec') != inputs['spec']):
        raise ValueError('Unexpected trained checkpoint identity/budget')
    seed_everything(inputs['config']['seed'])
    model = build_model(inputs['config'], inputs['spec']).to(device)
    if parameter_hash(model) != inputs['run_summary']['initial_parameter_sha256']:
        raise ValueError('Model architecture/initialization differs from original baseline')
    model.load_state_dict(payload['model'], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {'path': str(path.resolve()), 'sha256': file_hash(path), 'epoch': 100,
                   'parameter_sha256': parameter_hash(model), 'all_state_sha256': state_hash(model)}


def array_identity(array):
    array = np.asarray(array)
    return {'dtype': str(array.dtype), 'shape': list(array.shape),
            'sha256': hashlib.sha256(array.tobytes()).hexdigest()}


def extract(output, inputs, model_name, patches, layout, split, *, device='cuda:0',
            batch_size=256, views=('original',)):
    if split not in ('train', 'valid', 'test') or not set(views) <= {'original', 'rot90'}:
        raise ValueError('Unexpected split or inference view')
    if split == 'train' and (model_name != 'baseline' or tuple(views) != ('original', 'rot90')):
        raise ValueError('Train extraction is exclusively the fixed two-view baseline teacher')
    path = checkpoint_path(output, inputs, model_name)
    destination = Path(output)/'features'/model_name/split
    patch_path = Path(patches.filename)
    science = {'version': 1, 'model': model_name, 'split': split, 'views': list(views),
        'checkpoint_path': str(path.resolve()), 'checkpoint_sha256': file_hash(path),
        'patch_path': str(patch_path.resolve()), 'patch_sha256': file_hash(patch_path),
        'token_layout': layout, 'batch_size': batch_size,
        'normalization': 'same legacy float64 L2 normalization of float32 encoder output',
        'inference_input': 'full original window; rotation applied after resize; no context deletion',
        'parent_context': inputs['refinement_context_signature'],
        'code': code_binding(['cnn_refinement/features.py',
            'experiments/proposed/_shared/ef_suite.py', 'wbm_retrieval/diagnostics.py'])}
    signature = digest(science)
    if verified(destination, signature):
        return {view: destination/(view+'.npy') for view in views}
    if (destination/'manifest.json').exists():
        raise ValueError('Incomplete feature stage retained; use a new output root or repair before restart')
    identity(destination, 'cnn_'+model_name+'_'+split+'_features', signature, split=split, variant=model_name)
    model, checkpoint = load_model(output, inputs, model_name, device)
    record(destination/'manifest.json', signed({**science, 'checkpoint': checkpoint,
        'feature_signature': signature, 'environment': {'torch': torch.__version__, 'numpy': np.__version__,
         'cuda': torch.version.cuda, 'device': str(device)}}))
    arrays = {view: np.lib.format.open_memmap(destination/(view+'.npy'), mode='w+',
              dtype=np.float64, shape=(len(patches), 512)) for view in views}
    with torch.inference_mode():
        for start in range(0, len(patches), batch_size):
            x = torch.as_tensor(np.array(patches[start:start+batch_size], dtype=np.float32), device=device)
            for view, dest in arrays.items():
                batch = torch.rot90(x, 1, (-2, -1)) if view == 'rot90' else x
                h = model['encoder'](batch).cpu().numpy()
                if not np.isfinite(h).all() or np.any(np.linalg.norm(h, axis=1) <= 1e-12):
                    raise ValueError('Nonfinite/zero learned feature')
                dest[start:start+len(x)] = normalized(h)
            if start == 0 or (start//batch_size+1) % 50 == 0 or start+batch_size >= len(patches):
                print(f'{model_name} {split} features {min(start+batch_size,len(patches))}/{len(patches)}', flush=True)
    for array in arrays.values():
        array.flush()
    if state_hash(model) != checkpoint['all_state_sha256'] or file_hash(path) != checkpoint['sha256']:
        raise ValueError('Inference modified frozen model state')
    record(destination/'verification.json', {'status': 'verified', 'token_count': len(patches),
        'all_state_unchanged': True, 'checkpoint_unchanged': True, 'normalization_dtype': 'float64',
        'token_layout': layout})
    seal(destination, signature)
    return {view: destination/(view+'.npy') for view in views}
