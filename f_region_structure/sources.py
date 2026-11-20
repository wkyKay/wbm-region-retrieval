"""Read-only frozen F inputs; new descriptor and training-view caches only."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import tempfile

import numpy as np
import torch

from b1_structure_fusion.run import assert_frozen
from b1_structure_fusion.descriptors import load_settings
from experiments.proposed._shared import ef_suite
from experiments.proposed.f_segmentation import suite as original
from experiments.proposed.f_segmentation_geometry import pipeline as geo
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from wbm_retrieval.data import Protocol, source_fingerprint
from . import descriptors
from .state import ROOT, record, seal, verified


def load(source_root, data_file=None):
    source_root = Path(source_root).resolve()
    config, suite, checkpoint, kind, inputs = geo.load_source(source_root, 'cc8', data_file, 'cpu')
    assert_frozen(config, suite, kind, load_settings())
    protocol = Protocol(config)
    if protocol.retrieval is None:
        raise ValueError('Frozen full-gallery protocol is required')
    if [len(protocol.ids[s]) for s in ('train','valid','test')] != [25193,3599,7199]:
        raise ValueError('Unexpected split identities/counts')
    for split, nq, ng in (('valid',500,3599), ('test',7199,7199)):
        pool = getattr(protocol.retrieval, split)
        if len(pool) != nq or len(pool.gallery) != ng or pool.candidates_per_query != ng-1:
            raise ValueError('Unexpected fixed retrieval pool')
    identity = {'source_files':inputs, 'source_fingerprint':source_fingerprint(config),
                'geometry':geo.new_identity(config, inputs), 'checkpoint':kind,
                'epoch':int(checkpoint['epoch']), 'seed':2026}
    return config, suite, checkpoint, protocol, identity


def feature_source(split, config, geometry, roots):
    if split == 'train':
        folder = Path(config['output_dir'])
        marker = read_json(folder / 'cache/train_prepared.json')
        manifest = read_json(folder / 'suite_manifest.json')
        if marker['signature'] != manifest['signature']:
            raise ValueError('Training cache has a different source signature')
        original.verify_files(folder, marker['files'])
        with np.load(folder / 'cache/train_regions.npz', allow_pickle=False) as archive:
            meta = dict(archive)
        labels = np.load(folder / 'cache/train_region_labels.npy', mmap_mode='r')
        patches = folder / 'cache/train_F.npy'
        binding = {str((folder / n).resolve()): h for n,h in marker['files'].items()}
        for name in ('cache/train_prepared.json','suite_manifest.json'):
            binding[str((folder/name).resolve())] = file_hash(folder/name)
        return meta, labels, None, patches, binding
    for root in roots:
        folder = Path(root) / split
        if (folder / 'completed.json').exists() and geo.completed(folder, digest(geometry)):
            with np.load(folder / 'regions.npz', allow_pickle=False) as archive:
                meta = dict(archive)
            labels = np.load(folder / 'region_labels.npy', mmap_mode='r')
            h = np.load(folder / 'h512.npy', mmap_mode='r')
            marker = read_json(folder / 'completed.json')
            binding = {str((folder / n).resolve()):v for n,v in marker['files'].items()}
            binding[str((folder / 'completed.json').resolve())] = file_hash(folder / 'completed.json')
            return meta, labels, h, None, binding
    raise FileNotFoundError(f'No completed frozen F feature cache for {split}: {roots}')


def _job(job):
    raw, labels = job
    return descriptors.describe_regions(raw, labels)


def _map_jobs(jobs, workers):
    if workers == 0:
        yield from map(_job, jobs)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as executor:
        pending = deque()
        for _ in range(2*workers):
            job = next(jobs, None)
            if job is not None:
                pending.append(executor.submit(_job, job))
        while pending:
            yield pending.popleft().result()
            job = next(jobs, None)
            if job is not None:
                pending.append(executor.submit(_job, job))


def encode_train_views(patches_path, config, suite, checkpoint, output, device):
    patches = np.load(patches_path, mmap_mode='r')
    model = ef_suite.build_model(config, suite['variants']['F']).to(device).eval()
    model.load_state_dict(checkpoint['model'])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    features = np.lib.format.open_memmap(output / 'h_views.npy', mode='w+', dtype=np.float32,
                                        shape=(len(patches),2,512))
    for start in range(0,len(patches),128):
        x = torch.from_numpy(np.array(patches[start:start+128], dtype=np.float32)).to(device)
        with torch.inference_mode():
            for view in range(2):
                h = model['encoder'](torch.rot90(x, view, dims=(-2,-1)))
                features[start:start+len(x),view] = torch.nn.functional.normalize(h, dim=1).cpu().numpy()
        if start % 12800 == 0 or start+128 >= len(patches):
            print(f'Train frozen CNN views {min(start+128,len(patches))}/{len(patches)}', flush=True)
    features.flush()
    del features, model
    if device == 'cuda':
        torch.cuda.empty_cache()


def prepare(output, split, maps, config, suite, checkpoint, protocol, source_identity,
            feature_roots, context_signature, workers=4, device='cpu'):
    target = Path(output) / 'cache' / split
    signature = digest({'context':context_signature,'split':split,'cache_version':1})
    if verified(target, signature):
        value = read_json(target / 'source_binding.json')
        for path, expected in value.items():
            if file_hash(path) != expected:
                raise ValueError('Frozen source changed since cache preparation: '+path)
        return read_cache(target)
    meta, region_labels, h, patches, binding = feature_source(split, config, source_identity['geometry'], feature_roots)
    ids, offsets = np.asarray(meta['map_ids']), np.asarray(meta['offsets'])
    if not np.array_equal(ids, protocol.ids[split]) or region_labels.shape != (len(ids), *maps.shape[1:]):
        raise ValueError('Source region IDs or map dimensions differ from frozen split')
    if offsets.shape != (len(ids)+1,) or offsets[0] != 0 or np.any(np.diff(offsets)<0) or np.any(np.diff(offsets)>4):
        raise ValueError('Invalid source region offsets')
    target.parent.mkdir(parents=True,exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix='.'+split+'_attempt_',dir=target.parent))
    record(attempt / 'source_binding.json', binding)
    jobs = ((maps[int(sid)],np.array(region_labels[i])) for i,sid in enumerate(ids))
    rows = []
    for index,row in enumerate(_map_jobs(jobs,workers),1):
        a,b = offsets[index-1:index+1]
        if not np.array_equal(row['original_count'],meta['area'][a:b]):
            raise ValueError('Region descriptor area differs from frozen region order')
        rows.append(row)
        if index % 100 == 0 or index == len(ids):
            print(f'Region J {split}: {index}/{len(ids)}',flush=True)
    names = ('joint','joint_all','shape','shape_all','original_count','retained_count','quality',
             'empty','empty_all','radial_hist16','relative_area')
    values = {name:np.concatenate([r[name] for r in rows],axis=0) for name in names}
    if split != 'train':
        np.testing.assert_allclose(values['radial_hist16'],meta['histograms'],atol=1e-14,rtol=0)
        np.testing.assert_allclose(values['relative_area'],meta['relative_area'],atol=1e-14,rtol=0)
    values.update(map_ids=ids,offsets=offsets,area=np.asarray(meta['area']),
                  histograms=values['radial_hist16'],owners=np.repeat(ids,np.diff(offsets)))
    np.savez_compressed(attempt/'regions.npz',**values)
    original_counts = np.array([r['global_original_count'] for r in rows])
    kept = np.array([int(r['original_count'].sum()) for r in rows])
    write_json(attempt/'audit.json', {'split':split,'maps':len(ids),'regions':int(offsets[-1]),
        'empty_region_J':int(values['empty'].sum()),'empty_maps':int((np.diff(offsets)==0).sum()),
        'mean_region_quality':float(values['quality'].mean()),
        'total_pixel_retention':float(kept.sum()/original_counts.sum()),
        'map_ids':ids.tolist(),'map_pixel_retention':(kept/np.maximum(original_counts,1)).tolist(),
        'voting':'all original defect pixels once; pool original cc8 regions',
        'normalization':'per-image channel maximum before filtering',
        'train_views':'identity and 90-degree rotation of original frozen F fixed-window patch' if split=='train' else None})
    if split=='train':
        encode_train_views(patches,config,suite,checkpoint,attempt,device)
        h = np.load(attempt/'h_views.npy',mmap_mode='r')[:,0]
    np.save(attempt/'h512.npy',np.asarray(h))
    seal(attempt,signature)
    if target.exists():
        raise ValueError('Refuse to overwrite an incomplete cache; preserve it and use a new output')
    attempt.rename(target)
    return read_cache(target)


def read_cache(folder):
    folder = Path(folder)
    with np.load(folder/'regions.npz',allow_pickle=False) as archive:
        data = dict(archive)
    data['features'] = np.load(folder/'h512.npy',mmap_mode='r')
    if (folder/'h_views.npy').exists():
        data['h_views'] = np.load(folder/'h_views.npy',mmap_mode='r')
    data['folder'] = folder
    return data
