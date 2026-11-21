"""Diagnose single-label separation and context sensitivity without training.

Run from the repository root: python -m f_representation_audit.run --device cuda:0
Outputs are isolated, time-stamped diagnostics, not official retrieval results.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

from wbm_retrieval.common import digest, file_hash, read_json, write_csv, write_json
from .context import build_context_views
from .geometry import audit_geometry, paired_bootstrap
from .negative_audit import (audit_negative_sampling, nce_gradient_diagnostic,
                             sameclass_crossmap_mask, singleton_prototypes)

ROOT = Path(__file__).resolve().parents[1]
LABEL_NAMES = ['center', 'donut', 'edge-loc', 'edge-ring', 'loc', 'near-full', 'scratch', 'random']
VERSION = 'f_representation_audit_v1'


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).replace(microsecond=0)


def csv(path, rows):
    if rows:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        write_csv(path, rows, fields)


def state_hash(model):
    result = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        result.update(key.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def select_singletons(dataset, cap, seed):
    """Largest original region of each single-label map; stable frozen IDs."""
    reps = dataset['rep_indices']
    rng = np.random.default_rng(seed)
    result = []
    for label in range(8):
        members = reps[dataset['singleton_labels'][reps] == label]
        if cap is not None and len(members) > cap:
            members = np.sort(rng.choice(members, cap, replace=False))
        result.extend(members.tolist())
    return np.asarray(sorted(result), dtype=np.int64)


def context_selection(dataset, labels, maps_per_combination, seed):
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for position, sid in enumerate(dataset['map_ids']):
        if dataset['offsets'][position + 1] > dataset['offsets'][position]:
            groups[tuple(labels[int(sid)].tolist())].append(position)
    indices = []
    for combination in sorted(groups):
        positions = np.asarray(groups[combination])
        chosen = rng.choice(positions, min(len(positions), maps_per_combination), replace=False)
        for position in sorted(chosen):
            indices.extend(range(int(dataset['offsets'][position]), int(dataset['offsets'][position + 1])))
    return np.asarray(sorted(indices), dtype=np.int64)


def original_windows(raw_maps, dataset, indices):
    """Reconstruct raw 52-grid windows; caller checks against immutable cache."""
    owner = {int(s): i for i, s in enumerate(dataset['map_ids'])}
    windows = []
    for index in indices:
        sid = int(dataset['token_map_ids'][index])
        rid = int(dataset['token_region_ids'][index])
        mask = dataset['region_masks'][owner[sid]] == rid
        points = np.argwhere(mask)
        if len(points) != int(dataset['area'][index]):
            raise ValueError('Original region area does not match mask')
        raw = raw_maps[sid]
        r0, c0 = (int(round(float(x))) - 26 for x in points.mean(0))
        r1, c1 = r0 + 52, c0 + 52
        a, b, c, d = max(r0, 0), max(c0, 0), min(r1, raw.shape[0]), min(c1, raw.shape[1])
        window = np.zeros((3, 52, 52), dtype=np.float32)
        full = np.stack((raw == 2, mask, (raw == 1) | (raw == 2)))
        if c > a and d > b:
            window[:, a-r0:c-r0, b-c0:d-c0] = full[:, a:c, b:d]
        windows.append(window)
    return np.stack(windows)


def inference(model, arrays, indices, device, batch_size):
    import torch
    h, z = [], []
    with torch.inference_mode():
        for begin in range(0, len(indices), batch_size):
            x = torch.as_tensor(np.array(arrays[indices[begin:begin+batch_size]], dtype=np.float32), device=device)
            value = model['encoder'](x)
            h.append(value.cpu().numpy())
            z.append(model['projector'](value).cpu().numpy())
    return {'h512': np.concatenate(h), 'z128': np.concatenate(z)}


def summarize_numeric(rows, keys, group_key='label'):
    groups = defaultdict(list)
    for row in rows:
        groups[row[group_key]].append(row)
    def mean(items, key):
        values = [float(r[key]) for r in items if r.get(key) is not None]
        return float(np.mean(values)) if values else None
    per_class = {str(key): {'count': len(items), **{m: mean(items, m) for m in keys}}
                 for key, items in sorted(groups.items(), key=lambda x: str(x[0]))}
    return {'count': len(rows), 'mean': {m: mean(rows, m) for m in keys},
            'group_balanced': {m: mean(list(per_class.values()), m) for m in keys},
            'per_group': per_class}


def cosine_rows(left, right):
    a = left.astype(np.float64)
    b = right.astype(np.float64)
    if a.shape != b.shape or a.ndim != 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Context features must be finite equal-shaped matrices')
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError('Context features contain zero-norm vectors')
    return np.clip(np.sum(a*b, axis=1) / norms, -1., 1.)


def evaluate_context(model, views, view_audits, indices, dataset, labels, device, batch_size, state):
    values = {name: inference(model, data, np.arange(len(data)), device, batch_size)
              for name, data in views.items()}
    audit_lookup = {(a['region_key'], a['view']): a for a in view_audits}
    rows = []
    for name in views:
        if name == 'original':
            continue
        for space in ('h512', 'z128'):
            distances = 1 - cosine_rows(values['original'][space], values[name][space])
            rotation = 1 - cosine_rows(values['original'][space], values['rot90'][space])
            for i, index in enumerate(indices):
                sid = int(dataset['token_map_ids'][index])
                rid = int(dataset['token_region_ids'][index])
                source = audit_lookup[(f'{sid}:{rid}', name)]
                rows.append({'state': state, 'space': space, 'view': name, 'map_id': sid,
                             'region_id': rid, 'token_index': int(index),
                             'label_count': int(labels[sid].sum()),
                             'label_signature': '|'.join(map(str, np.flatnonzero(labels[sid]))),
                             'cosine_distance': float(distances[i]),
                             'rotation_distance': float(rotation[i]),
                             'distance_minus_rotation': float(distances[i]-rotation[i]),
                             'larger_than_rotation': float(distances[i] > rotation[i] + 1e-8),
                             'has_context': source['has_context'], 'is_noop': source['is_noop'],
                             'valid_region_input': source['valid_region_input'],
                             'context_changed_fraction': source['context_changed_fraction'],
                             'source_target_pixels': source['source_target_pixels'],
                             'source_context_pixels': source['source_context_pixels'],
                             'source_all_defect_changed_pixels': source['source_all_defect_changed_pixels']})
    summaries = {}
    keys = ['cosine_distance', 'rotation_distance', 'distance_minus_rotation', 'larger_than_rotation']
    common = {(r['map_id'], r['region_id']) for r in rows
              if r['view'] == 'context_drop' and not r['is_noop'] and r['has_context']}
    for name in views:
        if name == 'original':
            continue
        for space in ('h512', 'z128'):
            for group in ('all', 'single', 'mixed'):
                total = [r for r in rows if r['view'] == name and r['space'] == space
                          and (group == 'all' or
                          (r['label_count'] == 1 if group == 'single' else r['label_count'] > 1))]
                chosen = [r for r in total if not r['is_noop']]
                # Cluster tokens within each map before averaging maps/combination groups.
                by_map = defaultdict(list)
                for row in chosen:
                    by_map[row['map_id']].append(row)
                map_rows = [{'label_signature': members[0]['label_signature'],
                             **{k: float(np.mean([r[k] for r in members])) for k in keys}}
                            for members in by_map.values()]
                result = summarize_numeric(map_rows, keys, 'label_signature')
                result.update(total_regions=len(total), noop_regions=sum(r['is_noop'] for r in total),
                              effective_regions=len(chosen), total_maps=len({r['map_id'] for r in total}))
                common_rows = [r for r in total if (r['map_id'],r['region_id']) in common]
                by_common_map = defaultdict(list)
                for row in common_rows:
                    by_common_map[row['map_id']].append(row)
                common_maps = [{'label_signature': members[0]['label_signature'],
                                **{k: float(np.mean([r[k] for r in members])) for k in keys}}
                               for members in by_common_map.values()]
                result['common_context_cohort'] = summarize_numeric(common_maps, keys, 'label_signature')
                result['common_context_cohort']['regions'] = len(common_rows)
                result['common_context_cohort']['rule'] = 'same regions with nonzero actual context dropout; retain rotation no-ops'
                summaries[f'{name}/{space}/{group}'] = result
    return summaries, rows


def gradient_probe(model, checkpoint, dataset, query_indices, device, batch_size, repeats, seed, state):
    import torch
    n = len(dataset['token_map_ids'])
    rng = np.random.default_rng(seed)
    available = np.setdiff1d(np.arange(n), query_indices)
    full_batch = np.concatenate((query_indices, rng.choice(available, batch_size-len(query_indices), replace=False)))
    probabilities = torch.ones(n, device=device)
    probabilities[torch.as_tensor(full_batch, device=device)] = 0
    memory = checkpoint['memory'].numpy()
    anchors = memory[query_indices]
    prototypes = singleton_prototypes(memory, dataset['token_map_ids'], dataset['singleton_labels'], query_indices)
    original = np.asarray(dataset['patches'][query_indices], dtype=np.float32)
    transformed = np.ascontiguousarray(np.rot90(original, 1, axes=(-2, -1)))
    z = {name: inference(model, array, np.arange(len(array)), device, batch_size)['z128']
         for name, array in [('original', original), ('rot90', transformed)]}
    rows, checks, draws = [], [], []
    for repeat in range(repeats):
        generator = torch.Generator(device=device).manual_seed(seed + 100 + repeat)
        neg_ids = torch.multinomial(probabilities, 5000, replacement=True, generator=generator).cpu().numpy()
        draws.append(neg_ids)
        mask = sameclass_crossmap_mask(dataset['token_map_ids'], dataset['singleton_labels'], query_indices, neg_ids)
        for view in z:
            result = nce_gradient_diagnostic(z[view], anchors, memory[neg_ids], mask, prototypes,
                                             temperature=.07, verify_torch=(repeat == 0))
            if result['summary']['torch_verification']:
                checks.append({'state': state, 'view': view, **result['summary']['torch_verification']})
            for i, row in enumerate(result['rows']):
                index = int(query_indices[i])
                row.update(state=state, view=view, repeat=repeat, token_index=index,
                           map_id=int(dataset['token_map_ids'][index]), label=int(dataset['singleton_labels'][index]))
                for term in ('full', 'masked', 'sameclass_negative_term', 'full_minus_masked'):
                    val = row[f'{term}_descent_toward_class']
                    row[f'{term}_away_from_class'] = float(val < -1e-12) if val is not None else None
                rows.append(row)
    metrics = ['sampled_sameclass_crossmap_count', 'sameclass_negative_probability_mass',
               'loss_full', 'loss_masked', 'full_masked_gradient_cosine']
    metrics += [f'{term}_{suffix}' for term in ('full', 'masked', 'sameclass_negative_term', 'full_minus_masked')
                for suffix in ('descent_toward_class', 'away_from_class')]
    summaries = {view: summarize_numeric([r for r in rows if r['view'] == view], metrics)
                 for view in z}
    return {'summaries': summaries, 'checks': checks,
            'scope': 'Frozen evaluation-mode z gradients with checkpoint memory; no optimizer step, no historical minibatch claim',
            'counterfactual': 'Mask same-class cross-map logits; changes denominator, not a fixed-size replacement experiment',
            'historical_loss': 'Original and rotated terms each reported; historical objective averages their losses',
            'query_indices': query_indices.tolist(), 'excluded_batch': full_batch.tolist()}, rows, np.stack(draws)


def execute(args, output, identity):
    import torch
    from .inputs import load_inputs, verify_checkpoint_metadata
    from experiments.proposed._shared.ef_suite import build_model, fixed_window_patch
    from wbm_retrieval.common import seed_everything
    from wbm_retrieval.training import parameter_hash

    torch.set_num_threads(args.cpu_threads)
    inputs = load_inputs(ROOT, args.source_root, data_file=args.data_file)
    datasets, labels = inputs['datasets'], inputs['labels']
    config, spec = inputs['config'], inputs['spec']
    if config['local']['num_negatives'] != 5000 or config['local']['temperature'] != .07:
        raise ValueError('This diagnostic recipe targets the frozen 5000/.07 NCE settings')
    if args.gradient_per_class * 8 > config['batch_size']:
        raise ValueError('Gradient query sample exceeds original batch size')
    selections = {
        'train_representative': ('train', select_singletons(datasets['train'], args.train_per_class, args.seed)),
        'valid_representative': ('valid', select_singletons(datasets['valid'], None, args.seed)),
        'valid_all_regions': ('valid', np.flatnonzero(datasets['valid']['singleton_labels'] >= 0)),
    }
    context_indices = context_selection(datasets['valid'], labels, args.context_per_combination, args.seed)
    grad_indices = select_singletons(datasets['train'], args.gradient_per_class, args.seed + 1)
    source_hashes = inputs['protection']
    write_json(output/'source_protection.json', source_hashes)
    sample_rows = []
    for name, (split, selected) in {**selections, 'context': ('valid', context_indices),
                                   'gradient': ('train', grad_indices)}.items():
        ds = datasets[split]
        for index in selected:
            sid = int(ds['token_map_ids'][index])
            sample_rows.append({'subset': name, 'split': split, 'sample_id': sid, 'token_index': int(index),
                               'region_id': int(ds['token_region_ids'][index]),
                               'label_signature': '|'.join(map(str, np.flatnonzero(labels[sid]))),
                               'singleton_label': int(ds['singleton_labels'][index]),
                               'original_region_area': int(ds['area'][index])})
    csv(output/'selected_regions.csv', sample_rows)
    windows = original_windows(inputs['maps'], datasets['valid'], context_indices)
    keys = [f"{datasets['valid']['token_map_ids'][i]}:{datasets['valid']['token_region_ids'][i]}" for i in context_indices]
    views, view_audits = build_context_views(windows, keys, seed=args.seed, include_target_removed_control=True)
    cached = np.asarray(datasets['valid']['patches'][context_indices], dtype=np.float32)
    if not np.array_equal(views['original'], cached):
        raise ValueError('Reconstructed raw windows disagree with historical F input bytes')
    # Independently replay the canonical function on every selected original region.
    owners = {int(sid): i for i, sid in enumerate(datasets['valid']['map_ids'])}
    for i, index in enumerate(context_indices):
        sid = int(datasets['valid']['token_map_ids'][index])
        rid = int(datasets['valid']['token_region_ids'][index])
        mask = datasets['valid']['region_masks'][owners[sid]] == rid
        canonical, _ = fixed_window_patch(inputs['maps'][sid], mask)
        if not np.array_equal(canonical, views['original'][i]):
            raise ValueError('Independent original-window replay failed')
    csv(output/'context_input_audit.csv', view_audits)
    np.savez_compressed(output/'context_examples.npz', indices=context_indices[:12],
                        **{name: data[:12] for name, data in views.items()})
    print(f"Verified source; diagnostic subsets: {[(k,len(v[1])) for k,v in selections.items()]}; context={len(context_indices)}", flush=True)
    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(len(datasets['train']['token_map_ids']))
    batches = [permutation[i:i+config['batch_size']] for i in range(0, len(permutation), config['batch_size'])]
    exposure = audit_negative_sampling(datasets['train']['token_map_ids'], datasets['train']['singleton_labels'], batches)
    write_json(output/'negative_sampling_summary.json', exposure['summary'])
    csv(output/'negative_sampling_per_region.csv', exposure['rows'])
    np.save(output/'diagnostic_batch_permutation.npy', permutation)
    code_hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    manifest = {**identity, 'version': VERSION, 'kind': 'frozen_representation_diagnostic',
                'protocol': 'single_label_representation_diagnostic_v1',
                'source_root': str(args.source_root), 'config': vars(args), 'label_names': LABEL_NAMES,
                'code': code_hashes, 'source_files': source_hashes,
                'scope': 'train/valid only; labels are posthoc diagnostic proxies; not a new training or retrieval result',
                'selection': 'all valid single-label maps; largest region primary; all regions secondary; capped balanced train',
                'checkpoints': 'exact seed initialization plus pre-existing epoch15 best_retrieval and frozen epoch100 best_nce; no reselection',
                'bootstrap': 'paired map clusters within class, conditional on fixed gallery, not seed uncertainty',
                'context_control': 'target_removed_control violates target<=defects; location probe only, not a valid region or semantic counterexample',
                'source_test_policy': 'source archive/protocol identity may include test IDs; no test input extraction, ranking, or metrics',
                'environment': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                                'device': args.device, 'cuda': torch.version.cuda}}
    manifest['config'] = {k: str(v) if isinstance(v, Path) else v for k,v in manifest['config'].items()}
    write_json(output/'manifest.json', manifest)
    seed_everything(config['seed'])
    model = build_model(config, spec).to(args.device).eval()
    if parameter_hash(model) != inputs['run_summary']['initial_parameter_sha256']:
        raise ValueError('Seed-reconstructed initialization does not match the source training hash')
    all_geometry, context_summaries, gradients, state_metadata = {}, {}, {}, {}
    for state in ('random', 'best_retrieval', 'best_nce'):
        print(f'Extracting state {state}', flush=True)
        checkpoint = None
        if state != 'random':
            checkpoint = torch.load(inputs['checkpoint_paths'][state], map_location='cpu', weights_only=False)
            verify_checkpoint_metadata(checkpoint, state, inputs)
            model.load_state_dict(checkpoint['model'], strict=True)
        model.eval()
        before = state_hash(model)
        state_metadata[state] = {'epoch': 0 if checkpoint is None else checkpoint['epoch'],
                                 'all_state_sha256': before, 'parameter_sha256': parameter_hash(model)}
        state_dir = output/state
        state_dir.mkdir()
        for name, (split, indices) in selections.items():
            ds = datasets[split]
            features = inference(model, ds['patches'], indices, args.device, args.batch_size)
            np.savez_compressed(state_dir/f'{name}_features.npz', indices=indices, **features)
            for space, x in features.items():
                result = audit_geometry(x, ds['singleton_labels'][indices], ds['token_map_ids'][indices], ks=(1,5,10,20,50,100,500))
                write_json(state_dir/f'{name}_{space}_summary.json', result['summary'])
                csv(state_dir/f'{name}_{space}_per_region.csv', result['rows'])
                all_geometry[(state,name,space)] = result
        context_summaries[state], context_rows = evaluate_context(model, views, view_audits, context_indices,
                                                   datasets['valid'], labels, args.device, args.batch_size, state)
        write_json(state_dir/'context_summary.json', context_summaries[state])
        csv(state_dir/'context_per_region.csv', context_rows)
        if checkpoint is not None:
            gradients[state], gradient_rows, negative_draws = gradient_probe(model, checkpoint, datasets['train'],
                grad_indices, args.device, config['batch_size'], args.gradient_repeats, args.seed, state)
            write_json(state_dir/'gradient_summary.json', gradients[state])
            csv(state_dir/'gradient_per_region.csv', gradient_rows)
            np.save(state_dir/'gradient_negative_draws.npy', negative_draws)
        if state_hash(model) != before:
            raise ValueError('Frozen model tensors changed during diagnostic')
        print(f'Finished {state}; no model tensors changed', flush=True)
        del checkpoint
    comparisons = {}
    for a,b in [('random','best_nce'),('best_retrieval','best_nce')]:
        for name in selections:
            for space in ('h512','z128'):
                key = f'{a}_to_{b}/{name}/{space}'
                comparisons[key] = paired_bootstrap(all_geometry[(a,name,space)], all_geometry[(b,name,space)],
                                                   repetitions=args.bootstrap, seed=args.seed)
    write_json(output/'paired_comparisons.json', comparisons)
    summary = {'identity': identity, 'version': VERSION, 'states': state_metadata,
               'subsets': {name: {'split': split, 'regions': len(indices),
                                  'maps': len(np.unique(datasets[split]['token_map_ids'][indices]))}
                           for name,(split,indices) in selections.items()},
               'geometry': {'/'.join(key): result['summary'] for key,result in all_geometry.items()},
               'negative_sampling': exposure['summary'], 'gradients': gradients,
               'context': context_summaries, 'comparisons': comparisons,
               'verification': {'source_windows_byte_equal': len(context_indices), 'canonical_replay_equal': len(context_indices),
                                'random_initialization_hash_equal': True, 'all_models_unchanged': True},
               'interpretation': ['Absolute within-class cosine decreases do not prove worse class geometry.',
                                  'Same-class negative exposure and instantaneous gradients do not establish training causality.',
                                  'Single-label wafer labels are proxies; their individual components can be noise or incomplete patterns.',
                                  'Existing epoch15 checkpoint was selected using valid retrieval; this is exploratory, no checkpoint change.',
                                  'Context can contain parts of the same defect; sensitivity is not automatically harmful.']}
    write_json(output/'summary.json', summary)
    from .report import render
    render(output)
    print('Verifying source and scientific-code hashes after diagnostics', flush=True)
    for path, expected in source_hashes.items():
        if file_hash(path) != expected:
            raise ValueError(f'Protected source changed: {path}')
    for path, expected in code_hashes.items():
        if file_hash(ROOT/path) != expected:
            raise ValueError(f'Diagnostic source changed during run: {path}')
    write_json(output/'verification.json', {**summary['verification'], 'source_files_unchanged': len(source_hashes),
                                           'diagnostic_code_unchanged': True, 'verified_at': now().isoformat()})
    write_json(output/'status.json', {'status': 'completed', 'kind': 'diagnostic', 'completed_at': now().isoformat(), **identity})
    files = {str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output/'completed.json', {'version': VERSION, 'kind': 'diagnostic', **identity, 'files': files})
    print(f'COMPLETED {output}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT/'artifacts/f_segmentation_v1/cc8')
    parser.add_argument('--data-file', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--train-per-class', type=int, default=128)
    parser.add_argument('--context-per-combination', type=int, default=4)
    parser.add_argument('--gradient-per-class', type=int, default=8)
    parser.add_argument('--gradient-repeats', type=int, default=4)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if min(args.batch_size,args.cpu_threads,args.train_per_class,args.context_per_combination,
           args.gradient_per_class,args.gradient_repeats,args.bootstrap) < 1 or args.seed < 0:
        parser.error('Counts must be positive and seed nonnegative')
    args.source_root = args.source_root.resolve()
    if args.dry_run:
        print(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},indent=2))
        return
    stamp = now()
    identity = {'run_id': stamp.strftime('%Y%m%dT%H%M%S%z')+'__f_representation_audit',
                'started_at': stamp.isoformat(), 'split': 'train_valid', 'schema_version': 1}
    output = (args.output_root or ROOT/'artificial'/identity['run_id']).resolve()
    if output == args.source_root or args.source_root in output.parents or output in args.source_root.parents:
        parser.error('Output must be separate from source')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'run_identity.json', identity)
    write_json(output/'status.json', {'status':'running','kind':'diagnostic',**identity})
    print(f'OUTPUT_ROOT {output}', flush=True)
    try:
        execute(args, output, identity)
    except BaseException as error:
        write_json(output/'status.json', {'status':'failed','kind':'diagnostic',**identity,
                                         'error_type':type(error).__name__,'error':str(error)})
        raise


if __name__ == '__main__':
    main()
