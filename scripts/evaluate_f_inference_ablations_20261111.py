#!/usr/bin/env python3
"""Sequential, valid-only F rotation/radial ablations on a frozen checkpoint."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def radial_histogram(mask, support, bins=16):
    """Full original-grid region; support bbox centre, max support radius."""
    mask, support = np.asarray(mask, bool), np.asarray(support, bool)
    if mask.shape != support.shape or mask.ndim != 2 or bins < 2:
        raise ValueError("Invalid geometry")
    points, wafer = np.argwhere(mask), np.argwhere(support)
    if not len(points) or not len(wafer) or np.any(mask & ~support):
        raise ValueError("Region must be nonempty and contained in wafer support")
    center = (wafer.min(0) + wafer.max(0)) / 2
    radius = np.linalg.norm(wafer-center, axis=1).max()
    rho = np.linalg.norm(points-center, axis=1) / radius if radius > 0 else np.zeros(len(points))
    hist = np.histogram(np.clip(rho, 0, 1), bins=np.linspace(0, 1, bins+1))[0].astype(float)
    return hist / hist.sum()


def radial_distance(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape[-1] != b.shape[-1] or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Invalid radial histograms")
    if np.any(a < 0) or np.any(b < 0) or not np.allclose(a.sum(-1), 1) or not np.allclose(b.sum(-1), 1):
        raise ValueError("Radial histograms must be probability distributions")
    # W1 between equal-width bin centres in normalized radius units.
    return np.abs(np.cumsum(a, axis=-1)-np.cumsum(b, axis=-1)).sum(axis=-1)/a.shape[-1]


def pool_rotation_features(features):
    from wbm_retrieval.diagnostics import normalized
    features = np.asarray(features, float)
    if features.ndim != 3 or features.shape[0] != 4 or not np.isfinite(features).all():
        raise ValueError("Expected four [regions, dimensions] feature arrays")
    return normalized(np.mean([normalized(v) for v in features], axis=0))


def extract_features(model, patches, config, device):
    import torch
    from wbm_retrieval.diagnostics import normalized
    model.eval()
    rotations = []
    with torch.inference_mode():
        for turn in range(4):
            result = []
            for start in range(0, len(patches), config['inference_batch_size']):
                batch = np.array(patches[start:start+config['inference_batch_size']], dtype=np.float32)
                batch = np.ascontiguousarray(np.rot90(batch, turn, axes=(-2, -1)))
                result.append(model['encoder'](torch.from_numpy(batch).to(device)).cpu().numpy())
            rotations.append(normalized(np.concatenate(result)))
            print(f'Extracted rotation {turn*90}', flush=True)
    values = np.stack(rotations)
    return values[0], pool_rotation_features(values), {
        'mean_cosine_to_original_by_rotation': [float((values[0]*v).sum(1).mean()) for v in values],
        'min_cosine_to_original_by_rotation': [float((values[0]*v).sum(1).min()) for v in values],
        'pooled_cyclic_permutation_max_error': float(np.abs(pool_rotation_features(np.roll(values, 1, 0))-pool_rotation_features(values)).max())}


def paired_report(first, second):
    from wbm_retrieval.common import read_json
    a, b = read_json(first/'per_query.json'), read_json(second/'per_query.json')
    if [r['query_id'] for r in a] != [r['query_id'] for r in b]:
        raise ValueError('Query identities differ')
    rng = np.random.default_rng(2026)
    result = {}
    for group in ('all', 'single', 'mixed'):
        selected = [i for i, r in enumerate(a) if group == 'all' or r['group'] == group]
        result[group] = {}
        for key in ('Hit@1', 'Precision@10', 'mAP@10', 'mAP@100', 'mAP@500'):
            field = key[1:] if key.startswith('mAP') else key
            delta = np.array([b[i][field]-a[i][field] for i in selected])
            means = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)])
            result[group][key] = {'delta': float(delta.mean()), 'bootstrap_95ci': np.quantile(means, [.025, .975]).tolist(),
                                  'improved': int((delta > 1e-12).sum()), 'worsened': int((delta < -1e-12).sum()), 'queries': len(delta)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True, type=Path)
    parser.add_argument('--data-file', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    from experiments.proposed._shared.ef_suite import build_model
    from wbm_retrieval.common import read_json, write_json, file_hash, digest, get_device
    from wbm_retrieval.data import decode_pool
    from wbm_retrieval.diagnostics import batched_map_scores
    from wbm_retrieval.exact_evaluation import evaluate_suite_rankings
    from wbm_retrieval.exact_metrics import order_scores
    from wbm_retrieval.training import load_checkpoint
    from wbm_retrieval.regions import extract_regions
    source, output = args.source_run.resolve(), args.output_dir.resolve()
    if output.exists() or output == source or source in output.parents or output in source.parents:
        raise ValueError('Use a new output directory outside the source run')
    manifest = read_json(source/'suite_manifest.json')
    config = dict(manifest['settings'])
    config.update(seed=2026, device=args.device, num_workers=0)
    if config['data_protocol'] != 'deduplicated_v1' or config['evaluation']['metric_version'] != 'exact_v1':
        raise ValueError('Frozen exact protocol required')
    ckpt_path = source/'seed_2026/F/best_nce.pt'
    ckpt = load_checkpoint(ckpt_path)
    if ckpt['signature'] != digest({'suite': manifest['signature'], 'seed': 2026, 'variant': 'F'}):
        raise ValueError('Checkpoint signature mismatch')
    spec = manifest['suite']['variants']['F']
    if ckpt['input_spec'] != spec or ckpt['variant'] != 'F':
        raise ValueError('Not an F checkpoint')
    prepared = read_json(source/'cache/prepared.json')
    if prepared['signature'] != manifest['signature']:
        raise ValueError('Cache signature mismatch')
    required = ('cache/valid_regions.npz', 'cache/valid_F.npy', 'cache/valid_masks.npy', 'cache/labels.npy', 'valid_pool.json')
    hashes = {name: file_hash(source/name) for name in required}
    if any(prepared['files'].get(name) != sha for name, sha in hashes.items()):
        raise ValueError('Cache content changed')
    data_hash = file_hash(args.data_file)
    if data_hash != manifest['sources']['data_file']:
        raise ValueError('Original data hash mismatch')
    payload = read_json(source/'valid_pool.json')
    queries, candidates = decode_pool(payload)
    for name, ids in (('valid_queries.csv', queries), ('valid_gallery.csv', payload['gallery'])):
        path = ROOT/'protocol/retrieval_v1'/name
        if file_hash(path) != manifest['sources']['retrieval_protocol/'+name]:
            raise ValueError('Frozen protocol hash mismatch')
        with path.open() as f:
            frozen = [int(r['sample_id']) for r in csv.DictReader(f)]
        if list(ids) != frozen:
            raise ValueError('Pool identity mismatch')
    with np.load(source/'cache/valid_regions.npz') as archive:
        meta = dict(archive)
    if set(meta['map_ids'].tolist()) != set(payload['gallery']):
        raise ValueError('Invalid gallery coverage')
    output.mkdir(parents=True, exist_ok=False)
    betas = [0., 1., 2., 4.]
    plan = {'source_run': str(source), 'checkpoint': str(ckpt_path), 'checkpoint_sha256': file_hash(ckpt_path),
            'epoch': ckpt['epoch'], 'suite_signature': manifest['signature'], 'source_files': hashes,
            'data_sha256': data_hash, 'script_sha256': file_hash(__file__),
            'dependencies': {name: file_hash(ROOT/name) for name in ('wbm_retrieval/models.py', 'wbm_retrieval/diagnostics.py',
                'wbm_retrieval/exact_metrics.py', 'wbm_retrieval/regions.py', 'experiments/proposed/_shared/ef_suite.py')},
            'split': 'valid', 'test_evaluated': False, 'training_required': False,
            'sequence': ['baseline', 'rot4', 'radial_grid', 'conditional_combination'], 'radial_bins': 16,
            'beta_grid': betas, 'selection_metric': 'mAP@10', 'selection_uses_valid_labels': True,
            'beta_tie_rule': 'smaller beta',
            'combination_rule': 'both individual mAP@10 strictly improve and neither mixed mAP@10 decreases',
            'radial_geometry': 'support raw==1/2 bbox centre; max support radius; full original region mask',
            'radial_distance': 'sum(abs(CDF_a-CDF_b))/16; W1 between histogram bin centres',
            'pair_score': 'clip(cosine,0,1)*exp(-beta*radial_distance)',
            'aggregation': 'unchanged original F greedy and query sqrt(area) weighting',
            'num_queries': len(queries), 'num_gallery': len(payload['gallery'])}
    write_json(output/'plan_before_evaluation.json', plan)
    labels = np.load(source/'cache/labels.npy', mmap_mode='r')
    patches = np.load(source/'cache/valid_F.npy', mmap_mode='r')
    masks = np.load(source/'cache/valid_masks.npy', mmap_mode='r')
    if len(patches) != meta['offsets'][-1] or len(masks) != len(patches):
        raise ValueError('Region arrays not aligned')
    model = build_model(config, spec).to(get_device(args.device))
    model.load_state_dict(ckpt['model'])
    baseline, rot4, rotation_audit = extract_features(model, patches, config, get_device(args.device))
    write_json(output/'rotation_audit.json', rotation_audit)
    np.save(output/'baseline_h512.npy', baseline)
    np.save(output/'rot4_h512.npy', rot4)
    indices = {int(i): np.arange(a, b) for i, a, b in zip(meta['map_ids'], meta['offsets'][:-1], meta['offsets'][1:])}
    width = int(np.diff(meta['offsets']).max())
    padded = np.full((len(meta['map_ids']), width), -1, dtype=int)
    positions = {int(i): n for n, i in enumerate(meta['map_ids'])}
    for sample_id, tokens in indices.items():
        padded[positions[sample_id], :len(tokens)] = tokens
    results = {}
    def evaluate(name, features, beta=0., histograms=None):
        started = time.monotonic()
        target = output/name
        target.mkdir(exist_ok=False)
        with (target/'rankings.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['query_id', 'rank', 'candidate_id', 'similarity_score'])
            def rankings():
                for number, q in enumerate(queries, 1):
                    c, tokens = np.asarray(candidates[q]), indices[q]
                    cosine = features[tokens] @ features.T
                    similarity = np.clip(cosine, 0, 1)
                    if beta:
                        distance = radial_distance(histograms[tokens, None], histograms[None])
                        similarity *= np.exp(-beta*distance)
                    ci = padded[[positions[int(i)] for i in c]]
                    scores = batched_map_scores(similarity, ci, meta['area'][tokens], config['matching'])
                    if not beta:
                        reference = batched_map_scores(cosine, ci, meta['area'][tokens], config['matching'])
                        np.testing.assert_array_equal(scores, reference)
                    order = order_scores(c, scores, q, config['evaluation'].get('tie_seed', 2026))[:max(config['evaluation']['ks'])]
                    writer.writerows((int(q), rank, int(c[i]), float(scores[i])) for rank, i in enumerate(order, 1))
                    yield int(q), c[order]
                    if number % 100 == 0:
                        print(f'{name}: {number}/{len(queries)}', flush=True)
            result = evaluate_suite_rankings(config, labels, candidates, rankings(), target, 'valid', split_ids=meta['map_ids'],
                metadata={'representation': name, 'beta': beta, 'checkpoint': str(ckpt_path), 'epoch': ckpt['epoch'],
                          'aggregation': 'original_F', 'candidates_scored': 'full_pool', 'bootstrap_samples': 0,
                          'pool_sha256': hashes['valid_pool.json']})
        results[name] = result
        write_json(target/'timing.json', {'seconds': time.monotonic()-started})
        if name != 'baseline':
            write_json(target/'paired_vs_baseline.json', paired_report(output/'baseline', target))
        print('RESULT '+json.dumps({'name': name, 'mAP@10': result['metrics']['mAP@10'],
            'mAP@100': result['metrics']['mAP@100'], 'Hit@1': result['metrics']['Hit@1'],
            'mixed_mAP@10': result['groups']['mixed_label']['metrics']['mAP@10']}), flush=True)
        return result
    reference = evaluate('baseline', baseline)
    saved = read_json(source/'seed_2026/F/valid/best_nce/metrics.json')
    delta = {k: value-saved['metrics'][k] for k, value in reference['metrics'].items()}
    write_json(output/'baseline_verification.json', {'saved_metric_deltas': delta, 'identical': all(v == 0 for v in delta.values())})
    if any(v != 0 for v in delta.values()):
        raise ValueError('Baseline does not exactly reproduce saved F metrics; stop ablations')
    evaluate('rot4', rot4)
    # Only valid source rows are selected for geometry; archive contains the full immutable dataset.
    with np.load(args.data_file, allow_pickle=False) as archive:
        valid_maps = archive['arr_0'][meta['map_ids']]
        np.testing.assert_array_equal(archive['arr_1'][meta['map_ids']], labels[meta['map_ids']])
    histograms = np.empty((len(masks), 16))
    for raw, sample_id in zip(valid_maps, meta['map_ids']):
        tokens = indices[int(sample_id)]
        regions = extract_regions(raw, **config['proposal'])
        if len(regions) != len(tokens):
            raise ValueError('Re-extracted region count differs')
        for region, token in zip(regions, tokens):
            np.testing.assert_array_equal(region['mask'], masks[token])
            if int(masks[token].sum()) != meta['area'][token]:
                raise ValueError('Mask area mismatch')
            histograms[token] = radial_histogram(masks[token], np.isin(raw, [1, 2]))
    np.save(output/'radial_histograms.npy', histograms)
    write_json(output/'radial_identity_audit.json', {'all_masks_reextracted_equal': True, 'regions': len(masks),
        'maps': len(valid_maps), 'only_valid_ids_used': True, 'histogram_sum_max_error': float(abs(histograms.sum(1)-1).max())})
    for beta in betas[1:]:
        evaluate(f'radial_b{beta:g}', baseline, beta, histograms)
    names = {0.: 'baseline', **{b: f'radial_b{b:g}' for b in betas[1:]}}
    best = max(betas, key=lambda b: (results[names[b]]['metrics']['mAP@10'], -b))
    def improved(result):
        return (result['metrics']['mAP@10'] > reference['metrics']['mAP@10'] and
                result['groups']['mixed_label']['metrics']['mAP@10'] >= reference['groups']['mixed_label']['metrics']['mAP@10'])
    combine = improved(results['rot4']) and improved(results[names[best]])
    decision = {'selected_beta': best, 'selected_radial_variant': names[best], 'selection_metric': 'mAP@10',
                'combination_eligible': combine, 'selection_uses_valid_labels': True,
                'rule': plan['combination_rule'], 'beta_frozen_for_combination': True, 'test_evaluated': False}
    write_json(output/'selection_before_combination.json', decision)
    if combine:
        evaluate('rot4_radial', rot4, best, histograms)
        write_json(output/'rot4_radial/paired_vs_radial.json', paired_report(output/names[best], output/'rot4_radial'))
        write_json(output/'rot4_radial/paired_vs_rot4.json', paired_report(output/'rot4', output/'rot4_radial'))
    summary = {name: {'metrics': v['metrics'], 'class_macro_metrics': v['class_macro_metrics'],
                'single': v['groups']['single_label']['metrics'], 'mixed': v['groups']['mixed_label']['metrics']}
               for name, v in results.items()}
    write_json(output/'summary.json', {'results': summary, 'decision': decision, 'test_evaluated': False})
    write_json(output/'completed.json', {'status': 'completed', 'test_evaluated': False,
        'files': {str(p.relative_to(output)): file_hash(p) for p in output.rglob('*') if p.is_file()}})
    print('COMPLETE '+json.dumps(decision), flush=True)


if __name__ == '__main__':
    main()
