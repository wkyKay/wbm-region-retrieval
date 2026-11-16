#!/usr/bin/env python3
"""Frozen valid-only comparison: radial F, plus relative area, plus PCA axes.

No training, checkpoint selection, hyperparameter search, or test evaluation.
Original caches and existing results are read-only; output must be new.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
METHODS = ('radial', 'radial_area', 'radial_axes')
SIGMA = 1.5
SCALE_WEIGHT = .15


def scale_descriptor(mask, support):
    """[sqrt(relative area), relative long axis, relative short axis].

    Population covariance on full original-grid mask pixels. PDF's +1 pixel^2
    regularizer is inside sqrt; both axes are divided by sqrt(support area).
    This regularizer prevents log(0), but weakens small-axis scale sensitivity.
    """
    mask, support = np.asarray(mask, bool), np.asarray(support, bool)
    if mask.ndim != 2 or mask.shape != support.shape or np.any(mask & ~support):
        raise ValueError('Invalid mask/support geometry')
    points = np.argwhere(mask).astype(float)
    valid_area = int(support.sum())
    if not len(points) or not valid_area:
        raise ValueError('Empty region/support')
    centered = points - points.mean(0)
    eigenvalues = np.maximum(np.linalg.eigvalsh(centered.T @ centered / len(points)), 0)[::-1]
    axes = np.sqrt(eigenvalues + 1) / np.sqrt(valid_area)
    return np.r_[np.sqrt(len(points) / valid_area), axes], eigenvalues


def scale_distance(a, b, kind):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape[-1] != 3 or b.shape[-1] != 3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected finite three-component scale descriptors')
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError('Scale descriptors must be positive')
    delta = np.abs(np.log(a) - np.log(b))
    if kind == 'area':
        return delta[..., 0]
    if kind == 'axes':
        return .75 * delta[..., 1] + .25 * delta[..., 2]
    raise ValueError('Unknown scale kind')


def apply_scale(radial_scores, distance, weight=SCALE_WEIGHT, sigma=SIGMA):
    if not 0 <= weight <= 1 or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Invalid scale settings')
    distance = np.asarray(distance, float)
    if not np.isfinite(distance).all() or np.any(distance < 0):
        raise ValueError('Invalid scale distance')
    return np.asarray(radial_scores) * ((1 - weight) + weight * np.exp(-distance / sigma))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-run', 'valid-run', 'data-file', 'compared-root', 'implementation', 'output-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    source, valid, output = args.source_run.resolve(), args.valid_run.resolve(), args.output_dir.resolve()
    compared = args.compared_root.resolve()
    sys.path.insert(0, str(compared))
    from wbm_retrieval.common import read_json, write_json, file_hash, digest
    from wbm_retrieval.data import decode_pool, source_fingerprint
    from wbm_retrieval.diagnostics import batched_map_scores
    from wbm_retrieval.exact_evaluation import evaluate_suite_rankings
    from wbm_retrieval.exact_metrics import order_scores
    from wbm_retrieval.matching import greedy_assignment
    from wbm_retrieval.regions import extract_regions
    from shared.coverage_metrics import evaluate_rankings_file

    if output.exists() or any(output == x or x in output.parents or output in x.parents for x in (source, valid)):
        raise ValueError('Use a new output directory outside source runs')
    previous_plan = read_json(valid / 'plan_before_evaluation.json')
    previous_done = read_json(valid / 'completed.json')
    decision = read_json(valid / 'selection_before_combination.json')
    assert previous_done['status'] == 'completed'
    assert decision['selected_beta'] == 1 and decision['selected_radial_variant'] == 'radial_b1'
    assert previous_plan['radial_bins'] == 16
    assert file_hash(args.implementation) == previous_plan['script_sha256']
    reused = ('plan_before_evaluation.json', 'selection_before_combination.json', 'baseline_h512.npy',
              'radial_histograms.npy', 'radial_b1/rankings.csv', 'radial_b1/metrics.json')
    for name in reused:
        assert file_hash(valid / name) == previous_done['files'][name], name
    module_spec = importlib.util.spec_from_file_location('f_frozen_radial', args.implementation)
    radial = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(radial)

    manifest, prepared = read_json(source / 'suite_manifest.json'), read_json(source / 'cache/prepared.json')
    assert manifest['signature'] == prepared['signature'] == digest({k: v for k, v in manifest.items() if k != 'signature'})
    config = read_json(source / 'seed_2026/F/run_manifest.json')['config']
    assert config['matching'] == manifest['settings']['matching']
    assert config['matching']['lambda_pos'] == config['matching']['lambda_scale'] == 0
    assert config['matching']['shape_threshold'] == config['matching']['score_threshold'] == 0
    assert config['matching']['weight'] == 'sqrt_area'
    assert config['evaluation']['metric_version'] == 'exact_v1'
    assert config['evaluation']['ks'] == [1, 5, 10, 20, 50, 100, 500]
    assert source_fingerprint(config) == manifest['sources']
    for name, sha in manifest['code'].items():
        assert file_hash(ROOT / name) == sha, name
    assert file_hash(args.data_file) == manifest['sources']['data_file'] == previous_plan['data_sha256']
    assert file_hash(ROOT / 'wbm_retrieval/exact_metrics.py') == file_hash(compared / 'shared/exact_metrics.py')
    cache_names = ('cache/valid_regions.npz', 'cache/valid_masks.npy', 'cache/valid_F.npy', 'cache/labels.npy', 'valid_pool.json')
    for name in cache_names:
        assert file_hash(source / name) == prepared['files'][name], name
    checkpoint = source / 'seed_2026/F/best_nce.pt'
    assert file_hash(checkpoint) == previous_plan['checkpoint_sha256']
    assert read_json(source / 'seed_2026/F/training_summary.json')['completed']
    payload = read_json(source / 'valid_pool.json')
    queries, pool = decode_pool(payload)
    assert len(queries) == 500 and len(pool.gallery) == 3599 and pool.candidates_per_query == 3598
    protocol_files = [ROOT / 'protocol/retrieval_v1' / name for name in ('valid_queries.csv', 'valid_gallery.csv', 'metadata.json')]
    for path in protocol_files:
        assert file_hash(path) == manifest['sources']['retrieval_protocol/' + path.name]
        assert file_hash(path) == file_hash(compared / 'protocol/retrieval_v1' / path.name)
    for name, expected in (('valid_queries.csv', queries), ('valid_gallery.csv', pool.gallery)):
        with (ROOT / 'protocol/retrieval_v1' / name).open() as stream:
            np.testing.assert_array_equal([int(r['sample_id']) for r in csv.DictReader(stream)], expected)
    with np.load(source / 'cache/valid_regions.npz') as archive:
        meta = dict(archive)
    np.testing.assert_array_equal(meta['map_ids'], pool.gallery)
    labels = np.load(source / 'cache/labels.npy')
    features, histograms = np.load(valid / 'baseline_h512.npy'), np.load(valid / 'radial_histograms.npy')
    masks = np.load(source / 'cache/valid_masks.npy', mmap_mode='r')
    assert len(features) == len(histograms) == len(masks) == meta['offsets'][-1]
    assert features.shape[1] == 512 and histograms.shape[1] == 16
    np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1, atol=1e-12)
    indices = {int(i): np.arange(s, e) for i, s, e in zip(meta['map_ids'], meta['offsets'][:-1], meta['offsets'][1:])}
    positions = {int(i): r for r, i in enumerate(meta['map_ids'])}
    padded = np.full((len(meta['map_ids']), int(np.diff(meta['offsets']).max())), -1, dtype=int)
    for i, tokens in indices.items():
        padded[positions[i], :len(tokens)] = tokens
    protected = [source / n for n in cache_names] + [source / 'suite_manifest.json', source / 'cache/prepared.json', checkpoint]
    protected += [valid / n for n in reused] + [valid / 'completed.json'] + protocol_files
    protected += [ROOT / n for n in manifest['code']] + [args.data_file, args.implementation, compared / 'shared/coverage_metrics.py']
    snapshot = {str(p): {'sha256': file_hash(p), 'mtime_ns': p.stat().st_mtime_ns} for p in protected}
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    plan = {
        'methods': list(METHODS), 'split': 'valid', 'test_evaluated': False, 'training_required': False,
        'checkpoint': str(checkpoint), 'checkpoint_sha256': file_hash(checkpoint), 'epoch': previous_plan['epoch'],
        'source_run': str(source), 'valid_run': str(valid), 'source_snapshot': snapshot,
        'config': config, 'query_count': 500, 'gallery_count': 3599, 'candidates_per_query': 3598,
        'directed_pairs_per_method': 500 * 3598, 'candidate_rule': 'all_gallery_except_query', 'ranking_limit': 500,
        'retrieval_protocol_signature': pool.protocol_signature, 'script_sha256': file_hash(__file__),
        'radial_implementation_sha256': file_hash(args.implementation), 'beta': 1, 'bins': 16,
        'rotation_pooling': False, 'bidirectional': False, 'hyperparameter_search': False,
        'sigma_scale': SIGMA, 'scale_weight': SCALE_WEIGHT, 'axis_weights': [.75, .25],
        'scale_definitions': {
            'support': 'original raw pixels equal to 1 or 2; A_valid=count(support)',
            'relative_area': 'r=region_area/A_valid; t=sqrt(r); D_area=abs(log(t_q/t_c))',
            'axes': 'full original-mask population covariance; l=sqrt(lambda_max+1)/sqrt(A_valid); s=sqrt(lambda_min+1)/sqrt(A_valid)',
            'axes_distance': 'D_axes=0.75*abs(log(l_q/l_c))+0.25*abs(log(s_q/s_c))',
            'fusion': 'S_radial=clip(cos(h_q,h_c),0,1)*exp(-D_radial); S_new=S_radial*(0.85+0.15*exp(-D_scale/1.5))',
            'sensitivity': 'sqrt(relative area) and weighted axes share linear-scale log units; axes weights sum to 1; same sigma and fusion weight',
            'regularization_limit': 'PDF +1 pixel^2 makes axes sensitivity weaker for tiny/degenerate regions, so exact geometric scale equivariance is not claimed',
            'adaptation': 'axes divided by sqrt(A_valid) to use the same wafer-relative linear units as sqrt(relative area)',
        },
        'aggregation': 'unchanged one-direction greedy one-to-one and query sqrt(original area) weighting',
        'selection': 'compare these three predefined settings only; no parameter search or automatic test',
    }
    write_json(output / 'plan_before_evaluation.json', plan)

    # Re-extraction verifies token identity; only valid image rows enter geometry.
    with np.load(args.data_file, allow_pickle=False) as archive:
        maps = archive['arr_0'][meta['map_ids']]
        np.testing.assert_array_equal(archive['arr_1'][meta['map_ids']], labels[meta['map_ids']])
    descriptors, eigenvalues = np.empty((len(masks), 3)), np.empty((len(masks), 2))
    supports = []
    for raw, sample_id in zip(maps, meta['map_ids']):
        tokens, support = indices[int(sample_id)], np.isin(raw, [1, 2])
        supports.append(int(support.sum()))
        regions = extract_regions(raw, **config['proposal'])
        assert len(regions) == len(tokens)
        for region, token in zip(regions, tokens):
            np.testing.assert_array_equal(region['mask'], masks[token])
            assert int(masks[token].sum()) == meta['area'][token]
            np.testing.assert_array_equal(radial.radial_histogram(masks[token], support), histograms[token])
            descriptors[token], eigenvalues[token] = scale_descriptor(masks[token], support)
    np.savez(output / 'scale_geometry.npz', descriptors=descriptors, eigenvalues=eigenvalues,
             support_area=np.asarray(supports), map_ids=meta['map_ids'], offsets=meta['offsets'], area=meta['area'])
    geom = {'maps': len(maps), 'regions': len(masks), 'only_valid_ids_used': True,
            'all_masks_reextracted_equal': True, 'radial_histograms_identical': True,
            'support_area_min': min(supports), 'support_area_max': max(supports),
            'support_area_unique': sorted(set(supports)),
            'short_axis_eigenvalue_below_one': int((eigenvalues[:, 1] < 1).sum()),
            'short_axis_eigenvalue_zero': int((eigenvalues[:, 1] == 0).sum())}
    write_json(output / 'geometry_audit.json', geom)
    print('Geometry verified: ' + str(geom), flush=True)

    # Validate vectorized matching against independently assembled scalar pairs.
    sample_ids = meta['map_ids'][np.linspace(0, len(meta['map_ids']) - 1, 16, dtype=int)]
    parity = {}
    for method in METHODS:
        maximum = 0.
        for q in sample_ids:
            qt = indices[int(q)]
            similarity = np.clip(features[qt] @ features.T, 0, 1) * np.exp(-radial.radial_distance(histograms[qt, None], histograms[None]))
            if method != 'radial':
                similarity = apply_scale(similarity, scale_distance(descriptors[qt, None], descriptors[None], method.split('_')[1]))
            actual = batched_map_scores(similarity, padded[[positions[int(c)] for c in sample_ids]], meta['area'][qt], config['matching'])
            expected = []
            for c in sample_ids:
                ct = indices[int(c)]
                matrix = np.empty((len(qt), len(ct)))
                for r, ti in enumerate(qt):
                    for s, tj in enumerate(ct):
                        d = sum(abs(float(histograms[ti, :k].sum()) - float(histograms[tj, :k].sum())) for k in range(1, 17)) / 16
                        score = np.clip(np.dot(features[ti], features[tj]), 0, 1) * np.exp(-d)
                        if method == 'radial_area':
                            scale_d = .5 * abs(np.log(descriptors[ti, 0] ** 2 / descriptors[tj, 0] ** 2))
                        elif method == 'radial_axes':
                            scale_d = .75 * abs(np.log(descriptors[ti, 1] / descriptors[tj, 1])) + .25 * abs(np.log(descriptors[ti, 2] / descriptors[tj, 2]))
                        if method != 'radial':
                            score *= .85 + .15 * np.exp(-scale_d / 1.5)
                        matrix[r, s] = score
                weights = np.sqrt(meta['area'][qt])
                expected.append(sum(weights[r] * v for r, s, v in greedy_assignment(matrix)) / weights.sum() if len(qt) else 0.)
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
            maximum = max(maximum, float(np.max(np.abs(actual - expected))))
        parity[method] = {'scalar_batched_pairs': 256, 'max_error': maximum}
    write_json(output / 'matching_parity_audit.json', parity)

    taxonomy = sorted({tuple(labels[i]) for i in meta['map_ids']})
    assert len(taxonomy) == 37
    class_map = {x: i for i, x in enumerate(taxonomy)}
    class_ids = np.full(len(labels), -1, dtype=int)
    for i in meta['map_ids']:
        class_ids[i] = class_map[tuple(labels[i])]
    results = {}
    for method in METHODS:
        method_started = time.monotonic()
        target = output / method
        target.mkdir()
        metadata = {'method': method, 'beta': 1, 'sigma_scale': SIGMA,
                    'scale_weight': 0 if method == 'radial' else SCALE_WEIGHT,
                    'checkpoint_sha256': file_hash(checkpoint), 'representation': 'h512',
                    'aggregation': 'original_F', 'candidates_scored': 'full_pool', 'ranking_limit': 500,
                    'retrieval_protocol_signature': pool.protocol_signature, 'bootstrap_samples': 0,
                    'candidate_rule': 'all_gallery_except_query', 'smoke': False}
        with (target / 'rankings.csv').open('x', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['query_id', 'rank', 'candidate_id', 'similarity_score'])
            def rankings():
                for number, q in enumerate(queries, 1):
                    c, qt = np.asarray(pool[q]), indices[int(q)]
                    similarity = np.clip(features[qt] @ features.T, 0, 1) * np.exp(-radial.radial_distance(histograms[qt, None], histograms[None]))
                    if method != 'radial':
                        similarity = apply_scale(similarity, scale_distance(descriptors[qt, None], descriptors[None], method.split('_')[1]))
                    scores = batched_map_scores(similarity, padded[[positions[int(i)] for i in c]], meta['area'][qt], config['matching'])
                    assert len(scores) == 3598 and np.isfinite(scores).all()
                    order = order_scores(c, scores, q, config['evaluation'].get('tie_seed', 2026))[:500]
                    writer.writerows((int(q), rank, int(c[i]), float(scores[i])) for rank, i in enumerate(order, 1))
                    yield int(q), c[order]
                    if number % 100 == 0:
                        print(f'{method}: {number}/500 ({time.monotonic() - method_started:.1f}s)', flush=True)
            exact = evaluate_suite_rankings(config, labels, pool, rankings(), target, 'valid',
                                           split_ids=meta['map_ids'], metadata=metadata)
        if method == 'radial':
            previous_metrics = read_json(valid / 'radial_b1/metrics.json')
            np.testing.assert_array_equal(list(exact['metrics'].values()), list(previous_metrics['metrics'].values()))
            assert file_hash(target / 'rankings.csv') == file_hash(valid / 'radial_b1/rankings.csv')
            write_json(output / 'baseline_replay_audit.json', {'all_500_top500_rankings_and_scores_byte_identical': True,
                       'all_exact_metrics_identical': True, 'previous_result': str(valid / 'radial_b1'),
                       'rankings_sha256': file_hash(target / 'rankings.csv')})
        coverage = evaluate_rankings_file(SimpleNamespace(labels=labels, class_ids=class_ids), pool,
            target / 'rankings.csv', config['evaluation']['ks'], target, split_name='valid',
            metadata={**metadata, 'ranking_source_sha256': file_hash(target / 'rankings.csv')})
        results[method] = {'exact': exact['metrics'], 'coverage': coverage['metrics'],
                           'seconds': time.monotonic() - method_started}
        print(f"RESULT {method}: mAP10={exact['metrics']['mAP@10']}, nDCG10={coverage['metrics']['Coverage-nDCG@10']}", flush=True)
    for name, state in snapshot.items():
        path = Path(name)
        assert file_hash(path) == state['sha256'] and path.stat().st_mtime_ns == state['mtime_ns'], name
    write_json(output / 'summary.json', results)
    write_json(output / 'completed.json', {'status': 'completed', 'split': 'valid', 'test_evaluated': False,
        'source_files_unchanged': True, 'query_count': 500, 'gallery_count': 3599, 'candidates_per_query': 3598,
        'methods': list(METHODS), 'seconds': time.monotonic() - started,
        'files': {str(p.relative_to(output)): file_hash(p) for p in output.rglob('*') if p.is_file()}})
    print('COMPLETE: frozen valid three-way comparison', flush=True)


if __name__ == '__main__':
    main()
