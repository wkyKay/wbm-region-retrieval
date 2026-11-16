#!/usr/bin/env python3
"""One complete test of radial F + relative area, frozen after valid selection.

Reuses hash-bound, already normalized test h512 features from the radial test.
No training or parameter selection occurs here; all candidates are rescored.
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


def load_implementation(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def indexing(meta):
    indices = {int(i): np.arange(s, e) for i, s, e in
               zip(meta['map_ids'], meta['offsets'][:-1], meta['offsets'][1:])}
    positions = {int(i): n for n, i in enumerate(meta['map_ids'])}
    padded = np.full((len(meta['map_ids']), int(np.diff(meta['offsets']).max())), -1, dtype=int)
    for i, tokens in indices.items():
        padded[positions[i], :len(tokens)] = tokens
    return indices, positions, padded


def saved_selected(path, selected):
    selected = set(map(int, selected))
    rows = {q: [] for q in selected}
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            q = int(row['query_id'])
            if q in rows:
                rows[q].append((int(row['candidate_id']), float(row['similarity_score'])))
    if any(len(r) != 500 for r in rows.values()):
        raise ValueError('Incomplete reference Top500')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-run', 'valid-run', 'radial-valid-run', 'radial-test-run', 'data-file',
                 'compared-root', 'scale-implementation', 'radial-implementation', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    source, valid, old_valid, old_test, output = [p.resolve() for p in
        (args.source_run, args.valid_run, args.radial_valid_run, args.radial_test_run, args.output_dir)]
    compared = args.compared_root.resolve()
    sys.path.insert(0, str(compared))
    from wbm_retrieval.common import read_json, write_json, file_hash, digest
    from wbm_retrieval.data import Protocol, decode_pool, source_fingerprint
    from wbm_retrieval.diagnostics import batched_map_scores
    from wbm_retrieval.exact_evaluation import evaluate_suite_rankings
    from wbm_retrieval.exact_metrics import order_scores
    from wbm_retrieval.matching import greedy_assignment
    from wbm_retrieval.regions import extract_regions
    from shared.coverage_metrics import evaluate_rankings_file

    if output.exists() or any(output == x or x in output.parents or output in x.parents
                              for x in (source, valid, old_valid, old_test)):
        raise ValueError('Use a new output directory outside all existing runs')
    plan = read_json(valid / 'plan_before_evaluation.json')
    done = read_json(valid / 'completed.json')
    radial_plan = read_json(old_valid / 'plan_before_evaluation.json')
    radial_done = read_json(old_valid / 'completed.json')
    test_plan = read_json(old_test / 'selection_before_test.json')
    test_done = read_json(old_test / 'completed.json')
    assert all(d['status'] == 'completed' for d in (done, radial_done, test_done))
    assert plan['split'] == 'valid' and not plan['test_evaluated']
    assert plan['beta'] == test_plan['beta'] == 1 and plan['bins'] == test_plan['bins'] == 16
    assert plan['sigma_scale'] == 1.5 and plan['scale_weight'] == .15
    assert not plan['rotation_pooling'] and not plan['bidirectional'] and not plan['hyperparameter_search']
    assert Path(plan['source_run']) == source and Path(plan['valid_run']) == old_valid
    assert file_hash(args.scale_implementation) == plan['script_sha256']
    assert file_hash(args.radial_implementation) == plan['radial_implementation_sha256'] == radial_plan['script_sha256'] == test_plan['implementation_sha256']
    scale = load_implementation(args.scale_implementation, 'f_frozen_scale')
    radial = load_implementation(args.radial_implementation, 'f_frozen_radial')
    assert scale.SIGMA == plan['sigma_scale'] and scale.SCALE_WEIGHT == plan['scale_weight']
    reuse = {
        valid: ('plan_before_evaluation.json', 'scale_geometry.npz', 'radial_area/rankings.csv',
                'radial_area/metrics.json', 'radial_area/coverage_metrics.json'),
        old_valid: ('plan_before_evaluation.json', 'baseline_h512.npy', 'radial_histograms.npy'),
        old_test: ('selection_before_test.json', 'test_h512.npy', 'test_radial_histograms.npy',
                   'test/rankings.csv', 'test/metrics.json', 'test/coverage_metrics.json'),
    }
    for directory, completion in ((valid, done), (old_valid, radial_done), (old_test, test_done)):
        for name in reuse[directory]:
            assert file_hash(directory / name) == completion['files'][name], str(directory / name)
    manifest, prepared = read_json(source / 'suite_manifest.json'), read_json(source / 'cache/prepared.json')
    run = read_json(source / 'seed_2026/F/run_manifest.json')
    config = run['config']
    assert manifest['signature'] == prepared['signature'] == radial_plan['suite_signature']
    assert manifest['signature'] == digest({k: v for k, v in manifest.items() if k != 'signature'})
    assert run['signature'] == digest({'suite': manifest['signature'], 'seed': 2026, 'variant': 'F'})
    assert run['variant'] == 'F' and config['seed'] == 2026
    assert config == plan['config'] == test_plan['config']
    assert config['matching'] == manifest['settings']['matching']
    assert config['matching']['lambda_pos'] == config['matching']['lambda_scale'] == 0
    assert config['matching']['shape_threshold'] == config['matching']['score_threshold'] == 0
    assert config['matching']['weight'] == 'sqrt_area'
    assert config['evaluation']['ks'] == [1, 5, 10, 20, 50, 100, 500]
    assert config['evaluation']['metric_version'] == 'exact_v1' and config['evaluation']['tie_seed'] == 2026
    assert source_fingerprint(config) == manifest['sources']
    for name, sha in manifest['code'].items():
        assert file_hash(ROOT / name) == sha, name
    checkpoint = source / 'seed_2026/F/best_nce.pt'
    assert file_hash(checkpoint) == plan['checkpoint_sha256'] == test_plan['checkpoint_sha256'] == radial_plan['checkpoint_sha256']
    assert read_json(source / 'seed_2026/F/training_summary.json')['completed']
    assert file_hash(args.data_file) == manifest['sources']['data_file'] == radial_plan['data_sha256']
    assert file_hash(ROOT / 'wbm_retrieval/exact_metrics.py') == file_hash(compared / 'shared/exact_metrics.py')
    cache_names = ('cache/test_regions.npz', 'cache/test_masks.npy', 'cache/test_F.npy', 'cache/labels.npy',
                   'cache/valid_regions.npz', 'cache/valid_masks.npy', 'cache/valid_F.npy', 'valid_pool.json')
    for name in cache_names:
        actual = file_hash(source / name)
        assert prepared['files'][name] == actual, name
        old_snapshot = test_plan['source_snapshot'] if name.startswith('cache/test') else plan['source_snapshot']
        assert old_snapshot[str(source / name)]['sha256'] == actual, name
    for name in ('suite_manifest.json', 'cache/prepared.json', 'seed_2026/F/best_nce.pt'):
        actual = file_hash(source / name)
        assert plan['source_snapshot'][str(source / name)]['sha256'] == test_plan['source_snapshot'][str(source / name)]['sha256'] == actual
    protocol = Protocol(config)
    pool = protocol.candidates
    assert len(protocol.queries) == 7199 and pool.candidates_per_query == 7198
    assert protocol.retrieval.signature == plan['retrieval_protocol_signature'] == test_plan['retrieval_protocol_signature']
    protocol_files = [ROOT / 'protocol/retrieval_v1' / n for n in
                      ('valid_queries.csv', 'valid_gallery.csv', 'test_queries.csv', 'test_gallery.csv', 'metadata.json')]
    for path in protocol_files:
        assert file_hash(path) == manifest['sources']['retrieval_protocol/' + path.name]
        assert file_hash(path) == file_hash(compared / 'protocol/retrieval_v1' / path.name)
    protected = [source / n for n in cache_names] + [source / n for n in
        ('suite_manifest.json', 'cache/prepared.json', 'seed_2026/F/best_nce.pt',
         'seed_2026/F/run_manifest.json', 'seed_2026/F/training_summary.json')]
    protected += [ROOT / n for n in manifest['code']] + protocol_files
    protected += [args.data_file, args.scale_implementation, args.radial_implementation,
                  Path(__file__), compared / 'shared/coverage_metrics.py']
    for directory, names in reuse.items():
        protected += [directory / n for n in (*names, 'completed.json')]
    snapshot = {str(f): {'sha256': file_hash(f), 'mtime_ns': f.stat().st_mtime_ns} for f in protected}
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    selection = {'method': 'F_radial_area', 'split': 'test', 'beta': 1, 'bins': 16,
        'sigma_scale': 1.5, 'scale_weight': .15, 'scale_kind': 'sqrt_relative_area',
        'pair_score': 'clip(cos(hq,hc),0,1)*exp(-D_radial)*(0.85+0.15*exp(-abs(log(sqrt((aq/Aq)/(ac/Ac))))/1.5))',
        'selection_rule': 'User selected radial+relative area after the completed three-way valid comparison; all parameters copied from valid',
        'valid_plan': str(valid / 'plan_before_evaluation.json'), 'valid_plan_sha256': file_hash(valid / 'plan_before_evaluation.json'),
        'selection_uses_valid_labels': True, 'parameter_search_on_test': False, 'training_required': False,
        'rotation_pooling': False, 'bidirectional': False, 'checkpoint': str(checkpoint),
        'checkpoint_sha256': file_hash(checkpoint), 'epoch': plan['epoch'], 'config': config,
        'query_count': 7199, 'gallery_count': 7199, 'candidates_per_query': 7198,
        'total_directed_pairs': 7199 * 7198, 'ranking_limit': 500, 'candidate_rule': 'all_gallery_except_query',
        'retrieval_protocol_signature': protocol.retrieval.signature,
        'feature_reuse': 'Read hash-bound normalized h512 from previous radial test without renormalization',
        'source_snapshot': snapshot, 'scale_implementation_sha256': file_hash(args.scale_implementation),
        'radial_implementation_sha256': file_hash(args.radial_implementation), 'script_sha256': file_hash(__file__)}
    write_json(output / 'selection_before_test.json', selection)

    def pair_scores(features, hist, desc, tokens, use_area=True):
        scores = np.clip(features[tokens] @ features.T, 0, 1) * np.exp(-radial.radial_distance(hist[tokens, None], hist[None]))
        return scale.apply_scale(scores, scale.scale_distance(desc[tokens, None], desc[None], 'area'),
                                 weight=plan['scale_weight'], sigma=plan['sigma_scale']) if use_area else scores

    def replay(meta, features, hist, desc, queries, reference, use_area):
        indices, positions, padded = indexing(meta)
        ids = np.array(queries)[np.linspace(0, len(queries) - 1, 32, dtype=int)]
        saved = saved_selected(reference, ids)
        maximum = 0.
        for q in ids:
            qt = indices[int(q)]
            sim = pair_scores(features, hist, desc, qt, use_area)
            candidates = meta['map_ids'][meta['map_ids'] != q]
            ci = padded[[positions[int(c)] for c in candidates]]
            scores = batched_map_scores(sim, ci, meta['area'][qt], config['matching'])
            order = order_scores(candidates, scores, q, 2026)[:500]
            old_ids, old_scores = zip(*saved[int(q)])
            np.testing.assert_array_equal(candidates[order], old_ids)
            np.testing.assert_allclose(scores[order], old_scores, atol=1e-12, rtol=0)
            maximum = max(maximum, float(np.max(abs(scores[order] - old_scores))))
        return {'queries': 32, 'top500_ids_identical': True, 'score_atol': 1e-12, 'max_score_error': maximum}

    # Reproduce frozen valid-area rankings before any new test scoring.
    with np.load(source / 'cache/valid_regions.npz') as z:
        vm = dict(z)
    with np.load(valid / 'scale_geometry.npz') as z:
        vd = z['descriptors']
        np.testing.assert_array_equal(z['map_ids'], vm['map_ids'])
        np.testing.assert_array_equal(z['offsets'], vm['offsets'])
        np.testing.assert_array_equal(z['area'], vm['area'])
    vq, _ = decode_pool(read_json(source / 'valid_pool.json'))
    vf, vh = np.load(old_valid / 'baseline_h512.npy'), np.load(old_valid / 'radial_histograms.npy')
    valid_audit = replay(vm, vf, vh, vd, vq, valid / 'radial_area/rankings.csv', True)
    write_json(output / 'valid_replay_audit.json', valid_audit)
    del vm, vf, vh, vd
    print('Frozen valid area ranks and scores reproduced', flush=True)

    with np.load(source / 'cache/test_regions.npz') as z:
        meta = dict(z)
    np.testing.assert_array_equal(meta['map_ids'], pool.gallery)
    features = np.load(old_test / 'test_h512.npy')
    hist = np.load(old_test / 'test_radial_histograms.npy')
    masks = np.load(source / 'cache/test_masks.npy', mmap_mode='r')
    labels = np.load(source / 'cache/labels.npy')
    assert len(features) == len(hist) == len(masks) == meta['offsets'][-1]
    assert features.shape[1] == 512 and hist.shape[1] == 16
    np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1, atol=1e-12)
    indices, positions, padded = indexing(meta)
    with np.load(args.data_file, allow_pickle=False) as z:
        maps = z['arr_0'][meta['map_ids']]
        np.testing.assert_array_equal(z['arr_1'][meta['map_ids']], labels[meta['map_ids']])
    desc = np.empty((len(masks), 3))
    support_area = np.empty(len(meta['map_ids']), dtype=int)
    token_support_area = np.empty(len(masks), dtype=int)
    for row, (raw, i) in enumerate(zip(maps, meta['map_ids'])):
        support, tokens = np.isin(raw, [1, 2]), indices[int(i)]
        support_area[row] = int(support.sum())
        token_support_area[tokens] = support_area[row]
        regions = extract_regions(raw, **config['proposal'])
        assert len(regions) == len(tokens)
        for region, t in zip(regions, tokens):
            np.testing.assert_array_equal(region['mask'], masks[t])
            assert int(masks[t].sum()) == meta['area'][t]
            np.testing.assert_array_equal(radial.radial_histogram(masks[t], support), hist[t])
            desc[t], _ = scale.scale_descriptor(masks[t], support)
    np.testing.assert_array_equal(desc[:, 0], np.sqrt(meta['area'] / token_support_area))
    np.savez(output / 'test_area_geometry.npz', sqrt_relative_area=desc[:, 0],
             map_ids=meta['map_ids'], offsets=meta['offsets'], area=meta['area'], support_area=support_area)
    write_json(output / 'test_radial_replay_audit.json', replay(meta, features, hist, desc,
        protocol.queries, old_test / 'test/rankings.csv', False))
    # Reference computes both geometry penalties from raw values, not scale helper.
    chosen = meta['map_ids'][np.linspace(0, len(meta['map_ids']) - 1, 32, dtype=int)]
    maximum = 0.
    for q in chosen:
        qt = indices[int(q)]
        sim = pair_scores(features, hist, desc, qt)
        actual = batched_map_scores(sim, padded[[positions[int(c)] for c in chosen]], meta['area'][qt], config['matching'])
        expected = []
        for c in chosen:
            ct = indices[int(c)]
            matrix = np.empty((len(qt), len(ct)))
            for r, ti in enumerate(qt):
                for s, tj in enumerate(ct):
                    d_radial = sum(abs(float(hist[ti, :k].sum()) - float(hist[tj, :k].sum())) for k in range(1, 17)) / 16
                    rq = int(masks[ti].sum()) / int(token_support_area[ti])
                    rc = int(masks[tj].sum()) / int(token_support_area[tj])
                    d_area = .5 * abs(np.log(rq / rc))
                    matrix[r, s] = np.clip(np.dot(features[ti], features[tj]), 0, 1) * np.exp(-d_radial) * (.85 + .15 * np.exp(-d_area / 1.5))
            weights = np.sqrt(meta['area'][qt])
            expected.append(sum(weights[r] * v for r, s, v in greedy_assignment(matrix)) / weights.sum() if len(qt) else 0.)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
        maximum = max(maximum, float(np.max(abs(actual - expected))))
    write_json(output / 'geometry_matching_audit.json', {'maps': len(maps), 'regions': len(masks),
        'all_masks_reextracted_equal': True, 'all_radial_histograms_identical': True,
        'sqrt_relative_area_raw_count_identity': True, 'only_area_component_enters_score': True,
        'support_area_min': int(support_area.min()), 'support_area_max': int(support_area.max()),
        'scalar_batched_pairs': 1024, 'max_error': maximum})
    print('Test geometry, old radial ranks and scalar matching verified', flush=True)

    target = output / 'test'
    target.mkdir()
    metadata = {'method': 'F_radial_area', 'beta': 1, 'sigma_scale': 1.5, 'scale_weight': .15,
        'scale_kind': 'sqrt_relative_area', 'representation': 'h512', 'aggregation': 'original_F',
        'checkpoint_sha256': file_hash(checkpoint), 'retrieval_protocol_signature': protocol.retrieval.signature,
        'candidates_scored': 'full_pool', 'ranking_limit': 500, 'bootstrap_samples': 0,
        'candidate_rule': 'all_gallery_except_query', 'smoke': False}
    with (target / 'rankings.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['query_id', 'rank', 'candidate_id', 'similarity_score'])
        def rankings():
            for number, q in enumerate(protocol.queries, 1):
                candidates, qt = np.asarray(pool[q]), indices[int(q)]
                similarity = pair_scores(features, hist, desc, qt)
                scores = batched_map_scores(similarity, padded[[positions[int(c)] for c in candidates]], meta['area'][qt], config['matching'])
                assert len(scores) == 7198 and np.isfinite(scores).all()
                order = order_scores(candidates, scores, q, 2026)[:500]
                writer.writerows((int(q), rank, int(candidates[i]), float(scores[i])) for rank, i in enumerate(order, 1))
                yield int(q), candidates[order]
                if number % 250 == 0:
                    print(f'Test area {number}/7199 ({time.monotonic() - started:.1f}s)', flush=True)
        exact = evaluate_suite_rankings(config, labels, pool, rankings(), target, 'test',
                                       split_ids=meta['map_ids'], metadata=metadata)
    taxonomy = sorted({tuple(labels[i]) for i in protocol.ids['train']})
    assert len(taxonomy) == 37
    class_map = {x: i for i, x in enumerate(taxonomy)}
    class_ids = np.full(len(labels), -1, dtype=int)
    for i in meta['map_ids']:
        class_ids[i] = class_map[tuple(labels[i])]
    print('Exact complete; computing coverage from the same Top500 rankings', flush=True)
    coverage = evaluate_rankings_file(SimpleNamespace(labels=labels, class_ids=class_ids), pool,
        target / 'rankings.csv', config['evaluation']['ks'], target, split_name='test',
        metadata={**metadata, 'ranking_source_sha256': file_hash(target / 'rankings.csv')})
    for name, state in snapshot.items():
        path = Path(name)
        assert file_hash(path) == state['sha256'] and path.stat().st_mtime_ns == state['mtime_ns'], name
    write_json(output / 'completed.json', {'status': 'completed', 'test_evaluated': True,
        'query_count': 7199, 'candidates_per_query': 7198, 'source_files_unchanged': True,
        'parameters_copied_from_valid': True, 'parameter_search_on_test': False,
        'seconds': time.monotonic() - started,
        'files': {str(p.relative_to(output)): file_hash(p) for p in output.rglob('*') if p.is_file()}})
    print({'exact': exact['metrics'], 'coverage': coverage['metrics']}, flush=True)
    print('COMPLETE: frozen radial F + relative area full test', flush=True)


if __name__ == '__main__':
    main()
