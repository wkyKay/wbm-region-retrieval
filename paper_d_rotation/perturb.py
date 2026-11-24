"""Frozen query-region evidence deletion diagnostics for the paper's D4 section.

This module reports a diagnostic subset, not a replacement formal benchmark.
Each query's original rank-one candidate defines its explanation. The same
deleted query row is applied to every candidate before rematching/reranking.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from itertools import combinations
import json
import math
import multiprocessing
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('test', 'cross_test')
STRATEGIES = ('baseline', 'high', 'low', 'random')
METRICS = ('Precision@10', 'Coverage-nDCG@10', 'reference_score',
           'reference_rank', 'reference_top10')
COMMON = ('stage', 'query_id', 'class_id', 'label_count', 'num_regions',
          'candidate_count', 'reference_id', 'eligible', 'area_pair')
BASELINE_FIELDS = (*COMMON, *METRICS)
DELETION_FIELDS = (*COMMON, 'deleted_region', 'deleted_area', 'deleted_weight',
                   'reference_contribution', *METRICS,
                   *('delta_'+k for k in METRICS))
STRATEGY_FIELDS = (*COMMON, 'subset', 'strategy', 'selected_regions',
                  'deleted_weight', 'reference_contribution', *METRICS,
                  *('delta_'+k for k in METRICS))
SUMMARY_FIELDS = ('stage', 'subset', 'strategy', 'num_queries', 'metric',
                  'valid_queries', 'mean', 'baseline_mean', 'delta')
_WORKER = None


def scalar_match(pair, areas):
    """Independent scalar greedy replay, returning query-wise contributions."""
    pair = np.asarray(pair, dtype=np.float64)
    areas = np.asarray(areas, dtype=np.float64)
    if pair.ndim != 2 or areas.shape != (len(pair),) or np.any(areas <= 0):
        raise ValueError('Invalid scalar matching input')
    if not np.isfinite(pair).all() or np.any(pair < 0) or np.any(pair > 1):
        raise ValueError('Pair scores must be finite and in [0,1]')
    weights = np.sqrt(areas)
    denominator = float(weights.sum())
    contributions = np.zeros(len(areas), dtype=np.float64)
    used_q, used_c, total = set(), set(), 0.
    for negative, qi, ci in sorted((-float(pair[q, c]), q, c)
            for q in range(pair.shape[0]) for c in range(pair.shape[1])):
        if negative >= 0.:
            break
        if qi in used_q or ci in used_c:
            continue
        used_q.add(qi)
        used_c.add(ci)
        value = float(weights[qi]) * -negative
        total += value
        contributions[qi] = value / denominator
    return (total / denominator if denominator else 0.), contributions


def area_matched_pair(areas, max_weight_ratio=1.2):
    """Choose by area alone before examining contribution or outcomes."""
    areas = np.asarray(areas, dtype=np.float64)
    if areas.ndim != 1 or not np.isfinite(areas).all() or np.any(areas <= 0):
        raise ValueError('Areas must be positive finite values')
    weights = np.sqrt(areas)
    pairs = []
    for i, j in combinations(range(len(areas)), 2):
        ratio = float(max(weights[i], weights[j]) / min(weights[i], weights[j]))
        if ratio <= max_weight_ratio:
            pairs.append((abs(math.log(float(weights[i] / weights[j]))), i, j))
    return None if not pairs else min(pairs)[1:]


def high_low(contributions, indices=None):
    """Both extrema break contribution ties by lowest original region index."""
    values = np.asarray(contributions, dtype=np.float64)
    ids = list(range(len(values))) if indices is None else list(indices)
    if not ids:
        raise ValueError('At least one eligible region is required')
    return (min(ids, key=lambda i: (-float(values[i]), i)),
            min(ids, key=lambda i: (float(values[i]), i)))


def metric_values(ids, scores, query, labels, *, reference_id=None, hashes=None, cross=False):
    """Full-gallery linear-coverage nDCG and original deterministic ranking."""
    from wbm_retrieval.exact_metrics import tie_keys
    ids, scores = np.asarray(ids), np.asarray(scores)
    if (ids.ndim != 1 or scores.shape != ids.shape or not len(ids)
            or not np.isfinite(scores).all() or int(query) in ids):
        raise ValueError('Nonempty complete candidate scores are required')
    hashes = tie_keys(int(query), ids, 2026) if hashes is None else np.asarray(hashes)
    if hashes.shape != ids.shape:
        raise ValueError('Tie keys must align with candidates')
    order = np.lexsort((ids, hashes, -scores))
    qlabels = np.asarray(labels[int(query)], dtype=np.float64)
    if qlabels.sum() <= 0:
        raise ValueError('Query labels cannot be empty')
    coverage = np.asarray(labels[ids], dtype=np.float64) @ qlabels / qlabels.sum()
    discount = 1. / np.log2(np.arange(min(10, len(ids)), dtype=np.float64) + 2.)
    ideal = float(np.sum(np.sort(coverage)[::-1][:10] * discount))
    dcg = float(np.sum(coverage[order[:10]] * discount))
    exact = np.all(labels[ids[order[:10]]] == labels[int(query)], axis=1)
    ref = int(ids[order[0]]) if reference_id is None else int(reference_id)
    position = np.flatnonzero(ids == ref)
    if len(position) != 1:
        raise ValueError('Frozen reference must remain in the complete candidate pool')
    rank = int(np.flatnonzero(order == position[0])[0]) + 1
    return {'Precision@10': None if cross else float(exact.sum() / 10.),
            'Coverage-nDCG@10': dcg / ideal if ideal > 0 else None,
            'reference_score': float(scores[position[0]]), 'reference_rank': rank,
            'reference_top10': float(rank <= 10)}, order, ref


def _mean(values):
    finite = [v for v in values if v is not None]
    if finite and len(finite) != len(values):
        raise ValueError('A query metric cannot change validity across deletion strategies')
    return float(np.mean(finite)) if finite else None


def strategy_rows(common, baseline, deletions, contributions, areas):
    """Random is the exact expectation of metrics, never of score matrices."""
    weights = np.sqrt(np.asarray(areas, dtype=np.float64))
    weights /= weights.sum()
    pair = area_matched_pair(areas)
    subsets = {'main': tuple(range(len(areas)))}
    if pair is not None:
        subsets['area_matched'] = pair
    rows = []
    for subset, indices in subsets.items():
        high, low = high_low(contributions, indices)
        for strategy, selected in (('baseline', ()), ('high', (high,)),
                                   ('low', (low,)), ('random', indices)):
            values = baseline if not selected else {
                key: _mean([deletions[i][key] for i in selected]) for key in METRICS}
            row = {**common, 'subset': subset, 'strategy': strategy,
                   'selected_regions': '|'.join(map(str, selected)),
                   'deleted_weight': float(np.mean(weights[list(selected)])) if selected else 0.,
                   'reference_contribution': float(np.mean(np.asarray(contributions)[list(selected)])) if selected else 0.,
                   **values}
            row.update({'delta_'+key: None if values[key] is None else float(values[key]-baseline[key])
                        for key in METRICS})
            rows.append(row)
    return rows


def _init_worker(cache_folder, feature_path, pool, labels, classes, baseline_root, audit_queries):
    global _WORKER
    from paper_d_rotation.scoring import Provider
    from paper_c_rotation.pipeline import read_scoring_cache, CachedProvider
    provider = Provider(read_scoring_cache(cache_folder, feature_path), pool)
    baseline = CachedProvider(baseline_root, check=False)
    _WORKER = (provider, np.asarray(labels), np.asarray(classes), baseline, set(audit_queries))


def _query(query):
    from paper_d_rotation.scoring import score_components, batched_score
    from wbm_retrieval.exact_metrics import tie_keys, order_scores
    provider, labels, classes, cached, audit_queries = _WORKER
    query = int(query)
    ids = np.asarray(provider.pool[query], dtype=np.int64)
    qt = provider.tokens(query)
    areas = provider.area[qt]
    parts = provider.parts(query)
    pair = score_components(parts)
    rows = provider.padded[[provider.positions[int(c)] for c in ids]]
    baseline_scores = batched_score(pair, rows, areas)
    expected = cached.scores(query, ids)
    if not np.array_equal(baseline_scores, expected):
        raise ValueError(f'Baseline fails byte-exact C-primary replay: query={query}, '
                         f'max_error={np.max(np.abs(baseline_scores-expected))}')
    hashes = tie_keys(query, ids, 2026)
    eligible = len(qt) >= 2
    area_pair = area_matched_pair(areas)
    masks = {'test': np.ones(len(ids), dtype=bool),
             'cross_test': np.any(labels[ids] != labels[query], axis=1)}
    baseline_rows, deletion_rows, strategies = [], [], []
    stages = {}
    for stage, mask in masks.items():
        sid, ss = ids[mask], baseline_scores[mask]
        metrics, order, reference = metric_values(sid, ss, query, labels,
                                                   hashes=hashes[mask], cross=stage == 'cross_test')
        reference_tokens = provider.tokens(reference)
        scalar_score, contributions = scalar_match(pair[:, reference_tokens], areas)
        if abs(scalar_score - metrics['reference_score']) > 1e-12:
            raise ValueError('Reference explanation does not reproduce its baseline score')
        if not np.array_equal(order, order_scores(sid, ss, query, 2026)):
            raise ValueError('Diagnostic ranking differs from frozen tie policy')
        common = {'stage': stage, 'query_id': query, 'class_id': int(classes[query]),
                  'label_count': int(labels[query].sum()), 'num_regions': len(qt),
                  'candidate_count': len(sid), 'reference_id': reference, 'eligible': int(eligible),
                  'area_pair': '' if area_pair is None else '|'.join(map(str, area_pair))}
        baseline_rows.append({**common, **metrics})
        stages[stage] = (common, metrics, contributions, reference_tokens, [])
    independent_checks, max_error = 0, 0.
    audit_positions = np.argsort(ids)[:20] if query in audit_queries else []
    if query in audit_queries:
        from experiments.proposed.f_segmentation_geometry.coverage_metrics import query_metrics
        for stage, mask in masks.items():
            common, metrics, _, _, _ = stages[stage]
            sid, ss = ids[mask], baseline_scores[mask]
            order = order_scores(sid, ss, query, 2026)
            check = query_metrics(labels[query], labels[sid], labels[sid[order[:10]]], [10])
            actual = metrics['Coverage-nDCG@10']
            expected_ndcg = check['Coverage-nDCG@10']
            if (actual is None) != (expected_ndcg is None) or (actual is not None and abs(actual-expected_ndcg)>1e-12):
                raise ValueError('Independent linear-coverage nDCG audit failed')
        for pos in audit_positions:
            scalar, _ = scalar_match(pair[:, provider.tokens(int(ids[pos]))], areas)
            error = abs(scalar-baseline_scores[pos])
            max_error = max(max_error, error)
            independent_checks += 1
            if error > 1e-12:
                raise ValueError('Independent unmasked matching audit failed')
    if eligible:
        for deleted in range(len(qt)):
            masked = pair.copy()
            masked[deleted, :] = 0.
            scores = batched_score(masked, rows, areas)
            for pos in audit_positions:
                scalar, _ = scalar_match(masked[:, provider.tokens(int(ids[pos]))], areas)
                error = abs(scalar-scores[pos])
                max_error = max(max_error, error)
                independent_checks += 1
                if error > 1e-12:
                    raise ValueError('Independent masked matching audit failed')
            for stage, mask in masks.items():
                common, original, contribution, reference_tokens, result = stages[stage]
                metrics, _, _ = metric_values(ids[mask], scores[mask], query, labels,
                    reference_id=common['reference_id'], hashes=hashes[mask], cross=stage == 'cross_test')
                scalar, _ = scalar_match(masked[:, reference_tokens], areas)
                if abs(scalar-metrics['reference_score']) > 1e-12:
                    raise ValueError('Deleted reference matching audit failed')
                result.append(metrics)
                row = {**common, 'deleted_region': deleted, 'deleted_area': float(areas[deleted]),
                       'deleted_weight': float(np.sqrt(areas[deleted])/np.sqrt(areas).sum()),
                       'reference_contribution': float(contribution[deleted]), **metrics}
                row.update({'delta_'+k: None if metrics[k] is None else float(metrics[k]-original[k])
                            for k in METRICS})
                deletion_rows.append(row)
        for common, original, contribution, _, results in stages.values():
            strategies.extend(strategy_rows(common, original, results, contribution, areas))
    return baseline_rows, deletion_rows, strategies, independent_checks, max_error


def _write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def _shard(task):
    shard, queries, folder = task
    folder = Path(folder)
    handles = [(folder/f'{name}_{shard:02d}.csv').open('x', newline='', encoding='utf-8')
               for name in ('baseline', 'deletions', 'strategies')]
    checks, max_error = 0, 0.
    try:
        writers = [csv.DictWriter(h, fieldnames=f) for h, f in zip(handles,
                    (BASELINE_FIELDS, DELETION_FIELDS, STRATEGY_FIELDS))]
        for writer in writers:
            writer.writeheader()
        for number, query in enumerate(queries, 1):
            baseline, deleted, strategies, count, error = _query(query)
            for writer, rows in zip(writers, (baseline, deleted, strategies)):
                writer.writerows(rows)
            checks += count
            max_error = max(max_error, error)
            if number % 100 == 0 or number == len(queries):
                print(f'D4 shard {shard}: {number}/{len(queries)} queries', flush=True)
        return {'shard': shard, 'queries': len(queries), 'independent_candidate_checks': checks,
                'max_scalar_match_error': max_error}
    finally:
        for handle in handles:
            handle.close()


def _read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def summarize(folder):
    """Read the saved per-query strategies, not in-memory experiment state."""
    folder = Path(folder)
    rows = _read_csv(folder/'strategies.csv')
    baseline = _read_csv(folder/'baseline.csv')
    output = []
    for stage in STAGES:
        for subset in ('main', 'area_matched'):
            subset_rows = [r for r in rows if r['stage'] == stage and r['subset'] == subset]
            by_strategy = {s: [r for r in subset_rows if r['strategy'] == s] for s in STRATEGIES}
            original = {int(r['query_id']): r for r in by_strategy['baseline']}
            for strategy, selected in by_strategy.items():
                if len(selected) != len(original) or {int(r['query_id']) for r in selected} != set(original):
                    raise ValueError('All strategies must use the exact same diagnostic queries')
                for metric in (*METRICS, 'deleted_weight', 'reference_contribution'):
                    values, originals = [], []
                    for row in selected:
                        base = original[int(row['query_id'])]
                        if bool(row[metric]) != bool(base[metric]):
                            raise ValueError('Metric validity differs from its paired baseline')
                        if row[metric] != '':
                            values.append(float(row[metric]))
                            originals.append(float(base[metric]))
                    mean = float(np.mean(values)) if values else None
                    base_mean = float(np.mean(originals)) if originals else None
                    output.append({'stage': stage, 'subset': subset, 'strategy': strategy,
                        'num_queries': len(selected), 'metric': metric, 'valid_queries': len(values),
                        'mean': mean, 'baseline_mean': base_mean,
                        'delta': None if mean is None else float(np.mean(np.asarray(values)-originals))})
    with (folder/'summary.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(output)
    counts = {stage: {str(n): sum(int(r['num_regions']) == n for r in baseline if r['stage'] == stage)
                     for n in range(5)} for stage in STAGES}
    result = {'status': 'completed', 'kind': 'posthoc_query_evidence_deletion_diagnostic',
        'baseline_scope': 'all frozen test queries; strategy comparisons restricted to >=2 regions',
        'reference': 'unperturbed rank1 separately for ordinary and cross-combination',
        'random_policy': 'exact uniform expectation of per-region-deletion metrics',
        'area_control': {'max_sqrt_area_weight_ratio': 1.2,
            'selection': 'closest log weight ratio, then original region indices; before contributions',
            'scope': 'approximate area match; does not remove all area confounding'},
        'delta_convention': 'perturbed minus paired unperturbed; negative retrieval delta is worse',
        'undefined_ndcg': 'retain missing; exclude only from nDCG aggregate and report valid count',
        'region_counts': counts, 'rows': output}
    _write_json(folder/'summary.json', result)
    return result


def run(output, data, workers=8, baseline_root=None):
    """Run once into a new d4 directory; historical data and scores stay read-only."""
    from paper_c_rotation.pipeline import CachedProvider
    from wbm_retrieval.common import file_hash
    output = Path(output)
    folder = output/'d4'
    if folder.exists():
        raise FileExistsError('D4 output exists; preserve it and use an unchanged controlled resume or new run')
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError('Use 1 through 8 spawned workers')
    pool = data.get('pool')
    if pool is None:
        pool = data['protocol'].retrieval.test
    if baseline_root is None:
        raise ValueError('An explicit sealed rotation primary score cache is required')
    baseline_root = Path(baseline_root)
    reference = CachedProvider(baseline_root)
    if (not np.array_equal(reference.query_ids, list(pool))
            or not np.array_equal(reference.gallery_ids, pool.gallery)
            or reference.protocol_signature != pool.protocol_signature):
        raise ValueError('C-primary baseline identities differ from the frozen diagnostic pool')
    cache = data['cache']
    if not cache.get('feature_path'):
        raise ValueError('Explicit rotation feature path is required by spawned workers')
    ids = np.asarray(cache['map_ids'])
    region_counts = np.diff(cache['offsets'])
    if np.any(region_counts > 4) or not np.array_equal(ids, pool.gallery):
        raise ValueError('The original fixed Top-4 cache and gallery order are required')
    queries = list(map(int, pool))
    bundle = data['bundle']
    classes = np.asarray(bundle.class_ids)
    audit_queries = {}
    for q in queries:
        audit_queries.setdefault(int(classes[q]), q)
    folder.mkdir(parents=True, exist_ok=False)
    shards = folder/'shards'
    shards.mkdir()
    chunks = [list(map(int, x)) for x in np.array_split(queries, min(workers, len(queries)))]
    started = time.perf_counter()
    checks = []
    with ProcessPoolExecutor(max_workers=len(chunks), mp_context=multiprocessing.get_context('spawn'),
            initializer=_init_worker, initargs=(str(cache['folder']), str(cache['feature_path']), pool, bundle.labels, classes,
                                               str(baseline_root), list(audit_queries.values()))) as executor:
        futures = [executor.submit(_shard, (i, chunk, str(shards))) for i, chunk in enumerate(chunks)]
        for future in as_completed(futures):
            checks.append(future.result())
    for name, fields in (('baseline', BASELINE_FIELDS), ('deletions', DELETION_FIELDS),
                          ('strategies', STRATEGY_FIELDS)):
        with (folder/f'{name}.csv').open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            for i in range(len(chunks)):
                with (shards/f'{name}_{i:02d}.csv').open(newline='', encoding='utf-8') as part:
                    reader = csv.DictReader(part)
                    if reader.fieldnames != list(fields):
                        raise ValueError('D4 shard schema differs')
                    writer.writerows(reader)
    baseline_rows = _read_csv(folder/'baseline.csv')
    deletion_rows = _read_csv(folder/'deletions.csv')
    for stage in STAGES:
        selected = [r for r in baseline_rows if r['stage'] == stage]
        if [int(r['query_id']) for r in selected] != queries:
            raise ValueError('Diagnostic baseline misses, duplicates or reorders fixed queries')
        expected_deletions = sum(int(r['num_regions']) for r in selected if int(r['eligible']))
        deleted = [r for r in deletion_rows if r['stage'] == stage]
        if len(deleted) != expected_deletions:
            raise ValueError('Diagnostic deletion enumeration is incomplete')
    summary = summarize(folder)
    verification = {'status': 'verified', 'queries': len(queries),
        'eligible_queries': int(np.sum(region_counts >= 2)),
        'single_region_queries': int(np.sum(region_counts == 1)),
        'zero_region_queries': int(np.sum(region_counts == 0)),
        'area_matched_queries': sum(area_matched_pair(cache['area'][a:b]) is not None
                                   for a, b in zip(cache['offsets'][:-1], cache['offsets'][1:])),
        'baseline_full_scores_byte_exact': True, 'baseline_top1_reference_replayed': True,
        'full_candidate_baseline_pairs': len(queries)*(len(pool.gallery)-1),
        'masked_score_rows': len(deletion_rows),
        'independent_candidate_checks': sum(x['independent_candidate_checks'] for x in checks),
        'independent_reference_checks': 2*len(queries)+len(deletion_rows),
        'max_scalar_match_error': max(x['max_scalar_match_error'] for x in checks),
        'metric_audit_queries': len(audit_queries),
        'baseline_source': str(baseline_root),
        'baseline_completion_sha256': file_hash(baseline_root/'completed.json'),
        'source_cache_completion_sha256': file_hash(Path(cache['folder'])/'completed.json'),
        'rotation_feature_path': str(Path(cache['feature_path']).resolve()),
        'rotation_feature_sha256': file_hash(cache['feature_path']),
        'duration_seconds': time.perf_counter()-started, 'workers': workers,
        'files': {name: file_hash(folder/name) for name in ('baseline.csv', 'deletions.csv',
                                                         'strategies.csv', 'summary.csv', 'summary.json')}}
    _write_json(folder/'verification.json', verification)
    return {'summary': summary, 'verification': verification}
