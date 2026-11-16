#!/usr/bin/env python3
"""Independent CSV replay of exact_v1 and coverage_v1; no evaluator imports.

Only Top-500 rows are saved; all denominators and ideal rankings use the full
frozen gallery. This checks saved ranking identity/order and reported metrics,
not whether unrecorded candidates were scored by the retrieval implementation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

KS = (1, 5, 10, 20, 50, 100, 500)
POPCOUNT = np.array([i.bit_count() for i in range(256)], dtype=int)


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def read_ids(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        values = np.array([int(row['sample_id']) for row in csv.DictReader(stream)], dtype=int)
    if len(np.unique(values)) != len(values) or np.any(values < 0):
        raise ValueError(f'Invalid original IDs: {path}')
    return values


def tie_key(query, candidate, seed=2026):
    raw = f'exact_v1:tie:{seed}:{query}:{candidate}'.encode('ascii')
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big'), candidate


def query_metrics(query_class, ranked_classes, pool_counts, ks=KS):
    """pool_counts counts complete nonempty label bitsets, already excluding self."""
    pool_counts = np.asarray(pool_counts, dtype=int)
    ranked_classes = np.asarray(ranked_classes, dtype=int)
    if (query_class <= 0 or query_class >= 256 or pool_counts.shape != (256,)
            or np.any(pool_counts < 0) or pool_counts[0] or np.any(ranked_classes <= 0)
            or np.any(ranked_classes >= 256)):
        raise ValueError('Invalid class identities/counts')
    relevant = (ranked_classes == query_class).astype(float)
    r_exact = int(pool_counts[query_class])
    classes = np.arange(256)
    fractions = POPCOUNT[classes & query_class] / POPCOUNT[query_class]
    full_classes = (classes & query_class) == query_class
    r_full = int(pool_counts[full_classes].sum())
    ranked_coverage = fractions[ranked_classes]
    full_flags = full_classes[ranked_classes].astype(float)
    ranks = np.arange(1, len(ranked_classes)+1)
    exact_precision = np.cumsum(relevant) / ranks
    full_precision = np.cumsum(full_flags) / ranks
    # Sorting only the <=256 distinct bitsets then expanding their multiplicities
    # produces the ideal full-gallery graded ranking independently of the engine.
    order = np.argsort(-fractions, kind='stable')
    ideal = np.repeat(fractions[order], pool_counts[order])[:max(ks)]
    metrics = {}
    for k in ks:
        n, full_n = relevant[:k].sum(), full_flags[:k].sum()
        discounts = 1 / np.log2(np.arange(1, k+1)+1)
        observed = ranked_coverage[:k]
        idcg = np.dot(ideal[:k], discounts[:len(ideal[:k])])
        dcg = np.dot(observed, discounts[:len(observed)])
        metrics.update({
            f'Hit@{k}': float(n > 0), f'Precision@{k}': float(n/k),
            f'Recall@{k}': float(n/r_exact) if r_exact else np.nan,
            f'mAP@{k}': float(np.dot(exact_precision[:k], relevant[:k])/r_exact) if r_exact else np.nan,
            f'Coverage-nDCG@{k}': float(dcg/idcg) if idcg else np.nan,
            f'MeanCoverage@{k}': float(observed.sum()/k),
            f'FullCoverageHit@{k}': float(full_n > 0),
            f'FullCoveragePrecision@{k}': float(full_n/k),
            f'FullCoverageRecall@{k}': float(full_n/r_full) if r_full else np.nan,
            f'FullCoverage_mAP@{k}': float(np.dot(full_precision[:k], full_flags[:k])/r_full) if r_full else np.nan,
        })
    return metrics


def self_test():
    # Five candidates: two exact/full matches and three half-coverage candidates.
    counts = np.zeros(256, dtype=int)
    counts[1], counts[2], counts[3] = 2, 1, 2
    result = query_metrics(3, [1, 3, 2], counts, (1, 3, 5))
    assert result['Hit@1'] == 0
    assert result['Precision@3'] == 1/3
    assert result['Recall@3'] == .5
    assert result['mAP@3'] == .25  # R=2, never min(R,K).
    assert result['FullCoverage_mAP@3'] == .25
    assert result['MeanCoverage@3'] == 2/3
    assert result['MeanCoverage@5'] == .4  # Missing returns contribute zero.
    assert result['Precision@5'] == .2
    assert result['FullCoveragePrecision@5'] == .2
    expected = (.5 + 1/np.log2(3) + .25)/(1 + 1/np.log2(3) + .25)
    assert abs(result['Coverage-nDCG@3'] - expected) < 1e-15
    # A strict superset is full coverage but not exact relevance.
    counts = np.zeros(256, dtype=int)
    counts[3], counts[7] = 1, 1
    result = query_metrics(3, [7], counts, (1,))
    assert result['mAP@1'] == 0 and result['FullCoverage_mAP@1'] == .5
    assert result['Coverage-nDCG@1'] == result['MeanCoverage@1'] == 1
    # No relevant candidates: recall/AP excluded, hit/precision remain defined.
    counts = np.zeros(256, dtype=int)
    counts[1] = 3
    result = query_metrics(2, [1], counts, (1,))
    assert np.isnan(result['Recall@1']) and np.isnan(result['FullCoverage_mAP@1'])
    assert result['Hit@1'] == result['FullCoveragePrecision@1'] == 0
    assert np.isnan(result['Coverage-nDCG@1'])
    print('Independent verifier self-test passed', flush=True)


def verify(args):
    started = time.monotonic()
    result_dir, protocol, output = args.result.resolve(), args.protocol_dir.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError('Independent verification output already exists')
    metadata = json.loads((protocol/'metadata.json').read_text())
    if metadata['candidate_rule'] != 'all_gallery_except_query':
        raise ValueError('Unsupported candidate protocol')
    for name in ('test_queries.csv', 'test_gallery.csv'):
        if file_hash(protocol/name) != metadata['files'][name]:
            raise ValueError(f'Frozen protocol hash mismatch: {name}')
    if file_hash(args.data_file) != metadata['source_sha256']:
        raise ValueError('Source dataset hash mismatch')
    queries, gallery = read_ids(protocol/'test_queries.csv'), read_ids(protocol/'test_gallery.csv')
    if len(queries) != 7199 or len(gallery) != 7199 or not np.array_equal(queries, gallery):
        raise ValueError('Expected complete official 7199-by-7198 test protocol')
    with np.load(args.data_file, allow_pickle=False) as archive:
        labels = archive['arr_1']
    if labels.ndim != 2 or labels.shape[1] != 8 or not np.isin(labels, [0, 1]).all():
        raise ValueError('Expected original-ID indexed eight-bit labels')
    if gallery.max() >= len(labels):
        raise ValueError('IDs outside original NPZ')
    classes = labels.astype(np.int64) @ (1 << np.arange(8))
    query_classes = classes[queries]
    if np.any(query_classes == 0) or len(np.unique(query_classes)) != 37:
        raise ValueError('Expected 37 nonempty test combinations')
    counts = np.bincount(classes[gallery], minlength=256)
    gallery_set = set(int(x) for x in gallery)
    rows, ties, saved_rows = [], 0, 0
    ranking_path = result_dir/'rankings.csv'
    with ranking_path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not {'query_id', 'rank', 'candidate_id', 'similarity_score'} <= set(reader.fieldnames or []):
            raise ValueError('Unexpected ranking CSV schema')
        for number, query in enumerate(queries, 1):
            q = int(query)
            candidate_ids, previous_score, previous_id = [], None, None
            for expected_rank in range(1, 501):
                record = next(reader, None)
                if record is None or int(record['query_id']) != q or int(record['rank']) != expected_rank:
                    raise ValueError(f'Noncontiguous or missing ranking at query={q}, rank={expected_rank}')
                candidate, score = int(record['candidate_id']), float(record['similarity_score'])
                if candidate == q or candidate not in gallery_set or not np.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f'Invalid candidate/score for query={q}')
                if previous_score is not None:
                    if score > previous_score:
                        raise ValueError(f'Ranking scores increase at query={q}')
                    if score == previous_score:
                        ties += 1
                        if tie_key(q, previous_id) > tie_key(q, candidate):
                            raise ValueError(f'Invalid SHA tie order for query={q}')
                candidate_ids.append(candidate)
                previous_score, previous_id = score, candidate
            if len(set(candidate_ids)) != 500:
                raise ValueError(f'Duplicate ranking candidate for query={q}')
            pool_counts = counts.copy()
            pool_counts[classes[q]] -= 1
            rows.append(query_metrics(int(classes[q]), classes[candidate_ids], pool_counts))
            saved_rows += len(candidate_ids)
            if number % 1000 == 0:
                print(f'Independent replay {number}/{len(queries)}', flush=True)
        if next(reader, None) is not None:
            raise ValueError('Unexpected extra ranking records')
    names = list(rows[0])
    columns = {name: np.array([row[name] for row in rows]) for name in names}
    label_counts = POPCOUNT[query_classes]
    masks = {'overall': np.ones(len(queries), dtype=bool),
             'single_label': label_counts == 1, 'mixed_label': label_counts > 1}
    masks.update({f'by_label_count/{i}': label_counts == i for i in sorted(set(label_counts))})
    checks = {}
    for filename, version in (('metrics.json', 'exact_v1'), ('coverage_metrics.json', 'coverage_v1')):
        report = json.loads((result_dir/filename).read_text())
        if report['metric_version'] != version or report['split'] != 'test' or not report['test_evaluated']:
            raise ValueError(f'Wrong metric version/split: {filename}')
        if report['num_queries'] != len(queries) or report['ks'] != list(KS) or report['num_classes'] != 37:
            raise ValueError(f'Wrong report identities: {filename}')
        expected_names = ({f'{prefix}@{k}' for prefix in ('Hit','Precision','Recall','mAP') for k in KS}
                          if version == 'exact_v1' else {f'{prefix}@{k}' for prefix in
                          ('Coverage-nDCG','MeanCoverage','FullCoverageHit','FullCoveragePrecision','FullCoverageRecall','FullCoverage_mAP') for k in KS})
        if set(report['metrics']) != expected_names:
            raise ValueError(f'Incomplete or unexpected metric set: {filename}')
        comparisons, maximum = 0, 0.
        def check(actual, expected, description):
            nonlocal comparisons, maximum
            comparisons += 1
            if expected is None:
                if not np.isnan(actual):
                    raise ValueError(f'Expected undefined metric: {description}')
            elif not np.isfinite(actual) or abs(actual-float(expected)) >= 1e-12:
                raise ValueError(f'Metric mismatch {description}: {actual} vs {expected}')
            else:
                maximum = max(maximum, abs(actual-float(expected)))
        for name in expected_names:
            vector = columns[name]
            def mean(mask):
                finite = vector[mask]
                finite = finite[np.isfinite(finite)]
                return float(finite.mean()) if len(finite) else np.nan
            check(mean(masks['overall']), report['metrics'][name], filename+'/overall/'+name)
            class_means = np.array([mean(query_classes == c) for c in np.unique(query_classes)])
            check(float(np.nanmean(class_means)), report['class_macro_metrics'][name], filename+'/macro/'+name)
            for group in ('single_label', 'mixed_label'):
                if group in report.get('groups', {}):
                    check(mean(masks[group]), report['groups'][group]['metrics'][name], filename+'/'+group+'/'+name)
            for count, group_report in report.get('groups', {}).get('by_label_count', {}).items():
                check(mean(masks['by_label_count/'+count]), group_report['metrics'][name], filename+'/count'+count+'/'+name)
        checks[version] = {'metric_count': len(expected_names), 'aggregate_values_checked': comparisons,
                           'max_abs_error': maximum, 'query_mean_class_macro_and_groups_verified': True}
    audit = {'status': 'passed', 'verification': 'Independent ranking CSV replay; no imports from existing metric engines',
             'num_queries': len(queries), 'num_gallery': len(gallery), 'candidates_per_query': len(gallery)-1,
             'saved_top_k': 500, 'saved_ranking_rows': saved_rows, 'adjacent_equal_score_ties_checked': ties,
             'ranking_identity_order_and_self_exclusion_verified': True,
             'scope': 'Saved Top-500 identities, score order, SHA ties and all reported aggregate metrics; does not reconstruct omitted candidate scores',
             'results': checks, 'seconds': time.monotonic()-started,
             'source_hashes': {'data_file': metadata['source_sha256'], 'rankings.csv': file_hash(ranking_path),
                               'metrics.json': file_hash(result_dir/'metrics.json'),
                               'coverage_metrics.json': file_hash(result_dir/'coverage_metrics.json'),
                               'verifier_script': file_hash(__file__)}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(audit, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(audit, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path)
    parser.add_argument('--data-file', type=Path)
    parser.add_argument('--protocol-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if any(getattr(args, name) is None for name in ('result', 'data_file', 'protocol_dir', 'output')):
        parser.error('--result, --data-file, --protocol-dir and --output are required')
    verify(args)


if __name__ == '__main__':
    main()
