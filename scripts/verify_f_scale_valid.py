#!/usr/bin/env python3
"""Independently replay the three frozen valid scale-ablation rankings.

No production metric evaluator is imported. The mathematical helpers come from
the previous independent test verifier. Only saved Top-500 rows are available;
AP/Recall denominators and nDCG ideals use all 3,598 candidates per query.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from verify_f_radial_test import (KS, POPCOUNT, file_hash, query_metrics,
                                  read_ids, self_test as metric_self_test, tie_key)

METHODS = ('radial', 'radial_area', 'radial_axes')
VERSIONS = {'exact_v1': ('metrics.json', ('Hit', 'Precision', 'Recall', 'mAP')),
            'coverage_v1': ('coverage_metrics.json', ('Coverage-nDCG', 'MeanCoverage',
                'FullCoverageHit', 'FullCoveragePrecision', 'FullCoverageRecall', 'FullCoverage_mAP'))}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_rankings(path, queries, gallery, classes, counts):
    """Check CSV identity/order and compute every query metric independently."""
    gallery_set = set(map(int, gallery))
    rows, ties = [], 0
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not {'query_id', 'rank', 'candidate_id', 'similarity_score'} <= set(reader.fieldnames or []):
            raise ValueError(f'Unexpected ranking CSV schema: {path}')
        for query in queries:
            q = int(query)
            candidate_ids, previous_score, previous_id = [], None, None
            for rank in range(1, max(KS)+1):
                record = next(reader, None)
                if record is None or int(record['query_id']) != q or int(record['rank']) != rank:
                    raise ValueError(f'Noncontiguous/missing ranking at {path}: query={q}, rank={rank}')
                candidate, score = int(record['candidate_id']), float(record['similarity_score'])
                if candidate == q or candidate not in gallery_set or not np.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f'Invalid candidate/score at {path}: query={q}, rank={rank}')
                if previous_score is not None:
                    if score > previous_score:
                        raise ValueError(f'Increasing scores at {path}: query={q}, rank={rank}')
                    if score == previous_score:
                        ties += 1
                        if tie_key(q, previous_id) > tie_key(q, candidate):
                            raise ValueError(f'Invalid SHA tie order at {path}: query={q}, rank={rank}')
                candidate_ids.append(candidate)
                previous_score, previous_id = score, candidate
            if len(set(candidate_ids)) != max(KS):
                raise ValueError(f'Duplicate candidate at {path}: query={q}')
            pool_counts = counts.copy()
            pool_counts[classes[q]] -= 1
            rows.append(query_metrics(int(classes[q]), classes[candidate_ids], pool_counts))
        if next(reader, None) is not None:
            raise ValueError(f'Unexpected extra ranking rows: {path}')
    return rows, ties


def compare_reports(directory, rows, query_classes):
    columns = {name: np.array([row[name] for row in rows]) for name in rows[0]}
    label_counts = POPCOUNT[query_classes]
    masks = {'all': np.ones(len(rows), dtype=bool), 'single_label': label_counts == 1,
             'mixed_label': label_counts > 1}
    masks.update({f'count_{i}': label_counts == i for i in range(1, 5)})
    checks = {}
    for version, (filename, prefixes) in VERSIONS.items():
        report = read_json(directory/filename)
        if report['metric_version'] != version or report['split'] != 'valid' or report['test_evaluated'] is not False:
            raise ValueError(f'Wrong metric version/split: {directory/filename}')
        if report['num_queries'] != len(rows) or report['ks'] != list(KS) or report['num_classes'] != 37:
            raise ValueError(f'Wrong report identities: {directory/filename}')
        expected_names = {f'{prefix}@{k}' for prefix in prefixes for k in KS}
        if set(report['metrics']) != expected_names or set(report['class_macro_metrics']) != expected_names:
            raise ValueError(f'Incomplete or unexpected metric set: {directory/filename}')
        groups = report.get('groups', {})
        if not {'single_label', 'mixed_label', 'by_label_count'} <= set(groups):
            raise ValueError(f'Missing query groups: {directory/filename}')
        if set(groups['by_label_count']) != {'1', '2', '3', '4'}:
            raise ValueError(f'Wrong defect-count groups: {directory/filename}')
        grouped = {name: groups[name] for name in ('single_label', 'mixed_label')}
        grouped.update({f'count_{i}': groups['by_label_count'][str(i)] for i in range(1, 5)})
        for group, summary in grouped.items():
            if (summary['num_queries'] != int(masks[group].sum()) or
                    summary['num_classes'] != len(np.unique(query_classes[masks[group]])) or
                    set(summary['metrics']) != expected_names):
                raise ValueError(f'Wrong counts or incomplete metrics: {directory/filename}/{group}')
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

        for name in sorted(expected_names):
            vector = columns[name]

            def mean(mask):
                finite = vector[mask]
                finite = finite[np.isfinite(finite)]
                return float(finite.mean()) if len(finite) else np.nan

            check(mean(masks['all']), report['metrics'][name], filename+'/all/'+name)
            per_class = [mean(query_classes == c) for c in np.unique(query_classes)]
            check(float(np.nanmean(per_class)), report['class_macro_metrics'][name], filename+'/macro/'+name)
            for group, summary in grouped.items():
                check(mean(masks[group]), summary['metrics'][name], filename+'/'+group+'/'+name)
        checks[version] = {'metric_count': len(expected_names), 'aggregate_values_checked': comparisons,
                          'max_abs_error': maximum, 'all_macro_single_mixed_label_counts_verified': True}
    return checks


def verify(args):
    started = time.monotonic()
    run, protocol, output = args.run_dir.resolve(), args.protocol_dir.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError('Independent verification output already exists')
    metadata = read_json(protocol/'metadata.json')
    if metadata['candidate_rule'] != 'all_gallery_except_query':
        raise ValueError('Unsupported candidate protocol')
    for name in ('valid_queries.csv', 'valid_gallery.csv'):
        if file_hash(protocol/name) != metadata['files'][name]:
            raise ValueError(f'Frozen protocol hash mismatch: {name}')
    if file_hash(args.data_file) != metadata['source_sha256']:
        raise ValueError('Original dataset hash mismatch')
    queries, gallery = read_ids(protocol/'valid_queries.csv'), read_ids(protocol/'valid_gallery.csv')
    if len(queries) != 500 or len(gallery) != 3599 or not set(queries).issubset(gallery):
        raise ValueError('Expected frozen valid 500-query / 3599-gallery protocol')
    with np.load(args.data_file, allow_pickle=False) as archive:
        labels = archive['arr_1']
    if labels.ndim != 2 or labels.shape[1] != 8 or not np.isin(labels, [0, 1]).all():
        raise ValueError('Expected original-ID indexed eight-bit labels')
    if gallery.max() >= len(labels):
        raise ValueError('IDs outside original NPZ')
    classes = labels.astype(np.int64) @ (1 << np.arange(8))
    query_classes = classes[queries]
    if np.any(classes[gallery] == 0) or len(np.unique(query_classes)) != 37 or len(np.unique(classes[gallery])) != 37:
        raise ValueError('Expected 37 nonempty valid combinations')
    counts = np.bincount(classes[gallery], minlength=256)
    methods = {}
    for method in METHODS:
        directory = run/method
        rows, ties = read_rankings(directory/'rankings.csv', queries, gallery, classes, counts)
        checks = compare_reports(directory, rows, query_classes)
        methods[method] = {'status': 'passed', 'results': checks,
            'saved_ranking_rows': len(rows)*max(KS), 'adjacent_equal_score_ties_checked': ties,
            'ranking_identity_order_and_self_exclusion_verified': True,
            'source_hashes': {name: file_hash(directory/name) for name in
                             ('rankings.csv', 'metrics.json', 'coverage_metrics.json')}}
        print(f'{method}: 500 queries, 70 metrics and all groups verified', flush=True)
    audit = {'status': 'passed', 'split': 'valid', 'test_evaluated': False,
        'verification': 'Independent CSV replay; no production metric evaluator imports',
        'num_queries': 500, 'num_gallery': 3599, 'candidates_per_query': 3598,
        'saved_top_k': max(KS), 'methods': methods,
        'scope': 'Saved Top-500 identities, score order, SHA ties and all aggregate metrics; omitted candidate scores are not reconstructed',
        'seconds': time.monotonic()-started,
        'source_hashes': {'data_file': metadata['source_sha256'],
                         'verifier_script': file_hash(__file__),
                         'independent_helper': file_hash(Path(__file__).with_name('verify_f_radial_test.py')),
                         **{name: file_hash(protocol/name) for name in
                            ('metadata.json', 'valid_queries.csv', 'valid_gallery.csv')}}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(audit, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': audit['status'], 'methods': list(methods),
                      'seconds': audit['seconds'], 'output': str(output)}), flush=True)


def self_test():
    metric_self_test()
    # Full-gallery R=10 verifies that a Top-1 exact hit does not become AP=1.
    counts = np.zeros(256, dtype=int)
    counts[3], counts[7] = 10, 20
    result = query_metrics(3, [3], counts, (1,))
    assert result['mAP@1'] == .1
    assert result['FullCoverage_mAP@1'] == 1/30
    assert result['Coverage-nDCG@1'] == 1
    assert len(query_metrics(3, [3], counts)) == 70
    print('Valid scale verifier self-test passed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'data-file', 'protocol-dir', 'output'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if any(getattr(args, name) is None for name in ('run_dir', 'data_file', 'protocol_dir', 'output')):
        parser.error('--run-dir, --data-file, --protocol-dir and --output are required')
    verify(args)


if __name__ == '__main__':
    main()
