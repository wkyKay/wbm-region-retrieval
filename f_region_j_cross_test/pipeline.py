"""One fixed coverage test, with a frozen ordinary-ranking replay control."""
import csv
from itertools import groupby
from pathlib import Path
import tempfile
import time

import numpy as np

from experiments.proposed.f_segmentation_geometry import coverage_metrics
from fgw_cross_test.verify import independent_order
from wbm_retrieval.common import file_hash, read_json, write_json
from wbm_retrieval.exact_metrics import order_scores
from .state import (VARIANT, SPEC, SETTINGS, PROTOCOL, CANDIDATE_RULE, identity,
                    evaluation_folder, variant_signature, record, verified)

REPORT_FILES = ('run_identity.json', 'evaluation_manifest.json', 'rankings.csv',
                'coverage_metrics.json', 'coverage_per_query.csv', 'coverage_per_class.csv')
RANKING_FIELDS = ('query_id', 'rank', 'candidate_id', 'similarity_score')


def _index(output, context_signature, folder, report):
    record(Path(output) / 'evaluation_index_test.json', {'context_signature': context_signature, 'split': 'test',
           'variant': VARIANT, 'spec': SPEC, 'coverage_metrics': report['metrics'],
           'metric_query_counts': report['metric_query_counts'], 'completed_sha256': file_hash(folder / 'completed.json')})


def historical_replay(output, provider, data, context_signature):
    old = Path(data['source_metadata']['source_winner_folder']) / 'rankings.csv'
    path = Path(output) / 'historical_replay.json'
    if path.exists():
        previous = read_json(path)
        expected = {'status': 'verified', 'context_signature': context_signature,
                    'num_queries': len(provider.query_ids), 'ranking_rows': len(provider.query_ids)*500,
                    'top500_identical': True, 'max_score_error': 0., 'source': str(old),
                    'source_sha256': file_hash(old), 'score_cache_completed_sha256': file_hash(provider.marker)}
        if any(previous.get(k) != v for k, v in expected.items()):
            raise ValueError('Existing historical replay binding differs')
        return previous
    count = 0
    maximum = 0.
    started = time.perf_counter()
    with old.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(RANKING_FIELDS):
            raise ValueError('Unexpected original regional ranking columns')
        for q, iterator in groupby(reader, key=lambda r: int(r['query_id'])):
            if count >= len(provider.query_ids) or q != int(provider.query_ids[count]):
                raise ValueError('Original regional query ordering differs')
            rows = list(iterator)
            ids, scores = provider.full_row(q)
            order = independent_order(q, ids, scores, SETTINGS['tie_seed'])[:500]
            saved_ids = np.asarray([int(r['candidate_id']) for r in rows])
            saved = np.asarray([float(r['similarity_score']) for r in rows])
            if (len(rows) != 500 or [int(r['rank']) for r in rows] != list(range(1, 501))
                    or not np.array_equal(saved_ids, ids[order]) or not np.array_equal(saved, scores[order])):
                raise ValueError('Original full-pool regional Top-500/score failed exact replay')
            maximum = max(maximum, float(np.max(np.abs(saved-scores[order]))))
            count += 1
            if count % 1000 == 0:
                print(f'Original regional test replay: {count}/{len(provider.query_ids)}', flush=True)
    if count != len(provider.query_ids):
        raise ValueError('Original regional test is incomplete')
    result = {'status': 'verified', 'context_signature': context_signature, 'num_queries': count,
              'ranking_rows': count*500, 'top500_identical': True, 'max_score_error': maximum,
              'source': str(old), 'source_sha256': file_hash(old),
              'score_cache_completed_sha256': file_hash(provider.marker),
              'wall_seconds': time.perf_counter()-started,
              'scope': 'Recompute original full-pool scores, then reproduce ordinary Top-500; never filter old Top-500'}
    record(Path(output) / 'historical_replay.json', result)
    return result


def evaluate(output, data, context_signature, provider, *, formal=True):
    from .verify import verify_evaluation
    output = Path(output)
    pool, bundle = data['pool'], data['bundle']
    ks, seed = SETTINGS['ks'], SETTINGS['tie_seed']
    signature = variant_signature(context_signature)
    folder = evaluation_folder(output)
    if verified(folder, signature):
        report = read_json(folder / 'coverage_metrics.json')
        _index(output, context_signature, folder, report)
        return report
    if folder.exists():
        raise FileExistsError('Incomplete final evaluation; inspect without overwriting')
    folder.parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix='.'+VARIANT+'_attempt_', dir=folder.parent))
    ident = identity(output, signature)
    manifest = {**ident, 'signature': signature, 'context_signature': context_signature,
        'variant': VARIANT, 'spec': SPEC, 'settings': SETTINGS, 'ks': ks, 'tie_seed': seed,
        'split': 'test', 'stage': 'test', 'protocol': PROTOCOL, 'candidate_rule': CANDIDATE_RULE,
        'derived_protocol': pool.metadata, 'retrieval_protocol_signature': pool.protocol_signature,
        'base_protocol_signature': data['base_pool'].protocol_signature,
        'provider_source': {'signature': provider.signature, 'completed_sha256': file_hash(provider.marker)},
        'source_binding_sha256': data['source_binding_signature'], 'num_queries': len(pool),
        'gallery_count': len(pool.gallery), 'formal_protocol': formal,
        'f_version': 'regional_J_area_greedy', 'matching': 'original_F_greedy_sqrt_area',
        'structure_strength': .5, 'origin_valid_winner': True, 'valid_selection_rank': 1,
        'training_required': False, 'posthoc_research': True, 'test_parameter_search': False,
        'selection_rule': 'user_requested_cross_evaluation_of_frozen_valid_winner',
        'primary_variant': None, 'normalization_scope': 'none', 'ranking_scope': 'top_max_k'}
    record(attempt / 'run_identity.json', ident)
    record(attempt / 'evaluation_manifest.json', manifest)
    started = time.perf_counter()
    write_json(output / 'status.json', {'status': 'running', 'phase': 'cross_ranking'})
    with (attempt / 'rankings.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(RANKING_FIELDS)
        for number, q in enumerate(pool, 1):
            ids = np.asarray(pool[q])
            values = provider.scores(q, ids)
            order = order_scores(ids, values, int(q), seed)[:max(ks)]
            writer.writerows((int(q), rank, int(ids[i]), float(values[i]))
                             for rank, i in enumerate(order, 1))
            if number % 1000 == 0 or number == len(pool):
                print(f'Regional cross ranking: {number}/{len(pool)}', flush=True)
    ranking_seconds = time.perf_counter()-started
    write_json(output / 'status.json', {'status': 'running', 'phase': 'coverage_metrics'})
    started = time.perf_counter()
    coverage_metrics.evaluate_rankings_file(bundle, pool, attempt / 'rankings.csv', ks, attempt,
        split_name='test', metadata={**manifest, 'candidates_scored': 'full_filtered_pool', 'smoke': False})
    metrics_seconds = time.perf_counter()-started
    print('Independent full-candidate ranking and coverage verification', flush=True)
    write_json(output / 'status.json', {'status': 'running', 'phase': 'independent_verification'})
    started = time.perf_counter()
    checked = verify_evaluation(attempt, pool, bundle, ks, signature, provider,
                                data['source_binding'], tie_seed=seed)
    record(attempt / 'verification.json', checked)
    record(attempt / 'timing.json', {'ranking_wall_seconds': ranking_seconds,
           'metrics_wall_seconds': metrics_seconds, 'verification_wall_seconds': time.perf_counter()-started,
           'score_source': 'new complete original-formula cache; no repeated regional score calculation'})
    record(attempt / 'completed.json', {'status': 'completed', 'signature': signature,
           'files': {name: file_hash(attempt / name) for name in (*REPORT_FILES, 'verification.json', 'timing.json')}})
    if folder.exists():
        raise FileExistsError('Concurrent evaluation publication')
    attempt.rename(folder)
    report = read_json(folder / 'coverage_metrics.json')
    _index(output, context_signature, folder, report)
    return report


def summarize(output, data, report, context_signature):
    output = Path(output)
    references = {}
    rows = []
    for name, raw_folder in data['source_metadata']['reference_folders'].items():
        folder = Path(raw_folder)
        references[name] = {'folder': str(folder), 'completed_sha256': file_hash(folder / 'completed.json'),
                            'coverage_metrics_sha256': file_hash(folder / 'coverage_metrics.json')}
        previous = read_json(folder / 'coverage_metrics.json')
        if (previous['metadata']['derived_protocol'] != data['pool'].metadata
                or previous['metric_query_counts'] != report['metric_query_counts']
                or set(previous['metrics']) != set(report['metrics'])):
            raise ValueError('Comparison requires identical candidate protocol and metric counts')
        manifest = read_json(folder / 'evaluation_manifest.json')
        rows.append({'variant': name, 'formula': manifest['spec']['formula'], 'is_new_evaluation': False,
                     'source': str(folder), **previous['metrics']})
    rows.append({'variant': VARIANT, 'formula': 'regional 0.75*(0.4*C+0.6*T)+0.25*A; greedy sqrt(area)',
                 'is_new_evaluation': True, 'source': str(evaluation_folder(output)), **report['metrics']})
    metrics = list(report['metrics'])
    rows.sort(key=lambda row: (-row['Coverage-nDCG@10'], row['variant']))
    with (output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['variant', 'formula', 'is_new_evaluation', 'source', *metrics])
        writer.writeheader()
        writer.writerows(rows)
    with (output / 'metric_orderings.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['metric', 'rank', 'variant', 'value', 'direction'])
        for metric in metrics:
            for row in sorted(rows, key=lambda r: (-r[metric], r['variant'])):
                writer.writerow([metric, 1+sum(r[metric] > row[metric] for r in rows),
                                 row['variant'], row[metric], 'higher_is_better'])
    record(output / 'comparison_references.json', {'status': 'verified', 'context_signature': context_signature,
                                                  'references': references})
    record(output / 'comparison_design.json', {'primary_display_metric': 'Coverage-nDCG@10',
           'ranking_is_descriptive_only': True, 'test_selects_parameters': False,
           'primary_variant': None, 'new_evaluations': 1, 'existing_reference_evaluations': 6})
    for row in rows:
        print(row['variant']+': '+', '.join(f'{m}={100*row[m]:.4f}%' for m in
            ['Coverage-nDCG@10', 'MeanCoverage@10', 'FullCoveragePrecision@10', 'FullCoverage_mAP@100']), flush=True)
