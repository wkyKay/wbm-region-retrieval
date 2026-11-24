"""Independent offline acceptance of D1--D4 artifacts before final sealing.

Recomputes saved contribution arithmetic, original-pixel mask counts, strategy
selection, exact random expectations and all diagnostic aggregates. It does not
repeat the full-gallery scoring computation; that has separate baseline replay
and scalar-matching audits recorded by the compute stage.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import re
import struct

import numpy as np

STAGES = ('test', 'cross_test')
STRATEGIES = ('baseline', 'high', 'low', 'random')
METRICS = ('Precision@10', 'Coverage-nDCG@10', 'reference_score',
           'reference_rank', 'reference_top10')
SUMMARY_METRICS = (*METRICS, 'deleted_weight', 'reference_contribution')
REGION_COUNTS = {1: 38, 2: 45, 3: 75, 4: 7041}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def _number(value):
    if value is None or value == '':
        return None
    result = float(value)
    _require(math.isfinite(result), 'Nonfinite diagnostic number')
    return result


def _close(actual, expected, message, tolerance=1e-12):
    actual, expected = _number(actual), _number(expected)
    _require((actual is None) == (expected is None), message + ': validity differs')
    if actual is not None:
        _require(abs(actual - expected) <= tolerance, message + ': numeric mismatch')


def _mean(values):
    values = [_number(v) for v in values]
    present = [v for v in values if v is not None]
    _require(not present or len(present) == len(values), 'Within-query metric validity changed')
    return math.fsum(present) / len(present) if present else None


def _indices(value):
    return tuple(int(x) for x in value.split('|')) if value else ()


def _files(folder, hashes):
    _require(isinstance(hashes, dict) and bool(hashes), 'Missing artifact hash inventory')
    for relative, expected in hashes.items():
        path = Path(relative)
        target = Path(folder) / path
        _require(not path.is_absolute() and '..' not in path.parts
                 and target.is_file() and not target.is_symlink(), 'Unsafe/missing artifact: ' + relative)
        _require(_sha(target) == expected, 'Artifact hash mismatch: ' + relative)


def verify_examples(folder, expected_anchor_count=222):
    """Recheck D1/D2 algebra and masks without calling the exporter."""
    folder = Path(folder)
    manifest = _json(folder / 'examples.json')
    selection = _json(folder / 'selection.json')
    audit = _json(folder / 'mask_audit.json')
    receipt = _json(folder / 'verification.json')
    _require(receipt.get('status') == 'verified', 'D1/D2 export is not verified')
    _files(folder, receipt['files'])
    rows = manifest['examples']
    _require(manifest.get('method') == 'add_a0p4_b0p15'
             and manifest.get('formula') == '0.85*(0.4*C+0.6*T)+0.15*A',
             'Explanation method does not match current rotation primary')
    _require(receipt.get('source_feature_dtype') == 'float64'
             and receipt.get('algebraic_tolerance') == 1e-12,
             'Explanation precision does not match normalized rotation features')
    ids = [r['example_id'] for r in rows]
    _require(len(ids) == len(set(ids)), 'Duplicate example IDs')
    anchors = [r for r in rows if r.get('anchor')]
    _require(len(anchors) == expected_anchor_count == selection['anchor_count'], 'Incorrect anchor count')
    _require(len(rows) == selection['exported_count'] == receipt['exported_pairs'], 'Export count mismatch')
    representatives = set(map(int, selection['representative_queries_by_class'].values()))
    _require(len(representatives) * 6 == expected_anchor_count, 'Representative query count mismatch')
    for stage in STAGES:
        keys = {(int(r['query_id']), int(r['rank'])) for r in anchors if r['stage'] == stage}
        _require(keys == {(q, rank) for q in representatives for rank in (1, 10, 100)},
                 'Anchors do not cover each representative and rank in ' + stage)
    with np.load(folder / 'assets.npz', allow_pickle=False) as archive:
        sample_ids = archive['sample_ids']
        raw_maps, masks = archive['raw_maps'], archive['region_masks']
    _require(sample_ids.ndim == 1 and len(set(map(int, sample_ids))) == len(sample_ids), 'Duplicate asset IDs')
    _require(raw_maps.shape == masks.shape and raw_maps.ndim == 3
             and len(raw_maps) == len(sample_ids), 'Invalid asset shapes')
    _require(masks.dtype.kind in 'iu', 'Masks must contain integer local labels')
    needed = {int(r[k]) for r in rows for k in ('query_id', 'candidate_id')}
    _require(set(map(int, sample_ids)) == needed, 'Assets do not match exported sample identities')
    by_audit = {int(r['sample_id']): r for r in audit['maps']}
    _require(audit['status'] == 'verified' and len(by_audit) == len(audit['maps'])
             and set(by_audit) == needed, 'Mask audit identity mismatch')
    areas = {}
    for sid, raw, mask in zip(sample_ids, raw_maps, masks):
        item = by_audit[int(sid)]
        count = int(item['region_count'])
        _require(0 <= count <= 4 and np.all((mask >= 0) & (mask <= count)), 'Invalid local mask labels')
        _require(np.all(raw[mask > 0] == 2), 'Mask includes non-defect original pixels')
        actual = np.bincount(mask.ravel(), minlength=count + 1)[1:]
        _require(actual.tolist() == item['region_areas'] and np.all(actual > 0), 'Mask area audit mismatch')
        _require(int((raw == 2).sum()) == item['original_defect_pixels']
                 and int((mask > 0).sum()) == item['retained_defect_pixels'], 'Mask retention counts mismatch')
        areas[int(sid)] = actual.tolist()
    maximum_actual, maximum_algebraic, match_count = 0., 0., 0
    for row in rows:
        q, c = int(row['query_id']), int(row['candidate_id'])
        _require(q != c and row['stage'] in STAGES, 'Invalid example candidate/protocol')
        query_areas = areas[q]
        _require(query_areas == row['query_region_areas'], 'Example area differs from original pixels')
        weights = [math.sqrt(x) for x in query_areas]
        denominator = math.fsum(weights)
        _close(row['query_weight_denominator'], denominator, 'Query denominator')
        used_q, used_c, totals, algebra = set(), set(), [], []
        for match in row['matches']:
            qi, ci = int(match['query_local_index']), int(match['candidate_local_index'])
            _require(0 <= qi < len(query_areas) and 0 <= ci < len(areas[c]), 'Match index outside mask')
            _require(qi not in used_q and ci not in used_c, 'One-to-one correspondence violated')
            used_q.add(qi); used_c.add(ci)
            w = weights[qi] / denominator
            _close(match['query_normalized_weight'], w, 'Normalized query weight')
            _require(match['query_region_area'] == query_areas[qi], 'Matched query area mismatch')
            terms = []
            for channel, coefficient in (('C', .34), ('T', .51), ('A', .15)):
                value = float(match[channel])
                _require(0 <= value <= 1, 'Out-of-range similarity component')
                term = w * coefficient * value
                _close(match['contribution_' + channel], term, 'Algebraic channel contribution')
                terms.append(term)
            _close(match['contribution'], w * float(match['raw_score']), 'Actual matched contribution')
            _close(match['rounding_residual'], float(match['contribution']) - math.fsum(terms), 'Rounding residual')
            totals.append(float(match['contribution'])); algebra.extend(terms)
        match_count += len(totals)
        actual, symbolic = math.fsum(totals), math.fsum(algebra)
        maximum_actual = max(maximum_actual, abs(actual - row['ranking_score']))
        maximum_algebraic = max(maximum_algebraic, abs(symbolic - row['ranking_score']))
        for key in ('score', 'ranking_score', 'actual_contribution_sum'):
            _close(row[key], actual, 'Score reconstruction ' + key)
        _close(row['algebraic_component_sum'], symbolic, 'Algebraic subtotal')
        _close(symbolic, row['ranking_score'], 'Algebraic score reconstruction', 1e-12)
        qlabel, clabel = row['query_labels'], row['candidate_labels']
        _require(len(qlabel) == len(clabel) == 8 and all(v in (0, 1) for v in qlabel + clabel)
                 and sum(qlabel) > 0 and sum(clabel) > 0, 'Invalid example labels')
        coverage = sum(a * b for a, b in zip(qlabel, clabel)) / sum(qlabel)
        relation = 'exact' if qlabel == clabel else 'full_extra' if coverage == 1 else 'partial' if coverage > 0 else 'none'
        _require(row['relation'] == relation, 'Example relevance category mismatch')
        _close(row['coverage'], coverage, 'Example coverage')
        _require(row['stage'] != 'cross_test' or relation != 'exact', 'Cross example contains exact combination')
    _require(receipt['source_rankings_hash_checked'] is True
             and receipt['independent_scalar_matching_checked_pairs'] == len(rows)
             and receipt['matched_region_pairs'] == match_count, 'D1 source/matching audit coverage differs')
    _close(receipt['max_score_reconstruction_error'], maximum_actual, 'D1 recorded reconstruction bound')
    _close(receipt['max_algebraic_decomposition_error'], maximum_algebraic, 'D1 recorded algebra bound')
    expected_categories = {'ordinary_exact': ('test', 'exact'), 'cross_full': ('cross_test', 'full_extra'),
                           'cross_partial': ('cross_test', 'partial'), 'cross_none': ('cross_test', 'none')}
    cases = selection['display_cases']
    _require({c['category'] for c in cases} == set(expected_categories) and len(cases) == 4, 'Display categories changed')
    lookup = {r['example_id']: r for r in rows}
    for case in cases:
        stage, kind = expected_categories[case['category']]
        eligible = [r for r in anchors if r['stage'] == stage and r['relation'] == kind
                    and (kind != 'none' or r.get('gallery_has_coverage') is True)]
        if case['status'] == 'selected':
            chosen = lookup[case['example_id']]
            _require(chosen['stage'] == stage and chosen['relation'] == kind, 'Display case category changed')
            _require(kind != 'none' or chosen.get('gallery_has_coverage') is True,
                     'No-overlap failure example has no positive-coverage candidate in its gallery')
            if eligible:
                first = min(eligible, key=lambda r: (r['query_id'], r['rank'], r['candidate_id']))
                _require(case['selection_pool'] == 'anchor_pool' and chosen['example_id'] == first['example_id'],
                         'Display did not use deterministic first anchor')
            else:
                _require(case['selection_pool'] == 'all_test_rank1_fallback' and chosen['rank'] == 1,
                         'Invalid display fallback')
        else:
            _require(case['status'] == 'missing' and not eligible, 'Available anchor category marked missing')
    return {'anchor_pairs': len(anchors), 'exported_pairs': len(rows), 'original_maps_recounted': len(sample_ids),
            'matched_pairs_recomputed': match_count, 'max_score_error': maximum_actual,
            'max_algebraic_error': maximum_algebraic,
            'display_cases_selected': sum(x['status'] == 'selected' for x in cases)}


def _nearest_area_pair(areas):
    candidates = []
    for i in range(len(areas)):
        for j in range(i + 1, len(areas)):
            wi, wj = math.sqrt(areas[i]), math.sqrt(areas[j])
            if max(wi, wj) / min(wi, wj) <= 1.2:
                candidates.append((abs(math.log(wi / wj)), i, j))
    return min(candidates)[1:] if candidates else ()


def verify_d4(folder, expected_region_counts=None):
    """Independently reconstruct every strategy from saved single-region deletions."""
    folder = Path(folder)
    expected_region_counts = REGION_COUNTS if expected_region_counts is None else expected_region_counts
    expected_n = sum(expected_region_counts.values())
    eligible_n = sum(n for regions, n in expected_region_counts.items() if regions >= 2)
    receipt = _json(folder / 'verification.json')
    _require(receipt['status'] == 'verified', 'D4 compute audit missing')
    _files(folder, receipt['files'])
    baseline_rows, deletion_rows, strategy_rows = (_csv(folder / name) for name in
                                                  ('baseline.csv', 'deletions.csv', 'strategies.csv'))
    baseline = {(r['stage'], int(r['query_id'])): r for r in baseline_rows}
    _require(len(baseline) == len(baseline_rows) == 2 * expected_n, 'Baseline query/protocol count mismatch')
    for stage in STAGES:
        rows = [r for r in baseline_rows if r['stage'] == stage]
        _require(dict(Counter(int(r['num_regions']) for r in rows)) == expected_region_counts,
                 'Unexpected original query region counts')
        for row in rows:
            eligible = int(row['num_regions']) >= 2
            _require(int(row['eligible']) == int(eligible), 'Incorrect diagnostic eligibility')
            _require(int(row['reference_rank']) == 1 and float(row['reference_top10']) == 1., 'Baseline is not original rank1')
            count = int(row['candidate_count'])
            _require(0 < count <= expected_n - 1 and (stage != 'test' or count == expected_n - 1), 'Candidate counts changed')
    ordinary_ids = {q for stage, q in baseline if stage == 'test'}
    _require(ordinary_ids == {q for stage, q in baseline if stage == 'cross_test'}, 'Protocols use different query IDs')
    deleted = defaultdict(dict)
    for row in deletion_rows:
        key = row['stage'], int(row['query_id'])
        region = int(row['deleted_region'])
        _require(key in baseline and region not in deleted[key], 'Unknown/duplicate deletion')
        deleted[key][region] = row
    expected_strategies, area_ids = {}, set()
    for key, base in baseline.items():
        n = int(base['num_regions'])
        items = deleted.get(key, {})
        if n < 2:
            _require(not items, 'Single-region query included in strategy comparison')
            continue
        _require(set(items) == set(range(n)), 'Incomplete single-region deletion enumeration')
        areas = [float(items[i]['deleted_area']) for i in range(n)]
        _require(all(a > 0 and a.is_integer() for a in areas), 'Invalid original pixel areas')
        denominator = math.fsum(math.sqrt(a) for a in areas)
        weights = [math.sqrt(a) / denominator for a in areas]
        contributions = [float(items[i]['reference_contribution']) for i in range(n)]
        _close(math.fsum(contributions), base['reference_score'], 'Reference contribution reconstruction')
        pair = _nearest_area_pair(areas)
        _require(_indices(base['area_pair']) == pair, 'Area subset selected after seeing contributions')
        if pair:
            area_ids.add(key)
        for i, row in items.items():
            for name in ('reference_id', 'candidate_count', 'num_regions', 'area_pair', 'eligible'):
                _require(row[name] == base[name], 'Deletion reference/protocol metadata mismatch')
            _close(row['deleted_weight'], weights[i], 'Deleted normalized area weight')
            _require(1 <= float(row['reference_rank']) <= int(base['candidate_count']), 'Reranked reference out of pool')
            _close(row['reference_top10'], float(float(row['reference_rank']) <= 10), 'Reference Top10 flag')
            for metric in METRICS:
                value, original = _number(row[metric]), _number(base[metric])
                _require((value is None) == (original is None), 'Metric validity changes after deletion')
                _close(row['delta_' + metric], None if value is None else value - original, 'Per-deletion metric delta')
                if metric in ('Precision@10', 'Coverage-nDCG@10', 'reference_score') and value is not None:
                    _require(-1e-12 <= value <= 1 + 1e-12, 'Bounded diagnostic metric out of range')
            _require((key[0] == 'cross_test') == (row['Precision@10'] == ''), 'Cross exact metric must be absent')
        subsets = {'main': tuple(range(n))}
        if pair:
            subsets['area_matched'] = pair
        for subset, indices in subsets.items():
            high = min(indices, key=lambda i: (-contributions[i], i))
            low = min(indices, key=lambda i: (contributions[i], i))
            for strategy, selected in (('baseline', ()), ('high', (high,)), ('low', (low,)), ('random', indices)):
                values = {metric: _number(base[metric]) if not selected else _mean([items[i][metric] for i in selected])
                          for metric in METRICS}
                values.update(deleted_weight=0. if not selected else _mean([weights[i] for i in selected]),
                              reference_contribution=0. if not selected else _mean([contributions[i] for i in selected]))
                expected_strategies[(key[0], key[1], subset, strategy)] = (selected, values)
    _require(len(strategy_rows) == len(expected_strategies), 'Strategy row count mismatch')
    observed, grouped = set(), defaultdict(list)
    for row in strategy_rows:
        key = row['stage'], int(row['query_id']), row['subset'], row['strategy']
        _require(key not in observed and key in expected_strategies, 'Unknown/duplicate strategy row')
        observed.add(key)
        for name in ('reference_id', 'candidate_count', 'num_regions', 'area_pair', 'eligible'):
            _require(row[name] == baseline[key[:2]][name], 'Strategy reference/protocol metadata mismatch')
        selected, values = expected_strategies[key]
        _require(_indices(row['selected_regions']) == selected, 'High/low/random region selection changed')
        for metric, expected in values.items():
            _close(row[metric], expected, 'Strategy expectation: ' + metric)
        for metric in METRICS:
            original = _number(baseline[key[:2]][metric])
            expected = values[metric]
            _close(row['delta_' + metric], None if expected is None else expected - original, 'Strategy delta')
        grouped[(key[0], key[2], key[3])].append((key[1], values))
    summaries = _csv(folder / 'summary.csv')
    summary_json = _json(folder / 'summary.json')
    _require(len(summaries) == len(grouped) * len(SUMMARY_METRICS), 'Summary rows missing')
    by_json = {(r['stage'], r['subset'], r['strategy'], r['metric']): r for r in summary_json['rows']}
    seen_summary = set()
    for row in summaries:
        group = row['stage'], row['subset'], row['strategy']
        metric = row['metric']
        key = (*group, metric)
        _require(key not in seen_summary and group in grouped and metric in SUMMARY_METRICS, 'Unknown/duplicate summary')
        seen_summary.add(key)
        pairs = grouped[group]
        values, originals = [], []
        for query, metrics in pairs:
            value = metrics[metric]
            original = expected_strategies[(group[0], query, group[1], 'baseline')][1][metric]
            _require((value is None) == (original is None), 'Summary pairs have different valid sets')
            if value is not None:
                values.append(value); originals.append(original)
        _require(int(row['num_queries']) == len(pairs) and int(row['valid_queries']) == len(values), 'Summary query counts changed')
        expected = {'mean': _mean(values), 'baseline_mean': _mean(originals),
                    'delta': _mean([a-b for a, b in zip(values, originals)])}
        _require(key in by_json, 'JSON summary missing CSV row')
        for field, value in expected.items():
            _close(row[field], value, 'Independently recomputed summary ' + field)
            _close(by_json[key][field], value, 'JSON summary ' + field)
        for field in ('num_queries', 'valid_queries'):
            _require(int(row[field]) == by_json[key][field], 'JSON summary count differs')
    _require(set(by_json) == seen_summary, 'JSON summary has extra/duplicate rows')
    _require(receipt['queries'] == expected_n and receipt['eligible_queries'] == eligible_n
             and receipt['single_region_queries'] == expected_region_counts.get(1, 0)
             and receipt['zero_region_queries'] == expected_region_counts.get(0, 0), 'D4 audit counts differ')
    _require(receipt['masked_score_rows'] == len(deletion_rows)
             and receipt['area_matched_queries'] * 2 == len(area_ids), 'D4 mask/area audit counts differ')
    _require(receipt['baseline_full_scores_byte_exact'] is True and receipt['baseline_top1_reference_replayed'] is True,
             'D4 baseline replay not verified')
    _require(receipt['full_candidate_baseline_pairs'] == expected_n * (expected_n - 1), 'Baseline complete-pool replay count differs')
    _require(receipt['independent_reference_checks'] == 2 * expected_n + len(deletion_rows)
             and receipt['independent_candidate_checks'] > 0 and receipt['max_scalar_match_error'] <= 1e-12,
             'Independent scalar matching checks absent/inconsistent')
    return {'baseline_queries_per_protocol': expected_n, 'eligible_queries_per_protocol': eligible_n,
            'single_region_queries': expected_region_counts.get(1, 0),
            'area_matched_queries_per_protocol': len(area_ids) // 2,
            'single_region_deletion_rows': len(deletion_rows), 'strategy_rows_recomputed': len(strategy_rows),
            'summary_rows_recomputed': len(summaries),
            'baseline_replay_scope': 'compute stage byte-exact full-gallery scores; independent audit fields checked here',
            'random_expectation_scope': 'each metric independently averaged across enumerated deletion outcomes'}


def verify_output(output):
    """Validate an unsealed or sealed run, without mutating any artifact."""
    output = Path(output)
    manifest, frozen = _json(output / 'run_manifest.json'), _json(output / 'frozen_plan.json')
    for item in (manifest, frozen):
        payload = {k: v for k, v in item.items() if k != 'signature'}
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        _require(item.get('signature') == expected, 'Scientific metadata signature mismatch')
    stamp = datetime.fromisoformat(manifest['started_at'])
    _require(stamp.utcoffset() == timedelta(hours=8), 'Execution timestamp must use Asia/Shanghai offset')
    _require(manifest['run_id'] == output.name and output.name.startswith(stamp.strftime('%Y%m%dT%H%M%S%z') + '__'),
             'Run ID does not match actual execution timestamp')
    plan = manifest['context']['plan']
    _require(manifest.get('version') == 'paper_d_rotation_v1'
             and plan.get('source_variant') == 'add_a0p4_b0p15'
             and plan.get('checkpoint_selection') == 'fixed_epoch_100'
             and plan.get('inference_view') == 'original',
             'D identity does not match the frozen rotation primary')
    feature_path = manifest['context'].get('rotation_feature_path')
    _require(manifest['context'].get('rotation_feature_dtype') == 'float64'
             and feature_path in manifest['context']['source_binding']
             and manifest['context']['source_binding'][feature_path]
             == manifest['context'].get('rotation_feature_sha256'),
             'Rotation feature provenance is missing or detached')
    _require(frozen['plan'] == plan and frozen['started_at'] == manifest['started_at'], 'Frozen plan mismatch')
    _require(plan['training_required'] is False and plan['test_parameter_search'] is False
             and plan['new_formal_model_evaluations'] == 0 and plan['posthoc_research'] is True,
             'D analysis boundaries changed')
    _require(manifest['context']['source_binding'], 'Missing frozen source bindings')
    examples = verify_examples(output / 'explanations')
    perturb = verify_d4(output / 'd4')
    baselines = {(r['stage'], int(r['query_id'])): r for r in _csv(output / 'd4/baseline.csv')}
    for row in _json(output / 'explanations/examples.json')['examples']:
        valid = baselines[(row['stage'], int(row['query_id']))]['Coverage-nDCG@10'] != ''
        _require(row.get('gallery_has_coverage') is valid, 'Example positive-gallery flag differs from full-gallery IDCG audit')
    figures = verify_figures(output)
    return {'status': 'verified', 'D1_D2': examples, 'D3': figures, 'D4': perturb,
            'verification_scope': 'independent saved-asset pixel recount, contribution arithmetic, selection, strategy expectation and summary recomputation; source scoring replay/scalar checks are separate compute-stage audits',
            'analysis_only': True, 'new_formal_model_evaluations': 0,
            'limitations': ['score-evidence intervention, not pixel erasure',
                           'rank1 reference is protocol specific',
                           'area matching is approximate, not complete control of confounding',
                           'no regional semantic ground truth or physical causal claim']}


def verify_figures(output):
    """Check source binding, complete image inventory and explicit visual review."""
    output = Path(output)
    manifest = _json(output / 'figures/figures_manifest.json')
    selection = _json(output / 'explanations/selection.json')
    selected = [x for x in selection['display_cases'] if x['status'] == 'selected']
    _require(manifest['status'] == 'rendered' and manifest['main_display_cases'] == len(selected),
             'Figure case count differs from frozen selection')
    _require(manifest['anchor_gallery_examples'] == 222 and manifest['anchor_gallery_pages'] == 37,
             'All 222 anchor examples must be rendered in the full gallery')
    _require(manifest['source_explanation_verification_sha256'] == _sha(output / 'explanations/verification.json'),
             'Figure source explanation binding differs')
    _require(re.fullmatch(r'[0-9a-f]{64}', manifest['renderer_source_sha256']) is not None,
             'Renderer identity is missing')
    _files(output, manifest['files'])
    expected = {'figures/main_cases.png', 'figures/main_cases.pdf', 'figures/display_cases.csv',
                'figures/gallery/all_222_anchor_examples.pdf'}
    expected.update('figures/gallery/page_%02d.png' % i for i in range(1, 38))
    for case in selected:
        expected.update('figures/individual/' + case['category'] + ext for ext in ('.png', '.pdf'))
    _require(expected <= set(manifest['files']), 'Figure inventory is missing required output')
    png_count, pdf_count = 0, 0
    for name in manifest['files']:
        path = output / name
        if path.suffix == '.png':
            with path.open('rb') as stream:
                header = stream.read(24)
            _require(header[:8] == b'\x89PNG\r\n\x1a\n' and header[12:16] == b'IHDR', 'Invalid PNG image')
            dimensions = list(struct.unpack('>II', header[16:24]))
            _require(dimensions == manifest['png_dimensions'].get(name)
                     and min(dimensions) >= 600, 'Invalid or unexpectedly small image dimensions')
            png_count += 1
        elif path.suffix == '.pdf':
            with path.open('rb') as stream:
                _require(stream.read(5) == b'%PDF-', 'Invalid PDF image')
            pdf_count += 1
    display_rows = _csv(output / 'figures/display_cases.csv')
    _require(len(display_rows) == len(selected), 'Figure caption count mismatch')
    captions = {x['category']: x for x in display_rows}
    for case in selected:
        _require(case['category'] in captions, 'Missing figure caption')
        row = captions[case['category']]
        _require(row['example_id'] == case['example_id'] and row['selection_pool'] == case['selection_pool'],
                 'Caption source selection mismatch')
        for key in ('query_id', 'candidate_id', 'rank'):
            _require(int(row[key]) == case[key], 'Caption identity/rank mismatch')
    review = _json(output / 'figure_review.json')
    _require(review.get('status') == 'verified' and isinstance(review.get('reviewed_images'), dict)
             and 'figures/main_cases.png' in review['reviewed_images'], 'Main figure has no explicit visual review')
    _files(output, review['reviewed_images'])
    _require(set(review['reviewed_images']) <= set(manifest['files']), 'Visual review references unrelated images')
    return {'main_cases': len(selected), 'gallery_examples': 222, 'gallery_pages': 37,
            'png_files': png_count, 'pdf_files': pdf_count,
            'visually_reviewed_files': sorted(review['reviewed_images']),
            'review_scope': 'the named images were visually inspected; remaining gallery pages have artifact/content-source checks'}
