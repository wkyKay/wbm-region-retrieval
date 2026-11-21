"""Traceable D1--D3 exports from the frozen primary test ranking.

This module never changes masks, regions, features, rankings or source files.
The 222 anchor pairs and the four display categories are declared before the
new run. Labels classify examples for presentation only, never for scoring.
"""
from __future__ import annotations

import csv
import hashlib
import json
from itertools import groupby
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / 'results/20261121T175924+0800__paper_c_ablation'
LABEL_NAMES = ('Center', 'Donut', 'Edge-loc', 'Edge-ring', 'Loc', 'Near-full', 'Scratch', 'Random')
RANKS = (1, 10, 100)
STAGES = ('test', 'cross_test')
DISPLAY_CATEGORIES = (
    ('ordinary_exact', 'test', 'exact'),
    ('cross_full', 'cross_test', 'full_extra'),
    ('cross_partial', 'cross_test', 'partial'),
    ('cross_none', 'cross_test', 'none'),
)
SELECTION_RULE = {
    'anchors': 'first query in each complete-label class in frozen query order; ranks 1,10,100 in each protocol',
    'display': 'within anchor pool, ascending (query_id, rank, candidate_id) in each declared category',
    'fallback': 'only if category absent from anchors: all fixed test queries at original rank1 in that protocol, same order',
    'missing_category': 'report missing; no further search or visual selection',
    'cross_none_eligibility': 'complete eligible cross gallery must contain at least one candidate sharing a query label (IDCG@10 > 0); otherwise no-overlap is not called model failure',
    'categories': [list(x) for x in DISPLAY_CATEGORIES],
    'labels_only_used_for_reporting': True,
}


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def relation(query_labels, candidate_labels):
    q, c = np.asarray(query_labels), np.asarray(candidate_labels)
    if (q.shape != (8,) or c.shape != (8,) or not np.isin(q, (0, 1)).all()
            or not np.isin(c, (0, 1)).all() or not q.any() or not c.any()):
        raise ValueError('Example labels must be nonempty 8-dimensional binary vectors')
    count = int(np.count_nonzero((q > 0) & (c > 0)))
    coverage = count / int(np.count_nonzero(q))
    kind = ('exact' if np.array_equal(q, c) else
            'full_extra' if coverage == 1 else 'partial' if coverage > 0 else 'none')
    return {'relation': kind, 'coverage': coverage,
            'query_labels': [int(x) for x in q], 'candidate_labels': [int(x) for x in c],
            'query_label_names': [name for name, bit in zip(LABEL_NAMES, q) if bit],
            'candidate_label_names': [name for name, bit in zip(LABEL_NAMES, c) if bit]}


def anchor_queries(pool, class_ids):
    result = {}
    for query in pool:
        result.setdefault(int(class_ids[int(query)]), int(query))
    return result


def choose_display(anchors, rank1_records):
    """Deterministic category-based selection; never rank by contribution gap."""
    selected = []
    for name, stage, kind in DISPLAY_CATEGORIES:
        def qualifies(x):
            return (x['stage'] == stage and x['relation'] == kind
                    and (name != 'cross_none' or x.get('gallery_has_coverage', True)))
        eligible = [x for x in anchors if qualifies(x)]
        source = 'anchor_pool'
        if not eligible:
            eligible = [x for x in rank1_records if qualifies(x)]
            source = 'all_test_rank1_fallback'
        if eligible:
            row = min(eligible, key=lambda x: (x['query_id'], x['rank'], x['candidate_id']))
            selected.append({'category': name, 'selection_pool': source,
                             'status': 'selected', **row})
        else:
            selected.append({'category': name, 'selection_pool': source, 'status': 'missing',
                             'stage': stage, 'relation': kind})
    return selected


def _ranking_records(source, stage, queries, representatives, labels):
    folder = Path(source) / stage / 'primary'
    path = folder / 'rankings.csv'
    marker = json.loads((folder / 'completed.json').read_text())
    expected = marker.get('files', {}).get('rankings.csv')
    sha = _hash(path)
    if marker.get('status') != 'completed' or not expected or sha != expected:
        raise ValueError('Unsealed or changed primary ranking: ' + str(path))
    anchors, first, seen = [], [], []
    representatives = set(representatives.values())
    unique, counts = np.unique(labels[sorted(queries)], axis=0, return_counts=True)
    available = {}
    for values in unique:
        same = np.all(unique == values, axis=1)
        overlap = np.any((unique > 0) & (values > 0), axis=1)
        available[tuple(values)] = bool(np.any(overlap & ~same)) if stage == 'cross_test' else bool(
            np.sum(counts[overlap]) > 1)
    with path.open(newline='') as f:
        for q, rows in groupby(csv.DictReader(f), key=lambda r: int(r['query_id'])):
            if q not in queries or q in seen:
                raise ValueError('Unknown or repeated ranking query')
            seen.append(q)
            count = 0
            for count, row in enumerate(rows, 1):
                rank = int(row['rank'])
                if rank != count:
                    raise ValueError('Nonsequential ranking rows')
                if rank == 1 or (q in representatives and rank in RANKS):
                    c = int(row['candidate_id'])
                    if c == q or c not in queries:
                        raise ValueError('Invalid candidate identity in ranking')
                    record = {'stage': stage, 'query_id': q, 'candidate_id': c,
                              'rank': rank, 'ranking_score': float(row['similarity_score']),
                              'gallery_has_coverage': available[tuple(labels[q])],
                              **relation(labels[q], labels[c])}
                    if stage == 'cross_test' and record['relation'] == 'exact':
                        raise ValueError('Cross-combination ranking contains exact-label candidate')
                    if rank == 1:
                        first.append(record)
                    if q in representatives and rank in RANKS:
                        anchors.append(record)
            if count != 500:
                raise ValueError('Expected sealed Top500 source ranking')
    if set(seen) != queries or len(anchors) != len(representatives) * len(RANKS):
        raise ValueError('Incomplete query or anchor coverage')
    return anchors, first, {'path': str(path.resolve()), 'sha256': sha,
                            'completed_sha256': _hash(folder / 'completed.json'),
                            'queries': len(seen), 'ranking_rows': len(seen) * 500}


def validate_mask(sample_id, raw, mask, cache, row):
    """Bind each original-pixel mask to its exact h512/J token ordering."""
    raw, mask = np.asarray(raw), np.asarray(mask)
    offsets = np.asarray(cache['offsets'])
    start, stop = map(int, offsets[row:row + 2])
    count = stop - start
    if int(cache['map_ids'][row]) != int(sample_id) or raw.shape != mask.shape:
        raise ValueError('Mask map identity or dimensions disagree')
    if (mask.dtype.kind not in 'iu' or np.any(mask < 0) or np.any(mask > count)
            or np.any(raw[mask > 0] != 2)):
        raise ValueError('Region mask contains invalid labels or non-defect pixels')
    areas = np.bincount(mask.ravel(), minlength=count + 1)[1:]
    if not np.array_equal(areas, np.asarray(cache['area'][start:stop])):
        raise ValueError('Mask pixel areas disagree with original region feature order')
    return {'sample_id': int(sample_id), 'region_count': count,
            'region_areas': areas.astype(int).tolist(), 'original_defect_pixels': int((raw == 2).sum()),
            'retained_defect_pixels': int((mask > 0).sum()),
            'local_mask_label_equals_local_index_plus_one': True}


def enrich_explanation(explanation, ranking_record, areas):
    """Keep original numeric scores and separately audit algebraic C/T/A terms."""
    row = {**explanation, **ranking_record}
    weights = np.sqrt(np.asarray(areas, dtype=np.float64))
    denominator = float(weights.sum())
    if abs(float(explanation['query_weight_denominator']) - denominator) > 1e-12:
        raise ValueError('Explanation changes original complete-query denominator')
    sum_actual, sum_algebraic = 0., 0.
    matches = []
    used_q, used_c = set(), set()
    for old in explanation['matches']:
        m = dict(old)
        qi, ci = int(m['query_local_index']), int(m['candidate_local_index'])
        if qi in used_q or ci in used_c:
            raise ValueError('Explanation violates one-to-one matching')
        used_q.add(qi); used_c.add(ci)
        w = float(weights[qi] / denominator)
        m['query_normalized_weight'] = w
        m['query_region_area'] = int(areas[qi])
        for channel, coefficient in (('C', .30), ('T', .45), ('A', .25)):
            m['contribution_' + channel] = w * coefficient * float(m[channel])
        subtotal = sum(m['contribution_' + c] for c in ('C', 'T', 'A'))
        m['rounding_residual'] = float(m['contribution']) - subtotal
        sum_actual += float(m['contribution'])
        sum_algebraic += subtotal
        matches.append(m)
    row['matches'] = matches
    row['query_regions'] = len(areas)
    row['query_region_areas'] = np.asarray(areas, dtype=int).tolist()
    row['actual_contribution_sum'] = sum_actual
    row['algebraic_component_sum'] = sum_algebraic
    row['score_reconstruction_error'] = abs(sum_actual - row['ranking_score'])
    row['explanation_score_error'] = abs(float(explanation['score']) - row['ranking_score'])
    row['algebraic_decomposition_error'] = abs(sum_algebraic - row['ranking_score'])
    if (row['score_reconstruction_error'] > 1e-12 or row['explanation_score_error'] > 1e-12
            or row['algebraic_decomposition_error'] > 2e-6):
        raise ValueError('Explanation or component contribution does not reconstruct primary score')
    return row


def _independent_matches(provider, query, candidate, parts):
    from f_region_evidence.scoring import score_components
    qt, ct = provider.tokens(query), provider.tokens(candidate)
    pair = score_components(parts)[:, ct]
    available = sorted(((float(pair[i, j]), i, j) for i in range(len(qt)) for j in range(len(ct))),
                       key=lambda t: (-t[0], t[1], t[2]))
    uq, uc, chosen = set(), set(), []
    for value, i, j in available:
        if value > 0 and i not in uq and j not in uc:
            chosen.append((i, j)); uq.add(i); uc.add(j)
    return chosen


def run(output, data, source_root=None):
    """Export D1--D3 data; plotting is a separate, source-independent operation.

    Additional input keys are maps (indexed by original NPZ row ID),
    region_labels (map-major masks), and mask_map_ids (their original IDs).
    """
    from f_region_evidence.scoring import Provider
    from fgw_cross_test.state import cross_pool

    output, source = Path(output), Path(source_root or DEFAULT_SOURCE)
    folder = output / 'explanations'
    if folder.exists():
        raise FileExistsError('Explanation export refuses to replace an existing directory')
    folder.mkdir(parents=True)
    cache, pool, bundle = data['cache'], data['pool'], data['bundle']
    labels = np.asarray(bundle.labels)
    queries = set(map(int, pool))
    representatives = anchor_queries(pool, bundle.class_ids)
    if len(queries) != 7199 or len(representatives) != 37:
        raise ValueError('D explanations require all 7199 fixed test queries and 37 classes')
    if not np.array_equal(np.asarray(data['mask_map_ids']), np.asarray(cache['map_ids'])):
        raise ValueError('Mask row ordering differs from region feature/J ordering')
    anchors, rank1, sources = [], [], {}
    for stage in STAGES:
        a, f, binding = _ranking_records(source, stage, queries, representatives, labels)
        anchors.extend(a); rank1.extend(f); sources[stage] = binding
    selected = choose_display(anchors, rank1)
    wanted = {(x['stage'], x['query_id'], x['candidate_id']): dict(x, anchor=True) for x in anchors}
    for x in selected:
        if x['status'] == 'selected':
            key = (x['stage'], x['query_id'], x['candidate_id'])
            if key not in wanted:
                wanted[key] = {k: v for k, v in x.items() if k not in ('category', 'selection_pool', 'status')}
                wanted[key]['anchor'] = False
    providers = {'test': Provider(cache, pool),
                 'cross_test': Provider(cache, cross_pool(pool, labels))}
    examples, matched_count = [], 0
    # The same full-gallery C/T/A arrays are shared by both protocol explanations.
    by_query = {}
    for value in wanted.values():
        by_query.setdefault(value['query_id'], []).append(value)
    for number, (q, records) in enumerate(sorted(by_query.items()), 1):
        parts = providers['test'].parts(q)
        qt = providers['test'].tokens(q)
        for record in sorted(records, key=lambda x: (x['stage'], x['rank'], x['candidate_id'])):
            provider = providers[record['stage']]
            explanation = provider.explain(q, record['candidate_id'], parts=parts)
            actual_matches = [(x['query_local_index'], x['candidate_local_index']) for x in explanation['matches']]
            if actual_matches != _independent_matches(provider, q, record['candidate_id'], parts):
                raise ValueError('Independent scalar matching differs from displayed correspondence')
            enriched = enrich_explanation(explanation, record, cache['area'][qt])
            enriched['example_id'] = f"{record['stage']}__q{q}__r{record['rank']}__c{record['candidate_id']}"
            examples.append(enriched)
            matched_count += len(enriched['matches'])
        print(f'D1/D2 explanation queries: {number}/{len(by_query)}', flush=True)
    examples.sort(key=lambda x: (x['stage'], x['query_id'], x['rank'], x['candidate_id']))
    examples_by_key = {(x['stage'], x['query_id'], x['candidate_id']): x for x in examples}
    for chosen in selected:
        if chosen['status'] == 'selected':
            chosen['example_id'] = examples_by_key[(chosen['stage'], chosen['query_id'], chosen['candidate_id'])]['example_id']
    sample_ids = sorted({x[key] for x in examples for key in ('query_id', 'candidate_id')})
    positions = {int(s): i for i, s in enumerate(cache['map_ids'])}
    maps, masks, audits = [], [], []
    for sample_id in sample_ids:
        pos = positions[sample_id]
        raw, mask = data['maps'][sample_id], data['region_labels'][pos]
        audits.append(validate_mask(sample_id, raw, mask, cache, pos))
        maps.append(np.array(raw, copy=True)); masks.append(np.array(mask, copy=True))
    np.savez_compressed(folder / 'assets.npz', sample_ids=np.asarray(sample_ids, dtype=np.int64),
                        raw_maps=np.asarray(maps), region_masks=np.asarray(masks))
    _write_json(folder / 'examples.json', {'schema_version': 1, 'split': 'test',
        'method': 'add_a0p4_b0p25', 'formula': '0.75*(0.4*C+0.6*T)+0.25*A',
        'label_names': list(LABEL_NAMES), 'selection_rule': SELECTION_RULE,
        'sources': sources, 'examples': examples})
    _write_json(folder / 'selection.json', {'rule': SELECTION_RULE,
        'representative_queries_by_class': {str(k): v for k, v in representatives.items()},
        'anchor_count': len(anchors), 'exported_count': len(examples), 'display_cases': selected})
    _write_json(folder / 'mask_audit.json', {'status': 'verified', 'maps': audits})
    fields = ['example_id', 'stage', 'query_id', 'candidate_id', 'rank', 'query_local_index',
              'candidate_local_index', 'query_token', 'candidate_token', 'query_region_area',
              'query_normalized_weight', 'C', 'T', 'A', 'raw_score', 'contribution_C',
              'contribution_T', 'contribution_A', 'rounding_residual', 'contribution']
    with (folder / 'matches.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for e in examples:
            for match in e['matches']:
                writer.writerow({k: (match[k] if k in match else e[k]) for k in fields})
    verify = {'status': 'verified', 'anchor_pairs': len(anchors), 'exported_pairs': len(examples),
        'pairs_by_protocol': {s: sum(e['stage'] == s for e in examples) for s in STAGES},
        'independent_scalar_matching_checked_pairs': len(examples), 'matched_region_pairs': matched_count,
        'source_rankings_hash_checked': True, 'unique_source_maps_checked': len(sample_ids),
        'max_score_reconstruction_error': max(x['score_reconstruction_error'] for x in examples),
        'max_explanation_score_error': max(x['explanation_score_error'] for x in examples),
        'max_algebraic_decomposition_error': max(x['algebraic_decomposition_error'] for x in examples),
        'score_tolerance': 1e-12, 'algebraic_tolerance': 2e-6,
        'algebraic_rounding_note': 'C originates from float32 GEMM; original operation order is retained for scores; nominal 0.30C+0.45T+0.25A terms have recorded rounding residuals',
        'region_semantic_ground_truth_available': False, 'selection_rule': SELECTION_RULE,
        'files': {p.name: _hash(p) for p in sorted(folder.iterdir()) if p.is_file()}}
    _write_json(folder / 'verification.json', verify)
    return verify
