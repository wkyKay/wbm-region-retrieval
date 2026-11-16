"""Compare the fixed threshold intervention against the original F test."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
p.add_argument('--baseline', required=True)
p.add_argument('--coverage', required=True)
a = p.parse_args()
run, baseline, coverage = map(Path, (a.run, a.baseline, a.coverage))
def read(p):
    return json.loads(p.read_text())
def rows(p):
    with p.open() as f:
        return {int(row['query_id']): row for row in csv.DictReader(f)}
assert read(run / 'completed.json')['query_count'] == 7199
assert read(run / 'independent_verification.json')['queries'] == 7199
selected = read(run / 'selection_before_test.json')['selected_threshold']
comparison = {'intervention': {'shape_threshold': selected, 'score_threshold': selected},
              'baseline_threshold': 0.0, 'checkpoint_sha256': read(run / 'run_manifest.json')['checkpoint_sha256'],
              'protocol': '7199 test queries, 7198 candidates per query, identical frozen IDs',
              'interpretation': 'Threshold selected solely by valid Coverage-nDCG@10 from the predefined grid, then frozen before this test',
              'metrics': {}, 'per_query': {}}
for family, old, metric_file, query_file in (
        ('exact', baseline, 'metrics.json', 'per_query.csv'),
        ('coverage', coverage, 'coverage_metrics.json', 'coverage_per_query.csv')):
    original, changed = read(old / metric_file)['metrics'], read(run / 'test' / metric_file)['metrics']
    comparison['metrics'][family] = {k: {'threshold_0': v, 'selected_threshold': changed[k],
        'delta': changed[k] - v, 'delta_percentage_points': 100 * (changed[k] - v)} for k, v in original.items()}
    before, after = rows(old / query_file), rows(run / 'test' / query_file)
    assert before.keys() == after.keys() and len(before) == 7199
    selected = ('Precision@10', 'AP@10') if family == 'exact' else ('Coverage-nDCG@10', 'MeanCoverage@10', 'FullCoverageAP@10')
    for key in selected:
        ids = list(before)
        delta = np.array([float(after[q][key]) - float(before[q][key]) for q in ids])
        counts = np.array([int(before[q]['label_count']) for q in ids])
        comparison['per_query'][key] = {
            'improved': int((delta > 1e-12).sum()), 'unchanged': int((np.abs(delta) <= 1e-12).sum()),
            'worsened': int((delta < -1e-12).sum()), 'mean_delta': float(delta.mean()),
            'by_label_count': {str(n): {'queries': int((counts == n).sum()),
                'mean_delta': float(delta[counts == n].mean())} for n in sorted(set(counts))}}
ks = [1, 5, 10, 20, 50, 100, 500]
changes = {str(k): {'ordered_rankings_changed': 0, 'candidate_sets_changed': 0} for k in ks}
with (baseline / 'rankings.csv').open() as f, (run / 'test/rankings.csv').open() as g:
    left = itertools.groupby(csv.DictReader(f), key=lambda r: int(r['query_id']))
    right = itertools.groupby(csv.DictReader(g), key=lambda r: int(r['query_id']))
    n = 0
    for l, r in itertools.zip_longest(left, right):
        assert l is not None and r is not None and l[0] == r[0]
        lrows, rrows = list(l[1]), list(r[1])
        lids = [int(x['candidate_id']) for x in lrows]
        rids = [int(x['candidate_id']) for x in rrows]
        for k in ks:
            changes[str(k)]['ordered_rankings_changed'] += lids[:k] != rids[:k]
            changes[str(k)]['candidate_sets_changed'] += set(lids[:k]) != set(rids[:k])
        n += 1
    assert n == 7199
comparison['ranking_changes'] = changes
with (run / 'comparison.json').open('x') as f:
    json.dump(comparison, f, indent=2)
print(json.dumps(comparison, indent=2))
