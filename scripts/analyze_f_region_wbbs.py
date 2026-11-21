#!/usr/bin/env python3
"""Analyze a completely sealed regional-WBBS study; never select from test."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np

KS = (1, 5, 10, 20, 50, 100, 500)
BASE = 'region_j'
WBBS = 'global_wbbs'
BASE_PROTOCOL = '6a428fb8d8bdf34fb6c60852b6d9a7467bf6880cae8c802a6a9c5a36f96cb8a1'
CROSS_VALID_PROTOCOL = '8cce967e52a329e26b5a970e0740f1da5c3cd814df8046acbd36a2f21df319ae'
CROSS_PROTOCOL = '3aaf4895309e5d67c2aa1a196581a70237b0b72a8bc0c8f806df7b9947065bf0'
EXACT = [f'{m}@{k}' for k in KS for m in ('Hit', 'Precision', 'Recall', 'AP')]
COVERAGE = [f'{m}@{k}' for k in KS for m in ('Coverage-nDCG', 'MeanCoverage',
    'FullCoverageHit', 'FullCoveragePrecision', 'FullCoverageRecall', 'FullCoverageAP')]
CORE = {'test': ('exact/Precision@10', 'exact/AP@100', 'coverage/Coverage-nDCG@10',
                'coverage/FullCoveragePrecision@10', 'coverage/FullCoverageAP@100'),
        'cross_test': ('coverage/Coverage-nDCG@10', 'coverage/MeanCoverage@10',
                'coverage/FullCoverageHit@10', 'coverage/FullCoveragePrecision@10', 'coverage/FullCoverageAP@100')}
LIMITATIONS = ('Pointwise percentile intervals conditional on the frozen models, rankings, gallery and observed complete-label-combination proportions. '
               'Only queries are resampled within complete label combinations; no gallery, training-seed or dataset resampling. '
               'No correction for multiple comparisons, validation selection or prior test-informed research. '
               'Test never selects a variant or parameter; confidence intervals are descriptive conditional uncertainty, not a simultaneous significance claim.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def write_csv(path, rows):
    require(bool(rows), 'No output rows for '+str(path))
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Binding:
    def __init__(self):
        self.files = {}

    def bind(self, path, expected=None):
        require(not Path(path).is_symlink(), 'Symbolic link source is not permitted '+str(path))
        path = Path(path).resolve()
        require(path.is_file() and not path.is_symlink(), 'Missing/invalid source '+str(path))
        value = sha(path)
        require(expected is None or value == expected, 'Source hash mismatch '+str(path))
        require(str(path) not in self.files or self.files[str(path)] == value, 'Source changed '+str(path))
        self.files[str(path)] = value
        return path

    def sealed(self, root, names):
        root = Path(root).resolve()
        marker = read(self.bind(root/'completed.json'))
        require(marker.get('status') == 'completed' or
                (marker.get('test_evaluated') is True and marker.get('split') == 'test'
                 and marker.get('smoke') is not True), 'Source is not a completed formal evaluation')
        require(isinstance(marker.get('files'), dict) and marker['files'], 'Missing completion file manifest')
        for name in names:
            rel = Path(name)
            require(not rel.is_absolute() and '..' not in rel.parts, 'Source path escape')
            require(name in marker['files'], 'Completion does not bind '+name)
            self.bind(root/rel, marker['files'][name])
        return marker

    def signed(self, path):
        obj = read(self.bind(path))
        require(obj.get('signature') == digest({k: v for k, v in obj.items() if k != 'signature'}), 'Bad signed JSON')
        return obj

    def recheck(self):
        for path, value in self.files.items():
            require(sha(path) == value, 'Analysis source changed '+path)


def label_data(repo, binding):
    path = binding.bind(repo/'protocol/deduplicated/split.csv')
    with path.open() as f:
        rows = [r for r in csv.DictReader(f) if r['split'] == 'test']
    rows.sort(key=lambda r: int(r['sample_id']))
    ids = np.array([int(r['sample_id']) for r in rows], dtype=np.int64)
    labels = np.array([[int(r[f'label_{i}']) for i in range(8)] for r in rows], dtype=np.uint8)
    require(len(ids) == len(np.unique(ids)) == 7199 and np.isin(labels, [0, 1]).all(), 'Not frozen test identities')
    count = labels.sum(axis=1)
    patterns = [''.join(map(str, x)) for x in labels]
    groups = {'all': np.ones(len(ids), bool), 'single': count == 1, 'mixed': count > 1}
    groups.update({f'label_count_{int(n)}': count == n for n in sorted(set(count))})
    return ids, labels, patterns, groups


def numeric(value):
    if value in ('', None):
        return np.nan
    number = float(value)
    require(math.isfinite(number) and 0 <= number <= 1, 'Invalid metric value')
    return number


def aggregate_key(metric):
    if metric.startswith('FullCoverageAP@'):
        return metric.replace('FullCoverageAP@', 'FullCoverage_mAP@')
    return 'm'+metric if metric.startswith('AP@') else metric


def load_family(folder, kind, ids, labels, cross, binding):
    folder = Path(folder)
    csv_name, json_name = ('per_query.csv', 'metrics.json') if kind == 'exact' else ('coverage_per_query.csv', 'coverage_metrics.json')
    with binding.bind(folder/csv_name).open() as f:
        records = list(csv.DictReader(f))
    by_id = {int(row['query_id']): row for row in records}
    require(len(records) == len(by_id) == len(ids) and set(by_id) == set(map(int, ids)), 'Mismatched/duplicate report query IDs')
    fields = EXACT if kind == 'exact' else COVERAGE
    matrix = np.empty((len(ids), len(fields)))
    pool_sizes, positives = [], []
    # Recompute eligibility/positive counts by complete combination, without class-ID assumptions.
    expected = {}
    for label in {tuple(map(int, x)) for x in labels}:
        q = np.asarray(label)
        same = np.all(labels == q, axis=1)
        eligible = ~same if cross else np.ones(len(ids), bool)
        exact_count = int(same.sum())-1
        full_count = int(np.all(labels >= q, axis=1)[eligible].sum()) - (0 if cross else 1)
        any_count = int(np.any((labels & q) != 0, axis=1)[eligible].sum()) - (0 if cross else 1)
        expected[label] = (int(eligible.sum())-(0 if cross else 1), exact_count, full_count, any_count)
    for i, query in enumerate(ids):
        row, label = by_id[int(query)], labels[i]
        pool, exact_n, full_n, any_n = expected[tuple(map(int, label))]
        signature = '|'.join(map(str, np.flatnonzero(label))) if kind == 'exact' else ''.join(map(str, label))
        require(row['label_signature'] == signature and int(row['label_count']) == int(label.sum())
                and int(row['pool_size']) == pool and int(row['returned']) == 500, 'Query metadata differs')
        npositive = int(row['relevant_in_pool'] if kind == 'exact' else row['full_coverage_in_pool'])
        require(npositive == (exact_n if kind == 'exact' else full_n), 'Full eligible-pool positive denominator differs')
        values = np.array([numeric(row[k]) for k in fields])
        for j, key in enumerate(fields):
            missing = ((key.startswith(('FullCoverageRecall@', 'FullCoverageAP@')) and full_n == 0)
                       or (key.startswith('Coverage-nDCG@') and any_n == 0)
                       or (kind == 'exact' and key.startswith(('Recall@', 'AP@')) and exact_n == 0))
            require(bool(np.isnan(values[j])) == bool(missing), 'Unexpected metric validity mask')
        matrix[i] = values
        pool_sizes.append(pool); positives.append(npositive)
    report = read(binding.bind(folder/json_name))
    require(report['split'] == 'test' and report['num_queries'] == len(ids)
            and report['ks'] == list(KS) and report['metric_version'] == kind+'_v1', 'Aggregate protocol mismatch')
    if kind == 'exact':
        require(report['retrieval_protocol_signature'] == BASE_PROTOCOL, 'Wrong exact retrieval protocol')
    for j, key in enumerate(fields):
        valid = np.isfinite(matrix[:, j])
        mean = float(np.mean(matrix[valid, j])) if valid.any() else None
        wanted = report['metrics'][aggregate_key(key)]
        require((wanted is None and mean is None) or
                (wanted is not None and mean is not None and abs(wanted-mean) < 1e-12), 'Aggregate differs from query reports')
    return {kind+'/'+key: matrix[:, j] for j, key in enumerate(fields)}


def evaluation_files(stage):
    names = ['evaluation_manifest.json', 'coverage_metrics.json', 'coverage_per_query.csv']
    if stage == 'test':
        names += ['metrics.json', 'per_query.csv']
    return names


def check_new_identity(folder, stage, name, selection, context_signature, binding):
    """Bind a frozen chosen child to its completed root, not just its protocol."""
    folder = Path(folder)
    run = folder.parent.parent
    names = ['completed.json', *evaluation_files(stage)]
    binding.sealed(run, [str((folder/n).relative_to(run)) for n in names])
    marker = binding.sealed(folder, evaluation_files(stage))
    manifest = read(folder/'evaluation_manifest.json')
    require(name in selection['selected_variants'] and name in selection['selected_specs'],
            'Child is not a frozen prespecified test item')
    spec = selection['selected_specs'][name]
    signature = digest({'context': context_signature, 'stage': stage, 'variant': name, 'spec': spec})
    require(marker.get('signature') == signature
            and manifest.get('signature') == signature
            and manifest.get('evaluation_signature') == signature
            and manifest.get('context_signature') == context_signature
            and manifest.get('variant') == name and manifest.get('stage') == stage
            and manifest.get('split') == 'test' and manifest.get('spec') == spec
            and manifest.get('frozen_selection_sha256') == sha(run/'selection.json')
            and manifest.get('primary_variant') == selection['winner']
            and manifest.get('best_regional_wbbs') == selection['best_regional_wbbs'],
            'Child evaluation identity/specification/selection differs from the completed run')


def load_evaluation(folder, stage, ids, labels, binding):
    binding.sealed(folder, evaluation_files(stage))
    manifest = read(folder/'evaluation_manifest.json')
    require(manifest.get('split') == 'test' and manifest.get('retrieval_protocol_signature') ==
            (CROSS_PROTOCOL if stage == 'cross_test' else BASE_PROTOCOL), 'Evaluation candidate protocol differs')
    out = load_family(folder, 'coverage', ids, labels, stage == 'cross_test', binding)
    if stage == 'test':
        out.update(load_family(folder, 'exact', ids, labels, False, binding))
    return out


def bootstrap(differences, patterns, repetitions=2000, seed=2026):
    """Same stratified query draws for all contrasts/metrics; preserve missingness."""
    d = np.asarray(differences, dtype=np.float64)
    require(d.ndim == 2 and len(d) == len(patterns), 'Bootstrap shape mismatch')
    total = np.zeros((repetitions, d.shape[1])); counts = np.zeros_like(total)
    rng = np.random.default_rng(seed)
    for label in sorted(set(patterns)):
        at = np.flatnonzero(np.asarray(patterns) == label)
        value = d[at]; valid = np.isfinite(value)
        for first in range(0, repetitions, 100):
            n = min(100, repetitions-first)
            draws = rng.integers(0, len(at), size=(n, len(at)))
            total[first:first+n] += np.nan_to_num(value)[draws].sum(axis=1)
            counts[first:first+n] += valid[draws].sum(axis=1)
    return np.divide(total, counts, out=np.full_like(total, np.nan), where=counts > 0)



def frozen_grid():
    result={BASE:{'family':'j_baseline'}}
    for prefix,mode in [('all','all_points'),('denoised','wbbs_denoised')]:
        for label,penalty in [('bb',0.),('wbbs',5.)]:
            for weight,suffix in [(.25,'025'),(.4,'04'),(.6,'06')]:
                result[f'{prefix}_{label}_a{suffix}']={'family':'wbbs','point_mode':mode,
                    'count_alpha':penalty,'cnn_weight':weight,'area_weight':.25}
    return result


def independently_check_selection(run, selection, context, binding):
    grid=frozen_grid();require(context['grid']==grid,'Unexpected regional grid')
    context_signature=digest(context)
    complete=read(run/'valid_completed.json')
    require(complete.get('status')=='completed'
        and complete.get('context_signature')==selection.get('context_signature')==context_signature
        and complete.get('selection_sha256')==sha(run/'selection.json'),
        'Frozen valid completion/selection identity differs')
    stages=('valid','cross_valid')
    for stage in stages:
        names=['evaluation_index_'+stage+'.json']
        binding.sealed(run,names)
        path=run/names[0];index=read(path)
        require(complete['indexes'][stage]==sha(path)==selection[stage+'_index_sha256'], 'Selection index bytes differ')
        expected=grid if stage=='valid' else {**grid,WBBS:{'family':'global_wbbs'}}
        require(index.get('context_signature')==context_signature and index.get('stage')==stage
            and index.get('split')=='valid' and index['specs']==expected
            and set(index['results'])==set(expected), 'Incomplete or detached frozen valid grid')
        for name,spec in expected.items():
            child=run/stage/name
            marker=binding.sealed(child,['coverage_metrics.json','evaluation_manifest.json'])
            binding.sealed(run,[str((child/'completed.json').relative_to(run))])
            require(complete['evaluations'][stage][name]==sha(child/'completed.json')==index['results'][name]['completed_sha256'], 'Valid child completion differs')
            child_manifest=read(child/'evaluation_manifest.json')
            signature=digest({'context':context_signature,'stage':stage,'variant':name,'spec':spec})
            protocol=CROSS_VALID_PROTOCOL if stage=='cross_valid' else BASE_PROTOCOL
            rule='all_gallery_except_query_and_exact_label_combination' if stage=='cross_valid' else 'all_gallery_except_query'
            require(marker.get('signature')==signature
                and child_manifest.get('signature')==child_manifest.get('evaluation_signature')==signature
                and child_manifest.get('context_signature')==context_signature
                and child_manifest.get('stage')==stage and child_manifest.get('split')=='valid'
                and child_manifest.get('variant')==name and child_manifest.get('spec')==spec
                and child_manifest.get('num_queries')==500 and child_manifest.get('gallery_count')==3599
                and child_manifest.get('ks')==list(KS) and child_manifest.get('tie_seed')==2026
                and child_manifest.get('retrieval_protocol_signature')==protocol
                and child_manifest.get('base_protocol_signature')==BASE_PROTOCOL
                and child_manifest.get('candidate_rule')==rule
                and child_manifest.get('primary_variant') is None
                and child_manifest.get('best_regional_wbbs') is None
                and child_manifest.get('frozen_selection_sha256') is None,
                'Valid child identity/stage/protocol/specification differs')
            report=read(child/'coverage_metrics.json')
            require(report.get('split')=='valid' and report.get('num_queries')==500
                and report.get('ks')==list(KS) and report.get('metric_version')=='coverage_v1'
                and all(report.get('metadata',{}).get(k)==v for k,v in child_manifest.items()),
                'Valid aggregate metadata differs from its child identity')
            require(report['metrics']==index['results'][name]['coverage_metrics'], 'Valid metric index differs from report')
    values={name:row['coverage_metrics']['Coverage-nDCG@10'] for name,row in index['results'].items()}
    def tie(name):
        if name==BASE:return (0,0.,0,0.,name)
        s=grid[name];return (1,abs(s['cnn_weight']-.4),s['point_mode']!='all_points',s['count_alpha'],name)
    def choose(names):
        best=max(values[n] for n in names)
        return min((n for n in names if best-values[n]<=1e-12),key=tie)
    primary=choose(list(grid));best=choose([n for n in grid if n!=BASE])
    controls=[n for n,s in grid.items() if s.get('cnn_weight')==.4]
    selected=list(dict.fromkeys([*controls,best]))
    require(selection['winner']==primary and selection['best_regional_wbbs']==best
        and selection['fixed_controls']==controls and selection['selected_variants']==selected
        and selection['selected_specs']=={n:grid[n] for n in selected}
        and selection['selection_values']==values
        and selection['valid_order']==sorted(grid,key=lambda n:(-values[n],tie(n)))
        and selection['test_reselection'] is False, 'Selection differs from independent cross-valid rule')
    return grid


def comparison_plan(selection):
    pairs=[];aliases=[]
    for name in selection['selected_variants']:
        roles=[]
        if name in selection['fixed_controls']:roles.append('fixed_weight_0.4_control')
        if name==selection['best_regional_wbbs']:roles.append('best_regional_wbbs')
        if name==selection['winner']:roles.append('primary')
        for reference in (BASE,WBBS):
            pair=(name,reference);pairs.append(pair)
            aliases.append({'left':name,'right':reference,'roles':roles,'pair_index':len(pairs)-1,
                'primary_inference':name==selection['best_regional_wbbs']})
    return pairs,aliases


def role_names(name, selection):
    roles=[]
    if name==selection['winner']:roles.append('primary')
    if name==selection['best_regional_wbbs']:roles.append('best_regional_wbbs')
    if name in selection['fixed_controls']:roles.append('fixed_weight_0.4_control')
    if name==BASE:roles.append('regional_J_reference')
    if name==WBBS:roles.append('global_WBBS_reference')
    return '|'.join(roles)


def summaries(loaded,selection,groups,patterns):
    rows,differences=[],[];pairs,aliases=comparison_plan(selection)
    patterns=np.asarray(patterns)
    class_groups={p:patterns==p for p in sorted(set(patterns))}
    all_groups={**groups,**{'class_'+p:m for p,m in class_groups.items()}}
    for stage,methods in loaded.items():
        for name,metrics in methods.items():
            for metric,vector in metrics.items():
                for group,mask in all_groups.items():
                    valid=mask&np.isfinite(vector)
                    rows.append({'stage':stage,'method':name,'roles':role_names(name,selection),'group':group,
                        'metric':metric,'queries':int(mask.sum()),'valid_queries':int(valid.sum()),
                        'valid_classes':len(set(patterns[valid])),
                        'value':float(vector[valid].mean()) if valid.any() else None})
                means=[float(vector[m&np.isfinite(vector)].mean()) for m in class_groups.values() if np.any(m&np.isfinite(vector))]
                rows.append({'stage':stage,'method':name,'roles':role_names(name,selection),'group':'class_macro',
                    'metric':metric,'queries':len(vector),'valid_queries':int(np.isfinite(vector).sum()),
                    'valid_classes':len(means),'value':float(np.mean(means)) if means else None})
        for left,right in pairs:
            require(set(methods[left])==set(methods[right]),'Paired metric sets differ')
            for metric in methods[left]:
                a,b=methods[left][metric],methods[right][metric]
                require(np.array_equal(np.isfinite(a),np.isfinite(b)), 'Paired defined-query masks differ')
                delta=a-b
                for group,mask in all_groups.items():
                    valid=mask&np.isfinite(a)&np.isfinite(b);d=delta[valid]
                    differences.append({'stage':stage,'left':left,'right':right,
                        'primary_inference':left==selection['best_regional_wbbs'],'group':group,'metric':metric,
                        'queries':int(mask.sum()),'valid_queries':int(valid.sum()),'valid_classes':len(set(patterns[valid])),
                        'difference':float(d.mean()) if len(d) else None,'wins':int((d>1e-12).sum()),
                        'ties':int((np.abs(d)<=1e-12).sum()),'losses':int((d < -1e-12).sum())})
                means=[float(delta[m&np.isfinite(delta)].mean()) for m in class_groups.values() if np.any(m&np.isfinite(delta))]
                differences.append({'stage':stage,'left':left,'right':right,
                    'primary_inference':left==selection['best_regional_wbbs'],'group':'class_macro','metric':metric,
                    'queries':len(delta),'valid_queries':int(np.isfinite(delta).sum()),'valid_classes':len(means),
                    'difference':float(np.mean(means)) if means else None,
                    'wins':sum(x>1e-12 for x in means),'ties':sum(abs(x)<=1e-12 for x in means),
                    'losses':sum(x < -1e-12 for x in means)})
    return rows,differences,pairs,aliases


CI_GROUPS=('all','single','mixed','label_count_2','label_count_3','label_count_4')

def intervals(loaded,selection,pairs,groups,patterns,repetitions=2000,seed=2026):
    rows=[]
    for stage,methods in loaded.items():
        keys=[(a,b,k) for a,b in pairs for k in CORE[stage]]
        differences=np.column_stack([methods[a][k]-methods[b][k] for a,b,k in keys])
        for group in CI_GROUPS:
            mask=groups[group];sub=differences[mask]
            draws=bootstrap(sub,np.asarray(patterns)[mask],repetitions,seed)
            for j,(a,b,metric) in enumerate(keys):
                valid=np.isfinite(sub[:,j]);finite=draws[np.isfinite(draws[:,j]),j]
                ci=np.quantile(finite,[.025,.975],method='linear') if len(finite) else (None,None)
                values=[None if x is None else float(x) for x in ci]
                mean=float(sub[valid,j].mean()) if valid.any() else None
                rows.append({'stage':stage,'left':a,'right':b,'roles':role_names(a,selection),
                    'primary_inference':a==selection['best_regional_wbbs'],'group':group,'metric':metric,
                    'valid_queries':int(valid.sum()),'difference':mean,'ci_low':values[0],'ci_high':values[1],
                    'difference_pp':None if mean is None else 100*mean,
                    'ci_low_pp':None if values[0] is None else 100*values[0],
                    'ci_high_pp':None if values[1] is None else 100*values[1],
                    'repetitions':repetitions,'seed':seed,'pointwise_only':True})
    return rows


def point_diagnostics(run,ids,groups,binding):
    """Read prepared counts and genuine self BBP diagonals; never force self=1."""
    folder=run/'cache/test';count_folder=run/'counts/test'
    names=['audit.json','regions.npz',*[f'{m}_offsets.npy' for m in ('all_points','wbbs_denoised')]]
    binding.sealed(run,[f'cache/test/{n}' for n in ['completed.json',*names]])
    binding.sealed(folder,names)
    audit=read(folder/'audit.json')
    with np.load(folder/'regions.npz',allow_pickle=False) as z:
        owners=np.array(z['owners']);region_ids=np.array(z['map_ids']);region_offsets=np.array(z['offsets'])
    require(np.array_equal(region_ids,ids) and np.array_equal(owners,np.repeat(ids,np.diff(region_offsets))), 'Point-diagnostic original region ownership differs')
    counts_marker=binding.sealed(count_folder,['metadata.json','all_points.npy','wbbs_denoised.npy'])
    binding.sealed(run,['counts/test/completed.json'])
    count_meta=read(count_folder/'metadata.json')
    require(count_meta['upper_triangle'] is True and count_meta['diagonal']=='original exact mutual-nearest-neighbour count; never forced', 'Unexpected count diagonal scope')
    rows=[]
    for mode in ('all_points','wbbs_denoised'):
        offsets=np.load(folder/f'{mode}_offsets.npy');totals=np.diff(offsets)
        matrix=np.load(count_folder/f'{mode}.npy',mmap_mode='r');diag=np.asarray(matrix.diagonal()).copy()
        require(matrix.shape==(len(totals),len(totals)) and np.all(diag<=totals), 'Invalid real self BBP counts')
        self_scores=np.divide(diag,totals,out=np.zeros(len(totals),dtype=np.float64),where=totals>0)
        for group,mask in groups.items():
            use=np.isin(owners,ids[mask]);n=totals[use];score=self_scores[use]
            rows.append({'point_mode':mode,'group':group,'queries':int(mask.sum()),'regions':int(use.sum()),
                'points':int(n.sum()),'empty_regions':int((n==0).sum()),'singleton_regions':int((n==1).sum()),
                'self_score_below_one_regions':int((score<1).sum()),
                'nonempty_self_score_below_one_regions':int(((n>0)&(score<1)).sum()),
                'mean_self_score':float(score.mean()) if len(score) else None,
                'minimum_self_score':float(score.min()) if len(score) else None,
                'duplicate_descriptor_rows_all_only':audit['point_sets'][mode]['duplicate_descriptor_rows'] if group=='all' else None,
                'scope':'region-weighted diagnostics; real full-count diagonal, count penalty=1 on self; not a retrieved self candidate or causal attribution'})
    return rows


def explanation_summary(run,selection,binding):
    rows=[]
    for stage in ('test','cross_test'):
        for name in selection['selected_variants']:
            path=run/'explanations'/stage/name/'examples.json'
            binding.sealed(run,[str(path.relative_to(run))]);source=read(path)
            require(source['selection']=='first fixed query per complete combination; ranks 1,10,100', 'Unexpected explanation example choice')
            for rank in (1,10,100):
                records=[r for r in source['examples'] if r['rank']==rank]
                require(len(records)==37 and len({r['query_id'] for r in records})==37, 'Incomplete fixed explanation set')
                matches=[m for r in records for m in r['matches']]
                for item in records:
                    require(abs(sum(m['contribution'] for m in item['matches'])-item['score'])<1e-12, 'Explanation contributions disagree with score')
                scores=[m['W'] for m in matches]
                rows.append({'stage':stage,'method':name,'rank':rank,'sampled_pairs':len(records),
                    'accepted_region_pairs':len(matches),'empty_point_set_region_pairs':sum(min(m['query_points'],m['candidate_points'])==0 for m in matches),
                    'mean_structure_W':float(np.mean(scores)) if scores else None,
                    'mean_BBP':float(np.mean([m['BBP'] for m in matches])) if matches else None,
                    'mean_count_penalty':float(np.mean([m['count_penalty'] for m in matches])) if matches else None,
                    'mean_C':float(np.mean([m['C'] for m in matches])) if matches else None,
                    'mean_A':float(np.mean([m['A'] for m in matches])) if matches else None,
                    'scope':'fixed 37-class queries at ranks1/10/100; candidates may differ by method; not all-query correspondence statistics or causal evidence'})
    return rows


def self_test_valid_identity():
    """Resealed synthetic wrong-stage/context sources must still be rejected."""
    import tempfile
    def fixture(root, mutation):
        grid=frozen_grid();context={'grid':grid};ctx=digest(context);indexes={};evaluations={}
        for stage in ('valid','cross_valid'):
            specs=grid if stage=='valid' else {**grid,WBBS:{'family':'global_wbbs'}}
            results={};evaluations[stage]={}
            for name,spec in specs.items():
                folder=root/stage/name;folder.mkdir(parents=True)
                sig=digest({'context':ctx,'stage':stage,'variant':name,'spec':spec})
                child={'signature':sig,'evaluation_signature':sig,'context_signature':ctx,
                    'stage':stage,'split':'valid','variant':name,'spec':spec,'num_queries':500,
                    'gallery_count':3599,'ks':list(KS),'tie_seed':2026,
                    'retrieval_protocol_signature':CROSS_VALID_PROTOCOL if stage=='cross_valid' else BASE_PROTOCOL,
                    'base_protocol_signature':BASE_PROTOCOL,
                    'candidate_rule':'all_gallery_except_query_and_exact_label_combination' if stage=='cross_valid' else 'all_gallery_except_query',
                    'primary_variant':None,'best_regional_wbbs':None,'frozen_selection_sha256':None}
                if stage=='valid' and name==BASE and mutation.startswith('child_'):
                    key=mutation.removeprefix('child_');child[key]={'split':'test','stage':'cross_test',
                        'context_signature':'wrong','spec':{'family':'wrong'},'num_queries':499,
                        'retrieval_protocol_signature':'wrong'}[key]
                metrics={'Coverage-nDCG@10':.9 if name==BASE else .99 if name==WBBS else .5}
                write(folder/'evaluation_manifest.json',child)
                write(folder/'coverage_metrics.json',{'split':'valid','num_queries':500,'ks':list(KS),
                    'metric_version':'coverage_v1','metadata':child,'metrics':metrics})
                marker={'status':'completed','signature':sig,'files':{n:sha(folder/n)
                    for n in ('evaluation_manifest.json','coverage_metrics.json')}}
                if stage=='valid' and name==BASE and mutation=='marker_signature':marker['signature']='wrong'
                write(folder/'completed.json',marker)
                results[name]={'coverage_metrics':metrics,'completed_sha256':sha(folder/'completed.json')}
                evaluations[stage][name]=sha(folder/'completed.json')
            index={'context_signature':ctx,'stage':stage,'split':'valid','specs':specs,'results':results}
            if stage=='valid' and mutation in ('index_stage','index_context_signature'):
                index[mutation.removeprefix('index_')]='wrong'
            path=root/('evaluation_index_'+stage+'.json');write(path,index);indexes[stage]=sha(path)
        values={n:r['coverage_metrics']['Coverage-nDCG@10'] for n,r in results.items()}
        def tie(n):
            if n==BASE:return (0,0.,0,0.,n)
            v=grid[n];return (1,abs(v['cnn_weight']-.4),v['point_mode']!='all_points',v['count_alpha'],n)
        controls=[n for n,v in grid.items() if v.get('cnn_weight')==.4]
        selection={'context_signature':ctx,'winner':BASE,'best_regional_wbbs':'all_bb_a04',
            'fixed_controls':controls,'selected_variants':controls,'selected_specs':{n:grid[n] for n in controls},
            'selection_values':values,'valid_order':sorted(grid,key=lambda n:(-values[n],tie(n))),
            'test_reselection':False,**{k+'_index_sha256':v for k,v in indexes.items()}}
        write(root/'selection.json',selection)
        complete={'status':'completed','context_signature':ctx,'selection_sha256':sha(root/'selection.json'),
            'indexes':indexes,'evaluations':evaluations}
        if mutation.startswith('complete_'):complete[mutation.removeprefix('complete_')]='wrong'
        write(root/'valid_completed.json',complete)
        write(root/'completed.json',{'status':'completed','signature':ctx,
            'files':{str(f.relative_to(root)):sha(f) for f in root.rglob('*.json')}})
        return selection,context
    mutations=['complete_status','complete_context_signature','complete_selection_sha256',
        'index_stage','index_context_signature','marker_signature','child_split','child_stage',
        'child_context_signature','child_spec','child_num_queries','child_retrieval_protocol_signature']
    for mutation in ['',*mutations]:
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);selection,context=fixture(root,mutation)
            try:independently_check_selection(root,selection,context,Binding())
            except ValueError:
                require(bool(mutation),'Valid synthetic provenance unexpectedly failed')
            else:require(not mutation,'Resealed invalid provenance was accepted: '+mutation)


def self_test():
    patterns=['00','00','01','01','01','10']
    values=np.array([[.1,np.nan],[.1,np.nan],[.1,.2],[.1,.2],[.1,.2],[.1,.2]])
    draws=bootstrap(values,patterns,2000,2026)
    np.testing.assert_allclose(draws,np.broadcast_to([.1,.2],draws.shape),rtol=0,atol=1e-14)
    np.testing.assert_array_equal(draws,bootstrap(values,patterns,2000,2026))
    empty=bootstrap(np.full((6,1),np.nan),patterns,20,2026)
    require(np.isnan(empty).all(),'Undefined intervals must not become zero')
    require(len(frozen_grid())==13 and len([s for s in frozen_grid().values() if s.get('cnn_weight')==.4])==4,'Unexpected fixed grid')
    self_test_valid_identity()
    print('Self-test passed: resealed valid identity corruption rejected; paired stratified means, shared finite masks, undefined endpoints, fixed grid; no experiment files read')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=Path.cwd())
    parser.add_argument('--run-root',type=Path)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--baseline-ordinary',type=Path)
    parser.add_argument('--baseline-cross',type=Path)
    parser.add_argument('--wbbs-cross',type=Path)
    parser.add_argument('--wbbs-exact-root',type=Path)
    parser.add_argument('--wbbs-coverage-root',type=Path)
    parser.add_argument('--wbbs-proof-root',type=Path)
    parser.add_argument('--repetitions',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--plan',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args(argv)
    require(args.repetitions==2000 and args.seed==2026,'Frozen analysis uses 2000 resamples and seed2026')
    if args.self_test:self_test();return
    if args.plan:
        print(json.dumps({'point_estimate_ks':KS,'point_estimate_groups':'all/class_macro/single/mixed/1-4 labels/37 original classes',
            'confidence_endpoints':CORE,'ci_groups':CI_GROUPS,
            'comparisons':'all frozen selected regional WBBS test items versus original J and whole-map WBBS',
            'primary_inference':'valid-frozen best regional WBBS versus J/WBBS, cross C-nDCG@10 and FullCoverage P@10',
            'limitations':LIMITATIONS},indent=2));return
    require(args.run_root is not None,'--run-root is required after formal completion')
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).replace(microsecond=0)
    repo=args.repo_root.resolve();run=(repo/args.run_root).resolve();binding=Binding()
    root_marker=binding.sealed(run,['run_manifest.json','selection.json','valid_completed.json','inputs_test.json','status.json'])
    manifest=binding.signed(run/'run_manifest.json');selection=binding.signed(run/'selection.json')
    status=read(run/'status.json');context=manifest['context'];ctx=digest(context)
    require(manifest['version']=='f_region_wbbs_v1' and status['status']=='completed'
        and root_marker['signature']==selection['context_signature']==ctx,'The complete study must be sealed before test analysis')
    require(datetime.fromisoformat(manifest['started_at'])<=datetime.fromisoformat(selection['frozen_at'])<=datetime.fromisoformat(status['completed_at']), 'Invalid run/selection/completion time order')
    independently_check_selection(run,selection,context,binding)
    test_inputs=read(run/'inputs_test.json')
    require(test_inputs['selection_sha256']==sha(run/'selection.json') and test_inputs['context_signature']==ctx,'Test lacks its frozen valid-selection binding')
    ids,labels,patterns,groups=label_data(repo,binding)
    require(len(set(patterns))==37,'Expected the original37 multi-hot combinations')
    for stage in ('test','cross_test'):
        for name in selection['selected_variants']:
            check_new_identity(run/stage/name,stage,name,selection,ctx,binding)
            child=read(run/stage/name/'evaluation_manifest.json')
            require(datetime.fromisoformat(child['started_at'])>=datetime.fromisoformat(selection['frozen_at']), 'Test child predates frozen selection')
    loaded={stage:{name:load_evaluation(run/stage/name,stage,ids,labels,binding)
                   for name in selection['selected_variants']} for stage in ('test','cross_test')}
    oldbase=(repo/(args.baseline_ordinary or Path('artificial/f_region_area_fusion_test_20261120T160426+0800/test/add_a0p4_b0p25'))).resolve()
    crossbase=(repo/(args.baseline_cross or Path('artificial/f_region_j_cross_test_20261120T202638+0800/test/add_a0p4_b0p25'))).resolve()
    crosswbbs=(repo/(args.wbbs_cross or Path('artificial/fgj_cross_test_20261120T193642+0800/test/WBBS'))).resolve()
    for folder in (oldbase,crossbase,crosswbbs):
        expected=test_inputs['source_binding'].get(str(folder/'completed.json'))
        require(expected is not None,'Historical test reference absent from frozen test source binding')
        binding.bind(folder/'completed.json',expected)
    loaded['test'][BASE]=load_evaluation(oldbase,'test',ids,labels,binding)
    loaded['cross_test'][BASE]=load_evaluation(crossbase,'cross_test',ids,labels,binding)
    loaded['cross_test'][WBBS]=load_evaluation(crosswbbs,'cross_test',ids,labels,binding)
    proofroot=(repo/(args.wbbs_proof_root or Path('artificial/gj_fusion_analysis_20261120T190052+0800'))).resolve()
    binding.sealed(proofroot,['analysis_manifest.json']);proof=binding.signed(proofroot/'analysis_manifest.json')
    wbbs=proof['wbbs_source_validation']
    require(wbbs['status']=='verified' and wbbs['retrieval_protocol_signature']==BASE_PROTOCOL
        and wbbs['method']=='wbbs' and wbbs['variant']=='global','Missing independent WBBS same-ranking provenance proof')
    exact=(repo/(args.wbbs_exact_root or Path('../compared_methods/artifacts/formal_exact_20261109_120738/evaluation/wbbs_test_r2'))).resolve()
    coverage=(repo/(args.wbbs_coverage_root or Path('../compared_methods/artifacts/coverage_v1_test_20261109_191909/wbbs'))).resolve()
    for folder,names in [(exact,['test/per_query.csv','test/metrics.json']),
                         (coverage,['test/coverage_per_query.csv','test/coverage_metrics.json'])]:
        binding.sealed(folder,names)
        for name in ['completed.json',*names]:
            path=folder/name;binding.bind(path,proof['source_hashes'][str(path)])
    loaded['test'][WBBS]=load_family(exact/'test','exact',ids,labels,False,binding)
    loaded['test'][WBBS].update(load_family(coverage/'test','coverage',ids,labels,False,binding))
    inherited={**context['source_binding'],**test_inputs['source_binding']}
    models={p:s for p,s in inherited.items() if Path(p).suffix.lower() in ('.pt','.pth','.ckpt')}
    require(bool(models),'Missing frozen source-model provenance')
    for path,expected in models.items():binding.bind(path,expected)
    summary,differences,pairs,aliases=summaries(loaded,selection,groups,patterns)
    ci=intervals(loaded,selection,pairs,groups,patterns,args.repetitions,args.seed)
    point_rows=point_diagnostics(run,ids,groups,binding)
    examples=explanation_summary(run,selection,binding)
    output=(repo/(args.output_root or Path('artificial/f_region_wbbs_analysis_'+stamp.strftime('%Y%m%dT%H%M%S%z')))).resolve()
    require(not output.exists() and not output.is_relative_to(run),'Analysis output must be a new independent directory')
    output.mkdir(parents=True,exist_ok=False);shutil.copyfile(__file__,output/'analysis_script.py')
    for name,rows in [('summary.csv',summary),('paired_differences.csv',differences),('paired_bootstrap.csv',ci),
                      ('point_diagnostics.csv',point_rows),('sampled_match_explanations.csv',examples)]:write_csv(output/name,rows)
    binding.recheck()
    evidence={'schema_version':1,'analysis_version':'f_region_wbbs_analysis_v1',
        'run_id':stamp.strftime('%Y%m%dT%H%M%S%z')+'__f_region_wbbs_analysis','started_at':stamp.isoformat(),
        'finished_at':datetime.now(ZoneInfo('Asia/Shanghai')).replace(microsecond=0).isoformat(),
        'run_root':str(run),'source_context_signature':ctx,'frozen_selection_signature':selection['signature'],
        'primary':selection['winner'],'best_regional_wbbs':selection['best_regional_wbbs'],
        'fixed_controls':selection['fixed_controls'],'selected_variants':selection['selected_variants'],
        'comparison_aliases':aliases,'unique_pairs':pairs,'source_hashes':binding.files,'frozen_model_hashes':models,
        'inherited_scientific_source_binding':inherited,
        'source_verification_scope':'Directly consumed report/cache/audit/manifest/model files SHA checked and rechecked; remaining dependencies inherit completed-run binding. No rankings or regional formulas recomputed.',
        'query_ids_sha256':digest(ids.tolist()),'query_label_sha256':digest(labels.tolist()),
        'repetitions':args.repetitions,'seed':args.seed,'confidence_endpoints':CORE,'ci_groups':CI_GROUPS,
        'primary_inference':'valid-frozen best regional WBBS versus J/WBBS on cross Coverage-nDCG@10 and FullCoveragePrecision@10; fixed controls are mechanism comparisons',
        'units':'summary/differences are ratios; bootstrap includes ratios plus explicit *_pp percentage-point fields',
        'sampling':'same paired query draws for all contrasts/endpoints within each stage/group; replacement within raw complete 8-bit label combination; common per-metric defined-query mask',
        'interval':'pointwise95% percentile; numpy quantile(method=linear)','limitations':LIMITATIONS,
        'mechanism_scope':'whole-test per-mode point counts and actual BBP self-count diagonal; fixed37-class match examples at ranks1/10/100, candidates may differ by method; descriptive not causal',
        'test_parameter_search':False,'numpy':np.__version__}
    write(output/'analysis_manifest.json',{**evidence,'signature':digest(evidence)})
    write(output/'completed.json',{'status':'completed','signature':digest(evidence),
        'files':{str(p.relative_to(output)):sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    print(json.dumps({'status':'completed','output':str(output),'primary':selection['winner'],
        'best_regional_wbbs':selection['best_regional_wbbs'],'summary_rows':len(summary),
        'comparison_rows':len(differences),'bootstrap_rows':len(ci)},indent=2))


if __name__=='__main__':main()
