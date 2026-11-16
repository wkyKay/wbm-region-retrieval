"""Select a region threshold using frozen valid queries, then evaluate test once."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

p = argparse.ArgumentParser()
for name in ('project', 'compared', 'source', 'feature-run', 'output'):
    p.add_argument('--' + name, required=True)
a = p.parse_args()
project, compared, source, feature_run, output = [Path(getattr(a, n)).resolve()
    for n in ('project', 'compared', 'source', 'feature_run', 'output')]
sys.path[:0] = [str(project), str(compared)]
from wbm_retrieval.common import read_json, write_json, file_hash, digest
from wbm_retrieval.data import Protocol, source_fingerprint
from wbm_retrieval.training import load_checkpoint, parameter_hash
from wbm_retrieval.matching import map_similarity
from wbm_retrieval.diagnostics import batched_map_scores
from experiments.proposed._shared import ef_suite
from experiments.proposed._shared.ef_training import run_signature
from shared.coverage_metrics import evaluate_rankings_file

THRESHOLDS = (0.0, 0.75, 0.80, 0.85, 0.90, 0.95)
assert not output.exists(), 'A fresh output directory is required'
assert source not in output.parents and feature_run not in output.parents
output.mkdir(parents=True)
started = time.monotonic()
torch.set_num_threads(4)
manifest = read_json(source / 'suite_manifest.json')
prepared = read_json(source / 'cache/prepared.json')
assert prepared['signature'] == manifest['signature'] == digest({k:v for k,v in manifest.items() if k != 'signature'})
config = copy.deepcopy(read_json(source / 'seed_2026/F/run_manifest.json')['config'])
assert config['matching'] == manifest['settings']['matching']
assert config['matching']['shape_threshold'] == config['matching']['score_threshold'] == 0
assert config['matching']['lambda_pos'] == config['matching']['lambda_scale'] == 0
assert source_fingerprint(config) == manifest['sources']
for name, h in manifest['code'].items():
    assert file_hash(project / name) == h, name
cache_names = ['cache/labels.npy', 'cache/valid_regions.npz', 'cache/valid_F.npy']
for name in cache_names:
    assert file_hash(source / name) == prepared['files'][name], name
checkpoint_path = source / 'seed_2026/F/best_nce.pt'
original_selection = read_json(source.parent / 'selection_before_test.json')
checkpoint_hash = file_hash(checkpoint_path)
assert checkpoint_hash == original_selection['sha256']
checkpoint = load_checkpoint(checkpoint_path)
assert checkpoint['signature'] == run_signature(manifest['signature'], config['seed'], 'F')
assert checkpoint['epoch'] == 100
config['output_dir'], config['device'] = str(source), 'cuda:0'
protocol = Protocol(config)
assert len(protocol.retrieval.valid) == 500 and protocol.retrieval.valid.candidates_per_query == 3598
for name in ('valid_queries.csv', 'valid_gallery.csv', 'test_queries.csv', 'test_gallery.csv', 'metadata.json'):
    assert file_hash(project / 'protocol/retrieval_v1' / name) == file_hash(compared / 'protocol/retrieval_v1' / name)
spec = manifest['suite']['variants']['F']
model = ef_suite.build_model(config, spec).to(config['device'])
model.load_state_dict(checkpoint['model'])
model.requires_grad_(False)
model.eval()
model_hash = parameter_hash(model)
labels = np.load(source / 'cache/labels.npy')
classes = sorted({tuple(labels[i]) for i in protocol.ids['train']})
mapping = {c:i for i,c in enumerate(classes)}
class_ids = np.full(len(labels), -1, dtype=int)
for split in ('valid', 'test'):
    for q in protocol.ids[split]:
        class_ids[q] = mapping[tuple(labels[q])]
bundle = SimpleNamespace(labels=labels, class_ids=class_ids)
paths = [source / 'suite_manifest.json', source / 'cache/prepared.json', source / 'valid_pool.json', checkpoint_path,
         source / 'seed_2026/F/valid/best_nce/rankings.csv', Path(__file__).resolve(), compared / 'shared/coverage_metrics.py']
paths += [source / n for n in cache_names] + [project / n for n in manifest['code']]
snapshot = {str(f): {'sha256':file_hash(f), 'mtime_ns':f.stat().st_mtime_ns} for f in paths}
run = {'method':'F_best_nce_valid_selected_threshold', 'training_required':False, 'smoke':False,
       'threshold_grid':list(THRESHOLDS), 'selection_metric':'Coverage-nDCG@10',
       'tie_rule':'smaller threshold if the unrounded metric is exactly equal',
       'source_run':str(source), 'checkpoint':str(checkpoint_path), 'checkpoint_sha256':checkpoint_hash,
       'epoch':100, 'model_parameter_sha256':model_hash, 'baseline_config':config, 'input_spec':spec,
       'retrieval_protocol_signature':protocol.retrieval.signature, 'snapshot':snapshot,
       'test_policy':'Evaluate only the valid-selected threshold; freeze selection before test',
       'context':'Threshold-0.5 test and test feature diagnostics had already been inspected in preceding work.'}
write_json(output / 'run_manifest.json', run)

valid_features = ef_suite.embeddings(model, source / 'cache/valid_F.npy', config, config['device'])
np.save(output / 'valid_features.npy', valid_features)
with np.load(source / 'cache/valid_regions.npz') as z:
    meta = dict(z)
np.testing.assert_array_equal(meta['map_ids'], protocol.retrieval.valid.gallery)
selected = np.unique(np.linspace(0, len(meta['map_ids'])-1, 64, dtype=int))
padded = np.full((len(selected), int(np.diff(meta['offsets']).max())), -1, dtype=int)
groups = {}
for n,i in enumerate(selected):
    start,end = meta['offsets'][i:i+2]
    padded[n,:end-start] = np.arange(start,end)
    groups[int(i)] = {'embedding':valid_features[start:end], 'area':meta['area'][start:end],
                     'position':np.zeros((end-start,2)), 'scale':np.ones(end-start)}

def verify_selection_score(target, pool):
    """Independent coverage-nDCG@10 recomputation before the selection decision."""
    ranks = {}
    with (target / 'rankings.csv').open() as f:
        for row in csv.DictReader(f):
            if int(row['rank']) <= 10:
                ranks.setdefault(int(row['query_id']), []).append(int(row['candidate_id']))
    assert set(ranks) == set(pool)
    discounts = 1 / np.log2(np.arange(2,12))
    values = []
    for q in pool:
        full = labels[pool[q]] @ labels[q] / labels[q].sum()
        ideal = np.sort(full)[::-1][:10] @ discounts
        got = labels[ranks[q]] @ labels[q] / labels[q].sum()
        values.append(float(got @ discounts / ideal))
    value = float(np.mean(values))
    expected = read_json(target / 'coverage_metrics.json')['metrics']['Coverage-nDCG@10']
    np.testing.assert_allclose(value, expected, atol=1e-12, rtol=0)
    return value

def evaluate_one(split, threshold, target, features):
    settings = copy.deepcopy(config)
    settings['matching'].update(shape_threshold=threshold, score_threshold=threshold)
    pool = getattr(protocol.retrieval, split)
    original = ef_suite.embeddings
    def cached(model_arg, patch_path, cfg, device):
        assert Path(patch_path) == source / f'cache/{split}_F.npy'
        return features
    ef_suite.embeddings = cached
    try:
        exact = ef_suite.evaluate(settings, manifest['suite'], split, target, model=model, spec=spec,
                                  device=settings['device'], rankings=True, bootstrap=settings['evaluation']['bootstrap_samples'])
    finally:
        ef_suite.embeddings = original
    write_json(target / 'config.json', settings)
    coverage = evaluate_rankings_file(bundle, pool, target / 'rankings.csv', settings['evaluation']['ks'], target,
        split_name=split, metadata={'method':run['method'], 'matching':settings['matching'],
        'checkpoint_sha256':checkpoint_hash, 'ranking_source_sha256':file_hash(target / 'rankings.csv'),
        'candidate_rule':'all_gallery_except_query', 'ranking_limit':500, 'smoke':False,
        'retrieval_protocol_signature':protocol.retrieval.signature})
    return exact, coverage

results = []
for threshold in THRESHOLDS:
    settings = {**config['matching'], 'shape_threshold':threshold, 'score_threshold':threshold}
    max_error = 0.
    for i in selected[:16]:
        start,end = meta['offsets'][i:i+2]
        actual = batched_map_scores(valid_features[start:end] @ valid_features.T, padded, meta['area'][start:end], settings)
        expected = np.array([map_similarity(groups[int(i)],groups[int(j)],settings) for j in selected])
        np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
        max_error = max(max_error,float(np.max(np.abs(actual-expected))))
    target = output / 'valid' / f'threshold_{int(round(threshold*100)):03d}'
    exact, coverage = evaluate_one('valid', threshold, target, valid_features)
    verified = verify_selection_score(target, protocol.retrieval.valid)
    if threshold == 0:
        assert file_hash(target / 'rankings.csv') == file_hash(source / 'seed_2026/F/valid/best_nce/rankings.csv')
    result = {'threshold':threshold, 'Coverage-nDCG@10':coverage['metrics']['Coverage-nDCG@10'],
              'independent_selection_score':verified, 'exact':exact['metrics'], 'coverage':coverage['metrics'],
              'directory':str(target.relative_to(output)), 'coverage_metrics_sha256':file_hash(target / 'coverage_metrics.json'),
              'scalar_batched_parity_pairs':1024, 'max_abs_parity_error':max_error}
    results.append(result)
    print(f'VALID threshold={threshold:.2f}: Coverage-nDCG@10={result["Coverage-nDCG@10"]:.12f}',flush=True)
write_json(output / 'valid_results.json', results)
best = max(results, key=lambda r:(r['Coverage-nDCG@10'],-r['threshold']))
selection = {'selected_threshold':best['threshold'], 'selection_metric':'Coverage-nDCG@10',
             'selected_valid_score':best['Coverage-nDCG@10'], 'grid_results':results, 'tie_rule':run['tie_rule'],
             'checkpoint_sha256':checkpoint_hash, 'retrieval_protocol_signature':protocol.retrieval.signature,
             'selected_at_utc':datetime.now(timezone.utc).isoformat(), 'test_has_started':False,
             'independent_valid_selection_score_verification':True}
write_json(output / 'selection_before_test.json', selection)
selection_hash = file_hash(output / 'selection_before_test.json')
print(f'SELECTED threshold={best["threshold"]:.2f}; selection frozen before test',flush=True)

# Only after the valid decision, load the previously verified test embeddings.
feature_manifest = read_json(feature_run / 'run_manifest.json')
assert feature_manifest['checkpoint_sha256'] == checkpoint_hash
assert feature_manifest['cache_signature'] == prepared['signature']
feature_path = feature_run / 'test_features.npy'
assert file_hash(feature_path) == read_json(feature_run / 'bundle_hashes.json')['test_features.npy']
assert file_hash(source / 'cache/test_regions.npz') == prepared['files']['cache/test_regions.npz']
test_features = np.load(feature_path, mmap_mode='r')
with np.load(source / 'cache/test_regions.npz') as z:
    np.testing.assert_array_equal(z['map_ids'], protocol.retrieval.test.gallery)
    assert len(test_features) == z['offsets'][-1]
write_json(output / 'test_feature_source.json', {'path':str(feature_path), 'sha256':file_hash(feature_path),
    'checkpoint_sha256':checkpoint_hash, 'cache_signature':prepared['signature']})
exact, coverage = evaluate_one('test', best['threshold'], output / 'test', test_features)
assert parameter_hash(model) == model_hash
assert file_hash(output / 'selection_before_test.json') == selection_hash
for name,state in snapshot.items():
    f=Path(name)
    assert file_hash(f) == state['sha256'] and f.stat().st_mtime_ns == state['mtime_ns'],name
write_json(output / 'completed.json', {'method':run['method'], 'split':'test', 'smoke':False,
    'selected_threshold':best['threshold'], 'query_count':7199, 'candidates_per_query':7198,
    'valid_queries_per_threshold':500, 'valid_candidates_per_query':3598,
    'source_files_unchanged':True, 'model_unchanged':True, 'selection_unchanged':True,
    'selection_sha256':selection_hash, 'seconds':time.monotonic()-started,
    'files':{str(f.relative_to(output)):file_hash(f) for f in output.rglob('*') if f.is_file()}})
print(json.dumps({'selected_threshold':best['threshold'],'test_exact':exact['metrics'],'test_coverage':coverage['metrics']},indent=2),flush=True)
