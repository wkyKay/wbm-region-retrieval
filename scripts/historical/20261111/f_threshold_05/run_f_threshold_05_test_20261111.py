"""Fixed-checkpoint F threshold experiment; original source run stays read-only."""
import argparse
import copy
import csv
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

p = argparse.ArgumentParser()
for name in ('project', 'compared', 'source', 'output'):
    p.add_argument('--' + name, required=True)
a = p.parse_args()
project, compared, source, output = [Path(getattr(a, n)).resolve() for n in ('project', 'compared', 'source', 'output')]
sys.path[:0] = [str(project), str(compared)]
from wbm_retrieval.common import read_json, write_json, file_hash, digest
from wbm_retrieval.data import Protocol, source_fingerprint
from wbm_retrieval.training import load_checkpoint, parameter_hash
from wbm_retrieval.matching import map_similarity
from wbm_retrieval.diagnostics import batched_map_scores
from wbm_retrieval.exact_metrics import order_scores
from experiments.proposed._shared import ef_suite
from experiments.proposed._shared.ef_training import run_signature
from shared.coverage_metrics import evaluate_rankings_file

assert not output.exists(), 'Output must be fresh'
assert output != source and source not in output.parents and output not in source.parents
output.mkdir(parents=True)
started = time.monotonic()
torch.set_num_threads(4)
manifest = read_json(source / 'suite_manifest.json')
prepared = read_json(source / 'cache/prepared.json')
assert prepared['signature'] == manifest['signature'] == digest({k: v for k, v in manifest.items() if k != 'signature'})
config = copy.deepcopy(read_json(source / 'seed_2026/F/run_manifest.json')['config'])
assert config['matching'] == manifest['settings']['matching']
assert config['matching']['shape_threshold'] == config['matching']['score_threshold'] == 0
assert config['matching']['lambda_pos'] == config['matching']['lambda_scale'] == 0
assert source_fingerprint(config) == manifest['sources']
for name, h in manifest['code'].items():
    assert file_hash(project / name) == h, name
cache_names = ('cache/labels.npy', 'cache/test_regions.npz', 'cache/test_F.npy')
for name in cache_names:
    assert file_hash(source / name) == prepared['files'][name], name
checkpoint_path = source / 'seed_2026/F/best_nce.pt'
selection = read_json(source.parent / 'selection_before_test.json')
assert file_hash(checkpoint_path) == selection['sha256']
checkpoint = load_checkpoint(checkpoint_path)
assert checkpoint['signature'] == run_signature(manifest['signature'], config['seed'], 'F')
assert checkpoint['epoch'] == selection['training_summary']['best_nce_epoch'] == 100
baseline_settings = copy.deepcopy(config['matching'])
config['matching'].update(shape_threshold=0.5, score_threshold=0.5)
config['output_dir'] = str(source)  # evaluate() has a separate explicit destination.
config['device'] = 'cuda:0'
protocol = Protocol(config)
assert len(protocol.queries) == 7199 and protocol.candidates.candidates_per_query == 7198
for name in ('test_queries.csv', 'test_gallery.csv', 'metadata.json'):
    assert file_hash(project / 'protocol/retrieval_v1' / name) == file_hash(compared / 'protocol/retrieval_v1' / name)
with np.load(source / 'cache/test_regions.npz') as z:
    meta = dict(z)
np.testing.assert_array_equal(meta['map_ids'], protocol.candidates.gallery)
labels = np.load(source / 'cache/labels.npy')
spec = manifest['suite']['variants']['F']
model = ef_suite.build_model(config, spec).to(config['device'])
model.load_state_dict(checkpoint['model'])
model.requires_grad_(False)
model_hash = parameter_hash(model)
baseline_test = source / 'seed_2026/F/test/best_nce'
paths = [source / 'suite_manifest.json', source / 'cache/prepared.json', checkpoint_path,
         source.parent / 'selection_before_test.json', baseline_test / 'rankings.csv', baseline_test / 'metrics.json']
paths += [source / n for n in cache_names] + [project / n for n in manifest['code']]
paths += [compared / 'shared/coverage_metrics.py', Path(__file__).resolve()]
snapshot = {str(f): {'sha256': file_hash(f), 'mtime_ns': f.stat().st_mtime_ns} for f in paths}
run = {'method': 'F_best_nce_threshold_0.5', 'training_required': False, 'smoke': False,
       'experiment_role': 'user-specified post-hoc test comparison; no validation selection',
       'source_run': str(source), 'checkpoint': str(checkpoint_path), 'checkpoint_sha256': selection['sha256'],
       'epoch': checkpoint['epoch'], 'model_parameter_sha256': model_hash,
       'config': config, 'baseline_matching': baseline_settings, 'input_spec': spec,
       'query_count': 7199, 'gallery_count': 7199, 'candidates_per_query': 7198,
       'candidate_rule': 'all_gallery_except_query', 'ranking_limit': 500,
       'retrieval_protocol_signature': protocol.retrieval.signature,
       'cache_signature': prepared['signature'], 'snapshot': snapshot}
write_json(output / 'config.json', config)
write_json(output / 'run_manifest.json', run)

# Capture the evaluator's unchanged feature extraction and audit it before full retrieval.
original_embeddings = ef_suite.embeddings
def audited_embeddings(*args, **kwargs):
    features = original_embeddings(*args, **kwargs)
    np.save(output / 'test_features.npy', features)
    selected = np.unique(np.linspace(0, len(meta['map_ids']) - 1, 128, dtype=int))
    selected_ids = set(int(meta['map_ids'][i]) for i in selected)
    historical = {q: [] for q in selected_ids}
    with (baseline_test / 'rankings.csv').open() as f:
        for row in csv.DictReader(f):
            q = int(row['query_id'])
            if q in historical:
                historical[q].append((int(row['candidate_id']), float(row['similarity_score'])))
    width = int(np.diff(meta['offsets']).max())
    padded = np.full((len(meta['map_ids']), width), -1, dtype=np.int64)
    groups = {}
    for i, q in enumerate(meta['map_ids']):
        start, end = meta['offsets'][i:i+2]
        padded[i, :end-start] = np.arange(start, end)
        if i in selected:
            groups[int(q)] = {'embedding': features[start:end], 'area': meta['area'][start:end],
                             'position': np.zeros((end-start, 2)), 'scale': np.ones(end-start)}
    max_baseline_error = max_scalar_error = 0.
    scalar_pairs = 0
    for n, i in enumerate(selected):
        q = int(meta['map_ids'][i])
        start, end = meta['offsets'][i:i+2]
        similarities = features[start:end] @ features.T
        scores = batched_map_scores(similarities, padded, meta['area'][start:end], baseline_settings)
        keep = meta['map_ids'] != q
        ids = meta['map_ids'][keep]
        scores = scores[keep]
        order = order_scores(ids, scores, q, config['evaluation']['tie_seed'])[:500]
        old_ids, old_scores = zip(*historical[q])
        np.testing.assert_array_equal(ids[order], old_ids)
        np.testing.assert_allclose(scores[order], old_scores, atol=2e-6, rtol=0)
        max_baseline_error = max(max_baseline_error, float(np.max(np.abs(scores[order] - old_scores))))
        if n < 16:
            actual = batched_map_scores(similarities, padded[selected], meta['area'][start:end], config['matching'])
            expected = np.array([map_similarity(groups[q], groups[int(meta['map_ids'][j])], config['matching']) for j in selected])
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
            max_scalar_error = max(max_scalar_error, float(np.max(np.abs(actual-expected))))
            scalar_pairs += len(expected)
    audit = {'baseline_queries_reproduced': len(selected), 'baseline_top500_rows': len(selected)*500,
             'baseline_max_score_error': max_baseline_error, 'threshold_scalar_batched_pairs': scalar_pairs,
             'threshold_scalar_batched_max_error': max_scalar_error}
    write_json(output / 'preflight_verification.json', audit)
    print('Feature and matching audit passed: ' + json.dumps(audit), flush=True)
    return features
ef_suite.embeddings = audited_embeddings
try:
    result = ef_suite.evaluate(config, manifest['suite'], 'test', output / 'test', model=model, spec=spec,
                               device=config['device'], rankings=True, bootstrap=config['evaluation']['bootstrap_samples'])
finally:
    ef_suite.embeddings = original_embeddings
assert parameter_hash(model) == model_hash
classes = sorted({tuple(labels[i]) for i in protocol.ids['train']})
mapping = {c: i for i, c in enumerate(classes)}
class_ids = np.full(len(labels), -1, dtype=int)
for q in protocol.ids['test']:
    class_ids[q] = mapping[tuple(labels[q])]
coverage = evaluate_rankings_file(SimpleNamespace(labels=labels, class_ids=class_ids), protocol.candidates,
    output / 'test/rankings.csv', config['evaluation']['ks'], output / 'test', split_name='test',
    metadata={'method': run['method'], 'matching': config['matching'], 'checkpoint_sha256': selection['sha256'],
              'ranking_source_sha256': file_hash(output / 'test/rankings.csv'), 'candidate_rule': 'all_gallery_except_query',
              'ranking_limit': 500, 'smoke': False, 'retrieval_protocol_signature': protocol.retrieval.signature})
for name, state in snapshot.items():
    f = Path(name)
    assert file_hash(f) == state['sha256'] and f.stat().st_mtime_ns == state['mtime_ns'], name
write_json(output / 'completed.json', {'method': run['method'], 'split': 'test', 'smoke': False,
    'query_count': 7199, 'candidates_per_query': 7198, 'source_files_unchanged': True, 'model_unchanged': True,
    'seconds': time.monotonic()-started,
    'files': {str(f.relative_to(output)): file_hash(f) for f in (output / 'test').rglob('*') if f.is_file()}})
print(json.dumps({'exact': result['metrics'], 'coverage': coverage['metrics']}, indent=2), flush=True)
