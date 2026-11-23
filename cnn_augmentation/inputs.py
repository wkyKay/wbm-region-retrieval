"""Read-only baseline, original cache and deduplicated protocol adapters."""
from pathlib import Path
import numpy as np

from f_representation_audit.inputs import load_inputs, verify_protection
from wbm_retrieval.common import file_hash, read_json
from .state import ROOT, record, signed, run_context


BASELINE_RUN = ROOT/'results/20261123T105220+0800__cnn_refinement'

def _bind_baseline_run(binding):
    marker = read_json(BASELINE_RUN/'completed.json')
    if marker.get('status') != 'completed':
        raise ValueError('Historical baseline evaluation must be sealed')
    names = ['run_manifest.json', 'frozen_plan.json']
    for split in ('valid', 'test'):
        names += [f'inputs_{split}.json', f'evaluation/baseline/evaluation_index_{split}.json',
                  f'evaluation/baseline/model_context_{split}.json', f'evaluation/baseline/formula_audit_{split}.json']
    for name in names:
        path = BASELINE_RUN/name
        actual = file_hash(path)
        if actual != marker['files'].get(name):
            raise ValueError('Historical baseline evaluation binding changed: '+name)
        binding[str(path)] = actual
    binding[str(BASELINE_RUN/'completed.json')] = file_hash(BASELINE_RUN/'completed.json')


def load(output=None):
    inputs = load_inputs(ROOT, ROOT/'artifacts/f_segmentation_v1/cc8')
    _bind_baseline_run(inputs['protection'])
    if output is not None:
        manifest = run_context(output)
        if manifest['source_binding'] != inputs['protection']:
            raise ValueError('Baseline/protocol/source binding changed since launch')
        inputs['refinement_context_signature'] = manifest['signature']
        inputs['refinement_plan_signature'] = manifest['plan_signature']
        inputs['plan_signature'] = manifest['plan_signature']
    return inputs


def evaluation_data(split, inputs, output):
    from paper_c_ablation.inputs import load as load_evaluation
    from f_region_evidence.inputs import bind_sealed
    from f_region_wbbs.protocol import BASE_PROTOCOL_SIGNATURE, FROZEN, PROTOCOL, RULE
    if split not in ('valid', 'test'):
        raise ValueError('Formal retrieval split required')
    data = load_evaluation(split)
    if split == 'valid':
        folder = ROOT/'artificial/f_region_wbbs_20261121T143545+0800/cross_valid/region_j'
        meta = read_json(folder/'evaluation_manifest.json')
        check = read_json(folder/'verification.json')
        marker = read_json(folder/'completed.json')
        required = {'evaluation_manifest.json', 'verification.json', 'rankings.csv',
                    'coverage_metrics.json', 'coverage_per_query.csv', 'coverage_per_class.csv'}
        key = 'f_region_wbbs_f_region_wbbs_20261121T143545_0800_cross_valid_region_j'
        expected = {'key': key, 'stage': 'cross_valid', 'split': 'valid',
            'variant': 'region_j', 'spec': {'family': 'j_baseline'},
            'protocol': PROTOCOL, 'candidate_rule': RULE,
            'retrieval_protocol_signature': FROZEN['valid']['derived_signature'],
            'base_protocol_signature': BASE_PROTOCOL_SIGNATURE,
            'num_queries': 500, 'gallery_count': 3599, 'ks': [1,5,10,20,50,100,500]}
        if (any(meta.get(k) != v for k,v in expected.items())
                or not required <= set(marker.get('files', {}))
                or check.get('status') != 'verified'
                or any(check.get(k) != meta.get(k) for k in
                       ('signature', 'context_signature', 'stage', 'split', 'variant', 'spec', 'ks'))
                or check.get('num_queries') != 500 or check.get('ranking_rows') != 250000
                or check.get('all_queries_full_candidate_topk_exact') is not True
                or check.get('full_candidates_checked') != FROZEN['valid']['candidate_pairs']
                or check.get('candidate_identity_sha256') != FROZEN['valid']['candidate_identity_sha256']
                or check.get('ranking_sha256') != file_hash(folder/'rankings.csv')):
            raise ValueError('Historical cross-valid baseline identity/protocol is incompatible')
        bind_sealed(folder, data['source_binding'], meta['signature'])
        data['replay_references']['cross_valid'] = {'primary': {
            'key': key, 'folder': str(folder.resolve()),
            'ranking_sha256': file_hash(folder/'rankings.csv'),
            'completed_sha256': file_hash(folder/'completed.json'), 'spec': {'family': 'j_baseline'}}}
    cache = inputs['source_root']/'cache'
    marker = read_json(cache/f'{split}_prepared.json')
    if marker['signature'] != inputs['source_manifest']['signature']:
        raise ValueError('Evaluation patch cache belongs to another suite')
    binding = dict(data['source_binding'])
    for suffix in ('F.npy', 'regions.npz', 'region_labels.npy'):
        name = f'cache/{split}_{suffix}'
        path = inputs['source_root']/name
        expected = marker['files'][name]
        if file_hash(path) != expected:
            raise ValueError('Frozen evaluation input cache changed: '+str(path))
        binding[str(path)] = expected
    binding[str(cache/f'{split}_prepared.json')] = file_hash(cache/f'{split}_prepared.json')
    with np.load(cache/f'{split}_regions.npz', allow_pickle=False) as raw:
        for name in ('map_ids', 'offsets', 'area'):
            if not np.array_equal(raw[name], data['cache'][name]):
                raise ValueError('Retrieval regions differ from model input token order: '+name)
    patches = np.load(cache/f'{split}_F.npy', mmap_mode='r', allow_pickle=False)
    if patches.shape != (int(data['cache']['offsets'][-1]), 3, 96, 96) or patches.dtype != np.uint8:
        raise ValueError('Unexpected source input patch shape/dtype')
    data['refinement_patches'] = patches
    data['baseline_evaluation_root'] = BASELINE_RUN
    _bind_baseline_run(binding)
    old_index = read_json(BASELINE_RUN/f'evaluation/baseline/evaluation_index_{split}.json')
    for row in old_index['evaluations'].values():
        bind_sealed(Path(row['path']), binding, row['signature'])
    data['source_binding'] = dict(sorted(binding.items()))
    record(Path(output)/f'inputs_{split}.json', signed({
        'split': split, 'context_signature': run_context(output)['signature'],
        'baseline_evaluation_root': str(BASELINE_RUN.resolve()),
        'source_binding': data['source_binding'], 'source_metadata': data['source_metadata'],
        'replay_references': data['replay_references']}))
    return data
