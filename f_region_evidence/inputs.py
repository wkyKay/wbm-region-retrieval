"""Frozen model/protocol/caches, without test access during valid selection."""
from pathlib import Path
import numpy as np

from f_region_area_fusion import inputs as area_inputs
from f_region_score_search.inputs import _bind
from f_region_structure import sources
from f_region_structure.state import read_signed
from wbm_retrieval.common import digest, read_json
from .state import ROOT

REGION_SOURCE = ROOT/'artificial/f_region_structure_20261120T120331+0800'
VALID_SOURCE = ROOT/'artificial/f_region_area_fusion_valid_20261120T155825+0800'
TEST_SOURCE = ROOT/'artificial/f_region_area_fusion_test_20261120T160426+0800'
SOURCE_VARIANT = 'add_a0p4_b0p25'


def bind_sealed(folder, binding, signature=None):
    folder = Path(folder).resolve()
    _bind(folder/'completed.json', binding)
    marker = read_json(folder/'completed.json')
    if marker.get('status') != 'completed' or (signature is not None and marker.get('signature') != signature):
        raise ValueError('Incomplete or incompatible source: '+str(folder))
    for name, expected in marker['files'].items():
        p = Path(name)
        if p.is_absolute() or '..' in p.parts:
            raise ValueError('Source path escapes its root')
        _bind(folder/p, binding, expected)


def load(split='valid'):
    if split == 'valid':
        data = area_inputs.load_valid(REGION_SOURCE)
        manifest, frozen, binding = area_inputs.frozen_choice(VALID_SOURCE)
        if (frozen['selection']['winner'] != SOURCE_VARIANT
                or manifest['context']['source_binding'] != data['source_binding']
                or manifest['context']['matching'] != data['config']['matching']):
            raise ValueError('Expected the existing valid-selected C/T/A recipe')
        binding.update(data['source_binding'])
        baseline = VALID_SOURCE/'valid'/SOURCE_VARIANT
    elif split == 'test':
        data = area_inputs.load_test(VALID_SOURCE)
        binding = dict(data['source_binding'])
        manifest = read_signed(TEST_SOURCE/'run_manifest.json')
        context = manifest['context']
        if (manifest.get('version') != 'f_region_area_fusion_v1' or context.get('split') != 'test'
                or context.get('source_run') != str(VALID_SOURCE.resolve())
                or context.get('cache_source_run') != str(REGION_SOURCE.resolve())
                or context.get('source_binding') != data['source_binding']
                or context.get('matching') != data['config']['matching']
                or context.get('frozen_selection_signature') != data['frozen_selection']['signature']
                or read_signed(TEST_SOURCE/'frozen_selection.json') != data['frozen_selection']):
            raise ValueError('Historical test must use the exact frozen valid selection and inputs')
        baseline = TEST_SOURCE/'test'/SOURCE_VARIANT
        bind_sealed(TEST_SOURCE, binding, digest(context))
    else:
        raise ValueError('Only frozen valid/test inputs are allowed')
    bind_sealed(baseline, binding, digest({'context': digest(manifest['context']), 'split': split,
                'variant': SOURCE_VARIANT, 'spec': {'family': 'add', 'alpha': .4, 'beta': .25}}))
    old_manifest = read_signed(REGION_SOURCE/'run_manifest.json')
    config, suite, checkpoint, protocol, source_identity = sources.load(old_manifest['source_root'])
    if source_identity != old_manifest['context']['source']:
        raise ValueError('Original CNN/model/protocol changed')
    if data['protocol'].ids.keys() != protocol.ids.keys() or any(
            not np.array_equal(data['protocol'].ids[s], protocol.ids[s]) for s in protocol.ids):
        raise ValueError('Frozen sample identities changed')
    meta, region_labels, _h, _patches, feature_binding = sources.feature_source(
        split, config, source_identity['geometry'], old_manifest['feature_roots'])
    for p, sha in feature_binding.items():
        _bind(p, binding, sha)
    train_folder = REGION_SOURCE/'cache/train'
    bind_sealed(train_folder, binding, digest({'context': data['old_context_signature'], 'split': 'train', 'cache_version': 1}))
    train_cache = sources.read_cache(train_folder)
    train_cache.pop('h_views', None)
    if not np.array_equal(train_cache['map_ids'], protocol.ids['train']):
        raise ValueError('Reference bank must use the entire frozen original train cache')
    if np.intersect1d(train_cache['map_ids'], data['cache']['map_ids']).size:
        raise ValueError('Reference bank overlaps the held-out gallery')
    for p, sha in read_json(train_folder/'source_binding.json').items():
        _bind(p, binding, sha)
    with np.load(config['data_file'], allow_pickle=False) as archive:
        maps = archive['arr_0']
    if not np.array_equal(meta['map_ids'], data['cache']['map_ids']):
        raise ValueError('Frozen mask cache and region-J cache use different IDs')
    data.update(config=config, suite=suite, checkpoint=checkpoint, protocol=protocol,
        maps=maps, old_labels=region_labels, train_cache=train_cache,
        source_identity=source_identity, source_binding=dict(sorted(binding.items())),
        baseline_folder=baseline, source_run=REGION_SOURCE)
    return data
