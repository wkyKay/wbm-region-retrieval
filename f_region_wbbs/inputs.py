"""Read-only frozen regional inputs and the complete ordinary-valid WBBS matrix.

This follows the existing area/evidence provenance chain without loading a
train reference bank. Test input access is guarded by the caller's new valid
selection. The global benchmark is a valid-only score replay, never a selector.
"""
from pathlib import Path
import tempfile

import numpy as np

from f_region_area_fusion import inputs as area_inputs
from f_region_evidence.inputs import bind_sealed
from f_region_score_search.inputs import _bind
from f_region_structure import sources
from f_region_structure.state import read_signed, record, seal, verified, now
from fgw_grid_search.cache import FullScoreCache
from wbm_retrieval.common import digest, file_hash, read_json
from .state import ROOT, GLOBAL, KS
from .protocol import PROTOCOL, RULE, FROZEN, BASE_PROTOCOL_SIGNATURE

REGION_SOURCE = ROOT/'artificial/f_region_structure_20261120T120331+0800'
VALID_SOURCE = ROOT/'artificial/f_region_area_fusion_valid_20261120T155825+0800'
TEST_SOURCE = ROOT/'artificial/f_region_area_fusion_test_20261120T160426+0800'
GLOBAL_SOURCE = ROOT/'artificial/fgw_grid_20261119T200653+0800'
CROSS_J_SOURCE = ROOT/'artificial/f_region_j_cross_test_20261120T202638+0800'
CROSS_GLOBAL_SOURCE = ROOT/'artificial/fgj_cross_test_20261120T193642+0800'
SOURCE_VARIANT = 'add_a0p4_b0p25'
GLOBAL_SPEC = {'family': 'global_wbbs'}


class GlobalWBBSProvider:
    """Exact frozen WBBS values for any unique nonself subset of the gallery."""
    def __init__(self, cache, pool, source_receipt):
        queries, gallery = np.asarray(list(pool)), np.asarray(pool.gallery)
        if (pool.split_name != 'valid' or cache.split != 'valid'
                or cache.protocol_signature != pool.protocol_signature
                or not np.array_equal(cache.query_ids, queries) or not np.array_equal(cache.gallery_ids, gallery)
                or pool.candidates_per_query != len(gallery)-1):
            raise ValueError('Global WBBS source is not the identical ordinary-valid pool')
        self.cache = cache
        self.source_receipt = dict(source_receipt)
        self.query_ids, self.gallery_ids = cache.query_ids, cache.gallery_ids
        self.signature, self.protocol_signature = cache.signature, cache.protocol_signature
        self.marker = cache.marker
        self.matrix = cache.matrices['WBBS']
        self.query_positions = {int(q): i for i, q in enumerate(self.query_ids)}
        self.gallery_positions = {int(c): i for i, c in enumerate(self.gallery_ids)}
        for q in pool:
            if not np.array_equal(pool[int(q)], gallery[gallery != q]):
                raise ValueError('Global source pool must retain complete frozen gallery order')

    def scores(self, query, candidates):
        ids = np.asarray(candidates)
        if (isinstance(query, (bool, np.bool_)) or not isinstance(query, (int, np.integer))
                or int(query) not in self.query_positions or ids.ndim != 1 or ids.dtype.kind not in 'iu'
                or len(np.unique(ids)) != len(ids) or np.any(ids == query)
                or not np.isin(ids, self.gallery_ids).all()):
            raise ValueError('Expected a frozen query and unique nonself gallery candidate IDs')
        return np.asarray(self.matrix[self.query_positions[int(query)],
                          [self.gallery_positions[int(c)] for c in ids]])

    def full_row(self, query):
        ids = self.gallery_ids[self.gallery_ids != query]
        return ids, self.scores(query, ids)


def global_valid_source(pool, binding, source=GLOBAL_SOURCE):
    """Bind a completed matrix and its original signed input/source chain."""
    source = Path(source).resolve()
    manifest = read_signed(source/'run_manifest.json')
    context = manifest['context']
    inp = read_signed(source/'inputs.json')
    completed = read_json(source/'completed.json')
    if (manifest.get('version') != 'fgw_grid_search_v1' or manifest.get('stage') != 'search'
            or context.get('split') != 'valid' or context.get('stage') != 'search'
            or inp.get('split') != 'valid' or inp.get('parent_signature') != manifest['signature']
            or inp.get('context_digest') != digest(context) or inp.get('provenance') != context.get('provenance')
            or completed.get('status') != 'completed' or completed.get('version') != manifest['version']
            or completed.get('signature') != manifest['signature'] or completed.get('stage') != 'search'
            or len(pool) != 500 or len(pool.gallery) != 3599 or pool.split_name != 'valid'):
        raise ValueError('Global WBBS source must be the completed same-protocol valid component cache')
    for name in ('run_manifest.json', 'inputs.json'):
        _bind(source/name, binding, completed['files'].get(name))
        if name not in completed['files']:
            raise ValueError('Global source root did not bind its manifest/inputs')
    _bind(source/'completed.json', binding)
    folder = source/'component_cache/valid'
    for name in ('plan.json', 'completed/completed.json'):
        key = 'component_cache/valid/'+name
        if key not in completed['files']:
            raise ValueError('Global source completion omits component cache provenance')
        _bind(folder/name, binding, completed['files'][key])
    cache = FullScoreCache(folder)
    if cache.signature != digest({'inputs': inp['signature'], 'operation': 'full_F_G_W_scores'}):
        raise ValueError('Global full-score matrix is detached from its signed original inputs')
    for name, expected in cache.completion['files'].items():
        _bind(folder/'completed'/name, binding, expected)
    # These are the exact unchanged scientific sources referenced by the old run.
    for name, expected in context['code'].items():
        _bind(ROOT/name, binding, expected)
    receipt = {'status': 'verified', 'source_root': str(source), 'source_cache': str(folder),
        'source_manifest_signature': manifest['signature'], 'source_context_signature': digest(context),
        'source_inputs_signature': inp['signature'], 'source_cache_signature': cache.signature,
        'source_completed_sha256': file_hash(source/'completed.json'),
        'source_cache_completed_sha256': file_hash(cache.marker),
        'source_scores_sha256': cache.completion['files']['WBBS.npy'],
        'source_query_ids_sha256': cache.completion['files']['query_ids.npy'],
        'source_gallery_ids_sha256': cache.completion['files']['gallery_ids.npy'],
        'split': 'valid', 'num_queries': len(pool), 'gallery_count': len(pool.gallery),
        'nonself_pairs': len(pool)*(len(pool.gallery)-1), 'component': 'WBBS',
        'retrieval_protocol_signature': pool.protocol_signature,
        'scope': 'unchanged completed full WBBS score matrix, no new global scoring or tuning'}
    return GlobalWBBSProvider(cache, pool, receipt), receipt


def _reference_folders(split, binding):
    if split != 'test':
        return {}
    result = {'region_j_cross': CROSS_J_SOURCE/'test'/SOURCE_VARIANT,
              'global_wbbs_cross': CROSS_GLOBAL_SOURCE/'test/WBBS'}
    for name, folder in result.items():
        parent = CROSS_J_SOURCE if name == 'region_j_cross' else CROSS_GLOBAL_SOURCE
        manifest = read_signed(parent/'run_manifest.json')
        root = read_json(parent/'completed.json')
        context_signature = digest(manifest['context'])
        if root.get('status') != 'completed' or root.get('signature') != context_signature:
            raise ValueError('Historical cross test reference root identity differs')
        _bind(parent/'run_manifest.json', binding, root['files']['run_manifest.json'])
        _bind(parent/'completed.json', binding)
        rel = str(folder.relative_to(parent))+'/completed.json'
        _bind(folder/'completed.json', binding, root['files'][rel])
        variant = SOURCE_VARIANT if name == 'region_j_cross' else 'WBBS'
        spec = {'family': 'add', 'alpha': .4, 'beta': .25} if name == 'region_j_cross' else {'formula': 'WBBS'}
        signature = digest({'context_signature': context_signature, 'variant': variant, 'spec': spec})
        bind_sealed(folder, binding, signature)
        report = read_json(folder/'coverage_metrics.json')
        child = read_json(folder/'evaluation_manifest.json')
        expected_metadata = {**child, 'candidates_scored': 'full_filtered_pool', 'smoke': False}
        if (child.get('split') != 'test' or child.get('context_signature') != context_signature
                or child.get('signature') != signature or child.get('variant') != variant or child.get('spec') != spec
                or child.get('candidate_rule') != RULE or child.get('protocol') != PROTOCOL
                or child.get('retrieval_protocol_signature') != FROZEN['test']['derived_signature']
                or child.get('base_protocol_signature') != BASE_PROTOCOL_SIGNATURE
                or child.get('ks') != KS or child.get('tie_seed') != 2026
                or report['metadata'] != expected_metadata or child.get('num_queries') != 7199
                or child.get('gallery_count') != 7199):
            raise ValueError('Historical coverage reference is not the identical cross test protocol')
    return {k: str(v.resolve()) for k,v in result.items()}


def load(split='valid'):
    if split == 'valid':
        data = area_inputs.load_valid(REGION_SOURCE)
        manifest, frozen, binding = area_inputs.frozen_choice(VALID_SOURCE)
        if (frozen['selection']['winner'] != SOURCE_VARIANT
                or manifest['context']['source_binding'] != data['source_binding']
                or manifest['context']['matching'] != data['config']['matching']):
            raise ValueError('Expected the frozen valid-selected C/T/A regional baseline')
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
            raise ValueError('Historical test no longer matches its frozen valid model/selection')
        baseline = TEST_SOURCE/'test'/SOURCE_VARIANT
        bind_sealed(TEST_SOURCE, binding, digest(context))
    else:
        raise ValueError('Only frozen valid/test input loading is supported')
    bind_sealed(baseline, binding, digest({'context': digest(manifest['context']), 'split': split,
        'variant': SOURCE_VARIANT, 'spec': {'family': 'add', 'alpha': .4, 'beta': .25}}))
    old_manifest = read_signed(REGION_SOURCE/'run_manifest.json')
    config, suite, checkpoint, protocol, source_identity = sources.load(old_manifest['source_root'])
    if source_identity != old_manifest['context']['source']:
        raise ValueError('Original CNN/model/protocol changed')
    if data['protocol'].ids.keys() != protocol.ids.keys() or any(
            not np.array_equal(data['protocol'].ids[s], protocol.ids[s]) for s in protocol.ids):
        raise ValueError('Frozen sample identities changed')
    meta, old_labels, features, _patches, feature_binding = sources.feature_source(
        split, config, source_identity['geometry'], old_manifest['feature_roots'])
    for path, expected in feature_binding.items():
        _bind(path, binding, expected)
    if (not np.array_equal(meta['map_ids'], data['cache']['map_ids'])
            or np.asarray(features).dtype != np.asarray(data['cache']['features']).dtype
            or np.asarray(features).tobytes() != np.asarray(data['cache']['features']).tobytes()):
        raise ValueError('Frozen mask/h512 and regional-J caches differ')
    with np.load(config['data_file'], allow_pickle=False) as archive:
        maps = archive['arr_0']
    global_provider, global_source = (global_valid_source(protocol.retrieval.valid, binding)
                                     if split == 'valid' else (None, None))
    references = _reference_folders(split, binding)
    data.update(config=config, suite=suite, checkpoint=checkpoint, protocol=protocol,
        maps=maps, old_labels=old_labels, source_identity=source_identity,
        source_binding=dict(sorted(binding.items())), baseline_folder=baseline, source_run=REGION_SOURCE,
        global_provider=global_provider, global_source=global_source, reference_folders=references,
        source_metadata={'regional_source': str(REGION_SOURCE.resolve()),
            'source_valid': str(VALID_SOURCE.resolve()), 'source_test': str(TEST_SOURCE.resolve()) if split=='test' else None,
            'source_variant': SOURCE_VARIANT, 'source_spec': {'family': 'add', 'alpha': .4, 'beta': .25},
            'baseline_folder': str(baseline.resolve()), 'structure_strength': .5,
            'training_required': False, 'train_reference_bank_loaded': False,
            'global_valid_source': global_source, 'reference_folders': references})
    return data


def export_global(output, provider, pool, context_signature):
    """Copy all ordinary-valid nonself WBBS values, changing only stored self NaNs."""
    from .pipeline import CachedProvider
    output = Path(output)
    if (pool.split_name != 'valid' or provider.protocol_signature != pool.protocol_signature
            or not np.array_equal(provider.query_ids, list(pool))
            or not np.array_equal(provider.gallery_ids, pool.gallery)):
        raise ValueError('Global score export requires its identical ordinary-valid pool')
    folder = output/'score_cache/valid'/GLOBAL
    signature = digest({'context': context_signature, 'split': 'valid', 'variant': GLOBAL,
                        'spec': GLOBAL_SPEC, 'score_cache_version': 1})
    expected = {'signature': signature, 'context_signature': context_signature, 'variant': GLOBAL,
        'spec': GLOBAL_SPEC, 'split': 'valid', 'retrieval_protocol_signature': pool.protocol_signature,
        'num_queries': len(pool), 'gallery_count': len(pool.gallery), 'dtype': 'float64',
        'self_scores': 'zero; never ranked', 'source_receipt': provider.source_receipt}
    if verified(folder, signature):
        metadata = read_json(folder/'metadata.json')
        if any(metadata.get(k) != v for k,v in expected.items()):
            raise ValueError('Completed global copy source or protocol changed')
        copied = CachedProvider(folder)
    else:
        if folder.exists():
            raise ValueError('Incomplete global score copy requires a fresh output')
        folder.parent.mkdir(parents=True, exist_ok=True)
        attempt = Path(tempfile.mkdtemp(prefix='.'+GLOBAL+'_attempt_', dir=folder.parent))
        started = now().isoformat()
        matrix = np.lib.format.open_memmap(attempt/'scores.npy', mode='w+', dtype=np.float64,
                                          shape=(len(pool), len(pool.gallery)))
        for row, q in enumerate(pool):
            matrix[row] = provider.matrix[row]
            matrix[row, provider.gallery_ids == q] = 0.
        matrix.flush();del matrix
        np.savez(attempt/'identities.npz', queries=provider.query_ids, gallery=provider.gallery_ids)
        record(attempt/'metadata.json', {**expected, 'started_at': started, 'finished_at': now().isoformat()})
        seal(attempt, signature)
        if folder.exists():
            raise ValueError('Concurrent global cache publication detected')
        attempt.rename(folder)
        copied = CachedProvider(folder)
    checked = 0
    for q in pool:
        ids = np.asarray(pool[q]);a,b = copied.scores(q,ids),provider.scores(q,ids)
        if a.dtype != np.float64 or a.tobytes() != b.tobytes():
            raise ValueError('Copied nonself global scores lost byte identity')
        if copied.matrix[copied.query_positions[int(q)],copied.gallery_positions[int(q)]] != 0.:
            raise ValueError('Copied global self storage must be zero and never ranked')
        checked += len(ids)
    receipt = {'status': 'verified', 'context_signature': context_signature, 'split': 'valid',
        'variant': GLOBAL, 'num_queries': len(pool), 'gallery_count': len(pool.gallery),
        'full_nonself_pairs_checked': checked, 'nonself_scores_bitwise_identical': True,
        'max_score_absolute_error': 0., 'self_storage_conversion': 'source NaN to zero; excluded before ranking',
        'source_receipt': provider.source_receipt, 'score_cache_signature': signature,
        'score_cache_completed_sha256': file_hash(copied.marker)}
    record(output/'global_wbbs_copy_valid.json', receipt)
    return copied
