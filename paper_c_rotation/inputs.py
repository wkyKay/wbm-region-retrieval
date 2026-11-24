"""Read-only rotation-epoch100 features and canonical new-weight primary controls."""
from pathlib import Path

from cnn_augmentation.evaluation import replaced_cache, token_layout
from cnn_rotation_fusion import inputs as rotation_inputs
from cnn_rotation_fusion.state import run_context, check_binding
from f_region_evidence.inputs import bind_sealed
from f_region_score_search.inputs import _bind
from wbm_retrieval.common import digest, file_hash, read_json
from .state import ROOT, REPLAY_KEYS, read_signed

PRIMARY_SOURCE = ROOT/'results/20261124T093446+0800__cnn_rotation_fusion'
PRIMARY_VARIANT = 'add_a0p4_b0p15'
PRIMARY_SPEC = {'family': 'add', 'alpha': .4, 'beta': .15}


def load(split='valid'):
    """Replace only feature rows after verifying frozen geometry and model identity."""
    if split not in ('valid', 'test'):
        raise ValueError('Load only the frozen valid or test split')
    data = rotation_inputs.load(split)
    binding = dict(data['source_binding'])
    root = PRIMARY_SOURCE.resolve()
    manifest = run_context(root)
    marker = read_json(root/'completed.json')
    if marker.get('status') != 'completed' or marker.get('signature') != manifest['signature']:
        raise ValueError('Current primary source is not sealed')
    _bind(root/'completed.json', binding)

    def anchored(relative):
        path = root/relative
        _bind(path, binding, marker['files'][str(relative)])
        return path

    for name in ('run_manifest.json', 'frozen_plan.json', 'selection.json'):
        anchored(name)
    selection = read_signed(root/'selection.json')
    if (selection.get('context_signature') != manifest['signature']
            or selection.get('winner') != PRIMARY_VARIANT or selection.get('spec') != PRIMARY_SPEC
            or selection.get('selection_split') != 'valid' or selection.get('test_used') is not False):
        raise ValueError('Current primary does not match the frozen valid-selected weights')
    feature_path = Path(data['features']['rotation']).resolve()
    feature_sha = file_hash(feature_path)
    _bind(feature_path, binding)
    context_path = anchored('evaluation/rotation/model_context_'+split+'.json')
    model_context = read_signed(context_path)
    if (model_context.get('parent_context') != manifest['signature']
            or model_context.get('model') != 'rotation' or model_context.get('split') != split
            or model_context.get('h512_path') != str(feature_path)
            or model_context.get('h512_sha256') != feature_sha
            or model_context.get('token_layout') != token_layout(data['cache'])
            or model_context.get('specifications', {}).get(PRIMARY_VARIANT) != PRIMARY_SPEC):
        raise ValueError('Primary replay is detached from rotation features or regional row order')
    index_path = anchored('evaluation_index_'+split+'.json')
    index = read_json(index_path)
    if index.get('context_signature') != manifest['signature']:
        raise ValueError('Primary evaluation index context changed')
    stages = ('valid',) if split == 'valid' else ('test', 'cross_test')
    references = {}
    for stage in stages:
        item = index['evaluations']['rotation/'+stage+'/'+PRIMARY_VARIANT]
        folder = root/'evaluation/rotation'/stage/PRIMARY_VARIANT
        anchored(str((folder/'completed.json').relative_to(root)))
        row = rotation_inputs.reference_row(folder, 'rotation', stage, PRIMARY_VARIANT, canonical=item)
        meta = read_json(folder/'evaluation_manifest.json')
        if (row['key'] != REPLAY_KEYS[stage]['primary']
                or meta.get('model') != 'rotation' or meta.get('stage') != stage
                or meta.get('variant') != PRIMARY_VARIANT or meta.get('spec') != PRIMARY_SPEC
                or meta.get('context_signature') != model_context['signature']):
            raise ValueError('Primary replay model, formula or canonical key changed')
        binding.update(row['source_binding'])
        cache_folder = root/'evaluation/rotation/score_cache'/split/PRIMARY_VARIANT
        anchored(str((cache_folder/'completed.json').relative_to(root)))
        bind_sealed(cache_folder, binding)
        cache_meta = read_json(cache_folder/'metadata.json')
        if (cache_meta.get('context_signature') != model_context['signature']
                or cache_meta.get('h512_sha256') != feature_sha
                or cache_meta.get('token_layout') != model_context['token_layout']
                or cache_meta.get('spec') != PRIMARY_SPEC
                or meta.get('provider_source') != {'signature': cache_meta['signature'],
                      'completed_sha256': file_hash(cache_folder/'completed.json')}):
            raise ValueError('Primary score cache is detached from model identity or formula')
        references[stage] = {'primary': {'key': row['key'], 'folder': str(folder),
            'ranking_sha256': file_hash(folder/'rankings.csv'),
            'completed_sha256': file_hash(folder/'completed.json'), 'spec': PRIMARY_SPEC,
            'score_cache': str(cache_folder),
            'score_cache_completed_sha256': file_hash(cache_folder/'completed.json')}}
    cache = replaced_cache(data['cache'], feature_path)
    cache['feature_path'] = str(feature_path)
    data.update(cache=cache, source_binding=dict(sorted(binding.items())), replay_references=references,
        source_metadata={**data['source_metadata'], 'primary_source': str(root),
            'source_variant': PRIMARY_VARIANT, 'source_spec': PRIMARY_SPEC,
            'feature_path': str(feature_path), 'feature_sha256': feature_sha,
            'feature_token_layout': token_layout(cache), 'checkpoint': 'fixed_epoch_100',
            'model_source': data['model_source']['rotation'],
            'training_required': False, 'geometry_reused': True, 'original_masks_and_features_reused': False,
            'features': 'hash-bound rotation epoch100 original-view normalized h512'})
    return data
