"""Read-only legacy regional-J/CNN inputs, with strict administrative compatibility."""
from pathlib import Path

from f_region_area_fusion import inputs as area_inputs
from f_region_evidence.inputs import bind_sealed
from f_region_score_search.inputs import _bind
from f_region_wbbs import compat
from f_region_wbbs.protocol import BASE_PROTOCOL_SIGNATURE, FROZEN, PROTOCOL, RULE
from wbm_retrieval.common import digest, file_hash, read_json
from .state import ROOT, KS, REPLAY_KEYS, read_signed

REGION_SOURCE = ROOT/'artificial/f_region_structure_20261120T120331+0800'
VALID_SOURCE = ROOT/'artificial/f_region_area_fusion_valid_20261120T155825+0800'
TEST_SOURCE = ROOT/'artificial/f_region_area_fusion_test_20261120T160426+0800'
CROSS_SOURCE = ROOT/'artificial/f_region_j_cross_test_20261120T202638+0800'
PRIMARY_VARIANT = 'add_a0p4_b0p25'
PRIMARY_SPEC = {'family': 'add', 'alpha': .4, 'beta': .25}
NO_A_VARIANT = 'no_area_a0p4'
NO_A_SPEC = {'family': 'no_area', 'alpha': .4, 'beta': 0.}


def _ordinary_reference(root, manifest, split, name, old_variant, old_spec, binding):
    folder = root/split/old_variant
    context_signature = digest(manifest['context'])
    signature = digest({'context': context_signature, 'split': split,
                        'variant': old_variant, 'spec': old_spec})
    bind_sealed(folder, binding, signature)
    meta = read_json(folder/'evaluation_manifest.json')
    check = read_json(folder/'verification.json')
    nq, ng = (500, 3599) if split == 'valid' else (7199, 7199)
    if (meta.get('signature') != signature or meta.get('spec') != old_spec
            or meta.get('split') != split or meta.get('variant') != old_variant
            or meta.get('ks') != KS or meta.get('tie_seed') != 2026
            or meta.get('num_queries') != nq or meta.get('gallery_count') != ng
            or meta.get('candidate_rule') != 'all_gallery_except_query'
            or meta.get('retrieval_protocol_signature') != BASE_PROTOCOL_SIGNATURE
            or check.get('status') != 'verified' or check.get('num_queries') != nq
            or check.get('ranking_rows') != nq*max(KS)
            or check.get('full_candidates_checked') != nq*(ng-1)
            or check.get('all_queries_full_candidate_topk_exact') is not True
            or check.get('ranking_sha256') != file_hash(folder/'rankings.csv')):
        raise ValueError('Historical replay control has an incompatible ordinary protocol')
    return {'key': REPLAY_KEYS[split][name], 'folder': str(folder.resolve()),
            'ranking_sha256': file_hash(folder/'rankings.csv'),
            'completed_sha256': file_hash(folder/'completed.json'), 'spec': old_spec}


def _cross_reference(binding):
    manifest = read_signed(CROSS_SOURCE/'run_manifest.json')
    context = manifest['context']
    context_signature = digest(context)
    root_marker = read_json(CROSS_SOURCE/'completed.json')
    if (manifest.get('version') != 'f_region_j_cross_test_v1'
            or root_marker.get('status') != 'completed'
            or root_marker.get('signature') != context_signature):
        raise ValueError('Historical primary cross source is not sealed')
    # The immutable original source chain is preserved, not replaced by current hashes.
    _bind(CROSS_SOURCE/'completed.json', binding)
    _bind(CROSS_SOURCE/'run_manifest.json', binding, root_marker['files']['run_manifest.json'])
    folder = CROSS_SOURCE/'test'/PRIMARY_VARIANT
    relative = str(folder.relative_to(CROSS_SOURCE))+'/completed.json'
    _bind(folder/'completed.json', binding, root_marker['files'][relative])
    signature = digest({'context_signature': context_signature, 'variant': PRIMARY_VARIANT,
                        'spec': PRIMARY_SPEC})
    bind_sealed(folder, binding, signature)
    meta = read_json(folder/'evaluation_manifest.json')
    check = read_json(folder/'verification.json')
    if (meta.get('signature') != signature or meta.get('spec') != PRIMARY_SPEC
            or meta.get('split') != 'test' or meta.get('variant') != PRIMARY_VARIANT
            or meta.get('ks') != KS or meta.get('tie_seed') != 2026
            or meta.get('num_queries') != 7199 or meta.get('gallery_count') != 7199
            or meta.get('candidate_rule') != RULE or meta.get('protocol') != PROTOCOL
            or meta.get('retrieval_protocol_signature') != FROZEN['test']['derived_signature']
            or meta.get('base_protocol_signature') != BASE_PROTOCOL_SIGNATURE
            or check.get('status') != 'verified' or check.get('num_queries') != 7199
            or check.get('ranking_rows') != 7199*max(KS)
            or check.get('full_candidates_checked') != FROZEN['test']['candidate_pairs']
            or check.get('all_queries_full_candidate_topk_exact') is not True
            or check.get('ranking_sha256') != file_hash(folder/'rankings.csv')):
        raise ValueError('Historical primary cross replay identity or full-gallery verification differs')
    report = read_json(folder/'coverage_metrics.json')
    if any(report.get('metadata', {}).get(k) != v for k, v in meta.items()):
        raise ValueError('Historical primary cross report is detached from its manifest')
    return {'key': REPLAY_KEYS['cross_test']['primary'], 'folder': str(folder.resolve()),
            'ranking_sha256': file_hash(folder/'rankings.csv'),
            'completed_sha256': file_hash(folder/'completed.json'), 'spec': PRIMARY_SPEC}


def load(split='valid'):
    """Load only requested split; the caller must validate the new valid gate before test."""
    if split == 'valid':
        data = area_inputs.load_valid(REGION_SOURCE)
        manifest, frozen, binding, compatibility = compat.frozen_choice(VALID_SOURCE)
        if (frozen['selection']['winner'] != PRIMARY_VARIANT
                or manifest['context']['source_binding'] != data['source_binding']
                or manifest['context']['matching'] != data['config']['matching']):
            raise ValueError('Frozen regional inputs differ from the original valid-selected primary')
        binding.update(data['source_binding'])
        references = {'valid': {
            'primary': _ordinary_reference(VALID_SOURCE, manifest, split, 'primary',
                                          PRIMARY_VARIANT, PRIMARY_SPEC, binding),
            'no_a': _ordinary_reference(VALID_SOURCE, manifest, split, 'no_a',
                                       NO_A_VARIANT, NO_A_SPEC, binding)}}
    elif split == 'test':
        # This adapter only permits the exact archived/current build.py byte pair;
        # all scientific code hashes and historical binding equalities remain exact.
        data = compat.load_test(VALID_SOURCE)
        compatibility = data['source_compatibility']
        binding = dict(data['source_binding'])
        manifest = read_signed(TEST_SOURCE/'run_manifest.json')
        context = manifest['context']
        if (manifest.get('version') != 'f_region_area_fusion_v1' or context.get('split') != split
                or context.get('source_run') != str(VALID_SOURCE.resolve())
                or context.get('cache_source_run') != str(REGION_SOURCE.resolve())
                or context.get('source_binding') != data['source_binding']
                or context.get('matching') != data['config']['matching']
                or context.get('frozen_selection_signature') != data['frozen_selection']['signature']
                or read_signed(TEST_SOURCE/'frozen_selection.json') != data['frozen_selection']):
            raise ValueError('Historical test model, protocol or valid selection changed')
        bind_sealed(TEST_SOURCE, binding, digest(context))
        compatibility += compat.validate_code(context['code'])
        references = {'test': {'primary': _ordinary_reference(TEST_SOURCE, manifest, split,
                         'primary', PRIMARY_VARIANT, PRIMARY_SPEC, binding)},
                      'cross_test': {'primary': _cross_reference(binding)}}
    else:
        raise ValueError('Load only the frozen valid or test split')
    compatibility = compat.unique_receipts(compatibility)
    # Only append truthful administrative receipt hashes after all exact legacy equalities.
    compat.bind_compatibility(compatibility, binding)
    data.update(source_binding=dict(sorted(binding.items())), source_compatibility=compatibility,
                replay_references=references, source_run=REGION_SOURCE, cache_source_run=REGION_SOURCE,
                source_metadata={'region_source': str(REGION_SOURCE.resolve()),
                    'source_valid': str(VALID_SOURCE.resolve()),
                    'source_test': str(TEST_SOURCE.resolve()) if split == 'test' else None,
                    'source_variant': PRIMARY_VARIANT, 'source_spec': PRIMARY_SPEC,
                    'training_required': False, 'original_masks_and_features_reused': True})
    return data
