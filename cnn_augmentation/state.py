"""Frozen plan for five paper-derived single augmentation experiments."""
from pathlib import Path
import re
from f_region_structure.state import now, record, signed, read_signed, seal, verified
from wbm_retrieval.common import digest, file_hash
ROOT = Path(__file__).resolve().parents[1]
VERSION = 'cnn_augmentation_v1'
MODELS = ('baseline', 'crop', 'cutout', 'noise', 'rotation', 'shift')
VARIANTS = MODELS[1:]
KS = [1, 5, 10, 20, 50, 100, 500]
PLAN = {'baseline': {'source': 'artifacts/f_segmentation_v1/cc8',
              'seed': 2026,
              'checkpoint': 'best_nce',
              'epoch': 100,
              'retrain': False},
 'common_training': {'initialization': 'same original seed2026, not fine-tuning',
                     'epochs': 100,
                     'batch_size': 256,
                     'optimizer_steps': 39000,
                     'checkpoint_policy': 'fixed epoch100; snapshots15/25/50/75/100 diagnostic only',
                     'architecture': 'unchanged three-channel ResNet18 stride1 GAP h512 linear z128',
                     'segmentation': 'cc8 min_area5 Top4; original raw masks',
                     'input': 'original cached three-channel 96x96 windows (52->96 nearest); '
                              'augmentation of second view only',
                     'optimizer': 'SGD',
                     'learning_rate': 0.01,
                     'momentum': 0.9,
                     'weight_decay': 0.001,
                     'scheduler': 'wapirl_cosine',
                     'temperature': 0.07,
                     'memory_momentum': 0.5,
                     'negatives': 5000,
                     'negative_sampling': 'with replacement, exclude current batch tokens',
                     'opening': False,
                     'early_stopping': False,
                     'validation_loss': 'common original rotation-only NCE, diagnostic only',
                     'extra_randomness': 'independent per-token/epoch/method augmentation RNG; original '
                                         'batch order and shared5000negative torch RNG unchanged'},
 'data': {'protocol': 'deduplicated_v1/retrieval_v1',
          'split_counts': {'train': 25193, 'valid': 3599, 'test': 7199},
          'valid_queries': 500,
          'valid_gallery': 3599,
          'test_queries': 7199,
          'test_gallery': 7199,
          'candidate_rule': 'all gallery except query',
          'normal': 'excluded',
          'sample_id': 'original NPZ row',
          'labels_in_training_or_graph': False},
 'version': 'cnn_augmentation_v1',
 'new_training_variants': ['crop', 'cutout', 'noise', 'rotation', 'shift'],
 'single_augmentations': {'crop': {'area_scale': [0.5, 1.0],
                                   'aspect_ratio': [0.9, 1.1],
                                   'interpolation': 'nearest',
                                   'probability': 1.0,
                                   'implementation': 'author RandomResizedCrop; albumentations0.4.6 '
                                                     'ten-attempt and center fallback semantics'},
                          'cutout': {'holes': 4,
                                     'max_height': 19,
                                     'max_width': 19,
                                     'probability': 0.5,
                                     'fill': 0,
                                     'implementation': 'author default cut_ratio.2 on96; '
                                                       'albumentations0.4.6 center-clipping semantics; '
                                                       'same holes all3channels'},
                          'noise': {'selection_probability': 0.05,
                                    'replacement_defect_probability': 0.5,
                                    'domain': 'valid wafer W only',
                                    'target_channel': 'T_new = T_original & D_new',
                                    'support_channel': 'unchanged',
                                    'implementation': 'author MaskedBernoulliNoise resets selected '
                                                      'valid dies to1or2 uniformly; ternary semantics '
                                                      'adapted to D/T/W'},
                          'rotation': {'angle_degrees': [-180.0, 180.0],
                                       'probability': 1.0,
                                       'interpolation': 'nearest',
                                       'fill': 0,
                                       'implementation': 'sharedcv2affine all3channels, '
                                                         'center(width/2,height/2); '
                                                         'equivalentorientationrange0-360'},
                          'shift': {'x_fraction': [-0.25, 0.25],
                                    'y_fraction': [-0.25, 0.25],
                                    'probability': 1.0,
                                    'interpolation': 'nearest',
                                    'fill': 0,
                                    'implementation': 'sharedcontinuouscv2affine shift all3channels; '
                                                      'noscale/norotation'}},
 'augmentation_policy': {'combination': 'one method per model; replaces baseline right-angle rotation',
                         'first_view_and_inference': 'unchanged full original cached window',
                         'target_preservation': 'no target-aware rejection/resampling; audit '
                                                'retained-target fraction and empty-target views',
                         'no_extra_negatives': 'original5000sampler; no pseudo-neighbor filter',
                         'model_names': ['baseline', 'crop', 'cutout', 'noise', 'rotation', 'shift']},
 'paper': {'doi': '10.1109/TSM.2020.3038165',
           'author_repository': 'https://github.com/hgkahng/WaPIRL',
           'author_commit': '10e24c3bbec1ebe98ed32bef8faa86206042ef58',
           'augmentation_source': 'datasets/transforms.py',
           'albumentations_version': '0.4.6',
           'albumentations_commit': '70452cd040f782dcaca1b19964d6eafe7e5d14eb',
           'scope': 'adaptation of five augmentation operators to existing F, not reproduction of '
                    'whole-wafer supervised classification or fine-tuning'},
 'evaluation': {'models': ['baseline', 'crop', 'cutout', 'noise', 'rotation', 'shift'],
                'scoring': ['cnn_only', 'fusion'],
                'cnn_only': 'clip(cos(h_q,h_c),0,1)',
                'fusion': '0.75*(0.4*C+0.6*T)+0.25*A; existing J and area unchanged',
                'matching': 'original unidirectional greedy one-to-one, sqrt(area) query weights',
                'stages': ['valid', 'cross_valid', 'test', 'cross_test'],
                'ks': [1, 5, 10, 20, 50, 100, 500],
                'relevance': 'ordinary exact+coverage on same ranking; cross coverage',
                'cross_rule': 'exclude same full label combination from complete gallery, then rank',
                'denominators': 'complete permitted candidate library',
                'groups': ['overall',
                           'class_macro',
                           'single',
                           'mixed',
                           'label_count',
                           'per_class',
                           'per_query'],
                'test_gate': 'implementation/protocol validation only, not favorable validation '
                             'performance',
                'model_or_hyperparameter_search': False,
                'single_seed': 2026,
                'existing_baseline': 'results/20261123T105220+0800__cnn_refinement; all8formalroles '
                                     'reusecanonicalkeys'},
 'decision_rules': {'fusion_exact_P10_absolute_improvement': 0.01,
                    'paired_95CI_lower_above': 0.0,
                    'fusion_cross_nDCG10_noninferiority_lower': -0.005,
                    'representation_claim': 'CNN-only retrieval must also improve',
                    'bootstrap': '2000 paired resamples stratified by37full label combinations; fixed '
                                 'model/gallery',
                    'report_all_five_regardless_of_sign': True},
 'diagnostics': {'augmentation': 'fixed train-only sample and actual all-epoch target/support '
                                 'retention, empty-target rate, applied/no-op rates and noise '
                                 'selection/change counts',
                 'training': 'common original rotation validNCE diagnostic, no checkpoint selection'},
 'limitations': ['posthoc augmentation study motivated by baseline/test observations',
                 'single seed does not establish training stability',
                 'original paper uses whole-wafer WM811K classification and supervised fine-tuning; '
                 'current study uses deduplicated MixedWM38 regional retrieval',
                 'geometric augmentation may clip target/support and noise may remove target pixels; '
                 'report exposure rather than silently rejecting views',
                 'conditional paired query intervals hold model and gallery fixed and do not correct '
                 'prior exploration or multiple comparisons']}


def code_binding(names):
    return {name: file_hash(ROOT/name) for name in sorted(names)}


def identity(output, key, signature, **fields):
    path = Path(output)/'run_identity.json'
    if path.exists():
        result = read_signed(path)
        if (result.get('key') != key or result.get('evaluation_signature') != signature
                or any(result.get(k) != v for k, v in fields.items())):
            raise ValueError('Stage identity changed')
        return result
    if not re.fullmatch(r'[A-Za-z0-9_]+', key):
        raise ValueError('Unsafe stage key')
    time = now()
    result = signed(dict(schema_version=1, key=key,
        run_id=time.strftime('%Y%m%dT%H%M%S%z')+'__'+key,
        started_at=time.isoformat(), evaluation_signature=signature, **fields))
    record(path, result)
    return result


def run_context(output):
    manifest = read_signed(Path(output)/'run_manifest.json')
    plan = read_signed(Path(output)/'frozen_plan.json')
    if manifest.get('version') != VERSION or plan.get('plan') != PLAN:
        raise ValueError('Frozen scientific plan changed')
    if plan.get('signature') != manifest.get('plan_signature'):
        raise ValueError('Plan detached from run manifest')
    return manifest
