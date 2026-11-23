"""Scientific plan and immutable stage identities; no historical source writes."""
from pathlib import Path
import re

from f_region_structure.state import now, record, signed, read_signed, seal, verified
from wbm_retrieval.common import digest, file_hash

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'cnn_refinement_v1'
MODELS = ('baseline', 'context_dropout', 'filtered_negatives')
VARIANTS = MODELS[1:]
KS = [1, 5, 10, 20, 50, 100, 500]
PLAN = {
    'version': VERSION, 'new_training_variants': list(VARIANTS),
    'baseline': {'source': 'artifacts/f_segmentation_v1/cc8', 'seed': 2026,
                 'checkpoint': 'best_nce', 'epoch': 100, 'retrain': False},
    'common_training': {'initialization': 'same original seed2026, not fine-tuning',
        'epochs': 100, 'batch_size': 256, 'optimizer_steps': 39000,
        'checkpoint_policy': 'fixed epoch100; snapshots15/25/50/75/100 diagnostic only',
        'architecture': 'unchanged three-channel ResNet18 stride1 GAP h512 linear z128',
        'segmentation': 'cc8 min_area5 Top4; original raw masks',
        'input': 'original 52x52 centered window, nearest resize96; D/T/W',
        'optimizer': 'SGD', 'learning_rate': .01, 'momentum': .9, 'weight_decay': .001,
        'scheduler': 'wapirl_cosine', 'temperature': .07, 'memory_momentum': .5,
        'negatives': 5000, 'negative_sampling': 'with replacement, exclude current batch tokens',
        'opening': False, 'early_stopping': False,
        'validation_loss': 'common original rotation-only NCE, diagnostic only',
        'extra_randomness': 'independent generators; retain original base draw/batch/rotation RNG'},
    'context_dropout': {'second_view': 'delete each non-target defect pixel with p=.5 on raw52 before resize and original rotation',
        'probability': .5, 'preserve': 'target evidence, target mask, support',
        'first_view_and_inference': 'original full window', 'loss_and_sampler': 'unchanged'},
    'filtered_negatives': {'teacher': 'frozen baseline epoch100 h512; no labels',
        'data': 'train only', 'views': ['original', 'rot90'], 'top_k': 500,
        'neighbor_rule': 'cross-wafer mutual cosine Top500 in BOTH views; fixed graph',
        'ties': 'ascending original token index',
        'intervention': 'per-anchor replace pseudo-neighbor slots in original shared draw until exactly5000 allowed negatives',
        'same_wafer_negatives': 'retained unless current batch token; no extra exclusion',
        'positives': 'unchanged; no attraction term', 'augmentations': 'original rotation only'},
    'data': {'protocol': 'deduplicated_v1/retrieval_v1',
        'split_counts': {'train': 25193, 'valid': 3599, 'test': 7199},
        'valid_queries': 500, 'valid_gallery': 3599, 'test_queries': 7199, 'test_gallery': 7199,
        'candidate_rule': 'all gallery except query', 'normal': 'excluded',
        'sample_id': 'original NPZ row', 'labels_in_training_or_graph': False},
    'evaluation': {'models': list(MODELS), 'scoring': ['cnn_only', 'fusion'],
        'cnn_only': 'clip(cos(h_q,h_c),0,1)',
        'fusion': '0.75*(0.4*C+0.6*T)+0.25*A; existing J and area unchanged',
        'matching': 'original unidirectional greedy one-to-one, sqrt(area) query weights',
        'stages': ['valid', 'cross_valid', 'test', 'cross_test'], 'ks': KS,
        'relevance': 'ordinary exact+coverage on same ranking; cross coverage',
        'cross_rule': 'exclude same full label combination from complete gallery, then rank',
        'denominators': 'complete permitted candidate library',
        'groups': ['overall', 'class_macro', 'single', 'mixed', 'label_count', 'per_class', 'per_query'],
        'test_gate': 'implementation/protocol validation only, not favorable validation performance',
        'model_or_hyperparameter_search': False, 'single_seed': 2026},
    'diagnostics': {'data': 'train/valid only; labels posthoc proxies',
        'geometry': 'all699valid single-label maps largest region primary; all2713single regions secondary; h512/z128',
        'context': 'same fixed valid region IDs and legal original/rot90/drop50/dropall raw52 probes',
        'metrics': ['cosine distance', 'same/different class margin', '1NN', 'P@10', 'variance', 'effective rank'],
        'negative_filter': ['neighbor counts', 'replacement exposure', 'zero-filter rate', 'posthoc single-label proxy precision/recall']},
    'decision_rules': {'context_distance_relative_reduction': .30,
        'context_conditions': 'both drop50/dropall reduce mean h512 distance; no collapse or nearest-neighbor degradation',
        'fusion_exact_P10_absolute_improvement': .01, 'paired_95CI_lower_above': 0.,
        'fusion_cross_nDCG10_noninferiority_lower': -.005,
        'representation_claim': 'CNN-only retrieval must also improve',
        'bootstrap': '2000 paired resamples stratified by37full label combinations; fixed model/gallery',
        'report_both_variants_regardless_of_sign': True},
    'limitations': ['posthoc hypotheses motivated by previously observed baseline/test evidence',
        'single seed does not establish training stability',
        'filter changes hardness as well as possible false-negative rate; no causal isolation of semantic mechanism',
        'single-label wafer labels are region proxies, not pixel-level ground truth',
        'teacher-dependent graph may have low exposure; report effective intervention strength'],
}


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
