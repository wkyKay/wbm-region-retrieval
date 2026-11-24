"""Frozen twelve-point additive search using already trained CNNs."""
from pathlib import Path
from f_region_structure.state import now, record, signed, read_signed, seal, verified
from wbm_retrieval.common import digest, file_hash, read_json
from f_region_area_fusion.scoring import frozen_grid
ROOT = Path(__file__).resolve().parents[1]
VERSION = 'cnn_rotation_fusion_v1'
MODELS = ('baseline', 'rotation')
KS = [1, 5, 10, 20, 50, 100, 500]
GRID = {name: spec for name, spec in frozen_grid().items() if spec['family'] == 'add'}
THETA0 = 'add_a0p4_b0p25'
SOURCE = ROOT/'results/20261123T211208+0800__cnn_augmentation'
TIE_RULE = 'maximum integer exact Top10 hits; ties prefer theta0, then lexicographic variant name'
PLAN = {'version': VERSION, 'models': list(MODELS), 'grid': GRID, 'theta0': THETA0,
    'training_required': False, 'source': str(SOURCE.relative_to(ROOT)),
    'features': 'reuse baseline best_nce and rotation epoch100 original-view normalized h512',
    'formula': '(1-beta)*(alpha*C+(1-alpha)*T)+beta*A at region-pair level, then original greedy sqrt-area matching',
    'selection': {'split': 'valid', 'num_queries': 500, 'gallery_count': 3599,
                  'metric': 'integer exact Top10 hits', 'tie_rule': TIE_RULE, 'unique_winner': True},
    'test': {'num_queries': 7199, 'gallery_count': 7199, 'matrix': 'old/new C x theta0/theta_star; alias identical endpoints'},
    'ks': KS, 'candidate_rule': 'full gallery except self; cross additionally excludes exact label combination',
    'all_groups': True, 'fixed_seed': 2026, 'bootstrap_resamples': 2000,
    'limits': ['posthoc study after observing earlier test results', 'single trained seed and fixed gallery',
               'twelve additive weights only; no test tuning or expansion', 'conditional paired intervals do not correct prior exploration']}
sha = file_hash


def code_binding():
    from cnn_augmentation.evaluation import code_identity
    binding = code_identity()
    names = ['paper_c_ablation/inputs.py', 'f_region_area_fusion/inputs.py',
             'f_region_area_fusion/pipeline.py', 'f_region_wbbs/compat.py',
             'f_region_evidence/inputs.py', 'f_region_score_search/inputs.py']
    names += [str(p.relative_to(ROOT)) for p in (ROOT/'cnn_rotation_fusion').glob('*.py')]
    binding.update({str((ROOT/name).resolve()): sha(ROOT/name) for name in names})
    return dict(sorted(binding.items()))


def check_binding(binding):
    for name, expected in binding.items():
        path = Path(name)
        if not path.is_file() or path.is_symlink() or sha(path) != expected:
            raise ValueError('Protected source changed: '+name)


def run_context(output):
    output = Path(output)
    manifest = read_signed(output/'run_manifest.json')
    plan = read_signed(output/'frozen_plan.json')
    if (manifest.get('version') != VERSION or plan.get('plan') != PLAN
            or manifest.get('plan_signature') != plan['signature']
            or manifest.get('code') != code_binding()):
        raise ValueError('Frozen plan or executable code changed')
    return manifest
