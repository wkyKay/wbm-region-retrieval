"""Prespecified scientific grid, time identities and immutable output helpers."""
from pathlib import Path
from datetime import datetime
import re
from f_region_structure.state import now, record, signed, read_signed, seal, verified, protection
from wbm_retrieval.common import digest, read_json

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'f_region_wbbs_v1'
KS = [1, 5, 10, 20, 50, 100, 500]
BASELINE = 'region_j'
GLOBAL = 'global_wbbs'
WEIGHTS = (.25, .4, .6)
SELECTION_RULE = {
    'split': 'valid', 'protocol': 'coverage_cross_combination_v1',
    'metric': 'coverage/all/Coverage-nDCG@10', 'tie_tolerance': 1e-12,
    'tie_break': ['original J first', 'CNN weight closest to 0.4',
                  'all original region points first', 'smaller count penalty', 'variant name'],
    'primary_candidates': 'original regional J and twelve regional WBBS recipes; global WBBS is benchmark only',
    'test': 'four fixed CNN-weight-0.4 controls plus best regional WBBS, deduplicated',
    'test_selects_parameters': False,
}
SETTINGS = {'sigma_d': 9., 'noise_threshold': .32, 'alpha': 5., 'mu': 1.,
            'total_dies': 2061, 'block_size': 256}


def grid():
    result = {BASELINE: {'family': 'j_baseline'}}
    for prefix, mode in [('all', 'all_points'), ('denoised', 'wbbs_denoised')]:
        for label, penalty in [('bb', 0.), ('wbbs', 5.)]:
            for weight, suffix in [(.25, '025'), (.4, '04'), (.6, '06')]:
                result[f'{prefix}_{label}_a{suffix}'] = {
                    'family': 'wbbs', 'point_mode': mode, 'count_alpha': penalty,
                    'cnn_weight': weight, 'area_weight': .25}
    return result


def evaluation_specs(stage, selection=None):
    if stage == 'valid':
        return grid()
    if stage == 'cross_valid':
        return {**grid(), GLOBAL: {'family': 'global_wbbs'}}
    if stage in ('test', 'cross_test') and selection is not None:
        return selection['selected_specs']
    raise ValueError('Test requires a frozen valid selection')


def evaluation_signature(context_signature, stage, name, spec=None):
    if spec is None:
        spec = {'family': 'global_wbbs'} if name == GLOBAL else grid()[name]
    return digest({'context': context_signature, 'stage': stage, 'variant': name, 'spec': spec})


def identity(output, stage, name, signature):
    output = Path(output)
    split = 'valid' if stage.endswith('valid') else 'test'
    protocol = 'coverage_cross_combination_v1' if stage.startswith('cross_') else 'retrieval_v1'
    key = 'f_region_wbbs_'+re.sub(r'[^A-Za-z0-9_.-]', '_', output.name)+'_'+stage+'_'+name
    path = output/'identities'/(stage+'__'+name+'.json')
    stamp = datetime.fromisoformat(read_json(path)['started_at']) if path.exists() else now()
    value = {'schema_version': 1, 'key': key,
             'run_id': stamp.strftime('%Y%m%dT%H%M%S%z')+'__'+key,
             'started_at': stamp.isoformat(), 'split': split, 'stage': stage,
             'protocol': protocol, 'variant': name, 'evaluation_signature': signature}
    record(path, value)
    return value
