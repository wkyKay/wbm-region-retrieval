"""Scientific design, isolated time identities, and immutable metadata."""
from pathlib import Path
from datetime import datetime
import re

from f_region_structure.state import now, record, signed, read_signed, seal, verified, protection
from wbm_retrieval.common import digest, read_json

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'f_region_evidence_v1'
KS = [1, 5, 10, 20, 50, 100, 500]
BASELINE = 'base4_none'
QUANTILES = {'q75': .75, 'q90': .90, 'q95': .95}
REFERENCE = {'size': 4096, 'seed': 2026, 'source': 'frozen_original_train_top4',
             'sampling': 'one uniformly sampled region per sampled nonempty training wafer',
             'labels_used': False, 'quantile_method': 'linear'}


def grid():
    return {p+'_'+r: {'proposal': p, 'rejection': r,
            'formula': '0.75*(0.4*C+0.6*exp(-0.5*H))+0.25*A',
            'matching': 'original_greedy_query_sqrt_area',
            'rejection_formula': 'identity' if r == 'none' else 'max((S-tau_i)/(1-tau_i),0); tau_i==1 gives zero',
            'reference_quantile': QUANTILES.get(r)}
            for p in ('base4', 'top8', 'residual8') for r in ('none', *QUANTILES)}


def family(spec):
    if spec['proposal'] == 'base4':
        return 'baseline' if spec['rejection'] == 'none' else 'rejection'
    return 'completion' if spec['rejection'] == 'none' else 'combined'


def evaluation_signature(context_signature, stage, name):
    return digest({'context': context_signature, 'stage': stage, 'variant': name, 'spec': grid()[name]})


def identity(output, stage, name, signature):
    output = Path(output)
    split = 'valid' if stage == 'valid' else 'test'
    protocol = 'coverage_cross_combination_v1' if stage == 'cross_test' else 'retrieval_v1'
    key = 'f_region_evidence_'+re.sub(r'[^A-Za-z0-9_.-]', '_', output.name)+'_'+stage+'_'+name
    path = output/'identities'/(stage+'__'+name+'.json')
    if path.exists():
        value = read_json(path)
        time = datetime.fromisoformat(value['started_at'])
    else:
        time = now()
    expected = {'schema_version': 1, 'key': key, 'run_id': time.strftime('%Y%m%dT%H%M%S%z')+'__'+key,
                'started_at': time.isoformat(), 'split': split, 'stage': stage,
                'protocol': protocol, 'variant': name, 'evaluation_signature': signature}
    record(path, expected)
    return expected
