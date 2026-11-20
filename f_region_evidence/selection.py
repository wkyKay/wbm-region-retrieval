"""Prespecified valid-only selection and fixed family controls."""
from collections.abc import Mapping
import math
from numbers import Real
from .state import BASELINE, grid, family

RULE = {'split': 'valid', 'metric': 'exact/all/Precision@10', 'objective': 'integer_top10_hits',
        'tie_break': ['baseline first', 'fewer maximum regions', 'weaker rejection', 'variant name'],
        'test': 'one frozen winner from each nonbaseline family; primary is overall valid winner',
        'test_selects_parameters': False}


def tie_key(name):
    spec = grid()[name]
    return (name != BASELINE, 4 if spec['proposal'] == 'base4' else 8,
            {'none': 0, 'q75': 1, 'q90': 2, 'q95': 3}[spec['rejection']], name)


def select(results):
    if not isinstance(results, Mapping) or set(results) != set(grid()):
        raise ValueError('All twelve valid results are required')
    for name, row in results.items():
        if not isinstance(row, Mapping) or not isinstance(row.get('metrics'), Mapping):
            raise ValueError('Selection requires a valid result row and exact metrics')
        hits = row.get('top10_hits')
        if type(hits) is not int or not 0 <= hits <= 5000:
            raise ValueError('Selection requires integer hits from all 500 valid queries')
        precision = row['metrics'].get('Precision@10')
        if (isinstance(precision, bool) or not isinstance(precision, Real)
                or not math.isfinite(precision) or not 0 <= precision <= 1
                or abs(precision-hits/5000) > 1e-12):
            raise ValueError('Integer hits disagree with valid precision')
    order = sorted(results, key=lambda n: (-results[n]['top10_hits'], tie_key(n)))
    winners = {f: next(n for n in order if family(grid()[n]) == f)
               for f in ('baseline', 'rejection', 'completion', 'combined')}
    selected = [winners[f] for f in ('rejection', 'completion', 'combined')]
    return {'selection_rule': RULE, 'winner': order[0], 'valid_order': order,
            'family_winners': winners, 'selected_variants': selected,
            'selected_specs': {n: grid()[n] for n in selected}, 'synthetic': False}
