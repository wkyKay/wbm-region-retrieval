"""Select only from complete cross-valid reports; never from held-out test."""
import math
from .state import BASELINE, GLOBAL, SELECTION_RULE, grid


def tie_key(name):
    spec = grid()[name]
    if name == BASELINE:
        return (0, 0., 0, 0., name)
    return (1, abs(spec['cnn_weight']-.4), spec['point_mode'] != 'all_points', spec['count_alpha'], name)


def select(results):
    expected = {*grid(), GLOBAL}
    if set(results) != expected:
        raise ValueError('Selection requires all thirteen regional recipes and the global WBBS benchmark')
    values = {}
    for name in expected:
        value = results[name]['coverage_metrics']['Coverage-nDCG@10']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid cross-valid selection metric')
        values[name] = float(value)
    def choose(names):
        best = max(values[n] for n in names)
        return min((n for n in names if best-values[n] <= SELECTION_RULE['tie_tolerance']), key=tie_key)
    winner = choose(list(grid()))
    best_regional = choose([n for n in grid() if n != BASELINE])
    controls = [n for n,s in grid().items() if s.get('cnn_weight') == .4]
    selected = list(dict.fromkeys([*controls, best_regional]))
    return {'selection_rule': SELECTION_RULE, 'winner': winner,
            'best_regional_wbbs': best_regional,
            'fixed_controls': controls, 'selected_variants': selected,
            'selected_specs': {n: grid()[n] for n in selected},
            'valid_order': sorted(grid(), key=lambda n: (-values[n], tie_key(n))),
            'selection_values': values, 'test_reselection': False}
