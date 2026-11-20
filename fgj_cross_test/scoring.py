"""Original-scale user-specified formulas; no fitting or parameter search."""
import numpy as np
from .state import SPECS


def component_vectors(components):
    names = ('F', 'G', 'joint_distance', 'WBBS')
    if set(components) != set(names):
        raise ValueError('Expected exactly F/G/joint_distance/WBBS')
    result, shape = {}, None
    for name in names:
        a = np.asarray(components[name])
        if a.ndim != 1 or a.dtype.kind not in 'fiu' or not len(a):
            raise ValueError('Invalid component array: '+name)
        a = a.astype(np.float64, copy=False)
        if shape is None: shape = a.shape
        if a.shape != shape or not np.isfinite(a).all() or np.any(a < 0) or np.any(a > 1):
            raise ValueError('Nonfinite, unaligned, or non-unit component: '+name)
        result[name] = a
    return result


def score_all(components):
    c = component_vectors(components)
    j = np.exp(-2.0*c['joint_distance'])
    return {'GJ_ADD': .5*c['G']+.5*j, 'FJ_ADD': .5*c['F']+.5*j,
            'F_ONLY': c['F'], 'G_ONLY': c['G'], 'J_ONLY': j, 'WBBS': c['WBBS']}


def score_variant(components, name, spec):
    if name not in SPECS or spec != SPECS[name]:
        raise ValueError('Only the six fixed user-specified formulas are allowed')
    return score_all(components)[name]
