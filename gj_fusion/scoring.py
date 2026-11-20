"""Frozen whole-map G/J fusion without angle, score fitting or training.

G is the existing clipped whole-map CNN cosine and J=exp(-2*D_H).
The product role keeps the exact historical G*J operation. Its square root
is the alpha=.5 weighted geometric score, with the same mathematical ranking.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real

import numpy as np


STRUCTURE_STRENGTH = 2.0


def frozen_grid():
    """Return fresh specifications for the eight deduplicated score roles."""
    return {
        "J_only": {"family": "endpoint", "alpha": 0.0},
        "G_only": {"family": "endpoint", "alpha": 1.0},
        "add_a0p25": {"family": "additive", "alpha": 0.25},
        "add_a0p5": {"family": "additive", "alpha": 0.5},
        "add_a0p75": {"family": "additive", "alpha": 0.75},
        "geo_a0p25": {"family": "geometric", "alpha": 0.25},
        "product": {"family": "product", "alpha": 0.5},
        "geo_a0p75": {"family": "geometric", "alpha": 0.75},
    }


def checked_spec(spec):
    """Accept only canonical members of the fixed eight-role experiment."""
    if isinstance(spec, Mapping) and set(spec) == {"family", "alpha"}:
        alpha = spec["alpha"]
        if (isinstance(spec["family"], str) and isinstance(alpha, Real)
                and not isinstance(alpha, (bool, np.bool_)) and math.isfinite(alpha)):
            for expected in frozen_grid().values():
                if spec["family"] == expected["family"] and alpha == expected["alpha"]:
                    return expected
    raise ValueError("Score specification is outside the frozen G/J grid")


def _unit_vector(value, label):
    value = np.asarray(value)
    if (value.ndim != 1 or value.size == 0 or value.dtype.kind not in "fiu"
            or not np.isfinite(value).all() or np.any(value < 0.0)
            or np.any(value > 1.0)):
        raise ValueError(f"{label} must be a nonempty finite real vector in [0,1]")
    return np.asarray(value, dtype=np.float64)


def score_variant(global_scores, distances, spec):
    """Score one query's full candidate vector, with no candidate filtering.

    Both components must be finite, real, one-dimensional and in [0,1].
    In an active geometric or product branch, G=0 gives zero without epsilon
    smoothing; additive fusion can instead retain positive structural evidence.
    The endpoints bypass powers, avoiding any dependence on a 0**0 convention.
    """
    spec = checked_spec(spec)
    global_scores = _unit_vector(global_scores, "G")
    distances = _unit_vector(distances, "D_H")
    if global_scores.shape != distances.shape:
        raise ValueError("G and D_H must have identical candidate shapes")
    joint = np.exp(-STRUCTURE_STRENGTH * distances)
    family, alpha = spec["family"], spec["alpha"]
    if family == "endpoint":
        return joint if alpha == 0.0 else global_scores.copy()
    if family == "product":
        return global_scores * joint
    if family == "additive":
        return alpha * global_scores + (1.0 - alpha) * joint
    return np.power(global_scores, alpha) * np.power(joint, 1.0 - alpha)
