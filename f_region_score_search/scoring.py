"""Frozen valid-only controls for arithmetic versus multiplicative region scores.

All variants retain the original area factor B and use clipped CNN cosine C.
T = exp(-0.5 * Hellinger(Pq, Pc)) uses the same all-point 128D descriptor.
Missing structure falls back to C*B, except the structure-only endpoint, which
returns zero without structure evidence. Padded map regions are masked by the
caller; these functions score only actual region tokens.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real

import numpy as np

from f_region_structure.scoring import (
    frozen_grid as original_grid,
    score_components as original_score_components,
)


OLD_SPEC = original_grid()["replace_radial_all_points_l0p5"]
EMPTY_POLICY = "missing_structure_falls_back_to_CB_except_j_only_zero"


def _number(value):
    return str(float(value)).replace(".", "p")


def frozen_grid():
    """Return fresh, insertion-ordered specifications for all 15 candidates."""
    common = {"descriptor": "all_points", "strength": .5, "area": True}
    grid = {"product": {"family": "product", **common}}
    for alpha in (.25, .5, .75):
        for eta in (.25, .5, 1.):
            grid[f"mixed_a{_number(alpha)}_e{_number(eta)}"] = {
                "family": "mixed", **common, "alpha": alpha, "eta": eta,
                "gate_zero": False,
            }
    for alpha in (.25, .5, .75):
        grid[f"gated_add_a{_number(alpha)}"] = {
            "family": "mixed", **common, "alpha": alpha, "eta": 1.,
            "gate_zero": True,
        }
    for family in ("cnn_only", "j_only"):
        grid[family] = {"family": family, **common}
    return grid


def checked_spec(spec):
    """Reject undeclared variants and return their canonical specification."""
    if isinstance(spec, Mapping):
        for expected in frozen_grid().values():
            if set(spec) != set(expected):
                continue
            good = True
            for key, target in expected.items():
                value = spec[key]
                if isinstance(target, bool):
                    good &= isinstance(value, bool) and value == target
                elif isinstance(target, float):
                    good &= (isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
                             and math.isfinite(value) and value == target)
                else:
                    good &= isinstance(value, str) and value == target
                if not good:
                    break
            if good:
                return expected
    raise ValueError("Score specification is outside the frozen 15-candidate grid")


def _components(parts):
    try:
        c = np.asarray(parts["cosine"])
        b = np.asarray(parts["area"])
        structure = parts["structure"]["all_points"]
        distance = np.asarray(structure["distance"])
        valid = np.asarray(structure["valid"])
    except (KeyError, TypeError) as error:
        raise ValueError("Expected frozen regional all-point score components") from error
    if c.ndim != 2 or any(x.shape != c.shape for x in (b, distance, valid)):
        raise ValueError("Score components must have identical query-by-region shapes")
    for name, value, lower, upper in (("cosine", c, 0., 1.), ("area", b, .85, 1.),
                                      ("Hellinger distance", distance, 0., 1.)):
        if (value.dtype.kind not in "fi" or not np.isfinite(value).all()
                or np.any(value < lower) or np.any(value > upper)):
            raise ValueError(f"Invalid {name} component")
    if valid.dtype.kind != "b":
        raise ValueError("Structure validity must be a boolean matrix")
    return c, b, distance, valid


def score_components(parts, spec):
    """Score components from RegionPairScorer using specs={'selected': OLD_SPEC}."""
    spec = checked_spec(spec)
    c, b, distance, valid = _components(parts)
    family = spec["family"]
    if family == "product":
        # Keep the old implementation and its operation order for exact replay.
        return original_score_components(parts, OLD_SPEC)
    cb = c * b
    if family == "cnn_only":
        return cb
    t = np.exp(-.5 * distance)
    if family == "j_only":
        return np.where(valid, t * b, 0.)
    alpha, eta = spec["alpha"], spec["eta"]
    score = b * ((1. - eta) * c * t + eta * (alpha * c + (1. - alpha) * t))
    score = np.where(valid, score, cb)
    if spec["gate_zero"]:
        score = np.where(c > 0., score, 0.)
    return np.clip(score, 0., 1.)


def independent_pair_scores(query_indices, features, relative_area, joint, spec):
    """Independent scalar formulas for audits, without production score helpers."""
    spec = checked_spec(spec)
    h = np.asarray(features)
    area = np.asarray(relative_area, dtype=np.float64)
    p = np.asarray(joint, dtype=np.float64)
    q = np.asarray(query_indices)
    if (h.ndim != 2 or h.shape[1] != 512 or h.dtype.kind != "f"
            or not np.isfinite(h).all()):
        raise ValueError("Expected a finite floating-point N by 512 CNN matrix")
    n = len(h)
    norms = np.linalg.norm(h.astype(np.float64), axis=1)
    if np.any((norms != 0.) & ~np.isclose(norms, 1., rtol=1e-5, atol=1e-6)):
        raise ValueError("CNN rows must be normalized or zero")
    if (area.shape != (n,) or not np.isfinite(area).all()
            or np.any(area <= 0.) or np.any(area > 1.)):
        raise ValueError("Relative areas must be in (0, 1]")
    if p.shape != (n, 128) or not np.isfinite(p).all() or np.any(p < 0.):
        raise ValueError("Expected finite nonnegative N by 128 structure probabilities")
    mass = p.sum(axis=1)
    if not np.all((mass == 0.) | np.isclose(mass, 1., rtol=0., atol=1e-10)):
        raise ValueError("Structure rows must sum to one or be empty")
    if (q.ndim != 1 or q.dtype.kind not in "iu" or np.any(q < 0) or np.any(q >= n)
            or len(np.unique(q)) != len(q)):
        raise ValueError("Query indices must be a unique in-range integer vector")
    result = np.empty((len(q), n), dtype=np.float64)
    h64 = h.astype(np.float64)
    roots = np.sqrt(p)
    for row, i in enumerate(q):
        for j in range(n):
            c = min(1., max(0., float(np.dot(h64[i], h64[j]))))
            da = .5 * abs(math.log(float(area[i])) - math.log(float(area[j])))
            b = .85 + .15 * math.exp(-da / 1.5)
            has_structure = bool(mass[i] > 0. and mass[j] > 0.)
            family = spec["family"]
            if family == "cnn_only" or not has_structure:
                value = 0. if family == "j_only" else c * b
            else:
                h_distance = math.sqrt(.5 * math.fsum(
                    (float(roots[i, k]) - float(roots[j, k])) ** 2 for k in range(128)))
                t = math.exp(-.5 * min(1., max(0., h_distance)))
                if family == "product":
                    value = c * b * t
                elif family == "j_only":
                    value = t * b
                else:
                    alpha, eta = spec["alpha"], spec["eta"]
                    value = b * ((1. - eta) * c * t
                                 + eta * (alpha * c + (1. - alpha) * t))
            if spec.get("gate_zero", False) and c <= 0.:
                value = 0.
            result[row, j] = min(1., max(0., value))
    return result
