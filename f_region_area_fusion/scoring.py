"""Frozen 29-candidate area fusion on actual cc8 region tokens.

C is the existing clipped CNN cosine, T=exp(-0.5*Hellinger), and
A=exp(-0.5*abs(log(rq)-log(rc))/1.5). No score normalization is fitted.
Entropy is per query REGION over all real gallery regions, excluding every
region of that query's map; its three weights are shared across candidates.
Padding and empty maps have no tokens here and are handled by the matcher.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real

import numpy as np

from f_region_score_search.scoring import OLD_SPEC
from f_region_structure.scoring import RegionPairScorer, _features, _indices, _probabilities


BASELINE = "mul_a0p5_b0p15"
EMPTY_POLICY = "reject_missing_all_point_structure_and_padded_tokens"
ENTROPY_RULE = {
    "level": "query_region",
    "reference": "all_real_gallery_regions_excluding_same_map",
    "channels": ["C", "T", "A"],
    "probability": "similarity_divided_by_channel_sum",
    "entropy": "negative_sum_p_log_p_natural_log",
    "weights": "softmax_negative_entropy_temperature_1",
    "zero_mass_channel": "zero_weight",
    "constant_positive_channel": "retain_entropy_log_reference_count",
    "all_zero_channels": "equal_weights",
    "score_rescaling": "none",
}


def _number(value):
    return str(float(value)).replace(".", "p")


def frozen_grid():
    """Return fresh ordered specifications, with beta=0 deduplicated."""
    grid = {f"no_area_a{_number(a)}": {"family": "no_area", "alpha": a, "beta": 0.}
            for a in (.4, .5, .6)}
    for family in ("add", "mul"):
        for alpha in (.4, .5, .6):
            for beta in (.05, .1, .15, .25):
                grid[f"{family}_a{_number(alpha)}_b{_number(beta)}"] = {
                    "family": family, "alpha": alpha, "beta": beta}
    grid["equal3"] = {"family": "equal3"}
    grid["entropy"] = {"family": "entropy"}
    return grid


def checked_spec(spec):
    """Accept only canonical members of the predeclared grid."""
    if isinstance(spec, Mapping):
        for expected in frozen_grid().values():
            if set(spec) != set(expected):
                continue
            if all((isinstance(spec[k], str) and spec[k] == target)
                   if isinstance(target, str) else
                   (isinstance(spec[k], Real) and not isinstance(spec[k], (bool, np.bool_))
                    and math.isfinite(spec[k]) and spec[k] == target)
                   for k, target in expected.items()):
                return expected
    raise ValueError("Score specification is outside the frozen 29-candidate grid")


def _owners(value, n):
    value = np.asarray(value)
    if value.shape != (n,) or value.dtype.kind not in "iu" or np.any(value < 0):
        raise ValueError("Owners must be nonnegative integer map IDs for actual region tokens")
    return value


def _channels(parts):
    try:
        channels = tuple(np.asarray(parts[k]) for k in
                         ("cosine", "structure_similarity", "area_similarity"))
    except (KeyError, TypeError) as error:
        raise ValueError("Expected C, T and A regional components") from error
    c, t, a = channels
    if c.ndim != 2 or any(x.shape != c.shape for x in channels):
        raise ValueError("Components must have identical query-by-region shapes")
    for key, value, lower in (("C", c, 0.), ("T", t, math.exp(-.5)), ("A", a, 0.)):
        if (value.dtype.kind not in "fi" or not np.isfinite(value).all()
                or np.any(value < lower) or np.any(value > 1.)
                or (key == "A" and np.any(value == 0.))):
            raise ValueError(f"Invalid {key} component")
    return channels


def entropy_weights(channels, reference_mask):
    """Return Q by 3 weights; mask selects full-gallery real reference tokens.

    The helper allows zero channels for an explicit mathematical edge policy.
    Formal all-point T and positive relative-area A have positive evidence.
    A positive constant channel is retained, not dropped as zero-variance.
    """
    if not isinstance(channels, (list, tuple)) or len(channels) != 3:
        raise ValueError("Entropy requires three channel matrices")
    channels = tuple(np.asarray(value, dtype=np.float64) for value in channels)
    shape = channels[0].shape
    mask = np.asarray(reference_mask)
    if (len(shape) != 2 or any(value.shape != shape for value in channels)
            or mask.shape != shape or mask.dtype.kind != "b"):
        raise ValueError("Entropy channels and boolean reference mask must share Q by N shape")
    if any(not np.isfinite(value).all() or np.any(value < 0.) or np.any(value > 1.)
           for value in channels):
        raise ValueError("Entropy similarities must be finite and in [0,1]")
    entropies = np.zeros((shape[0], 3), dtype=np.float64)
    active = np.zeros_like(entropies, dtype=bool)
    for k, value in enumerate(channels):
        selected = np.where(mask, value, 0.)
        mass = selected.sum(axis=1, keepdims=True)
        active[:, k] = mass[:, 0] > 0.
        p = np.divide(selected, mass, out=np.zeros_like(selected), where=mass > 0.)
        logp = np.log(p, out=np.zeros_like(p), where=p > 0.)
        entropies[:, k] = -(p * logp).sum(axis=1)
    any_active = active.any(axis=1, keepdims=True)
    minimum = np.min(np.where(active, entropies, np.inf), axis=1, keepdims=True)
    minimum = np.where(any_active, minimum, 0.)
    weights = np.exp(np.where(active, minimum - entropies, -np.inf))
    total = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, total, out=np.zeros_like(weights), where=total > 0.)
    return np.where(any_active, weights, 1. / 3.)


class AreaPairScorer(RegionPairScorer):
    """Reuse the frozen C/T computation while deriving A from raw geometry."""

    def __init__(self, features, geometry, descriptors):
        super().__init__(features, geometry, descriptors)
        self.owners = _owners(geometry["owners"], self.n)
        if not self.valid["all_points"].all():
            raise ValueError("All-point structure is required for every actual region token")

    def components(self, query_indices, specs=None):
        if specs is not None:
            if not isinstance(specs, Mapping) or not specs:
                raise ValueError("Expected a nonempty name-to-spec mapping")
            for spec in specs.values():
                checked_spec(spec)
        q = _indices(query_indices, self.n)
        parts = super().components(q, specs={"selected": OLD_SPEC})
        parts["structure_similarity"] = np.exp(-.5 * parts["structure"]["all_points"]["distance"])
        # Derive A directly, preserving the source area-distance operation order.
        distance = .5 * np.abs(np.log(self.relative_area[q])[:, None]
                                - np.log(self.relative_area)[None])
        parts["area_similarity"] = np.exp(-distance / 1.5)
        reference = self.owners[q, None] != self.owners[None, :]
        parts["entropy_weights"] = entropy_weights(_channels(parts), reference)
        parts["entropy_reference_counts"] = reference.sum(axis=1)
        return parts

    def scores(self, query_indices, spec):
        return score_components(self.components(query_indices), spec)

    @staticmethod
    def score_components(parts, spec):
        return score_components(parts, spec)


def _weights(parts, rows):
    try:
        value = np.asarray(parts["entropy_weights"])
    except (KeyError, TypeError) as error:
        raise ValueError("Entropy requires weights computed over the full gallery") from error
    if (value.shape != (rows, 3) or value.dtype.kind not in "fi"
            or not np.isfinite(value).all() or np.any(value < 0.) or np.any(value > 1.)
            or not np.allclose(value.sum(axis=1), 1., rtol=0., atol=1e-12)):
        raise ValueError("Invalid entropy weights")
    return value


def score_components(parts, spec):
    spec = checked_spec(spec)
    c, t, a = _channels(parts)
    family = spec["family"]
    if family == "equal3":
        score = (c + t + a) / 3.
    elif family == "entropy":
        w = _weights(parts, len(c))
        score = w[:, 0, None] * c + w[:, 1, None] * t + w[:, 2, None] * a
    else:
        alpha, beta = spec["alpha"], spec["beta"]
        mixed = alpha * c + (1. - alpha) * t
        if family == "no_area":
            score = mixed
        elif family == "add":
            score = (1. - beta) * mixed + beta * a
        else:
            # For alpha=.5,beta=.15 this exactly preserves old B*(.5*C+.5*T).
            score = ((1. - beta) + beta * a) * mixed
    return np.clip(score, 0., 1.)


def independent_components(query_indices, features, relative_area, joint_all, owners,
                           *, candidate_indices=None, with_entropy=True):
    """Independent pair arithmetic and entropy, without production components.

    Full source arrays are required when with_entropy=True. Candidate selection
    is applied AFTER full-gallery weights are calculated. with_entropy=False
    permits a small source subset for fixed-weight formula audits. Returned
    components can be reused by independent_score_components for the full grid.
    """
    h = _features(features, width=512).astype(np.float64)
    n = len(h)
    q = _indices(query_indices, n)
    area = np.asarray(relative_area, dtype=np.float64)
    if (area.shape != (n,) or not np.isfinite(area).all()
            or np.any(area <= 0.) or np.any(area > 1.)):
        raise ValueError("Relative areas must be in (0,1]")
    p = _probabilities(joint_all, n, 128, "all-point J", False)
    owners = _owners(owners, n)
    selected = np.arange(n) if candidate_indices is None else _indices(candidate_indices, n)
    if not isinstance(with_entropy, bool):
        raise ValueError("with_entropy must be boolean")
    columns = np.arange(n) if with_entropy else selected
    c = np.empty((len(q), len(columns)), dtype=np.float64)
    t, a = np.empty_like(c), np.empty_like(c)
    roots = np.sqrt(p)
    weights = np.empty((len(q), 3), dtype=np.float64)
    for row, i in enumerate(q):
        # Separate row-wise norm and scalar dot operations avoid the production
        # GEMM and batched squared-difference reduction, without Python bin loops.
        hellinger = np.linalg.norm(roots[columns] - roots[i], axis=1) / math.sqrt(2.)
        for col, j in enumerate(columns):
            c[row, col] = min(1., max(0., float(np.dot(h[i], h[j]))))
            t[row, col] = math.exp(-.5 * min(1., max(0., float(hellinger[col]))))
            a[row, col] = math.exp(-abs(math.log(float(area[i]))
                                          - math.log(float(area[j]))) / 3.)
        if with_entropy:
            reference = owners[columns] != owners[i]
            hs = []
            for channel in (c, t, a):
                values = channel[row, reference]
                mass = math.fsum(values)
                hs.append(-math.fsum((float(s) / mass) * math.log(float(s) / mass)
                                     for s in values if s > 0.) if mass > 0. else None)
            active = [value for value in hs if value is not None]
            if active:
                minimum = min(active)
                z = [math.exp(minimum - value) if value is not None else 0. for value in hs]
                weights[row] = [value / math.fsum(z) for value in z]
            else:
                weights[row] = 1. / 3.
    if with_entropy:
        c, t, a = (value[:, selected] for value in (c, t, a))
    result = {"cosine": c, "structure_similarity": t, "area_similarity": a}
    if with_entropy:
        result["entropy_weights"] = weights
    return result


def independent_score_components(parts, spec):
    """Scalar fusion formulas independent of the vectorized production scorer."""
    spec = checked_spec(spec)
    c, t, a = _channels(parts)
    w = _weights(parts, len(c)) if spec["family"] == "entropy" else None
    result = np.empty_like(c, dtype=np.float64)
    for row in range(len(c)):
        for col in range(c.shape[1]):
            cv, tv, av = float(c[row, col]), float(t[row, col]), float(a[row, col])
            family = spec["family"]
            if family == "equal3":
                score = (cv + tv + av) / 3.
            elif family == "entropy":
                score = math.fsum((float(w[row, 0]) * cv, float(w[row, 1]) * tv,
                                   float(w[row, 2]) * av))
            else:
                alpha, beta = spec["alpha"], spec["beta"]
                mixed = alpha * cv + (1. - alpha) * tv
                score = (mixed if family == "no_area" else
                         (1. - beta) * mixed + beta * av if family == "add" else
                         (1. - beta + beta * av) * mixed)
            result[row, col] = min(1., max(0., score))
    return result


def independent_pair_scores(query_indices, features, relative_area, joint_all, owners, spec,
                            *, candidate_indices=None):
    spec = checked_spec(spec)
    parts = independent_components(query_indices, features, relative_area, joint_all, owners,
                                   candidate_indices=candidate_indices,
                                   with_entropy=spec["family"] == "entropy")
    return independent_score_components(parts, spec)
