"""Original-map TV distributions, independent of labels and CNN regions.

The vendored TV primitives are read-only imports. This module never calls cc8,
Top-4 selection, crop/resize, a learned model, or any label-dependent function.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.proposed.f_segmentation.tensor_voting import (
    retained_points,
    tensor_saliencies,
    tensor_vote,
)


CONFIG_PATH = Path(__file__).with_name("config.json")
FAMILIES = ("R", "T", "I", "J")
ROUNDING_TOLERANCE = 1e-12


def load_settings(path=None):
    """Load a versioned recipe; callers bind the complete object to a signature."""
    with Path(path or CONFIG_PATH).open(encoding="utf-8") as handle:
        result = json.load(handle)
    return validate_settings(result)


def validate_settings(settings):
    """Reject ambiguous recipes rather than silently choosing missing settings."""
    result = dict(settings)
    required = {
        "schema_version", "recipe", "sigma_d", "noise_threshold", "block_size",
        "bins", "axis_order", "normalization", "soft_bins", "empty_policy",
        "distance", "families", "lambda_grid", "ks", "tie_seed", "selection_metric",
        "selection_tie_rule", "baseline", "posthoc_design", "test_parameter_search",
        "training_required",
    }
    if set(result) != required:
        raise ValueError(f"Unexpected settings keys: missing={required-set(result)}, extra={set(result)-required}")
    fixed = {
        "schema_version": 1,
        "recipe": "b1_tv_joint_v1",
        "bins": [8, 4, 4],
        "axis_order": ["rho", "stick", "ball"],
        "normalization": "per_image_channel_max_before_denoising; zero_channel_to_zero",
        "soft_bins": "uniform_endpoint_nodes; trilinear_hat_weights; equal_point_mass",
        "empty_policy": "both_empty_distance_zero; one_empty_distance_one",
        "distance": "ordinary_hellinger_float64",
        "families": list(FAMILIES),
        "selection_metric": "exact/all/Precision@10",
        "selection_tie_rule": "smaller_strength_including_zero",
        "baseline": "(0.5*F_optimal+0.5*G)*exp(-0.5*D_angle)",
        "posthoc_design": True,
        "test_parameter_search": False,
        "training_required": False,
    }
    if any(result[key] != value for key, value in fixed.items()):
        raise ValueError("This implementation only supports the declared B1 v1 recipe")
    for key in ("sigma_d", "noise_threshold"):
        if isinstance(result[key], bool) or not np.isscalar(result[key]) or not np.isfinite(result[key]):
            raise ValueError(f"Invalid {key}")
    if result["sigma_d"] <= 0 or not 0 <= result["noise_threshold"] <= 1:
        raise ValueError("Expected positive TV scale and threshold in [0,1]")
    for key in ("block_size", "tie_seed"):
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < (1 if key == "block_size" else 0):
            raise ValueError(f"Invalid {key}")
    grid = result["lambda_grid"]
    if (not isinstance(grid, list) or not grid or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in grid)
            or not np.isfinite(grid).all() or min(grid) < 0 or grid != sorted(set(grid)) or grid[0] != 0):
        raise ValueError("lambda_grid must be unique sorted nonnegative finite values starting at zero")
    ks = result["ks"]
    if (not isinstance(ks, list) or not ks or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in ks)
            or ks != sorted(set(ks)) or 10 not in ks):
        raise ValueError("ks must be unique sorted positive integers including 10")
    return result


def _unit_values(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < -ROUNDING_TOLERANCE) or np.any(values > 1 + ROUNDING_TOLERANCE):
        raise ValueError("Descriptor coordinates must be finite in [0,1]")
    return np.clip(values, 0., 1.)


def hat_weights(values, nodes):
    """Partition of unity on endpoint nodes k/(nodes-1), including 0 and 1."""
    if isinstance(nodes, bool) or not isinstance(nodes, (int, np.integer)) or nodes < 2:
        raise ValueError("At least two soft-bin nodes are required")
    values = _unit_values(values)
    return np.maximum(1. - np.abs(values[..., None] * (nodes-1) - np.arange(nodes)), 0.)


def soft_histogram(points, bins=(8, 4, 4)):
    """Equal-mass trilinear soft distribution in rho, stick, ball axis order."""
    points = _unit_values(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected N by 3 normalized descriptor points")
    if len(bins) != 3:
        raise ValueError("Expected three histogram axes")
    weights = [hat_weights(points[:, i], size) for i, size in enumerate(bins)]
    if len(points) == 0:
        return np.zeros(tuple(bins), dtype=np.float64)
    histogram = np.einsum("ni,nj,nk->ijk", *weights, optimize=True) / len(points)
    if not np.isclose(histogram.sum(), 1., rtol=0, atol=ROUNDING_TOLERANCE):
        raise ValueError("Soft histogram lost probability mass")
    return histogram


def describe(raw, settings=None):
    """Compute a single wafer's structure from all original defective pixels."""
    settings = load_settings() if settings is None else validate_settings(settings)
    raw = np.asarray(raw)
    if raw.ndim != 2 or min(raw.shape, default=0) == 0 or not np.isin(raw, [0, 1, 2, 3]).all():
        raise ValueError("Expected a nonempty 2D raw wafer with codes 0/1/2/3")
    support = np.argwhere((raw == 1) | (raw == 2)).astype(np.float64)
    if len(support) == 0:
        raise ValueError("Wafer has no valid support")
    points = np.argwhere(raw == 2).astype(np.float64)
    center = (support.min(0) + support.max(0)) / 2
    radius = float(np.linalg.norm(support-center, axis=1).max())
    if len(points):
        saliency = tensor_saliencies(tensor_vote(points, settings["sigma_d"], settings["block_size"]))
        maxima = saliency.max(axis=0)
        normalized = np.divide(saliency, maxima, out=np.zeros_like(saliency), where=maxima > 0)
        keep = retained_points(saliency, settings["noise_threshold"])
        rho = np.linalg.norm(points-center, axis=1)/radius if radius > 0 else np.zeros(len(points))
        features = np.column_stack((rho, normalized))[keep]
    else:
        maxima = np.zeros(2, dtype=np.float64)
        features = np.empty((0, 3), dtype=np.float64)
    histogram = soft_histogram(features, tuple(settings["bins"])).reshape(-1)
    return {
        "joint128": histogram,
        "empty": len(features) == 0,
        "original_count": len(points),
        "retained_count": len(features),
        "saliency_max": maxima,
    }


def _probabilities(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim not in (1, 2) or not values.shape[-1] or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Expected finite nonnegative probability rows")
    mass = values.sum(axis=-1)
    valid = np.isclose(mass, 1., rtol=0, atol=ROUNDING_TOLERANCE) | (mass == 0)
    if not np.all(valid):
        raise ValueError("Probability rows must have mass one (or zero with an explicit empty flag)")
    return values


def distributions(joint):
    """Same-source marginals and shape-position independence control.

    Leading dimension is the number of wafers; input/output also accept one row.
    I retains the complete stick/ball joint marginal, not a product of 3 marginals.
    """
    joint = _probabilities(joint)
    if joint.shape[-1] != 128:
        raise ValueError("B1 v1 requires a 128-dimensional joint distribution")
    shaped = joint.reshape(*joint.shape[:-1], 8, 4, 4)
    radial = shaped.sum(axis=(-2, -1))
    shape = shaped.sum(axis=-3)
    independent = radial[..., :, None, None] * shape[..., None, :, :]
    return {"R": radial, "T": shape.reshape(*joint.shape[:-1], 16),
            "I": independent.reshape(*joint.shape[:-1], 128), "J": joint}


def _empty_flags(q, candidates, q_empty=False, c_empty=None):
    if not isinstance(q_empty, (bool, np.bool_)):
        raise ValueError("q_empty must be boolean")
    flags = np.zeros(len(candidates), dtype=bool) if c_empty is None else np.asarray(c_empty)
    if flags.dtype != bool or flags.shape != (len(candidates),):
        raise ValueError("c_empty must have one boolean per candidate")
    if bool(q.sum() == 0) != bool(q_empty) or not np.array_equal(candidates.sum(axis=-1) == 0, flags):
        raise ValueError("Empty flags must agree with zero-mass rows")
    return flags


def hellinger(q, candidates, q_empty=False, c_empty=None):
    """Ordinary Hellinger; an explicit empty atom gives both-empty=0, one-empty=1."""
    q, candidates = _probabilities(q), _probabilities(candidates)
    if q.ndim != 1 or candidates.ndim != 2 or q.shape[0] != candidates.shape[1]:
        raise ValueError("Expected one query distribution and a matrix of candidates")
    flags = _empty_flags(q, candidates, q_empty, c_empty)
    difference = np.sqrt(candidates) - np.sqrt(q)
    result = np.sqrt(np.maximum(.5 * np.sum(difference*difference, axis=1), 0.))
    result = np.clip(result, 0., 1.)
    result[np.logical_xor(q_empty, flags)] = 1.
    result[np.logical_and(q_empty, flags)] = 0.
    return result
