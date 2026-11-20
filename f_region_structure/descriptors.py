"""Original-map tensor voting pooled within fixed F connected components.

Voting, channel normalization and denoising use every original defect pixel,
including pixels not assigned to an F token. Only pooling is region-specific.
The historical B1 descriptor implementation and all source artifacts are read-only.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from b1_structure_fusion.descriptors import load_settings, soft_histogram, validate_settings
from experiments.proposed.f_segmentation.tensor_voting import (
    retained_points,
    tensor_saliencies,
    tensor_vote,
)


def _validate_inputs(raw, region_labels, min_area, top_k):
    raw, labels = np.asarray(raw), np.asarray(region_labels)
    if raw.ndim != 2 or min(raw.shape, default=0) == 0 or not np.isin(raw, [0, 1, 2, 3]).all():
        raise ValueError("Expected a nonempty 2D original wafer with codes 0/1/2/3")
    if labels.shape != raw.shape or labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Region labels must be an integer map with the original wafer shape")
    for value, name in ((min_area, "min_area"), (top_k, "top_k")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if top_k > 4:
        raise ValueError("This experiment supports at most the frozen Top-4 F regions")
    if np.any(labels < 0) or np.any(labels > top_k):
        raise ValueError("Region labels must be zero or tokens within the Top-4 budget")
    tokens = np.unique(labels[labels > 0])
    if not np.array_equal(tokens, np.arange(1, len(tokens) + 1)):
        raise ValueError("Positive region tokens must be consecutive starting at one")
    if np.any((labels > 0) & (raw != 2)):
        raise ValueError("Regions may only contain original defective pixels")
    support = np.argwhere((raw == 1) | (raw == 2)).astype(np.float64)
    if not len(support):
        raise ValueError("Wafer has no valid support")
    # Reject cropped masks, disconnected unions and two tokens splitting one CC.
    # Source token order/Top-4 selection remains the caller's signed provenance.
    components, _ = ndimage.label(raw == 2, np.ones((3, 3), dtype=np.uint8))
    component_areas = np.bincount(components.ravel())
    areas = np.empty(len(tokens), dtype=np.int64)
    for index, token in enumerate(tokens):
        mask = labels == token
        area = int(mask.sum())
        if area < min_area:
            raise ValueError("Region area is smaller than the frozen minimum")
        ids = np.unique(components[mask])
        if len(ids) != 1 or ids[0] == 0 or component_areas[ids[0]] != area:
            raise ValueError("Each token must be one complete original cc8 component")
        areas[index] = area
    return raw, labels, support, areas


def describe_regions(raw, region_labels, *, min_area=5, top_k=4, settings=None):
    """Return joint/shape distributions for tokens ``1..N`` in a signed F map.

    ``joint`` and ``shape`` use the original global J denoising rule; ``*_all``
    pool all original pixels of the same region without hard denoising. Both
    versions use identical global TV values and pre-denoising channel maxima.
    Zero-mass rows carry ``empty=True`` and are not a perfect-match empty atom.
    All numeric descriptors are float64; point counts are int64.

    The caller must validate the source map's frozen token ordering/selection.
    This function validates complete original cc8 masks, IDs and area limits.
    """
    settings = load_settings() if settings is None else validate_settings(settings)
    raw, labels, support, areas = _validate_inputs(raw, region_labels, min_area, top_k)
    n_regions = len(areas)
    points_int = np.argwhere(raw == 2)
    points = points_int.astype(np.float64)
    center = (support.min(axis=0) + support.max(axis=0)) / 2
    radius = float(np.linalg.norm(support - center, axis=1).max())
    if len(points):
        saliencies = tensor_saliencies(tensor_vote(points, settings["sigma_d"], settings["block_size"]))
        maxima = saliencies.max(axis=0)
        normalized = np.divide(saliencies, maxima, out=np.zeros_like(saliencies), where=maxima > 0)
        keep = retained_points(saliencies, settings["noise_threshold"])
        rho = np.linalg.norm(points - center, axis=1) / radius if radius > 0 else np.zeros(len(points))
        point_features = np.column_stack((rho, normalized))
        point_tokens = labels[tuple(points_int.T)]
    else:
        maxima = np.zeros(2, dtype=np.float64)
        keep = np.zeros(0, dtype=bool)
        rho = np.zeros(0, dtype=np.float64)
        point_features = np.empty((0, 3), dtype=np.float64)
        point_tokens = np.zeros(0, dtype=labels.dtype)

    joint = np.zeros((n_regions, 128), dtype=np.float64)
    joint_all = np.zeros_like(joint)
    radial = np.zeros((n_regions, 16), dtype=np.float64)
    retained_count = np.zeros(n_regions, dtype=np.int64)
    for index in range(n_regions):
        region = point_tokens == index + 1
        retained = region & keep
        retained_count[index] = int(retained.sum())
        joint[index] = soft_histogram(point_features[retained], tuple(settings["bins"])).reshape(-1)
        joint_all[index] = soft_histogram(point_features[region], tuple(settings["bins"])).reshape(-1)
        radial[index] = np.histogram(np.clip(rho[region], 0, 1), bins=np.linspace(0, 1, 17))[0] / areas[index]

    return {
        "joint": joint,
        "joint_all": joint_all,
        "shape": joint.reshape(n_regions, 8, 4, 4).sum(axis=1).reshape(n_regions, 16),
        "shape_all": joint_all.reshape(n_regions, 8, 4, 4).sum(axis=1).reshape(n_regions, 16),
        "original_count": areas,
        "retained_count": retained_count,
        "quality": np.divide(retained_count, areas, out=np.zeros(n_regions, dtype=np.float64), where=areas > 0),
        "empty": retained_count == 0,
        "empty_all": np.zeros(n_regions, dtype=bool),
        "radial_hist16": radial,
        "relative_area": areas.astype(np.float64) / len(support),
        "saliency_max": maxima,
        "global_original_count": len(points),
        "global_retained_count": int(keep.sum()),
        "selected_fraction": float(areas.sum() / len(points)) if len(points) else 0.,
    }
