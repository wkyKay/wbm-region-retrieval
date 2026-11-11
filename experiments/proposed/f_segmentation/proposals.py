"""Six label-free proposal pipelines, matching the 2026-11-11 visual audit."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .npfc_pre import extract_npfc
from .tensor_voting import retained_points, tensor_saliencies, tensor_vote

METHODS = ("cc8", "cc4", "median8", "opening8", "tv8", "npfc_pre")
SETTINGS = {
    "cc8": {"connectivity": 8},
    "cc4": {"connectivity": 4},
    "median8": {"size": 3, "boundary": "constant_zero", "connectivity": 8,
                "output": "intersect_original_defects_after_labeling"},
    "opening8": {"kernel": "cross3", "iterations": 1, "connectivity": 8},
    "tv8": {"sigma_d": 9.0, "noise_threshold": 0.32, "block_size": 256,
            "connectivity": 8, "all_other_points_vote": True},
    "npfc_pre": {"initial_connectivity": 8, "minimum_initial_area": 3,
                 "merge_squared_distance": 8, "merge_policy": "transitive_closure",
                 "full_npfc": False},
}


def segment(raw, method, min_area=5, top_k=4):
    """Return uint8 region IDs (0=unassigned) and pixel-retention diagnostics.

    Area and ties are measured on original defective pixels, even for median
    support that fills holes. Empty results stay empty; no CC8 fallback.
    """
    raw = np.asarray(raw)
    if raw.ndim != 2 or not np.isin(raw, [0, 1, 2, 3]).all():
        raise ValueError("Expected a 2D raw wafer with codes 0/1/2/3")
    if method not in METHODS or min_area < 1 or not 1 <= top_k <= 255:
        raise ValueError("Invalid segmentation method or proposal budget")
    defect = raw == 2
    support = defect
    added = 0
    if method == "npfc_pre":
        masks = extract_npfc(defect)["masks"]
    else:
        if method == "median8":
            support = ndimage.median_filter(defect.astype(np.uint8), size=3,
                                            mode="constant", cval=0).astype(bool)
            support &= (raw == 1) | defect
            added = int((support & ~defect).sum())
        elif method == "opening8":
            support = ndimage.binary_opening(defect, ndimage.generate_binary_structure(2, 1),
                                             iterations=1)
        elif method == "tv8":
            points = np.argwhere(defect)
            settings = SETTINGS[method]
            saliency = tensor_saliencies(tensor_vote(points, settings["sigma_d"], settings["block_size"]))
            points = points[retained_points(saliency, settings["noise_threshold"])]
            support = np.zeros(raw.shape, dtype=bool)
            support[tuple(points.T)] = True
        labels, count = ndimage.label(support, ndimage.generate_binary_structure(2, 1 if method == "cc4" else 2))
        masks = [(labels == i) & defect for i in range(1, count + 1)]
    masks = [m for m in masks if m.any()]
    masks.sort(key=lambda m: (-int(m.sum()), int(np.flatnonzero(m)[0])))
    eligible = [m for m in masks if int(m.sum()) >= min_area]
    output = np.zeros(raw.shape, dtype=np.uint8)
    for i, mask in enumerate(eligible[:top_k], 1):
        output[mask] = i
    original = int(defect.sum())
    kept = int(np.count_nonzero(output))
    return output, {"original_pixels": original, "native_regions": len(masks),
                    "native_pixels": sum(int(m.sum()) for m in masks),
                    "eligible_regions": len(eligible), "regions": int(output.max()),
                    "retained_pixels": kept, "retained_fraction": kept / original if original else 0.,
                    "empty": not bool(kept), "median_added_support_pixels": added}


def segment_job(job):
    raw, method, proposal = job
    return segment(raw, method, proposal["min_area"], proposal["top_k"])
