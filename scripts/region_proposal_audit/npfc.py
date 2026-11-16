"""NPFC-pre: the reproducible preliminary stages of Hou et al. (2024).

This is NOT a full NPFC reproduction. It implements path-connected initial
clusters, removes components of one or two pixels, and combines clusters with
minimum Euclidean separation <= sqrt(8). No SD-based cluster-count selection,
radial core assignment, or Den-based filtering is claimed. See NPFC_NOTES.md.

The output masks contain only original defective pixels; clustering gaps are
never filled. No class labels or F-specific area/top-k postprocessing are used.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


SOURCE = "https://doi.org/10.1007/s10836-024-06132-8"


def extract_npfc(defect: np.ndarray) -> dict:
    """Return the NPFC preliminary filtering/near-cluster merging masks.

    Parameters
    ----------
    defect : two-dimensional array
        Boolean defective-pixel mask (the caller supplies ``raw == 2``).

    Returns
    -------
    dict
        ``masks`` is a list of nonempty, mutually exclusive Boolean masks on
        the input grid. ``noise_mask`` holds the initial <=2-pixel components.
        ``diagnostics`` describes the limited stage and all interpretation
        choices. The function name is retained for the audit runner's API;
        display names MUST say ``NPFC-pre``, not full ``NPFC``.

    The paper does not specify the order of proximity merging. Here it is the
    deterministic transitive closure of the graph of surviving initial
    components linked by minimum original-pixel separation <= sqrt(8).
    """
    mask = np.asarray(defect, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("defect must be a two-dimensional mask")
    labels, initial_count = ndimage.label(mask, np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel(), minlength=initial_count + 1)
    retained_ids = np.flatnonzero(sizes >= 3)
    retained_ids = retained_ids[retained_ids != 0]
    kept = np.isin(labels, retained_ids) & mask
    noise = mask & ~kept

    diagnostics = {
        "method_id": "npfc_pre",
        "display_name": "NPFC-pre",
        "stage_only": True,
        "source_doi": SOURCE,
        "source_sections": "2.1(2-7), 2.2.1, first step of 2.2.3",
        "initial_connectivity": 8,
        "minimum_initial_area": 3,
        "merge_distance": float(np.sqrt(8)),
        "merge_squared_distance": 8,
        "merge_policy": "transitive_closure_of_surviving_component_graph",
        "tie_order": "descending_area_then_first_row_major_pixel",
        "full_npfc_sd_selection": False,
        "full_npfc_radial_den_assignment": False,
        "initial_components": int(initial_count),
        "surviving_initial_components": int(len(retained_ids)),
        "original_pixels": int(mask.sum()),
        "retained_pixels": int(kept.sum()),
        "removed_pixels": int(noise.sum()),
        "warnings": [
            "Preliminary stages only; not a full NPFC reproduction.",
            "Proximity merge transitive closure is an explicit implementation convention.",
            "Original paper evaluated WM-811K; this audit applies the stages to MixedWM38.",
        ],
    }

    if not len(retained_ids):
        diagnostics["output_regions"] = 0
        return {"masks": [], "noise_mask": noise, "diagnostics": diagnostics}

    parent = {int(i): int(i) for i in retained_ids}

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    points = np.argwhere(kept)
    point_components = labels[tuple(points.T)]
    # A small tolerance makes all exact integer squared-distance-8 neighbors
    # discoverable. The subsequent integer check rejects any pair outside 8.
    pairs = cKDTree(points).query_pairs(np.sqrt(8) + 1e-12, output_type="ndarray")
    for left, right in pairs:
        if int(np.square(points[left] - points[right]).sum()) > 8:
            continue
        a, b = find(int(point_components[left])), find(int(point_components[right]))
        if a != b:
            # Canonical smallest initial component ID makes output independent
            # of cKDTree's pair iteration order.
            parent[max(a, b)] = min(a, b)

    grouped: dict[int, list[int]] = {}
    for component in retained_ids:
        grouped.setdefault(find(int(component)), []).append(int(component))
    result = [np.isin(labels, component_ids) for component_ids in grouped.values()]
    result.sort(key=lambda item: (-int(item.sum()), int(np.flatnonzero(item)[0])))
    diagnostics["output_regions"] = len(result)
    return {"masks": result, "noise_mask": noise, "diagnostics": diagnostics}
