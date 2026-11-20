"""Auditable additions to frozen cc8 Top-4, never modifications to old masks."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from experiments.proposed.f_segmentation.proposals import segment

PROPOSALS = ("base4", "top8", "residual8")
PROPOSAL_SPECS = {
    "base4": {"base": "frozen_cc8_min5_top4", "additional_regions": 0},
    "top8": {"base": "frozen_cc8_min5_top4", "additional_regions": 4,
             "addition": "remaining_complete_cc8", "minimum_area": 5},
    "residual8": {"base": "frozen_cc8_min5_top4", "additional_regions": 4,
                  "addition": "all_remaining_original_defect_pixels",
                  "merge_squared_distance": 8, "merge_policy": "transitive_closure",
                  "minimum_area_after_grouping": 5, "fill_nondefect_pixels": False},
}


def _validate(raw, old_labels):
    raw, old = np.asarray(raw), np.asarray(old_labels)
    if (raw.ndim != 2 or not all(raw.shape) or not np.isin(raw, [0, 1, 2, 3]).all()
            or not np.any((raw == 1) | (raw == 2))):
        raise ValueError("Expected a nonempty original wafer with valid support")
    if old.shape != raw.shape or old.dtype.kind not in "iu" or np.any(old > 4):
        raise ValueError("Frozen labels must be integer original-grid Top-4 masks")
    expected, _ = segment(raw, "cc8", min_area=5, top_k=4)
    if not np.array_equal(old, expected):
        raise ValueError("Frozen labels differ from original cc8 min-area-5 Top-4")
    return raw, old.astype(np.uint8, copy=False)


def _ordered(masks):
    return sorted(masks, key=lambda m: (-int(m.sum()), int(np.flatnonzero(m)[0])))


def residual_groups(residual):
    """Original pixels connected by squared distance <= 8, including tiny CCs.

    Group the point-adjacency graph by transitive closure. A two-pixel gap may
    connect components, but neither intervening normal pixels nor base regions
    become graph members. This is a fixed grouping rule, not a new segmenter.
    """
    mask = np.asarray(residual)
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("Residual must be a two-dimensional Boolean mask")
    labels, count = ndimage.label(mask, np.ones((3, 3), dtype=np.uint8))
    parent = np.arange(count + 1)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    height, width = mask.shape
    # One half of the integer-radius-two neighbourhood; integer distances are
    # exact, independent of floating point radius or pair iteration order.
    for dy in range(3):
        for dx in range(-2, 3):
            if (dy == 0 and dx <= 0) or dy * dy + dx * dx > 8:
                continue
            y0, y1 = max(0, -dy), min(height, height - dy)
            x0, x1 = max(0, -dx), min(width, width - dx)
            if y0 >= y1 or x0 >= x1:
                continue
            a = labels[y0:y1, x0:x1].ravel()
            b = labels[y0+dy:y1+dy, x0+dx:x1+dx].ravel()
            ok = (a > 0) & (b > 0) & (a != b)
            for left, right in np.unique(np.column_stack((a[ok], b[ok])), axis=0):
                left, right = find(int(left)), find(int(right))
                if left != right:
                    parent[max(left, right)] = min(left, right)
    roots = np.array([find(i) for i in range(count + 1)])
    grouped = roots[labels]
    return _ordered([grouped == root for root in np.unique(roots[1:])])


def propose_all(raw, old_labels):
    """Return ``name: (labels, audit)`` with full original-pixel provenance."""
    raw, old = _validate(raw, old_labels)
    defect = raw == 2
    components, count = ndimage.label(defect, np.ones((3, 3), dtype=np.uint8))
    original_sizes = np.bincount(components.ravel(), minlength=count + 1)
    base_count = int(old.max())
    residual = defect & (old == 0)
    cc_masks = _ordered([components == i for i in range(1, count + 1)
                         if original_sizes[i] >= 5 and not np.any(old[components == i])])
    grouped = residual_groups(residual)
    additions = {"base4": [], "top8": cc_masks[:4],
                 "residual8": [m for m in grouped if int(m.sum()) >= 5][:4]}
    result = {}
    for name in PROPOSALS:
        labels = old.copy()
        for token, mask in enumerate(additions[name], base_count + 1):
            if np.any(mask & ~residual) or np.any(labels[mask]):
                raise AssertionError("A new region overlaps preserved regions or nondefect pixels")
            labels[mask] = token
        trace = []
        for token in range(1, int(labels.max()) + 1):
            indices = np.flatnonzero(labels == token)
            original_ids = np.unique(components.ravel()[indices]).tolist()
            trace.append({"token": token, "origin": "frozen_top4" if token <= base_count else name,
                          "area": len(indices), "first_flat_index": int(indices[0]),
                          "original_cc_ids": original_ids,
                          "original_cc_areas": [int(original_sizes[i]) for i in original_ids],
                          "flat_indices": indices.tolist()})
        retained = int(np.count_nonzero(labels))
        total = int(defect.sum())
        result[name] = (labels, {
            "proposal": name, "base_regions": base_count,
            "additional_regions": len(additions[name]), "regions": int(labels.max()),
            "original_pixels": total, "base_retained_pixels": int(np.count_nonzero(old)),
            "retained_pixels": retained, "added_pixels": retained - int(np.count_nonzero(old)),
            "remaining_pixels": total - retained,
            "retained_fraction": retained / total if total else 0.,
            "residual_graph_components": len(grouped), "region_trace": trace,
        })
    return result


def propose(raw, old_labels, name):
    if name not in PROPOSALS:
        raise ValueError("Unknown frozen region proposal")
    return propose_all(raw, old_labels)[name]
