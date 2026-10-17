"""Reusable filtered 8-CC, token patches, PCA geometry and image augmentation."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def resize_nearest(array, size):
    h, w = array.shape
    rr = np.minimum((np.arange(size) * h / size).astype(int), h - 1)
    cc = np.minimum((np.arange(size) * w / size).astype(int), w - 1)
    return array[np.ix_(rr, cc)]


def square_pad(array, padding=0):
    h, w = array.shape
    side = max(h, w)
    result = np.zeros((side + 2 * padding, side + 2 * padding), dtype=array.dtype)
    row, col = padding + (side - h) // 2, padding + (side - w) // 2
    result[row:row + h, col:col + w] = array
    return result


def token_patch(mask, size=96, padding=4):
    points = np.argwhere(mask)
    if not len(points):
        raise ValueError("Cannot build a patch for an empty token")
    low, high = points.min(0), points.max(0) + 1
    crop = mask[low[0]:high[0], low[1]:high[1]].astype(np.uint8)
    return resize_nearest(square_pad(crop, padding), size)


def whole_map_patch(raw, size=96):
    return resize_nearest(square_pad((raw == 2).astype(np.uint8)), size)


def extract_regions(raw, min_area=5, top_k=6, padding=4):
    del padding  # Padding belongs exclusively to encoder input construction.
    defect = raw == 2
    valid = (raw == 1) | defect
    valid_points = np.argwhere(valid)
    if not len(valid_points):
        return []
    wafer_center = valid_points.mean(axis=0)
    wafer_radius = max(float(np.linalg.norm(valid_points - wafer_center, axis=1).max()), 1.0)
    labels, count = ndimage.label(defect, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    retained = [i for i in range(1, count + 1) if areas[i] >= min_area]
    # Equal-area ties follow raster-scan CC IDs for deterministic proposals.
    retained.sort(key=lambda index: (-areas[index], index))
    result = []
    for component in retained[:top_k]:
        mask = labels == component
        points = np.argwhere(mask).astype(np.float64)
        centroid = points.mean(0)
        # Population covariance, in wafer-radius units; trace is rotation invariant.
        centered = (points - centroid) / wafer_radius
        covariance = centered.T @ centered / len(points)
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)
        result.append({"mask": mask, "area": int(areas[component]),
                       "position": (centroid - wafer_center) / wafer_radius,
                       "scale": float(np.sqrt(eigenvalues.sum())),
                       "pca_eigenvalues": eigenvalues[::-1]})
    return result


def augment(image, rng, rotation=True, max_shift=0, dropout=0.0, noise=0.0):
    output = np.array(image, dtype=np.float32, copy=True)
    if rotation:
        output = np.rot90(output, int(rng.integers(4)), axes=(-2, -1)).copy()
    if max_shift:
        h, w = output.shape[-2:]
        rows, cols = np.nonzero((output.reshape(-1, h, w) > 0).any(0))
        if len(rows):
            # Optional translations must retain every foreground pixel.
            row = int(rng.integers(max(-max_shift, -int(rows.min())), min(max_shift, h - 1 - int(rows.max())) + 1))
            col = int(rng.integers(max(-max_shift, -int(cols.min())), min(max_shift, w - 1 - int(cols.max())) + 1))
        else:
            row = col = 0
        shifted = np.zeros_like(output)
        r0, r1 = max(0, -row), min(h, h - row)
        c0, c1 = max(0, -col), min(w, w - col)
        shifted[..., r0 + row:r1 + row, c0 + col:c1 + col] = output[..., r0:r1, c0:c1]
        output = shifted
    output *= rng.random(output.shape) >= dropout
    output = np.maximum(output, (rng.random(output.shape) < noise).astype(np.float32))
    return np.ascontiguousarray(output)
