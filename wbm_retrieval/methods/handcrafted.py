"""Method one: handcrafted token descriptors and shared greedy local retrieval."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def handcrafted_descriptor(mask):
    """Hu + radial mass + occupancy statistics, without absolute position or size."""
    points = np.argwhere(mask).astype(np.float64)
    area = len(points)
    if not area:
        raise ValueError("Empty token")
    centered = points - points.mean(0)
    x, y = centered[:, 1], centered[:, 0]
    def moment(p, q):
        return float(np.sum(x ** p * y ** q) / area ** (1 + (p + q) / 2))
    n20, n02, n11 = moment(2, 0), moment(0, 2), moment(1, 1)
    n30, n12, n21, n03 = moment(3, 0), moment(1, 2), moment(2, 1), moment(0, 3)
    a, b, c, d = n30 - 3 * n12, 3 * n21 - n03, n30 + n12, n21 + n03
    hu = np.array([n20 + n02, (n20 - n02) ** 2 + 4 * n11 ** 2,
                   a * a + b * b, c * c + d * d,
                   a * c * (c * c - 3 * d * d) + b * d * (3 * c * c - d * d),
                   (n20 - n02) * (c * c - d * d) + 4 * n11 * c * d,
                   b * c * (c * c - 3 * d * d) - a * d * (3 * c * c - d * d)])
    transformed = np.where(np.abs(hu) > 1e-12, -np.sign(hu) * np.log10(np.maximum(np.abs(hu), 1e-12)) / 12, 0)
    radial = np.linalg.norm(centered, axis=1)
    radial /= max(float(radial.max()), 1e-8)
    histogram = np.histogram(radial, bins=12, range=(0, 1))[0].astype(float) / area
    covariance = centered.T @ centered / area
    ev = np.maximum(np.linalg.eigvalsh(covariance), 0)
    h, w = (points.max(0) - points.min(0) + 1)
    filled = int(ndimage.binary_fill_holes(mask).sum())
    # Four-neighbour exposed pixel edges give a scale-normalized perimeter statistic.
    padded = np.pad(mask.astype(np.int32), 1)
    perimeter = np.abs(np.diff(padded, axis=0)).sum() + np.abs(np.diff(padded, axis=1)).sum()
    stats = np.array([area / (h * w), ev[0] / max(ev[1], 1e-8),
                      min(4 * np.pi * area / max(perimeter ** 2, 1), 1),
                      (filled - area) / max(filled, 1), radial.mean(), radial.std()])
    groups = [transformed, histogram, stats]
    groups = [value / max(np.linalg.norm(value), 1e-8) for value in groups]
    descriptor = np.concatenate(groups).astype(np.float32)
    return descriptor / max(np.linalg.norm(descriptor), 1e-8)


def retrieve(config, protocol):
    """Rank cached handcrafted CC descriptors without fitting any model."""
    from ..data import load_regions
    from ..matching import group_tokens
    from ..retrieval import rank_local

    metadata = load_regions(config, "test")
    groups = group_tokens(metadata, metadata["descriptor"])
    rank_local("handcrafted", groups, config, protocol)

