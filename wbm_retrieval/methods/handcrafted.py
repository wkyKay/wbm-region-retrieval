"""Zernike-magnitude region baseline; no learned or concatenated descriptors.

Reference: A. Khotanzad and Y. H. Hong, "Invariant Image Recognition by
Zernike Moments", IEEE TPAMI 12(5), 489-497, 1990, doi:10.1109/34.55109.
Order, discrete normalization and distance-to-score mapping are documented
adaptations for this project's CC-token retrieval protocol.
"""

from __future__ import annotations

from functools import lru_cache
from math import factorial

import numpy as np


@lru_cache(maxsize=16)
def moment_orders(max_order):
    """Nonnegative m; omit constant Z00 and centroid-zero Z11."""
    if not isinstance(max_order, int) or not 2 <= max_order <= 20:
        raise ValueError("Zernike max_order must be an integer from 2 to 20")
    return tuple((n, m) for n in range(2, max_order + 1) for m in range(n + 1) if (n - m) % 2 == 0)


def radial_polynomial(n, m, radius):
    value = np.zeros_like(radius, dtype=np.float64)
    for s in range((n - m) // 2 + 1):
        coefficient = ((-1) ** s * factorial(n - s) /
                       (factorial(s) * factorial((n + m) // 2 - s) * factorial((n - m) // 2 - s)))
        value += coefficient * radius ** (n - 2 * s)
    return value


def handcrafted_descriptor(mask, max_order=12):
    """Return |Z_nm / Z_00| for the complete centroid-centered binary region.

    Scale by the farthest foreground radius, keeping all pixels in the unit
    disk. Holes are not filled. Uniform foreground mass gives the discrete
    ratio (n+1)*mean(R_nm*exp(-i*m*theta)). Do not add auxiliary statistics
    or normalize the resulting vector to unit length.
    """
    orders = moment_orders(max_order)
    binary = np.asarray(mask)
    if binary.ndim != 2:
        raise ValueError("A Zernike region mask must be two-dimensional")
    points = np.argwhere(binary > 0).astype(np.float64)
    if not len(points):
        raise ValueError("Empty token")
    points -= points.mean(0)
    radius = np.linalg.norm(points, axis=1)
    radius /= max(float(radius.max()), 1e-12)
    angle = np.arctan2(points[:, 0], points[:, 1])
    magnitudes = [abs((n + 1) * np.mean(radial_polynomial(n, m, radius) * np.exp(-1j * m * angle)))
                  for n, m in orders]
    return np.asarray(magnitudes, dtype=np.float32)


def retrieve(config, protocol):
    """Rank Zernike Euclidean scores with the shared outer region assignment."""
    from ..data import load_regions
    from ..matching import group_tokens
    from ..retrieval import rank_local

    metadata = load_regions(config, "test")
    groups = group_tokens(metadata, metadata["descriptor"])
    rank_local("handcrafted", groups, config, protocol, representation="zernike")
