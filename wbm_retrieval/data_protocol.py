"""Exact-image deduplication and deterministic, class-stratified splitting.

Source row IDs remain unchanged. Hashes only locate possible matches; raw pixel
equality confirms them before retaining the smallest original row ID.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib

import numpy as np


BASE_CLASS_NAMES = (
    "center", "donut", "edge-loc", "edge-ring", "loc", "near-full", "scratch", "random",
)
DEDUPLICATION_POLICY = "exact_raw_pixels_uint8_sha256_then_array_equal_min_original_id_v1"


@dataclass
class Deduplication:
    representative_ids: np.ndarray
    representative_of: np.ndarray
    duplicate_groups: list
    _normal_count: int
    _unique_normal_count: int

    def summary(self):
        total, unique = len(self.representative_of), len(self.representative_ids)
        return {
            "source_count": total,
            "unique_count": unique,
            "removed_count": total - unique,
            "duplicate_group_count": len(self.duplicate_groups),
            "duplicate_member_count": sum(map(len, self.duplicate_groups)),
            "source_normal_count": self._normal_count,
            "unique_normal_count": self._unique_normal_count,
            "source_defect_count": total - self._normal_count,
            "unique_defect_count": unique - self._unique_normal_count,
        }


def _validate_labels(labels, n=None):
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.shape[1] != 8 or (n is not None and len(labels) != n):
        raise ValueError("Expected eight binary base-defect labels with shape (N, 8)")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected eight binary base-defect labels")
    return labels


def deduplicate_maps(maps, labels):
    """Globally deduplicate complete raw maps, including Normal samples.

    Pixel codes 0, 1, 2 and 3 all participate in identity. Rotations, flips,
    defect-mask equivalence and approximate matches are not merged. Conflicting
    labels on identical raw images are rejected instead of silently selecting one.
    """
    if not isinstance(maps, np.ndarray) or maps.ndim != 3 or not all(maps.shape):
        raise ValueError("Expected nonempty raw wafer maps with shape (N, H, W)")
    labels = _validate_labels(labels, len(maps))
    for start in range(0, len(maps), 1024):
        if not np.isin(maps[start:start + 1024], [0, 1, 2, 3]).all():
            raise ValueError("Unsupported WBM pixel code; expected 0/1/2/3")

    buckets = defaultdict(list)
    representatives = []
    representative_of = np.empty(len(maps), dtype=np.int64)
    repeated = {}
    for sample_id, wafer in enumerate(maps):
        canonical = np.ascontiguousarray(wafer, dtype=np.uint8)
        digest = hashlib.sha256(canonical.tobytes()).digest()
        representative = None
        for candidate in buckets[digest]:
            if np.array_equal(maps[candidate], wafer):
                representative = candidate
                break
        if representative is None:
            representative = sample_id
            buckets[digest].append(sample_id)
            representatives.append(sample_id)
        else:
            if not np.array_equal(labels[representative], labels[sample_id]):
                raise ValueError(
                    f"Identical raw maps have conflicting labels: IDs {representative} and {sample_id}"
                )
            repeated.setdefault(representative, [representative]).append(sample_id)
        representative_of[sample_id] = representative

    representative_ids = np.asarray(representatives, dtype=np.int64)
    normal = labels.sum(axis=1) == 0
    return Deduplication(
        representative_ids, representative_of,
        [repeated[key] for key in sorted(repeated)],
        int(normal.sum()), int(normal[representative_ids].sum()),
    )


def stratified_split(labels, representative_ids, seed=2026, ratios=(0.7, 0.1, 0.2)):
    """Split unique defect representatives; empty-label Normal is excluded.

    Each combination must have at least three unique maps so all three splits
    contain that combination. Counts use floor for train/valid and the remainder
    for test, with a minimum of one per split for small classes.
    """
    labels = _validate_labels(labels)
    values = np.asarray(representative_ids)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("representative_ids must be a one-dimensional integer array")
    if len(values) != len(np.unique(values)) or np.any(values < 0) or np.any(values >= len(labels)):
        raise ValueError("representative_ids must be unique source row IDs in range")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    ratios = np.asarray(ratios, dtype=float)
    if ratios.shape != (3,) or not np.isfinite(ratios).all() or np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1.):
        raise ValueError("Split ratios must be three positive finite values summing to one")

    groups = defaultdict(list)
    for sample_id in sorted(values.tolist()):
        signature = tuple(np.flatnonzero(labels[sample_id]).tolist())
        if signature:
            groups[signature].append(sample_id)
    if not groups:
        raise ValueError("No nonempty-label representatives to split")
    rng = np.random.default_rng(seed)
    result = {name: [] for name in ("train", "valid", "test")}
    for signature, ids in sorted(groups.items()):
        n = len(ids)
        if n < 3:
            raise ValueError(f"Combination {signature} has only {n} unique maps; need at least three")
        shuffled = rng.permutation(ids)
        counts = [int(np.floor(n * ratios[0])), int(np.floor(n * ratios[1]))]
        counts.append(n - sum(counts))
        counts = [max(1, count) for count in counts]
        while sum(counts) > n:
            largest = max(range(3), key=lambda i: counts[i])
            counts[largest] -= 1
        nt, nv, _ = counts
        result["train"].extend(shuffled[:nt].tolist())
        result["valid"].extend(shuffled[nt:nt + nv].tolist())
        result["test"].extend(shuffled[nt + nv:].tolist())
    return {name: np.asarray(sorted(ids), dtype=np.int64) for name, ids in result.items()}
