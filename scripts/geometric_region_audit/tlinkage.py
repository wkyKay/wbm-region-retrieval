"""T-Linkage line fitting: Python port for a label-free wafer pixel audit.

Reference: Magri & Fusiello, CVPR 2014, equations 2, 4, 5 / Algorithm 1.
Author MATLAB: https://fusiello.github.io/demo/jlk/Tlinkage.zip
This is a port and pixel-coordinate adaptation, not the author's executable.
See TLINKAGE_NOTES.md for exact provenance and the outlier-rejection caveat.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import asdict, dataclass

import numpy as np
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class TLinkageConfig:
    epsilon_pixels: float = 2.5  # preference cutoff, equal to 5*tau
    hypotheses_per_point: int = 5
    outlier_rejection: str = "author_cardinality"


def _line_hypotheses(points, rng, count):
    """Cycle first point as in author demo; uniformly choose distinct second."""
    n = len(points)
    first = np.arange(count, dtype=np.int64) % n
    second = rng.integers(0, n - 1, size=count)
    second += second >= first
    pairs = np.column_stack((first, second))
    # Match the author's duplicate MSS removal: keep the last sampled pair.
    keys = np.sort(pairs, axis=1)
    _, reverse_indices = np.unique(keys[::-1], axis=0, return_index=True)
    keep = np.sort(len(keys) - 1 - reverse_indices)
    pairs = pairs[keep]
    p, q = points[pairs[:, 0]], points[pairs[:, 1]]
    normal = np.column_stack((p[:, 1] - q[:, 1], q[:, 0] - p[:, 0]))
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    offset = -np.einsum("ij,ij->i", normal, p)
    return np.column_stack((normal, offset)), pairs


def preference_matrix(points, hypotheses, epsilon):
    residual = np.abs(points @ hypotheses[:, :2].T + hypotheses[:, 2])
    preferences = np.exp(-residual / (epsilon / 5.0))
    preferences[residual >= epsilon] = 0.0
    return preferences


def linkage(preferences):
    """Exact greedy Tanimoto linkage, with row minima cached in a lazy heap.

    No approximate nearest neighbors or truncated linkage. Each merge takes
    the global minimum distance, updates the PF by componentwise minimum,
    and stops at distance >= 1. Tie-breaking is deterministic by point index.
    """
    p = np.array(preferences, dtype=np.float64, copy=True)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=np.int32), {"merges": 0}
    norm2 = np.einsum("ij,ij->i", p, p)
    sparse = csr_matrix(p)
    inner = (sparse @ sparse.T).toarray()
    denominator = norm2[:, None] + norm2[None, :] - inner
    distances = np.ones((n, n), dtype=np.float64)
    np.divide(inner, denominator, out=distances, where=denominator > 0)
    distances = 1.0 - distances
    # An empty preference is an unsupported point, not a zero-distance group.
    distances[denominator <= 0] = 1.0
    np.clip(distances, 0.0, 1.0, out=distances)
    np.fill_diagonal(distances, np.inf)
    del sparse, inner, denominator

    active = np.ones(n, dtype=bool)
    versions = np.zeros(n, dtype=np.int64)
    nearest = np.argmin(distances, axis=1)
    best = distances[np.arange(n), nearest].copy()
    members = [[i] for i in range(n)]
    heap = []

    def push(i):
        j = int(nearest[i])
        heapq.heappush(heap, (float(best[i]), min(i, j), max(i, j), i, int(versions[i])))

    for i in range(n):
        push(i)
    merges = 0
    while heap:
        distance, x, y, source, version = heapq.heappop(heap)
        if not active[source] or version != versions[source]:
            continue
        if not active[x] or not active[y]:
            continue
        if distance >= 1.0:
            break
        p[x] = np.minimum(p[x], p[y])
        norm2[x] = np.dot(p[x], p[x])
        active[y] = False
        versions[y] += 1
        members[x].extend(members[y])
        members[y] = []
        distances[y, :] = np.inf
        distances[:, y] = np.inf
        ids = np.flatnonzero(active)
        cols = np.flatnonzero(p[x])
        dot = p[np.ix_(ids, cols)] @ p[x, cols]
        denom = norm2[ids] + norm2[x] - dot
        new = np.ones(len(ids), dtype=np.float64)
        np.divide(dot, denom, out=new, where=denom > 0)
        new = 1.0 - new
        new[denom <= 0] = 1.0
        np.clip(new, 0.0, 1.0, out=new)
        distances[x, ids] = new
        distances[ids, x] = new
        distances[x, x] = np.inf

        # Recompute only rows whose previous minimum has changed. Other rows
        # need a cheap check against their one changed distance, to x.
        must = ids[(nearest[ids] == x) | (nearest[ids] == y) | (ids == x)]
        changed = set(map(int, must))
        for i in must:
            nearest[i] = np.argmin(distances[i])
            best[i] = distances[i, nearest[i]]
        for i in ids:
            i = int(i)
            if i in changed:
                continue
            candidate = distances[i, x]
            if candidate < best[i] or (candidate == best[i] and x < nearest[i]):
                nearest[i] = x
                best[i] = candidate
                changed.add(i)
        for i in changed:
            versions[i] += 1
            push(i)
        merges += 1

    labels = np.zeros(n, dtype=np.int32)
    for label, i in enumerate(np.flatnonzero(active), 1):
        labels[members[i]] = label
    return labels, {"merges": merges, "clusters_before_rejection": int(active.sum())}


def author_cardinality_rejection(labels, minimal_sample_size=2):
    """Port of supplied outlier_rejection_card.m, including >=2-model bias."""
    labels = np.asarray(labels, dtype=np.int32)
    ids, counts = np.unique(labels, return_counts=True)
    if not len(ids):
        return labels.copy(), {"outlier_rule": "author_cardinality", "rejected_cluster_ids": []}
    cardinality = np.append(counts, minimal_sample_size)
    cluster_ids = np.append(ids, 0)
    order = np.argsort(cardinality, kind="stable")
    cardinality, cluster_ids = cardinality[order], cluster_ids[order]
    delta = np.diff(cardinality)
    cut = int(np.argmax(delta)) + 1  # MATLAB's 1-based cut
    result = labels.copy()
    rejected = []
    assumed_two = False
    if cut > 1:
        if cut == len(delta):
            assumed_two = True
            delta[cut - 1] = 0
            cut = int(np.argmax(delta)) + 1
        rejected = cluster_ids[:cut].astype(int).tolist()
        result[np.isin(result, rejected)] = 0
    # Stable spatial IDs (area normalization, if needed, belongs to renderer).
    surviving = np.unique(result[result > 0])
    out = np.zeros_like(result)
    for k, old in enumerate(surviving, 1):
        out[result == old] = k
    return out, {"outlier_rule": "author_cardinality", "rejected_cluster_ids": rejected,
                 "author_at_least_two_models_branch": assumed_two,
                 "cardinalities_ascending_including_dummy": cardinality.astype(int).tolist()}


def fit_line(points):
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=True)
    normal = vh[-1]
    if normal[0] < 0 or (normal[0] == 0 and normal[1] < 0):
        normal = -normal
    return np.r_[normal, -np.dot(normal, center)]


def tlinkage_points(points, seed=2026, config=None):
    config = config or TLinkageConfig()
    points = np.asarray(points, dtype=np.float64)
    start = time.perf_counter()
    diagnostics = {"method": "T-Linkage (line), Python port", "config": asdict(config),
                   "seed": int(seed), "num_points": len(points), "model_family": "line",
                   "uses_class_labels": False, "coordinate_order": "row,column"}
    if len(points) < 2:
        diagnostics.update({"seconds": time.perf_counter() - start, "models": [], "num_hypotheses": 0})
        return np.zeros(len(points), dtype=np.int32), diagnostics
    if config.epsilon_pixels <= 0:
        raise ValueError("epsilon_pixels must be positive")
    rng = np.random.default_rng(seed)
    hypotheses, _ = _line_hypotheses(points, rng, config.hypotheses_per_point * len(points))
    preferences = preference_matrix(points, hypotheses, config.epsilon_pixels)
    labels, info = linkage(preferences)
    diagnostics.update(info)
    diagnostics["num_hypotheses"] = len(hypotheses)
    diagnostics["preference_nonzero_fraction"] = float(np.count_nonzero(preferences) / preferences.size)
    if config.outlier_rejection == "author_cardinality":
        labels, info = author_cardinality_rejection(labels)
        diagnostics.update(info)
    elif config.outlier_rejection != "none":
        raise ValueError(config.outlier_rejection)
    models = []
    for k in np.unique(labels[labels > 0]):
        selected = points[labels == k]
        model = fit_line(selected)
        residual = np.abs(selected @ model[:2] + model[2])
        models.append({"region_id": int(k), "type": "line", "parameters": model.tolist(),
                       "area": len(selected), "rms_residual_pixels": float(np.sqrt(np.mean(residual ** 2)))})
    diagnostics.update({"models": models, "num_regions": len(models), "unassigned_points": int((labels == 0).sum()),
                        "seconds": time.perf_counter() - start})
    return labels, diagnostics


def tlinkage_regions(raw, seed=2026, config=None):
    """Return original-grid int32 labels (0=unassigned), and JSON diagnostics."""
    raw = np.asarray(raw)
    points = np.argwhere(raw == 2)
    memberships, diagnostics = tlinkage_points(points, seed=seed, config=config)
    result = np.zeros(raw.shape, dtype=np.int32)
    if len(points):
        result[tuple(points.T)] = memberships
    return result, diagnostics
