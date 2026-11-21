"""Read-only, NumPy diagnostics for class geometry of frozen representations.

Wafer labels are diagnostic proxies for region labels, not pixel-level truth.
An audit never clips cosine, changes features, or fits a model. For multiple
tokens from a wafer, every token of that wafer is excluded from its gallery.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


_GEOMETRY_METRICS = (
    "intra_mean_cosine", "inter_mean_cosine", "cosine_margin",
    "balanced_intra_mean_cosine", "balanced_inter_mean_cosine",
    "balanced_cosine_margin", "nearest_same_rank",
)


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def _metrics(ks):
    return [*_GEOMETRY_METRICS,
            *(name for k in ks for name in (f"hit_at_{k}", f"precision_at_{k}"))]


def _direction(name):
    if name == "nearest_same_rank":
        return "lower"
    if name.endswith("mean_cosine"):
        return "descriptive_only"
    return "higher"


def _inputs(features, labels, map_ids, ks, chunk_size):
    x, labels, ids = np.asarray(features), np.asarray(labels), np.asarray(map_ids)
    if (x.ndim != 2 or not all(x.shape) or x.dtype.kind not in "fiu"
            or not np.isfinite(x).all()):
        raise ValueError("Features must be a nonempty finite real matrix")
    n = len(x)
    if (labels.shape != (n,) or labels.dtype.kind not in "iuUS"
            or ids.shape != (n,) or ids.dtype.kind not in "iu"
            or np.any(ids < 0)):
        raise ValueError("Supply one integer/string class and nonnegative integer map ID per row")
    if labels.dtype.kind in "iu" and np.any(labels < 0):
        raise ValueError("Class IDs must be nonnegative")
    empty_label = b"" if labels.dtype.kind == "S" else ""
    if labels.dtype.kind in "US" and np.any(labels == empty_label):
        raise ValueError("Class labels cannot be empty")
    if labels.dtype.kind == "S":
        labels = labels.astype(str)
    ks = tuple(ks)
    if (not ks or any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
                      or k < 1 for k in ks) or len(set(ks)) != len(ks)):
        raise ValueError("ks must contain unique positive integers")
    if (isinstance(chunk_size, (bool, np.bool_))
            or not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1):
        raise ValueError("chunk_size must be a positive integer")
    x = x.astype(np.float64, copy=False)
    norms = np.linalg.norm(x, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("Cosine requires finite nonzero feature norms")
    owner_labels = {}
    for label, sid in zip(labels.tolist(), ids.tolist()):
        if sid in owner_labels and owner_labels[sid] != label:
            raise ValueError("Tokens belonging to one map must have the same diagnostic label")
        owner_labels[sid] = label
    return x / norms[:, None], labels, ids, tuple(int(k) for k in ks), owner_labels


def _aggregate(rows, names):
    """Query, map, and class averages without treating pair edges as queries."""
    by_map = defaultdict(list)
    for row in rows:
        by_map[row["map_id"]].append(row)
    map_rows = []
    for sid, items in sorted(by_map.items()):
        map_rows.append({"map_id": sid, "label": items[0]["label"],
                         **{name: _mean(row[name] for row in items) for name in names}})
    by_class = defaultdict(list)
    for row in map_rows:
        by_class[row["label"]].append(row)
    per_class = []
    for label, items in sorted(by_class.items()):
        class_queries = [row for row in rows if row["label"] == label]
        per_class.append({
            "label": label, "maps": len(items), "queries": len(class_queries),
            "metrics": {name: _mean(row[name] for row in items) for name in names},
            "valid_maps": {name: sum(row[name] is not None for row in items) for name in names},
            "valid_queries": {name: sum(row[name] is not None for row in class_queries) for name in names},
        })
    return {
        "query_macro": {
            "metrics": {name: _mean(row[name] for row in rows) for name in names},
            "valid_queries": {name: sum(row[name] is not None for row in rows) for name in names},
        },
        "map_macro": {
            "metrics": {name: _mean(row[name] for row in map_rows) for name in names},
            "valid_maps": {name: sum(row[name] is not None for row in map_rows) for name in names},
        },
        "class_balanced": {
            "metrics": {name: _mean(item["metrics"][name] for item in per_class) for name in names},
            "valid_classes": {
                name: sum(item["metrics"][name] is not None for item in per_class) for name in names},
        },
        "per_class": per_class,
    }


def audit_geometry(features, labels, map_ids, *, ks=(1, 5, 10), chunk_size=256):
    """Return JSON-safe ``summary`` and one diagnostic row per feature vector.

    ``labels`` is a one-dimensional array of single-class IDs, not multi-hot
    labels. Run independently for h and z. Cosine uses float64 L2-normalized
    features without clipping. Ranking ties use ascending source map ID, then
    original feature row; KNN is token-level unless there is one token per map.

    Raw intra/inter means average candidate tokens. Balanced means first average
    tokens within each candidate map; inter then weights other classes equally.
    Summary ``class_balanced`` first averages query tokens within each query map,
    then maps within each class, and finally classes equally. No sample-pair
    edge is treated as an independent query. Undefined metrics are ``None``.
    """
    x, labels, ids, ks, owner_labels = _inputs(features, labels, map_ids, ks, chunk_size)
    n = len(x)
    unique_maps, map_counts = np.unique(ids, return_counts=True)
    map_labels = np.asarray([owner_labels[int(sid)] for sid in unique_maps])
    classes = np.unique(labels)
    map_order = np.argsort(ids, kind="stable")
    starts = np.r_[0, np.cumsum(map_counts)[:-1]]
    all_indices = np.arange(n)
    token_indices = np.empty(n, dtype=np.int64)
    for start, count in zip(starts, map_counts):
        token_indices[map_order[start:start + count]] = np.arange(count)
    rows = []
    for begin in range(0, n, chunk_size):
        similarities = x[begin:begin + chunk_size] @ x.T
        candidate_map_means = np.add.reduceat(similarities[:, map_order], starts, axis=1) / map_counts
        for offset, scores in enumerate(similarities):
            i = begin + offset
            eligible = ids != ids[i]
            same = eligible & (labels == labels[i])
            other = eligible & (labels != labels[i])
            same_maps = (unique_maps != ids[i]) & (map_labels == labels[i])
            other_maps = map_labels != labels[i]
            intra = float(scores[same].mean()) if same.any() else None
            inter = float(scores[other].mean()) if other.any() else None
            map_means = candidate_map_means[offset]
            balanced_intra = float(map_means[same_maps].mean()) if same_maps.any() else None
            balanced_inter = _mean(
                map_means[map_labels == label].mean() for label in classes if label != labels[i])
            candidates = all_indices[eligible]
            order = np.lexsort((candidates, ids[candidates], -scores[candidates]))
            relevant = labels[candidates[order]] == labels[i]
            positives = np.flatnonzero(relevant)
            row = {
                "source_index": int(i), "map_id": int(ids[i]), "token_index": int(token_indices[i]),
                "label": labels[i].item(), "candidate_count": int(eligible.sum()),
                "same_class_candidates": int(same.sum()), "other_class_candidates": int(other.sum()),
                "same_class_candidate_maps": int(same_maps.sum()),
                "other_class_candidate_maps": int(other_maps.sum()),
                "intra_mean_cosine": intra, "inter_mean_cosine": inter,
                "cosine_margin": intra - inter if intra is not None and inter is not None else None,
                "balanced_intra_mean_cosine": balanced_intra,
                "balanced_inter_mean_cosine": balanced_inter,
                "balanced_cosine_margin": (balanced_intra - balanced_inter
                                           if balanced_intra is not None and balanced_inter is not None else None),
                "nearest_same_rank": int(positives[0] + 1) if len(positives) else None,
            }
            for k in ks:
                effective = min(k, len(candidates))
                row[f"effective_k_{k}"] = effective
                # Absence of any same-class candidate is structurally undefined.
                defined = bool(effective and len(positives))
                row[f"hit_at_{k}"] = float(relevant[:effective].any()) if defined else None
                row[f"precision_at_{k}"] = float(relevant[:effective].mean()) if defined else None
            rows.append(row)
    names = _metrics(ks)
    summary = {
        "schema_version": 1, "queries": n, "maps": len(unique_maps), "classes": len(classes),
        "feature_dimension": x.shape[1], "ks": list(ks), "metric_names": names,
        "metric_directions": {name: _direction(name) for name in names},
        "intra_inter_direction_note": "Absolute cosine shifts alone do not establish semantic improvement; inspect margins and KNN.",
        "label_scope": "single-class wafer labels are diagnostic proxies, not region/pixel semantic truth",
        "gallery_rule": "all feature rows except every token of the query map",
        "knn_unit": "map" if len(unique_maps) == n else "token",
        "cosine": "float64 L2 normalization, no clipping",
        "ties": "descending cosine, ascending map_id, ascending source_index",
        "small_gallery_k": "effective_k=min(k, eligible candidate tokens); retained in each query row",
        "missing_positive_policy": "same-class geometry, rank, Hit and Precision are null when no same-class candidate exists",
        "aggregation": "query tokens averaged within map; maps within class; classes equally; undefined values omitted with counts",
        **_aggregate(rows, names),
    }
    return {"summary": summary, "rows": rows}


def _paired_rows(reference, candidate):
    if (not isinstance(reference, dict) or not isinstance(candidate, dict)
            or "summary" not in reference or "summary" not in candidate
            or "rows" not in reference or "rows" not in candidate):
        raise ValueError("Both inputs must be geometry audits")
    names = reference["summary"].get("metric_names")
    if (not names or names != candidate["summary"].get("metric_names")
            or reference["summary"].get("ks") != candidate["summary"].get("ks")):
        raise ValueError("Paired audits require identical metric definitions and K values")
    def indexed(audit):
        result = {}
        for row in audit["rows"]:
            key = (row["map_id"], row["token_index"])
            if key in result:
                raise ValueError("Duplicate query identity in audit")
            result[key] = row
        return result
    left, right = indexed(reference), indexed(candidate)
    if not left or left.keys() != right.keys():
        raise ValueError("Paired audits require the same query maps and token identities")
    pairs = []
    for key in sorted(left):
        a, b = left[key], right[key]
        for field in ("label", "candidate_count", "same_class_candidates", "other_class_candidates"):
            if a[field] != b[field]:
                raise ValueError("Labels or candidate identities changed between audits")
        for name in names:
            for row in (a, b):
                value = row[name]
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                          or not math.isfinite(value)):
                    raise ValueError("Paired audit metrics must be finite numbers or null")
            if (a[name] is None) != (b[name] is None):
                raise ValueError("Paired audits disagree on metric eligibility")
        pairs.append((a, b))
    return names, pairs


def paired_bootstrap(reference, candidate, *, repetitions=2000, seed=2026, confidence=.95):
    """Class-balanced candidate-minus-reference effects with map-cluster CIs.

    Token metrics are paired before averaging within each map. Maps are sampled
    with replacement within their diagnostic class, preserving all their tokens
    as a cluster. The gallery and learned representations remain fixed; these
    intervals do not measure training-seed or gallery uncertainty. For ranks,
    a negative difference is an improvement. No pair-edge bootstrap is used.
    """
    if (isinstance(repetitions, (bool, np.bool_)) or not isinstance(repetitions, (int, np.integer))
            or repetitions < 1):
        raise ValueError("repetitions must be a positive integer")
    if (isinstance(confidence, (bool, np.bool_)) or not isinstance(confidence, (float, int))
            or not 0 < confidence < 1):
        raise ValueError("confidence must be between zero and one")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    names, pairs = _paired_rows(reference, candidate)
    maps = defaultdict(list)
    for a, b in pairs:
        maps[a["map_id"]].append((a, b))
    classes = defaultdict(list)
    for sid, items in sorted(maps.items()):
        values = []
        left_values, right_values = [], []
        for name in names:
            left = _mean(a[name] for a, _ in items)
            right = _mean(b[name] for _, b in items)
            left_values.append(left)
            right_values.append(right)
            values.append(right - left if left is not None else None)
        classes[items[0][0]["label"]].append((sid, values, left_values, right_values))
    arrays = {}
    for label, items in sorted(classes.items()):
        arrays[label] = tuple(np.asarray([[np.nan if v is None else v for v in item[j]]
                                         for item in items], dtype=np.float64) for j in (1, 2, 3))
    # Eligibility for the implemented metrics is constant within a class: it
    # depends on labels and gallery membership, never on model scores. Preserve
    # that contract so bootstrap samples cannot silently drop a class.
    for differences, _, _ in arrays.values():
        valid = np.isfinite(differences)
        if np.any(valid.any(axis=0) != valid.all(axis=0)):
            raise ValueError("Metric eligibility must be constant within each diagnostic class")
    rng = np.random.default_rng(seed)
    draws = np.zeros((repetitions, len(names)), dtype=np.float64)
    valid_classes = np.zeros(len(names), dtype=np.int64)
    effect = np.zeros(len(names))
    reference_mean = np.zeros(len(names))
    candidate_mean = np.zeros(len(names))
    valid_maps = np.zeros(len(names), dtype=np.int64)
    for differences, left, right in arrays.values():
        valid = np.isfinite(differences).all(axis=0)
        if not valid.any():
            continue
        valid_classes += valid
        valid_maps += valid * len(differences)
        effect[valid] += differences[:, valid].mean(axis=0)
        reference_mean[valid] += left[:, valid].mean(axis=0)
        candidate_mean[valid] += right[:, valid].mean(axis=0)
        # A bounded block avoids allocating repetitions * maps * metrics.
        for begin in range(0, repetitions, 128):
            size = min(128, repetitions - begin)
            indices = rng.integers(len(differences), size=(size, len(differences)))
            draws[begin:begin + size, valid] += differences[:, valid][indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    effects = {}
    for column, name in enumerate(names):
        count = int(valid_classes[column])
        if count:
            samples = draws[:, column] / count
            lower, upper = np.quantile(samples, [alpha, 1 - alpha])
            effects[name] = {
                "reference": float(reference_mean[column] / count),
                "candidate": float(candidate_mean[column] / count),
                "difference": float(effect[column] / count),
                "ci_low": float(lower), "ci_high": float(upper),
                "valid_classes": count, "valid_maps": int(valid_maps[column]),
                "valid_queries": sum(a[name] is not None for a, _ in pairs),
                "improvement_direction": _direction(name),
            }
        else:
            effects[name] = {key: None for key in ("reference", "candidate", "difference", "ci_low", "ci_high")}
            effects[name].update(valid_classes=0, valid_maps=0, valid_queries=0,
                                 improvement_direction=_direction(name))
    return {
        "schema_version": 1, "difference": "candidate minus reference",
        "estimand": "equal-class mean of equal-map means of paired query-token metrics",
        "resampling": "map clusters with replacement, stratified by diagnostic class",
        "confidence": float(confidence), "repetitions": int(repetitions), "seed": int(seed),
        "scope": "fixed representations and gallery; no training-seed, gallery, or multiple-comparison uncertainty",
        "effects": effects,
    }
