"""Versioned exact-combination retrieval metrics, independent of training code.

AP@K is sum(Precision@r * relevant[r], r=1..K) / R, where R counts
all relevant candidates in the fixed gallery excluding the query. Missing
returns contribute zero. This is truncated AP, not unrestricted full AP.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np


DEFAULT_KS = (1, 5, 10, 20, 50, 100, 500)
METRIC_VERSION = "exact_v1"
TIE_POLICY = (
    "ascending uint64 from the first 8 bytes (big-endian) of SHA-256 of "
    "ASCII exact_v1:tie:{seed}:{query_id}:{candidate_id}; final key sample_id"
)


def _integer(value, name, minimum=0):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or int(value) < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _ids(values, name, *, nonempty=False):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be one-dimensional integer source IDs")
    try:
        raw = list(values)
    except TypeError as error:
        raise ValueError(f"{name} must be one-dimensional integer source IDs") from error
    result = [_integer(value, name) for value in raw]
    if any(value > np.iinfo(np.int64).max for value in result):
        raise ValueError(f"{name} contains an out-of-range source ID")
    if (nonempty and not result) or len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique nonnegative source IDs")
    return np.asarray(result, dtype=np.int64)


def tie_keys(query_id, candidate_ids, seed=2026):
    """Return label-independent, input-order-independent SHA-256 uint64 keys.

    The serialized bytes are ASCII ``exact_v1:tie:{seed}:{query_id}:{id}``
    with decimal integers and no whitespace or newline. Use the first eight
    digest bytes as an unsigned big-endian integer. Labels are never read.
    """
    query_id = _integer(query_id, "query_id")
    seed = _integer(seed, "seed")
    ids = _ids(candidate_ids, "candidate_ids")
    prefix = f"{METRIC_VERSION}:tie:{seed}:{query_id}:"
    return np.fromiter(
        (int.from_bytes(hashlib.sha256(f"{prefix}{int(i)}".encode("ascii")).digest()[:8], "big")
         for i in ids), dtype=np.uint64, count=len(ids),
    )


def _scores(values, name, length, *, allow_negative_infinity):
    try:
        raw = np.asarray(values)
        if raw.ndim != 1 or len(raw) != length or raw.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be a one-dimensional real array aligned to candidate_ids")
        result = raw.astype(np.float64)
    except (TypeError, OverflowError) as error:
        raise ValueError(f"{name} must contain real scores") from error
    invalid = np.isnan(result) | np.isposinf(result)
    if not allow_negative_infinity:
        invalid |= np.isneginf(result)
    if np.any(invalid):
        raise ValueError(f"{name} contains invalid nonfinite values")
    return result


def order_scores(candidate_ids, scores, query_id, seed=2026, secondary=None):
    """Return sorting indices: descending score, optional ascending secondary.

    Hash keys, then original sample IDs, break remaining ties. ``-inf`` is a
    valid worst primary score (e.g. a distance of +inf). Secondary distances
    must be finite; NaN and +inf primary scores are rejected.
    """
    ids = _ids(candidate_ids, "candidate_ids")
    values = _scores(scores, "scores", len(ids), allow_negative_infinity=True)
    hashes = tie_keys(query_id, ids, seed)
    if secondary is None:
        return np.lexsort((ids, hashes, -values))
    secondary_values = _scores(secondary, "secondary", len(ids), allow_negative_infinity=False)
    return np.lexsort((ids, hashes, secondary_values, -values))


def _mean_metrics(rows, ks):
    metrics, counts = {}, {}
    for k in ks:
        for prefix in ("Hit", "Precision", "Recall", "AP"):
            field = f"{prefix}@{k}"
            key = f"mAP@{k}" if prefix == "AP" else field
            values = [row[field] for row in rows if row[field] is not None]
            metrics[key] = float(np.mean(values)) if values else None
            counts[key] = len(values)
    return metrics, counts


def _summary(rows, ks):
    metrics, counts = _mean_metrics(rows, ks)
    return {
        "num_queries": len(rows),
        "num_classes": len({row["class_id"] for row in rows}),
        "queries_without_relevant_candidates": sum(row["relevant_in_pool"] == 0 for row in rows),
        "metrics": metrics,
        "metric_query_counts": counts,
    }


def _label_info(bundle, allowed):
    """Check the class mapping against full nonempty multi-hot combinations."""
    classes = np.asarray(bundle.class_ids)
    labels = np.asarray(bundle.labels)
    if classes.ndim != 1 or classes.dtype.kind not in "iu":
        raise ValueError("class_ids must be a one-dimensional integer array")
    if labels.ndim != 2 or len(labels) != len(classes) or labels.shape[1] == 0:
        raise ValueError("labels must be a multi-hot matrix aligned to class_ids")
    if np.any(allowed >= len(classes)):
        raise ValueError("Held-out source ID is outside the label arrays")
    selected = labels[allowed]
    if not np.all((selected == 0) | (selected == 1)) or np.any(selected.sum(axis=1) == 0):
        raise ValueError("Exact evaluation requires nonempty binary defect labels")
    if np.any(classes[allowed] < 0):
        raise ValueError("Exact evaluation requires valid nonempty combination class IDs")
    mapping, inverse = {}, {}
    for i in allowed:
        class_id = int(classes[i])
        label = tuple(int(v) for v in labels[i])
        if class_id in mapping and mapping[class_id] != label:
            raise ValueError("A class ID maps to multiple complete label combinations")
        if label in inverse and inverse[label] != class_id:
            raise ValueError("A complete label combination maps to multiple class IDs")
        mapping[class_id], inverse[label] = label, class_id
    return classes, mapping


def _write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def evaluate_rankings(bundle, pool, rankings, ks, output_dir, split_name="valid", metadata=None):
    """Evaluate exactly one (query_id, ranked_ids) record per frozen query.

    The iterator is consumed once, retaining per-query statistics rather than
    all rankings. Query means and equal-weight class means are both reported.
    R=0 queries remain in Hit/Precision, and are excluded from Recall/AP.
    An existing metric report is never overwritten.
    """
    if split_name not in ("valid", "test"):
        raise ValueError("Retrieval split must be valid or test")
    if hasattr(pool, "split_name") and pool.split_name != split_name:
        raise ValueError("Retrieval pool split does not match requested evaluation")
    if isinstance(ks, (str, bytes)):
        raise ValueError("ks must contain unique positive integer cutoffs")
    try:
        cutoffs = [_integer(k, "K", 1) for k in ks]
    except TypeError as error:
        raise ValueError("ks must contain unique positive integer cutoffs") from error
    if not cutoffs or len(set(cutoffs)) != len(cutoffs):
        raise ValueError("ks must contain unique positive integer cutoffs")
    cutoffs.sort()
    if metadata is not None and not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    directory = Path(output_dir)
    paths = [directory / name for name in ("metrics.json", "per_query.csv", "per_class.csv")]
    if any(path.exists() for path in paths):
        raise FileExistsError(f"Exact metric outputs already exist in {directory}")

    allowed_ids = bundle.valid_ids if split_name == "valid" else bundle.reserved_ids
    allowed_array = _ids(allowed_ids, "held-out IDs", nonempty=True)
    allowed = set(allowed_array.tolist())
    classes, label_mapping = _label_info(bundle, allowed_array)
    query_ids = set(_ids(pool, "pool query IDs", nonempty=True).tolist())
    if not query_ids <= allowed:
        raise ValueError("Frozen queries must belong to the requested held-out split")

    rows, seen = [], set()
    items = rankings.items() if isinstance(rankings, Mapping) else iter(rankings)
    for item in items:
        try:
            q, ranking = item
        except (TypeError, ValueError) as error:
            raise ValueError("Rankings must yield (query_id, ranked_ids) records") from error
        q = _integer(q, "query_id")
        if q not in query_ids or q in seen:
            raise ValueError("Rankings must contain exactly one result per frozen query")
        seen.add(q)
        candidates = _ids(pool[q], "candidate IDs", nonempty=True)
        candidate_set = set(candidates.tolist())
        if not candidate_set <= allowed or q in candidate_set:
            raise ValueError("Candidate pool must use held-out IDs and exclude self")
        if max(cutoffs) > len(candidates):
            raise ValueError("K cannot exceed the full candidate pool size")
        ranked = _ids(ranking, "ranked IDs")
        if not set(ranked.tolist()) <= candidate_set or q in ranked:
            raise ValueError("Ranking contains unknown, wrong-split or self candidates")
        relevant_total = int(np.sum(classes[candidates] == classes[q]))
        relevant = (classes[ranked] == classes[q]).astype(np.float64)
        cumulative = np.cumsum(relevant)
        precision_at_rank = cumulative / np.arange(1, len(ranked) + 1)
        label = label_mapping[int(classes[q])]
        row = {
            "query_id": q, "class_id": int(classes[q]),
            "label_signature": "|".join(str(i) for i, value in enumerate(label) if value),
            "label_count": sum(label), "pool_size": len(candidates),
            "relevant_in_pool": relevant_total, "returned": len(ranked),
        }
        for k in cutoffs:
            hits = float(relevant[:k].sum())
            row[f"Hit@{k}"] = float(hits > 0)
            row[f"Precision@{k}"] = hits / k
            row[f"Recall@{k}"] = hits / relevant_total if relevant_total else None
            row[f"AP@{k}"] = (
                float(np.sum(precision_at_rank[:k] * relevant[:k]) / relevant_total)
                if relevant_total else None
            )
        rows.append(row)
    if seen != query_ids:
        raise ValueError("Rankings must contain exactly the frozen query IDs")
    rows.sort(key=lambda row: row["query_id"])

    by_class = defaultdict(list)
    for row in rows:
        by_class[row["class_id"]].append(row)
    class_rows = []
    for class_id, class_queries in sorted(by_class.items()):
        summary = _summary(class_queries, cutoffs)
        class_rows.append({
            "class_id": class_id,
            "label_signature": class_queries[0]["label_signature"],
            "label_count": class_queries[0]["label_count"],
            "num_queries": len(class_queries),
            "queries_without_relevant_candidates": summary["queries_without_relevant_candidates"],
            **summary["metrics"],
        })
    aggregate = _summary(rows, cutoffs)
    macro, macro_counts = {}, {}
    for key in aggregate["metrics"]:
        values = [row[key] for row in class_rows if row[key] is not None]
        macro[key] = float(np.mean(values)) if values else None
        macro_counts[key] = len(values)
    result = {
        "metric_version": METRIC_VERSION,
        "split": split_name, "test_evaluated": split_name == "test",
        "ks": cutoffs,
        "relevance": "exact equality of the entire nonempty defect-label combination",
        "ap_definition": "AP@K = sum_{r=1..K}(Precision@r * relevant_at_r) / R; truncated AP",
        "ap_denominator": "all relevant candidates in the fixed pool, including screened/routed-out candidates",
        "precision_denominator": "K; short or empty returns are misses",
        "zero_relevant_policy": "exclude from mAP/Recall; include in Hit/Precision",
        "aggregation": "query means; class macro means give each represented query class equal weight",
        **aggregate,
        "class_macro_metrics": macro,
        "class_macro_metric_class_counts": macro_counts,
        "groups": {
            "single_label": _summary([row for row in rows if row["label_count"] == 1], cutoffs),
            "mixed_label": _summary([row for row in rows if row["label_count"] > 1], cutoffs),
            "by_label_count": {
                str(count): _summary([row for row in rows if row["label_count"] == count], cutoffs)
                for count in sorted({row["label_count"] for row in rows})
            },
        },
        "retrieval_protocol_signature": getattr(pool, "protocol_signature", None),
        "metadata": dict(metadata or {}),
    }
    # Serialize before touching disk, so malformed metadata cannot leave a report.
    document = json.dumps(result, indent=2, sort_keys=True, allow_nan=False, default=_json_default) + "\n"
    directory.mkdir(parents=True, exist_ok=True)
    _write_csv(paths[1], rows, list(rows[0]))
    _write_csv(paths[2], class_rows, list(class_rows[0]))
    paths[0].write_text(document, encoding="utf-8")
    return result
