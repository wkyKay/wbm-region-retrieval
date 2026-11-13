"""Directional category-coverage metrics for the second retrieval task.

The query is the reference side: extra labels on a candidate do not reduce
coverage.  This module is deliberately separate from the exact-combination
metric implementation so the two relevance definitions remain auditable.
"""
from __future__ import annotations

import csv
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


METRIC_VERSION = "coverage_v1"


def _labels(values, name, *, width=None, allow_empty=False):
    array = np.asarray(values)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or (width is not None and array.shape[1] != width):
        raise ValueError(f"{name} must be a two-dimensional binary label matrix")
    if not np.isin(array, [0, 1]).all():
        raise ValueError(f"{name} must contain only binary labels")
    if not allow_empty and np.any(array.sum(axis=1) == 0):
        raise ValueError(f"{name} must contain nonempty defect labels")
    return array.astype(np.float64, copy=False)


def coverage_values(query_labels, candidate_labels):
    """Return directional query-to-candidate coverage in ``[0, 1]``."""
    query = _labels(query_labels, "query_labels")
    if len(query) != 1:
        raise ValueError("query_labels must describe exactly one query")
    candidates = _labels(candidate_labels, "candidate_labels", width=query.shape[1])
    denominator = float(query.sum())
    return (candidates @ query[0]) / denominator


def query_metrics(query_labels, candidate_labels, ranked_labels, ks):
    """Compute one query's graded and full-coverage metrics.

    ``candidate_labels`` is the complete gallery excluding the query.  The
    ranked list may be truncated; missing positions are zero coverage/misses.
    """
    query = _labels(query_labels, "query_labels")
    candidates = _labels(candidate_labels, "candidate_labels", width=query.shape[1])
    ranked = _labels(ranked_labels, "ranked_labels", width=query.shape[1], allow_empty=True)
    if len(ranked) == 0:
        ranked = np.empty((0, query.shape[1]), dtype=np.float64)
    cutoffs = sorted({int(k) for k in ks})
    if not cutoffs or any(k < 1 for k in cutoffs):
        raise ValueError("ks must contain positive integers")
    full_coverage = coverage_values(query, candidates)
    returned_coverage = coverage_values(query, ranked) if len(ranked) else np.empty(0, dtype=float)
    full_flags = np.isclose(full_coverage, 1.0, rtol=0.0, atol=1e-12)
    returned_flags = np.isclose(returned_coverage, 1.0, rtol=0.0, atol=1e-12)
    full_count = int(full_flags.sum())
    precision_by_rank = (np.cumsum(returned_flags).astype(float) /
                         np.arange(1, len(returned_flags) + 1))
    result = {"full_coverage_in_pool": full_count, "returned": len(returned_coverage)}
    for k in cutoffs:
        values = np.zeros(k, dtype=float)
        values[:min(k, len(returned_coverage))] = returned_coverage[:k]
        discounts = 1.0 / np.log2(np.arange(k, dtype=float) + 2.0)
        ideal = np.sort(full_coverage)[::-1][:k]
        idcg = float(np.sum(ideal * discounts[:len(ideal)]))
        result[f"Coverage-nDCG@{k}"] = float(np.sum(values * discounts) / idcg) if idcg else None
        result[f"MeanCoverage@{k}"] = float(values.mean())
        hits = float(returned_flags[:k].sum())
        result[f"FullCoverageHit@{k}"] = float(hits > 0)
        result[f"FullCoveragePrecision@{k}"] = hits / k
        result[f"FullCoverageRecall@{k}"] = hits / full_count if full_count else None
        result[f"FullCoverageAP@{k}"] = (
            float(np.sum(precision_by_rank[:k] * returned_flags[:k]) / full_count)
            if full_count else None
        )
    return result


def _metric_fields(ks):
    return [field for k in ks for field in (
        f"Coverage-nDCG@{k}", f"MeanCoverage@{k}", f"FullCoverageHit@{k}",
        f"FullCoveragePrecision@{k}", f"FullCoverageRecall@{k}", f"FullCoverageAP@{k}")]


def _mean(rows, fields):
    metrics, counts = {}, {}
    for field in fields:
        values = [row[field] for row in rows if row[field] is not None]
        key = f"FullCoverage_mAP@{field.split('@', 1)[1]}" if field.startswith("FullCoverageAP@") else field
        metrics[key] = float(np.mean(values)) if values else None
        counts[key] = len(values)
    return metrics, counts


def _summary(rows, fields):
    metrics, counts = _mean(rows, fields)
    return {
        "num_queries": len(rows),
        "num_classes": len({row["class_id"] for row in rows}),
        "queries_without_full_coverage_candidates": sum(row["full_coverage_in_pool"] == 0 for row in rows),
        "metrics": metrics,
        "metric_query_counts": counts,
    }


def _metric_output_fields(ks):
    fields = []
    for k in ks:
        fields.extend((f"Coverage-nDCG@{k}", f"MeanCoverage@{k}", f"FullCoverageHit@{k}",
                       f"FullCoveragePrecision@{k}", f"FullCoverageRecall@{k}",
                       f"FullCoverage_mAP@{k}"))
    return fields


def aggregate(rows, ks, *, split_name, test_evaluated, metadata=None):
    """Aggregate per-query coverage rows and produce class/group summaries."""
    cutoffs = sorted({int(k) for k in ks})
    fields = _metric_fields(cutoffs)
    rows = list(rows)
    if not rows:
        raise ValueError("coverage evaluation requires at least one query")
    by_class = defaultdict(list)
    for row in rows:
        by_class[row["class_id"]].append(row)
    class_rows, class_summaries = [], {}
    for class_id, class_queries in sorted(by_class.items()):
        summary = _summary(class_queries, fields)
        class_summaries[class_id] = summary
        class_rows.append({
            "class_id": class_id,
            "label_signature": class_queries[0]["label_signature"],
            "label_count": class_queries[0]["label_count"],
            "num_queries": len(class_queries),
            "queries_without_full_coverage_candidates": summary["queries_without_full_coverage_candidates"],
            **summary["metrics"],
        })
    overall = _summary(rows, fields)
    macro, macro_counts = {}, {}
    for key in overall["metrics"]:
        values = [summary["metrics"][key] for summary in class_summaries.values()
                  if summary["metrics"][key] is not None]
        macro[key] = float(np.mean(values)) if values else None
        macro_counts[key] = len(values)

    def group(selected):
        if not selected:
            output_fields = _metric_output_fields(cutoffs)
            return {"num_queries": 0, "num_classes": 0,
                    "queries_without_full_coverage_candidates": 0,
                    "metrics": {field: None for field in output_fields},
                    "metric_query_counts": {field: 0 for field in output_fields}}
        return _summary(selected, fields)

    result = {
        "metric_version": METRIC_VERSION,
        "split": split_name,
        "test_evaluated": test_evaluated,
        "ks": cutoffs,
        "relevance": "directional query-label coverage |Lq intersection Lc| / |Lq|",
        "gain": "linear coverage",
        "full_coverage_condition": "Lq subset Lc; coverage == 1",
        "idcg_scope": "complete fixed gallery excluding query",
        "zero_idcg_policy": "exclude from Coverage-nDCG means and report valid counts",
        "full_coverage_denominator": "complete fixed gallery count with coverage == 1",
        "zero_full_coverage_policy": "exclude FullCoverage Recall/AP means; include Hit/Precision",
        "precision_denominator": "K; short or empty returns are zero coverage/misses",
        "aggregation": "query means; class macro means give each represented combination equal weight",
        **overall,
        "class_macro_metrics": macro,
        "class_macro_metric_class_counts": macro_counts,
        "groups": {
            "single_label": group([row for row in rows if row["label_count"] == 1]),
            "mixed_label": group([row for row in rows if row["label_count"] > 1]),
            "by_label_count": {
                str(count): group([row for row in rows if row["label_count"] == count])
                for count in sorted({row["label_count"] for row in rows})
            },
        },
        "metadata": dict(metadata or {}),
    }
    return result, class_rows


def _label_signature(labels):
    values = _labels(labels, "labels")[0].astype(int)
    return "".join(str(value) for value in values)


def _ids(values, name):
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return array.astype(np.int64, copy=False)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_csv(path, rows, fieldnames):
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_rankings_file(bundle, pool, ranking_path, ks, output_dir, *, split_name="valid", metadata=None):
    """Write coverage reports from the bounded Top-K ranking CSV."""
    cutoffs = sorted({int(k) for k in ks})
    if not cutoffs or any(k < 1 for k in cutoffs):
        raise ValueError("ks must contain positive integers")
    directory = Path(output_dir)
    paths = [directory / name for name in
             ("coverage_metrics.json", "coverage_per_query.csv", "coverage_per_class.csv")]
    if any(path.exists() for path in paths):
        raise FileExistsError(f"Coverage metric outputs already exist in {directory}")

    ranking_file = Path(ranking_path).open(newline="", encoding="utf-8")
    reader = csv.DictReader(ranking_file)
    expected = {"query_id", "rank", "candidate_id"}
    if not expected <= set(reader.fieldnames or ()):
        ranking_file.close()
        raise ValueError("ranking file is missing query_id, rank, or candidate_id")

    def next_record():
        try:
            record = next(reader)
        except StopIteration:
            return None
        query_id = int(record["query_id"])
        rank = int(record["rank"])
        candidate_id = int(record["candidate_id"])
        if rank < 1:
            raise ValueError("ranking ranks must be positive")
        return query_id, rank, candidate_id

    pending = next_record()
    labels = np.asarray(bundle.labels)
    class_ids = np.asarray(bundle.class_ids)
    rows, seen = [], set()
    items = pool.items() if hasattr(pool, "items") else pool
    for query_value, candidates_value in items:
        query_id = int(query_value)
        if query_id in seen:
            raise ValueError("duplicate query ID in retrieval pool")
        seen.add(query_id)
        candidates = _ids(candidates_value, "candidate IDs")
        if query_id in candidates or len(np.unique(candidates)) != len(candidates):
            raise ValueError("candidate pool must exclude the query and contain unique IDs")
        if pending is not None and pending[0] < query_id:
            ranking_file.close()
            raise ValueError(f"ranking file contains query {pending[0]} outside the retrieval pool")
        records = []
        while pending is not None and pending[0] == query_id:
            records.append((pending[1], pending[2]))
            pending = next_record()
        if [rank for rank, _ in records] != list(range(1, len(records) + 1)):
            ranking_file.close()
            raise ValueError(f"ranking ranks are not contiguous for query {query_id}")
        ranked = np.asarray([candidate for _, candidate in records], dtype=np.int64)
        if len(ranked) and (len(np.unique(ranked)) != len(ranked) or not np.isin(ranked, candidates).all()):
            ranking_file.close()
            raise ValueError(f"ranking contains an invalid candidate for query {query_id}")
        query_labels = labels[query_id]
        candidate_labels = labels[candidates]
        ranked_labels = labels[ranked] if len(ranked) else np.empty((0, labels.shape[1]), dtype=labels.dtype)
        label_signature = _label_signature(query_labels)
        row = {"query_id": query_id, "class_id": int(class_ids[query_id]),
               "label_signature": label_signature, "label_count": int(np.sum(query_labels)),
               "pool_size": len(candidates)}
        row.update(query_metrics(query_labels, candidate_labels, ranked_labels, cutoffs))
        rows.append(row)
    if pending is not None:
        ranking_file.close()
        raise ValueError(f"ranking file contains query {pending[0]} outside the retrieval pool")
    ranking_file.close()
    if not rows:
        raise ValueError("coverage evaluation requires at least one query")
    rows.sort(key=lambda row: row["query_id"])
    result, class_rows = aggregate(rows, cutoffs, split_name=split_name,
                                   test_evaluated=split_name == "test", metadata=metadata)
    _write_csv(paths[1], rows, list(rows[0]))
    _write_csv(paths[2], class_rows, list(class_rows[0]))
    paths[0].write_text(json.dumps(result, indent=2, sort_keys=True,
                                   allow_nan=False, default=_json_default) + "\n", encoding="utf-8")
    return result
