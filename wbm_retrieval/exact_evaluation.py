"""Adapters for the byte-identical compared_methods exact_v1 metric engine.

Source IDs always index the original label matrix. Canonical reports retain the
same schema as compared_methods; auxiliary files support the experiment suites.
"""
from collections.abc import Mapping
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .common import write_csv, write_json
from .exact_metrics import DEFAULT_KS, TIE_POLICY, evaluate_rankings


def is_exact(config_or_settings):
    settings = config_or_settings.get("evaluation", config_or_settings)
    version = settings.get("metric_version", "containment")
    if version not in ("exact_v1", "containment", "legacy"):
        raise ValueError(f"Unknown metric_version: {version}")
    return version == "exact_v1"


def per_query_metric_name(key):
    return key[1:] if key.startswith("mAP@") else key


def group_rows(result):
    """Return suite summary rows without changing the canonical report schema."""
    if isinstance(result["groups"], list):
        return result["groups"]
    fields = ("num_queries", "num_classes", "queries_without_relevant_candidates",
              "metrics", "metric_query_counts", "query_bootstrap_95ci")
    rows = [{"group": name, **{key: value[key] for key in fields if key in value}}
            for name, value in (("all", result),
                                ("single", result["groups"]["single_label"]),
                                ("mixed", result["groups"]["mixed_label"]))]
    rows.append({"group": "class_macro", "num_queries": result["num_queries"],
                 "num_classes": result["num_classes"], "metrics": result["class_macro_metrics"]})
    return rows


def evaluate_suite_rankings(config, labels, candidates, rankings, output, split,
                            split_ids=None, metadata=None):
    """Stream ranked source IDs and keep full-pool denominators.

    Existing reports are immutable. Callers must select a fresh evaluation
    directory when intentionally evaluating another checkpoint or protocol.
    """
    if not is_exact(config):
        raise ValueError("This adapter requires metric_version=exact_v1")
    settings = config["evaluation"]
    output = Path(output)
    if any((output / name).exists() for name in
           ("metrics.json", "per_query.csv", "per_class.csv", "per_query.json", "per_query_metrics.csv")):
        raise FileExistsError(f"Exact metric outputs already exist in {output}")
    if isinstance(labels, Mapping):
        if not labels:
            raise ValueError("Labels must not be empty")
        width = len(next(iter(labels.values())))
        matrix = np.zeros((max(labels) + 1, width), dtype=np.uint8)
        for sample_id, label in labels.items():
            value = np.asarray(label)
            if value.shape != (width,) or not np.isin(value, [0, 1]).all():
                raise ValueError("Labels must be binary vectors of equal width")
            matrix[sample_id] = value
    else:
        matrix = np.asarray(labels)
    if matrix.ndim != 2 or not np.isin(matrix, [0, 1]).all():
        raise ValueError("Labels must be an original-ID-indexed binary matrix")
    nonempty = np.flatnonzero(matrix.sum(axis=1))
    taxonomy = sorted({tuple(int(x) for x in matrix[i]) for i in nonempty})
    mapping = {label: index for index, label in enumerate(taxonomy)}
    classes = np.full(len(matrix), -1, dtype=np.int64)
    for sample_id in nonempty:
        classes[sample_id] = mapping[tuple(int(x) for x in matrix[sample_id])]
    if split_ids is None:
        split_ids = getattr(candidates, "gallery", None)
    if split_ids is None:
        split_ids = sorted(set(candidates).union(*(set(v) for v in candidates.values())))
    bundle = SimpleNamespace(labels=matrix, class_ids=classes,
                             valid_ids=split_ids, reserved_ids=split_ids)
    extra = {"ranking_scope": "top_max_k", "ties": TIE_POLICY,
             "tie_seed": settings.get("tie_seed", 2026), **dict(metadata or {})}
    result = evaluate_rankings(bundle, candidates, rankings, settings.get("ks", DEFAULT_KS),
                               output, split_name=split, metadata=extra)
    with (output / "per_query.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    integer_fields = {"query_id", "class_id", "label_count", "pool_size", "relevant_in_pool", "returned"}
    for row in rows:
        for key, value in row.items():
            if key in integer_fields:
                row[key] = int(value)
            elif key != "label_signature":
                row[key] = float(value) if value else None
        row["group"] = "single" if row["label_count"] == 1 else "mixed"
    write_json(output / "per_query.json", rows)
    write_csv(output / "per_query_metrics.csv", rows, list(rows[0]))
    # These identifiers are used by the existing paired-suite reports.
    for key in ("representation", "pool_sha256", "matching", "aggregation"):
        if key in extra and key not in result:
            result[key] = extra[key]
    repetitions = int(extra.get("bootstrap_samples", 0))
    if repetitions:
        from .evaluation import bootstrap_interval
        for target, selected in ((result, rows),
                                 (result["groups"]["single_label"], [r for r in rows if r["group"] == "single"]),
                                 (result["groups"]["mixed_label"], [r for r in rows if r["group"] == "mixed"])):
            rng = np.random.default_rng(config.get("seed", 2026))
            target["query_bootstrap_95ci"] = {
                key: bootstrap_interval([r[per_query_metric_name(key)] for r in selected], repetitions, rng)
                for key in result["metrics"]}
    write_json(output / "metrics.json", result)
    return result


def fresh_training_evaluation_dir(output, config):
    """Preserve interrupted exact reports and recompute into a fresh sibling.

    The suite lock serializes training, so choosing a new attempt cannot race
    another training process. Final evaluation does not use this helper.
    """
    output = Path(output)
    if not is_exact(config) or not output.exists() or not any(output.iterdir()):
        return output
    attempt = 2
    while True:
        candidate = output.with_name(f"{output.name}_attempt_{attempt:03d}")
        if not candidate.exists():
            return candidate
        attempt += 1
