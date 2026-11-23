"""Read-only, posthoc train diagnostics of a previously frozen exclusion graph.

Labels are wafer-level proxies for single-label wafers, not region ground truth.
They are used only to describe the existing graph; nothing here updates the graph,
teacher, student, optimizer, checkpoint selection, or training sampler.
"""
from pathlib import Path
import hashlib

import numpy as np
import torch

from cnn_refinement.neighbors import _normalize, load_graph
from cnn_refinement.training import FilteredNegativeSampler
from cnn_refinement.state import identity, record, seal, signed, verified
from wbm_retrieval.common import digest, file_hash, write_csv
from wbm_retrieval.training import loader


def _array_identity(array):
    array = np.asarray(array)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def _mean(value):
    value = np.asarray(value)
    return float(value.mean(dtype=np.float64)) if value.size else None


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _distribution(values):
    values = np.asarray(values)
    return {"count": int(values.size), "mean": _mean(values),
            "p05": float(np.percentile(values, 5)) if values.size else None,
            "median": float(np.median(values)) if values.size else None,
            "p95": float(np.percentile(values, 95)) if values.size else None}


def _proxy_counts(graph, singleton):
    """Exact counts over the full graph and all eligible single-label train tokens."""
    ids = np.asarray(graph["token_map_ids"])
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    same_wafer_count = counts[inverse]
    class_counts = np.bincount(singleton[singleton >= 0], minlength=8)
    evaluable = np.zeros(len(ids), dtype=np.int64)
    hits = np.zeros(len(ids), dtype=np.int64)
    relevant = np.zeros(len(ids), dtype=np.int64)
    for row in np.flatnonzero(singleton >= 0):
        neighbors = graph["indices"][graph["indptr"][row]:graph["indptr"][row + 1]]
        neighbor_labels = singleton[neighbors]
        evaluable[row] = np.count_nonzero(neighbor_labels >= 0)
        hits[row] = np.count_nonzero(neighbor_labels == singleton[row])
        relevant[row] = class_counts[singleton[row]] - same_wafer_count[row]
    return evaluable, hits, relevant


def _expected_exposure(graph, batches, count=5000):
    """Exact expectation under iid uniform draws outside the fixed current batch."""
    n = len(graph["token_map_ids"])
    batch_number = np.empty(n, dtype=np.int64)
    batch_sizes = np.empty(n, dtype=np.int64)
    for number, members in enumerate(batches):
        batch_number[members] = number
        batch_sizes[members] = len(members)
    blocked = np.zeros(n, dtype=np.int64)
    for row in range(n):
        neighbors = graph["indices"][graph["indptr"][row]:graph["indptr"][row + 1]]
        blocked[row] = np.count_nonzero(batch_number[neighbors] != batch_number[row])
    available = n - batch_sizes
    allowed = available - blocked
    if np.any(allowed <= 0):
        raise ValueError("Posthoc diagnostic has no allowed negative for at least one anchor")
    probability = blocked / available
    # Expected distinct values in count iid draws from an m-member pool.
    def duplicate_rate(pool):
        with np.errstate(divide="ignore"):
            distinct = -pool * np.expm1(count * np.log1p(-1 / pool))
        return 1 - distinct / count
    return {"batch_number": batch_number, "batch_size": batch_sizes, "blocked_outside_batch": blocked,
            "available_before": available, "available_after": allowed,
            "replacement_probability": probability, "expected_replaced_slots": count * probability,
            "probability_zero_replacements": np.power(1 - probability, count),
            "expected_duplicate_rate_before": duplicate_rate(available),
            "expected_duplicate_rate_after": duplicate_rate(allowed)}


def _sample_audit(graph, features, batches, count=5000):
    """Four fixed epoch-0 batches, separate diagnostic draws, no training RNG state."""
    selected = np.unique(np.linspace(0, len(batches) - 1, min(4, len(batches)), dtype=np.int64))
    base_generator = torch.Generator().manual_seed(2026 + 3381)
    sampler = FilteredNegativeSampler(graph, len(features), "cpu", seed=2026 + 7001)
    rows, before_values, after_values, removed_values, replacement_values = [], [], [], [], []
    for number in selected:
        anchors = batches[int(number)]
        probabilities = torch.ones(len(features))
        probabilities[anchors] = 0
        base = torch.multinomial(probabilities, count, replacement=True, generator=base_generator)
        changes = sampler.replacements(torch.as_tensor(anchors), base, torch.as_tensor(anchors))
        actual = base.repeat(len(anchors), 1)
        actual[changes["rows"], changes["columns"]] = changes["ids"]
        before = features[anchors] @ features[base.numpy()].T
        after = before.copy()
        changed_rows = changes["rows"].numpy()
        changed_cols = changes["columns"].numpy()
        changed_ids = changes["ids"].numpy()
        replacements = np.sum(features[anchors[changed_rows]] * features[changed_ids], axis=1, dtype=np.float32)
        removed = before[changed_rows, changed_cols]
        after[changed_rows, changed_cols] = replacements
        before_values.append(before.ravel())
        after_values.append(after.ravel())
        removed_values.append(removed)
        replacement_values.append(replacements)
        duplicate_before = 1 - len(torch.unique(base)) / count
        duplicate_after = np.array([1 - len(np.unique(row)) / count for row in actual.numpy()])
        rows.append({"batch_number_zero_based": int(number), "anchors": len(anchors),
                     "negative_slots": len(anchors) * count, "replaced_slots": changes["rejected"],
                     "replacement_fraction": changes["rejected"] / (len(anchors) * count),
                     "zero_filter_anchors": changes["zero_filter_anchor_count"],
                     "duplicate_rate_before": duplicate_before, "duplicate_rate_after": float(duplicate_after.mean()),
                     "teacher_cosine_before": float(before.mean(dtype=np.float64)),
                     "teacher_cosine_after": float(after.mean(dtype=np.float64)),
                     "removed_cosine_mean": _mean(removed), "replacement_cosine_mean": _mean(replacements)})
    values = {name: np.concatenate(arrays) for name, arrays in (
        ("before", before_values), ("after", after_values), ("removed", removed_values), ("replacement", replacement_values))}
    histogram = []
    edges = np.linspace(-1.000001, 1.000001, 101)
    counts = {name: np.histogram(value, edges)[0] for name, value in values.items()}
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        histogram.append({"low": float(low), "high": float(high), **{name: int(v[index]) for name, v in counts.items()}})
    weights = np.array([row["anchors"] for row in rows])
    summary = {"batch_numbers_zero_based": selected.tolist(), "anchors": int(weights.sum()),
               "negative_slots": int(weights.sum() * count), "replaced_slots": sum(row["replaced_slots"] for row in rows),
               "zero_filter_anchors": sum(row["zero_filter_anchors"] for row in rows),
               "duplicate_rate_before": float(np.average([row["duplicate_rate_before"] for row in rows], weights=weights)),
               "duplicate_rate_after": float(np.average([row["duplicate_rate_after"] for row in rows], weights=weights)),
               "teacher_cosine": {name: _distribution(value) for name, value in values.items()}}
    summary["replacement_fraction"] = summary["replaced_slots"] / summary["negative_slots"]
    summary["zero_filter_fraction"] = summary["zero_filter_anchors"] / summary["anchors"]
    return summary, rows, histogram


def audit_negatives(output, inputs, graph):
    """Write an independently sealed sibling diagnostic; return summary.json data."""
    output = Path(output) / "negative_diagnostics"
    verified_graph = load_graph(graph["path"])
    if graph["signature"] != verified_graph["signature"]:
        raise ValueError("Supplied graph differs from the sealed graph")
    graph = verified_graph
    train = inputs["datasets"]["train"]
    ids = np.asarray(train["token_map_ids"])
    if not np.array_equal(ids, graph["token_map_ids"]):
        raise ValueError("Posthoc labels and frozen graph do not share train token order")
    labels = np.asarray(inputs["labels"])[ids]
    if labels.shape != (len(ids), 8) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected binary 8-label train wafer proxies")
    singleton = np.where(labels.sum(1) == 1, labels.argmax(1), -1).astype(np.int64)
    if "singleton_labels" in train and not np.array_equal(singleton, train["singleton_labels"]):
        raise ValueError("Train singleton labels disagree with source wafer labels")
    teacher_binding = graph["metadata"]["identity"]["original"]
    teacher_path = Path(teacher_binding["source_path"])
    if file_hash(teacher_path) != teacher_binding["file_sha256"]:
        raise ValueError("Frozen teacher features changed before posthoc diagnostics")
    science = {"version": 1, "scope": "train_only_posthoc_no_feedback_to_graph_or_training",
        "context_signature": inputs["refinement_context_signature"], "graph_signature": graph["signature"],
        "teacher": teacher_binding, "token_map_ids": _array_identity(ids), "proxy_labels": _array_identity(labels),
        "source_binding_sha256": digest(inputs["protection"]), "batch_size": inputs["config"]["batch_size"],
        "permutation": "legacy loader epoch0 seed2026; all train tokens exactly once",
        "negative_count": 5000, "sample_batches": "four evenly spaced batches including first and last",
        "base_diagnostic_seed": 2026 + 3381, "replacement_diagnostic_seed_argument": 2026 + 7001,
        "score": "frozen original teacher normalized FP64 then rounded FP32; raw cosine",
        "implementation_sha256": file_hash(__file__)}
    signature = digest(science)
    if verified(output, signature):
        from wbm_retrieval.common import read_json
        return read_json(output / "summary.json")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Incomplete negative diagnostic is retained; use a fresh experiment output")
    identity(output, "cnn_train_negative_diagnostics", signature)
    record(output / "manifest.json", signed({"science": science, "diagnostic_signature": signature}))
    config = {"batch_size": inputs["config"]["batch_size"], "num_workers": 0, "device": "cpu"}
    batches = [batch.numpy() for batch in loader(torch.arange(len(ids)), config, 2026, True)]
    evaluable, hits, relevant = _proxy_counts(graph, singleton)
    expected = _expected_exposure(graph, batches)
    degree = np.diff(graph["indptr"])
    with_precision = evaluable > 0
    with_recall = relevant > 0
    per_anchor = []
    for row in range(len(ids)):
        per_anchor.append({"token_index": row, "map_id": int(ids[row]), "singleton_proxy": int(singleton[row]),
            "degree": int(degree[row]), "evaluable_single_label_neighbors": int(evaluable[row]),
            "same_label_neighbors": int(hits[row]), "all_cross_wafer_same_label_tokens": int(relevant[row]),
            "proxy_precision": _ratio(hits[row], evaluable[row]), "proxy_recall": _ratio(hits[row], relevant[row]),
            **{key: value[row].item() for key, value in expected.items()}})
    write_csv(output / "per_anchor.csv", per_anchor, list(per_anchor[0]))
    features = _normalize(np.load(teacher_path, mmap_mode="r", allow_pickle=False))
    observed, observed_rows, histogram = _sample_audit(graph, features, batches)
    write_csv(output / "observed_batches.csv", observed_rows, list(observed_rows[0]))
    write_csv(output / "hardness_histogram.csv", histogram, list(histogram[0]))
    class_rows = []
    for label in range(8):
        members = singleton == label
        class_rows.append({"label_index": label, "anchor_tokens": int(members.sum()),
            "evaluable_neighbor_edges": int(evaluable[members].sum()), "same_label_edges": int(hits[members].sum()),
            "proxy_precision_micro": _ratio(hits[members].sum(), evaluable[members].sum()),
            "proxy_recall_micro": _ratio(hits[members].sum(), relevant[members].sum())})
    write_csv(output / "proxy_per_class.csv", class_rows, list(class_rows[0]))
    summary = {"signature": signature, "scope": science["scope"], "graph_signature": graph["signature"],
        "tokens": len(ids), "single_label_anchor_tokens": int(np.count_nonzero(singleton >= 0)),
        "graph": {"degree": _distribution(degree), "zero_degree_count": int(np.count_nonzero(degree == 0)),
                  "zero_degree_fraction": float(np.mean(degree == 0)), "directed_edges": int(degree.sum())},
        "single_label_proxy": {"evaluable_neighbor_edges": int(evaluable.sum()), "same_label_edges": int(hits.sum()),
            "edge_fraction_evaluable": _ratio(evaluable.sum(), degree.sum()),
            "precision_micro": _ratio(hits.sum(), evaluable.sum()),
            "precision_anchor_mean": _mean(hits[with_precision] / evaluable[with_precision]),
            "recall_micro": _ratio(hits.sum(), relevant.sum()),
            "recall_anchor_mean": _mean(hits[with_recall] / relevant[with_recall]),
            "anchors_with_defined_precision": int(with_precision.sum()), "anchors_with_defined_recall": int(with_recall.sum()),
            "precision_definition": "same-single-label graph edges / graph edges between two single-label wafer tokens",
            "recall_definition": "same-single-label graph edges / ALL cross-wafer train tokens with that single label"},
        "expected_epoch0": {key: _distribution(value) for key, value in expected.items()
                              if key not in ("batch_number", "batch_size")},
        "observed_diagnostic_sample": observed,
        "limitations": ["Wafer labels are region proxies; mixed-wafer graph edges have no evaluated proxy label.",
            "Four-batch samples use independent diagnostic draws, not the actual training negative sequence.",
            "Expected statistics are conditional on the fixed epoch0 batch partition and uniform sampling with replacement.",
            "Teacher cosine measures hardness in frozen h512 space, not the evolving student z128 NCE objective.",
            "High proxy precision cannot isolate semantic false negatives from changes to negative hardness."]}
    record(output / "summary.json", summary)
    seal(output, signature)
    return summary
