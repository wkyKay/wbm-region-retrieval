"""One label-containment evaluator for all methods and all frozen candidate pools."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import iter_csv, write_csv, write_json
from .data import containment


def dcg(values):
    return float(np.sum(np.asarray(values, dtype=float) / np.log2(np.arange(len(values)) + 2)))


def query_metrics(relevance, ideal_relevance, ks):
    relevance = np.asarray(relevance, dtype=float)
    ideal = np.sort(np.asarray(ideal_relevance, dtype=float))[::-1]
    full = np.isclose(relevance, 1.0)
    total_full = int(np.isclose(ideal, 1.0).sum())
    values = {}
    for k in ks:
        n = min(k, len(relevance))
        denominator = dcg(ideal[:n])
        values[f"ContainmentNDCG@{k}"] = dcg(relevance[:n]) / denominator if denominator else None
        values[f"MeanContainment@{k}"] = float(relevance[:n].mean()) if n else 0.
        values[f"FullCoveragePrecision@{k}"] = float(full[:n].mean()) if n else 0.
        values[f"FullCoverageRecall@{k}"] = float(full[:n].sum() / total_full) if total_full else None
        values[f"FullCoverageHit@{k}"] = float(full[:n].any()) if total_full else None
    positives = np.flatnonzero(full)
    values["mAP_full"] = float((np.cumsum(full)[positives] / (positives + 1)).sum() / total_full) if total_full else None
    values["MRR_full"] = 1.0 / (int(positives[0]) + 1) if len(positives) else 0. if total_full else None
    return values


def bootstrap_interval(values, repetitions, rng):
    data = np.array([value for value in values if value is not None], dtype=float)
    if not len(data) or not repetitions:
        return None
    # Small chunks avoid allocating repetitions * num_queries all at once.
    means = []
    for start in range(0, repetitions, 100):
        indices = rng.integers(len(data), size=(min(100, repetitions - start), len(data)))
        means.extend(data[indices].mean(1).tolist())
    low, high = np.quantile(means, [0.025, 0.975])
    return [float(low), float(high)]


def evaluate_ranking(path, protocol, settings, seed):
    by_query = {q: [] for q in protocol.queries}
    for row in iter_csv(path):
        q, c = int(row["query_id"]), int(row["candidate_id"])
        score = float(row["similarity_score"])
        if q not in by_query or not np.isfinite(score):
            raise ValueError("Unknown query or non-finite ranking score")
        by_query[q].append((c, score, int(row["rank"])))
    rows = []
    for q in protocol.queries:
        values = by_query[q]
        ids = [value[0] for value in values]
        if len(ids) != len(set(ids)) or set(ids) != set(protocol.candidates[q]):
            raise ValueError(f"Ranking candidate set differs from the frozen pool for query {q}")
        values.sort(key=lambda value: (-value[1], value[0]))
        if [value[2] for value in values] != list(range(1, len(values) + 1)):
            raise ValueError(f"Ranking order/rank column mismatch for query {q}")
        relevance = containment(protocol.labels[q], np.array([protocol.labels[c] for c, _, _ in values]))
        # Complete candidate set, not merely returned top K, determines IDCG and recall.
        ideal = containment(protocol.labels[q], np.array([protocol.labels[c] for c in protocol.candidates[q]]))
        rows.append({"query_id": q, **query_metrics(relevance, ideal, settings["ks"])})
    metrics = {}
    intervals = {}
    rng = np.random.default_rng(seed)
    for key in rows[0]:
        if key == "query_id":
            continue
        numbers = [row[key] for row in rows if row[key] is not None]
        metrics[key] = float(np.mean(numbers)) if numbers else None
        intervals[key] = bootstrap_interval([row[key] for row in rows], settings["bootstrap_samples"], rng)
    result = {"relevance": "query_label_containment", "gain": "identity", "metrics": metrics,
              "query_bootstrap_95ci": intervals, "num_queries": len(rows),
              "queries_with_full_coverage_candidates": sum(row["mAP_full"] is not None for row in rows),
              "ties": "descending score, then ascending candidate_id", "ranking": str(path)}
    parent = Path(path).parent
    write_json(parent / "metrics.json", result)
    write_csv(parent / "per_query_metrics.csv", rows, list(rows[0]))
    return result


def evaluate_all(config, protocol, methods=None):
    from .common import budget_name
    root = Path(config["output_dir"]) / "retrieval"
    names = methods or ["handcrafted", "proposed"] + [budget_name(b) for b in config["global"]["budgets"]]
    summary = []
    for name in names:
        path = root / name / "rankings.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing ranking {path}; run retrieval first")
        result = evaluate_ranking(path, protocol, config["evaluation"], config["seed"])
        summary.append({"method": name, **result["metrics"]})
        print(f"Evaluated {name}: {result['metrics']}", flush=True)
    write_csv(Path(config["output_dir"]) / "comparison.csv", summary, list(summary[0]))
    write_json(Path(config["output_dir"]) / "comparison.json", summary)
    return summary


def compare_paired(config, left="proposed", right="handcrafted", metric="ContainmentNDCG@10"):
    root = Path(config["output_dir"]) / "retrieval"
    left_rows = {int(r["query_id"]): r for r in iter_csv(root / left / "per_query_metrics.csv")}
    right_rows = {int(r["query_id"]): r for r in iter_csv(root / right / "per_query_metrics.csv")}
    if set(left_rows) != set(right_rows):
        raise ValueError("Paired comparison requires identical query IDs")
    differences = [float(left_rows[q][metric]) - float(right_rows[q][metric]) for q in left_rows
                   if left_rows[q][metric] and right_rows[q][metric]]
    interval = bootstrap_interval(differences, config["evaluation"]["bootstrap_samples"], np.random.default_rng(config["seed"]))
    result = {"left": left, "right": right, "metric": metric,
              "mean_difference": float(np.mean(differences)) if differences else None,
              "paired_query_bootstrap_95ci": interval}
    write_json(Path(config["output_dir"]) / f"paired_{left}_vs_{right}.json", result)
    return result
