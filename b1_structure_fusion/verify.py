"""Read-only independent checks of saved B1 scores and per-query metrics.

This module does not call either production metric evaluator. Results returned
here are written outside completed evaluation directories by the entry point.
"""
import csv
from itertools import groupby
from pathlib import Path

import numpy as np

from experiments.proposed.f_segmentation_geometry import pipeline as geo
from wbm_retrieval.common import read_json, file_hash
from wbm_retrieval.exact_metrics import tie_keys


def _records(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return {int(row["query_id"]): row for row in csv.DictReader(handle)}


def independent_query_metrics(query_labels, candidate_labels, ranked_labels, ks):
    """Compute exact and coverage directly from full multi-hot label vectors."""
    query = np.asarray(query_labels, dtype=int)
    candidates = np.asarray(candidate_labels, dtype=int)
    ranked = np.asarray(ranked_labels, dtype=int)
    if query.ndim != 1 or query.sum() <= 0:
        raise ValueError("Expected a nonempty multi-hot query")
    exact_pool = np.all(candidates == query, axis=1)
    exact_rank = np.all(ranked == query, axis=1)
    denominator = int(exact_pool.sum())
    pool_coverage = np.count_nonzero(candidates & query, axis=1)/query.sum()
    ranked_coverage = np.count_nonzero(ranked & query, axis=1)/query.sum()
    full_pool = np.all(candidates[:, query == 1] == 1, axis=1)
    full_rank = np.all(ranked[:, query == 1] == 1, axis=1)
    full_denominator = int(full_pool.sum())
    # A scalar prefix loop is independent of the evaluator's vectorized AP implementation.
    exact_hits = full_hits = 0
    exact_ap = full_ap = 0.
    prefixes = {0: (0, 0, 0., 0.)}
    for position, (exact, full) in enumerate(zip(exact_rank, full_rank), 1):
        exact_hits += int(exact)
        full_hits += int(full)
        if exact:
            exact_ap += exact_hits/position
        if full:
            full_ap += full_hits/position
        prefixes[position] = (exact_hits, full_hits, exact_ap, full_ap)
    ideal_all = sorted(pool_coverage.tolist(), reverse=True)
    result = {"exact": {}, "coverage": {}}
    for k in ks:
        exact_hits, full_hits, exact_ap_numerator, full_ap_numerator = prefixes[min(k, len(ranked))]
        ideal = ideal_all[:k]
        dcg = sum(float(value)/np.log2(r+2) for r, value in enumerate(ranked_coverage[:k]))
        idcg = sum(float(value)/np.log2(r+2) for r, value in enumerate(ideal))
        result["exact"].update({f"Hit@{k}": float(exact_hits > 0), f"Precision@{k}": exact_hits/k,
              f"Recall@{k}": exact_hits/denominator if denominator else None,
              f"AP@{k}": exact_ap_numerator/denominator if denominator else None})
        result["coverage"].update({f"Coverage-nDCG@{k}": dcg/idcg if idcg else None,
              f"MeanCoverage@{k}": float(ranked_coverage[:k].sum())/k,
              f"FullCoverageHit@{k}": float(full_hits > 0), f"FullCoveragePrecision@{k}": full_hits/k,
              f"FullCoverageRecall@{k}": full_hits/full_denominator if full_denominator else None,
              f"FullCoverageAP@{k}": full_ap_numerator/full_denominator if full_denominator else None})
    return result


def verify_evaluation(folder, pool, bundle, ks, signature, tie_seed=2026):
    folder = Path(folder)
    if not geo.completed(folder, signature):
        raise ValueError("Cannot verify an incomplete evaluation")
    manifest = read_json(folder / "evaluation_manifest.json")
    if manifest["split"] != pool.split_name or manifest["ks"] != list(ks):
        raise ValueError("Verification split/K differs from the completed evaluation")
    expected = {"exact": _records(folder / "per_query.csv"),
                "coverage": _records(folder / "coverage_per_query.csv")}
    query_ids = set(map(int, pool))
    if any(set(rows) != query_ids for rows in expected.values()):
        raise ValueError("Per-query reports do not cover the frozen queries exactly")
    sums = {"exact": {}, "coverage": {}}
    seen, count, maximum_error = set(), 0, 0.
    labels = np.asarray(bundle.labels, dtype=int)
    with (folder / "rankings.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for query, records in groupby(reader, key=lambda row: int(row["query_id"])):
            if query not in query_ids or query in seen:
                raise ValueError("Ranking query coverage is invalid")
            seen.add(query)
            rows = list(records)
            ranked = np.array([int(row["candidate_id"]) for row in rows], dtype=np.int64)
            candidates = pool[query]
            if ([int(row["rank"]) for row in rows] != list(range(1, max(ks)+1)) or
                    len(set(ranked.tolist())) != len(ranked) or not np.isin(ranked, candidates).all()):
                raise ValueError("Saved ranking has missing ranks, duplicates, self or invalid candidates")
            scores = np.array([float(row["similarity_score"]) for row in rows])
            if not np.isfinite(scores).all() or np.any(scores[:-1] < scores[1:]):
                raise ValueError("Saved scores are nonfinite or not descending")
            order = np.lexsort((ranked, tie_keys(query, ranked, tie_seed), -scores))
            if not np.array_equal(order, np.arange(len(ranked))):
                raise ValueError("Saved exact-score ties violate the frozen SHA ordering")
            spec = manifest["spec"]
            b0 = np.array([float(row["b0_score"]) for row in rows])
            distances = np.array([float(row["structure_distance"]) for row in rows])
            if not np.isfinite(distances).all() or np.any(distances < 0) or np.any(distances > 1):
                raise ValueError("Invalid saved structure distance")
            calculated_score = (b0 if spec["family"] == "B0" else np.exp(-distances)
                                if spec["family"] == "J_only" else b0*np.exp(-spec["strength"]*distances))
            np.testing.assert_allclose(scores, calculated_score, atol=1e-13, rtol=1e-13)
            reconstructed_b0 = np.array([(.5*float(row["optimal_f_score"])+.5*float(row["global_score"]))
                                         *np.exp(-.5*float(row["angular_distance"])) for row in rows])
            np.testing.assert_allclose(b0, reconstructed_b0, atol=1e-13, rtol=1e-13)
            metrics = independent_query_metrics(labels[query], labels[candidates], labels[ranked], ks)
            for family, values in metrics.items():
                for field, value in values.items():
                    saved = expected[family][query][field]
                    saved = None if saved == "" else float(saved)
                    if (value is None) != (saved is None):
                        raise ValueError(f"Undefined metric mismatch: {query}/{field}")
                    if value is not None:
                        error = abs(value-saved)
                        maximum_error = max(maximum_error, error)
                        if error > 1e-12:
                            raise ValueError(f"Independent metric mismatch: {query}/{field}")
                        sums[family].setdefault(field, []).append(value)
            count += len(rows)
    if seen != query_ids:
        raise ValueError("Ranking is missing frozen queries")
    for family, filename in (("exact", "metrics.json"), ("coverage", "coverage_metrics.json")):
        report = read_json(folder / filename)
        for field, values in sums[family].items():
            name = "mAP"+field[2:] if field.startswith("AP@") else (
                "FullCoverage_mAP"+field[len("FullCoverageAP"):] if field.startswith("FullCoverageAP@") else field)
            if abs(float(np.mean(values))-report["metrics"][name]) > 1e-12:
                raise ValueError(f"Overall aggregation mismatch: {family}/{name}")
    return {"status": "verified", "split": pool.split_name, "variant": manifest["variant"],
            "num_queries": len(seen), "ranking_rows": count, "max_metric_absolute_error": maximum_error,
            "scope": "independent saved-score formulas, candidate IDs, SHA ties, all per-query exact/coverage metrics and overall means",
            "ranking_sha256": file_hash(folder / "rankings.csv"),
            "completion_sha256": file_hash(folder / "completed.json")}
