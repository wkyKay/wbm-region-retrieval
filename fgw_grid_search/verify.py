"""Independent full-candidate grid formulas, SHA ordering and label metrics.

Verification runs before completion sealing. It binds every scientific report
without referencing its own JSON or the later completion marker.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path

import numpy as np

from b1_structure_fusion.verify import independent_query_metrics

FORMULA_ATOL = 1e-12
SOURCE_ATOL = 2e-11


def _json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def independent_order(query, ids, scores, seed=2026):
    ids, scores = np.asarray(ids, dtype=np.int64), np.asarray(scores, dtype=float)
    if ids.ndim != 1 or scores.shape != ids.shape or len(set(ids.tolist())) != len(ids):
        raise ValueError("Candidate IDs and scores must be unique aligned vectors")
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite scores")
    keys = np.array([int.from_bytes(hashlib.sha256(
        f"exact_v1:tie:{int(seed)}:{int(query)}:{int(c)}".encode("ascii")
    ).digest()[:8], "big") for c in ids], dtype=np.uint64)
    return np.lexsort((ids, keys, -scores))


def compare_full_ranking(query, candidates, scores, saved_ids, seed=2026,
                         tolerance=SOURCE_ATOL):
    """Prove TopK membership/order against ALL candidates; never mask a mismatch."""
    candidates = np.asarray(candidates, dtype=np.int64)
    saved_ids = np.asarray(saved_ids, dtype=np.int64)
    if len(set(saved_ids.tolist())) != len(saved_ids) or not np.isin(saved_ids, candidates).all():
        raise ValueError("Saved TopK contains duplicate or non-gallery IDs")
    order = independent_order(query, candidates, scores, seed)
    expected = candidates[order[:len(saved_ids)]]
    positions = {int(c): i for i, c in enumerate(candidates)}
    bad = np.flatnonzero(expected != saved_ids)
    changes = []
    for i in bad:
        old, new = int(saved_ids[i]), int(expected[i])
        gap = float(scores[positions[new]] - scores[positions[old]])
        changes.append({"rank": int(i + 1), "saved_id": old, "recomputed_id": new,
                        "signed_score_gap": gap, "within_numeric_tolerance": abs(gap) <= tolerance})
    boundary = None
    if len(order) > len(saved_ids) and len(saved_ids):
        boundary = float(scores[order[len(saved_ids)-1]] - scores[order[len(saved_ids)]])
    return {"exact_order_match": not len(bad), "mismatch_count": len(changes),
            "numeric_only": bool(changes) and all(c["within_numeric_tolerance"] for c in changes),
            "changes": changes, "topk_boundary_gap": boundary}


def _unique_records(path):
    result = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            query = int(row["query_id"])
            if query in result:
                raise ValueError(f"Duplicate per-query row in {path}: {query}")
            result[query] = row
    return result


def _metric_name(name):
    if name.startswith("AP@"):
        return "mAP" + name[2:]
    if name.startswith("FullCoverageAP@"):
        return "FullCoverage_mAP" + name[len("FullCoverageAP"):]
    return name


def _label_signature(labels, family):
    if family == "exact":
        return "|".join(str(i) for i, value in enumerate(labels) if value)
    return "".join(map(str, labels.tolist()))


def independent_score(components, spec):
    """Explicit formulas independent of the production scoring module."""
    if not isinstance(spec, dict) or spec.get("family") not in (
            "sum_raw", "sum_minmax", "weighted_product", "legacy_fgw"):
        raise ValueError("Unknown grid score specification")
    weights = spec.get("weights")
    if (not isinstance(weights, list) or len(weights) != 3 or
            any(isinstance(x, bool) or not isinstance(x, (int, float)) or
                not math.isfinite(x) or x < 0 for x in weights) or weights[0] <= 0):
        raise ValueError("Weights must be finite nonnegative with F > 0")
    family = spec["family"]
    if family == "legacy_fgw":
        if set(spec) != {"family", "weights"} or weights != [.5, .5, 1.]:
            raise ValueError("Legacy reference is fixed")
    else:
        d, ticks = spec.get("denominator"), spec.get("ticks")
        if (set(spec) != {"family", "weights", "ticks", "denominator"} or
                isinstance(d, bool) or not isinstance(d, int) or not 1 <= d <= 20 or
                not isinstance(ticks, list) or len(ticks) != 3 or
                any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in ticks) or
                ticks[0] < 1 or sum(ticks) != d or
                any(abs(w - t / d) > 1e-15 for w, t in zip(weights, ticks))):
            raise ValueError("Grid ticks/weights must sum to one with F > 0")
    if not isinstance(components, dict) or set(components) != {"F", "G", "WBBS"}:
        raise ValueError("Expected full-candidate F/G/WBBS vectors")
    arrays = [np.asarray(components[key], dtype=np.float64) for key in ("F", "G", "WBBS")]
    if (any(x.ndim != 1 or not len(x) or x.shape != arrays[0].shape for x in arrays) or
            any(not np.isfinite(x).all() or np.any((x < 0) | (x > 1)) for x in arrays)):
        raise ValueError("Full-candidate component lies outside aligned finite [0,1]")
    if family == "legacy_fgw":
        return (.5 * arrays[0] + .5 * arrays[1]) * arrays[2]
    if weights == [1., 0., 0.]:
        return arrays[0].copy()
    if family == "weighted_product":
        result = np.ones(arrays[0].shape, dtype=np.float64)
        for weight, values in zip(weights, arrays):
            if weight:
                result *= np.power(values, weight)
        return result
    result = np.zeros(arrays[0].shape, dtype=np.float64)
    for weight, values in zip(weights, arrays):
        if not weight:
            continue
        if family == "sum_minmax":
            low, high = float(values.min()), float(values.max())
            values = (values - low) / (high - low) if high > low else np.zeros_like(values)
        result += weight * values
    return result


def _assert_value(saved, actual, context, tolerance=FORMULA_ATOL):
    saved = None if saved in ("", None) else float(saved)
    if (saved is None) != (actual is None):
        raise ValueError(f"Undefined metric mismatch: {context}")
    if actual is None:
        return 0.
    error = abs(float(saved) - actual)
    if not np.isfinite(saved) or error > tolerance:
        raise ValueError(f"Metric mismatch: {context}: {saved} != {actual}")
    return error


def verify_evaluation(folder, pool, bundle, ks, signature, provider, tie_seed=2026):
    folder = Path(folder)
    manifest = _json(folder / "evaluation_manifest.json")
    binding = {"signature": provider.signature, "completed_sha256": _sha(Path(provider.marker))}
    if (manifest.get("signature") != signature or manifest["split"] != pool.split_name or
            manifest["ks"] != list(ks) or manifest.get("tie_seed") != tie_seed or
            manifest.get("provider_source") != binding or provider.split != pool.split_name or
            provider.protocol_signature != pool.protocol_signature or
            not np.array_equal(provider.query_ids, list(pool)) or
            not np.array_equal(provider.gallery_ids, pool.gallery)):
        raise ValueError("Evaluation manifest/source differs from verification inputs")
    expected = {"exact": _unique_records(folder / "per_query.csv"),
                "coverage": _unique_records(folder / "coverage_per_query.csv")}
    query_ids = set(map(int, pool))
    gallery = set(map(int, pool.gallery))
    if len(gallery) != len(pool.gallery) or not query_ids <= gallery or pool.candidates_per_query != len(gallery)-1:
        raise ValueError("Invalid complete-gallery protocol")
    if any(set(rows) != query_ids for rows in expected.values()):
        raise ValueError("Per-query report does not cover each frozen query exactly once")
    labels = np.asarray(bundle.labels, dtype=int)
    classes = np.asarray(bundle.class_ids)
    sums = {family: defaultdict(lambda: defaultdict(list)) for family in expected}
    seen, count, maximum_error, score_error, component_error = set(), 0, 0., 0., 0.
    full_candidate_count = 0
    with (folder / "rankings.csv").open(newline="", encoding="utf-8") as handle:
        for query, iterator in groupby(csv.DictReader(handle), key=lambda r: int(r["query_id"])):
            if query not in query_ids or query in seen:
                raise ValueError("Missing, repeated, or unexpected ranking query")
            seen.add(query)
            rows = list(iterator)
            ranked = np.array([int(r["candidate_id"]) for r in rows], dtype=np.int64)
            candidates = np.asarray(pool[query], dtype=np.int64)
            if len(candidates) != len(gallery)-1 or set(map(int, candidates)) != gallery-{query}:
                raise ValueError("Candidates must cover the complete gallery except self")
            if (len(rows) != max(ks) or [int(r["rank"]) for r in rows] != list(range(1, max(ks)+1))
                    or len(set(ranked.tolist())) != len(ranked) or query in ranked
                    or not np.isin(ranked, candidates).all()):
                raise ValueError("Invalid saved TopK candidate identities or ranks")
            scores = np.array([float(r["similarity_score"]) for r in rows])
            values = provider.scores(query, candidates)
            full_scores = independent_score(values, manifest["spec"])
            expected_order = independent_order(query, candidates, full_scores, tie_seed)
            expected_ids = candidates[expected_order[:len(rows)]]
            if not np.array_equal(ranked, expected_ids):
                raise ValueError("Saved TopK differs from independent full-candidate ordering")
            formula = full_scores[expected_order[:len(rows)]]
            if not np.isfinite(scores).all() or not np.allclose(scores, formula, atol=FORMULA_ATOL, rtol=0):
                raise ValueError("Saved fusion score formula differs")
            score_error = max(score_error, float(np.max(np.abs(scores - formula))))
            for field, branch in (("optimal_f_score", "F"), ("global_score", "G"), ("wbbs_score", "WBBS")):
                saved_values = np.array([float(row[field]) for row in rows], dtype=np.float64)
                source_values = np.asarray(values[branch])[expected_order[:len(rows)]]
                if not np.array_equal(saved_values, source_values):
                    raise ValueError("Saved component differs from full-score provider: " + field)
                component_error = max(component_error, float(np.max(np.abs(saved_values - source_values))))
            full_candidate_count += len(candidates)
            values = independent_query_metrics(labels[query], labels[candidates], labels[ranked], ks)
            label_count, cls = int(labels[query].sum()), str(int(classes[query]))
            groups = ["all", "class/"+cls, "by_label_count/"+str(label_count),
                      "single_label" if label_count == 1 else "mixed_label"]
            for family, metrics in values.items():
                saved_row = expected[family][query]
                if (int(saved_row["class_id"]) != int(classes[query]) or
                    int(saved_row["label_count"]) != label_count or
                    int(saved_row["pool_size"]) != len(candidates) or
                    int(saved_row["returned"]) != len(ranked) or
                    saved_row["label_signature"] != _label_signature(labels[query], family)):
                    raise ValueError("Per-query metadata differs from frozen identities/labels")
                denominator_field = "relevant_in_pool" if family == "exact" else "full_coverage_in_pool"
                relevant = (np.all(labels[candidates] == labels[query], axis=1) if family == "exact" else
                            np.all(labels[candidates][:, labels[query].astype(bool)] == 1, axis=1))
                if int(saved_row[denominator_field]) != int(relevant.sum()):
                    raise ValueError("Per-query denominator differs from full-gallery labels")
                for field, value in metrics.items():
                    maximum_error = max(maximum_error, _assert_value(saved_row[field], value, (query, family, field)))
                    if value is not None:
                        for group in groups:
                            sums[family][group][_metric_name(field)].append(value)
            count += len(rows)
    if seen != query_ids:
        raise ValueError("Ranking misses frozen queries")
    aggregation_count = 0
    for family, filename, classfile in (("exact", "metrics.json", "per_class.csv"),
                                       ("coverage", "coverage_metrics.json", "coverage_per_class.csv")):
        report = _json(folder/filename)
        if (report.get("split") != pool.split_name or report.get("num_queries") != len(query_ids) or
                report.get("ks") != list(ks) or report.get("metric_version") != family + "_v1" or
                report.get("metadata", {}).get("signature") != signature or
                report.get("retrieval_protocol_signature", report.get("metadata", {}).get("retrieval_protocol_signature")) != pool.protocol_signature):
            raise ValueError("Aggregate report protocol/signature differs")
        means = {g: {k: math.fsum(v)/len(v) for k,v in metrics.items()} for g,metrics in sums[family].items()}
        classes_present = [g for g in means if g.startswith("class/")]
        if report.get("num_classes") != len(classes_present):
            raise ValueError("Aggregate class count differs")
        means["class_macro"] = {metric: math.fsum(means[g][metric] for g in classes_present
                              if metric in means[g]) / sum(metric in means[g] for g in classes_present)
                              for metric in means["all"]}
        saved = {"all": report["metrics"], "class_macro": report["class_macro_metrics"]}
        for name, group in report["groups"].items():
            if name == "by_label_count":
                for key, value in group.items():
                    if value["num_queries"] != sum(int(labels[q].sum()) == int(key) for q in query_ids):
                        raise ValueError("Label-count group size differs")
                saved.update({"by_label_count/"+str(k): v["metrics"] for k,v in group.items()})
            else:
                members = sum((int(labels[q].sum()) == 1) == (name == "single_label") for q in query_ids)
                if group["num_queries"] != members:
                    raise ValueError("Single/mixed group size differs")
                saved[name] = group["metrics"]
        with (folder/classfile).open(newline="", encoding="utf-8") as handle:
            seen_classes = set()
            for row in csv.DictReader(handle):
                key = "class/"+str(int(row["class_id"]))
                if key in seen_classes:
                    raise ValueError("Duplicate class report")
                seen_classes.add(key)
                members = [q for q in query_ids if int(classes[q]) == int(row["class_id"])]
                if (not members or int(row["num_queries"]) != len(members) or
                        row["label_signature"] != _label_signature(labels[members[0]], family)):
                    raise ValueError("Per-class identities/counts differ")
                saved[key] = {k:v for k,v in row.items() if "@" in k}
        if set(means) != set(saved):
            raise ValueError("Class/group aggregation identities differ")
        stems = (("Hit", "Precision", "Recall", "mAP") if family == "exact" else
                 ("Coverage-nDCG", "MeanCoverage", "FullCoverageHit", "FullCoveragePrecision", "FullCoverageRecall", "FullCoverage_mAP"))
        metric_names = {f"{stem}@{k}" for stem in stems for k in ks}
        for group, metrics in means.items():
            if set(saved[group]) != metric_names:
                raise ValueError("Aggregate metric keys differ")
            for metric in metric_names:
                maximum_error = max(maximum_error, _assert_value(saved[group][metric], metrics.get(metric), (family, group, metric)))
                aggregation_count += 1
    files = ("run_identity.json", "evaluation_manifest.json", "rankings.csv",
             "metrics.json", "per_query.csv", "per_class.csv",
             "coverage_metrics.json", "coverage_per_query.csv", "coverage_per_class.csv")
    if binding != {"signature": provider.signature, "completed_sha256": _sha(Path(provider.marker))}:
        raise ValueError("Source marker changed during verification")
    return {"status": "verified", "signature": signature, "provider_source": binding,
            "split": pool.split_name, "variant": manifest["variant"], "ks": list(ks), "tie_seed": tie_seed,
            "num_queries": len(seen), "ranking_rows": count,
            "full_candidates_checked": full_candidate_count,
            "all_queries_full_candidate_topk_exact": True,
            "max_metric_absolute_error": maximum_error, "max_saved_score_absolute_error": score_error,
            "max_saved_component_absolute_error": component_error,
            "aggregation_values_checked": aggregation_count,
            "scope": "Every query: independent full-candidate fusion formula and SHA TopK ordering; exact saved raw components; full-gallery label denominators; every K/query/class/macro/count-group metric.",
            "ranking_sha256": _sha(folder / "rankings.csv"),
            "files": {name: _sha(folder / name) for name in files}}
