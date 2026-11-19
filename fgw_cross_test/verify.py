"""Independent verification of coverage-only, cross-combination evaluations.

No production scorer or metric evaluator is called. Every saved Top-K is
checked against all eligible source scores, and every metric is recomputed
from labels with the original coverage_v1 definitions.
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

ATOL = 1e-12
PROTOCOL = "coverage_cross_combination_v1"
RULE = "all_gallery_except_query_and_exact_label_combination"
FORMAL_KS = [1, 5, 10, 20, 50, 100, 500]
REPORT_FILES = ("run_identity.json", "evaluation_manifest.json", "rankings.csv",
                "coverage_metrics.json", "coverage_per_query.csv", "coverage_per_class.csv")
STEMS = ("Coverage-nDCG", "MeanCoverage", "FullCoverageHit", "FullCoveragePrecision",
         "FullCoverageRecall", "FullCoverage_mAP")


def _json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def independent_order(query, candidates, scores, seed=2026):
    candidates, scores = np.asarray(candidates), np.asarray(scores, dtype=float)
    if (candidates.ndim != 1 or candidates.dtype.kind not in "iu" or
            scores.shape != candidates.shape or len(np.unique(candidates)) != len(candidates)
            or not np.isfinite(scores).all()):
        raise ValueError("Invalid candidate/score vectors")
    keys = np.array([int.from_bytes(hashlib.sha256(
        f"exact_v1:tie:{int(seed)}:{int(query)}:{int(candidate)}".encode("ascii")
    ).digest()[:8], "big") for candidate in candidates], dtype=np.uint64)
    return np.lexsort((candidates, keys, -scores))


def independent_score(components, spec):
    """Compute the frozen formulas on the already-filtered complete vectors."""
    family, weights = spec.get("family"), spec.get("weights")
    if (family not in ("sum_raw", "sum_minmax", "weighted_product", "legacy_fgw")
            or not isinstance(weights, list) or len(weights) != 3
            or any(isinstance(w, bool) or not isinstance(w, (int, float)) or
                   not math.isfinite(w) or w < 0 for w in weights) or weights[0] <= 0):
        raise ValueError("Invalid frozen score specification")
    if family == "legacy_fgw":
        if set(spec) != {"family", "weights"} or weights != [.5, .5, 1.]:
            raise ValueError("Legacy specification changed")
    else:
        denominator, ticks = spec.get("denominator"), spec.get("ticks")
        if (set(spec) != {"family", "weights", "denominator", "ticks"}
                or type(denominator) is not int or not 1 <= denominator <= 20
                or not isinstance(ticks, list) or len(ticks) != 3
                or any(type(tick) is not int or tick < 0 for tick in ticks)
                or ticks[0] < 1 or sum(ticks) != denominator
                or weights != [tick / denominator for tick in ticks]):
            raise ValueError("Frozen weight ticks changed")
    if not isinstance(components, dict) or set(components) != {"F", "G", "WBBS"}:
        raise ValueError("Expected exactly F/G/WBBS source components")
    arrays = [np.asarray(components[name], dtype=np.float64) for name in ("F", "G", "WBBS")]
    if (any(a.ndim != 1 or not len(a) or a.shape != arrays[0].shape for a in arrays)
            or any(not np.isfinite(a).all() or np.any((a < 0) | (a > 1)) for a in arrays)):
        raise ValueError("Invalid full eligible source scores")
    if family == "legacy_fgw":
        return (.5 * arrays[0] + .5 * arrays[1]) * arrays[2]
    # This endpoint is explicitly frozen in the source scorer, including minmax.
    if weights == [1., 0., 0.]:
        return arrays[0].copy()
    if family == "weighted_product":
        result = np.ones_like(arrays[0])
        for weight, values in zip(weights, arrays):
            if weight:
                result *= np.power(values, weight)
        return result
    result = np.zeros_like(arrays[0])
    for weight, values in zip(weights, arrays):
        if not weight:
            continue
        if family == "sum_minmax":
            minimum, maximum = float(values.min()), float(values.max())
            values = ((values - minimum) / (maximum - minimum)
                      if maximum > minimum else np.zeros_like(values))
        result += weight * values
    return result


def independent_coverage(query, candidates, ranked, ks):
    """Scalar-prefix AP and discounted gains; denominators use every candidate."""
    query, candidates, ranked = (np.asarray(x, dtype=int) for x in (query, candidates, ranked))
    if query.ndim != 1 or query.sum() <= 0:
        raise ValueError("Query must have nonempty labels")
    if (candidates.ndim != 2 or ranked.ndim != 2 or candidates.shape[1] != len(query)
            or ranked.shape[1] != len(query)):
        raise ValueError("Invalid candidate/ranked labels")
    query_count = int(query.sum())
    pool_overlap = np.count_nonzero(candidates & query, axis=1)
    returned_overlap = np.count_nonzero(ranked & query, axis=1)
    relevant = int(np.count_nonzero(pool_overlap == query_count))
    ideal = sorted((int(x) / query_count for x in pool_overlap), reverse=True)
    gain = [int(x) / query_count for x in returned_overlap]
    hits = 0
    ap_terms, prefixes = [], [0]
    for rank, overlap in enumerate(returned_overlap, 1):
        if int(overlap) == query_count:
            hits += 1
            ap_terms.append(hits / rank)
        else:
            ap_terms.append(0.)
        prefixes.append(hits)
    metrics = {}
    for k in ks:
        hits = prefixes[min(k, len(gain))]
        numerator = math.fsum(ap_terms[:k])
        dcg = math.fsum(value / math.log2(rank + 2) for rank, value in enumerate(gain[:k]))
        idcg = math.fsum(value / math.log2(rank + 2) for rank, value in enumerate(ideal[:k]))
        metrics.update({f"Coverage-nDCG@{k}": dcg / idcg if idcg else None,
                        f"MeanCoverage@{k}": math.fsum(gain[:k]) / k,
                        f"FullCoverageHit@{k}": float(hits > 0),
                        f"FullCoveragePrecision@{k}": hits / k,
                        f"FullCoverageRecall@{k}": hits / relevant if relevant else None,
                        f"FullCoverageAP@{k}": numerator / relevant if relevant else None})
    return metrics, relevant


def _assert_value(saved, actual, context):
    saved = None if saved in (None, "") else float(saved)
    if (saved is None) != (actual is None):
        raise ValueError(f"Undefined metric mismatch: {context}")
    if actual is None:
        return 0.
    error = abs(saved - actual)
    if not math.isfinite(saved) or error > ATOL:
        raise ValueError(f"Metric mismatch: {context}: {saved} != {actual}")
    return error


def _records(path, key):
    result = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            identity = int(row[key])
            if identity in result:
                raise ValueError(f"Duplicate {key} report: {path}")
            result[identity] = row
    return result


def _metric_name(name):
    return name.replace("FullCoverageAP@", "FullCoverage_mAP@")


def _verify_selection(manifest):
    source = manifest.get("frozen_selection", {})
    if set(source) != {"path", "sha256"} or _sha(source["path"]) != source["sha256"]:
        raise ValueError("Frozen selection source hash changed")
    frozen = _json(source["path"])
    names, specs = frozen.get("ranked_variants"), frozen.get("specs")
    if (frozen.get("signature") != _digest({k: v for k, v in frozen.items() if k != "signature"})
            or frozen.get("top_k") != 10 or not isinstance(names, list)
            or len(names) != 10 or len(set(names)) != 10 or not isinstance(specs, dict)
            or set(specs) != set(names) or manifest.get("variant") not in names
            or manifest.get("spec") != specs[manifest["variant"]]
            or frozen.get("primary_variant") != names[0]
            or frozen.get("selection_split") != "valid"
            or frozen.get("test_parameter_search") is not False):
        raise ValueError("Evaluation differs from the ten frozen valid recipes")
    valid = frozen.get("valid_selection", {})
    if [row.get("variant") for row in valid.get("ranking", [])[:10]] != names:
        raise ValueError("Ten recipes are not the original valid prefix")
    return source


def verify_evaluation(folder, pool, bundle, ks, signature, provider, tie_seed=2026):
    folder = Path(folder)
    if any((folder / name).exists() for name in ("metrics.json", "per_query.csv", "per_class.csv")):
        raise ValueError("Cross-combination evaluation must not emit exact reports")
    initial_hashes = {name: _sha(folder / name) for name in REPORT_FILES}
    manifest, identity = _json(folder / "evaluation_manifest.json"), _json(folder / "run_identity.json")
    binding = {"signature": provider.signature, "completed_sha256": _sha(provider.marker)}
    queries, gallery = np.asarray(list(pool)), np.asarray(pool.gallery)
    labels, classes = np.asarray(bundle.labels), np.asarray(bundle.class_ids)
    if (queries.ndim != 1 or gallery.ndim != 1 or queries.dtype.kind not in "iu"
            or gallery.dtype.kind not in "iu" or not len(queries)
            or len(np.unique(queries)) != len(queries) or len(np.unique(gallery)) != len(gallery)
            or not np.isin(queries, gallery).all() or np.any(gallery < 0)
            or labels.ndim != 2 or not np.isin(labels, (0, 1)).all()
            or max(gallery) >= len(labels) or len(classes) != len(labels)
            or np.any(labels[gallery].sum(axis=1) == 0)):
        raise ValueError("Invalid query/gallery/label identities")
    if (manifest.get("signature") != signature or manifest.get("split") != "test"
            or pool.split_name != "test" or manifest.get("ks") != list(ks)
            or manifest.get("tie_seed") != tie_seed or manifest.get("num_queries") != len(queries)
            or manifest.get("provider_source") != binding or provider.split != "test"
            or not np.array_equal(provider.query_ids, queries)
            or not np.array_equal(provider.gallery_ids, gallery)
            or manifest.get("candidate_rule") != RULE
            or pool.metadata.get("protocol_version") != PROTOCOL
            or pool.metadata.get("candidate_rule") != RULE
            or manifest.get("derived_protocol") != pool.metadata
            or manifest.get("retrieval_protocol_signature") != pool.protocol_signature
            or hashlib.sha256(json.dumps(pool.metadata, sort_keys=True).encode()).hexdigest() != pool.protocol_signature
            or manifest.get("base_protocol_signature") != provider.protocol_signature
            or pool.metadata.get("base_protocol_signature") != provider.protocol_signature):
        raise ValueError("Evaluation manifest/source/candidate protocol differs")
    if (identity.get("evaluation_signature") != signature or
            any(manifest.get(k) != identity.get(k) for k in ("run_id", "started_at", "variant", "split"))):
        raise ValueError("Evaluation time identity differs")
    selection_source = _verify_selection(manifest)
    formal = manifest.get("formal_protocol", True)
    if formal and (len(queries) != 7199 or len(gallery) != 7199
                   or set(map(int, queries)) != set(map(int, gallery)) or list(ks) != FORMAL_KS):
        raise ValueError("Formal evaluation requires all 7199 queries/gallery and seven frozen K")
    if not ks or list(ks) != sorted(set(ks)) or any(type(k) is not int or k < 1 for k in ks):
        raise ValueError("Invalid cutoffs")
    expected = _records(folder / "coverage_per_query.csv", "query_id")
    if set(expected) != set(map(int, queries)):
        raise ValueError("Per-query report does not cover all frozen queries")
    perquery, seen, counts = {}, [], []
    candidate_digest = hashlib.sha256()
    max_error = score_error = component_error = 0.
    with (folder / "rankings.csv").open(newline="", encoding="utf-8") as stream:
        for query, records in groupby(csv.DictReader(stream), key=lambda r: int(r["query_id"])):
            if len(seen) >= len(queries) or query != int(queries[len(seen)]):
                raise ValueError("Ranking query order/identity differs")
            seen.append(query)
            candidates = np.asarray(pool[query])
            eligible = gallery[(gallery != query) & np.any(labels[gallery] != labels[query], axis=1)]
            if not np.array_equal(candidates, eligible):
                raise ValueError("Eligible candidate identities differ from independent label exclusion")
            counts.append(len(candidates))
            candidate_digest.update(json.dumps([query, candidates.tolist()], separators=(",", ":")).encode("ascii"))
            candidate_digest.update(b"\n")
            rows = list(records)
            ranked = np.array([int(row["candidate_id"]) for row in rows], dtype=np.int64)
            if (len(rows) != max(ks) or [int(row["rank"]) for row in rows] != list(range(1, max(ks) + 1))
                    or len(np.unique(ranked)) != len(ranked) or not np.isin(ranked, candidates).all()):
                raise ValueError("Invalid saved TopK identities/ranks")
            components = provider.scores(query, candidates)
            scores = independent_score(components, manifest["spec"])
            order = independent_order(query, candidates, scores, tie_seed)[:max(ks)]
            if not np.array_equal(ranked, candidates[order]):
                raise ValueError("Saved TopK differs from independent full-candidate ordering")
            saved_scores = np.array([float(row["similarity_score"]) for row in rows])
            if not np.isfinite(saved_scores).all() or not np.allclose(saved_scores, scores[order], atol=ATOL, rtol=0):
                raise ValueError("Saved fusion score formula differs")
            score_error = max(score_error, float(np.max(np.abs(saved_scores - scores[order]))))
            for field, key in (("optimal_f_score", "F"), ("global_score", "G"), ("wbbs_score", "WBBS")):
                saved = np.array([float(row[field]) for row in rows])
                if not np.array_equal(saved, np.asarray(components[key])[order]):
                    raise ValueError("Saved component differs from source: " + field)
            metrics, relevant = independent_coverage(labels[query], labels[candidates], labels[ranked], ks)
            label_count, cls = int(labels[query].sum()), int(classes[query])
            saved = expected[query]
            metadata = {"class_id": cls, "label_count": label_count, "pool_size": len(candidates),
                        "returned": len(ranked), "full_coverage_in_pool": relevant}
            if (any(int(saved[k]) != v for k, v in metadata.items()) or
                    saved["label_signature"] != "".join(map(str, labels[query].astype(int).tolist()))):
                raise ValueError("Per-query labels/counts/full-pool denominator differs")
            if {key for key in saved if "@" in key} != set(metrics):
                raise ValueError("Per-query metric names changed")
            for key, value in metrics.items():
                max_error = max(max_error, _assert_value(saved[key], value, (query, key)))
            perquery[query] = {"class": cls, "label_count": label_count, "relevant": relevant,
                               "metrics": {_metric_name(k): v for k, v in metrics.items()}}
    if seen != queries.tolist():
        raise ValueError("Ranking misses frozen queries")
    if (candidate_digest.hexdigest() != pool.metadata.get("candidate_identity_sha256")
            or min(counts) != pool.metadata.get("candidate_count_min")
            or max(counts) != pool.metadata.get("candidate_count_max")
            or pool.candidates_per_query != min(counts) or min(counts) < max(ks)):
        raise ValueError("Candidate identity digest/count range differs")
    if formal and (min(counts), max(counts)) != (6999, 7170):
        raise ValueError("Formal eligible candidate count range differs")
    report = _json(folder / "coverage_metrics.json")
    if (report.get("split") != "test" or report.get("test_evaluated") is not True
            or report.get("ks") != list(ks) or report.get("metric_version") != "coverage_v1"
            or report.get("metadata", {}).get("signature") != signature
            or report.get("metadata", {}).get("retrieval_protocol_signature") != pool.protocol_signature):
        raise ValueError("Coverage report metadata/signature differs")
    names = {f"{stem}@{k}" for stem in STEMS for k in ks}
    aggregated = 0

    def check_summary(saved, members, context):
        nonlocal max_error, aggregated
        if (saved.get("num_queries") != len(members)
                or saved.get("num_classes") != len({perquery[q]["class"] for q in members})
                or saved.get("queries_without_full_coverage_candidates") != sum(perquery[q]["relevant"] == 0 for q in members)
                or set(saved.get("metrics", {})) != names
                or set(saved.get("metric_query_counts", {})) != names):
            raise ValueError("Aggregate identities/counts/metric keys differ: " + context)
        means = {}
        for metric in names:
            values = [perquery[q]["metrics"][metric] for q in members if perquery[q]["metrics"][metric] is not None]
            means[metric] = math.fsum(values) / len(values) if values else None
            max_error = max(max_error, _assert_value(saved["metrics"][metric], means[metric], (context, metric)))
            if saved["metric_query_counts"][metric] != len(values):
                raise ValueError("Metric effective query count differs: " + context)
            aggregated += 1
        return means

    all_queries = list(perquery)
    check_summary(report, all_queries, "all")
    if set(report.get("groups", {})) != {"single_label", "mixed_label", "by_label_count"}:
        raise ValueError("Coverage groups changed")
    for group, single in (("single_label", True), ("mixed_label", False)):
        members = [q for q in all_queries if (perquery[q]["label_count"] == 1) == single]
        check_summary(report["groups"][group], members, group)
    label_counts = {str(perquery[q]["label_count"]) for q in all_queries}
    if set(report["groups"]["by_label_count"]) != label_counts:
        raise ValueError("Label-count groups changed")
    for label_count in label_counts:
        members = [q for q in all_queries if perquery[q]["label_count"] == int(label_count)]
        check_summary(report["groups"]["by_label_count"][label_count], members, "label_count/" + label_count)
    class_rows = _records(folder / "coverage_per_class.csv", "class_id")
    class_ids = {perquery[q]["class"] for q in all_queries}
    if set(class_rows) != class_ids:
        raise ValueError("Per-class report identities differ")
    class_means = defaultdict(list)
    for cls in class_ids:
        members = [q for q in all_queries if perquery[q]["class"] == cls]
        saved = class_rows[cls]
        if (int(saved["num_queries"]) != len(members)
                or int(saved["queries_without_full_coverage_candidates"]) != sum(perquery[q]["relevant"] == 0 for q in members)
                or int(saved["label_count"]) != perquery[members[0]]["label_count"]
                or any(not np.array_equal(labels[q], labels[members[0]]) for q in members)
                or saved["label_signature"] != "".join(map(str, labels[members[0]].astype(int).tolist()))
                or {key for key in saved if "@" in key} != names):
            raise ValueError("Per-class labels/counts differ")
        for metric in names:
            values = [perquery[q]["metrics"][metric] for q in members if perquery[q]["metrics"][metric] is not None]
            actual = math.fsum(values) / len(values) if values else None
            max_error = max(max_error, _assert_value(saved[metric], actual, (cls, metric)))
            if actual is not None:
                class_means[metric].append(actual)
            aggregated += 1
    if (set(report.get("class_macro_metrics", {})) != names
            or set(report.get("class_macro_metric_class_counts", {})) != names):
        raise ValueError("Class-macro metric keys differ")
    for metric in names:
        values = class_means[metric]
        actual = math.fsum(values) / len(values) if values else None
        max_error = max(max_error, _assert_value(report["class_macro_metrics"][metric], actual, ("class_macro", metric)))
        if report["class_macro_metric_class_counts"][metric] != len(values):
            raise ValueError("Class-macro effective count differs")
        aggregated += 1
    if formal:
        if report["num_classes"] != 37 or report["queries_without_full_coverage_candidates"] != 999:
            raise ValueError("Formal class/no-full-coverage counts differ")
        for metric, count in report["metric_query_counts"].items():
            actual = 7000 if metric.startswith("Coverage-nDCG@") else (6200 if metric.startswith(("FullCoverageRecall@", "FullCoverage_mAP@")) else 7199)
            if count != actual:
                raise ValueError("Formal effective query counts differ")
    final_hashes = {name: _sha(folder / name) for name in REPORT_FILES}
    if (final_hashes != initial_hashes or _sha(selection_source["path"]) != selection_source["sha256"]
            or binding != {"signature": provider.signature, "completed_sha256": _sha(provider.marker)}):
        raise ValueError("Reports/source changed during verification")
    return {"status": "verified", "signature": signature, "provider_source": binding,
            "split": "test", "variant": manifest["variant"], "ks": list(ks), "tie_seed": tie_seed,
            "num_queries": len(seen), "ranking_rows": len(seen) * max(ks),
            "candidate_identity_sha256": candidate_digest.hexdigest(),
            "full_candidates_checked": sum(counts), "candidate_count_min": min(counts), "candidate_count_max": max(counts),
            "all_queries_full_candidate_topk_exact": True,
            "max_metric_absolute_error": max_error, "max_saved_score_absolute_error": score_error,
            "max_saved_component_absolute_error": component_error, "aggregation_values_checked": aggregated,
            "metric_query_counts": report["metric_query_counts"], "exact_reports": False,
            "scope": "Every eligible candidate identity; original frozen ten specs; independent filtered-pool fusion/minmax and SHA TopK; scalar coverage_v1 and all query/class/macro/group counts.",
            "ranking_sha256": final_hashes["rankings.csv"], "files": final_hashes}
