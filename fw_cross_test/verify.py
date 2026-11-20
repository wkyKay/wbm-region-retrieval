"""Independent full-pool verification of user-specified F and F*WBBS.

Reuse the existing independent scalar coverage definitions and tie hashing.
No production scorer or metric evaluator is called. The comparison preserves
all six metrics and candidate-denominator rules from the preceding cross test.
"""
from collections import defaultdict
import csv
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path

import numpy as np

from fgw_cross_test.verify import (
    ATOL, PROTOCOL, RULE, FORMAL_KS, REPORT_FILES, STEMS,
    _json, _sha, _digest, independent_order, independent_coverage,
    _assert_value, _records, _metric_name,
)

# Keep these independently specified; never import the production spec/scorer.
_SPECS = {"F": {"family": "local_f", "formula": "F"},
          "FW": {"family": "product_fw", "formula": "F*WBBS"}}


def independent_score(components, spec):
    if spec not in _SPECS.values():
        raise ValueError("Only the user-specified F or literal F*WBBS formula is allowed")
    if not isinstance(components, dict) or set(components) != {"F", "G", "WBBS"}:
        raise ValueError("Expected exactly F/G/WBBS source components")
    arrays = [np.asarray(components[name]) for name in ("F", "G", "WBBS")]
    if any(np.iscomplexobj(a) for a in arrays):
        raise ValueError("Source scores must be real")
    arrays = [np.asarray(a, dtype=np.float64) for a in arrays]
    if (any(a.ndim != 1 or not len(a) or a.shape != arrays[0].shape for a in arrays)
            or any(not np.isfinite(a).all() or np.any((a < 0) | (a > 1)) for a in arrays)):
        raise ValueError("Invalid full eligible source scores")
    return arrays[0].copy() if spec == _SPECS["F"] else arrays[0] * arrays[2]


def _verify_variants(manifest):
    source = manifest.get("frozen_variants", {})
    if set(source) != {"path", "sha256"} or _sha(source["path"]) != source["sha256"]:
        raise ValueError("User-specified variant source hash changed")
    frozen = _json(source["path"])
    if (frozen.get("signature") != _digest({k: v for k, v in frozen.items() if k != "signature"})
            or frozen.get("variants") != ["F", "FW"] or frozen.get("specs") != _SPECS
            or manifest.get("variant") not in _SPECS
            or manifest.get("spec") != _SPECS[manifest["variant"]]
            or frozen.get("primary_variant") is not None
            or frozen.get("selection_rule") != "user_specified_posthoc_components"
            or frozen.get("test_parameter_search") is not False
            or frozen.get("settings") != manifest.get("settings")
            or frozen.get("settings", {}).get("ks") != manifest.get("ks")
            or frozen.get("settings", {}).get("tie_seed") != manifest.get("tie_seed")
            or manifest.get("test_parameter_search") is not False
            or manifest.get("posthoc_research") is not True
            or manifest.get("training_required") is not False):
        raise ValueError("Evaluation differs from the two user-specified component formulas/settings")
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
    selection_source = _verify_variants(manifest)
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
            "scope": "Every eligible candidate identity; user-specified F and literal F*WBBS; independent filtered-pool scoring and SHA TopK; scalar coverage_v1 and all query/class/macro/group counts.",
            "ranking_sha256": final_hashes["rankings.csv"], "files": final_hashes}
