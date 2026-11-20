"""Independent, query-batched verification of the six fixed cross-combination runs.

Each complete eligible component vector and SHA tie vector is read once per
query, shared across six independently scored rankings. No production scorer,
ranking function or metric evaluator is called.
"""
from collections import defaultdict
from contextlib import ExitStack
import csv
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import re

import numpy as np

from fgw_cross_test.verify import (
    ATOL, PROTOCOL, RULE, FORMAL_KS, REPORT_FILES, STEMS,
    _json, _sha, _digest, independent_coverage, _assert_value, _records, _metric_name,
)

# Independent literal definitions; intentionally do not import state/scoring.
_SPECS = {
    "GJ_ADD": {"formula": "0.5*G+0.5*J"},
    "FJ_ADD": {"formula": "0.5*F+0.5*J"},
    "F_ONLY": {"formula": "F"}, "G_ONLY": {"formula": "G"},
    "J_ONLY": {"formula": "J"}, "WBBS": {"formula": "WBBS"},
}
RANKING_FIELDS = ("query_id", "rank", "candidate_id", "similarity_score",
                  "optimal_f_score", "global_score", "joint_distance", "wbbs_score")
_COMPONENT_FIELDS = {"optimal_f_score": "F", "global_score": "G",
                     "joint_distance": "joint_distance", "wbbs_score": "WBBS"}
_MANIFEST_KEYS = ("signature", "context_signature", "spec", "split", "variant", "ks",
                  "tie_seed", "num_queries", "gallery_count", "candidate_rule",
                  "derived_protocol", "retrieval_protocol_signature")


def _components(values, count=None):
    if not isinstance(values, dict) or set(values) != set(_COMPONENT_FIELDS.values()):
        raise ValueError("Expected exactly F/G/joint_distance/WBBS components")
    arrays = {name: np.asarray(value) for name, value in values.items()}
    if any(np.iscomplexobj(value) for value in arrays.values()):
        raise ValueError("Source components must be real")
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in arrays.items()}
    shape = arrays["F"].shape
    if (len(shape) != 1 or not shape[0] or count is not None and shape != (count,)
            or any(value.shape != shape or not np.isfinite(value).all()
                   or np.any((value < 0) | (value > 1)) for value in arrays.values())):
        raise ValueError("Invalid complete eligible component vectors")
    return arrays


def _score_all(values):
    joint = np.exp(-2.0 * values["joint_distance"])
    return {"GJ_ADD": .5 * values["G"] + .5 * joint,
            "FJ_ADD": .5 * values["F"] + .5 * joint,
            "F_ONLY": values["F"], "G_ONLY": values["G"],
            "J_ONLY": joint, "WBBS": values["WBBS"]}


def independent_score(components, spec):
    matches = [name for name, expected in _SPECS.items() if spec == expected]
    if len(matches) != 1:
        raise ValueError("Only the six fixed literal formulas are allowed")
    return _score_all(_components(components))[matches[0]].copy()


def _tie_keys(query, candidates, seed):
    return np.asarray([int.from_bytes(hashlib.sha256(
        f"exact_v1:tie:{int(seed)}:{int(query)}:{int(candidate)}".encode("ascii")
    ).digest()[:8], "big") for candidate in candidates], dtype=np.uint64)


def _check_aggregates(folder, perquery, labels, ks):
    report = _json(folder / "coverage_metrics.json")
    names = {f"{stem}@{k}" for stem in STEMS for k in ks}
    max_error, aggregated = 0., 0

    def check_summary(saved, members, context):
        nonlocal max_error, aggregated
        if (saved.get("num_queries") != len(members)
                or saved.get("num_classes") != len({perquery[q]["class"] for q in members})
                or saved.get("queries_without_full_coverage_candidates") != sum(perquery[q]["relevant"] == 0 for q in members)
                or set(saved.get("metrics", {})) != names
                or set(saved.get("metric_query_counts", {})) != names):
            raise ValueError("Aggregate identities/counts/metric keys differ: " + context)
        for metric in names:
            values = [perquery[q]["metrics"][metric] for q in members if perquery[q]["metrics"][metric] is not None]
            actual = math.fsum(values) / len(values) if values else None
            max_error = max(max_error, _assert_value(saved["metrics"][metric], actual, (context, metric)))
            if saved["metric_query_counts"][metric] != len(values):
                raise ValueError("Metric effective query count differs: " + context)
            aggregated += 1

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
    for count in label_counts:
        check_summary(report["groups"]["by_label_count"][count],
                      [q for q in all_queries if perquery[q]["label_count"] == int(count)], "label_count/" + count)
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
                or saved["label_signature"] != "".join(map(str, labels[members[0]].astype(int).tolist()))
                or {key for key in saved if "@" in key} != names):
            raise ValueError("Per-class labels/counts/metric names differ")
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
    return max_error, aggregated, report["metric_query_counts"]


def verify_evaluations(folders, pool, bundle, ks, signatures, score_components, source_binding, tie_seed=2026):
    """Return shared audit and six per-variant receipts without writing anything."""
    if (not isinstance(folders, dict) or set(folders) != set(_SPECS)
            or not isinstance(signatures, dict) or set(signatures) != set(_SPECS)
            or any(not isinstance(value, str) or not value for value in signatures.values())):
        raise ValueError("Exactly the six fixed evaluations and signatures are required")
    if (not isinstance(source_binding, dict) or not source_binding
            or any(not isinstance(path, str) or not Path(path).is_absolute()
                   or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha)
                   for path, sha in source_binding.items())):
        raise ValueError("Source binding must contain absolute paths and SHA-256 hashes")
    source_digest = _digest(source_binding)
    if (not ks or list(ks) != sorted(set(ks)) or any(type(k) is not int or k < 1 for k in ks)
            or type(tie_seed) is not int or tie_seed != 2026):
        raise ValueError("Invalid cutoffs or frozen tie seed")
    folders = {name: Path(path).resolve() for name, path in folders.items()}
    if len(set(folders.values())) != len(folders):
        raise ValueError("Each variant requires its own report directory")
    queries, gallery = np.asarray(list(pool)), np.asarray(pool.gallery)
    labels, classes = np.asarray(bundle.labels), np.asarray(bundle.class_ids)
    if (queries.ndim != 1 or gallery.ndim != 1 or queries.dtype.kind not in "iu"
            or gallery.dtype.kind not in "iu" or not len(queries) or not len(gallery)
            or len(np.unique(queries)) != len(queries) or len(np.unique(gallery)) != len(gallery)
            or not np.isin(queries, gallery).all() or np.any(gallery < 0)
            or labels.ndim != 2 or labels.shape[1] != 8 or not np.isin(labels, (0, 1)).all()
            or max(gallery) >= len(labels) or classes.shape != (len(labels),) or classes.dtype.kind not in "iu"
            or np.any(labels[gallery].sum(axis=1) == 0)):
        raise ValueError("Invalid query/gallery/raw-label identities")
    label_classes = {tuple(labels[q]): int(classes[q]) for q in gallery}
    if (len(set(label_classes.values())) != len(label_classes)
            or any(label_classes[tuple(labels[q])] != int(classes[q]) for q in gallery)):
        raise ValueError("Class IDs do not identify unique complete raw label combinations")
    if (pool.split_name != "test" or pool.metadata.get("protocol_version") != PROTOCOL
            or pool.metadata.get("candidate_rule") != RULE
            or hashlib.sha256(json.dumps(pool.metadata, sort_keys=True).encode()).hexdigest() != pool.protocol_signature):
        raise ValueError("Derived cross-combination protocol differs")
    initial_hashes, manifests, expected, allmetrics = {}, {}, {}, {}
    errors = {name: {"metric": 0., "score": 0., "component": 0.} for name in _SPECS}
    for name, folder in folders.items():
        if any((folder / filename).exists() for filename in ("metrics.json", "per_query.csv", "per_class.csv")):
            raise ValueError("Cross-combination evaluation must not emit exact reports")
        initial_hashes[name] = {filename: _sha(folder / filename) for filename in REPORT_FILES}
        manifest, identity = _json(folder / "evaluation_manifest.json"), _json(folder / "run_identity.json")
        if (manifest.get("signature") != signatures[name] or manifest.get("spec") != _SPECS[name]
                or manifest.get("split") != "test" or manifest.get("variant") != name
                or manifest.get("ks") != list(ks) or manifest.get("tie_seed") != tie_seed
                or manifest.get("num_queries") != len(queries) or manifest.get("gallery_count") != len(gallery)
                or manifest.get("candidate_rule") != RULE or manifest.get("derived_protocol") != pool.metadata
                or manifest.get("retrieval_protocol_signature") != pool.protocol_signature
                or not isinstance(manifest.get("context_signature"), str) or not manifest["context_signature"]):
            raise ValueError("Evaluation manifest/formula/candidate protocol differs: " + name)
        if (identity.get("evaluation_signature") != signatures[name]
                or any(manifest.get(k) != identity.get(k) for k in ("run_id", "started_at", "variant", "split"))):
            raise ValueError("Evaluation time identity differs: " + name)
        report = _json(folder / "coverage_metrics.json")
        if (report.get("split") != "test" or report.get("test_evaluated") is not True
                or report.get("ks") != list(ks) or report.get("metric_version") != "coverage_v1"
                or any(report.get("metadata", {}).get(key) != manifest[key] for key in _MANIFEST_KEYS)):
            raise ValueError("Coverage report metadata/signature differs: " + name)
        manifests[name] = manifest
        expected[name] = _records(folder / "coverage_per_query.csv", "query_id")
        if set(expected[name]) != set(map(int, queries)):
            raise ValueError("Per-query report misses/duplicates frozen queries: " + name)
        allmetrics[name] = {}
    contexts = {manifest["context_signature"] for manifest in manifests.values()}
    formals = {manifest.get("formal_protocol", True) for manifest in manifests.values()}
    if len(contexts) != 1 or len(formals) != 1:
        raise ValueError("The six evaluations must share one context and formal scope")
    formal = next(iter(formals))
    if formal and (len(queries) != 7199 or len(gallery) != 7199
                   or set(map(int, queries)) != set(map(int, gallery)) or list(ks) != FORMAL_KS):
        raise ValueError("Formal evaluation requires 7199 queries/gallery and seven frozen cutoffs")
    counts, candidate_digest = [], hashlib.sha256()
    with ExitStack() as stack:
        streams = {}
        for name, folder in folders.items():
            reader = csv.DictReader(stack.enter_context((folder / "rankings.csv").open(newline="", encoding="utf-8")))
            if reader.fieldnames != list(RANKING_FIELDS):
                raise ValueError("Ranking CSV must contain exactly the eight frozen columns")
            streams[name] = iter(groupby(reader, key=lambda row: int(row["query_id"])))
        for query in map(int, queries):
            candidates = np.asarray(pool[query])
            eligible = gallery[(gallery != query) & np.any(labels[gallery] != labels[query], axis=1)]
            if not np.array_equal(candidates, eligible) or len(candidates) < max(ks):
                raise ValueError("Candidate identities differ from independent full label exclusion")
            counts.append(len(candidates))
            candidate_digest.update(json.dumps([query, candidates.tolist()], separators=(",", ":")).encode("ascii") + b"\n")
            values = _components(score_components(query, candidates), len(candidates))
            scores = _score_all(values)
            ties = _tie_keys(query, candidates, tie_seed)
            for name in _SPECS:
                group = next(streams[name], None)
                if group is None or group[0] != query:
                    raise ValueError("Ranking query order/identity differs: " + name)
                rows = list(group[1])
                ranked = np.asarray([int(row["candidate_id"]) for row in rows], dtype=np.int64)
                if (len(rows) != max(ks) or [int(row["rank"]) for row in rows] != list(range(1, max(ks) + 1))
                        or len(np.unique(ranked)) != len(ranked) or not np.isin(ranked, candidates).all()):
                    raise ValueError("Invalid saved TopK identities/ranks: " + name)
                order = np.lexsort((candidates, ties, -scores[name]))[:max(ks)]
                if not np.array_equal(ranked, candidates[order]):
                    raise ValueError("Saved TopK differs from independent full-candidate ordering: " + name)
                for field, actual in [("similarity_score", scores[name][order]),
                                      *((field, values[key][order]) for field, key in _COMPONENT_FIELDS.items())]:
                    saved = np.asarray([float(row[field]) for row in rows])
                    if not np.isfinite(saved).all() or not np.allclose(saved, actual, atol=ATOL, rtol=0):
                        raise ValueError("Saved score formula/component differs: " + name + "/" + field)
                    key = "score" if field == "similarity_score" else "component"
                    errors[name][key] = max(errors[name][key], float(np.max(np.abs(saved - actual))))
                metrics, relevant = independent_coverage(labels[query], labels[candidates], labels[ranked], ks)
                label_count, cls = int(labels[query].sum()), int(classes[query])
                row = expected[name][query]
                metadata = {"class_id": cls, "label_count": label_count, "pool_size": len(candidates),
                            "returned": len(ranked), "full_coverage_in_pool": relevant}
                if (any(int(row[key]) != value for key, value in metadata.items())
                        or row["label_signature"] != "".join(map(str, labels[query].astype(int).tolist()))):
                    raise ValueError("Per-query labels/counts/full-pool denominator differs: " + name)
                if {key for key in row if "@" in key} != set(metrics):
                    raise ValueError("Per-query metric names changed: " + name)
                for metric, value in metrics.items():
                    errors[name]["metric"] = max(errors[name]["metric"], _assert_value(row[metric], value, (name, query, metric)))
                allmetrics[name][query] = {"class": cls, "label_count": label_count, "relevant": relevant,
                                           "metrics": {_metric_name(key): value for key, value in metrics.items()}}
        if any(next(stream, None) is not None for stream in streams.values()):
            raise ValueError("Ranking contains extra or duplicate query groups")
    if (candidate_digest.hexdigest() != pool.metadata.get("candidate_identity_sha256")
            or min(counts) != pool.metadata.get("candidate_count_min")
            or max(counts) != pool.metadata.get("candidate_count_max")
            or pool.candidates_per_query != min(counts)):
        raise ValueError("Candidate identity digest/count range differs")
    shared = {"status": "verified", "split": "test", "context_signature": next(iter(contexts)),
              "ks": list(ks), "tie_seed": tie_seed, "variants": list(_SPECS),
              "num_queries": len(queries), "components_calls": len(queries), "tie_key_computations": len(queries),
              "full_candidates_checked": sum(counts), "candidate_identity_sha256": candidate_digest.hexdigest(),
              "candidate_count_min": min(counts), "candidate_count_max": max(counts),
              "source_binding_sha256": source_digest, "source_file_count": len(source_binding)}
    receipts = {}
    for name, folder in folders.items():
        error, aggregated, metric_counts = _check_aggregates(folder, allmetrics[name], labels, ks)
        final_hashes = {filename: _sha(folder / filename) for filename in REPORT_FILES}
        if final_hashes != initial_hashes[name]:
            raise ValueError("Reports changed during verification: " + name)
        receipts[name] = {**shared, "signature": signatures[name], "variant": name, "spec": _SPECS[name],
                          "ranking_rows": len(queries) * max(ks), "all_queries_full_candidate_topk_exact": True,
                          "independent_formulas": True, "max_metric_absolute_error": max(error, errors[name]["metric"]),
                          "max_saved_score_absolute_error": errors[name]["score"],
                          "max_saved_component_absolute_error": errors[name]["component"],
                          "aggregation_values_checked": aggregated, "metric_query_counts": metric_counts,
                          "exact_reports": False, "ranking_sha256": final_hashes["rankings.csv"], "files": final_hashes,
                          "scope": "All eligible candidates, independent six formulas and SHA ordering; all coverage_v1 query/class/macro/group values and effective denominator counts."}
    if _digest(source_binding) != source_digest:
        raise ValueError("Source binding changed during verification")
    return {"status": "verified", "shared": shared, "variants": receipts}
