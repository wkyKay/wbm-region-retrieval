"""Pre-seal full-pool replay and independent ranking/metric verification.

The provider replays production scores. SHA ordering, label denominators and
all metric aggregations are independent; regional formula/greedy audits are a
separate pipeline check and are deliberately not claimed by this module.
"""
from __future__ import annotations

import csv
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path

import numpy as np

from b1_structure_fusion.verify import independent_query_metrics
from fgw_grid_search.verify import independent_order


ATOL = 1e-12
REPORT_FILES = ("evaluation_manifest.json", "rankings.csv", "metrics.json",
                "per_query.csv", "per_class.csv", "coverage_metrics.json",
                "coverage_per_query.csv", "coverage_per_class.csv")


def _json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids(values, name):
    array = np.asarray(values)
    if (array.ndim != 1 or array.dtype.kind not in "iu" or not len(array) or
            np.any(array < 0) or len(np.unique(array)) != len(array)):
        raise ValueError(f"Invalid unique integer {name}")
    return array.astype(np.int64, copy=False)


def _records(path, identity):
    rows = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = int(row[identity])
            if key in rows:
                raise ValueError(f"Duplicate {identity} in {path.name}")
            rows[key] = row
    return rows


def _metric_name(field):
    if field.startswith("AP@"):
        return "mAP" + field[2:]
    if field.startswith("FullCoverageAP@"):
        return "FullCoverage_mAP" + field[len("FullCoverageAP"):]
    return field


def _label_signature(label, family):
    return ("|".join(str(i) for i, value in enumerate(label) if value) if family == "exact"
            else "".join(str(int(value)) for value in label))


def _assert_value(saved, actual, context):
    saved = None if saved in (None, "") else float(saved)
    if (saved is None) != (actual is None):
        raise ValueError(f"Undefined metric mismatch: {context}")
    if actual is None:
        return 0.
    error = abs(saved - actual)
    if not np.isfinite(saved) or error > ATOL:
        raise ValueError(f"Metric mismatch: {context}: {saved} != {actual}")
    return error


def _summary(rows, metric_names):
    metrics, counts = {}, {}
    for key in metric_names:
        values = [row["metrics"][key] for row in rows if row["metrics"][key] is not None]
        counts[key] = len(values)
        metrics[key] = math.fsum(values) / len(values) if values else None
    return {"num_queries": len(rows), "num_classes": len({row["class_id"] for row in rows}),
            "without": sum(row["denominator"] == 0 for row in rows),
            "metrics": metrics, "metric_query_counts": counts}


def _check_summary(saved, actual, zero_field, context):
    for key in ("num_queries", "num_classes", "metric_query_counts"):
        if saved.get(key) != actual[key]:
            raise ValueError(f"Aggregate counts differ: {context}/{key}")
    if saved.get(zero_field) != actual["without"]:
        raise ValueError(f"Zero-denominator count differs: {context}")
    if set(saved.get("metrics", {})) != set(actual["metrics"]):
        raise ValueError(f"Aggregate metric keys differ: {context}")
    return max((_assert_value(saved["metrics"][key], value, (context, key))
                for key, value in actual["metrics"].items()), default=0.)


def verify_evaluation(folder, pool, bundle, ks, signature, provider, tie_seed=2026):
    """Verify an unsealed evaluation; provider(query) returns all IDs and scores."""
    folder = Path(folder)
    ks = list(ks)
    if (not ks or any(isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1 for k in ks)
            or ks != sorted(set(ks)) or 10 not in ks):
        raise ValueError("Verification requires sorted unique positive Ks including 10")
    if isinstance(tie_seed, bool) or not isinstance(tie_seed, (int, np.integer)) or tie_seed < 0:
        raise ValueError("Invalid tie seed")
    manifest = _json(folder / "evaluation_manifest.json")
    if (manifest.get("signature") != signature or manifest.get("split") != pool.split_name or
            manifest.get("ks") != ks or manifest.get("tie_seed") != tie_seed or
            not isinstance(manifest.get("variant"), str) or not manifest["variant"] or
            not isinstance(manifest.get("spec"), dict) or not manifest["spec"]):
        raise ValueError("Evaluation manifest differs from verification inputs")
    query_array, gallery_array = _ids(list(pool), "query IDs"), _ids(pool.gallery, "gallery IDs")
    query_ids, gallery = set(query_array.tolist()), set(gallery_array.tolist())
    heldout = _ids(bundle.valid_ids if pool.split_name == "valid" else bundle.reserved_ids,
                   "held-out original IDs")
    if (not query_ids <= gallery or pool.candidates_per_query != len(gallery) - 1
            or max(ks) > pool.candidates_per_query or gallery != set(heldout.tolist())):
        raise ValueError("Invalid complete-gallery protocol or K")
    labels, classes = np.asarray(bundle.labels), np.asarray(bundle.class_ids)
    if (labels.ndim != 2 or classes.ndim != 1 or len(classes) != len(labels) or
            classes.dtype.kind not in "iu" or max(gallery) >= len(labels) or
            not labels.shape[1] or not np.isin(labels[gallery_array], [0, 1]).all() or
            np.any(labels[gallery_array].sum(1) == 0) or np.any(classes[gallery_array] < 0)):
        raise ValueError("Invalid original-ID label/class arrays")
    labels = labels.astype(np.int64, copy=False)
    mapping, inverse = {}, {}
    for sample in gallery_array:
        cls, label = int(classes[sample]), tuple(labels[sample].tolist())
        if (cls in mapping and mapping[cls] != label) or (label in inverse and inverse[label] != cls):
            raise ValueError("Class IDs differ from complete label combinations")
        mapping[cls], inverse[label] = label, cls
    saved_query = {family: _records(folder / filename, "query_id") for family, filename in
                   (("exact", "per_query.csv"), ("coverage", "coverage_per_query.csv"))}
    if any(set(rows) != query_ids for rows in saved_query.values()):
        raise ValueError("Per-query report must cover each frozen query exactly once")
    files = list(REPORT_FILES)
    if (folder / "run_identity.json").exists():
        files.append("run_identity.json")
    initial_hashes = {name: _sha(folder / name) for name in files}
    recomputed = {"exact": [], "coverage": []}
    seen, ranking_rows, full_count, top10_hits = set(), 0, 0, 0
    metric_error, score_error = 0., 0.
    with (folder / "rankings.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["query_id", "rank", "candidate_id", "similarity_score"]:
            raise ValueError("Unexpected ranking columns")
        for query, iterator in groupby(reader, key=lambda row: int(row["query_id"])):
            if query not in query_ids or query in seen:
                raise ValueError("Missing, repeated, or unexpected ranking query")
            seen.add(query)
            rows = list(iterator)
            ranked = _ids([int(row["candidate_id"]) for row in rows], "ranked IDs")
            candidates = _ids(pool[query], "pool candidate IDs")
            if set(candidates.tolist()) != gallery - {query}:
                raise ValueError("Pool candidates do not cover the gallery except self")
            if (len(rows) != max(ks) or [int(row["rank"]) for row in rows] != list(range(1, max(ks)+1))
                    or not set(ranked.tolist()) <= set(candidates.tolist())):
                raise ValueError("Invalid saved TopK identities or ranks")
            returned = provider(query)
            if not isinstance(returned, (tuple, list)) or len(returned) != 2:
                raise ValueError("Provider must return full candidate IDs and aligned scores")
            replay_ids = _ids(returned[0], "provider candidate IDs")
            replay_scores = np.asarray(returned[1], dtype=np.float64)
            if (set(replay_ids.tolist()) != gallery - {query} or replay_scores.shape != replay_ids.shape
                    or not np.isfinite(replay_scores).all()):
                raise ValueError("Provider must score the complete gallery except self")
            order = independent_order(query, replay_ids, replay_scores, tie_seed)[:max(ks)]
            if not np.array_equal(ranked, replay_ids[order]):
                raise ValueError("Saved TopK differs from independent full-candidate ordering")
            saved_scores = np.asarray([float(row["similarity_score"]) for row in rows])
            if not np.isfinite(saved_scores).all() or not np.allclose(saved_scores, replay_scores[order], rtol=0, atol=ATOL):
                raise ValueError("Saved score differs from full-candidate production replay")
            score_error = max(score_error, float(np.abs(saved_scores - replay_scores[order]).max()))
            values = independent_query_metrics(labels[query], labels[candidates], labels[ranked], ks)
            top10_hits += int(np.all(labels[ranked[:10]] == labels[query], axis=1).sum())
            for family, metrics in values.items():
                saved = saved_query[family][query]
                count, cls = int(labels[query].sum()), int(classes[query])
                if (int(saved["class_id"]) != cls or int(saved["label_count"]) != count or
                        int(saved["pool_size"]) != len(candidates) or int(saved["returned"]) != len(ranked) or
                        saved["label_signature"] != _label_signature(labels[query], family)):
                    raise ValueError("Per-query identity/label metadata differs")
                relevant = (np.all(labels[candidates] == labels[query], axis=1) if family == "exact" else
                            np.all(labels[candidates][:, labels[query].astype(bool)] == 1, axis=1))
                denominator = int(relevant.sum())
                field = "relevant_in_pool" if family == "exact" else "full_coverage_in_pool"
                if int(saved[field]) != denominator:
                    raise ValueError("Per-query denominator differs from complete-gallery labels")
                if {key for key in saved if "@" in key} != set(metrics):
                    raise ValueError("Per-query metric keys differ")
                for key, value in metrics.items():
                    metric_error = max(metric_error, _assert_value(saved[key], value, (query, family, key)))
                recomputed[family].append({"query_id": query, "class_id": cls, "label_count": count,
                    "label_signature": _label_signature(labels[query], family), "denominator": denominator,
                    "metrics": {_metric_name(key): value for key, value in metrics.items()}})
            ranking_rows += len(rows)
            full_count += len(candidates)
    if seen != query_ids:
        raise ValueError("Ranking misses frozen queries")

    aggregation_count = 0
    for family, filename, classfile in (("exact", "metrics.json", "per_class.csv"),
                                        ("coverage", "coverage_metrics.json", "coverage_per_class.csv")):
        report = _json(folder / filename)
        if (report.get("split") != pool.split_name or report.get("ks") != ks or
                report.get("test_evaluated") != (pool.split_name == "test") or
                report.get("metric_version") != family + "_v1" or
                report.get("metadata", {}).get("signature") != signature or
                report.get("retrieval_protocol_signature", report.get("metadata", {}).get("retrieval_protocol_signature")) != pool.protocol_signature):
            raise ValueError("Aggregate report protocol/signature differs")
        rows = recomputed[family]
        names = set(rows[0]["metrics"])
        zero_field = "queries_without_relevant_candidates" if family == "exact" else "queries_without_full_coverage_candidates"
        metric_error = max(metric_error, _check_summary(report, _summary(rows, names), zero_field, family+"/all"))
        groups = {"single_label": [r for r in rows if r["label_count"] == 1],
                  "mixed_label": [r for r in rows if r["label_count"] > 1]}
        if set(report.get("groups", {})) != {"single_label", "mixed_label", "by_label_count"}:
            raise ValueError("Group identities differ")
        by_count = {str(count): [r for r in rows if r["label_count"] == count]
                    for count in sorted({r["label_count"] for r in rows})}
        if set(report["groups"]["by_label_count"]) != set(by_count):
            raise ValueError("Label-count group identities differ")
        for key, members in groups.items():
            metric_error = max(metric_error, _check_summary(report["groups"][key], _summary(members, names), zero_field, (family, key)))
        for key, members in by_count.items():
            metric_error = max(metric_error, _check_summary(report["groups"]["by_label_count"][key], _summary(members, names), zero_field, (family, "count", key)))
        by_class = {cls: [r for r in rows if r["class_id"] == cls] for cls in sorted({r["class_id"] for r in rows})}
        saved_classes = _records(folder / classfile, "class_id")
        if set(saved_classes) != set(by_class):
            raise ValueError("Per-class identities differ")
        class_summaries = []
        for cls, members in by_class.items():
            saved, actual = saved_classes[cls], _summary(members, names)
            if (int(saved["num_queries"]) != len(members) or
                    int(saved["label_count"]) != members[0]["label_count"] or
                    saved["label_signature"] != members[0]["label_signature"] or
                    int(saved[zero_field]) != actual["without"] or {key for key in saved if "@" in key} != names):
                raise ValueError("Per-class counts/labels/metric keys differ")
            for key, value in actual["metrics"].items():
                metric_error = max(metric_error, _assert_value(saved[key], value, (family, cls, key)))
            class_summaries.append(actual)
        macro, macro_counts = {}, {}
        for key in names:
            values = [row["metrics"][key] for row in class_summaries if row["metrics"][key] is not None]
            macro[key], macro_counts[key] = (math.fsum(values)/len(values) if values else None), len(values)
        if set(report.get("class_macro_metrics", {})) != names or report.get("class_macro_metric_class_counts") != macro_counts:
            raise ValueError("Class macro metric keys/counts differ")
        for key, value in macro.items():
            metric_error = max(metric_error, _assert_value(report["class_macro_metrics"][key], value, (family, "macro", key)))
        aggregation_count += len(names) * (2 + len(groups) + len(by_count) + len(by_class))
    final_hashes = {name: _sha(folder / name) for name in files}
    if final_hashes != initial_hashes:
        raise ValueError("Evaluation reports changed during verification")
    return {"status": "verified", "signature": signature, "split": pool.split_name,
            "variant": manifest["variant"], "ks": ks, "tie_seed": int(tie_seed),
            "num_queries": len(seen), "ranking_rows": ranking_rows,
            "full_candidates_checked": full_count, "top10_hits": top10_hits,
            "all_queries_full_candidate_topk_exact": True,
            "max_metric_absolute_error": metric_error, "max_saved_score_absolute_error": score_error,
            "aggregation_values_checked": aggregation_count,
            "scope": "Every query: production full-candidate score replay; independent SHA TopK ordering, full-gallery label denominators and every K/query/class/macro/count-group metric and count. Independent regional formulas and greedy matching are audited separately.",
            "ranking_sha256": final_hashes["rankings.csv"], "files": final_hashes}
