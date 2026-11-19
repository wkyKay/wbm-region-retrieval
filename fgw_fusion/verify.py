"""Independent, read-only F/G/WBBS score and metric verification.

No production descriptor/scoring function is called. The original independent
label-metric implementation is reused; its formulas do not call either evaluator.
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


def independent_saved_score(rows, spec):
    """Explicit fixed formulas; never import the production scoring module."""
    common = {"wbbs_power": 1., "angular_gamma": 0., "local_mode": "optimal"}
    allowed = [dict(common, local_weight=lw, global_weight=gw)
               for lw, gw in ((.5, .5), (0., 1.), (1., 0.))]
    if not isinstance(spec, dict) or spec not in allowed:
        raise ValueError("Unknown fixed F/G/WBBS score specification")
    if any(isinstance(spec[k], bool) for k in ("local_weight", "global_weight", "wbbs_power", "angular_gamma")):
        raise ValueError("Boolean score parameters are invalid")
    try:
        values = {key: np.array([float(row[key]) for row in rows], dtype=np.float64)
                  for key in ("optimal_f_score", "global_score", "wbbs_score")}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Missing or invalid F/G/WBBS saved components") from error
    if any(not np.isfinite(value).all() or np.any((value < 0) | (value > 1))
           for value in values.values()):
        raise ValueError("Saved F/G/WBBS component lies outside finite [0,1]")
    f, g, w = (values[key] for key in ("optimal_f_score", "global_score", "wbbs_score"))
    if spec["local_weight"] == 0.:
        return g * w
    if spec["global_weight"] == 0.:
        return f * w
    return (.5 * f + .5 * g) * w


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


def verify_evaluation(folder, pool, bundle, ks, signature, tie_seed=2026):
    folder = Path(folder)
    marker = _json(folder / "completed.json")
    if marker.get("signature") != signature:
        raise ValueError("Completed signature mismatch")
    for name, expected in marker["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or _sha(folder / relative) != expected:
            raise ValueError(f"Completed file hash mismatch: {name}")
    manifest = _json(folder / "evaluation_manifest.json")
    if manifest.get("signature") != signature or manifest["split"] != pool.split_name or manifest["ks"] != list(ks):
        raise ValueError("Evaluation manifest differs from verification inputs")
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
    seen, count, maximum_error, score_error = set(), 0, 0., 0.
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
            if not np.array_equal(independent_order(query, ranked, scores, tie_seed), np.arange(len(rows))):
                raise ValueError("Saved score order or SHA ties differ")
            formula = independent_saved_score(rows, manifest["spec"])
            np.testing.assert_allclose(scores, formula, atol=FORMULA_ATOL, rtol=FORMULA_ATOL)
            score_error = max(score_error, float(np.max(np.abs(scores-formula))))
            values = independent_query_metrics(labels[query], labels[candidates], labels[ranked], ks)
            label_count, cls = int(labels[query].sum()), str(int(classes[query]))
            groups = ["all", "class/"+cls, "by_label_count/"+str(label_count),
                      "single_label" if label_count == 1 else "mixed_label"]
            for family, metrics in values.items():
                saved_row = expected[family][query]
                if (int(saved_row["class_id"]) != int(classes[query]) or
                    int(saved_row["label_count"]) != label_count or
                    int(saved_row["pool_size"]) != len(candidates) or
                    int(saved_row["returned"]) != len(ranked)):
                    raise ValueError("Per-query metadata differs from frozen identities/labels")
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
        means = {g: {k: math.fsum(v)/len(v) for k,v in metrics.items()} for g,metrics in sums[family].items()}
        classes_present = [g for g in means if g.startswith("class/")]
        means["class_macro"] = {metric: math.fsum(means[g][metric] for g in classes_present
                              if metric in means[g]) / sum(metric in means[g] for g in classes_present)
                              for metric in means["all"]}
        saved = {"all": report["metrics"], "class_macro": report["class_macro_metrics"]}
        for name, group in report["groups"].items():
            if name == "by_label_count":
                saved.update({"by_label_count/"+str(k): v["metrics"] for k,v in group.items()})
            else:
                saved[name] = group["metrics"]
        with (folder/classfile).open(newline="", encoding="utf-8") as handle:
            seen_classes = set()
            for row in csv.DictReader(handle):
                key = "class/"+str(int(row["class_id"]))
                if key in seen_classes:
                    raise ValueError("Duplicate class report")
                seen_classes.add(key)
                saved[key] = {k:v for k,v in row.items() if "@" in k}
        if set(means) != set(saved):
            raise ValueError("Class/group aggregation identities differ")
        for group, metrics in means.items():
            for metric, value in metrics.items():
                maximum_error = max(maximum_error, _assert_value(saved[group][metric], value, (family, group, metric)))
                aggregation_count += 1
    return {"status": "verified", "split": pool.split_name, "variant": manifest["variant"],
            "num_queries": len(seen), "ranking_rows": count,
            "max_metric_absolute_error": maximum_error, "max_saved_score_absolute_error": score_error,
            "aggregation_values_checked": aggregation_count,
            "scope": "all saved TopK IDs/SHA ties/independent F/G/WBBS score formulas; full-gallery label denominators; all query/class/macro/count-group metrics. Unsaved candidate ordering is separately sampled.",
            "ranking_sha256": _sha(folder/"rankings.csv"), "completion_sha256": _sha(folder/"completed.json")}
