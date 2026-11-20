"""Independent G/J fusion, full-candidate rankings and dual-metric verification.

The source callback supplies frozen G and Hellinger-distance components. This
module deliberately never calls the production fusion scorer or metric
functions. It reuses independent aggregation helpers from an earlier verifier.
"""
from __future__ import annotations

import csv
from itertools import groupby
import math
from pathlib import Path

import numpy as np

from b1_ablation.verify import independent_order
from b1_structure_fusion.verify import independent_query_metrics
from f_region_structure.verify import (REPORT_FILES, _json, _sha, _ids, _records,
    _metric_name, _label_signature, _assert_value, _summary, _check_summary)

ATOL = 1e-12
RANKING_FIELDS = ["query_id", "rank", "candidate_id", "similarity_score",
                  "global_score", "joint_distance"]


def _unit_vector(values, name):
    vector = np.asarray(values, dtype=np.float64)
    if (vector.ndim != 1 or not np.isfinite(vector).all()
            or np.any(vector < 0) or np.any(vector > 1)):
        raise ValueError("Expected a finite unit-interval vector: " + name)
    return vector


def independent_score(global_scores, joint_distances, spec):
    """Explicit frozen formulas without importing gj_fusion.scoring."""
    g = _unit_vector(global_scores, "G")
    d = _unit_vector(joint_distances, "D_J")
    if g.shape != d.shape:
        raise ValueError("G and D_J must be aligned")
    if not isinstance(spec, dict) or set(spec) != {"family", "alpha"}:
        raise ValueError("Unknown independent fusion specification")
    family, alpha = spec["family"], spec["alpha"]
    allowed = {"endpoint": (0., 1.), "additive": (.25, .5, .75),
               "geometric": (.25, .75), "product": (.5,)}
    if (family not in allowed or isinstance(alpha, (bool, np.bool_))
            or not isinstance(alpha, (int, float, np.integer, np.floating))
            or not np.isfinite(alpha) or alpha not in allowed[family]):
        raise ValueError("Unknown independent fusion family/weight")
    j = np.exp(-2. * d)
    if family == "endpoint":
        return g.copy() if alpha == 1. else j
    if family == "additive":
        return alpha * g + (1. - alpha) * j
    if family == "product":
        return g * j
    return np.power(g, alpha) * np.power(j, 1. - alpha)


def audit_components(data, score_components, max_candidates=20):
    """Check frozen G/H extraction on minimum-ID representatives of each class.

    G is reconstructed from saved float32 embeddings using a float64 dot
    product; it is not renormalized, matching the frozen source convention.
    Hellinger is independently accumulated using scalar ``math.fsum``. This is
    a sampled extraction audit, not a reconstruction of CNN or tensor voting.
    """
    if (isinstance(max_candidates, (bool, np.bool_))
            or not isinstance(max_candidates, (int, np.integer)) or max_candidates < 1):
        raise ValueError("max_candidates must be a positive integer")
    pool, bundle = data["pool"], data["bundle"]
    global_ids = _ids(data["global_ids"], "global feature IDs")
    descriptor_ids = _ids(data["descriptors"]["sample_ids"], "descriptor IDs")
    gallery = _ids(pool.gallery, "gallery IDs")
    if set(global_ids.tolist()) != set(gallery.tolist()) or set(descriptor_ids.tolist()) != set(gallery.tolist()):
        raise ValueError("Component sources must cover the exact complete gallery")
    features = np.asarray(data["global_features"], dtype=np.float64)
    joint = np.asarray(data["descriptors"]["joint128"], dtype=np.float64)
    empty = np.asarray(data["descriptors"]["empty"])
    if (features.ndim != 2 or features.shape[0] != len(global_ids) or not features.shape[1]
            or not np.isfinite(features).all() or joint.shape != (len(descriptor_ids), 128)
            or not np.isfinite(joint).all() or np.any(joint < 0)
            or empty.shape != (len(descriptor_ids),) or empty.dtype != np.bool_
            or not np.array_equal(joint.sum(1) == 0, empty)
            or not np.allclose(joint[~empty].sum(1), 1., rtol=0, atol=1e-10)):
        raise ValueError("Invalid frozen component arrays or empty descriptor flags")
    grows = {int(sample): i for i, sample in enumerate(global_ids)}
    drows = {int(sample): i for i, sample in enumerate(descriptor_ids)}
    representatives = {}
    for query in sorted(map(int, pool)):
        representatives.setdefault(int(bundle.class_ids[query]), query)
    query_ids = sorted(representatives.values())
    max_g, max_d, checked, empty_checked = 0., 0., 0, 0
    audited_candidates = {}
    roots = np.sqrt(joint)
    for query in query_ids:
        candidates = np.sort(_ids(pool[query], "audit candidates"))[:max_candidates]
        audited_candidates[str(query)] = candidates.tolist()
        supplied = score_components(query, candidates)
        if not isinstance(supplied, (tuple, list)) or len(supplied) != 2:
            raise ValueError("Component provider must return G and D_J")
        g = _unit_vector(supplied[0], "audit global scores")
        d = _unit_vector(supplied[1], "audit joint distances")
        if g.shape != candidates.shape or d.shape != candidates.shape:
            raise ValueError("Component provider must align with every requested candidate")
        qi = drows[query]
        for position, candidate in enumerate(candidates):
            candidate = int(candidate)
            gi, ci = grows[candidate], drows[candidate]
            reference_g = float(np.clip(np.dot(features[grows[query]], features[gi]), 0., 1.))
            if empty[qi] or empty[ci]:
                reference_d = 0. if empty[qi] and empty[ci] else 1.
                if d[position] != reference_d:
                    raise ValueError("Frozen empty-descriptor distance must be exactly zero or one")
                empty_checked += 1
            else:
                squared = math.fsum((float(a)-float(b))**2 for a, b in zip(roots[qi], roots[ci]))
                reference_d = min(1., math.sqrt(max(0., .5*squared)))
            g_error, d_error = abs(float(g[position])-reference_g), abs(float(d[position])-reference_d)
            if g_error > 2e-7:
                raise ValueError(f"Independent float64 global dot differs: {query}/{candidate}: {g_error}")
            if d_error > 1e-12:
                raise ValueError(f"Independent scalar Hellinger differs: {query}/{candidate}: {d_error}")
            max_g, max_d = max(max_g, g_error), max(max_d, d_error)
            checked += 1
    return {"status": "verified", "split": pool.split_name, "query_ids": query_ids,
            "num_queries": len(query_ids), "candidate_pairs": checked,
            "candidate_ids": audited_candidates, "max_candidates_per_query": int(max_candidates),
            "empty_pairs_checked": empty_checked, "max_global_absolute_error": max_g,
            "max_joint_distance_absolute_error": max_d,
            "global_tolerance": 2e-7, "joint_distance_tolerance": 1e-12,
            "scope": "Minimum original query ID per complete-label class, smallest candidate IDs: independent float64 dot product from frozen global embeddings and scalar Hellinger accumulation from frozen joint histograms; exact empty semantics. No CNN inference or tensor voting is rerun."}


def verify_evaluation(folder, pool, bundle, ks, signature, score_components, tie_seed=2026):
    """Verify before sealing; score_components(query, candidates) returns (G, D_J)."""
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
    independent_score(np.array([.5]), np.array([.5]), manifest["spec"])
    if (manifest.get("num_queries") != len(pool) or manifest.get("gallery_count") != len(pool.gallery)
            or manifest.get("candidate_rule") != "all_gallery_except_query"
            or manifest.get("retrieval_protocol_signature", manifest.get("protocol_signature")) != pool.protocol_signature):
        raise ValueError("Evaluation manifest protocol/counts differ")
    query_array, gallery_array = _ids(list(pool), "query IDs"), _ids(pool.gallery, "gallery IDs")
    query_ids, gallery = set(query_array.tolist()), set(gallery_array.tolist())
    heldout = _ids(bundle.valid_ids if pool.split_name == "valid" else bundle.reserved_ids,
                   "held-out original IDs")
    if (not query_ids <= gallery or pool.candidates_per_query != len(gallery) - 1
            or max(ks) > pool.candidates_per_query or gallery != set(heldout.tolist())):
        raise ValueError("Invalid complete-gallery protocol or K")
    if pool.split_name == "test" and query_ids != gallery:
        raise ValueError("Test requires every gallery sample as a query")
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
    metric_error, score_error, component_error = 0., 0., 0.
    with (folder / "rankings.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != RANKING_FIELDS:
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
            returned = score_components(query, candidates)
            if not isinstance(returned, (tuple, list)) or len(returned) != 2:
                raise ValueError("Component provider must return aligned full-gallery G and D_J")
            replay_ids = candidates
            global_scores = _unit_vector(returned[0], "full-gallery global scores")
            joint_distances = _unit_vector(returned[1], "full-gallery joint distances")
            if global_scores.shape != candidates.shape or joint_distances.shape != candidates.shape:
                raise ValueError("Component provider must cover the complete gallery except self")
            replay_scores = independent_score(global_scores, joint_distances, manifest["spec"])
            order = independent_order(query, replay_ids, replay_scores, tie_seed)[:max(ks)]
            if not np.array_equal(ranked, replay_ids[order]):
                raise ValueError("Saved TopK differs from independent full-candidate ordering")
            saved_scores = np.asarray([float(row["similarity_score"]) for row in rows])
            if not np.isfinite(saved_scores).all() or not np.allclose(saved_scores, replay_scores[order], rtol=0, atol=ATOL):
                raise ValueError("Saved score differs from independent full-candidate fusion")
            score_error = max(score_error, float(np.abs(saved_scores - replay_scores[order]).max()))
            stored_g = _unit_vector([float(row["global_score"]) for row in rows], "saved global scores")
            stored_d = _unit_vector([float(row["joint_distance"]) for row in rows], "saved joint distances")
            error = max(float(np.abs(stored_g - global_scores[order]).max()),
                        float(np.abs(stored_d - joint_distances[order]).max()))
            if error > ATOL:
                raise ValueError("Saved G/J components differ from full-gallery source replay")
            component_error = max(component_error, error)
            saved_formula = independent_score(stored_g, stored_d, manifest["spec"])
            error = float(np.abs(saved_scores - saved_formula).max())
            if error > ATOL:
                raise ValueError("Saved score differs from independent saved-component formula")
            score_error = max(score_error, error)
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
            "max_component_absolute_error": component_error,
            "independent_fusion_all_candidates": True,
            "aggregation_values_checked": aggregation_count,
            "scope": "Every query: independently computed G/J fusion on all candidate components, independent SHA TopK ordering, saved-component equality, full-gallery label denominators, and every K/query/class/macro/count-group metric and count. Frozen G/J feature and distance extraction are supplied by the source provider, not independently reimplemented here.",
            "ranking_sha256": final_hashes["rankings.csv"], "files": final_hashes}
