"""Independent, read-only B1 ablation and frozen-source verification.

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
    """Explicit ablation formulas, independent of production score_variant."""
    required = {"family", "strength", "local_mode", "local_weight", "global_weight", "angular_gamma"}
    if not required.issubset(spec) or spec["family"] not in ("J", "I"):
        raise ValueError("Unrecognized fixed ablation specification")
    if spec["local_mode"] not in ("none", "greedy", "optimal"):
        raise ValueError("Unknown local matching")
    for key in ("strength", "local_weight", "global_weight", "angular_gamma"):
        if not np.isfinite(spec[key]) or spec[key] < 0:
            raise ValueError("Invalid fixed ablation weight")
    values = {key: np.array([float(r[key]) for r in rows]) for key in
              ("f_score", "optimal_f_score", "global_score", "angular_distance",
               "joint_distance", "independent_distance", "structure_distance",
               "b0_score", "full_b1_score")}
    if any(not np.isfinite(v).all() for v in values.values()):
        raise ValueError("Nonfinite saved score component")
    if any(np.any((v < -FORMULA_ATOL) | (v > 1 + FORMULA_ATOL)) for v in values.values()):
        raise ValueError("Saved score component lies outside [0,1]")
    angular = values["angular_distance"]
    original = (.5 * values["optimal_f_score"] + .5 * values["global_score"]) * np.exp(-.5 * angular)
    np.testing.assert_allclose(values["b0_score"], original, atol=FORMULA_ATOL, rtol=FORMULA_ATOL)
    np.testing.assert_allclose(values["full_b1_score"], original * np.exp(-2 * values["joint_distance"]),
                               atol=FORMULA_ATOL, rtol=FORMULA_ATOL)
    distance = values["joint_distance"] if spec["family"] == "J" else values["independent_distance"]
    np.testing.assert_allclose(values["structure_distance"], distance, atol=FORMULA_ATOL, rtol=0)
    local = (values["optimal_f_score"] if spec["local_mode"] == "optimal" else
             values["f_score"] if spec["local_mode"] == "greedy" else np.zeros(len(rows)))
    if spec["local_weight"] == 0 and spec["global_weight"] == 0:
        base = np.ones(len(rows))
    else:
        base = spec["local_weight"] * local + spec["global_weight"] * values["global_score"]
    return base * np.exp(-spec["angular_gamma"] * angular) * np.exp(-spec["strength"] * distance)


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
            "scope": "all saved TopK IDs/SHA ties/independent score formulas; full-gallery label denominators; all query/class/macro/count-group metrics. Unsaved candidate ordering is separately sampled.",
            "ranking_sha256": _sha(folder/"rankings.csv"), "completion_sha256": _sha(folder/"completed.json")}


def independent_descriptor(raw, sigma=9., threshold=.32, block_size=128):
    """Independent scalar tensor entries + analytic eigenvalues + scatter soft bins."""
    raw = np.asarray(raw)
    support = np.argwhere((raw == 1) | (raw == 2)).astype(float)
    if raw.ndim != 2 or not len(support) or not np.isin(raw, [0,1,2,3]).all():
        raise ValueError("Invalid original wafer/support")
    points = np.argwhere(raw == 2).astype(float)
    center = (support.min(axis=0) + support.max(axis=0))/2.
    radius = float(np.sqrt(((support-center)**2).sum(axis=1)).max())
    saliency = np.zeros((len(points), 2))
    for begin in range(0, len(points), block_size):
        delta = points[begin:begin+block_size,None,:] - points[None,:,:]
        dx, dy = delta[:,:,0], delta[:,:,1]
        d2 = dx*dx + dy*dy
        nz = d2 > 0
        w = np.exp(-d2/sigma) * nz
        ratio = np.divide(w, d2, out=np.zeros_like(w), where=nz)
        xx = np.sum(w - .5*ratio*dx*dx, axis=1)
        yy = np.sum(w - .5*ratio*dy*dy, axis=1)
        xy = np.sum(-.5*ratio*dx*dy, axis=1)
        gap = np.sqrt(np.maximum((xx-yy)**2+4*xy*xy, 0))
        low = np.maximum((xx+yy-gap)/2, 0)
        saliency[begin:begin+len(delta)] = np.column_stack((gap, low))
    maximum = saliency.max(axis=0) if len(points) else np.zeros(2)
    keep = ~np.all(saliency < threshold*maximum, axis=1)
    normalized = np.divide(saliency, maximum, out=np.zeros_like(saliency), where=maximum>0)
    radial = np.sqrt(((points-center)**2).sum(axis=1))/radius if radius else np.zeros(len(points))
    triple = np.column_stack((radial, normalized))[keep]
    histogram = np.zeros((8,4,4))
    for xyz in triple:
        position = np.clip(xyz, 0, 1)*np.array([7,3,3])
        lower = np.floor(position).astype(int)
        choices = [[(int(k), 1-f)] + ([(int(k+1), f)] if f>0 else [])
                   for k,f in zip(lower, position-lower)]
        for i,wi in choices[0]:
            for j,wj in choices[1]:
                for k,wk in choices[2]:
                    histogram[i,j,k] += wi*wj*wk
    if len(triple):
        histogram /= len(triple)
    angle, outer = independent_angular(raw)
    return {"joint128": histogram.reshape(-1), "empty": len(triple)==0,
            "original_count": len(points), "retained_count": int(keep.sum()),
            "saliency_max": maximum, "moments": angle, "outer_count": outer}


def independent_angular(raw):
    raw = np.asarray(raw)
    support = np.argwhere((raw == 1) | (raw == 2))
    points = np.argwhere(raw == 2)
    if not len(support):
        raise ValueError("Empty wafer support")
    center = (support.min(axis=0)+support.max(axis=0))/2
    radius = np.sqrt(np.sum((support-center)**2, axis=1)).max()
    radius_values = np.sqrt(np.sum((points-center)**2,axis=1))/radius if radius else np.zeros(len(points))
    delta = points[radius_values >= .65] - center
    if not len(delta):
        return np.zeros(4), 0
    theta = np.arctan2(delta[:,0],delta[:,1])
    # Complex exponential preserves the frozen sin/cos and reduction semantics.
    moments = np.array([abs(np.exp(1j*k*theta).mean()) for k in range(1,5)])
    return moments, len(delta)


def _independent_components(query, candidates, scorer, angular_by_id):
    """Frozen arrays -> F optimal by subset DP, G dot product, and raw-mask angle."""
    meta = scorer.meta
    qrow = scorer.rows[int(query)]
    q = np.arange(meta["offsets"][qrow], meta["offsets"][qrow+1])
    n = len(candidates)
    global_part = np.clip(scorer.global_features[[scorer.global_rows[int(c)] for c in candidates]] @
                          scorer.global_features[scorer.global_rows[int(query)]], 0, 1)
    angles = np.array([np.abs(angular_by_id[int(query)][0]-angular_by_id[int(c)][0]).mean()
                       if angular_by_id[int(query)][1] and angular_by_id[int(c)][1] else 0.
                       for c in candidates])
    if not len(q):
        return np.zeros(n), global_part, angles
    cosine = np.clip(scorer.features[q] @ scorer.features.T, 0, 1)
    cdf = np.cumsum(meta["histograms"], axis=1)
    radial = np.abs(cdf[q,None,:]-cdf[None,:,:]).sum(axis=2)/16
    distance = .5*np.abs(np.log(meta["relative_area"][q])[:,None]-np.log(meta["relative_area"])[None,:])
    pair = cosine*np.exp(-radial)*(.85+.15*np.exp(-distance/1.5))
    slots = np.asarray(scorer.padded[[scorer.rows[int(c)] for c in candidates]])
    width = slots.shape[1]
    if not width:
        return np.zeros(n), global_part, angles
    matrix = np.clip(pair[:,np.maximum(slots,0)].transpose(1,0,2),0,1)
    matrix *= (slots>=0)[:,None,:]
    weights = np.sqrt(np.asarray(meta["area"][q],dtype=float))
    # State: which candidate slots have been used. Skipping handles dummy matches.
    dp = {0:np.zeros(n)}
    for qi in range(len(q)):
        next_dp = {mask:value.copy() for mask,value in dp.items()}
        for mask,value in dp.items():
            for cj in range(width):
                if not (mask >> cj)&1:
                    target = mask | (1 << cj)
                    trial = value + weights[qi]*matrix[:,qi,cj]
                    if target in next_dp:
                        next_dp[target] = np.maximum(next_dp[target],trial)
                    else:
                        next_dp[target] = trial
        dp = next_dp
    optimal = np.maximum.reduce(list(dp.values()))/weights.sum()
    return optimal, global_part, angles


def _distance(query_hist, all_hist, empty_q, empty_c, device):
    if device == "cpu":
        delta = np.sqrt(all_hist)-np.sqrt(query_hist)
        value = np.sqrt(np.maximum(.5*np.sum(delta*delta,axis=1),0))
    else:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA source audit requested without available CUDA")
        with torch.no_grad():
            candidate = torch.as_tensor(np.sqrt(all_hist),dtype=torch.float64,device=device)
            query = torch.as_tensor(np.sqrt(query_hist),dtype=torch.float64,device=device)
            delta = candidate-query
            value = (.5*delta.square().sum(dim=1)).clamp_min(0).sqrt().cpu().numpy()
    value = np.clip(value,0,1)
    value[np.logical_xor(empty_q,empty_c)] = 1.
    value[np.logical_and(empty_q,empty_c)] = 0.
    return value


def audit_source_sample(source_root, pool, bundle, scorer, descriptor_values, device="cuda"):
    """37 per-split frozen queries; independently rebuild raw-query TV and full Top500.

    Gallery TV descriptors and frozen CNN features remain audited inputs. All
    gallery angular descriptors are rebuilt from raw masks. No outputs are written.
    Numeric rank changes are explicit failures, never silently tolerance-accepted.
    """
    source_root = Path(source_root)
    if not hasattr(bundle,"maps"):
        raise ValueError("Source audit requires bundle.maps indexed by original sample_id")
    labels = np.asarray(bundle.labels,dtype=int)
    selected = {}
    for query in pool:
        label = tuple(labels[int(query)].tolist())
        if label not in selected or int(query)<selected[label]:
            selected[label] = int(query)
    queries = sorted(selected.values())
    if len(queries) != 37:
        raise ValueError("Formal source audit requires all 37 complete label combinations")
    split = pool.split_name
    source_folder = source_root/split/"J_2"
    marker = _json(source_folder/"completed.json")
    manifest = _json(source_folder/"evaluation_manifest.json")
    if manifest["spec"] != {"family":"J","strength":2.0} or manifest["split"] != split:
        raise ValueError("Expected the frozen original J_2 source")
    ranking_file = source_folder/"rankings.csv"
    actual_hash = _sha(ranking_file)
    if actual_hash != marker["files"]["rankings.csv"]:
        raise ValueError("Source rankings hash changed")
    wanted = set(queries)
    saved = {}
    with ranking_file.open(newline="",encoding="utf-8") as handle:
        for query, rows in groupby(csv.DictReader(handle),key=lambda r:int(r["query_id"])):
            if query in wanted:
                if query in saved:
                    raise ValueError("Duplicate sampled source query")
                saved[query] = list(rows)
            else:
                for _ in rows:
                    pass
    if set(saved) != wanted:
        raise ValueError("Sample query missing from original ranking")
    cache_rows = {int(x):i for i,x in enumerate(descriptor_values["sample_ids"])}
    if set(cache_rows) != set(map(int,pool.gallery)):
        raise ValueError("Descriptor gallery identities differ")
    angular = {int(c):independent_angular(bundle.maps[int(c)]) for c in pool.gallery}
    maximum = {"descriptor":0., "saliency_max":0., "angular":0.,
               "optimal_f":0., "global":0., "structure_distance":0., "score":0.}
    cases = []
    for query in queries:
        raw = independent_descriptor(bundle.maps[query])
        qi = cache_rows[query]
        if any(raw[k] != descriptor_values[k][qi] for k in ("empty","original_count","retained_count","outer_count")):
            raise ValueError(f"Independent raw descriptor count differs for {query}")
        for key,destination in (("joint128","descriptor"),("saliency_max","saliency_max"),("moments","angular")):
            err = float(np.max(np.abs(raw[key]-descriptor_values[key][qi])))
            maximum[destination] = max(maximum[destination],err)
            np.testing.assert_allclose(raw[key],descriptor_values[key][qi],atol=SOURCE_ATOL,rtol=SOURCE_ATOL,
                                       err_msg=f"Original descriptor mismatch for {query}/{key}")
        candidates = np.asarray(pool[query],dtype=np.int64)
        rows = np.array([cache_rows[int(c)] for c in candidates])
        optimal, global_part, angle = _independent_components(query,candidates,scorer,angular)
        # Independent query TV reconstruction participates in every gallery score.
        distances = _distance(raw["joint128"],descriptor_values["joint128"][rows],
                              raw["empty"],descriptor_values["empty"][rows],device)
        scores = (.5*optimal+.5*global_part)*np.exp(-.5*angle)*np.exp(-2*distances)
        records = saved[query]
        if len(records)!=500 or [int(r["rank"]) for r in records] != list(range(1,501)):
            raise ValueError("Original source sampled ranking must contain all Top500 rows")
        ids = np.array([int(r["candidate_id"]) for r in records])
        lookup = {int(c):i for i,c in enumerate(candidates)}
        selected_indices = np.array([lookup[int(c)] for c in ids])
        for field,array,destination in (
                ("optimal_f_score",optimal,"optimal_f"),("global_score",global_part,"global"),
                ("angular_distance",angle,"angular"),("structure_distance",distances,"structure_distance"),
                ("similarity_score",scores,"score")):
            recorded = np.array([float(r[field]) for r in records])
            recomputed = array[selected_indices]
            error = float(np.max(np.abs(recorded-recomputed)))
            maximum[destination] = max(maximum[destination],error)
            np.testing.assert_allclose(recorded,recomputed,atol=SOURCE_ATOL,rtol=SOURCE_ATOL,
                                       err_msg=f"Independent source score mismatch: {query}/{field}")
        comparison = compare_full_ranking(query,candidates,scores,ids,manifest["tie_seed"])
        cases.append({"query_id":query,"label_signature":"".join(map(str,labels[query])),
                      "candidate_count":len(candidates),**comparison})
        print(f"Independent source audit {split}: {len(cases)}/{len(queries)}; query={query}; "
              f"rank_changes={comparison['mismatch_count']}",flush=True)
    mismatches = [case for case in cases if not case["exact_order_match"]]
    return {"status":"verified" if not mismatches else "failed_rank_identity",
            "split":split,"num_queries":len(queries),"sample_rule":"minimum original sample_id among frozen queries in each full multi-hot class",
            "query_ids":queries,"device":device,"absolute_tolerance":SOURCE_ATOL,
            "max_absolute_errors":maximum,"ranking_cases":cases,
            "numeric_rank_change_queries":[case["query_id"] for case in mismatches if case["numeric_only"]],
            "source_hashes":{"rankings.csv":actual_hash,"completed.json":_sha(source_folder/"completed.json"),
                             "evaluation_manifest.json":_sha(source_folder/"evaluation_manifest.json"),
                             "descriptors.npz":_sha(source_root/"descriptors"/split/f"{split}.npz")},
            "scope":"Independent raw TV/soft histogram for 37 queries; raw angular moments for complete gallery; frozen CNN features/region geometry plus independent weighted matching; full-gallery J_2 Top500 recomputation.",
            "limitations":"Gallery TV values and CNN/region feature extraction are frozen inputs, not independently re-extracted. Sampling does not establish all-query rank equivalence."}
