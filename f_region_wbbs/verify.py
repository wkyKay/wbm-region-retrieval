"""Independent full-cache ranking/denominator verification for four fixed stages.

Ordinary exact/coverage metrics reuse the established independent implementation.
Cross metrics reuse only independent arithmetic/aggregation, with a new split-
aware candidate audit; none of the old experimental verifiers are modified.
"""
from __future__ import annotations
import csv
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import re
import numpy as np

from f_region_structure import verify as ordinary
from fgw_cross_test.verify import (ATOL, REPORT_FILES as CROSS_REPORT_FILES, _json, _sha,
    _digest, independent_order, independent_coverage, _assert_value, _records, _metric_name)
from fgj_cross_test.verify import _check_aggregates
from f_region_evidence.verify import independent_components, independent_greedy
from fgw_fusion.wbbs_backend import original_module
from .protocol import PROTOCOL, RULE, KS as FORMAL_KS, audit_pool

REPORT_FILES = (*ordinary.REPORT_FILES, "run_identity.json")
RANKING_FIELDS = ("query_id", "rank", "candidate_id", "similarity_score")
_METADATA_KEYS = ("signature", "context_signature", "variant", "spec", "split", "stage",
    "ks", "tie_seed", "num_queries", "gallery_count", "candidate_rule",
    "retrieval_protocol_signature", "provider_source", "source_binding_sha256")


def independent_grid():
    result = {"region_j": {"family": "j_baseline"}}
    for prefix, mode in (("all", "all_points"), ("denoised", "wbbs_denoised")):
        for label, penalty in (("bb", 0.), ("wbbs", 5.)):
            for weight, suffix in ((.25, "025"), (.4, "04"), (.6, "06")):
                result[f"{prefix}_{label}_a{suffix}"] = {"family": "wbbs", "point_mode": mode,
                    "count_alpha": penalty, "cnn_weight": weight, "area_weight": .25}
    return result


def audit_formulas(provider, bundle, specs, max_candidates=20):
    """Original raster-order WBBS reference, scalar regional fusion and greedy.

    The production provider should read sealed full BBP count matrices. This
    samples regional counts/formulas independently; it does not recompute all
    51.8 million map pairs and does not audit the separate global benchmark.
    """
    grid=independent_grid()
    if (not specs or any(name not in grid or spec!=grid[name] for name,spec in specs.items())
            or type(max_candidates) is not int or max_candidates<1):
        raise ValueError("Expected a nonempty frozen regional-formula audit grid")
    representatives={}
    for query in sorted(map(int,provider.pool)):
        representatives.setdefault(int(bundle.class_ids[query]),query)
    queries=[representatives[cls] for cls in sorted(representatives)]
    result={"status":"verified","query_ids":queries,"num_queries":len(queries),
            "candidate_pairs":0,"bb_region_pairs_checked":0,"bb_counts_exact":True,
            "max_raw_formula_absolute_error":0.,"max_matching_absolute_error":0.,
            "max_wbbs_absolute_error":0.,"variants":{name:{"query_ids":queries,
                "candidate_pairs":0,"max_raw_formula_absolute_error":0.,"max_matching_absolute_error":0.}
                for name in specs}}
    original=original_module()
    for query in queries:
        ids=np.sort(np.asarray(provider.pool[query],dtype=np.int64))[:max_candidates]
        qt=provider.tokens(query);tokens=[provider.tokens(candidate) for candidate in ids]
        ct=np.concatenate(tokens) if tokens else np.array([],dtype=np.int64)
        parts=provider.parts(query)
        direct=independent_components(provider.cache,qt,ct)
        component_error=max((float(np.max(np.abs(np.asarray(parts[key])[:,ct]-direct[key]),initial=0.))
            for key in ("cosine","joint_distance","structure_similarity","area_similarity")),default=0.)
        if component_error>2e-6:
            raise ValueError("Independent C/H/T/A regional components differ")
        weights={}
        for mode in ("all_points","wbbs_denoised"):
            source=provider.cache["point_sets"][mode]
            offsets=np.asarray(source["offsets"]);descriptors=np.asarray(source["descriptors"])
            if (descriptors.dtype!=np.float64 or descriptors.ndim!=2 or descriptors.shape[1]!=3
                    or not np.isfinite(descriptors).all() or np.any(descriptors<0) or np.any(descriptors>1)):
                raise ValueError("Original WBBS audit requires complete float64 point descriptors")
            values={alpha:np.zeros((len(qt),len(ct)),dtype=np.float64) for alpha in (0.,5.)}
            for i,qi in enumerate(qt):
                qpoints=descriptors[offsets[qi]:offsets[qi+1]]
                for j,ci in enumerate(ct):
                    cpoints=descriptors[offsets[ci]:offsets[ci+1]]
                    # This is the unchanged original blocked-cdist/first-argmin
                    # implementation, independent of accelerated cached counts.
                    count=len(original.best_buddies(qpoints,cpoints,mu=1.,block_size=256))
                    if int(parts["bb_counts"][mode][i,ci])!=count:
                        raise ValueError("Cached BBP count differs from original raster-order WBBS")
                    n,m=len(qpoints),len(cpoints)
                    for alpha in (0.,5.):
                        value=0. if min(n,m)==0 else count/min(n,m)*float(np.exp(-alpha*abs(n-m)/2061))
                        values[alpha][i,j]=value
                        error=abs(value-float(parts["wbbs_scores"][mode][alpha][i,ci]))
                        if error>1e-15:
                            raise ValueError("WBBS normalization/count penalty uses different full point counts")
                        result["max_wbbs_absolute_error"]=max(result["max_wbbs_absolute_error"],error)
                    result["bb_region_pairs_checked"]+=1
            weights[mode]=values
        for name,spec in specs.items():
            if spec["family"]=="j_baseline":
                raw=.75*(.4*parts["cosine"]+.6*parts["structure_similarity"])+.25*parts["area_similarity"]
                expected=direct["raw_score"]
            else:
                alpha=spec["cnn_weight"];w=parts["wbbs_scores"][spec["point_mode"]][spec["count_alpha"]]
                raw=.75*(alpha*parts["cosine"]+(1.-alpha)*w)+.25*parts["area_similarity"]
                expected=.75*(alpha*direct["cosine"]+(1.-alpha)*weights[spec["point_mode"]][spec["count_alpha"]])+.25*direct["area_similarity"]
            formula_error=max(component_error,float(np.max(np.abs(raw[:,ct]-expected),initial=0.)))
            if formula_error>2e-6:
                raise ValueError("Independent regional J/WBBS fusion formula differs")
            all_ids,scores=provider.score(query,spec,parts)
            lookup=dict(zip(map(int,all_ids),map(float,scores)))
            start=0
            detail=result["variants"][name]
            for candidate,tokens_ in zip(ids,tokens):
                # Preserve the production raw matrix when testing matching:
                # tiny float32 cosine differences can change an exact pair tie.
                value=independent_greedy(raw[:,ct[start:start+len(tokens_)]],provider.cache["area"][qt])
                error=abs(value-lookup[int(candidate)])
                if error>ATOL:
                    raise ValueError("Independent original greedy sqrt-area aggregation differs")
                detail["max_matching_absolute_error"]=max(detail["max_matching_absolute_error"],error)
                detail["candidate_pairs"]+=1;result["candidate_pairs"]+=1
                start+=len(tokens_)
            detail["max_raw_formula_absolute_error"]=max(detail["max_raw_formula_absolute_error"],formula_error)
            result["max_raw_formula_absolute_error"]=max(result["max_raw_formula_absolute_error"],formula_error)
            result["max_matching_absolute_error"]=max(result["max_matching_absolute_error"],detail["max_matching_absolute_error"])
    result["scope"]="Each complete-class minimum query ID and first sorted candidate IDs; original WBBS blocked squared-Euclidean/first-raster mutual nearest neighbours, exact integer counts, full-point normalization and fixed 2061-die count penalty. Independent C/H/T/A and regional fusion formulas; scalar greedy on identical production raw pairs preserves float32 ties and the full query sqrt-area denominator. Sampled audit, not all-pair independent recomputation; global WBBS has separate source validation."
    return result


def _header(folder, pool, bundle, ks, signature, provider, source_binding, tie_seed):
    if (not isinstance(source_binding, dict) or not source_binding
            or any(not isinstance(path, str) or not Path(path).is_absolute()
                   or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha)
                   for path, sha in source_binding.items())):
        raise ValueError("Source binding must contain absolute paths and SHA-256 hashes")
    if (not ks or list(ks) != sorted(set(ks)) or any(type(k) is not int or k < 1 for k in ks)
            or type(tie_seed) is not int or tie_seed != 2026):
        raise ValueError("Invalid cutoffs or frozen tie seed")
    manifest = _json(folder / "evaluation_manifest.json")
    identity = _json(folder / "run_identity.json")
    stage, variant, spec = manifest.get("stage"), manifest.get("variant"), manifest.get("spec")
    grid = independent_grid()
    if stage == "cross_valid":
        grid["global_wbbs"] = {"family": "global_wbbs"}
    context = manifest.get("context_signature")
    split = "valid" if isinstance(stage, str) and stage.endswith("valid") else "test"
    cross = isinstance(stage, str) and stage.startswith("cross_")
    protocol = PROTOCOL if cross else "retrieval_v1"
    expected_signature = _digest({"context": context, "stage": stage, "variant": variant, "spec": spec})
    queries, gallery = ordinary._ids(list(pool), "queries"), ordinary._ids(pool.gallery, "gallery")
    labels, classes = np.asarray(bundle.labels), np.asarray(bundle.class_ids)
    provider_binding = {"signature": provider.signature, "completed_sha256": _sha(provider.marker)}
    if (stage not in ("valid", "cross_valid", "test", "cross_test") or variant not in grid or spec != grid[variant]
            or not isinstance(context, str) or not context or signature != expected_signature
            or manifest.get("signature") != signature or manifest.get("split") != split
            or pool.split_name != split or manifest.get("protocol") != protocol
            or manifest.get("ks") != list(ks) or manifest.get("tie_seed") != tie_seed
            or manifest.get("num_queries") != len(queries) or manifest.get("gallery_count") != len(gallery)
            or manifest.get("retrieval_protocol_signature") != pool.protocol_signature
            or manifest.get("source_binding_sha256") != _digest(source_binding)
            or manifest.get("provider_source") != provider_binding
            or not isinstance(provider.signature, str) or not provider.signature
            or not np.array_equal(provider.query_ids, queries) or not np.array_equal(provider.gallery_ids, gallery)
            or not np.isin(queries, gallery).all()
            or labels.ndim != 2 or labels.shape[1] != 8 or not np.isin(labels, (0, 1)).all()
            or max(gallery) >= len(labels) or classes.shape != (len(labels),) or classes.dtype.kind not in "iu"
            or np.any(labels[gallery].sum(axis=1) == 0)):
        raise ValueError("Evaluation identity/frozen specification/cache/source differs")
    label_classes = {tuple(labels[q]): int(classes[q]) for q in gallery}
    if (len(set(label_classes.values())) != len(label_classes)
            or any(label_classes[tuple(labels[q])] != int(classes[q]) for q in gallery)):
        raise ValueError("Class IDs do not identify unique complete raw label combinations")
    if (identity.get("evaluation_signature") != signature
            or any(manifest.get(k) != identity.get(k) for k in
                   ("key", "run_id", "started_at", "variant", "split", "stage", "protocol"))
            or not identity.get("key") or not identity.get("run_id") or not identity.get("started_at")):
        raise ValueError("Evaluation time identity differs")
    if cross:
        if (manifest.get("candidate_rule") != RULE or pool.metadata.get("protocol_version") != PROTOCOL
                or pool.metadata.get("candidate_rule") != RULE or manifest.get("derived_protocol") != pool.metadata
                or hashlib.sha256(json.dumps(pool.metadata, sort_keys=True).encode()).hexdigest() != pool.protocol_signature
                or pool.metadata.get("base_protocol_signature") != provider.protocol_signature):
            raise ValueError("Cross-combination candidate protocol differs")
        if any((folder / name).exists() for name in ("metrics.json", "per_query.csv", "per_class.csv")):
            raise ValueError("Cross-combination evaluation must not emit exact reports")
    elif (manifest.get("candidate_rule") != "all_gallery_except_query" or manifest.get("derived_protocol")
          or provider.protocol_signature != pool.protocol_signature):
        raise ValueError("Ordinary complete-gallery candidate protocol differs")
    formal = manifest.get("formal_protocol", True)
    if type(formal) is not bool:
        raise ValueError("Formal protocol flag must be boolean")
    if formal and (len(queries) != (500 if split == "valid" else 7199)
                   or len(gallery) != (3599 if split == "valid" else 7199) or list(ks) != FORMAL_KS
                   or (split == "test" and set(queries.tolist()) != set(gallery.tolist()))):
        raise ValueError("Formal evaluation requires complete frozen queries/gallery and seven K")
    for filename in (["coverage_metrics.json"] if cross else ["metrics.json", "coverage_metrics.json"]):
        metadata = _json(folder / filename).get("metadata", {})
        if (any(metadata.get(key) != manifest.get(key) for key in _METADATA_KEYS)
                or (cross and metadata.get("derived_protocol") != pool.metadata)):
            raise ValueError("Metric report metadata/signature differs")
    return manifest, queries, gallery, labels, classes, provider_binding, formal


def _scores(provider, q, candidates):
    scores = np.asarray(provider.scores(q, candidates))
    if (scores.dtype != np.float64 or scores.shape != candidates.shape
            or not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1)):
        raise ValueError("Cache must supply complete finite float64 similarity vectors in [0,1]")
    return scores


def verify_evaluation(folder, pool, bundle, ks, signature, provider, source_binding, tie_seed=2026):
    """Replay all sealed scores and independently verify all requested reports."""
    folder = Path(folder)
    manifest, queries, gallery, labels, classes, provider_binding, formal = _header(
        folder, pool, bundle, ks, signature, provider, source_binding, tie_seed)
    cross = manifest["stage"].startswith("cross_")
    candidate_audit = audit_pool(pool, labels, ks, formal=formal)
    files = CROSS_REPORT_FILES if cross else REPORT_FILES
    initial_hashes = {name: _sha(folder / name) for name in files}
    source_digest = _digest(source_binding)
    if cross:
        receipt = _verify_cross(folder, pool, labels, classes, queries, gallery, ks, provider, tie_seed, candidate_audit)
    else:
        seen = []
        candidate_digest = hashlib.sha256()
        def replay(q):
            if len(seen) >= len(queries) or q != int(queries[len(seen)]):
                raise ValueError("Ranking query order/identity differs")
            seen.append(int(q))
            candidates = np.asarray(pool[q], dtype=np.int64)
            candidate_digest.update(np.asarray([q, len(candidates)], dtype="<i8").tobytes())
            candidate_digest.update(np.asarray(candidates, dtype="<i8").tobytes())
            return candidates, _scores(provider, q, candidates)
        receipt = ordinary.verify_evaluation(folder, pool, bundle, ks, signature, replay, tie_seed)
        receipt.update(candidate_identity_sha256=candidate_digest.hexdigest(),
                       candidate_count_min=len(gallery)-1, candidate_count_max=len(gallery)-1,
                       metric_query_counts={family: _json(folder / filename)["metric_query_counts"]
                           for family, filename in (("exact", "metrics.json"), ("coverage", "coverage_metrics.json"))})
    final_hashes = {name: _sha(folder / name) for name in files}
    if (initial_hashes != final_hashes or _digest(source_binding) != source_digest
            or provider_binding != {"signature": provider.signature, "completed_sha256": _sha(provider.marker)}):
        raise ValueError("Reports, sealed cache marker or source binding changed during verification")
    receipt.update(status="verified", signature=signature, context_signature=manifest["context_signature"],
                   split=pool.split_name, stage=manifest["stage"], variant=manifest["variant"], spec=manifest["spec"],
                   ks=list(ks), tie_seed=tie_seed, provider_source=provider_binding,
                   source_binding_sha256=source_digest, source_file_count=len(source_binding),
                   exact_reports=not cross, ranking_sha256=final_hashes["rankings.csv"], files=final_hashes,
                   scope="Every eligible score read from the newly sealed complete float64 cache; independent candidate identities, SHA TopK and all query/class/macro/group values and full-pool denominators. Not a second regional score computation or all-pair independent formula audit; separate sampled original regional-WBBS formula and greedy audits apply; no calibration or new training.")
    if receipt["metric_query_counts"]["coverage"] != candidate_audit["metric_query_counts"]:
        raise ValueError("Coverage valid-query counts differ from independent candidate audit")
    receipt.update(candidate_protocol_audit=candidate_audit,
                   candidate_identity_sha256=candidate_audit["candidate_identity_sha256"],
                   query_ids_sha256=candidate_audit["query_ids_sha256"],
                   gallery_ids_sha256=candidate_audit["gallery_ids_sha256"])
    return receipt


def _verify_cross(folder, pool, labels, classes, queries, gallery, ks, provider, tie_seed, candidate_audit):
    report = _json(folder / "coverage_metrics.json")
    if (report.get("split") != pool.split_name or report.get("test_evaluated") != (pool.split_name == "test")
            or report.get("ks") != list(ks) or report.get("metric_version") != "coverage_v1"):
        raise ValueError("Coverage report metadata/signature differs")
    expected = _records(folder / "coverage_per_query.csv", "query_id")
    if set(expected) != set(map(int, queries)):
        raise ValueError("Per-query report misses/duplicates frozen queries")
    perquery, seen, counts = {}, [], []
    candidate_digest = hashlib.sha256()
    metric_error = score_error = 0.
    with (folder / "rankings.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(RANKING_FIELDS):
            raise ValueError("Ranking CSV must have exactly the four frozen columns")
        for query, records in groupby(reader, key=lambda row: int(row["query_id"])):
            if len(seen) >= len(queries) or query != int(queries[len(seen)]):
                raise ValueError("Ranking query order/identity differs")
            seen.append(query)
            candidates = np.asarray(pool[query])
            eligible = gallery[(gallery != query) & np.any(labels[gallery] != labels[query], axis=1)]
            if not np.array_equal(candidates, eligible) or len(candidates) < max(ks):
                raise ValueError("Candidate identities differ from independent full label exclusion")
            counts.append(len(candidates))
            candidate_digest.update(json.dumps([query, candidates.tolist()], separators=(",", ":")).encode("ascii") + b"\n")
            rows = list(records)
            ranked = np.asarray([int(row["candidate_id"]) for row in rows], dtype=np.int64)
            if (len(rows) != max(ks) or [int(row["rank"]) for row in rows] != list(range(1, max(ks) + 1))
                    or len(np.unique(ranked)) != len(ranked) or not np.isin(ranked, candidates).all()):
                raise ValueError("Invalid saved TopK identities/ranks")
            scores = _scores(provider, query, candidates)
            if (scores.dtype != np.float64 or scores.shape != candidates.shape
                    or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))):
                raise ValueError("Cache must supply complete finite float64 similarity vectors in [0,1]")
            order = independent_order(query, candidates, scores, tie_seed)[:max(ks)]
            if not np.array_equal(ranked, candidates[order]):
                raise ValueError("Saved TopK differs from independent full-candidate cache ordering")
            saved_scores = np.asarray([float(row["similarity_score"]) for row in rows])
            if not np.isfinite(saved_scores).all() or not np.allclose(saved_scores, scores[order], atol=ATOL, rtol=0):
                raise ValueError("Saved scores differ from sealed complete cache")
            score_error = max(score_error, float(np.max(np.abs(saved_scores - scores[order]))))
            metrics, relevant = independent_coverage(labels[query], labels[candidates], labels[ranked], ks)
            label_count, cls = int(labels[query].sum()), int(classes[query])
            row = expected[query]
            metadata = {"class_id": cls, "label_count": label_count, "pool_size": len(candidates),
                        "returned": len(ranked), "full_coverage_in_pool": relevant}
            if (any(int(row[key]) != value for key, value in metadata.items())
                    or row["label_signature"] != "".join(map(str, labels[query].astype(int).tolist()))):
                raise ValueError("Per-query labels/counts/full-pool denominator differs")
            if {key for key in row if "@" in key} != set(metrics):
                raise ValueError("Per-query metric names changed")
            for metric, value in metrics.items():
                metric_error = max(metric_error, _assert_value(row[metric], value, (query, metric)))
            perquery[query] = {"class": cls, "label_count": label_count, "relevant": relevant,
                               "metrics": {_metric_name(key): value for key, value in metrics.items()}}
    if seen != queries.tolist():
        raise ValueError("Ranking misses frozen queries")
    if (candidate_digest.hexdigest() != pool.metadata.get("candidate_identity_sha256")
            or min(counts) != pool.metadata.get("candidate_count_min")
            or max(counts) != pool.metadata.get("candidate_count_max")
            or pool.candidates_per_query != min(counts)):
        raise ValueError("Candidate identity digest/count range differs")
    if (sum(counts) != candidate_audit["full_candidates_checked"]
            or candidate_digest.hexdigest() != candidate_audit["candidate_identity_sha256"]):
        raise ValueError("Independent candidate audit differs from score replay")
    error, aggregated, metric_counts = _check_aggregates(folder, perquery, labels, ks)
    if (report["num_classes"] != candidate_audit["num_classes"]
            or report["queries_without_full_coverage_candidates"] != candidate_audit["queries_without_full_coverage_candidates"]
            or metric_counts != candidate_audit["metric_query_counts"]):
        raise ValueError("Complete candidate-pool effective query counts differ")
    return {"num_queries": len(seen), "ranking_rows": len(seen) * max(ks),
            "full_candidates_checked": sum(counts), "candidate_identity_sha256": candidate_digest.hexdigest(),
            "candidate_count_min": min(counts), "candidate_count_max": max(counts),
            "all_queries_full_candidate_topk_exact": True, "max_metric_absolute_error": max(error, metric_error),
            "max_saved_score_absolute_error": score_error, "aggregation_values_checked": aggregated,
            "metric_query_counts": {"coverage": metric_counts}}
