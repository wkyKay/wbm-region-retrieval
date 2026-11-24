"""Independent paper C1--C5 formula samples and complete-cache evaluation.

Every eligible float64 cached score is reranked and every exact/coverage metric
is independently recomputed. Regional scores themselves are independently
recomputed only for the explicitly reported deterministic sample. Matching is
separately checked on the production pair matrices after their formula check.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from f_region_structure import verify as ordinary
from f_region_evidence.verify import _verify_cross
from fgw_cross_test.verify import (
    ATOL, PROTOCOL, RULE, FORMAL_KS, REPORT_FILES as CROSS_REPORT_FILES,
    _json, _sha, _digest,
)

REPORT_FILES = (*ordinary.REPORT_FILES, "run_identity.json")
RAW_ATOL = 2e-6
_METADATA_KEYS = ("signature", "context_signature", "variant", "spec", "split", "stage",
                  "ks", "tie_seed", "num_queries", "gallery_count", "candidate_rule",
                  "retrieval_protocol_signature", "provider_source", "source_binding_sha256")


def independent_grid():
    """Independent literals; never import the production specification here."""
    return {name: {"variant": name,
                   "distribution": "independent" if name == "independent_j" else "joint",
                   "matching": "reuse_candidate" if name == "reuse_candidate" else "greedy_one_to_one"}
            for name in ("primary", "no_c", "no_t", "no_a", "independent_j", "reuse_candidate")}


def independent_distribution(joint):
    """Scalar p(rho) times p(stick,ball), retaining stick--ball dependence."""
    values = np.asarray(joint, dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] != 128 or not np.isfinite(values).all()
            or np.any(values < 0.) or not np.allclose(values.sum(1), 1., atol=1e-10, rtol=0)):
        raise ValueError("Invalid independent-J probability input")
    result = np.empty_like(values)
    for row, p in enumerate(values):
        radial = [math.fsum(float(p[16*r + s]) for s in range(16)) for r in range(8)]
        shape = [math.fsum(float(p[16*r + s]) for r in range(8)) for s in range(16)]
        for r in range(8):
            for s in range(16):
                result[row, 16*r + s] = radial[r] * shape[s]
    return result


def independent_components(cache, query_tokens, candidate_tokens):
    """Separate scalar float64 dot/log/root arithmetic, with no scoring helper."""
    h, p, a = (np.asarray(cache[key]) for key in ("features", "joint_all", "relative_area"))
    q, c = np.asarray(query_tokens), np.asarray(candidate_tokens)
    if (h.ndim != 2 or h.shape[1] != 512 or h.dtype.kind != "f"
            or p.shape != (len(h), 128) or a.shape != (len(h),)
            or any(x.ndim != 1 or x.dtype.kind not in "iu" or np.any(x < 0)
                   or np.any(x >= len(h)) or len(np.unique(x)) != len(x) for x in (q, c))):
        raise ValueError("Invalid independent regional formula inputs")
    selected = np.unique(np.r_[q, c]).astype(np.int64)
    if (not np.isfinite(h[selected]).all() or not np.isfinite(a[selected]).all()
            or np.any(a[selected] <= 0.) or np.any(a[selected] > 1.)):
        raise ValueError("Invalid independent regional feature/area input")
    norms = np.linalg.norm(h[selected].astype(np.float64), axis=1)
    if np.any((norms != 0) & ~np.isclose(norms, 1., rtol=1e-5, atol=1e-6)):
        raise ValueError("Independent cosine requires normalized or zero features")
    product = independent_distribution(p[selected])
    jroot = {int(t): [math.sqrt(float(x)) for x in p[t]] for t in selected}
    iroot = {int(t): [math.sqrt(float(x)) for x in product[i]] for i, t in enumerate(selected)}
    names = ("cosine", "joint_distance", "structure_similarity", "area_similarity",
             "independent_joint_distance", "independent_structure_similarity")
    result = {name: np.empty((len(q), len(c)), dtype=np.float64) for name in names}
    for row, qi in enumerate(q):
        for col, ci in enumerate(c):
            cosine = min(1., max(0., math.fsum(float(x)*float(y) for x, y in zip(h[qi], h[ci]))))
            distance = min(1., math.sqrt(.5*math.fsum((x-y)**2 for x, y in zip(jroot[int(qi)], jroot[int(ci)]))))
            independent = min(1., math.sqrt(.5*math.fsum((x-y)**2 for x, y in zip(iroot[int(qi)], iroot[int(ci)]))))
            area = math.exp(-abs(math.log(float(a[qi]))-math.log(float(a[ci])))/3.)
            for key, value in (("cosine", cosine), ("joint_distance", distance),
                               ("structure_similarity", math.exp(-.5*distance)), ("area_similarity", area),
                               ("independent_joint_distance", independent),
                               ("independent_structure_similarity", math.exp(-.5*independent))):
                result[key][row, col] = value
    return result


def independent_score_components(parts, spec):
    """Scalar fixed deletion/fusion formulas, separate from production scoring."""
    name = spec.get("variant") if isinstance(spec, dict) else None
    if name not in independent_grid() or spec != independent_grid()[name]:
        raise ValueError("Unknown fixed ablation specification")
    c, t, a = (np.asarray(parts[k], dtype=np.float64) for k in
               ("cosine", "independent_structure_similarity" if name == "independent_j" else "structure_similarity",
                "area_similarity"))
    if (c.ndim != 2 or any(x.shape != c.shape or not np.isfinite(x).all()
                          or np.any(x < 0.) or np.any(x > 1.) for x in (c, t, a))):
        raise ValueError("Invalid scalar fusion components")
    result = np.empty_like(c)
    for row in range(len(c)):
        for col in range(c.shape[1]):
            cv, tv, av = float(c[row, col]), float(t[row, col]), float(a[row, col])
            value = ((.51*tv+.15*av)/.66 if name == "no_c" else
                     (.34*cv+.15*av)/.49 if name == "no_t" else
                     .4*cv+.6*tv if name == "no_a" else
                     .85*(.4*cv+.6*tv)+.15*av)
            result[row, col] = min(1., max(0., value))
    return result


def independent_match(pair, area, reuse_candidate=False):
    """Scalar sorted-edge greedy or per-row maximum, using all query weights."""
    pair, area = np.asarray(pair), np.asarray(area, dtype=np.float64)
    if (pair.ndim != 2 or not np.isfinite(pair).all() or np.any(pair < 0.) or np.any(pair > 1.)
            or area.shape != (len(pair),) or not np.isfinite(area).all() or np.any(area <= 0.)
            or type(reuse_candidate) is not bool):
        raise ValueError("Invalid scalar matching inputs")
    weights = [math.sqrt(float(value)) for value in area]
    denominator = math.fsum(weights)
    if not denominator or not pair.shape[1]:
        return 0.
    if reuse_candidate:
        contributions = [weights[i]*max(map(float, pair[i])) for i in range(len(pair))]
    else:
        edges = sorted((-float(pair[i, j]), i, j) for i in range(len(pair)) for j in range(pair.shape[1]))
        usedq, usedc, contributions = set(), set(), []
        for negative, qi, ci in edges:
            if negative < 0. and qi not in usedq and ci not in usedc:
                usedq.add(qi)
                usedc.add(ci)
                contributions.append(-negative*weights[qi])
    return math.fsum(contributions)/denominator


def _token_layout(cache, gallery):
    ids = ordinary._ids(cache["map_ids"], "region map IDs")
    offsets = np.asarray(cache["offsets"])
    area = np.asarray(cache["area"])
    n = len(cache["features"])
    if (set(map(int, ids)) != set(map(int, gallery))
            or offsets.dtype.kind not in "iu" or offsets.shape != (len(ids)+1,)
            or offsets[0] != 0 or offsets[-1] != n
            or np.any(np.diff(offsets.astype(np.int64)) < 0)
            or np.any(np.diff(offsets.astype(np.int64)) > 4)
            or area.shape != (n,) or not np.isfinite(area).all() or np.any(area <= 0)
            or not np.array_equal(cache["owners"], np.repeat(ids, np.diff(offsets.astype(np.int64))))):
        raise ValueError("Independent Top-4 token ownership/layout differs")
    return {int(q): np.arange(offsets[i], offsets[i+1], dtype=np.int64) for i, q in enumerate(ids)}


def audit_formulas(provider, bundle, specs, max_candidates=20):
    """Sample each class's smallest query ID against first sorted candidates.

    The imported production pair matrix is the *object being checked*, first
    against independently derived scalar scores, then supplied to a separate
    scalar matcher to avoid false failures from float32 cosine tie changes.
    """
    from .scoring import score_components

    grid = independent_grid()
    if not specs or any(name not in grid or spec != grid[name] for name, spec in specs.items()):
        raise ValueError("Unknown fixed ablation specification")
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError("Invalid audit candidate count")
    representatives = {}
    for q in sorted(map(int, provider.pool)):
        representatives.setdefault(int(bundle.class_ids[q]), q)
    representatives = [representatives[k] for k in sorted(representatives)]
    pool = provider.pool
    formal = len(pool) in (500, 7199) or len(pool.gallery) in (3599, 7199)
    if formal and (len(pool) != (500 if pool.split_name == "valid" else 7199)
                   or len(pool.gallery) != (3599 if pool.split_name == "valid" else 7199)
                   or len(representatives) != 37 or max_candidates != 20 or specs != grid):
        raise ValueError("Formal formula audit requires all six variants and 37 classes times 20 candidates")
    map_tokens = _token_layout(provider.cache, pool.gallery)
    for q, expected in map_tokens.items():
        if not np.array_equal(provider.tokens(q), expected):
            raise ValueError("Provider token indices differ from independent map ownership")
    fields = ("max_raw_formula_absolute_error", "max_fusion_absolute_error", "max_matching_absolute_error")
    details = {name: {"query_ids": representatives, "class_count": len(representatives), "candidate_pairs": 0,
                      "candidate_ids_by_query": {}, **dict.fromkeys(fields, 0.)} for name in specs}
    for q in representatives:
        candidates = np.sort(np.asarray(provider.pool[q], dtype=np.int64))[:max_candidates]
        qt = map_tokens[q]
        token_sets = [map_tokens[int(c)] for c in candidates]
        ct = np.concatenate(token_sets) if token_sets else np.array([], dtype=np.int64)
        direct = independent_components(provider.cache, qt, ct)
        parts = provider.parts(q)
        component_error = 0.
        for key, expected in direct.items():
            actual = np.asarray(parts[key])[:, ct]
            error = float(np.max(np.abs(actual-expected), initial=0.))
            if not np.isfinite(actual).all() or error > RAW_ATOL:
                raise ValueError("Independent regional formula mismatch: "+key)
            component_error = max(component_error, error)
        for name, spec in specs.items():
            pair = np.asarray(score_components(parts, spec))[:, ct]
            expected_pair = independent_score_components(direct, spec)
            fusion_error = float(np.max(np.abs(pair-expected_pair), initial=0.))
            if not np.isfinite(pair).all() or fusion_error > RAW_ATOL:
                raise ValueError("Independent regional fusion formula mismatch: "+name)
            ids, scores = provider.score(q, spec, parts)
            ids, scores = np.asarray(ids), np.asarray(scores)
            if (not np.array_equal(ids, np.asarray(provider.pool[q])) or scores.shape != ids.shape
                    or not np.isfinite(scores).all() or np.any(scores < 0.) or np.any(scores > 1.)):
                raise ValueError("Sampled aggregation candidate identities/scores differ")
            lookup = dict(zip(map(int, ids), map(float, scores)))
            detail = details[name]
            detail["candidate_ids_by_query"][str(q)] = candidates.tolist()
            detail["max_raw_formula_absolute_error"] = max(detail["max_raw_formula_absolute_error"], component_error)
            detail["max_fusion_absolute_error"] = max(detail["max_fusion_absolute_error"], fusion_error)
            start = 0
            for candidate, tokens in zip(candidates, token_sets):
                expected = independent_match(pair[:, start:start+len(tokens)], provider.cache["area"][qt],
                                             reuse_candidate=spec["matching"] == "reuse_candidate")
                start += len(tokens)
                error = abs(expected-lookup[int(candidate)])
                if error > ATOL:
                    raise ValueError("Independent greedy/reuse-candidate aggregation mismatch: "+name)
                detail["max_matching_absolute_error"] = max(detail["max_matching_absolute_error"], error)
                detail["candidate_pairs"] += 1
    if formal and any(detail["candidate_pairs"] != 740 for detail in details.values()):
        raise ValueError("Formal formula audit must check 740 map pairs in every variant")
    return {"status": "verified", "formal_protocol": formal,
            "class_count": len(representatives), "query_ids": representatives,
            "max_candidates_per_query": max_candidates, "variants": details,
            "candidate_pairs": sum(d["candidate_pairs"] for d in details.values()),
            **{field: max(d[field] for d in details.values()) for field in fields},
            "formula_atol": RAW_ATOL, "matching_atol": ATOL,
            "scope": "For each class minimum query ID and first sorted candidate IDs: independent scalar float64 C/H/T/A, p(rho)*p(stick,ball), independent H/T and all six fusion formulas. Separate scalar greedy/reuse matching uses the checked production pair matrix to preserve float32 tie behavior, including every query sqrt-area weight in the denominator. This is a sampled formula audit, not an all-map-pair independent recomputation."}


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
    context = manifest.get("context_signature")
    split = "valid" if stage == "valid" else "test"
    cross = stage == "cross_test"
    protocol = PROTOCOL if cross else "retrieval_v1"
    expected_signature = _digest({"context": context, "stage": stage, "variant": variant, "spec": spec})
    queries, gallery = ordinary._ids(list(pool), "queries"), ordinary._ids(pool.gallery, "gallery")
    labels, classes = np.asarray(bundle.labels), np.asarray(bundle.class_ids)
    provider_binding = {"signature": provider.signature, "completed_sha256": _sha(provider.marker)}
    if (stage not in ("valid", "test", "cross_test") or variant not in grid or spec != grid[variant]
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
    cross = manifest["stage"] == "cross_test"
    files = CROSS_REPORT_FILES if cross else REPORT_FILES
    initial_hashes = {name: _sha(folder / name) for name in files}
    source_digest = _digest(source_binding)
    if cross:
        receipt = _verify_cross(folder, pool, labels, classes, queries, gallery, ks, provider, tie_seed, formal)
    else:
        seen = []
        candidate_digest = hashlib.sha256()
        def replay(q):
            if len(seen) >= len(queries) or q != int(queries[len(seen)]):
                raise ValueError("Ranking query order/identity differs")
            seen.append(int(q))
            candidates = np.asarray(pool[q], dtype=np.int64)
            if not np.array_equal(candidates, gallery[gallery != q]):
                raise ValueError("Ordinary candidate identities/order differ from complete gallery excluding self")
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
                   scope="Every eligible score read from the newly sealed complete float64 cache; independent candidate identities, SHA TopK and all query/class/macro/group values and full-pool denominators. Not a second regional score computation or all-pair independent formula audit; a separate sampled scalar regional-formula and greedy/reuse-candidate audit applies.")
    return receipt


