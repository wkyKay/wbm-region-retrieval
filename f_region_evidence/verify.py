"""Independent evidence arithmetic, preserved tokens and sealed-cache evaluation.

The complete cached scores are replayed, not independently recomputed for all
map pairs. Regional arithmetic and background calibration have separate sampled
audits; ranking ties, candidate identities, metrics and denominators are exact.
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
from fgw_cross_test.verify import (
    ATOL, PROTOCOL, RULE, FORMAL_KS, REPORT_FILES as CROSS_REPORT_FILES,
    _json, _sha, _digest, independent_order, independent_coverage,
    _assert_value, _records, _metric_name,
)
from fgj_cross_test.verify import _check_aggregates

REPORT_FILES = (*ordinary.REPORT_FILES, "run_identity.json")
RANKING_FIELDS = ("query_id", "rank", "candidate_id", "similarity_score")
RAW_ATOL = 2e-6
_FORMAL_CANDIDATE_HASH = "68c6875330ed70841586c654da776f8d0bc72bcd4cd341b79cea29705543f0b8"
_METADATA_KEYS = ("signature", "context_signature", "variant", "spec", "split", "stage",
                  "ks", "tie_seed", "num_queries", "gallery_count", "candidate_rule",
                  "retrieval_protocol_signature", "provider_source", "source_binding_sha256")


def independent_grid():
    """Independent literals: deliberately do not import production state/scoring."""
    return {p + "_" + r: {"proposal": p, "rejection": r,
        "formula": "0.75*(0.4*C+0.6*exp(-0.5*H))+0.25*A",
        "matching": "original_greedy_query_sqrt_area",
        "rejection_formula": "identity" if r == "none" else "max((S-tau_i)/(1-tau_i),0); tau_i==1 gives zero",
        "reference_quantile": {"q75": .75, "q90": .9, "q95": .95}.get(r)}
        for p in ("base4", "top8", "residual8") for r in ("none", "q75", "q90", "q95")}


def _layout(cache):
    ids = ordinary._ids(cache["map_ids"], "cache map IDs")
    offsets = np.asarray(cache["offsets"])
    n = len(cache["features"])
    if (offsets.dtype.kind not in "iu" or offsets.shape != (len(ids) + 1,)
            or offsets[0] != 0 or offsets[-1] != n
            or np.any(np.diff(offsets.astype(np.int64)) < 0)
            or np.any(np.diff(offsets.astype(np.int64)) > 8)):
        raise ValueError("Invalid region cache offsets")
    expected = np.repeat(ids, np.diff(offsets.astype(np.int64)))
    if not np.array_equal(cache["owners"], expected):
        raise ValueError("Cache region owners differ from map offsets")
    return ids, offsets.astype(np.int64), {int(q): i for i, q in enumerate(ids)}


def audit_preserved_regions(original_cache, new_cache):
    """Check every old token in each map's unchanged prefix, including dtypes."""
    ids, offsets, positions = _layout(original_cache)
    newids, newoffsets, newpositions = _layout(new_cache)
    if set(ids.tolist()) != set(newids.tolist()):
        raise ValueError("Preserved cache map identities differ")
    mandatory = {"features", "area", "relative_area", "joint_all", "owners"}
    descriptor_fields = {"histograms", "joint", "shape", "shape_all", "original_count",
                         "retained_count", "quality", "empty", "empty_all", "radial_hist16"}
    fields = sorted(mandatory | (descriptor_fields & set(original_cache)))
    if not set(fields) <= set(new_cache):
        raise ValueError("New cache omits an original regional field")
    oldrows, newrows = [], []
    for q in ids:
        i, j = positions[int(q)], newpositions[int(q)]
        a, b = offsets[i:i+2]
        c, d = newoffsets[j:j+2]
        if b - a > 4 or d - c < b - a:
            raise ValueError("Original Top-4 regions were removed")
        oldrows.extend(range(a, b)); newrows.extend(range(c, c + b - a))
    oldrows, newrows = np.asarray(oldrows, dtype=np.int64), np.asarray(newrows, dtype=np.int64)
    if "base_token_indices" in new_cache and not np.array_equal(new_cache["base_token_indices"], newrows):
        raise ValueError("Preserved token-index mapping differs")
    if "base_region_counts" in new_cache:
        expected_counts = [offsets[positions[int(q)]+1]-offsets[positions[int(q)]] for q in newids]
        if not np.array_equal(new_cache["base_region_counts"], expected_counts):
            raise ValueError("Preserved per-map region counts differ")
    hashes = {}
    for field in fields:
        old, new = np.asarray(original_cache[field]), np.asarray(new_cache[field])
        if (old.shape[0] != offsets[-1] or new.shape[0] != newoffsets[-1]
                or old.dtype != new.dtype or old.shape[1:] != new.shape[1:]
                or old[oldrows].tobytes() != new[newrows].tobytes()):
            raise ValueError("Original region field changed: " + field)
        hashes[field] = hashlib.sha256(old[oldrows].tobytes()).hexdigest()
    return {"status": "verified", "num_maps": len(ids), "original_regions": len(oldrows),
            "new_regions": int(newoffsets[-1]), "all_original_regions_byte_identical": True,
            "preserved_fields": fields, "preserved_sha256": hashes,
            "scope": "All original region feature/geometry/descriptor bytes and dtypes, in unchanged per-map prefix order; mask pixels have a separate proposal provenance audit."}


def independent_components(cache, query_tokens, candidate_tokens, reference_cache=None):
    """Scalar float64 C/H/T/A/S, without production feature/scoring helpers."""
    other = cache if reference_cache is None else reference_cache
    q, c = np.asarray(query_tokens, dtype=np.int64), np.asarray(candidate_tokens, dtype=np.int64)
    output = {name: np.empty((len(q), len(c)), dtype=np.float64)
              for name in ("cosine", "joint_distance", "structure_similarity", "area_similarity", "raw_score")}
    for cache_ in (cache, other):
        h, p, a = (np.asarray(cache_[key]) for key in ("features", "joint_all", "relative_area"))
        if (h.ndim != 2 or h.shape[1] != 512 or p.shape != (len(h), 128)
                or a.shape != (len(h),) or not np.isfinite(h).all() or not np.isfinite(p).all()
                or np.any(p < 0) or not np.allclose(p.sum(1), 1., atol=1e-6, rtol=0)
                or not np.isfinite(a).all() or np.any(a <= 0) or np.any(a > 1)):
            raise ValueError("Invalid independent regional formula inputs")
    for row, qi in enumerate(q):
        hq = np.asarray(cache["features"][qi], dtype=np.float64)
        pq = np.sqrt(np.asarray(cache["joint_all"][qi], dtype=np.float64))
        aq = float(cache["relative_area"][qi])
        for col, ci in enumerate(c):
            hc = np.asarray(other["features"][ci], dtype=np.float64)
            pc = np.sqrt(np.asarray(other["joint_all"][ci], dtype=np.float64))
            cosine = min(1., max(0., float(np.dot(hq, hc))))
            distance = min(1., math.sqrt(.5 * math.fsum(float(x) ** 2 for x in pq - pc)))
            structure = math.exp(-.5 * distance)
            area = math.exp(-abs(math.log(aq) - math.log(float(other["relative_area"][ci]))) / 3.)
            for key, value in (("cosine", cosine), ("joint_distance", distance),
                               ("structure_similarity", structure), ("area_similarity", area),
                               ("raw_score", .75 * (.4 * cosine + .6 * structure) + .25 * area)):
                output[key][row, col] = value
    return output


def independent_evidence(raw, tau=None):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw > 1):
        raise ValueError("Invalid raw evidence matrix")
    if tau is None:
        return raw.copy()
    tau = np.asarray(tau, dtype=np.float64)
    if tau.shape != (len(raw),) or not np.isfinite(tau).all() or np.any(tau < 0) or np.any(tau > 1):
        raise ValueError("Invalid region background thresholds")
    result = np.empty_like(raw)
    for i in range(len(raw)):
        for j in range(raw.shape[1]):
            result[i, j] = 0. if tau[i] == 1. else max((float(raw[i, j]) - float(tau[i])) / (1. - float(tau[i])), 0.)
    return result


def independent_greedy(pair, area):
    pair, area = np.asarray(pair), np.asarray(area, dtype=np.float64)
    if pair.ndim != 2 or area.shape != (len(pair),) or not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError("Invalid scalar greedy inputs")
    weights = [math.sqrt(float(value)) for value in area]
    denominator = math.fsum(weights)
    usedq, usedc, contributions = set(), set(), []
    for value, qi, ci in sorted(((float(pair[i, j]), i, j)
                                for i in range(len(pair)) for j in range(pair.shape[1])),
                               key=lambda item: (-item[0], item[1], item[2])):
        if value > 0 and qi not in usedq and ci not in usedc:
            usedq.add(qi); usedc.add(ci); contributions.append(value * weights[qi])
    return math.fsum(contributions) / denominator if denominator else 0.


def _representatives(pool, bundle):
    result = {}
    for q in sorted(map(int, pool)):
        result.setdefault(int(bundle.class_ids[q]), q)
    return [result[key] for key in sorted(result)]


def audit_formulas(providers, bundle, specs, max_candidates=20):
    """Sample all requested variants, independent raw formulas and scalar matching."""
    grid = independent_grid()
    if not specs or any(name not in grid or spec != grid[name] for name, spec in specs.items()):
        raise ValueError("Unknown fixed evidence specification")
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError("Invalid audit candidate count")
    result = {"status": "verified", "variants": {}, "candidate_pairs": 0,
              "max_raw_formula_absolute_error": 0., "max_matching_absolute_error": 0.}
    for proposal in sorted({spec["proposal"] for spec in specs.values()}):
        provider = providers[proposal]
        active = {name: spec for name, spec in specs.items() if spec["proposal"] == proposal}
        representatives = _representatives(provider.pool, bundle)
        details = {name: {"query_ids": representatives, "candidate_pairs": 0,
                          "max_raw_formula_absolute_error": 0., "max_matching_absolute_error": 0.}
                   for name in active}
        for q in representatives:
            candidates = np.sort(np.asarray(provider.pool[q], dtype=np.int64))[:max_candidates]
            qt = provider.tokens(q)
            token_sets = [provider.tokens(c) for c in candidates]
            ct = np.concatenate(token_sets) if token_sets else np.array([], dtype=np.int64)
            direct = independent_components(provider.cache, qt, ct)
            parts = provider.parts(q)
            production_raw = .75 * (.4 * parts["cosine"] + .6 * parts["structure_similarity"]) + .25 * parts["area_similarity"]
            raw_error = 0.
            for key in direct:
                actual = production_raw[:, ct] if key == "raw_score" else np.asarray(parts[key])[:, ct]
                error = float(np.max(np.abs(actual - direct[key]), initial=0.))
                raw_error = max(raw_error, error)
                if error > RAW_ATOL:
                    raise ValueError("Independent regional formula mismatch: " + key)
            for name, spec in active.items():
                rejection = spec["rejection"]
                tau = None if rejection == "none" else provider.thresholds[rejection][qt]
                evidence = independent_evidence(production_raw[:, ct], tau)
                ids, scores = provider.score(q, rejection, parts)
                lookup = dict(zip(map(int, ids), map(float, scores)))
                start = 0
                for candidate, tokens in zip(candidates, token_sets):
                    expected = independent_greedy(evidence[:, start:start+len(tokens)], provider.cache["area"][qt])
                    start += len(tokens)
                    error = abs(expected - lookup[int(candidate)])
                    if error > ATOL:
                        raise ValueError("Independent rejection/greedy aggregation mismatch")
                    details[name]["max_matching_absolute_error"] = max(details[name]["max_matching_absolute_error"], error)
                    details[name]["candidate_pairs"] += 1
                details[name]["max_raw_formula_absolute_error"] = max(details[name]["max_raw_formula_absolute_error"], raw_error)
        for name, detail in details.items():
            result["variants"][name] = detail
            result["candidate_pairs"] += detail["candidate_pairs"]
            for field in ("max_raw_formula_absolute_error", "max_matching_absolute_error"):
                result[field] = max(result[field], detail[field])
    result["scope"] = "Each class minimum query ID and first sorted candidate IDs: independent float64 C/H/T/A/raw-score formulas; scalar rejection and greedy on the same production raw-score matrix to preserve float32 tie behavior. All query sqrt-area weights remain in the denominator. Not an all-pair independent formula computation."
    return result


def audit_calibration(providers, bundle, train_cache, reference):
    """Rebuild train selection and scalar quantiles for one region per class."""
    ids, offsets, positions = _layout(train_cache)
    if np.any(np.diff(offsets) > 4):
        raise ValueError("Background reference must contain original train Top-4")
    size, seed = reference.get("size"), reference.get("seed")
    if type(size) is not int or size < 1 or type(seed) is not int or seed != 2026:
        raise ValueError("Invalid frozen background reference size/seed")
    eligible = sorted(int(q) for q in ids if offsets[positions[int(q)] + 1] > offsets[positions[int(q)]])
    if size > len(eligible):
        raise ValueError("Background reference exceeds distinct nonempty train wafers")
    rng = np.random.default_rng(seed)
    owners = np.sort(rng.choice(np.asarray(eligible, dtype=np.int64), size=size, replace=False))
    local = np.asarray([rng.integers(int(offsets[positions[int(q)]+1]-offsets[positions[int(q)]]))
                        for q in owners], dtype=np.int64)
    tokens = np.asarray([offsets[positions[int(q)]] for q in owners], dtype=np.int64) + local
    def ih(value):
        return hashlib.sha256(np.asarray(value, dtype="<i8").tobytes()).hexdigest()
    expected = {"token_indices": tokens, "owner_ids": owners, "local_indices": local,
                "size": size, "seed": seed,
                "rule": "sorted_train_owners_sample_without_replacement_then_one_uniform_region_per_owner",
                "train_map_ids_sha256": ih(ids), "train_offsets_sha256": ih(offsets),
                "reference_identity_sha256": ih(np.column_stack((owners, local, tokens)))}
    if set(reference) != set(expected):
        raise ValueError("Background reference fields differ")
    for key, value in expected.items():
        if isinstance(value, np.ndarray):
            actual = np.asarray(reference[key])
            same = actual.dtype.kind in "iu" and np.array_equal(actual, value)
        else:
            same = reference[key] == value
        if not same:
            raise ValueError("Background reference selection identity differs: " + key)
    result = {"status": "verified", "reference_size": size,
              "reference_identity_sha256": expected["reference_identity_sha256"],
              "train_query_disjoint": True, "proposals": {}, "max_threshold_absolute_error": 0.}
    for name, provider in providers.items():
        if np.intersect1d(provider.cache["map_ids"], ids).size:
            raise ValueError("Calibration query maps overlap train reference population")
        qt, queries = [], []
        for q in _representatives(provider.pool, bundle):
            actual = provider.tokens(q)
            if len(actual):
                queries.append(q); qt.append(int(actual[0]))
        raw = independent_components(provider.cache, qt, tokens, train_cache)["raw_score"]
        detail = {"query_ids": queries, "query_tokens": qt, "regions_checked": len(qt),
                  "region_reference_pairs": len(qt) * size, "max_threshold_absolute_error": 0.}
        for rejection, quantile in (("q75", .75), ("q90", .90), ("q95", .95)):
            actual = np.asarray(provider.thresholds[rejection])
            if (actual.shape != (len(provider.cache["features"]),) or not np.isfinite(actual).all()
                    or np.any(actual < 0) or np.any(actual > 1)):
                raise ValueError("Threshold array does not cover complete real-token cache")
            for i, token in enumerate(qt):
                values = sorted(map(float, raw[i]))
                at = (len(values) - 1) * quantile
                low, high = math.floor(at), math.ceil(at)
                direct = values[low] + (at - low) * (values[high] - values[low])
                error = abs(float(actual[token]) - direct)
                if error > RAW_ATOL:
                    raise ValueError("Independent train-reference linear quantile differs")
                detail["max_threshold_absolute_error"] = max(detail["max_threshold_absolute_error"], error)
        result["proposals"][name] = detail
        result["max_threshold_absolute_error"] = max(result["max_threshold_absolute_error"], detail["max_threshold_absolute_error"])
    result["scope"] = "Deterministic train-only distinct-owner reference identity; independent float64 raw regional scores against every reference token and scalar sorted linear quantiles for the first nonempty region of each class representative. Threshold error allows source float32 cosine arithmetic."
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
                   scope="Every eligible score read from the newly sealed complete float64 cache; independent candidate identities, SHA TopK and all query/class/macro/group values and full-pool denominators. Not a second regional score computation or all-pair independent formula audit; separate sampled regional-formula, rejection/greedy and train-reference quantile audits apply.")
    return receipt


def _verify_cross(folder, pool, labels, classes, queries, gallery, ks, provider, tie_seed, formal):
    report = _json(folder / "coverage_metrics.json")
    if (report.get("split") != "test" or report.get("test_evaluated") is not True
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
    if formal and (sum(counts) != 50395860 or (min(counts), max(counts)) != (6999, 7170)
                   or candidate_digest.hexdigest() != _FORMAL_CANDIDATE_HASH):
        raise ValueError("Formal cross-combination complete candidate identities differ")
    error, aggregated, metric_counts = _check_aggregates(folder, perquery, labels, ks)
    if formal:
        if report["num_classes"] != 37 or report["queries_without_full_coverage_candidates"] != 999:
            raise ValueError("Formal class/no-full-coverage counts differ")
        for metric, count in metric_counts.items():
            actual = 7000 if metric.startswith("Coverage-nDCG@") else (6200 if metric.startswith(("FullCoverageRecall@", "FullCoverage_mAP@")) else 7199)
            if count != actual:
                raise ValueError("Formal effective query counts differ")
    return {"num_queries": len(seen), "ranking_rows": len(seen) * max(ks),
            "full_candidates_checked": sum(counts), "candidate_identity_sha256": candidate_digest.hexdigest(),
            "candidate_count_min": min(counts), "candidate_count_max": max(counts),
            "all_queries_full_candidate_topk_exact": True, "max_metric_absolute_error": max(error, metric_error),
            "max_saved_score_absolute_error": score_error, "aggregation_values_checked": aggregated,
            "metric_query_counts": {"coverage": metric_counts}}
