"""Independent coverage-only verification from the sealed full score cache.

Every eligible cached score participates in independent SHA Top-500 ordering.
This is not a second computation of regional scores and not an all-pair
independent regional-formula audit. The separate 740-pair audit checks regional
arithmetic and greedy matching; this module checks cached rankings and metrics.
"""
import csv
import hashlib
from itertools import groupby
import json
from pathlib import Path
import re

import numpy as np

from fgw_cross_test.verify import (
    ATOL, PROTOCOL, RULE, FORMAL_KS, REPORT_FILES,
    _json, _sha, _digest, independent_order, independent_coverage,
    _assert_value, _records, _metric_name,
)
from fgj_cross_test.verify import _check_aggregates


# Independent literals; do not import production state or regional scoring.
VARIANT = "add_a0p4_b0p25"
SPEC = {"family": "add", "alpha": .4, "beta": .25}
RANKING_FIELDS = ("query_id", "rank", "candidate_id", "similarity_score")
_METADATA_KEYS = ("signature", "context_signature", "variant", "spec", "split", "ks",
                  "tie_seed", "num_queries", "gallery_count", "candidate_rule",
                  "derived_protocol", "retrieval_protocol_signature", "provider_source",
                  "source_binding_sha256")
_FORMAL_CANDIDATE_HASH = "68c6875330ed70841586c654da776f8d0bc72bcd4cd341b79cea29705543f0b8"


def verify_evaluation(folder, pool, bundle, ks, signature, provider, source_binding, tie_seed=2026):
    """Return a receipt after all-query cached ordering and metric replay; no writes."""
    folder = Path(folder)
    if any((folder / name).exists() for name in ("metrics.json", "per_query.csv", "per_class.csv")):
        raise ValueError("Cross-combination evaluation must not emit exact reports")
    if (not isinstance(source_binding, dict) or not source_binding
            or any(not isinstance(path, str) or not Path(path).is_absolute()
                   or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha)
                   for path, sha in source_binding.items())):
        raise ValueError("Source binding must contain absolute paths and SHA-256 hashes")
    source_digest = _digest(source_binding)
    if (not ks or list(ks) != sorted(set(ks)) or any(type(k) is not int or k < 1 for k in ks)
            or type(tie_seed) is not int or tie_seed != 2026):
        raise ValueError("Invalid cutoffs or frozen tie seed")
    initial_hashes = {name: _sha(folder / name) for name in REPORT_FILES}
    manifest, identity = _json(folder / "evaluation_manifest.json"), _json(folder / "run_identity.json")
    provider_binding = {"signature": provider.signature, "completed_sha256": _sha(provider.marker)}
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
    context = manifest.get("context_signature")
    expected_signature = _digest({"context_signature": context, "variant": VARIANT, "spec": SPEC})
    if (not isinstance(context, str) or not context or signature != expected_signature
            or manifest.get("signature") != signature or manifest.get("variant") != VARIANT
            or manifest.get("spec") != SPEC or manifest.get("split") != "test"
            or pool.split_name != "test" or manifest.get("ks") != list(ks)
            or manifest.get("tie_seed") != tie_seed or manifest.get("num_queries") != len(queries)
            or manifest.get("gallery_count") != len(gallery)
            or manifest.get("provider_source") != provider_binding
            or not isinstance(provider.signature, str) or not provider.signature
            or not np.array_equal(provider.query_ids, queries) or not np.array_equal(provider.gallery_ids, gallery)
            or manifest.get("source_binding_sha256") != source_digest
            or manifest.get("candidate_rule") != RULE or pool.metadata.get("protocol_version") != PROTOCOL
            or pool.metadata.get("candidate_rule") != RULE or manifest.get("derived_protocol") != pool.metadata
            or manifest.get("retrieval_protocol_signature") != pool.protocol_signature
            or hashlib.sha256(json.dumps(pool.metadata, sort_keys=True).encode()).hexdigest() != pool.protocol_signature
            or pool.metadata.get("base_protocol_signature") != provider.protocol_signature):
        raise ValueError("Evaluation identity/frozen formula/cache/candidate protocol differs")
    if (identity.get("evaluation_signature") != signature
            or any(manifest.get(k) != identity.get(k) for k in ("run_id", "started_at", "variant", "split"))):
        raise ValueError("Evaluation time identity differs")
    formal = manifest.get("formal_protocol", True)
    if type(formal) is not bool:
        raise ValueError("Formal protocol flag must be boolean")
    if formal and (len(queries) != 7199 or len(gallery) != 7199
                   or set(map(int, queries)) != set(map(int, gallery)) or list(ks) != FORMAL_KS):
        raise ValueError("Formal evaluation requires all 7199 queries/gallery and seven frozen K")
    report = _json(folder / "coverage_metrics.json")
    if (report.get("split") != "test" or report.get("test_evaluated") is not True
            or report.get("ks") != list(ks) or report.get("metric_version") != "coverage_v1"
            or any(report.get("metadata", {}).get(key) != manifest[key] for key in _METADATA_KEYS)):
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
            scores = np.asarray(provider.scores(query, candidates))
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
    final_hashes = {name: _sha(folder / name) for name in REPORT_FILES}
    if (initial_hashes != final_hashes or _digest(source_binding) != source_digest
            or provider_binding != {"signature": provider.signature, "completed_sha256": _sha(provider.marker)}):
        raise ValueError("Reports, sealed cache marker or source binding changed during verification")
    return {"status": "verified", "signature": signature, "context_signature": context,
            "split": "test", "variant": VARIANT, "spec": dict(SPEC), "ks": list(ks), "tie_seed": tie_seed,
            "provider_source": provider_binding, "source_binding_sha256": source_digest,
            "source_file_count": len(source_binding), "num_queries": len(seen), "ranking_rows": len(seen) * max(ks),
            "full_candidates_checked": sum(counts), "candidate_identity_sha256": candidate_digest.hexdigest(),
            "candidate_count_min": min(counts), "candidate_count_max": max(counts),
            "all_queries_full_candidate_topk_exact": True, "max_metric_absolute_error": max(error, metric_error),
            "max_saved_score_absolute_error": score_error, "aggregation_values_checked": aggregated,
            "metric_query_counts": metric_counts, "exact_reports": False,
            "ranking_sha256": final_hashes["rankings.csv"], "files": final_hashes,
            "scope": "Every eligible score read from the newly sealed complete float64 cache; independent candidate identities, SHA TopK and all coverage_v1 query/class/macro/group values and denominator counts. Not a second regional score computation or all-pair independent regional formula audit; the separate 740-pair formula/greedy audit validates regional arithmetic."}
