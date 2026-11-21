"""Read-only derived candidate pools and independent frozen-identity audits."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CROSS_SOURCE = ROOT.parent / "compared_methods/shared/cross_combination.py"
PROTOCOL = "coverage_cross_combination_v1"
RULE = "all_gallery_except_query_and_exact_label_combination"
BASE_PROTOCOL_SIGNATURE = "6a428fb8d8bdf34fb6c60852b6d9a7467bf6880cae8c802a6a9c5a36f96cb8a1"
KS = [1, 5, 10, 20, 50, 100, 500]
STEMS = ("Coverage-nDCG", "MeanCoverage", "FullCoverageHit", "FullCoveragePrecision",
         "FullCoverageRecall", "FullCoverage_mAP")
FROZEN = {
    "valid": {"num_queries": 500, "gallery_count": 3599, "candidate_pairs": 1750095,
        "candidate_count_min": 3499, "candidate_count_max": 3585,
        "candidate_identity_sha256": "e9e865ff288db79bf622f5016c17ac22d61d5ca071d39893a6202b9b0dd013b7",
        "derived_signature": "8cce967e52a329e26b5a970e0740f1da5c3cd814df8046acbd36a2f21df319ae",
        "query_ids_sha256": "7ad18576677a7b17f69f8400288a793e809ab7b81ebaee865398c27039f46b47",
        "gallery_ids_sha256": "127149b3d283aebcec92712a2f679e048db829cd140ff86bce13d9e4afeac6e7",
        "ndcg_valid": 484, "full_valid": 428},
    "test": {"num_queries": 7199, "gallery_count": 7199, "candidate_pairs": 50395860,
        "candidate_count_min": 6999, "candidate_count_max": 7170,
        "candidate_identity_sha256": "68c6875330ed70841586c654da776f8d0bc72bcd4cd341b79cea29705543f0b8",
        "derived_signature": "3aaf4895309e5d67c2aa1a196581a70237b0b72a8bc0c8f806df7b9947065bf0",
        "query_ids_sha256": "aadcaf330f605804ea4d2db52ddab0d83f56ead7cc4f4f901e38abc0e4e01eba",
        "gallery_ids_sha256": "aadcaf330f605804ea4d2db52ddab0d83f56ead7cc4f4f901e38abc0e4e01eba",
        "ndcg_valid": 7000, "full_valid": 6200},
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _ids(value, name):
    value = np.asarray(value)
    if (value.ndim != 1 or value.dtype.kind not in "iu" or not len(value)
            or np.any(value < 0) or len(np.unique(value)) != len(value)):
        raise ValueError("Expected unique nonnegative integer " + name)
    return value.astype(np.int64, copy=False)


def make_cross_pool(base, labels):
    """Use the shared eligibility rule without touching any frozen CSV."""
    spec = importlib.util.spec_from_file_location("regional_wbbs_shared_cross_protocol", CROSS_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.PROTOCOL_VERSION != PROTOCOL or module.CANDIDATE_RULE != RULE:
        raise ValueError("Shared cross-combination protocol definition changed")
    return module.CrossCombinationPool(base, labels)


def audit_pool(pool, labels, ks=KS, formal=True):
    """Independently rebuild all candidates and all metric validity denominators.

    This does not inspect or produce any score, ranking or experimental metric.
    Query and gallery order are part of the candidate hash and are not replaced.
    """
    if type(formal) is not bool or list(ks) != sorted(set(ks)) or not ks or any(type(k) is not int or k < 1 for k in ks):
        raise ValueError("Invalid formal flag or cutoffs")
    queries, gallery = _ids(list(pool), "query IDs"), _ids(pool.gallery, "gallery IDs")
    labels = np.asarray(labels)
    split = pool.split_name
    if (split not in FROZEN or not np.isin(queries, gallery).all()
            or labels.ndim != 2 or labels.shape[1] != 8 or max(gallery) >= len(labels)
            or not np.isin(labels[gallery], (0, 1)).all() or np.any(labels[gallery].sum(1) == 0)):
        raise ValueError("Invalid split or original-ID labels")
    metadata = getattr(pool, "metadata", None)
    cross = metadata is not None
    expected = FROZEN[split]
    query_hash, gallery_hash = _digest(queries.tolist()), _digest(gallery.tolist())
    base_signature = metadata.get("base_protocol_signature") if cross else pool.protocol_signature
    if formal and (base_signature != BASE_PROTOCOL_SIGNATURE or list(ks) != KS
                   or len(queries) != expected["num_queries"] or len(gallery) != expected["gallery_count"]
                   or query_hash != expected["query_ids_sha256"] or gallery_hash != expected["gallery_ids_sha256"]):
        raise ValueError("Formal query/gallery identities or base protocol differ")
    checksum = hashlib.sha256()
    counts, positives, overlap, label_counts, classes = [], [], [], [], []
    for query in queries:
        eligible = gallery[gallery != query]
        if cross:
            eligible = eligible[np.any(labels[eligible] != labels[query], axis=1)]
        actual = np.asarray(pool[int(query)])
        if actual.dtype.kind not in "iu" or not np.array_equal(actual, eligible) or len(actual) < max(ks):
            raise ValueError("Candidate pool differs from independent full-gallery eligibility")
        checksum.update(json.dumps([int(query), eligible.tolist()], separators=(",", ":")).encode("ascii") + b"\n")
        coverage = labels[eligible][:, labels[query].astype(bool)].sum(1)
        counts.append(len(eligible)); positives.append(int(np.sum(coverage == labels[query].sum())))
        overlap.append(int(np.count_nonzero(coverage))); label_counts.append(int(labels[query].sum()))
        classes.append(tuple(map(int, labels[query])))
    def summary(indices):
        full = sum(positives[i] > 0 for i in indices)
        ndcg = sum(overlap[i] > 0 for i in indices)
        return {"num_queries": len(indices), "num_classes": len({classes[i] for i in indices}),
            "queries_without_full_coverage_candidates": len(indices)-full,
            "queries_without_positive_coverage_candidates": len(indices)-ndcg,
            "metric_query_counts": {f"{stem}@{k}": ndcg if stem == "Coverage-nDCG" else
                full if stem in ("FullCoverageRecall", "FullCoverage_mAP") else len(indices)
                for stem in STEMS for k in ks}}
    indices = list(range(len(queries)))
    overall = summary(indices)
    groups = {"single_label": summary([i for i in indices if label_counts[i] == 1]),
              "mixed_label": summary([i for i in indices if label_counts[i] > 1]),
              "by_label_count": {str(n): summary([i for i in indices if label_counts[i] == n])
                                 for n in sorted(set(label_counts))}}
    if cross:
        independently_derived = {"protocol_version": PROTOCOL, "candidate_rule": RULE,
            "base_protocol_signature": base_signature, "candidate_identity_sha256": checksum.hexdigest(),
            "filter_uses_ground_truth": True,
            "filter_scope": "benchmark eligibility only; method scores never receive labels",
            "candidate_count_min": min(counts), "candidate_count_max": max(counts)}
        derived_signature = hashlib.sha256(json.dumps(independently_derived, sort_keys=True).encode()).hexdigest()
        if metadata != independently_derived or pool.protocol_signature != derived_signature:
            raise ValueError("Derived protocol hash or candidate metadata differs")
        if formal and (checksum.hexdigest() != expected["candidate_identity_sha256"]
                or sum(counts) != expected["candidate_pairs"]
                or (min(counts), max(counts)) != (expected["candidate_count_min"], expected["candidate_count_max"])
                or derived_signature != expected["derived_signature"]
                or overall["metric_query_counts"]["Coverage-nDCG@1"] != expected["ndcg_valid"]
                or overall["metric_query_counts"]["FullCoverage_mAP@1"] != expected["full_valid"]):
            raise ValueError("Formal cross-combination candidate identities or effective counts differ")
    elif pool.candidates_per_query != len(gallery)-1:
        raise ValueError("Ordinary candidates must cover the gallery except self")
    if pool.candidates_per_query != min(counts):
        raise ValueError("Candidate minimum differs from complete pool")
    return {"status": "verified", "split": split, "protocol": PROTOCOL if cross else "retrieval_v1",
            "retrieval_protocol_signature": pool.protocol_signature, "base_protocol_signature": base_signature,
            "num_queries": len(queries), "gallery_count": len(gallery), "num_classes": overall["num_classes"],
            "ks": list(ks), "candidate_rule": RULE if cross else "all_gallery_except_query",
            "query_ids_sha256": query_hash, "gallery_ids_sha256": gallery_hash,
            "candidate_identity_sha256": checksum.hexdigest(), "candidate_count_min": min(counts),
            "candidate_count_max": max(counts), "full_candidates_checked": sum(counts),
            "queries_without_full_coverage_candidates": overall["queries_without_full_coverage_candidates"],
            "queries_without_positive_coverage_candidates": overall["queries_without_positive_coverage_candidates"],
            "metric_query_counts": overall["metric_query_counts"], "groups": groups,
            "scope": "Independent full-gallery candidate identities, frozen query/gallery hashes and coverage denominator validity; no scoring or metric values computed."}
