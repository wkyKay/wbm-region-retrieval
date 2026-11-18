#!/usr/bin/env python3
"""Read-only full-gallery proof of J(lambda=1/2) ranking equivalence and AJ.

This is verification/postprocessing, never a new retrieval evaluation or a new
catalogue entry. Source formulas and source evaluation IDs remain unchanged.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
from itertools import groupby
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from b1_structure_fusion.scoring import StructuralDistances
from b1_ablation.verify import FORMULA_ATOL, independent_order
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from wbm_retrieval.exact_metrics import order_scores

TZ = ZoneInfo("Asia/Shanghai")
SOURCE_ROOT = "artificial/b1_structure_all_20261117T193244+0800"
AJ_ROOT = "artificial/b1_ablation_20261118T150805+0800"
OTHER_PROTECTED = ("artificial/b1_ablation_20261118T183806+0800",
                   "artificial/b1_ablation_20261118T190927+0800")
VERSION = "b1_structure_pair_verification_v1"
CSV_FIELDS = ["query_id", "candidate_count", "top_k", "J_source_topk_match",
              "J_lambda1_lambda2_topk_match", "AJ_source_topk_match", "J_saved_scores_match",
              "AJ_saved_scores_match", "J_score_max_abs_error", "AJ_score_max_abs_error", "status"]


class VerificationMismatch(ValueError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def _unit_vector(value, name, shape=None):
    vector = np.asarray(value, dtype=np.float64)
    if (vector.ndim != 1 or (shape is not None and vector.shape != shape)
            or not np.isfinite(vector).all() or np.any(vector < 0) or np.any(vector > 1)):
        raise ValueError("Expected finite aligned unit vector: "+name)
    return vector


def compute_scores(joint_distance, angular_distance):
    """Preserve the source NumPy float64 exp and multiplication order."""
    joint = _unit_vector(joint_distance, "joint_distance")
    angular = _unit_vector(angular_distance, "angular_distance", joint.shape)
    return {"J_lambda1": np.exp(-joint), "J_lambda2": np.exp(-2.*joint),
            "AJ": np.ones_like(angular) * np.exp(-.5*angular) * np.exp(-2.*joint)}


def angular_distance(query, candidates, descriptor_values):
    """Frozen outer-moment distance, without loading or scoring F/G features."""
    ids = np.asarray(descriptor_values["sample_ids"])
    if ids.ndim != 1 or len(set(map(int, ids))) != len(ids):
        raise ValueError("Descriptor IDs must be unique")
    rows = {int(sample): i for i, sample in enumerate(ids)}
    try:
        qi = rows[int(query)]
        ci = np.array([rows[int(c)] for c in candidates], dtype=np.int64)
    except KeyError as error:
        raise ValueError("Missing angular descriptor ID") from error
    moments = np.asarray(descriptor_values["moments"], dtype=np.float64)
    outer = np.asarray(descriptor_values["outer_count"])
    if (moments.shape != (len(ids), 4) or outer.shape != (len(ids),)
            or not np.isfinite(moments).all() or np.any(moments < 0) or np.any(moments > 1)
            or not np.issubdtype(outer.dtype, np.integer) or np.any(outer < 0)):
        raise ValueError("Invalid frozen angular descriptors")
    distance = np.abs(moments[ci]-moments[qi]).mean(axis=1)
    distance *= (outer[ci] > 0) & (outer[qi] > 0)
    return distance


def _ids(value, name):
    ids = np.asarray(value)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0):
        raise ValueError("Expected original integer IDs: "+name)
    return ids.astype(np.int64, copy=False)


def _saved(query, rows, top_k, candidate_set, label):
    if len(rows) != top_k:
        raise ValueError(f"{label} saved TopK length differs: query {query}")
    ids, scores = [], []
    for rank, row in enumerate(rows, 1):
        if int(row["query_id"]) != query or int(row["rank"]) != rank:
            raise ValueError(f"{label} query/rank sequence differs: query {query}")
        ids.append(int(row["candidate_id"]))
        scores.append(float(row["similarity_score"]))
    if len(set(ids)) != top_k or not set(ids) <= candidate_set:
        raise ValueError(f"{label} saved candidates are repeated or outside the full pool")
    return np.asarray(ids, dtype=np.int64), _unit_vector(scores, label+" saved scores")


def _changes(left_ids, right_ids, candidates, left_scores, right_scores):
    positions = {int(sample): i for i, sample in enumerate(candidates)}
    result = []
    for rank in np.flatnonzero(left_ids != right_ids):
        left, right = int(left_ids[rank]), int(right_ids[rank])
        li, ri = positions[left], positions[right]
        result.append({"rank": int(rank+1), "left_candidate_id": left, "right_candidate_id": right,
                       "left_candidate_left_formula_score": float(left_scores[li]),
                       "right_candidate_left_formula_score": float(left_scores[ri]),
                       "left_candidate_right_formula_score": float(right_scores[li]),
                       "right_candidate_right_formula_score": float(right_scores[ri])})
    return result


def check_query(query, candidates, gallery_ids, joint_distance, angular_distance_values,
                saved_j_rows, saved_aj_rows, *, top_k=500, tie_seed=2026):
    """Check all candidates before truncation; an identity/order mismatch is fatal."""
    query = int(query)
    candidates, gallery = _ids(candidates, "candidates"), _ids(gallery_ids, "gallery")
    candidate_set, gallery_set = set(map(int, candidates)), set(map(int, gallery))
    if (len(gallery_set) != len(gallery) or query not in gallery_set
            or len(candidates) != len(gallery)-1 or len(candidate_set) != len(candidates)
            or candidate_set != gallery_set-{query} or not 0 < top_k <= len(candidates)):
        raise VerificationMismatch("Candidates are not the complete gallery except self", {
            "query_id": query, "failure_kinds": ["full_candidate_identity_mismatch"]})
    try:
        j_ids, j_saved = _saved(query, saved_j_rows, top_k, candidate_set, "J_lambda1")
        aj_ids, aj_saved = _saved(query, saved_aj_rows, top_k, candidate_set, "AJ")
        scores = compute_scores(joint_distance, angular_distance_values)
        orders = {}
        for name, score in scores.items():
            score = _unit_vector(score, name, candidates.shape)
            production = order_scores(candidates, score, query, tie_seed)
            independent = independent_order(query, candidates, score, tie_seed)
            if not np.array_equal(production, independent):
                raise VerificationMismatch("Production and independent SHA ordering differ", {
                    "query_id": query, "formula": name, "failure_kinds": ["sort_implementation_mismatch"]})
            orders[name] = candidates[production[:top_k]]
    except VerificationMismatch:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise VerificationMismatch(str(error), {"query_id": query, "failure_kinds": ["invalid_source_or_score_fields"]}) from error
    positions = {int(sample): i for i, sample in enumerate(candidates)}
    j_recomputed = scores["J_lambda1"][[positions[int(c)] for c in j_ids]]
    aj_recomputed = scores["AJ"][[positions[int(c)] for c in aj_ids]]
    evidence = {"query_id": query, "candidate_count": len(candidates), "top_k": top_k,
                "J_source_topk_match": bool(np.array_equal(j_ids, orders["J_lambda1"])),
                "J_lambda1_lambda2_topk_match": bool(np.array_equal(orders["J_lambda1"], orders["J_lambda2"])),
                "AJ_source_topk_match": bool(np.array_equal(aj_ids, orders["AJ"])),
                "J_saved_scores_match": bool(np.allclose(j_saved, j_recomputed, atol=FORMULA_ATOL, rtol=FORMULA_ATOL)),
                "AJ_saved_scores_match": bool(np.allclose(aj_saved, aj_recomputed, atol=FORMULA_ATOL, rtol=FORMULA_ATOL)),
                "J_score_max_abs_error": float(np.max(np.abs(j_saved-j_recomputed))),
                "AJ_score_max_abs_error": float(np.max(np.abs(aj_saved-aj_recomputed)))}
    checks = {"J_source_topk_match": "J_source_topk_mismatch",
              "AJ_source_topk_match": "AJ_source_topk_mismatch",
              "J_saved_scores_match": "J_saved_score_mismatch",
              "AJ_saved_scores_match": "AJ_saved_score_mismatch",
              "J_lambda1_lambda2_topk_match": "lambda_topk_identity_changed"}
    evidence["failure_kinds"] = [reason for key, reason in checks.items() if not evidence[key]]
    if evidence["failure_kinds"]:
        evidence["changes"] = {
            "saved_J_vs_recomputed_J_lambda1": _changes(j_ids, orders["J_lambda1"], candidates, scores["J_lambda1"], scores["J_lambda1"]),
            "recomputed_J_lambda1_vs_lambda2": _changes(orders["J_lambda1"], orders["J_lambda2"], candidates, scores["J_lambda1"], scores["J_lambda2"]),
            "saved_AJ_vs_recomputed_AJ": _changes(aj_ids, orders["AJ"], candidates, scores["AJ"], scores["AJ"])}
        evidence["source_score_differences"] = {
            "J": [{"rank": int(i+1), "candidate_id": int(j_ids[i]), "saved": float(j_saved[i]), "recomputed": float(j_recomputed[i])}
                  for i in np.flatnonzero(~np.isclose(j_saved, j_recomputed, atol=FORMULA_ATOL, rtol=FORMULA_ATOL))],
            "AJ": [{"rank": int(i+1), "candidate_id": int(aj_ids[i]), "saved": float(aj_saved[i]), "recomputed": float(aj_recomputed[i])}
                   for i in np.flatnonzero(~np.isclose(aj_saved, aj_recomputed, atol=FORMULA_ATOL, rtol=FORMULA_ATOL))]}
        evidence["status"] = "mismatch"
        raise VerificationMismatch("Strict ranking/source verification failed for query "+str(query), evidence)
    evidence["status"] = "verified"
    return evidence


def _stream(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"query_id", "rank", "candidate_id", "similarity_score"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Missing source ranking fields: "+str(path))
        seen = set()
        for query, rows in groupby(reader, key=lambda row: int(row["query_id"])):
            if query in seen:
                raise ValueError("Repeated query group in source ranking: "+str(query))
            seen.add(query)
            yield query, list(rows)


def verify_rankings(pool, descriptor_values, j_ranking, aj_ranking, output, *, device="cuda",
                    top_k=500, tie_seed=2026, status_path=None):
    """Stream the entire frozen query pool; save lambda differences before refusing reuse."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = [output/name for name in ("per_query_checks.csv", "mismatches.jsonl", "summary.json", "failure.json")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Existing verification output cannot be overwritten")
    status_path = output/"status.json" if status_path is None else Path(status_path)
    queries = list(map(int, pool))
    if len(queries) != len(set(queries)) or len(queries) != len(pool):
        raise ValueError("Frozen query IDs must be complete and unique")
    gallery = _ids(pool.gallery, "gallery")
    if (pool.candidates_per_query != len(gallery)-1
            or not np.array_equal(np.sort(descriptor_values["sample_ids"]), np.sort(gallery))):
        raise ValueError("Descriptor IDs/candidate count differ from the frozen gallery")
    distances = StructuralDistances(descriptor_values["sample_ids"], descriptor_values["joint128"],
                                    descriptor_values["empty"], device=device)
    j_stream, aj_stream = _stream(j_ranking), _stream(aj_ranking)
    checked = mismatched_lambda = 0
    maxima = {"J_score_max_abs_error": 0., "AJ_score_max_abs_error": 0.}
    started = time.perf_counter()
    try:
        with paths[0].open("x", newline="", encoding="utf-8") as handle, paths[1].open("x", encoding="utf-8") as mismatch_file:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for query in queries:
                try:
                    jq, jrows = next(j_stream)
                    aq, arows = next(aj_stream)
                except StopIteration as error:
                    raise ValueError("Source ranking misses frozen queries") from error
                if jq != query or aq != query:
                    raise ValueError(f"Source query completeness/order differs: expected {query}, J={jq}, AJ={aq}")
                candidates = np.asarray(pool[query])
                joint = distances.for_query(query, candidates)["J"]
                angle = angular_distance(query, candidates, descriptor_values)
                try:
                    evidence = check_query(query, candidates, gallery, joint, angle, jrows, arows,
                                           top_k=top_k, tie_seed=tie_seed)
                except VerificationMismatch as error:
                    evidence = error.evidence
                    mismatch_file.write(json.dumps(evidence, ensure_ascii=False, allow_nan=False)+"\n")
                    mismatch_file.flush()
                    if evidence.get("failure_kinds") != ["lambda_topk_identity_changed"]:
                        if all(key in evidence for key in CSV_FIELDS):
                            writer.writerow({key: evidence[key] for key in CSV_FIELDS}); handle.flush()
                        raise
                    mismatched_lambda += 1
                writer.writerow({key: evidence[key] for key in CSV_FIELDS})
                checked += 1
                for key in maxima:
                    maxima[key] = max(maxima[key], evidence[key])
                if checked % 100 == 0 or checked == len(queries):
                    handle.flush()
                    write_json(status_path, {"status": "running", "phase": "full_gallery_ranking_verification",
                        "queries_checked": checked, "queries_total": len(queries), "lambda_mismatch_queries": mismatched_lambda,
                        "candidate_count": pool.candidates_per_query, "elapsed_seconds": time.perf_counter()-started})
                    print(f"Structure pair check: {checked}/{len(queries)} queries; lambda mismatches={mismatched_lambda}", flush=True)
            for name, iterator in (("J", j_stream), ("AJ", aj_stream)):
                if next(iterator, None) is not None:
                    raise ValueError("Unexpected extra source query groups: "+name)
        summary = {"status": "verified" if not mismatched_lambda else "strength_equivalence_failed",
                   "num_queries": checked, "candidates_per_query": pool.candidates_per_query, "top_k": top_k,
                   "source_J_lambda1_verified": True, "source_AJ_verified": True,
                   "J_lambda1_lambda2_topk_identical": not mismatched_lambda,
                   "lambda_mismatch_queries": mismatched_lambda, "requires_new_evaluation": bool(mismatched_lambda),
                   "reusable_max_k": top_k if not mismatched_lambda else None,
                   "verification_scope": "Every full-gallery candidate is rescored; exact identity/order equality is asserted only for TopK. No equality claim beyond TopK or across backends.",
                   "score_atol": FORMULA_ATOL, "score_rtol": FORMULA_ATOL,
                   "ranking_identity_tolerance": "none: exact candidate identity and order required", **maxima,
                   "ranking_source_sha256": {"J": file_hash(j_ranking), "AJ": file_hash(aj_ranking)},
                   "seconds": time.perf_counter()-started}
        write_json(output/"summary.json", summary)
        if mismatched_lambda:
            raise VerificationMismatch("Original J/AJ sources verified, but lambda=2 TopK differs; do not reuse J reports",
                                       {**summary, "failure_kinds": ["lambda_topk_identity_changed"],
                                        "changes_file": str(output/"mismatches.jsonl")})
        return summary
    except BaseException as error:
        evidence = getattr(error, "evidence", {"failure_kinds": ["source_query_or_runtime_failure"], "message": str(error)})
        strength_failure = evidence.get("failure_kinds") == ["lambda_topk_identity_changed"]
        classification = ("strength_equivalence_failed" if strength_failure else
                          "source_replay_failed" if isinstance(error, ValueError) else "verification_runtime_failed")
        evidence.setdefault("failure_classification", classification)
        evidence.setdefault("requires_new_evaluation", strength_failure)
        failure = {"status": "failed", "failure_classification": classification,
                   "requires_new_evaluation": strength_failure, "error_type": type(error).__name__, "error": str(error),
                   "queries_fully_checked": checked, "queries_total": len(queries), "evidence": evidence}
        write_json(output/"failure.json", failure)
        write_json(status_path, failure)
        raise
    finally:
        j_stream.close(); aj_stream.close()


def _path(value):
    return (ROOT/Path(value).expanduser()).resolve()


def _signed(value):
    return {**value, "signature": digest(value)}


def _ref(path):
    return {"path": str(Path(path).resolve()), "sha256": file_hash(path)}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", help="Fresh artificial/b1_ablation_structure_pair_check_<time> directory")
    p.add_argument("--device", choices=("cuda",), default="cuda")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    stamp = datetime.now(TZ).replace(microsecond=0)
    output = _path(args.output_root or ("artificial/b1_ablation_structure_pair_check_"+stamp.strftime("%Y%m%dT%H%M%S%z")))
    source, aj_root = _path(SOURCE_ROOT), _path(AJ_ROOT)
    protected = tuple(_path(p) for p in OTHER_PROTECTED)
    if ((ROOT/"artificial").resolve() not in output.parents
            or not output.name.startswith("b1_ablation_structure_pair_check_")):
        raise ValueError("Choose an isolated artificial/b1_ablation_structure_pair_check_<actual-time> output")
    from experiments.proposed.f_global_fusion.run import separate_output
    separate_output(output, [source, aj_root, *protected, ROOT/"protocol", ROOT/"results_catalog"])
    if output.exists():
        raise FileExistsError("Verification outputs are immutable; choose a new output")
    plan = {"version": VERSION, "operation": "verification_postprocessing", "creates_new_evaluation": False,
            "creates_catalog_entry": False, "split": "test", "query_count": 7199, "gallery_count": 7199,
            "candidates_per_query": 7198, "top_k": 500, "device": args.device, "tie_seed": 2026,
            "source_root": str(source), "AJ_root": str(aj_root), "protected_roots": list(map(str, protected)),
            "source_formulas": {"J": "exp(-D_J)", "AJ": "1*exp(-0.5*D_angle)*exp(-2*D_J)"},
            "requested_formulas": {"J": "exp(-2*D_J)", "AJ": "1*exp(-0.5*D_angle)*exp(-2*D_J)"},
            "reuse_gate": "All query Top500 J(lambda1)=J(lambda2)=saved J; recomputed AJ=saved AJ. No ranking tolerance.",
            "bootstrap": {"replicates": 10000, "seed": 2026, "strata": "37 complete label combinations", "anchor": "AJ"}}
    if args.dry_run:
        print(json.dumps({**plan, "output": str(output)}, ensure_ascii=False, indent=2)); return 0
    output.mkdir(parents=True, exist_ok=False)
    started_at = stamp.isoformat(timespec="seconds")
    verification_id = stamp.strftime("%Y%m%dT%H%M%S%z")+"__b1_structure_pair_check_test"
    write_json(output/"verification_identity.json", {"operation": "verification_postprocessing",
               "run_id": verification_id, "started_at": started_at, "creates_new_evaluation": False})
    write_json(output/"plan_before_verification.json", plan)
    write_json(output/"status.json", {"status": "running", "phase": "source_verification", "started_at": started_at})
    before = None
    try:
        import torch
        from b1_ablation import run, register
        from b1_ablation.analysis import postprocess
        from experiments.proposed.f_global_fusion.pipeline import signed_manifest, checked_files
        if not torch.cuda.is_available():
            raise RuntimeError("Original CUDA float64 structural scoring is required")
        for folder in (aj_root, *protected):
            if read_json(folder/"status.json").get("status") != "completed":
                raise ValueError("Protected ablation must be completed: "+str(folder))
            signed_manifest(folder/"run_manifest.json")
        before = run.inventory(source, aj_root, protected)
        write_json(output/"source_protection_before.json", before)
        bundle, contexts = run.load_inputs(source, args.device)
        register.proposal(aj_root)
        pool, _, descriptors, _, provenance = contexts["test"]
        if len(pool) != 7199 or len(pool.gallery) != 7199 or pool.candidates_per_query != 7198:
            raise ValueError("Expected full frozen 7199-query test protocol")
        query_ids = list(map(int, pool))
        if set(query_ids) != set(map(int, pool.gallery)):
            raise ValueError("Test queries must cover the complete gallery")
        j_folder, aj_folder = source/"test/J_only", aj_root/"test/structure_only"
        j_manifest = read_json(j_folder/"evaluation_manifest.json")
        aj_manifest = read_json(aj_folder/"evaluation_manifest.json")
        if j_manifest.get("spec") != {"family": "J_only", "strength": 1.}:
            raise ValueError("Original J source is not the frozen lambda=1 ranking")
        if aj_manifest.get("spec") != {"family": "J", "strength": 2., "local_mode": "none",
                                      "local_weight": 0., "global_weight": 0., "angular_gamma": .5}:
            raise ValueError("AJ source formula differs from the fixed structure-only recipe")
        for manifest in (j_manifest, aj_manifest):
            if (manifest.get("scoring_device") != "cuda" or manifest.get("tie_seed") != 2026
                    or manifest.get("retrieval_protocol_signature") != pool.protocol_signature):
                raise ValueError("Source device/tie/protocol differs")
        code = run.identity()
        source_manifest = signed_manifest(source/"run_manifest.json")
        inputs = [source/"run_manifest.json", source/"selection.json", source/"inputs_test.json",
                  source/"descriptors/test/completed.json", source/"descriptors/test/test.npz",
                  aj_root/"run_manifest.json", aj_root/"inputs_test.json",
                  j_folder/"completed.json", aj_folder/"completed.json"]
        source_identities = {"J": read_json(j_folder/"run_identity.json"),
                             "AJ": read_json(aj_folder/"run_identity.json")}
        manifest = _signed({"version": VERSION, "operation": "verification_postprocessing", "run_id": verification_id,
            "started_at": started_at, "output": str(output), "plan": plan, "source_files": [_ref(p) for p in inputs],
            "source_evaluation_identities": source_identities, "source_data_protocol": source_manifest["context"]["sources"],
            "retrieval_protocol_signature": pool.protocol_signature, "verified_input_provenance": provenance, "code": code,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "environment": {"numpy": np.__version__, "torch": torch.__version__, "python": sys.version,
                            "device": args.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                            "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda}})
        write_json(output/"run_manifest.json", manifest)
        verified = verify_rankings(pool, descriptors, j_folder/"rankings.csv", aj_folder/"rankings.csv",
                                   output/"ranking_check", device=args.device, top_k=500, tie_seed=2026,
                                   status_path=output/"status.json")
        reuse = {"operation": "report_reuse_after_strict_top500_verification", "verification": verified,
                 "creates_new_evaluation": False, "new_catalog_entries": 0,
                 "J_source_formula_unchanged": "exp(-D_J)", "J_requested_formula": "exp(-2*D_J)",
                 "identity_scope": "Exact Top500 equality for every query, supporting only reported K<=500 on this frozen test pool",
                 "AJ_source_formula_unchanged": "exp(-0.5*D_angle)*exp(-2*D_J)", "methods": {}}
        for name, folder in (("J", j_folder), ("AJ", aj_folder)):
            reuse["methods"][name] = {"source_directory": str(folder), "source_identity": source_identities[name],
                "reports": {family: {**_ref(folder/filename), "data": read_json(folder/filename)}
                            for family, filename in (("exact", "metrics.json"), ("coverage", "coverage_metrics.json"))}}
        write_json(output/"reused_reports.json", reuse)
        write_json(output/"status.json", {"status": "running", "phase": "direct_AJ_minus_J_postprocessing", "queries_checked": 7199})
        postprocess(output/"analysis/test", {"J": {"exact_dir": j_folder}, "AJ": {"exact_dir": aj_folder}},
                    bundle.labels, query_ids, list(map(int, pool.gallery)), split="test",
                    ks=[1, 5, 10, 20, 50, 100, 500], bootstrap_replicates=10000, seed=2026, anchor="AJ")
        checked_files(ROOT, code)
        after = run.inventory(source, aj_root, protected)
        if after != before:
            raise ValueError("Protected B1/component/double-deletion files changed")
        write_json(output/"source_protection_after.json", {"status": "unchanged", "files": len(after), "inventory_digest": digest(after)})
        write_json(output/"status.json", {"status": "completed", "operation": "verification_postprocessing",
                   "run_id": verification_id, "new_evaluations": 0, "new_catalog_entries": 0, "queries_checked": 7199})
        marker = output/"completed.json"
        files = {str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file() and p != marker}
        write_json(marker, {"status": "verified", "operation": "verification_postprocessing", "signature": manifest["signature"],
                   "run_id": verification_id, "started_at": started_at, "new_evaluations": 0, "new_catalog_entries": 0,
                   "source_evaluation_identities": source_identities, "query_count": 7199, "top_k": 500, "files": files})
        print("Verified J/AJ and completed direct comparison:", output, flush=True)
        return 0
    except BaseException as error:
        failure = {"status": "failed", "operation": "verification_postprocessing", "run_id": verification_id,
                   "started_at": started_at, "error_type": type(error).__name__, "error": str(error),
                   "evidence": getattr(error, "evidence", None),
                   "failure_classification": getattr(error, "evidence", {}).get("failure_classification", "verification_runtime_failed"),
                   "requires_new_evaluation": getattr(error, "evidence", {}).get("requires_new_evaluation", False),
                   "new_evaluations": 0, "new_catalog_entries": 0}
        if before is not None:
            try:
                from b1_ablation import run
                after = run.inventory(source, aj_root, protected)
                changed = [p for p in set(before)|set(after) if before.get(p) != after.get(p)]
                write_json(output/"source_protection_after.json", {"status": "unchanged" if not changed else "changed",
                    "files": len(after), "changed_paths": sorted(changed), "inventory_digest": digest(after)})
            except BaseException as protection_error:
                failure["protection_check_error"] = str(protection_error)
        write_json(output/"failure.json", failure)
        write_json(output/"status.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
