#!/usr/bin/env python3
"""Run unchanged coverage metrics for the frozen valid top ten, excluding exact candidates."""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import argparse
import csv
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from experiments.proposed.f_segmentation_geometry import coverage_metrics
from fgw_grid_search import scoring
from fgw_grid_search.cache import FullScoreCache
from fgw_grid_search.pipeline import RANKING_FIELDS, source_binding
from fgw_topk_test.state import verify_run as verify_original
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from wbm_retrieval.exact_metrics import order_scores
from fgw_cross_test.state import (CANDIDATE_RULE, CROSS_SOURCE, PROTOCOL, REPORT_FILES, TOP_K, TZ,
    VERSION, checked_files, choose_output, cross_pool, evaluation_folder, evaluation_paths,
    exclusive, freeze_top10, identity, load_bundle, protected_inventory, read_signed,
    record, separate, signed, verify_run)

_WORK = None


class FilteredScoreProvider:
    """Read-only rows from a fully verified original cache, restricted before fusion."""
    def __init__(self, source, pool):
        if (source.protocol_signature != pool.metadata["base_protocol_signature"] or
                not np.array_equal(source.query_ids, np.array(list(pool), dtype=np.int64)) or
                not np.array_equal(source.gallery_ids, pool.gallery)):
            raise ValueError("Filtered pool does not derive from the original score cache")
        self.source, self.pool = source, pool
        for field in ("signature", "marker", "split", "protocol_signature", "query_ids", "gallery_ids"):
            setattr(self, field, getattr(source, field))

    def scores(self, query, candidate_ids):
        candidates = np.asarray(candidate_ids)
        if candidates.dtype.kind not in "iu" or not np.array_equal(candidates, self.pool[int(query)]):
            raise ValueError("Scores require the entire filtered candidate row in frozen order")
        columns = np.searchsorted(self.source.gallery_ids, candidates)
        if not np.array_equal(self.source.gallery_ids[columns], candidates):
            raise ValueError("Candidate source IDs differ")
        return {name: np.asarray(matrix[self.source.rows[int(query)], columns])
                for name, matrix in self.source.matrices.items()}


def code_identity():
    return {str(path.relative_to(ROOT)): file_hash(path)
            for path in sorted(Path(__file__).parent.glob("*.py"))}


def summary(folder, name, spec):
    report = read_json(Path(folder) / "coverage_metrics.json")
    return {"variant": name, "family": spec["family"], "split": "test",
            "num_queries": report["num_queries"],
            **report["metrics"], "metric_query_counts": report["metric_query_counts"]}


def _complete(folder, signature, manifest):
    folder = Path(folder)
    if not folder.exists():
        return False
    marker = read_json(folder / "completed.json")
    if (marker.get("status") != "completed" or marker.get("signature") != signature or
            read_json(folder / "evaluation_manifest.json") != manifest or
            set(marker.get("files", {})) != set(REPORT_FILES) | {"verification.json", "timing.json"}):
        raise ValueError("Existing evaluation differs; use a new output")
    checked_files(folder, marker["files"])
    checked = read_json(folder / "verification.json")
    if (checked.get("status") != "verified" or checked.get("signature") != signature or
            checked.get("files") != {name: file_hash(folder / name) for name in REPORT_FILES}):
        raise ValueError("Completed evaluation independent check is stale")
    return True


def _evaluate_one(item):
    from fgw_cross_test.verify import verify_evaluation
    name, spec = item
    output, signature, provider, pool, bundle, settings, formal = _WORK
    sig = digest({"parent_signature": signature, "variant": name, "spec": spec,
                  "protocol_signature": pool.protocol_signature})
    ident = identity(output, name, sig)
    manifest = {**ident, "signature": sig, "parent_signature": signature, "variant": name,
        "spec": spec, "settings": settings, "ks": settings["ks"], "tie_seed": settings["tie_seed"],
        "stage": "test", "split": "test", "protocol": PROTOCOL, "candidate_rule": CANDIDATE_RULE,
        "derived_protocol": pool.metadata, "retrieval_protocol_signature": pool.protocol_signature,
        "base_protocol_signature": pool.metadata["base_protocol_signature"],
        "provider_source": source_binding(provider), "num_queries": len(pool), "gallery_count": len(pool.gallery),
        "frozen_selection": {"path": str(output / "frozen_selection.json"),
                             "sha256": file_hash(output / "frozen_selection.json")},
        "formal_protocol": formal, "training_required": False, "posthoc_research": True,
        "test_parameter_search": False, "normalization_scope": "filtered_full_candidate_pool",
        "metrics": "unchanged coverage_v1", "ranking_scope": "top_max_k"}
    folder = evaluation_folder(output, name)
    if _complete(folder, sig, manifest):
        print("Reused verified cross evaluation:", name, flush=True)
        return summary(folder, name, spec)
    folder.parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="." + name + "_attempt_", dir=folder.parent))
    record(attempt / "run_identity.json", ident)
    record(attempt / "evaluation_manifest.json", manifest)
    started = time.perf_counter()
    print("Ranking filtered full pool:", name, flush=True)
    with (attempt / "rankings.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(RANKING_FIELDS)
        for number, query in enumerate(pool, 1):
            candidates = pool[query]
            components = provider.scores(query, candidates)
            values = scoring.score_variant(components, spec)
            order = order_scores(candidates, values, query, settings["tie_seed"])[:max(settings["ks"])]
            writer.writerows((query, rank, int(candidates[i]), float(values[i]), float(components["F"][i]),
                              float(components["G"][i]), float(components["WBBS"][i]))
                             for rank, i in enumerate(order, 1))
            if number % 1000 == 0:
                print(name, "ranked", number, "/", len(pool), flush=True)
    ranking_seconds = time.perf_counter() - started
    print("Original coverage metrics:", name, flush=True)
    started = time.perf_counter()
    metadata = {**manifest, "candidates_scored": "full_filtered_pool", "smoke": False}
    coverage_metrics.evaluate_rankings_file(bundle, pool, attempt / "rankings.csv", settings["ks"],
                                            attempt, split_name="test", metadata=metadata)
    metric_seconds = time.perf_counter() - started
    print("Independent full-pool verification:", name, flush=True)
    started = time.perf_counter()
    verification = verify_evaluation(attempt, pool, bundle, settings["ks"], sig, provider, settings["tie_seed"])
    record(attempt / "verification.json", verification)
    record(attempt / "timing.json", {"ranking_wall_seconds": ranking_seconds, "metrics_wall_seconds": metric_seconds,
        "verification_wall_seconds": time.perf_counter() - started, "source_score_cache_reused": True})
    record(attempt / "completed.json", {"status": "completed", "signature": sig,
        "files": {p: file_hash(attempt / p) for p in (*REPORT_FILES, "verification.json", "timing.json")}})
    if folder.exists():
        raise FileExistsError("Concurrent cross evaluation publication")
    attempt.rename(folder)
    print("Completed cross evaluation:", name, flush=True)
    return summary(folder, name, spec)


def evaluate_fixed(output, signature, provider, pool, bundle, specs, settings, *, workers=1, formal=False):
    """Evaluate an unchanged frozen prefix; formal callers enforce full identities."""
    global _WORK
    output = Path(output).resolve()
    settings = scoring.validate_settings(settings)
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("workers must be in 1..8")
    frozen = read_signed(output / "frozen_selection.json")
    if (frozen != freeze_top10(frozen["valid_selection"]) or specs != frozen["specs"] or
            list(specs) != frozen["ranked_variants"] or len(specs) != TOP_K or
            settings != frozen["valid_selection"]["settings"]):
        raise ValueError("Only the exact original valid top ten/settings are allowed")
    if (pool.split_name != "test" or provider.split != "test" or
            pool.metadata.get("protocol_version") != PROTOCOL or max(settings["ks"]) > pool.candidates_per_query):
        raise ValueError("Invalid derived cross test pool")
    if formal and ((len(pool), len(pool.gallery)) != (7199, 7199) or
                   (pool.metadata["candidate_count_min"], pool.metadata["candidate_count_max"]) != (6999, 7170)):
        raise ValueError("Formal test requires all 7,199 queries and frozen cross candidates")
    if workers > 1 and "fork" not in multiprocessing.get_all_start_methods():
        raise ValueError("Parallel cross test requires fork")
    binding = source_binding(provider)
    _WORK = (output, signature, provider, pool, bundle, settings, formal)
    try:
        if workers == 1:
            rows = [_evaluate_one(item) for item in specs.items()]
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as executor:
                rows = list(executor.map(_evaluate_one, specs.items()))
    finally:
        _WORK = None
    if source_binding(provider) != binding:
        raise ValueError("Original source cache completion changed")
    index = {"signature": signature, "stage": "test", "split": "test", "protocol": PROTOCOL,
        "ranked_variants": list(specs), "specs": specs, "settings": settings,
        "paths": evaluation_paths(output, specs), "provider_source": binding,
        "derived_protocol": pool.metadata, "summary_rows": rows,
        "completed_sha256": {name: file_hash(evaluation_folder(output, name) / "completed.json") for name in specs}}
    record(output / "evaluation_index_test.json", index)
    return index


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-root", help="Completed ordinary fixed-top-ten test, with its full component cache")
    choice = result.add_mutually_exclusive_group()
    choice.add_argument("--output-root")
    choice.add_argument("--resume-root")
    result.add_argument("--workers", type=int, default=8)
    result.add_argument("--dry-run", action="store_true", help="Inspect signed selection only; no evaluation/output")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if type(args.workers) is not int or not 1 <= args.workers <= 8:
        raise ValueError("workers must be in 1..8")
    started = datetime.now(TZ).replace(microsecond=0)
    output = choose_output(args.output_root, args.resume_root, now=started)
    previous = read_signed(output / "run_manifest.json") if args.resume_root else None
    source = args.source_root or (previous["context"]["source_root"] if previous else None)
    if not source:
        raise ValueError("New cross evaluation requires --source-root")
    source = (ROOT / Path(source).expanduser()).resolve()
    if ROOT / "artificial" not in source.parents:
        raise ValueError("Source must be a completed ordinary test in this repository")
    original = read_signed(source / "run_manifest.json")
    frozen = read_signed(source / "frozen_selection.json")
    if original.get("version") != "fgw_topk_test_v1" or frozen != freeze_top10(frozen["valid_selection"]):
        raise ValueError("Source is not the original frozen valid top-ten test")
    settings = scoring.validate_settings(original["context"]["settings"])
    plan = {"version": VERSION, "stage": "test", "split": "test", "protocol": PROTOCOL,
        "source_root": str(source), "source_signature": original["signature"],
        "source_completed_sha256": file_hash(source / "completed.json"),
        "selection_source": original["context"]["selection_source"], "settings": settings,
        "specs": frozen["specs"], "ranked_variants": frozen["ranked_variants"],
        "primary_variant": frozen["primary_variant"], "frozen_selection_signature": frozen["signature"],
        "evaluation_paths": evaluation_paths(output, frozen["ranked_variants"]),
        "training_required": False, "posthoc_research": True, "test_parameter_search": False,
        "formal_protocol": True, "change_scope": "candidate eligibility only; existing coverage_v1 metrics"}
    separate(output, [source, ROOT / "artifacts", ROOT / "protocol", ROOT / "results_catalog"])
    if args.dry_run:
        print(json.dumps({**plan, "output": str(output), "workers": args.workers,
                          "full_source_verification": "performed by actual run before scoring"}, ensure_ascii=False, indent=2))
        return 0
    import scipy
    import torch
    from fgw_grid_search.run import code_identity as grid_code_identity
    torch.set_num_threads(1)
    environment = {"numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__, "python": sys.version}
    if environment != original["context"]["environment"] or grid_code_identity() != original["context"]["grid_code"]:
        raise ValueError("Original frozen scoring code/numeric environment changed")
    print("Verifying completed original test and valid selection (read-only)...", flush=True)
    checked_original, checked_frozen = verify_original(source, check_cache=False)
    if checked_original != original or checked_frozen != frozen:
        raise ValueError("Original source changed during verification")
    print("Opening and hashing original full F/G/W cache (read-only)...", flush=True)
    cache = FullScoreCache(source / "component_cache/test")
    bundle, base_pool, labels_origin = load_bundle(source, cache)
    print("Constructing exact-excluded candidate pool...", flush=True)
    pool = cross_pool(base_pool, bundle.labels)
    if (len(pool), len(pool.gallery), pool.metadata["candidate_count_min"], pool.metadata["candidate_count_max"]) != (7199, 7199, 6999, 7170):
        raise ValueError("Derived candidate pool differs from frozen formal protocol")
    provider = FilteredScoreProvider(cache, pool)
    code = code_identity()
    context = {**plan, "code": code, "environment": environment, "labels_origin": labels_origin,
        "external_code": {str(CROSS_SOURCE): file_hash(CROSS_SOURCE)},
        "provider_source": source_binding(provider), "derived_protocol": pool.metadata,
        "base_protocol_signature": base_pool.protocol_signature,
        "retrieval_protocol_signature": pool.protocol_signature}
    if previous and (previous.get("version") != VERSION or previous["context"] != context):
        raise ValueError("Resume source/settings/code/environment changed")
    if previous and (output / "completed.json").exists():
        verify_run(output, check_source=False)
        print("Verified completed cross test:", output, flush=True)
        return 0
    with exclusive(output):
        if previous:
            manifest, before = previous, read_json(output / "source_protection_before.json")
            print("Rechecking protected sources before resume...", flush=True)
            if protected_inventory(source) != before:
                raise ValueError("Protected sources changed before resume")
        else:
            print("Recording read-only source protection inventory...", flush=True)
            before = protected_inventory(source)
            manifest = signed({"version": VERSION, "stage": "test", "context": context,
                "batch_id": started.strftime("%Y%m%dT%H%M%S%z") + "__fgw_cross_test",
                "started_at": started.isoformat(), "output": str(output), "workers": args.workers,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "launch_command": list(map(str, sys.argv if argv is None else [__file__, *argv]))})
            record(output / "source_protection_before.json", before)
            record(output / "run_manifest.json", manifest)
        record(output / "settings.json", settings)
        record(output / "frozen_selection.json", frozen)
        record(output / "candidate_protocol.json", pool.metadata)
        audit_path = output / "protocol/cross_combination_candidates.csv"
        if not audit_path.exists():
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_path.open("x", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(pool.audit[0]))
                writer.writeheader()
                writer.writerows(pool.audit)
        else:
            with audit_path.open(newline="") as handle:
                actual = [{key: int(value) for key, value in row.items()} for row in csv.DictReader(handle)]
            if actual != pool.audit:
                raise ValueError("Candidate audit changed on resume")
        try:
            write_json(output / "status.json", {"status": "running", "phase": "cross_test", "new_evaluations": TOP_K})
            evaluate_fixed(output, manifest["signature"], provider, pool, bundle, frozen["specs"], settings,
                           workers=args.workers, formal=True)
            checked_files(ROOT, code)
            if file_hash(CROSS_SOURCE) != context["external_code"][str(CROSS_SOURCE)]:
                raise ValueError("Existing candidate-rule code changed during run")
            print("Rechecking all protected original sources...", flush=True)
            after = protected_inventory(source)
            if before != after:
                raise ValueError("Protected source contents or timestamps changed")
            record(output / "source_protection_after.json", {"status": "unchanged", "files": len(after),
                                                              "inventory_digest": digest(after)})
            write_json(output / "status.json", {"status": "completed", "stage": "test",
                "test_evaluated": True, "new_evaluations": TOP_K, "protocol": PROTOCOL})
            files = {str(path.relative_to(output)): file_hash(path) for path in sorted(output.rglob("*"))
                     if path.is_file() and not any(part.startswith(".") for part in path.relative_to(output).parts)
                     and path.name not in ("completed.json", "rankings.csv") and path.suffix != ".log"}
            # Child completion markers bind large rankings; the parent binds those markers.
            files.update({str(evaluation_folder(output, name).relative_to(output) / "completed.json"):
                          file_hash(evaluation_folder(output, name) / "completed.json") for name in frozen["ranked_variants"]})
            record(output / "completed.json", {"status": "completed", "version": VERSION,
                "signature": manifest["signature"], "stage": "test", "new_evaluations": TOP_K, "files": files})
            verify_run(output, check_source=False)
            print("Completed and verified frozen top-ten cross coverage:", output, flush=True)
        except BaseException as exc:
            if not (output / "completed.json").exists():
                write_json(output / "status.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
