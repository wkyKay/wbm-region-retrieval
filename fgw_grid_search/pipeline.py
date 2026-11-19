"""Isolated, resumable full-gallery evaluation of fixed F/G/WBBS recipes."""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import csv
import math
import multiprocessing
from pathlib import Path
import re
import tempfile
import time
from zoneinfo import ZoneInfo

import numpy as np

from experiments.proposed.f_global_fusion import pipeline as fusion
from experiments.proposed.f_segmentation_geometry import pipeline as geo
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from wbm_retrieval.exact_metrics import evaluate_rankings, order_scores
from . import scoring, verify

TZ = ZoneInfo("Asia/Shanghai")
RANKING_FIELDS = ["query_id", "rank", "candidate_id", "similarity_score",
                  "optimal_f_score", "global_score", "wbbs_score"]
REPORT_FILES = ("run_identity.json", "evaluation_manifest.json", "rankings.csv",
                "metrics.json", "per_query.csv", "per_class.csv",
                "coverage_metrics.json", "coverage_per_query.csv", "coverage_per_class.csv")
_WORK = None


def record(path, value):
    """Create immutable JSON atomically; an identical resume never writes."""
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Evaluation identity/settings/source changed; use a new output: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="." + path.name + "_", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        write_json(temporary, value)
        # No replacement of a concurrently created identity or index.
        import os
        os.link(temporary, path)
    finally:
        temporary.unlink()


def source_binding(provider):
    if not isinstance(provider.signature, str) or not provider.signature:
        raise ValueError("A signed full-score provider is required")
    return {"signature": provider.signature, "completed_sha256": file_hash(Path(provider.marker))}


def variant_signature(signature, variant, spec, split, settings, stage, binding, protocol_signature):
    return digest({"evaluation": signature, "variant": variant, "spec": spec,
                   "split": split, "settings": settings, "stage": stage,
                   "provider_source": binding, "device": "cpu",
                   "retrieval_protocol_signature": protocol_signature})


def _identity(output, split, variant, signature):
    path = output / "identities" / f"{split}__{variant}.json"
    batch = re.sub(r"[^a-zA-Z0-9_]+", "_", output.name)
    key = f"fgw_grid_{batch}_{split}_{variant}"
    science = {"schema_version": 1, "key": key, "evaluation_signature": signature,
               "variant": variant, "split": split}
    if path.exists():
        value = read_json(path)
        if any(value.get(k) != v for k, v in science.items()):
            raise ValueError("Resume identity signature changed")
        stamp = value.get("started_at", "")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00", stamp):
            raise ValueError("Resume identity lacks original +08:00 timestamp")
        if value.get("run_id") != datetime.fromisoformat(stamp).strftime("%Y%m%dT%H%M%S%z") + "__" + key:
            raise ValueError("Resume run_id and started_at differ")
        return value
    if (output / split / variant).exists():
        raise ValueError("Completed evaluation lacks its original persistent identity")
    now = datetime.now(TZ).replace(microsecond=0)
    value = {**science, "run_id": now.strftime("%Y%m%dT%H%M%S%z") + "__" + key,
             "started_at": now.isoformat(timespec="seconds")}
    record(path, value)
    return value


def _candidate_vector(pool, query, gallery):
    candidates = np.asarray(pool[query])
    if (candidates.ndim != 1 or candidates.dtype.kind not in "iu" or
            len(candidates) != len(gallery) - 1 or len(np.unique(candidates)) != len(candidates) or
            set(map(int, candidates)) != gallery - {int(query)}):
        raise ValueError("Candidates must contain the complete gallery except self")
    return candidates.astype(np.int64, copy=False)


def _complete(folder, signature, expected_manifest):
    if not folder.exists():
        return False
    marker = read_json(folder / "completed.json")
    required = set(REPORT_FILES) | {"verification.json", "timing.json"}
    if marker.get("status") != "completed" or marker.get("signature") != signature:
        raise ValueError("Completed evaluation signature changed")
    if set(marker.get("files", {})) != required:
        raise ValueError("Completed evaluation does not bind every required report")
    for name, expected in marker["files"].items():
        if file_hash(folder / name) != expected:
            raise ValueError("Completed evaluation hash mismatch: " + name)
    if read_json(folder / "evaluation_manifest.json") != expected_manifest:
        raise ValueError("Completed evaluation manifest changed")
    checked = read_json(folder / "verification.json")
    if (checked.get("status") != "verified" or checked.get("signature") != signature or
            checked.get("provider_source") != expected_manifest["provider_source"] or
            checked.get("num_queries") != expected_manifest["num_queries"] or
            checked.get("ranking_rows") != expected_manifest["num_queries"] * max(expected_manifest["ks"]) or
            checked.get("files") != {name: file_hash(folder / name) for name in REPORT_FILES}):
        raise ValueError("Completed independent verification is missing or stale")
    return True


def _summary(folder, variant, spec, split):
    with (folder / "per_query.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    hits = 0
    for row in rows:
        value = float(row["Precision@10"]) * 10
        if not math.isfinite(value) or abs(value - round(value)) > 1e-10 or not 0 <= value <= 10:
            raise ValueError("Precision@10 is not an integer relevant-hit count")
        hits += int(round(value))
    exact = read_json(folder / "metrics.json")["metrics"]
    coverage = read_json(folder / "coverage_metrics.json")["metrics"]
    if not rows or not math.isclose(exact["Precision@10"], hits / (10 * len(rows)), abs_tol=1e-12, rel_tol=0):
        raise ValueError("Precision@10 aggregate differs from integer hits")
    return {"variant": variant, "family": spec["family"], "split": split,
            "num_queries": len(rows), "precision10_hits": hits,
            "Precision@10": exact["Precision@10"], "mAP@100": exact["mAP@100"],
            "Coverage-nDCG@10": coverage["Coverage-nDCG@10"]}


def _evaluate_variant(item):
    name, spec, sig, manifest = item
    output, provider, pool, bundle, settings = _WORK
    split = pool.split_name
    target = output / split / name
    # Assign actual per-variant launch time, never a queued/planned batch time.
    identity = _identity(output, split, name, sig)
    manifest = {**identity, **manifest}
    if _complete(target, sig, manifest):
        return _summary(target, name, spec, split)
    attempt = Path(tempfile.mkdtemp(prefix=f".{name}_attempt_", dir=output / split))
    record(attempt / "run_identity.json", identity)
    record(attempt / "evaluation_manifest.json", manifest)
    started = time.perf_counter()
    gallery = set(map(int, pool.gallery))
    with (attempt / "rankings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(RANKING_FIELDS)
        for query in pool:
            candidates = _candidate_vector(pool, query, gallery)
            values = provider.scores(int(query), candidates)
            scores = scoring.score_variant(values, spec)
            order = order_scores(candidates, scores, int(query), settings["tie_seed"])[:max(settings["ks"])]
            writer.writerows((int(query), rank, int(candidates[i]), float(scores[i]),
                              float(values["F"][i]), float(values["G"][i]), float(values["WBBS"][i]))
                             for rank, i in enumerate(order, 1))
    ranking_seconds = time.perf_counter() - started
    metadata = {"signature": sig, "run_id": identity["run_id"], "started_at": identity["started_at"],
                "variant": name, "spec": spec, "settings": settings,
                "source_signature": manifest["parent_signature"], "provider_source": manifest["provider_source"],
                "stage": manifest["stage"], "candidates_scored": "full_pool", "ranking_scope": "top_max_k",
                "retrieval_protocol_signature": pool.protocol_signature, "posthoc_research": True,
                "test_parameter_search": False}
    started = time.perf_counter()
    evaluate_rankings(bundle, pool, fusion.iter_rankings(attempt / "rankings.csv"), settings["ks"], attempt,
                      split_name=split, metadata=metadata)
    geo.evaluate_rankings_file(bundle, pool, attempt / "rankings.csv", settings["ks"], attempt,
                               split_name=split, metadata=metadata)
    metric_seconds = time.perf_counter() - started
    started = time.perf_counter()
    checked = verify.verify_evaluation(attempt, pool, bundle, settings["ks"], sig, provider, settings["tie_seed"])
    record(attempt / "verification.json", checked)
    record(attempt / "timing.json", {"ranking_wall_seconds": ranking_seconds,
           "metrics_wall_seconds": metric_seconds, "verification_wall_seconds": time.perf_counter() - started,
           "source_feature_and_cache_time_excluded": True})
    record(attempt / "completed.json", {"status": "completed", "signature": sig,
           "files": {name: file_hash(attempt / name) for name in (*REPORT_FILES, "verification.json", "timing.json")}})
    if target.exists():
        raise FileExistsError("Concurrent variant publication: " + str(target))
    attempt.rename(target)
    return _summary(target, name, spec, split)


def evaluate(output, signature, provider, pool, bundle, specs, settings, *, workers=1, stage="search"):
    """Evaluate a valid grid, or exactly one previously selected test recipe.

    The caller must independently bind/freeze test selection. This function only
    enforces its structural guard. Linux fork workers share read-only full-score
    matrices and each writes its own atomic variant attempt; only the parent
    publishes the split index. Failed attempts remain intact for inspection.
    """
    global _WORK
    settings = scoring.validate_settings(settings)
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError("workers must be an integer in [1,8]")
    split = getattr(pool, "split_name", None)
    if (stage not in ("search", "test") or split != ("valid" if stage == "search" else "test") or
            not isinstance(specs, dict) or not specs or (stage == "test" and len(specs) != 1)):
        raise ValueError("search requires valid; test requires test and exactly one frozen variant")
    variants = {}
    for name, spec in specs.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError("Unsafe or empty variant ID")
        variants[name] = scoring.validate_spec(spec)
    if not isinstance(signature, str) or not signature:
        raise ValueError("A signed parent evaluation identity is required")
    queries = list(pool)
    gallery = np.asarray(pool.gallery)
    if (gallery.ndim != 1 or gallery.dtype.kind not in "iu" or np.any(gallery < 0) or
            len(np.unique(gallery)) != len(gallery) or not queries or len(queries) != len(pool) or
            len(set(queries)) != len(queries) or any(isinstance(q, (bool, np.bool_)) or
            not isinstance(q, (int, np.integer)) for q in queries) or
            not set(map(int, queries)) <= set(map(int, gallery)) or
            pool.candidates_per_query != len(gallery) - 1 or max(settings["ks"]) > pool.candidates_per_query):
        raise ValueError("Invalid frozen full-gallery query/candidate identities")
    if (provider.split != split or provider.protocol_signature != pool.protocol_signature or
            not np.array_equal(provider.query_ids, queries) or not np.array_equal(provider.gallery_ids, gallery)):
        raise ValueError("Full-score provider and evaluation protocol differ")
    for query in queries:
        _candidate_vector(pool, query, set(map(int, gallery)))
    if workers > 1 and "fork" not in multiprocessing.get_all_start_methods():
        raise ValueError("Parallel evaluation requires the fork multiprocessing context")
    binding = source_binding(provider)
    output = Path(output).resolve()
    request = {"signature": signature, "split": split, "stage": stage, "variants": variants,
               "settings": settings, "provider_source": binding,
               "retrieval_protocol_signature": pool.protocol_signature,
               "query_ids_digest": digest(list(map(int, queries))), "gallery_ids_digest": digest(gallery.tolist())}
    record(output / f"evaluation_request_{split}.json", request)
    (output / split).mkdir(parents=True, exist_ok=True)
    signatures, items = {}, []
    for name, spec in variants.items():
        sig = variant_signature(signature, name, spec, split, settings, stage, binding, pool.protocol_signature)
        signatures[name] = sig
        manifest = {"signature": sig, "parent_signature": signature,
                    "spec": spec, "settings": settings, "ks": settings["ks"], "tie_seed": settings["tie_seed"],
                    "provider_source": binding, "stage": stage, "num_queries": len(pool),
                    "gallery_count": len(gallery), "candidate_rule": "all_gallery_except_query",
                    "retrieval_protocol_signature": pool.protocol_signature, "scoring_device": "cpu",
                    "training_required": False, "posthoc_research": True, "test_parameter_search": False}
        items.append((name, spec, sig, manifest))
    _WORK = (output, provider, pool, bundle, settings)
    try:
        if workers == 1:
            summaries = [_evaluate_variant(item) for item in items]
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as executor:
                summaries = list(executor.map(_evaluate_variant, items))
    finally:
        _WORK = None
    if source_binding(provider) != binding:
        raise ValueError("Full-score source marker changed during evaluation")
    index = {**request, "variant_signatures": signatures,
             "completed_sha256": {name: file_hash(output / split / name / "completed.json") for name in variants},
             "summary_rows": summaries}
    record(output / f"evaluation_index_{split}.json", index)
    return index
