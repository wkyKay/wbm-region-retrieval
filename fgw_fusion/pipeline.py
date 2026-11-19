"""Full-gallery fixed FGW evaluation with immutable resumable Top-K shards."""
from contextlib import ExitStack
from datetime import datetime
import csv
from pathlib import Path
import re
import tempfile
import time
from zoneinfo import ZoneInfo

import numpy as np

from b1_structure_fusion import reporting
from experiments.proposed.f_global_fusion import pipeline as fusion
from experiments.proposed.f_segmentation_geometry import pipeline as geo
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from wbm_retrieval.exact_metrics import evaluate_rankings, order_scores
from . import scoring

TZ = ZoneInfo("Asia/Shanghai")
RANKING_FIELDS = ["query_id", "rank", "candidate_id", "similarity_score",
                  "optimal_f_score", "global_score", "wbbs_score"]


def record(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Immutable FGW metadata changed; use a new output: " + str(path))
    else:
        write_json(path, value)


def _identity(output, split, name, signature):
    path = Path(output) / "identities" / f"{split}__{name}.json"
    batch = re.sub(r"[^a-zA-Z0-9_]+", "_", Path(output).name)
    key = f"fgw_{batch}_{split}_{name}"
    science = {"schema_version": 1, "key": key, "evaluation_signature": signature,
               "variant": name, "split": split}
    if path.exists():
        value = read_json(path)
        if any(value.get(k) != v for k, v in science.items()):
            raise ValueError("Resume identity signature changed")
        started_at = value.get("started_at")
        if not isinstance(started_at, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00", started_at):
            raise ValueError("Resume identity requires original started_at in +08:00")
        started = datetime.fromisoformat(started_at)
        if value.get("run_id") != started.strftime("%Y%m%dT%H%M%S%z") + "__" + key:
            raise ValueError("Resume run_id disagrees with started_at")
        return value
    now = datetime.now(TZ).replace(microsecond=0)
    value = {**science, "run_id": now.strftime("%Y%m%dT%H%M%S%z") + "__" + key,
             "started_at": now.isoformat(timespec="seconds")}
    record(path, value)
    return value


def _status(output, split, phase, **details):
    write_json(Path(output)/"status.json", {"status": "running", "split": split,
               "phase": phase, "test_evaluated": split == "test", **details})


def _candidate_vector(pool, query, gallery_set):
    candidates = np.asarray(pool[query])
    if (candidates.ndim != 1 or candidates.dtype.kind not in "iu" or int(query) not in gallery_set
            or len(candidates) != pool.candidates_per_query
            or len(set(map(int, candidates))) != len(candidates)
            or set(map(int, candidates)) != gallery_set-{int(query)}):
        raise ValueError("Candidate pool must contain the entire gallery except self")
    return candidates.astype(np.int64, copy=False)


def _load_shard(folder, metadata, specs, candidates, tie_seed):
    marker = read_json(folder/"completed.json")
    if marker.get("metadata") != metadata or marker.get("signature") != digest(metadata):
        raise ValueError("Query shard identity or source signature changed")
    if marker.get("files") != {"scores.npz": file_hash(folder/"scores.npz")}:
        raise ValueError("Query shard file hash mismatch")
    with np.load(folder/"scores.npz", allow_pickle=False) as archive:
        arrays = dict(archive)
    fields = ("ids", "scores", "f", "g", "w")
    if set(arrays) != {name+"_"+key for name in specs for key in fields}:
        raise ValueError("Query shard fields differ from the fixed matrix")
    for name, spec in specs.items():
        ids = arrays[name+"_ids"]
        if (ids.dtype.kind not in "iu" or ids.shape != (metadata["top_k"],)
                or len(set(map(int, ids))) != len(ids) or not np.isin(ids, candidates).all()):
            raise ValueError("Query shard has invalid Top-K candidate identities")
        local, glob, wbbs, scores = (scoring.unit_vector(arrays[name+"_"+k], k, ids.shape)
                                     for k in ("f", "g", "w", "scores"))
        formula = scoring.score_variant({"optimal_f_score": local, "global_score": glob}, wbbs, spec)
        if not np.array_equal(scores, formula):
            raise ValueError("Query shard score formula differs")
        if not np.array_equal(order_scores(ids, scores, metadata["query_id"], tie_seed), np.arange(len(ids))):
            raise ValueError("Query shard score or SHA tie order differs")
    return arrays


def _query_shard(output, signature, split, query, candidates, scorer, wbbs_provider,
                 specs, signatures, top_k, tie_seed):
    parent = Path(output)/"shards"/split
    parent.mkdir(parents=True, exist_ok=True)
    target = parent/str(int(query))
    metadata = {"version": "fgw_topk_shard_v1", "source_signature": signature,
                "split": split, "query_id": int(query), "candidate_count": len(candidates),
                "candidate_ids_digest": digest(candidates.tolist()), "top_k": top_k,
                "variant_signatures": signatures, "tie_seed": tie_seed}
    if target.exists():
        return _load_shard(target, metadata, specs, candidates, tie_seed), True
    values = scorer.components(query, candidates)
    wbbs = scoring.unit_vector(wbbs_provider.scores(query, candidates), "WBBS", candidates.shape)
    local = scoring.unit_vector(values["optimal_f_score"], "optimal F", candidates.shape)
    glob = scoring.unit_vector(values["global_score"], "global G", candidates.shape)
    arrays = {}
    for name, spec in specs.items():
        scores = scoring.score_variant(values, wbbs, spec)
        order = order_scores(candidates, scores, int(query), tie_seed)[:top_k]
        arrays.update({name+"_ids": candidates[order], name+"_scores": scores[order],
                       name+"_f": local[order], name+"_g": glob[order], name+"_w": wbbs[order]})
    attempt = Path(tempfile.mkdtemp(prefix=f".{int(query)}_attempt_", dir=parent))
    with (attempt/"scores.npz").open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    write_json(attempt/"completed.json", {"metadata": metadata, "signature": digest(metadata),
               "files": {"scores.npz": file_hash(attempt/"scores.npz")}})
    if target.exists():
        raise FileExistsError("Concurrent query shard creation: " + str(target))
    attempt.rename(target)
    return arrays, False


def evaluate(output, signature, scorer, pool, bundle, ks, tie_seed, wbbs_provider,
             settings=scoring.SETTINGS):
    """Every candidate is scored once and shared by the three fixed variants.

    WBBS providers must validate their full-score source signature before use.
    Completed query shards bind that same parent signature and are reusable only
    with unchanged source, formula, protocol and tie metadata.
    """
    settings = scoring.validate_settings(settings)
    split = getattr(pool, "split_name", None)
    variants = scoring.specs(split, settings)
    if split not in ("valid", "test"):
        raise ValueError("FGW requires valid/test pool identity")
    if list(ks) != settings["ks"] or tie_seed != settings["tie_seed"]:
        raise ValueError("Evaluation K/tie seed differs from frozen settings")
    if max(ks) > pool.candidates_per_query:
        raise ValueError("K must fit the complete candidate pool")
    if not isinstance(signature, str) or not signature:
        raise ValueError("A signed source/evaluation identity is required")
    gallery = np.asarray(pool.gallery)
    queries = list(pool)
    if (gallery.ndim != 1 or gallery.dtype.kind not in "iu" or np.any(gallery < 0)
            or len(set(map(int, gallery))) != len(gallery) or len(queries) != len(pool)
            or len(set(map(int, queries))) != len(queries)
            or any(isinstance(q, (bool, np.bool_)) or not isinstance(q, (int, np.integer)) for q in queries)
            or pool.candidates_per_query != len(gallery)-1):
        raise ValueError("Invalid frozen query/gallery identities")
    gallery_set = set(map(int, gallery))
    output = Path(output)
    parent = output/split
    parent.mkdir(parents=True, exist_ok=True)
    signatures, pending = {}, []
    for name, spec in variants.items():
        variant_signature = digest({"evaluation": signature, "variant": name, "spec": spec,
                                    "split": split, "settings": settings, "device": "cpu",
                                    "retrieval_protocol_signature": getattr(pool, "protocol_signature", None)})
        signatures[name] = variant_signature
        target = parent/name
        if geo.completed(target, variant_signature):
            continue
        identity = _identity(output, split, name, variant_signature)
        attempt = Path(tempfile.mkdtemp(prefix=f".{name}_attempt_", dir=parent))
        record(attempt/"run_identity.json", identity)
        record(attempt/"evaluation_manifest.json", {**identity, "parent_signature": signature,
               "signature": variant_signature, "spec": spec, "ks": list(ks), "tie_seed": tie_seed,
               "retrieval_protocol_signature": getattr(pool, "protocol_signature", None),
               "num_queries": len(pool), "gallery_count": len(pool.gallery),
               "candidate_rule": "all_gallery_except_query", "scoring_device": "cpu",
               "settings": settings, "training_required": False, "posthoc_research": True,
               "test_parameter_search": False, "parameter_selection": settings["parameter_selection"]})
        pending.append((name, spec, target, attempt, identity))
    if pending:
        started = time.perf_counter(); reused = 0
        with ExitStack() as stack:
            writers = []
            for name, _, _, attempt, _ in pending:
                writer = csv.writer(stack.enter_context((attempt/"rankings.csv").open("w", newline="")))
                writer.writerow(RANKING_FIELDS); writers.append((name, writer))
            for number, query in enumerate(queries, 1):
                candidates = _candidate_vector(pool, query, gallery_set)
                arrays, was_reused = _query_shard(output, signature, split, query, candidates, scorer,
                                      wbbs_provider, variants, signatures, max(ks), tie_seed)
                reused += int(was_reused)
                for name, writer in writers:
                    ids = arrays[name+"_ids"]
                    writer.writerows((int(query), rank, int(candidate),
                                      *(float(arrays[name+"_"+key][i]) for key in ("scores", "f", "g", "w")))
                                     for i, (rank, candidate) in enumerate(zip(range(1, len(ids)+1), ids)))
                if number % 25 == 0 or number == len(queries):
                    print(f"FGW {split}: {number}/{len(queries)} queries; reused shards={reused}", flush=True)
                    _status(output, split, "ranking", queries_done=number, queries_total=len(queries),
                            variants=[item[0] for item in pending], query_shards_reused=reused)
        ranking_seconds = time.perf_counter()-started
        for index, (name, spec, target, attempt, identity) in enumerate(pending, 1):
            metric_started = time.perf_counter()
            _status(output, split, "metrics", variant=name, variants_done=index-1, variants_total=len(pending))
            metadata = {"signature": signatures[name], "run_id": identity["run_id"],
                        "started_at": identity["started_at"], "variant": name, **spec,
                        "candidates_scored": "full_pool", "ranking_scope": "top_max_k",
                        "retrieval_protocol_signature": getattr(pool, "protocol_signature", None),
                        "posthoc_research": True, "test_parameter_search": False}
            evaluate_rankings(bundle, pool, fusion.iter_rankings(attempt/"rankings.csv"), ks, attempt,
                              split_name=split, metadata=metadata)
            geo.evaluate_rankings_file(bundle, pool, attempt/"rankings.csv", ks, attempt,
                                       split_name=split, metadata=metadata)
            write_json(attempt/"timing.json", {"ranking_batch_wall_seconds": ranking_seconds,
                       "ranking_batch_variants": [item[0] for item in pending],
                       "ranking_time_is_shared_not_per_variant": True, "query_shards_reused": reused,
                       "metrics_wall_seconds": time.perf_counter()-metric_started})
            geo.commit_marker(attempt, signatures[name])
            attempt.rename(target)
            exact = read_json(target/"metrics.json")["metrics"]
            coverage = read_json(target/"coverage_metrics.json")["metrics"]
            print(f"FGW {split}/{name}: P@10={exact.get('Precision@10')}; "
                  f"mAP@100={exact.get('mAP@100')}; Coverage-nDCG@10={coverage.get('Coverage-nDCG@10')}", flush=True)
    reports = reporting.summarize(output, split, variants)
    record(output/f"evaluation_index_{split}.json", {"signature": signature, "split": split,
           "variants": variants, "variant_signatures": signatures,
           "completed_sha256": {name: file_hash(parent/name/"completed.json") for name in variants}})
    return reports
