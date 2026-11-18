"""Isolated fixed ablations, full-gallery scoring, and immutable completion records."""
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
from b1_structure_fusion.scoring import StructuralDistances, baseline_score
from experiments.proposed.f_global_fusion import pipeline as fusion
from experiments.proposed.f_segmentation_geometry import pipeline as geo
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from wbm_retrieval.exact_metrics import evaluate_rankings, order_scores
from . import scoring

TZ = ZoneInfo("Asia/Shanghai")
RANKING_FIELDS = ["query_id", "rank", "candidate_id", "similarity_score", "b0_score",
                  "structure_distance", "f_score", "optimal_f_score", "global_score",
                  "angular_distance", "joint_distance", "independent_distance", "full_b1_score"]


def record(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Immutable ablation metadata changed; use a new output: {path}")
    else:
        write_json(path, value)


def _identity(output, split, name, signature):
    path = Path(output) / "identities" / f"{split}__{name}.json"
    batch = re.sub(r"[^a-zA-Z0-9_]+", "_", Path(output).name)
    key = f"b1_ablation_{batch}_{split}_{name}"
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
    write_json(Path(output) / "status.json", {"status": "running", "split": split,
               "phase": phase, "test_evaluated": split == "test", **details})


def evaluate(output, signature, scorer, pool, bundle, ks, tie_seed, descriptor_values,
             settings, device="cuda", reference=None):
    """One full-gallery pass per split shares components across all fixed variants."""
    settings = scoring.validate_settings(settings)
    split = getattr(pool, "split_name", None)
    variants = scoring.specs(split, settings)
    if list(ks) != settings["ks"] or tie_seed != settings["tie_seed"]:
        raise ValueError("Evaluation K/tie seed differs from frozen settings")
    if max(ks) > pool.candidates_per_query:
        raise ValueError("K must fit the complete candidate pool")
    if not isinstance(signature, str) or not signature:
        raise ValueError("A signed source/evaluation identity is required")
    if not np.array_equal(np.sort(descriptor_values["sample_ids"]), np.sort(pool.gallery)):
        raise ValueError("Descriptors must cover exactly the complete gallery")
    gallery_set = set(map(int, pool.gallery))
    if len(gallery_set) != len(pool.gallery) or pool.candidates_per_query != len(gallery_set)-1:
        raise ValueError("Candidate pool must contain the entire gallery except self")
    distances = StructuralDistances(descriptor_values["sample_ids"], descriptor_values["joint128"],
                                    descriptor_values["empty"], device=device)
    output = Path(output)
    parent = output / split
    parent.mkdir(parents=True, exist_ok=True)
    signatures, pending = {}, []
    for name, spec in variants.items():
        variant_signature = digest({"evaluation": signature, "variant": name, "spec": spec,
                                    "split": split, "settings": settings, "device": str(device),
                                    "retrieval_protocol_signature": getattr(pool, "protocol_signature", None)})
        signatures[name] = variant_signature
        target = parent / name
        if geo.completed(target, variant_signature):
            continue
        identity = _identity(output, split, name, variant_signature)
        attempt = Path(tempfile.mkdtemp(prefix=f".{name}_attempt_", dir=parent))
        record(attempt / "run_identity.json", identity)
        record(attempt / "evaluation_manifest.json", {**identity, "parent_signature": signature,
               "signature": variant_signature, "spec": spec, "ks": list(ks), "tie_seed": tie_seed,
               "retrieval_protocol_signature": getattr(pool, "protocol_signature", None),
               "num_queries": len(pool), "gallery_count": len(pool.gallery),
               "candidate_rule": "all_gallery_except_query", "scoring_device": str(device),
               "settings": settings, "training_required": False, "posthoc_research": True,
               "test_parameter_search": False, "parameter_selection": settings["parameter_selection"]})
        pending.append((name, spec, target, attempt, identity))
    if pending:
        started = time.perf_counter()
        with ExitStack() as stack:
            writers = []
            for name, spec, _, attempt, _ in pending:
                writer = csv.writer(stack.enter_context((attempt / "rankings.csv").open("w", newline="")))
                writer.writerow(RANKING_FIELDS)
                writers.append((name, spec, writer))
            for number, query in enumerate(pool, 1):
                candidates = np.asarray(pool[query])
                if (int(query) not in gallery_set or len(candidates) != pool.candidates_per_query
                        or set(map(int, candidates)) != gallery_set-{int(query)}):
                    raise ValueError("Candidate pool must contain the entire gallery except self")
                values = scorer.components(query, candidates)
                structure = distances.for_query(query, candidates)
                base = baseline_score(values)
                anchor = scoring.full_b1_score(values, structure)
                component_keys = ("f_score", "optimal_f_score", "global_score", "angular_distance")
                for name, spec, writer in writers:
                    scores = scoring.score_variant(values, structure, spec)
                    order = order_scores(candidates, scores, int(query), tie_seed)[:max(ks)]
                    chosen_distance = structure[spec["family"]]
                    writer.writerows((int(query), rank, int(candidates[i]), float(scores[i]),
                                      float(base[i]), float(chosen_distance[i]),
                                      *(float(values[k][i]) for k in component_keys),
                                      float(structure["J"][i]), float(structure["I"][i]), float(anchor[i]))
                                     for rank, i in enumerate(order, 1))
                if number % 100 == 0 or number == len(pool):
                    print(f"B1 ablation {split}: {number}/{len(pool)} queries, {len(pending)} variants", flush=True)
                    _status(output, split, "ranking", queries_done=number, queries_total=len(pool),
                            variants=[item[0] for item in pending])
        ranking_seconds = time.perf_counter()-started
        for index, (name, spec, target, attempt, identity) in enumerate(pending, 1):
            metric_started = time.perf_counter()
            _status(output, split, "metrics", variant=name, variants_done=index-1, variants_total=len(pending))
            metadata = {"signature": signatures[name], "run_id": identity["run_id"],
                        "started_at": identity["started_at"], "variant": name, **spec,
                        "candidates_scored": "full_pool", "ranking_scope": "top_max_k",
                        "retrieval_protocol_signature": getattr(pool, "protocol_signature", None),
                        "posthoc_research": True, "test_parameter_search": False}
            evaluate_rankings(bundle, pool, fusion.iter_rankings(attempt / "rankings.csv"), ks, attempt,
                              split_name=split, metadata=metadata)
            geo.evaluate_rankings_file(bundle, pool, attempt / "rankings.csv", ks, attempt,
                                       split_name=split, metadata=metadata)
            write_json(attempt / "timing.json", {"ranking_batch_wall_seconds": ranking_seconds,
                       "ranking_batch_variants": [item[0] for item in pending],
                       "ranking_time_is_shared_not_per_variant": True,
                       "metrics_wall_seconds": time.perf_counter()-metric_started})
            geo.commit_marker(attempt, signatures[name])
            attempt.rename(target)
            reporting.print_report(name, split, {family: read_json(target / filename) for family, filename in
                                   (("exact", "metrics.json"), ("coverage", "coverage_metrics.json"))}, reference)
    reports = reporting.summarize(output, split, variants, reference, print_all=not pending)
    record(output / f"evaluation_index_{split}.json", {"signature": signature, "split": split,
           "variants": variants, "variant_signatures": signatures,
           "completed_sha256": {name: file_hash(parent / name / "completed.json") for name in variants}})
    return reports
