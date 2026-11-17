"""Isolated B1 descriptor caching, streaming evaluation, and frozen selection."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from datetime import datetime
import csv
import multiprocessing
from pathlib import Path
import re
import tempfile
import time
from zoneinfo import ZoneInfo

import numpy as np

from experiments.proposed.f_fusion_ablation.scoring import describe as describe_angular
from experiments.proposed.f_global_fusion import pipeline as fusion
from experiments.proposed.f_segmentation_geometry import pipeline as geo
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from wbm_retrieval.exact_metrics import evaluate_rankings, order_scores
from . import descriptors, scoring, reporting

TZ = ZoneInfo("Asia/Shanghai")


def record(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Immutable B1 metadata changed; use a new output: {path}")
    else:
        write_json(path, value)


def status(output, split, phase, **details):
    write_json(Path(output) / "status.json", {"status": "running", "split": split,
               "test_evaluated": split == "test", "phase": phase, **details})


def _describe_job(job):
    raw, settings = job
    result = descriptors.describe(raw, settings)
    density, moments, outer_count = describe_angular(raw)
    return {**result, "density": density, "moments": moments, "outer_count": outer_count}


def cache_descriptors(output, maps, ids, signature, settings, workers=0, split="valid"):
    """Only the supplied gallery IDs are described; old source caches are never written."""
    if split not in ("valid", "test") or workers < 0:
        raise ValueError("Invalid descriptor split/workers")
    output = Path(output)
    target = output / "descriptors" / split
    ids = np.asarray(ids)
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids) or len(np.unique(ids)) != len(ids) or
            np.any(ids < 0) or np.any(ids >= len(maps))):
        raise ValueError("Descriptor IDs must be unique original integer IDs")
    if geo.completed(target, signature):
        with np.load(target / f"{split}.npz", allow_pickle=False) as archive:
            values = dict(archive)
        if not np.array_equal(values["sample_ids"], ids):
            raise ValueError("Cached descriptor IDs differ from the complete gallery")
        return values
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=f".{split}_attempt_", dir=target.parent))
    jobs = ((maps[int(sample_id)], settings) for sample_id in ids)
    rows = []
    def consume(iterator):
        for index, value in enumerate(iterator, 1):
            rows.append(value)
            if index % 100 == 0 or index == len(ids):
                print(f"B1 {split} descriptors: {index}/{len(ids)}", flush=True)
                status(output, split, "descriptors", images_done=index, images_total=len(ids))
    if workers:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            consume(executor.map(_describe_job, jobs, chunksize=8))
    else:
        consume(map(_describe_job, jobs))
    values = {"sample_ids": ids.astype(np.int64), **{key: np.asarray([row[key] for row in rows])
              for key in ("joint128", "empty", "original_count", "retained_count", "saliency_max",
                          "density", "moments", "outer_count")}}
    np.savez_compressed(attempt / f"{split}.npz", **values)
    write_json(attempt / "preparation.json", {"split": split, "num_images": len(ids),
               "seconds": time.perf_counter()-started, "workers": workers,
               "source": "complete original masks; no Top-4 crop; no labels",
               "empty_count": int(values["empty"].sum()),
               "original_points": int(values["original_count"].sum()),
               "retained_points": int(values["retained_count"].sum())})
    geo.commit_marker(attempt, signature)
    attempt.rename(target)
    return values


def _identity(output, split, name, signature):
    folder = Path(output) / "identities"
    path = folder / f"{split}__{name}.json"
    batch = re.sub(r"[^a-zA-Z0-9_]+", "_", Path(output).name)
    safe_name = name.replace(".", "p")
    key = f"b1_{batch}_{split}_{safe_name}"
    science = {"schema_version": 1, "key": key, "evaluation_signature": signature, "variant": name, "split": split}
    if path.exists():
        value = read_json(path)
        if any(value.get(key) != expected for key, expected in science.items()):
            raise ValueError("Resume identity does not match the scientific evaluation")
        started_at = value.get("started_at")
        if not isinstance(started_at, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00", started_at):
            raise ValueError("Resume identity needs the original valid +08:00 start time")
        started = datetime.fromisoformat(started_at)
        if value.get("run_id") != started.strftime("%Y%m%dT%H%M%S%z") + "__" + key:
            raise ValueError("Resume run_id differs from its original start time and key")
        return value
    now = datetime.now(TZ).replace(microsecond=0)
    value = {**science,
             "run_id": now.strftime("%Y%m%dT%H%M%S%z") + "__" + key,
             "started_at": now.isoformat(timespec="seconds")}
    record(path, value)
    return value


def evaluate(output, signature, scorer, pool, bundle, ks, tie_seed, descriptor_values,
             settings, device="cpu", variants=None, reference=None):
    """All candidates are scored before Top-K truncation; each evaluation gets an identity."""
    output = Path(output)
    split = getattr(pool, "split_name", None)
    if split not in ("valid", "test"):
        raise ValueError("B1 requires an explicit valid/test pool")
    all_specs = scoring.valid_variants(settings)
    if split == "test" and variants is None:
        raise ValueError("Test requires variants from a verified frozen valid selection")
    specs = all_specs if variants is None else variants
    if not specs or any(name not in all_specs or spec != all_specs[name] for name, spec in specs.items()):
        raise ValueError("Unknown or modified B1 variant")
    if split == "valid" and specs != all_specs:
        raise ValueError("Valid must evaluate the complete predeclared B1 grid")
    if not ks or max(ks) > pool.candidates_per_query:
        raise ValueError("K must fit the complete candidate pool")
    if not np.array_equal(np.sort(descriptor_values["sample_ids"]), np.sort(pool.gallery)):
        raise ValueError("Structure descriptors must cover exactly the complete gallery")
    distances = scoring.StructuralDistances(descriptor_values["sample_ids"], descriptor_values["joint128"],
                                            descriptor_values["empty"], device=device)
    parent = output / split
    parent.mkdir(parents=True, exist_ok=True)
    pending = []
    signatures = {}
    for name, spec in specs.items():
        variant_signature = digest({"evaluation": signature, "variant": name, "spec": spec})
        signatures[name] = variant_signature
        target = parent / name
        if geo.completed(target, variant_signature):
            continue
        identity = _identity(output, split, name, variant_signature)
        attempt = Path(tempfile.mkdtemp(prefix=f".{name}_attempt_", dir=parent))
        record(attempt / "run_identity.json", identity)
        record(attempt / "evaluation_manifest.json", {**identity, "parent_signature": signature,
               "spec": spec, "ks": list(ks), "tie_seed": tie_seed,
               "retrieval_protocol_signature": getattr(pool, "protocol_signature", None),
               "num_queries": len(pool), "gallery_count": len(pool.gallery),
               "candidate_rule": "all_gallery_except_query", "scoring_device": device,
               "training_required": False, "posthoc_research": True,
               "test_parameter_search": False, "signature": variant_signature})
        pending.append((name, spec, target, attempt, identity))
    if pending:
        started = time.perf_counter()
        fields = ["query_id", "rank", "candidate_id", "similarity_score", "b0_score", "structure_distance",
                  "f_score", "optimal_f_score", "global_score", "angular_distance"]
        with ExitStack() as stack:
            writers = []
            for name, spec, _, attempt, _ in pending:
                writer = csv.writer(stack.enter_context((attempt / "rankings.csv").open("w", newline="")))
                writer.writerow(fields)
                writers.append((name, spec, writer))
            for number, query in enumerate(pool, 1):
                candidates = pool[query]
                values = scorer.components(query, candidates)
                if any(not np.isfinite(value).all() for value in values.values()):
                    raise ValueError("Nonfinite base score component")
                structure = distances.for_query(query, candidates)
                baseline = scoring.score_variant(values, structure, all_specs["B0"])
                zero_component = np.zeros(len(candidates), dtype=np.float64)
                saved_components = tuple(values.get(key, zero_component) for key in
                                         ("f_score", "optimal_f_score", "global_score", "angular_distance"))
                for name, spec, writer in writers:
                    scores = scoring.score_variant(values, structure, spec)
                    if not np.isfinite(scores).all():
                        raise ValueError("Nonfinite B1 scores")
                    order = order_scores(candidates, scores, int(query), tie_seed)[:max(ks)]
                    distance = (zero_component if spec["family"] == "B0" else
                                structure["J" if spec["family"] == "J_only" else spec["family"]])
                    writer.writerows((int(query), rank, int(candidates[i]), float(scores[i]),
                                      float(baseline[i]), float(distance[i]),
                                      *(float(component[i]) for component in saved_components))
                                     for rank, i in enumerate(order, 1))
                if number % 100 == 0 or number == len(pool):
                    print(f"B1 {split} full-gallery ranking: {number}/{len(pool)}, {len(pending)} variants", flush=True)
                    status(output, split, "ranking", queries_done=number, queries_total=len(pool), variants=len(pending))
        ranking_seconds = time.perf_counter()-started
        for index, (name, spec, target, attempt, identity) in enumerate(pending, 1):
            started_metrics = time.perf_counter()
            status(output, split, "metrics", variant=name, variants_done=index-1, variants_total=len(pending))
            metadata = {"signature": signatures[name], "run_id": identity["run_id"],
                        "started_at": identity["started_at"], "variant": name, **spec,
                        "candidates_scored": "full_pool", "ranking_scope": "top_max_k",
                        "retrieval_protocol_signature": getattr(pool, "protocol_signature", None)}
            evaluate_rankings(bundle, pool, fusion.iter_rankings(attempt / "rankings.csv"), ks, attempt,
                              split_name=split, metadata=metadata)
            geo.evaluate_rankings_file(bundle, pool, attempt / "rankings.csv", ks, attempt,
                                       split_name=split, metadata=metadata)
            write_json(attempt / "timing.json", {"ranking_batch_wall_seconds": ranking_seconds,
                       "ranking_batch_variants": [item[0] for item in pending],
                       "ranking_time_is_shared_not_per_variant": True,
                       "metrics_wall_seconds": time.perf_counter()-started_metrics})
            geo.commit_marker(attempt, signatures[name])
            attempt.rename(target)
            reporting.print_report(name, split, {family: read_json(target / filename) for family, filename in
                                   (("exact", "metrics.json"), ("coverage", "coverage_metrics.json"))}, reference)
    reports = reporting.summarize(output, split, specs, reference, print_all=not pending)
    write_json(output / f"evaluation_index_{split}.json", {"signature": signature, "split": split, "variants": specs,
               "variant_signatures": signatures,
               "completed_sha256": {name: file_hash(parent / name / "completed.json") for name in specs}})
    return reports


def select(output, signature, settings, context=None):
    """The primary hypothesis is J even if a control family has a higher valid score."""
    output = Path(output)
    specs = scoring.valid_variants(settings)
    index = read_json(output / "evaluation_index_valid.json")
    if index["signature"] != signature or index["split"] != "valid" or index["variants"] != specs:
        raise ValueError("Selection requires the complete matching valid evaluation")
    values, hashes = {}, {}
    for name in specs:
        folder = output / "valid" / name
        if not geo.completed(folder, index["variant_signatures"][name]):
            raise ValueError("Complete valid evaluation before selection")
        report = read_json(folder / "metrics.json")
        if report["split"] != "valid" or "Precision@10" not in report["metrics"]:
            raise ValueError("B1 selects only valid exact Precision@10")
        values[name] = report["metrics"]["Precision@10"]
        hashes[name] = file_hash(folder / "completed.json")
    choices = {}
    for family in ("R", "T", "I", "J"):
        candidates = [{"variant": name, "strength": spec["strength"], "value": values[name]}
                      for name, spec in specs.items() if spec["family"] in ("B0", family)]
        chosen = max(candidates, key=lambda row: (row["value"], -row["strength"]))
        choices[family] = {"selected": chosen, "candidates": candidates}
    payload = {"schema_version": 1, "split": "valid", "test_evaluated": False,
               "evaluation_signature": signature, "settings": settings, "context": context,
               "primary_family": "J", "selection_metric": "exact/all/Precision@10",
               "tie_rule": "smaller strength, zero aliases B0", "selections": choices,
               "variant_signatures": index["variant_signatures"], "completed_sha256": hashes}
    selection = {**payload, "signature": digest(payload)}
    record(output / "selection.json", selection)
    print("\nFrozen valid selection (J is the predeclared B1 hypothesis):", flush=True)
    for family, value in choices.items():
        selected = value["selected"]
        print(f"  {family}: {selected['variant']}, lambda={selected['strength']}, P@10={selected['value']*100:.4f}%", flush=True)
    if choices["J"]["selected"]["variant"] == "B0":
        print("  J selected zero: this valid experiment does not support a nonzero B1 structure penalty.", flush=True)
    reporting.print_selection_summary(output, selection, "valid")
    return selection


def load_selection(path, settings, expected_context=None):
    path = Path(path)
    selection = read_json(path)
    payload = {key: value for key, value in selection.items() if key != "signature"}
    if (selection.get("signature") != digest(payload) or selection.get("split") != "valid" or
            selection.get("test_evaluated") is not False or selection.get("settings") != settings or
            selection.get("primary_family") != "J"):
        raise ValueError("Invalid or changed frozen B1 valid selection")
    if expected_context is not None and selection.get("context") != expected_context:
        raise ValueError("Valid/test data, model, recipe or source context differ")
    specs = scoring.valid_variants(settings)
    if set(selection.get("completed_sha256", {})) != set(specs):
        raise ValueError("Selection is missing predeclared valid configurations")
    values = {}
    for name in specs:
        folder = path.parent / "valid" / name
        if (not geo.completed(folder, selection["variant_signatures"][name]) or
                file_hash(folder / "completed.json") != selection["completed_sha256"][name]):
            raise ValueError("Frozen valid report changed")
        values[name] = read_json(folder / "metrics.json")["metrics"]["Precision@10"]
    for family in ("R", "T", "I", "J"):
        expected = [{"variant": name, "strength": spec["strength"], "value": values[name]}
                    for name, spec in specs.items() if spec["family"] in ("B0", family)]
        chosen = max(expected, key=lambda row: (row["value"], -row["strength"]))
        if selection["selections"].get(family) != {"selected": chosen, "candidates": expected}:
            raise ValueError("Frozen choice is not the prescribed valid selection")
    return selection


def test_variants(selection, settings):
    specs = scoring.valid_variants(settings)
    names = {"B0", "J_only", *(selection["selections"][family]["selected"]["variant"]
                               for family in ("R", "T", "I", "J"))}
    return {name: spec for name, spec in specs.items() if name in names}
