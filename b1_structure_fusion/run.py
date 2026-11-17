#!/usr/bin/env python3
"""Isolated B1 evaluation: frozen features, full galleries, timestamped artificial/ outputs."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from b1_structure_fusion import descriptors, pipeline, reporting
from b1_structure_fusion.scoring import valid_variants
from experiments.proposed._shared.ef_suite import exclusive
from experiments.proposed.f_fusion_ablation.scoring import Scores
from experiments.proposed.f_global_fusion import pipeline as fusion
from experiments.proposed.f_global_fusion.run import separate_output
from experiments.proposed.f_segmentation_geometry import pipeline as geo
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from wbm_retrieval.data import Protocol, source_fingerprint
from wbm_retrieval.diagnostics import diagnostic_log

TZ = ZoneInfo("Asia/Shanghai")
VERSION = "b1_structure_fusion_v1"


def project_path(value):
    return (ROOT / Path(value).expanduser()).resolve()


def code_identity():
    """Hash scoring dependencies without including docs, tests or the result registry."""
    paths = set(Path(__file__).parent.glob("*.py"))
    paths.add(Path(__file__).parent / "config.json")
    for folder in ("f_global_fusion", "f_segmentation_geometry"):
        paths.update((ROOT / "experiments/proposed" / folder).glob("*.py"))
    paths.update(ROOT / p for p in (
        "experiments/proposed/f_fusion_ablation/scoring.py",
        "experiments/proposed/f_fusion_combination_test/pipeline.py",
        "experiments/proposed/f_segmentation/tensor_voting.py",
        "wbm_retrieval/common.py", "wbm_retrieval/data.py", "wbm_retrieval/diagnostics.py",
        "wbm_retrieval/exact_metrics.py", "wbm_retrieval/retrieval_protocol.py"))
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)}


def checked_reference(split, sources, *, root=ROOT):
    """Read only this split's WBBS reports from the verified, committed catalogue snapshot."""
    from results_catalog import build as catalog
    folder = root / "results_catalog"
    snapshot = read_json(folder / "snapshot.json")
    specs = read_json(folder / "sources.json")
    spec_sha = hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest()
    if snapshot["source_spec_sha256"] != spec_sha:
        raise ValueError("Result catalogue sources changed: collect/check before reading WBBS")
    generated = catalog.render(snapshot)
    for name in ("metrics.csv", "registry.csv"):
        if (folder / name).read_text() != generated[name]:
            raise ValueError(f"Stale result catalogue: {name}")
    expected_key = "wbbs_" + split
    matches = [r for r in snapshot["runs"] if r["key"] == expected_key]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one registered {expected_key}")
    run = matches[0]
    if (run["split"] != split or run["protocol"] != "retrieval_v1" or
            run["status"] != "report_hashes_verified" or run["method"] != "wbbs"):
        raise ValueError("WBBS reference must be a verified ordinary retrieval result")
    for name, expected in snapshot["protocol_files"]["wbm-region-retrieval"].items():
        if file_hash(root / "protocol" / name) != expected:
            raise ValueError("WBBS reference and current frozen protocol differ")
    expected_pool = (500, 3599, 3598) if split == "valid" else (7199, 7199, 7198)
    if tuple(run[k] for k in ("query_count", "gallery_count", "candidates_per_query")) != expected_pool:
        raise ValueError("WBBS reference has a different candidate pool")
    rows = []
    hashes = {}
    for family, item in run["reports"].items():
        report = item["data"]
        if report["split"] != split or report["num_queries"] != expected_pool[0]:
            raise ValueError("WBBS report split/count mismatch")
        hashes[family] = item["source"]
        # If full remote reports are available, verify them too. Local snapshots remain usable.
        original = root.parent / "compared_methods" / item["source"]["path"]
        if original.exists() and file_hash(original) != item["source"]["sha256"]:
            raise ValueError(f"WBBS original report changed: {original}")
        for group, metrics in catalog.metric_groups(report):
            for metric, value in metrics.items():
                name, k = metric.rsplit("@", 1)
                rows.append({"split": split, "group": group, "metric_version": family + "_v1",
                             "metric": name, "K": int(k), "value": value})
    return rows, {"key": expected_key, "run_id": run["run_id"], "reports": hashes,
                  "missing_coverage": "coverage" not in run["reports"],
                  "reference_run_sha256": digest(run)}


def choose_output(args, root=ROOT):
    if args.resume_root:
        target = (root / Path(args.resume_root).expanduser()).resolve()
        if not (target / "run_manifest.json").is_file():
            raise ValueError("--resume-root requires an existing signed B1 run")
    elif args.output_root:
        target = (root / Path(args.output_root).expanduser()).resolve()
    else:
        stamp = datetime.now(TZ).strftime("%Y%m%dT%H%M%S%z")
        target = root / "artificial" / f"b1_structure_{args.stage}_{stamp}"
    allowed = (root / "artificial").resolve()
    if target == allowed or allowed not in target.parents:
        raise ValueError("B1 outputs must be isolated below this project's artificial/ directory")
    for parent in target.parents:
        if parent == allowed:
            break
        if (parent / "run_manifest.json").exists():
            raise ValueError("Do not nest B1 outputs inside an existing B1 run")
    if not args.resume_root and target.exists():
        # A launcher may open its log before Python starts; no other pre-existing files allowed.
        if any(p.name != "launch.log" for p in target.iterdir()):
            raise ValueError("Existing output: use --resume-root for identical settings, or a new timestamp")
    return target


def frozen_local_cache(args, split, geometry_signature):
    candidates = [project_path(args.f_feature_root) / "cc8/features" / split,
                  project_path(args.fusion_root) / "f_cache/features" / split,
                  project_path(args.previous_test_root) / "f_cache/features" / split] if split == "test" else [
                  project_path(args.f_feature_root) / "cc8/features" / split,
                  project_path(args.fusion_root) / "f_cache/features" / split]
    for path in candidates:
        if geo.completed(path, geometry_signature):
            return path
    raise FileNotFoundError("B1 requires completed frozen F features; checked: " + ", ".join(map(str, candidates)))


def assert_frozen(config, suite, kind, settings):
    if (kind != "best_nce" or config["seed"] != 2026 or config["embedding_dim"] != 512 or
            config["proposal"]["min_area"] != 5 or config["proposal"]["top_k"] != 4 or
            suite["variants"]["F"]["window_size"] != 52 or config["matching"]["weight"] != "sqrt_area" or
            any(config["matching"][k] != 0 for k in ("lambda_pos", "lambda_scale", "shape_threshold", "score_threshold")) or
            config["evaluation"]["ks"] != settings["ks"] or
            config["evaluation"].get("tie_seed", 2026) != settings["tie_seed"]):
        raise ValueError("Expected frozen cc8 F best_nce/seed2026 and unchanged evaluation recipe")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("valid", "test", "all"), default="valid",
                   help="all explicitly runs valid then frozen test; default is valid only")
    output = p.add_mutually_exclusive_group()
    output.add_argument("--output-root", help="New timestamped directory under artificial/")
    output.add_argument("--resume-root", help="Resume identical inputs in an existing B1 directory")
    p.add_argument("--selection-file", help="Completed valid selection required for stage=test")
    p.add_argument("--source-root", default="artifacts/f_segmentation_v1")
    p.add_argument("--f-feature-root", default="artifacts/f_segmentation_radial_area_v1")
    p.add_argument("--fusion-root", default="artifacts/f_global_fusion_v1")
    p.add_argument("--previous-test-root", default="artifacts/f_fusion_ablation_test_v1")
    p.add_argument("--wapirl-valid-root", default="../compared_methods/artifacts/wapirl_valid_v1")
    p.add_argument("--wapirl-test-root", default="../compared_methods/artifacts/wapirl_test_v1")
    p.add_argument("--data-file")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda",
                   help="CUDA accelerates float64 structure distances; frozen B0 stays NumPy for exact fallback")
    p.add_argument("--prepare-workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true", help="Print plan only, no data loading or output")
    p.add_argument("--check-inputs", action="store_true", help="Verify available source models/features without evaluating")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.prepare_workers < 0:
        raise ValueError("prepare-workers must be nonnegative")
    if args.stage == "test" and not args.selection_file:
        raise ValueError("--stage test requires --selection-file from a completed B1 valid run")
    if args.stage != "test" and args.selection_file:
        raise ValueError("--selection-file is only used with --stage test")
    settings = descriptors.load_settings()
    output = choose_output(args)
    plan = {"stage": args.stage, "output": str(output), "settings": settings,
            "device": args.device, "prepare_workers": args.prepare_workers,
            "valid_variants": valid_variants(settings), "test_variants": "frozen valid choices plus B0/J_only",
            "posthoc_design": True, "test_parameter_search": False, "training_required": False,
            "all_zero_J_policy": "finish valid; skip test when primary J selects zero"}
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; activate the original CUDA environment or explicitly use --device cpu")
    source = project_path(args.source_root)
    data_file = project_path(args.data_file) if args.data_file else None
    protected = [source, project_path(args.f_feature_root), project_path(args.fusion_root),
                 project_path(args.previous_test_root), project_path(args.wapirl_valid_root),
                 project_path(args.wapirl_test_root), ROOT / "protocol", ROOT / "results_catalog"]
    separate_output(output, protected)
    if args.selection_file:
        separate_output(output, [project_path(args.selection_file).parent])
    selection = pipeline.load_selection(project_path(args.selection_file), settings) if args.selection_file else None
    print("Checking frozen cc8 F, protocol and existing feature caches", flush=True)
    with geo.source_lock(source):
        config, suite, checkpoint, kind, inputs = geo.load_source(source, "cc8", data_file, "cpu")
        del checkpoint  # No model inference or training in B1.
        assert_frozen(config, suite, kind, settings)
        separate_output(output, [Path(config[k]) for k in
                        ("data_file", "split_manifest", "deduplication_metadata", "retrieval_protocol_dir")])
        protocol = Protocol(config)
        if protocol.retrieval is None:
            raise ValueError("B1 requires the frozen deduplicated retrieval protocol")
        sources, geometry = source_fingerprint(config), geo.new_identity(config, inputs)
        code = code_identity()
        splits = ("valid", "test") if args.stage == "all" else (args.stage,)
        contexts = {}
        expected_context = selection.get("context") if selection else None
        for split in splits:
            pool = getattr(protocol.retrieval, split)
            local = frozen_local_cache(args, split, digest(geometry))
            global_root = project_path(getattr(args, f"wapirl_{split}_root"))
            expected_global = expected_context["global"] if expected_context else None
            gids, gf, global_id = fusion.load_global(global_root, split, pool, sources,
                expected_global["checkpoint"] if expected_global else None,
                expected_global["inference_recipe"] if expected_global else None)
            context = {"version": VERSION, "settings": settings, "sources": sources, "geometry": geometry,
                       "global": {k: global_id[k] for k in ("checkpoint", "inference_recipe")},
                       "code": code, "device": args.device}
            if expected_context is not None and context != expected_context:
                raise ValueError("B1 valid/test scientific settings, device, source model or code differ")
            expected_context = context
            reference, ref_identity = checked_reference(split, sources)
            contexts[split] = (pool, local, gids, gf, global_id, context, reference, ref_identity)
            print(f"Verified {split}: {len(pool)} queries, {len(pool.gallery)} gallery; F features {local}", flush=True)
            if args.stage == "all" and not args.check_inputs:
                break  # Do not read test features before the valid selection is complete.
        if args.check_inputs:
            print("Inputs verified. No descriptors, rankings or metrics computed.", flush=True)
            return 0
        with exclusive(output):
            manifest_path = output / "run_manifest.json"
            if manifest_path.exists():
                manifest = fusion.signed_manifest(manifest_path)
                if manifest["context"] != expected_context or manifest["stage"] != args.stage:
                    raise ValueError("Resume settings differ; create a new run")
            else:
                stamp = datetime.now(TZ)
                manifest = {"version": VERSION, "stage": args.stage, "context": expected_context,
                            "batch_id": stamp.strftime("%Y%m%dT%H%M%S%z") + "__b1_" + args.stage,
                            "started_at": stamp.isoformat(timespec="seconds"), "output": str(output),
                            "selection_file": str(project_path(args.selection_file)) if args.selection_file else None,
                            "posthoc_design": True, "test_parameter_search": False, "training_required": False}
                manifest["signature"] = digest(manifest)
                write_json(manifest_path, manifest)
                write_json(output / "plan_before_evaluation.json", plan)
            with diagnostic_log(output):
                print("B1 output:", output, flush=True)
                try:
                    with np.load(config["data_file"], allow_pickle=False) as z:
                        maps, labels = z["arr_0"], z["arr_1"]
                    bundle = fusion.bundle_for(protocol, labels)
                    for split in splits:
                        if split == "test" and args.stage == "all":
                            selection = pipeline.load_selection(output / "selection.json", settings, expected_context)
                            if selection["selections"]["J"]["selected"]["strength"] == 0:
                                write_json(output / "status.json", {"status": "completed", "test_evaluated": False,
                                    "reason": "primary J selected zero; no supported B1 gain on valid", "output": str(output)})
                                print("J selected lambda=0. Valid complete; test skipped by the frozen plan.", flush=True)
                                break
                            pool = protocol.retrieval.test
                            local = frozen_local_cache(args, split, digest(geometry))
                            gids, gf, global_id = fusion.load_global(project_path(args.wapirl_test_root), split, pool,
                                sources, expected_context["global"]["checkpoint"], expected_context["global"]["inference_recipe"])
                            reference, ref_identity = checked_reference(split, sources)
                            contexts[split] = (pool, local, gids, gf, global_id, expected_context, reference, ref_identity)
                        pool, local, gids, gf, global_id, context, reference, ref_identity = contexts[split]
                        if selection:
                            pipeline.load_selection(project_path(args.selection_file) if args.selection_file else output / "selection.json",
                                                    settings, context)
                        input_identity = {"context": context, "split": split,
                            "f_features": file_hash(local / "completed.json"), "global_features": global_id,
                            "reference": ref_identity,
                            "selection_signature": selection["signature"] if split == "test" else None}
                        input_signature = digest(input_identity)
                        pipeline.record(output / f"inputs_{split}.json", {**input_identity, "signature": input_signature})
                        description = pipeline.cache_descriptors(output, maps, pool.gallery,
                            digest({"sources": sources, "split": split, "settings": settings, "code": code}),
                            settings, args.prepare_workers, split)
                        scorer = Scores(local, gids, gf, pool.gallery, config["matching"], description)
                        specs = pipeline.test_variants(selection, settings) if split == "test" else None
                        pipeline.evaluate(output, input_signature, scorer, pool, bundle, settings["ks"], settings["tie_seed"],
                                          description, settings, args.device, specs, reference)
                        if split == "valid":
                            selection = pipeline.select(output, input_signature, settings, context)
                        from b1_structure_fusion.verify import verify_evaluation
                        verification = {}
                        for name, spec in (specs or valid_variants(settings)).items():
                            verification[name] = verify_evaluation(output / split / name, pool, bundle,
                                settings["ks"], digest({"evaluation": input_signature, "variant": name, "spec": spec}),
                                settings["tie_seed"])
                        pipeline.record(output / f"verification_{split}.json", verification)
                        if split == "test":
                            reporting.print_selection_summary(output, selection, split)
                        geo.original.verify_files(source / "cc8", inputs)
                        # Detect source changes during a long evaluation, not only on entry.
                        geo.completed(local, digest(geometry))
                        global_root = project_path(getattr(args, f"wapirl_{split}_root"))
                        if (file_hash(global_root / "completed.json") != global_id["completed_sha256"] or
                                file_hash(global_root / "evaluation_manifest.json") != global_id["manifest_sha256"]):
                            raise ValueError("WaPIRL source changed during B1 evaluation")
                        fusion.checked_files(global_root, read_json(global_root / "completed.json")["files"])
                        fusion.checked_files(ROOT, code)
                        write_json(output / "status.json", {"status": "completed", "split": split,
                                   "test_evaluated": split == "test", "output": str(output)})
                    from b1_structure_fusion.register import write_proposal
                    write_proposal(output)
                    print("Finished. exact/coverage and signed differences: comparison.csv; registration proposal: catalog_sources.json", flush=True)
                except BaseException as exc:
                    write_json(output / "status.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
                    raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
