#!/usr/bin/env python3
"""Add one fixed angular (.5) plus optimal-matching test experiment after test inspection."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from experiments.proposed._shared.ef_suite import exclusive
from experiments.proposed.f_global_fusion import pipeline as fusion
from experiments.proposed.f_global_fusion.run import project_path, separate_output, record
from experiments.proposed.f_segmentation_geometry import pipeline as geo
from experiments.proposed.f_fusion_ablation.scoring import Scores
from experiments.proposed.f_fusion_combination_test import pipeline
from wbm_retrieval.common import ROOT, digest, file_hash, write_json
from wbm_retrieval.data import Protocol, source_fingerprint
from wbm_retrieval.diagnostics import diagnostic_log


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("test",), default="test")
    parser.add_argument("--previous-test-root", default="artifacts/f_fusion_ablation_test_v1")
    parser.add_argument("--source-root", default="artifacts/f_segmentation_v1")
    parser.add_argument("--f-feature-root", default="artifacts/f_segmentation_radial_area_v1")
    parser.add_argument("--fusion-root", default="artifacts/f_global_fusion_v1")
    parser.add_argument("--wapirl-test-root", default="../compared_methods/artifacts/wapirl_test_v1")
    parser.add_argument("--output-root", default="artifacts/f_fusion_combination_test_v1")
    parser.add_argument("--data-file", help="Relocated NPZ with unchanged content")
    parser.add_argument("--dry-run", action="store_true", help="Show fixed plan without loading inputs or writing files")
    args = parser.parse_args(argv)
    source, features, old_fusion, global_root, previous, output = map(project_path, (
        args.source_root, args.f_feature_root, args.fusion_root, args.wapirl_test_root,
        args.previous_test_root, args.output_root))
    separate_output(output, [source, features, old_fusion, global_root, previous, ROOT / "protocol"])
    for parent in output.parents:
        if any((parent / n).exists() for n in ("ablation_manifest.json", "test_manifest.json", "combination_manifest.json")):
            raise ValueError("Cannot nest outputs inside another ablation")
    plan = {"stage": "test", "variant": pipeline.VARIANT, **pipeline.SETTINGS,
            "previous_test_root": str(previous), "output": str(output), "wapirl_test_root": str(global_root)}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if output.exists() and not (output / "combination_manifest.json").exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output; historical files are immutable")
    data_file = project_path(args.data_file) if args.data_file else None
    print("Checking prior test results, frozen feature caches and code signatures", flush=True)
    # Open existing source locks read-only; do not change source contents or timestamps.
    with geo.source_lock(previous), geo.source_lock(source):
        prior, prior_eval, reports, descriptors, provenance = pipeline.load_previous(previous)
        config, _, _, kind, inputs = geo.load_source(source, "cc8", data_file, "cpu")
        separate_output(output, [Path(config[k]).resolve() for k in
                                 ("data_file", "split_manifest", "deduplication_metadata", "retrieval_protocol_dir")])
        geometry_spec, sources = geo.new_identity(config, inputs), source_fingerprint(config)
        ks, tie_seed = config["evaluation"]["ks"], config["evaluation"].get("tie_seed", 2026)
        if (geometry_spec != prior["geometry"] or sources != prior["sources"] or
                ks != prior["ks"] or tie_seed != prior["tie_seed"] or kind != "best_nce"):
            raise ValueError("F model, data, protocol or evaluation settings differ from previous test")
        protocol = Protocol(config)
        if protocol.retrieval is None:
            raise ValueError("Requires frozen deduplicated protocol")
        pool = protocol.retrieval.test
        if max(ks) > pool.candidates_per_query or not np.array_equal(descriptors["sample_ids"], pool.gallery):
            raise ValueError("Previous descriptors/K differ from full test pool")
        gids, global_features, global_identity = fusion.load_global(
            global_root, "test", pool, sources, prior["global"]["checkpoint"], prior["global"]["inference_recipe"])
        if global_identity != prior["global"]:
            raise ValueError("WaPIRL test cache differs from previous test")
        geometry_signature = digest(geometry_spec)
        local_dir = None
        for candidate in (features / "cc8/features/test", old_fusion / "f_cache/features/test",
                          previous / "f_cache/features/test"):
            if geo.completed(candidate, geometry_signature):
                if file_hash(candidate / "completed.json") != prior_eval["f_features"]:
                    raise ValueError("F feature cache differs from previous test")
                local_dir = candidate
                print(f"Reusing verified F test features: {candidate}", flush=True)
                break
        if local_dir is None:
            raise FileNotFoundError("Keep the F test cache from the previous run; check --f-feature-root / --fusion-root / --previous-test-root")
        identity = {"version": "f_fusion_combination_test_v1", "settings": pipeline.SETTINGS,
                    "previous_test": provenance, "geometry": geometry_spec, "global": global_identity,
                    "sources": sources, "ks": ks, "tie_seed": tie_seed,
                    "f_features": prior_eval["f_features"], "descriptors": prior_eval["descriptors"],
                    "code": {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
        signature = digest(identity)
        with exclusive(output):
            if not (output / "combination_manifest.json").exists() and any(p.name != ".ablation.lock" for p in output.iterdir()):
                raise ValueError("Choose an empty output root")
            record(output / "combination_manifest.json", {**identity, "signature": signature})
            record(output / "plan_before_evaluation.json", {"stage": "test", "variant": pipeline.VARIANT,
                    **pipeline.SETTINGS, "query_count": len(pool), "gallery_count": len(pool.gallery),
                    "candidates_per_query": pool.candidates_per_query})
            with diagnostic_log(output):
                try:
                    pipeline.status(output, "loading_features")
                    print(f"Fixed combination: {pipeline.SETTINGS['formula']}", flush=True)
                    with np.load(config["data_file"], allow_pickle=False) as archive:
                        labels = archive["arr_1"]
                    scorer = Scores(local_dir, gids, global_features, pool.gallery, config["matching"], descriptors)
                    pipeline.evaluate(output, signature, scorer, pool, fusion.bundle_for(protocol, labels), ks, tie_seed)
                    pipeline.summarize(output, signature, reports)
                    geo.original.verify_files(source / "cc8", inputs)
                    write_json(output / "status.json", {"status": "completed", "split": "test",
                               "variant": pipeline.VARIANT, "test_evaluated": True, **pipeline.SETTINGS})
                    print("Completed fixed combination test", flush=True)
                except BaseException as error:
                    write_json(output / "status.json", {"status": "failed", "split": "test",
                               "error_type": type(error).__name__, "error": str(error), **pipeline.SETTINGS})
                    raise


if __name__ == "__main__":
    main()
