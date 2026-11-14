#!/usr/bin/env python3
"""Evaluate four frozen valid-selected groups on test, without training or tuning."""
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
from experiments.proposed.f_fusion_ablation.scoring import SETTINGS, Scores, variants
from experiments.proposed.f_fusion_ablation_test import pipeline
from wbm_retrieval.common import ROOT, digest, file_hash, write_json
from wbm_retrieval.data import Protocol, source_fingerprint
from wbm_retrieval.diagnostics import diagnostic_log


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("test",), default="test")
    parser.add_argument("--valid-root", default="artifacts/f_fusion_ablation_valid_v1")
    parser.add_argument("--source-root", default="artifacts/f_segmentation_v1")
    parser.add_argument("--f-feature-root", default="artifacts/f_segmentation_radial_area_v1")
    parser.add_argument("--fusion-root", default="artifacts/f_global_fusion_v1",
                        help="Read-only fallback for previously extracted F test features")
    parser.add_argument("--wapirl-test-root", default="../compared_methods/artifacts/wapirl_test_v1")
    parser.add_argument("--output-root", default="artifacts/f_fusion_ablation_test_v1")
    parser.add_argument("--data-file")
    parser.add_argument("--device", default="cpu", help="Only used for missing F feature extraction")
    parser.add_argument("--prepare-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Show paths and policy without reading artifacts")
    args = parser.parse_args(argv)
    if args.prepare_workers < 0:
        parser.error("prepare-workers must be nonnegative")
    source, features, previous, global_root, valid, output = map(project_path, (
        args.source_root, args.f_feature_root, args.fusion_root, args.wapirl_test_root,
        args.valid_root, args.output_root))
    separate_output(output, [source, features, previous, global_root, valid, ROOT / "protocol"])
    for parent in output.parents:
        if any((parent / name).exists() for name in ("ablation_manifest.json", "test_manifest.json")):
            raise ValueError("Cannot nest outputs inside another ablation")
    plan = {"stage": "test", "training_required": False, "alpha_global": .5,
            "selection": "frozen valid selection.json; no test search",
            "four_groups": ["baseline", "density", "angular", "optimal"],
            "valid_root": str(valid), "output": str(output), "wapirl_test_root": str(global_root)}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if output.exists() and not (output / "test_manifest.json").exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output; historical files are immutable")
    print("Checking completed valid selection and frozen source signatures", flush=True)
    valid_manifest, selection, selected = pipeline.load_selection(valid)
    if (valid_manifest["plan"]["settings"] != SETTINGS or
            valid_manifest["plan"]["variants"] != {name: list(spec) for name, spec in variants().items()}):
        raise ValueError("Valid scoring settings differ from current implementation")
    data_file = project_path(args.data_file) if args.data_file else None
    with geo.source_lock(source):
        config, suite, checkpoint, kind, inputs = geo.load_source(source, "cc8", data_file, args.device)
        separate_output(output, [Path(config[k]).resolve() for k in
                                 ("data_file", "split_manifest", "deduplication_metadata", "retrieval_protocol_dir")])
        geometry_spec = geo.new_identity(config, inputs)
        sources = source_fingerprint(config)
        ks, tie_seed = config["evaluation"]["ks"], config["evaluation"].get("tie_seed", 2026)
        if (geometry_spec != valid_manifest["geometry"] or sources != valid_manifest["sources"] or
                ks != valid_manifest["ks"] or tie_seed != valid_manifest["tie_seed"] or kind != "best_nce"):
            raise ValueError("F model, data, protocol or evaluation settings differ from valid")
        protocol = Protocol(config)
        if protocol.retrieval is None:
            raise ValueError("Requires frozen deduplicated protocol")
        pool = protocol.retrieval.test
        if max(ks) > pool.candidates_per_query:
            raise ValueError("K exceeds the full test pool")
        gids, global_features, global_identity = fusion.load_global(
            global_root, "test", pool, sources, valid_manifest["global"]["checkpoint"],
            valid_manifest["global"]["inference_recipe"])
        geometry_signature = digest(geometry_spec)
        identity = {"version": "f_fusion_four_test_v1", "alpha_global": .5,
                    "selected_variants": selected, "selection_signature": selection["signature"],
                    "valid_files": {name: file_hash(valid / name) for name in
                                    ("ablation_manifest.json", "evaluation_manifest.json", "selection.json")},
                    "geometry": geometry_spec, "global": global_identity, "sources": sources,
                    "ks": ks, "tie_seed": tie_seed, "valid_code": valid_manifest["code"],
                    "code": {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
        signature = digest(identity)
        with exclusive(output):
            if not (output / "test_manifest.json").exists() and any(p.name != ".ablation.lock" for p in output.iterdir()):
                raise ValueError("Choose an empty output root")
            record(output / "test_manifest.json", {**identity, "signature": signature})
            record(output / "frozen_selection.json", selection)
            record(output / "plan_before_evaluation.json", {"stage": "test", "alpha_global": .5,
                    "training_required": False, "test_parameter_search": False, "four_groups": selected,
                    "query_count": len(pool), "gallery_count": len(pool.gallery),
                    "candidates_per_query": pool.candidates_per_query})
            with diagnostic_log(output):
                print(f"Frozen four groups: {selected}; test queries={len(pool)}, candidates/query={pool.candidates_per_query}", flush=True)
                try:
                    pipeline.status(output, "preparing_features")
                    with np.load(config["data_file"], allow_pickle=False) as archive:
                        maps, labels = archive["arr_0"], archive["arr_1"]
                    local_dir = None
                    for candidate in (features / "cc8/features/test", previous / "f_cache/features/test",
                                      output / "f_cache/features/test"):
                        if geo.completed(candidate, geometry_signature):
                            local_dir = candidate
                            print(f"Reusing verified F test features: {candidate}", flush=True)
                            break
                    if local_dir is None:
                        print("Extracting missing F test features into new output (no training)", flush=True)
                        local_dir = geo.build_features(output / "f_cache", config, suite, checkpoint, protocol,
                                                       maps, "test", geometry_signature, args.prepare_workers)
                    description = pipeline.descriptors(output, maps, pool.gallery, signature)
                    del maps
                    evaluation = {"run_signature": signature, "f_features": file_hash(local_dir / "completed.json"),
                                  "descriptors": file_hash(output / "descriptors/completed.json")}
                    evaluation_signature = digest(evaluation)
                    record(output / "evaluation_manifest.json", {**evaluation, "signature": evaluation_signature})
                    scorer = Scores(local_dir, gids, global_features, pool.gallery, config["matching"], description)
                    bundle = fusion.bundle_for(protocol, labels)
                    pipeline.evaluate(output, evaluation_signature, scorer, pool, bundle, ks, tie_seed, selected)
                    pipeline.summarize(output, evaluation_signature, selected)
                    geo.original.verify_files(source / "cc8", inputs)
                    write_json(output / "status.json", {"status": "completed", "split": "test",
                                                        "test_evaluated": True, "four_groups": selected})
                    print(f"Completed test: {selected}", flush=True)
                except BaseException as error:
                    write_json(output / "status.json", {"status": "failed", "split": "test", "test_evaluated": True,
                                                        "error_type": type(error).__name__, "error": str(error)})
                    raise


if __name__ == "__main__":
    main()
