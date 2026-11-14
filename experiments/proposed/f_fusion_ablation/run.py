#!/usr/bin/env python3
"""Valid only: frozen alpha=.5 baseline, density, angular structure, optimal matching."""
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
from experiments.proposed.f_fusion_ablation import pipeline
from experiments.proposed.f_fusion_ablation.scoring import SETTINGS, Scores, variants
from wbm_retrieval.common import ROOT, digest, file_hash, write_json
from wbm_retrieval.data import Protocol, source_fingerprint
from wbm_retrieval.diagnostics import diagnostic_log


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("valid",), default="valid")
    parser.add_argument("--source-root", default="artifacts/f_segmentation_v1")
    parser.add_argument("--f-feature-root", default="artifacts/f_segmentation_radial_area_v1")
    parser.add_argument("--fusion-root", default="artifacts/f_global_fusion_v1",
                        help="Read-only fallback for F features built by the previous fusion run")
    parser.add_argument("--wapirl-valid-root", default="../compared_methods/artifacts/wapirl_valid_v1")
    parser.add_argument("--output-root", default="artifacts/f_fusion_ablation_valid_v1")
    parser.add_argument("--data-file")
    parser.add_argument("--device", default="cpu", help="Only used for missing F feature extraction")
    parser.add_argument("--prepare-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare_workers < 0:
        parser.error("prepare-workers must be nonnegative")
    source, features, previous, global_root, output = map(project_path, (
        args.source_root, args.f_feature_root, args.fusion_root, args.wapirl_valid_root, args.output_root))
    separate_output(output, [source, features, previous, global_root, ROOT / "protocol"])
    for parent in output.parents:
        if (parent / "ablation_manifest.json").exists():
            raise ValueError("Cannot nest outputs inside another ablation")
    plan = {"stage": "valid", "test_evaluated": False, "training_required": False, "settings": SETTINGS,
            "variants": {name: list(spec) for name, spec in variants().items()},
            "primary_metric": "exact/all/Precision@10",
            "four_groups": ["baseline", "density", "angular", "optimal"],
            "output": str(output), "wapirl_valid_root": str(global_root)}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if output.exists() and not (output / "ablation_manifest.json").exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output; historical files are immutable")
    data_file = project_path(args.data_file) if args.data_file else None
    print("Checking frozen F model, protocol and WaPIRL valid cache", flush=True)
    with geo.source_lock(source):
        config, suite, checkpoint, kind, inputs = geo.load_source(source, "cc8", data_file, args.device)
        separate_output(output, [Path(config[k]).resolve() for k in
                                 ("data_file", "split_manifest", "deduplication_metadata", "retrieval_protocol_dir")])
        if (kind != "best_nce" or config["seed"] != 2026 or config["embedding_dim"] != 512 or
                config["proposal"]["min_area"] != 5 or config["proposal"]["top_k"] != 4 or
                suite["variants"]["F"]["window_size"] != 52 or config["matching"]["weight"] != "sqrt_area" or
                any(config["matching"][k] != 0 for k in ("lambda_pos", "lambda_scale", "shape_threshold", "score_threshold"))):
            raise ValueError("Expected frozen cc8 main-method settings")
        protocol = Protocol(config)
        if protocol.retrieval is None:
            raise ValueError("Requires frozen deduplicated protocol")
        pool, ks = protocol.retrieval.valid, config["evaluation"]["ks"]
        if 10 not in ks or max(ks) > pool.candidates_per_query:
            raise ValueError("Expected Precision@10 selection and K within the full valid pool")
        sources = source_fingerprint(config)
        gids, global_features, global_identity = fusion.load_global(global_root, "valid", pool, sources)
        geometry_spec = geo.new_identity(config, inputs)
        geometry_signature = digest(geometry_spec)
        code_paths = [*Path(__file__).parent.glob("*.py"), *Path(fusion.__file__).parent.glob("*.py"),
                      ROOT / "wbm_retrieval/diagnostics.py", ROOT / "wbm_retrieval/exact_metrics.py"]
        identity = {"version": "f_fusion_four_valid_v1", "plan": {k:v for k,v in plan.items()
                    if k not in ("output", "wapirl_valid_root")}, "geometry": geometry_spec,
                    "global": global_identity, "sources": sources, "ks": ks,
                    "tie_seed": config["evaluation"].get("tie_seed", 2026),
                    "code": {str(p.relative_to(ROOT)): file_hash(p) for p in code_paths}}
        signature = digest(identity)
        with exclusive(output):
            if not (output / "ablation_manifest.json").exists() and any(p.name != ".ablation.lock" for p in output.iterdir()):
                raise ValueError("Choose an empty output root")
            record(output / "ablation_manifest.json", {**identity, "signature": signature})
            record(output / "plan_before_evaluation.json", identity["plan"])
            with diagnostic_log(output):
                print(json.dumps(identity["plan"]), flush=True)
                try:
                    pipeline.status(output, "preparing_features")
                    with np.load(config["data_file"], allow_pickle=False) as archive:
                        maps, labels = archive["arr_0"], archive["arr_1"]
                    local_dir = None
                    for candidate in (features / "cc8/features/valid", previous / "f_cache/features/valid",
                                      output / "f_cache/features/valid"):
                        if geo.completed(candidate, geometry_signature):
                            local_dir = candidate
                            print(f"Reusing verified F valid features: {candidate}", flush=True)
                            break
                    if local_dir is None:
                        print("Extracting missing F valid features into new output (no training)", flush=True)
                        local_dir = geo.build_features(output / "f_cache", config, suite, checkpoint, protocol,
                                                       maps, "valid", geometry_signature, args.prepare_workers)
                    description = pipeline.descriptors(output, maps, pool.gallery, signature)
                    del maps
                    evaluation = {"run_signature": signature, "f_features": file_hash(local_dir / "completed.json"),
                                  "descriptors": file_hash(output / "descriptors/completed.json")}
                    evaluation_signature = digest(evaluation)
                    record(output / "evaluation_manifest.json", {**evaluation, "signature": evaluation_signature})
                    scorer = Scores(local_dir, gids, global_features, pool.gallery, config["matching"], description)
                    bundle = fusion.bundle_for(protocol, labels)
                    pipeline.evaluate(output, evaluation_signature, scorer, pool, bundle, ks, identity["tie_seed"])
                    selected = pipeline.summarize(output, evaluation_signature)
                    geo.original.verify_files(source / "cc8", inputs)
                    write_json(output / "status.json", {"status": "completed", "split": "valid",
                                                        "test_evaluated": False, "four_groups": selected})
                    print(f"Completed valid only: {selected}", flush=True)
                except BaseException as error:
                    write_json(output / "status.json", {"status": "failed", "split": "valid", "test_evaluated": False,
                                                        "error_type": type(error).__name__, "error": str(error)})
                    raise


if __name__ == "__main__":
    main()
