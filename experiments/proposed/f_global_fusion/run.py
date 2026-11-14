#!/usr/bin/env python3
"""Fuse frozen cc8 F and WaPIRL; select weights on valid, then freeze for test."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from experiments.proposed._shared.ef_suite import exclusive
from experiments.proposed.f_segmentation_geometry import pipeline as geo
from experiments.proposed.f_global_fusion import pipeline as fusion
from wbm_retrieval.common import ROOT, digest, file_hash, read_json, write_json
from wbm_retrieval.data import Protocol, source_fingerprint
from wbm_retrieval.diagnostics import diagnostic_log


def project_path(value):
    return (ROOT / Path(value).expanduser()).resolve()


def separate_output(output, protected):
    if output == ROOT or output in ROOT.parents:
        raise ValueError("Output must not contain the repository")
    for source in protected:
        if output == source or source in output.parents or output in source.parents:
            raise ValueError(f"Output must be separate from protected input: {source}")
    for parent in output.parents:
        if any((parent / n).exists() for n in ("suite_manifest.json", "resolved_config.json",
                                              "segmentation_comparison.json", "fusion_manifest.json")):
            raise ValueError("Do not nest fusion outputs inside an existing run")


def record(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Inputs/settings changed; choose a new output root: {path}")
    else:
        write_json(path, value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("valid", "test", "all"), default="valid")
    parser.add_argument("--source-root", default="artifacts/f_segmentation_v1", help="Original frozen F training root")
    parser.add_argument("--f-feature-root", default="artifacts/f_segmentation_radial_area_v1",
                        help="Read-only existing geometry evaluation root; missing caches are built in new output")
    parser.add_argument("--wapirl-valid-root", default="../compared_methods/artifacts/wapirl_valid_v1")
    parser.add_argument("--wapirl-test-root", default="../compared_methods/artifacts/wapirl_test_v1")
    parser.add_argument("--output-root", default="artifacts/f_global_fusion_v1")
    parser.add_argument("--alphas", nargs="+", type=float, default=list(fusion.DEFAULT_ALPHAS))
    parser.add_argument("--selection-family", choices=("exact", "coverage"), default="exact")
    parser.add_argument("--selection-group", choices=("all", "class_macro"), default="all")
    parser.add_argument("--selection-metric", default="Precision@10")
    parser.add_argument("--data-file", help="Relocated NPZ; content must match frozen source hash")
    parser.add_argument("--device", default="cpu", help="Only used if F features need extraction")
    parser.add_argument("--prepare-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    alphas = fusion.validate_alphas(args.alphas)
    if args.prepare_workers < 0:
        parser.error("prepare-workers must be nonnegative")
    source, feature_root, valid_root, test_root, output = map(project_path, (
        args.source_root, args.f_feature_root, args.wapirl_valid_root, args.wapirl_test_root, args.output_root))
    data_file = project_path(args.data_file) if args.data_file else None
    separate_output(output, [source, feature_root, valid_root, test_root, ROOT / "protocol",
                             *([data_file] if data_file else [])])
    plan = {"stage": args.stage, "method": "cc8", "alpha_global": alphas,
            "selection": [args.selection_family, args.selection_group, args.selection_metric],
            "source_root": str(source), "f_feature_root": str(feature_root),
            "wapirl_valid_root": str(valid_root), "wapirl_test_root": str(test_root),
            "output_root": str(output), "training_required": False,
            "test_weights": "frozen valid winner plus alpha=0 and alpha=1"}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if output.exists() and not (output / "fusion_manifest.json").exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output root; existing artifacts are immutable")
    if args.stage == "test" and not (output / "selection.json").is_file():
        raise ValueError("Run --stage valid before --stage test")
    # Validate frozen training, checkpoint and protocol before writing output.
    with geo.source_lock(source):
        config, suite, checkpoint, kind, inputs = geo.load_source(source, "cc8", data_file, args.device)
        separate_output(output, [Path(config[k]).resolve() for k in
                                 ("data_file", "split_manifest", "deduplication_metadata", "retrieval_protocol_dir")])
        if (config["final_checkpoint"] != "best_nce" or config["seed"] != 2026 or
                config["embedding_dim"] != 512 or suite["variants"]["F"]["window_size"] != 52 or
                config["proposal"]["min_area"] != 5 or config["proposal"]["top_k"] != 4 or
                config["matching"]["weight"] != "sqrt_area" or
                config["matching"]["lambda_pos"] != 0 or config["matching"]["lambda_scale"] != 0 or
                config["matching"]["shape_threshold"] != 0 or config["matching"]["score_threshold"] != 0):
            raise ValueError("Expected the frozen cc8 main method (best_nce, seed 2026, min_area5, Top4, sqrt_area)")
        protocol = Protocol(config)
        if protocol.retrieval is None:
            raise ValueError("Frozen deduplicated full-gallery protocol is required")
        ks = config["evaluation"]["ks"]
        prefixes = (("Hit", "Precision", "Recall", "mAP") if args.selection_family == "exact" else
                    ("Coverage-nDCG", "MeanCoverage", "FullCoverageHit", "FullCoveragePrecision",
                     "FullCoverageRecall", "FullCoverage_mAP"))
        if args.selection_metric not in {f"{name}@{k}" for name in prefixes for k in ks}:
            raise ValueError("Selection metric must belong to the selected metric family and frozen K values")
        if max(ks) > min(protocol.retrieval.valid.candidates_per_query, protocol.retrieval.test.candidates_per_query):
            raise ValueError("K exceeds the complete candidate pool")
        sources = source_fingerprint(config)
        valid_ids, valid_features, valid_identity = fusion.load_global(valid_root, "valid", protocol.retrieval.valid, sources)
        geometry_spec = geo.new_identity(config, inputs)
        geometry_signature = digest(geometry_spec)
        code_paths = list(Path(__file__).parent.glob("*.py")) + [
            ROOT / "wbm_retrieval/diagnostics.py", ROOT / "wbm_retrieval/exact_metrics.py"]
        identity = {"version": "cc8_wapirl_score_fusion_v1", "geometry": geometry_spec,
                    "wapirl_valid": valid_identity, "sources": sources,
                    "alphas": alphas, "ks": ks, "tie_seed": config["evaluation"].get("tie_seed", 2026),
                    "selection": plan["selection"], "formula": "(1-alpha)*F+alpha*WaPIRL",
                    "code": {str(p.relative_to(ROOT)): file_hash(p) for p in code_paths}}
        signature = digest(identity)
        manifest = {**identity, "signature": signature}
        with exclusive(output), diagnostic_log(output):
            path = output / "fusion_manifest.json"
            if not path.exists() and any(p.name not in (".ablation.lock", "run.log") for p in output.iterdir()):
                raise ValueError("Choose an empty output root")
            record(path, manifest)
            print(json.dumps(plan), flush=True)
            write_json(output / "status.json", {"status": "running", "stage": args.stage})
            try:
                with np.load(config["data_file"], allow_pickle=False) as archive:
                    labels = archive["arr_1"]
                bundle = fusion.bundle_for(protocol, labels)

                def feature_dir(split):
                    existing = feature_root / "cc8" / "features" / split
                    if existing.exists():
                        if not geo.completed(existing, geometry_signature):
                            raise ValueError(f"Incomplete F geometry cache: {existing}")
                        return existing
                    cached = output / "f_cache/features" / split
                    if geo.completed(cached, geometry_signature):
                        return cached
                    print(f"Building missing F {split} features in new fusion output", flush=True)
                    with np.load(config["data_file"], allow_pickle=False) as archive:
                        maps = archive["arr_0"]
                    return geo.build_features(output / "f_cache", config, suite, checkpoint, protocol,
                                              maps, split, geometry_signature, args.prepare_workers)

                def evaluate(split, weights, gids, features, global_identity, selection=None):
                    local_dir = feature_dir(split)
                    split_identity = {"run_signature": signature, "split": split,
                                      "f_cache_sha256": file_hash(local_dir / "completed.json"),
                                      "global": global_identity, "weights": weights,
                                      "selection_signature": selection["signature"] if selection else None}
                    split_signature = digest(split_identity)
                    record(output / f"{split}_manifest.json", {**split_identity, "signature": split_signature})
                    pool = getattr(protocol.retrieval, split)
                    scorer = fusion.BranchScores(local_dir, gids, features, pool.gallery, config["matching"])
                    fusion.evaluate(output, split, weights, split_signature, scorer, pool, bundle, ks, identity["tie_seed"])
                    fusion.summarize(output)
                    return split_signature

                if args.stage in ("valid", "all"):
                    valid_signature = evaluate("valid", alphas, valid_ids, valid_features, valid_identity)
                else:
                    valid_manifest = fusion.signed_manifest(output / "valid_manifest.json")
                    if valid_manifest["run_signature"] != signature:
                        raise ValueError("Valid results belong to different inputs/settings")
                    valid_signature = valid_manifest["signature"]
                selection = fusion.select(output, alphas, valid_signature, args.selection_metric,
                                          args.selection_family, args.selection_group)
                fusion.save_selection(output, selection, require_existing=args.stage == "test")
                print(f"Frozen valid selection: alpha={selection['alpha']}, {args.selection_metric}={selection['value']}", flush=True)
                if args.stage in ("test", "all"):
                    gids, features, global_identity = fusion.load_global(test_root, "test", protocol.retrieval.test,
                                                                          sources, valid_identity["checkpoint"],
                                                                          valid_identity["inference_recipe"])
                    evaluate("test", sorted({0., selection["alpha"], 1.}), gids, features, global_identity, selection)
                geo.original.verify_files(source / "cc8", inputs)
                write_json(output / "status.json", {"status": "completed", "stage": args.stage,
                                                     "selected_alpha": selection["alpha"]})
            except BaseException as error:
                write_json(output / "status.json", {"status": "failed", "stage": args.stage,
                                                     "error": str(error), "error_type": type(error).__name__})
                raise


if __name__ == "__main__":
    main()
