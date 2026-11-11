#!/usr/bin/env python3
"""Run independent F segmentation training, valid and full test in one command."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.proposed._shared.ef_suite import exclusive
from experiments.proposed._shared.ef_training import train_variant
from experiments.proposed.f_segmentation.proposals import METHODS, SETTINGS
from experiments.proposed.f_segmentation.suite import (evaluate_final, initialize, load_suite,
                                                       method_config, prepare, identity)
from wbm_retrieval.common import digest, file_hash, get_device, read_json, write_csv, write_json
from wbm_retrieval.data import source_fingerprint
from wbm_retrieval.diagnostics import diagnostic_log


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--stage", choices=("all", "prepare", "train", "valid", "test"), default="all",
                        help="all explicitly includes test after training and valid")
    parser.add_argument("--checkpoint", choices=("best_nce", "best_retrieval"), default="best_nce",
                        help="Freeze before the run; best_retrieval uses valid labels")
    parser.add_argument("--output-root")
    parser.add_argument("--data-file", help="Path to the original MixedWM38 NPZ (content hash checked)")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--prepare-workers", type=int, default=0,
                        help="CPU segmentation workers, 0=serial; bounded jobs, spawn-safe")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Print resolved plan without writes/training")
    args = parser.parse_args(argv)
    if len(set(args.methods)) != len(args.methods) or args.prepare_workers < 0:
        parser.error("Methods must be unique and prepare-workers nonnegative")
    suite, config = load_suite(args.config, output_root=args.output_root, seed=args.seed,
                              device=args.device, batch_size=args.batch_size, workers=args.num_workers)
    if args.data_file:
        config["data_file"] = str(Path(args.data_file).expanduser().resolve())
    config["final_checkpoint"] = args.checkpoint
    root = Path(config["output_dir"])
    plan = {"stage": args.stage, "methods": {m: SETTINGS[m] for m in args.methods},
            "config": config, "valid_pool": suite["valid_pool"],
            "resume": "automatic; identical scientific settings and verified files only",
            "test": "all explicitly runs full test; test never selects a checkpoint"}
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    # Fail before cache work when the data/device is unavailable.
    source_fingerprint(config)
    if args.stage != "prepare":
        get_device(config["device"])
    for parent in root.parents:
        if (parent / "suite_manifest.json").exists() or (parent / "resolved_config.json").exists():
            raise ValueError("Do not nest comparisons inside historical runs")
    common_identity = identity(suite, config)
    with exclusive(root):
        marker = root / "segmentation_comparison.json"
        if marker.exists():
            if read_json(marker).get("signature") != digest(common_identity):
                raise ValueError("Comparison inputs/config/code changed; choose a new --output-root")
        else:
            if any(p.name != ".ablation.lock" for p in root.iterdir()):
                raise ValueError("Choose an empty --output-root for a new segmentation comparison")
            write_json(marker, {"signature": digest(common_identity), **common_identity})
        with diagnostic_log(root):
            for method in args.methods:
                current = method_config(config, method)
                with exclusive(current["output_dir"]):
                    signature = initialize(suite, current)
                    print(f"Starting {method}: {current['output_dir']}", flush=True)
                    write_json(Path(current["output_dir"]) / "execution_plan.json", plan)
                    if args.stage in ("all", "prepare", "train", "valid"):
                        prepare(suite, current, signature, workers=args.prepare_workers)
                    if args.stage in ("all", "train"):
                        train_variant(current, suite, signature, "F", resume=True)
                    if args.stage in ("all", "valid"):
                        evaluate_final(suite, current, signature, "valid", args.checkpoint)
                    if args.stage in ("all", "test"):
                        run = Path(current["output_dir"]) / f"seed_{current['seed']}" / "F"
                        if not read_json(run / "training_summary.json")["completed"]:
                            raise ValueError("Training must finish before test preprocessing")
                        prepare(suite, current, signature, splits=("test",), workers=args.prepare_workers)
                        evaluate_final(suite, current, signature, "test", args.checkpoint)
                summarize(root, config["seed"], args.checkpoint)


def summarize(root, seed, kind):
    rows = []
    for method in METHODS:
        for split in ("valid", "test"):
            path = root / method / f"seed_{seed}/F/{split}/{kind}/metrics.json"
            if not path.with_name("completed.json").exists():
                continue
            report = read_json(path)
            rows.append({"method": method, "split": split, "checkpoint": kind,
                         "epoch": report["epoch"], "num_queries": report["num_queries"],
                         **report["metrics"], "metrics_file": str(path.relative_to(root)),
                         "metrics_sha256": file_hash(path)})
    if rows:
        write_csv(root / f"comparison_seed_{seed}_{kind}.csv", rows, list(rows[0]))
        write_json(root / f"comparison_seed_{seed}_{kind}.json", rows)


if __name__ == "__main__":
    main()
