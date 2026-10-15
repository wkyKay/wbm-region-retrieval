#!/usr/bin/env python3
"""One entry point from preprocessing through all model fits and Experiment A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wbm_retrieval.common import ROOT, budget_name, load_config
from wbm_retrieval.data import Protocol, audit, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/experiment_a.json"))
    parser.add_argument("--stage", choices=["all", "audit", "prepare", "train-local", "train-global", "retrieve", "evaluate", "geometry"], default="all")
    parser.add_argument("--device", help="Override cuda/cpu/auto, or specify cuda:N")
    parser.add_argument("--resume", action="store_true", help="Resume compatible last checkpoints; reuse completed fits")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved settings and stage plan, without training or writes")
    parser.add_argument("--methods", nargs="+", help="For retrieve/evaluate only: handcrafted proposed global_5pct global_10pct global_50pct")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    valid_names = ["handcrafted", "proposed"] + [budget_name(b) for b in config["global"]["budgets"]]
    if args.methods and (args.stage not in ("retrieve", "evaluate") or not set(args.methods) <= set(valid_names)):
        parser.error("--methods only accepts configured methods for retrieve/evaluate")
    if args.dry_run:
        print(json.dumps({"config": config, "requested_stage": args.stage, "methods": args.methods or valid_names,
                          "all_stages": ["prepare", "train-local", "train-global", "retrieve", "evaluate"],
                          "relevance": "query label containment", "training_started": False}, indent=2, ensure_ascii=False))
        return
    if args.stage == "audit":
        print(json.dumps(audit(config), indent=2, ensure_ascii=False))
        return
    if args.stage in ("all", "prepare"):
        prepare(config)
        if args.stage == "prepare":
            return
    else:
        marker = Path(config["output_dir"]) / "cache/prepared.json"
        if not marker.is_file():
            raise FileNotFoundError("Run --stage prepare before this stage")
        # Check hashes/configuration before consuming any stale prepared artifacts.
        prepare(config)
    protocol = Protocol(config)
    if args.stage in ("all", "train-local"):
        from wbm_retrieval.methods import proposed
        proposed.train(config, resume=args.resume)
    if args.stage in ("all", "train-global"):
        from wbm_retrieval.methods import supervised_global
        supervised_global.train(config, protocol, resume=args.resume)
    if args.stage in ("all", "retrieve"):
        from wbm_retrieval.retrieval import retrieve
        retrieve(config, protocol, args.methods)
    if args.stage in ("all", "evaluate"):
        from wbm_retrieval.evaluation import compare_paired, evaluate_all
        evaluate_all(config, protocol, args.methods)
        if not args.methods or {"proposed", "handcrafted"} <= set(args.methods):
            compare_paired(config)
    if args.stage == "geometry":
        from wbm_retrieval.geometry import generate_examples
        generate_examples(config)


if __name__ == "__main__":
    main()
