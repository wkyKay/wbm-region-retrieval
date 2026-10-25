#!/usr/bin/env python3
"""Run isolated G: 1000 negatives and query region-count containment."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.g.suite import evaluate, exclusive, load_suite, prepare_suite, summarize
from experiments.g.training import evaluate_variant, train_variant
from wbm_retrieval.diagnostics import diagnostic_log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ablation_g.json")
    parser.add_argument("--stage", choices=("all", "prepare", "train", "evaluate", "summarize", "test"), default="all")
    parser.add_argument("--variants", nargs="+", choices=["G"], default=None)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved plan; no data loading/training/writes")
    parser.add_argument("--test-checkpoint", choices=("best_retrieval", "best_nce"), default="best_retrieval")
    args = parser.parse_args()
    if args.stage == "test" and args.variants is None:
        parser.error("Test requires explicit --variants after selecting a design on valid")
    variants = args.variants or ["G"]
    if len(set(variants)) != len(variants):
        parser.error("Duplicate variants")
    suite, config = load_suite(args.config, output_root=args.output_root, seed=args.seed, device=args.device,
                              batch_size=args.batch_size, workers=args.num_workers)
    if args.dry_run:
        print(json.dumps({"stage": args.stage, "variants": {n: suite["variants"][n] for n in variants},
                          "config": config, "valid_pool": suite["valid_pool"],
                          "selection_metric": suite["selection_metric"], "evaluate_every": suite["evaluate_every"]}, indent=2))
        return
    root = Path(config["output_dir"])
    with exclusive(root):
        # Validate directory identity before opening logs in an unrecognized directory.
        signature = prepare_suite(suite, config)
        with diagnostic_log(root):
            split = "test" if args.stage == "test" else "valid"
            if args.stage in ("all", "evaluate", "test"):
                target = root / "handcrafted" / split
                evaluate(config, suite, split, target, rankings=True, bootstrap=config["evaluation"]["bootstrap_samples"])
            for name in variants:
                if args.stage in ("all", "train"):
                    train_variant(config, suite, signature, name, resume=args.resume)
                if args.stage in ("all", "evaluate", "test"):
                    evaluate_variant(config, suite, signature, name, split,
                                     args.test_checkpoint if args.stage == "test" else "both")
            if args.stage != "prepare":
                summarize(config, suite, split)


if __name__ == "__main__":
    main()
