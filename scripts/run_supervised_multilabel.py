#!/usr/bin/env python3
"""Isolated supervised whole-map multilabel experiment; all trains/evaluates VALID only."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/supervised_multilabel.json")
    parser.add_argument("--stage", choices=("all", "prepare", "train", "evaluate", "test", "plot"), default="all")
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--budget-source", help="Optional old supervised experiment; verify identical labeled_train_ids")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Resolve configuration only; no data reads/training/writes")
    args = parser.parse_args()
    from experiments.supervised.suite import exclusive, load_suite, prepare
    config = load_suite(args.config, device=args.device, num_workers=args.num_workers, cpu_threads=args.cpu_threads,
                        output_root=args.output_root, budget_source=args.budget_source)
    if args.dry_run:
        print(json.dumps({"stage": args.stage, "config": config, "all_evaluates": "valid only",
                          "model_selection": "valid macro-F1, then lower BCE, then earlier epoch"}, indent=2))
        return
    # Import plotting only after configuring a writable font-cache directory.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(config["output_root"]) / "matplotlib"))
    import torch
    torch.set_num_threads(config["cpu_threads"])
    with exclusive(config["output_root"]):
        signature, protocol = prepare(config)
        from experiments.supervised.training import train, evaluate, summarize
        from experiments.supervised.suite import method_name
        from experiments.supervised.figures import learning_curves, classification_figures
        for budget in config["budgets"]:
            if args.stage in ("all", "train"):
                train(config, protocol, signature, budget, resume=args.resume)
            if args.stage in ("all", "evaluate", "test"):
                evaluate(config, protocol, signature, budget, "test" if args.stage == "test" else "valid")
            if args.stage == "plot":
                folder = Path(config["output_root"]) / method_name(budget)
                if (folder / "history.json").exists():
                    learning_curves(folder)
                for split in ("valid", "test"):
                    if (folder / "classification" / split / "predictions.npz").exists():
                        classification_figures(folder, split, config["output_root"])
        if args.stage in ("all", "evaluate", "test", "plot"):
            for split in (("valid", "test") if args.stage == "plot" else ("test" if args.stage == "test" else "valid",)):
                summarize(config, split)


if __name__ == "__main__":
    main()
