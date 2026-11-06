"""Scoped launchers for the canonical experiment implementations."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def invoke(script, arguments):
    """Run an experiment CLI while restoring the caller's argument list."""
    previous = sys.argv
    try:
        sys.argv = [str(ROOT / script), *arguments]
        runpy.run_path(str(ROOT / script), run_name="__main__")
    finally:
        sys.argv = previous


def launch(name, argv=None):
    catalog = json.loads((ROOT / "experiments/catalog.json").read_text())
    entry = catalog[name]
    arguments = list(sys.argv[1:] if argv is None else argv)
    if entry["kind"] == "main_method":
        return main_method(name, entry, arguments)
    if entry["kind"] == "handcrafted_count":
        return handcrafted_count(name, entry, arguments)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--config", default=str(ROOT / entry["config"]))
    # A per-experiment entry point cannot silently train another variant.
    parser.add_argument("--variants", nargs="+")
    args, remaining = parser.parse_known_args(arguments)
    variant = entry.get("variant")
    # Legacy argparse accepts long-option abbreviations. Do not let --var/--v
    # bypass the variant binding when forwarded to that parser.
    if any(token.startswith("--") and "--variants".startswith(token.split("=", 1)[0])
           for token in remaining):
        parser.error("Use the full --variants option; this entry point runs only its bound experiment")
    if args.variants is not None and args.variants != ([variant] if variant else []):
        parser.error(f"{name} is bound to {variant or 'its configured budgets'}; use that experiment's run.py")
    forwarded = ["--config", str(project_path(args.config))]
    if variant:
        forwarded += ["--variants", variant]
    invoke(entry["entrypoint"], forwarded + remaining)


def main_method(name, entry, arguments):
    """Run only the requested member of the historical five-method experiment."""
    parser = argparse.ArgumentParser(description=f"{name}: {entry['description']}", allow_abbrev=False)
    parser.add_argument("--config", default=str(ROOT / entry["config"]))
    stages = ["all", "audit", "prepare", "retrieve", "evaluate", "test"]
    if entry.get("training_stage"):
        stages.append("train")
    parser.add_argument("--stage", choices=stages, default="all")
    parser.add_argument("--device")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(arguments)
    from wbm_retrieval.common import budget_name, load_config

    config_path = project_path(args.config)
    config = load_config(config_path)
    if args.device:
        config["device"] = args.device
    methods = ([budget_name(b) for b in config["global"]["budgets"]]
               if entry["method"] == "supervised_global" else [entry["method"]])
    if args.stage == "all":
        plan = ["prepare"] + ([entry["training_stage"]] if entry.get("training_stage") else [])
        plan += ["retrieve", "evaluate"]
    elif args.stage == "test":
        plan = ["retrieve", "evaluate"]
    elif args.stage == "train":
        plan = [entry["training_stage"]]
    else:
        plan = [args.stage]
    if args.dry_run:
        print(json.dumps({"experiment": name, "config": config, "stages": plan,
                          "methods": methods, "retrieval_split": "test", "training_started": False},
                         indent=2, ensure_ascii=False))
        return
    for stage in plan:
        forwarded = ["--config", str(config_path), "--stage", stage]
        if args.device:
            forwarded += ["--device", args.device]
        if args.resume:
            forwarded.append("--resume")
        if stage in ("retrieve", "evaluate"):
            forwarded += ["--methods", *methods]
        invoke("experiments/_shared/main_cli.py", forwarded)


def handcrafted_count(name, entry, arguments):
    """Evaluate the G counting baseline without invoking any learned trainer."""
    parser = argparse.ArgumentParser(description=f"{name}: {entry['description']}", allow_abbrev=False)
    parser.add_argument("--config", default=str(ROOT / entry["config"]))
    parser.add_argument("--stage", choices=("all", "prepare", "evaluate", "test"), default="all")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(arguments)
    from experiments.proposed.g.suite import evaluate, exclusive, load_suite, prepare_suite

    suite, config = load_suite(project_path(args.config), output_root=args.output_root,
                               seed=args.seed, device=args.device, batch_size=args.batch_size,
                               workers=args.num_workers)
    split = "test" if args.stage == "test" else "valid"
    if args.dry_run:
        print(json.dumps({"experiment": name, "config": config, "stage": args.stage,
                          "split": split, "method": "handcrafted", "training_started": False}, indent=2))
        return
    root = Path(config["output_dir"])
    with exclusive(root):
        prepare_suite(suite, config)
        if args.stage != "prepare":
            evaluate(config, suite, split, root / "handcrafted" / split,
                     rankings=True, bootstrap=config["evaluation"]["bootstrap_samples"])
