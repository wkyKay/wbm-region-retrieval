#!/usr/bin/env python3
"""Inference-only single-label, largest-region retrieval using an existing best.pt."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-experiment", required=True, help="Existing experiment with resolved_config, cache and models/local/best.pt")
    parser.add_argument("--output-dir", help="Separate output tree; defaults to source name + _single_label_top1")
    parser.add_argument("--splits", nargs="+", choices=("valid", "test"), default=["valid", "test"])
    parser.add_argument("--device", default="cuda", help="cuda, cuda:N, mps or cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true", help="Reuse verified protocol/features with identical inputs and settings")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.bootstrap < 0 or len(set(args.splits)) != len(args.splits):
        parser.error("Use a positive batch size, nonnegative workers/bootstrap and unique splits")
    source = Path(args.source_experiment).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else source.with_name(source.name + "_single_label_top1")
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    if args.device == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    from wbm_retrieval.common import get_device
    from wbm_retrieval.single_label import run_single_label
    settings = {"splits": args.splits, "batch_size": args.batch_size, "num_workers": args.num_workers,
                "bootstrap": args.bootstrap, "seed": args.seed}
    run_single_label(source, output, get_device(args.device), settings, args.resume)


if __name__ == "__main__":
    main()
