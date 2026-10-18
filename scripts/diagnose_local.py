#!/usr/bin/env python3
"""Compare saved local representations on a frozen valid retrieval pool."""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Keep imports and inference from creating files inside the source experiment.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

from wbm_retrieval.common import get_device, load_config, seed_everything
from wbm_retrieval.diagnostics import run_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_a.json")
    parser.add_argument("--device", choices=("cuda", "mps", "cpu", "auto"), default=None)
    parser.add_argument("--output-dir", help="Separate directory, outside the source experiment; project-relative")
    parser.add_argument("--queries", type=int, default=500, help="Valid queries, capped to available maps; 0=all")
    parser.add_argument("--candidates", type=int, default=1000, help="Candidates per query, capped to valid minus self; 0=all")
    parser.add_argument("--seed", type=int, default=2026, help="Protocol and bootstrap seed; random-model seed comes from config")
    parser.add_argument("--pair-samples", type=int, default=20000, help="Random region pairs; rotated anchors capped to token count")
    parser.add_argument("--neighbor-anchors", type=int, default=24)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=1000, help="Query bootstrap repetitions; 0=disabled")
    parser.add_argument("--batch-size", type=int, help="Inference only; does not change training config or signatures")
    parser.add_argument("--num-workers", type=int, default=0, help="Inference DataLoader workers")
    parser.add_argument("--cpu-threads", type=int, default=4, help="PyTorch CPU threads")
    parser.add_argument("--resume", action="store_true", help="Reuse verified diagnostic-only features/protocol with identical settings")
    args = parser.parse_args()
    if args.queries < 0 or args.candidates < 0 or args.bootstrap < 0 or args.num_workers < 0 or args.seed < 0:
        parser.error("queries/candidates/bootstrap/num-workers/seed must be nonnegative")
    if min(args.pair_samples, args.neighbor_anchors, args.neighbors, args.cpu_threads) < 1:
        parser.error("pair-samples/neighbor-anchors/neighbors/cpu-threads must be positive")
    config_path = Path(args.config).expanduser()
    config_path = config_path.resolve() if config_path.is_absolute() else (ROOT / config_path).resolve()
    config = load_config(config_path)
    batch = args.batch_size or config.get("inference_batch_size", config["batch_size"])
    if batch < 1 or args.batch_size == 0:
        parser.error("batch-size must be positive")
    if 10 not in config["evaluation"]["ks"]:
        parser.error("Diagnostic comparison requires evaluation.ks to contain 10")
    output = Path(args.output_dir).expanduser() if args.output_dir else Path(config["output_dir"] + "_valid_diagnostics")
    output = output.resolve() if output.is_absolute() else (ROOT / output).resolve()
    device = get_device(args.device or config["device"])
    seed_everything(config["seed"])
    import torch
    torch.set_num_threads(args.cpu_threads)
    settings = {key: getattr(args, key) for key in ("queries", "candidates", "seed", "pair_samples", "neighbor_anchors",
                                                  "neighbors", "bootstrap", "num_workers", "cpu_threads")}
    settings["batch_size"] = batch
    run_diagnostics(config_path, config, output, device, settings, resume=args.resume)


if __name__ == "__main__":
    main()
