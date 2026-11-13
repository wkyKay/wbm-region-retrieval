#!/usr/bin/env python3
"""Evaluate six existing F models with radial/area scoring and both metric sets.

Source checkpoints and all original results are read-only. No training or
checkpoint selection occurs here. Repeating a command verifies completed work
and resumes missing evaluations in new, atomic attempt directories.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.proposed._shared.ef_suite import exclusive
from experiments.proposed.f_segmentation.proposals import METHODS
from experiments.proposed.f_segmentation_geometry.pipeline import evaluate_method, source_lock, summarize
from experiments.proposed.f_segmentation_geometry.scoring import SETTINGS
from wbm_retrieval.common import ROOT, get_device, read_json, write_json
from wbm_retrieval.diagnostics import diagnostic_log


def project_path(value):
    path = Path(value).expanduser()
    return (ROOT / path).resolve()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="artifacts/f_segmentation_v1")
    parser.add_argument("--output-root", default="artifacts/f_segmentation_radial_area_v1")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--splits", nargs="+", choices=("valid", "test"), default=["valid", "test"])
    parser.add_argument("--checkpoint", choices=("best_nce", "best_retrieval"),
                        help="Must match source final_checkpoint; default reads the frozen source rule")
    parser.add_argument("--data-file", help="Override relocated raw NPZ path, with content-hash validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare-workers", type=int, default=0,
                        help="CPU segmentation workers when a source split cache is missing")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.methods)) != len(args.methods) or len(set(args.splits)) != len(args.splits) or args.prepare_workers < 0:
        parser.error("Methods/splits must be unique; prepare-workers must be nonnegative")
    source, output = project_path(args.source_root), project_path(args.output_root)
    if source == output or source in output.parents or output in source.parents:
        parser.error("Output must be separate from source, not its ancestor or descendant")
    if output == ROOT or output in ROOT.parents:
        parser.error("Output cannot contain the repository")
    for parent in output.parents:
        if any((parent / n).exists() for n in ("suite_manifest.json", "resolved_config.json", "segmentation_comparison.json")):
            parser.error("Do not nest new output inside historical runs")
    data_file = str(Path(args.data_file).expanduser().resolve()) if args.data_file else None
    plan = {"source": str(source), "output": str(output), "methods": args.methods,
            "splits": args.splits, "scoring": SETTINGS, "source_read_only": True,
            "checkpoint": args.checkpoint or "source final_checkpoint", "training_required": False,
            "metrics": ["exact_v1", "coverage_v1"]}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    get_device(args.device)
    # Acquire source before creating output, so a live original job cannot race.
    with source_lock(source), exclusive(output):
        marker = output / "geometry_comparison.json"
        if marker.exists():
            if read_json(marker) != {"format": "f_segmentation_radial_area_v1", "source": str(source)}:
                raise ValueError("Output belongs to a different source/format")
        else:
            if any(p.name != ".ablation.lock" for p in output.iterdir()):
                raise ValueError("Choose an empty output root; historical outputs are immutable")
            write_json(marker, {"format": "f_segmentation_radial_area_v1", "source": str(source)})
        with diagnostic_log(output):
            print(json.dumps(plan, ensure_ascii=False), flush=True)
            state = {"status": "running", "methods": args.methods, "splits": args.splits}
            try:
                for method in args.methods:
                    state["current_method"] = method
                    write_json(output / "status.json", state)
                    evaluate_method(source, output, method, data_file, args.device, args.checkpoint,
                                    args.splits, args.prepare_workers)
                    summarize(output, METHODS)
                write_json(output / "status.json", {**state, "status": "completed"})
            except BaseException as error:
                write_json(output / "status.json", {**state, "status": "failed",
                           "error_type": type(error).__name__, "error": str(error)})
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
