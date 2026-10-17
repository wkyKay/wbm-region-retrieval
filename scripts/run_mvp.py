#!/usr/bin/env python3
"""Prepare a protected 300-map subset and run the real unified MPS pipeline."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from wbm_retrieval.common import (budget_name, csv_rows, file_hash, load_config, read_json,
                                  write_csv, write_json)
from wbm_retrieval.data import Protocol


def protected_files(source_config):
    source = load_config(source_config)
    paths = [Path(source_config).resolve()] + [Path(source[key]) for key in (
        "data_file", "split_manifest", "query_manifest", "candidate_manifest")]
    return {str(path): file_hash(path) for path in paths}


def build_subset(config):
    """Remap only copied rows, retaining an explicit original-ID provenance table."""
    if config["mvp"]["maps"] != 300:
        raise ValueError("This reviewed MVP protocol requires exactly 300 maps")
    source_path = ROOT / config["mvp"]["source_config"]
    source = load_config(source_path)
    output = Path(config["output_dir"])
    if output == Path(source["output_dir"]) or not output.is_relative_to(ROOT / "artifacts"):
        raise ValueError("MVP output must be a separate directory inside artifacts/")
    for key in ("data_file", "split_manifest", "query_manifest", "candidate_manifest"):
        if not Path(config[key]).is_relative_to(output):
            raise ValueError(f"MVP {key} must stay inside its output directory")
    fingerprints = protected_files(source_path)
    marker = output / "subset.json"
    if marker.exists():
        existing = read_json(marker)
        if existing["protected_sources"] != fingerprints:
            raise ValueError("MVP source data changed; use a new isolated output directory")
        for path, expected in existing["subset_sha256"].items():
            if file_hash(path) != expected:
                raise ValueError(f"MVP subset was modified: {path}")
        return existing
    rows = csv_rows(source["split_manifest"])
    label_keys = sorted((key for key in rows[0] if key.startswith("label_") and key[6:].isdigit()),
                        key=lambda key: int(key[6:]))
    if len(label_keys) != 8:
        raise ValueError("The 300-map MVP quotas expect the source dataset's eight base labels")
    groups = {}
    for row in rows:
        signature = tuple(i for i, key in enumerate(label_keys) if int(row[key]))
        groups.setdefault((row["split"], signature), []).append(int(row["sample_id"]))
    signatures = [(i,) for i in range(8)] + [tuple(value) for value in config["mvp"]["source_mixed_signatures"]]
    rng = np.random.default_rng(config["seed"])
    selected, splits = [], []
    for split, single_count, mixed_count in (("train", 20, 25), ("valid", 2, 7), ("test", 4, 14)):
        for signature in signatures:
            count = single_count if len(signature) == 1 else mixed_count
            available = groups.get((split, signature), [])
            if len(available) < count:
                raise ValueError(f"Insufficient source {split} samples for {signature}: need {count}")
            chosen = rng.choice(available, count, replace=False).tolist()
            selected.extend(chosen)
            splits.extend([split] * count)
    if len(selected) != 300 or len(set(selected)) != 300:
        raise ValueError("MVP must contain 300 distinct source wafers")
    with np.load(source["data_file"], allow_pickle=False) as archive:
        all_labels = archive["arr_1"]
        all_maps = archive["arr_0"]
        labels = all_labels[selected].astype(np.uint8)
        maps = all_maps[selected].astype(np.uint8)
    del all_maps, all_labels
    output.mkdir(parents=True, exist_ok=True)
    Path(config["data_file"]).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(config["data_file"], maps, labels)
    manifest, provenance = [], []
    for new_id, (old_id, split) in enumerate(zip(selected, splits)):
        row = {"sample_id": new_id, "original_index": new_id, "valid_index": new_id, "split": split}
        row.update({key: int(labels[new_id, i]) for i, key in enumerate(label_keys)})
        manifest.append(row)
        provenance.append({"sample_id": new_id, "source_map_id": old_id, "source_split": split})
    write_csv(config["split_manifest"], manifest, list(manifest[0]))
    queries = [i for i, split in enumerate(splits) if split == "test"]
    write_csv(config["query_manifest"], ({"sample_id": q} for q in queries), ["sample_id"])
    write_csv(config["candidate_manifest"], ({"query_id": q, "candidate_id": c} for q in queries for c in queries if q != c),
              ["query_id", "candidate_id"])
    write_csv(output / "source_mapping.csv", provenance, list(provenance[0]))
    protocol = Protocol(config, labels)
    result = {"protected_sources": fingerprints, "protocol": protocol.summary(), "seed": config["seed"],
              "subset_sha256": {config[key]: file_hash(config[key]) for key in (
                  "data_file", "split_manifest", "query_manifest", "candidate_manifest")},
              "signatures": [list(value) for value in signatures], "scope": "MVP correctness only, not formal performance"}
    write_json(marker, result)
    return read_json(marker)


def run_stage(config_path, stage, output, durations, resume=False):
    command = [sys.executable, "-u", str(ROOT / "run_experiment.py"), "--config", str(config_path), "--stage", stage]
    if resume:
        command.append("--resume")
    key = f"resume_{stage}" if resume else stage
    started = time.perf_counter()
    print(f"[MVP] Starting {key}", flush=True)
    with (output / "run.log").open("a", encoding="utf-8") as log:
        log.write(f"\n[MVP] {key}: {' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env=os.environ.copy())
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    durations[key] = time.perf_counter() - started
    if code:
        raise RuntimeError(f"Stage {key} failed with exit code {code}; see {output / 'run.log'}")
    print(f"[MVP] Completed {key} in {durations[key]:.1f}s", flush=True)


def validate_outputs(config):
    import torch
    from wbm_retrieval.training import load_checkpoint, parameter_hash

    output = Path(config["output_dir"])
    protocol = Protocol(config)
    names = ["handcrafted", "proposed"] + [budget_name(b) for b in config["global"]["budgets"]]
    models, checksums = {}, {}
    for model_name in ["local"] + names[2:]:
        folder = output / "models" / model_name
        history = read_json(folder / "history.json")
        summary = read_json(folder / "training_summary.json")
        if not history or not all(np.isfinite([h["train_loss"], h["valid_loss"]]).all() for h in history):
            raise AssertionError(f"Empty or non-finite history for {model_name}")
        if summary["training_device"] != "mps" or summary["optimizer_steps"] < 1:
            raise AssertionError(f"No actual MPS optimizer updates for {model_name}")
        for filename in ("best.pt", "last.pt"):
            path = folder / filename
            checkpoint = load_checkpoint(path)
            if checkpoint["training_device"] != "mps" or "mps_rng" not in checkpoint:
                raise AssertionError(f"Missing MPS training/checkpoint provenance for {model_name}")
            state = checkpoint["model"]
            if any(not torch.isfinite(value).all() for value in state.values() if value.is_floating_point()):
                raise AssertionError(f"Non-finite checkpoint parameters for {model_name}")
            if parameter_hash(state) == checkpoint["initial_parameter_sha256"]:
                raise AssertionError(f"Parameters did not change for {model_name}")
            if checkpoint["early_stopping"]["last_epoch"] != checkpoint["epoch"]:
                raise AssertionError("Early stopping/checkpoint epochs disagree")
            if checkpoint["optimizer_steps"] < 1:
                raise AssertionError("Checkpoint has no optimizer steps")
            if filename == "best.pt" and not np.isclose(checkpoint["history"][-1]["valid_loss"], min(h["valid_loss"] for h in history)):
                raise AssertionError(f"best.pt is not the lowest validation loss for {model_name}")
            checksums[str(path)] = file_hash(path)
            del checkpoint, state
            gc.collect()
        for filename in ("history.json", "training_summary.json"):
            path = folder / filename
            checksums[str(path)] = file_hash(path)
        models[model_name] = summary
    initial_encoders = {summary["initial_encoder_sha256"] for summary in models.values()}
    specs = [summary["encoder_spec"] for summary in models.values()]
    if len(initial_encoders) != 1 or any(spec != specs[0] for spec in specs):
        raise AssertionError("Local and supervised models must share architecture and initial encoder weights")
    if specs[0]["embedding_dim"] != 512 or any(summary["optimizer"] != "SGD" for summary in models.values()):
        raise AssertionError("The MVP must exercise the revised 512-D WaPIRL/SGD models")
    with np.load(output / "cache/test_regions.npz", allow_pickle=False) as metadata:
        from wbm_retrieval.methods.handcrafted import moment_orders
        expected_dim = len(moment_orders(config["handcrafted"]["max_order"]))
        if metadata["descriptor"].shape[1] != expected_dim:
            raise AssertionError("MVP did not prepare the requested Zernike descriptors")
    descriptor_info = read_json(output / "retrieval/handcrafted/retrieval_info.json")
    if descriptor_info["representation"] != "zernike":
        raise AssertionError("Handcrafted retrieval must use Zernike Euclidean scores")
    rankings = {}
    for name in names:
        folder = output / "retrieval" / name
        rows = csv_rows(folder / "rankings.csv")
        by_query = {}
        for row in rows:
            by_query.setdefault(int(row["query_id"]), []).append(int(row["candidate_id"]))
            if not np.isfinite(float(row["similarity_score"])):
                raise AssertionError("Non-finite retrieval score")
        if set(by_query) != set(protocol.queries):
            raise AssertionError(f"Missing test queries for {name}")
        for q, candidates in by_query.items():
            if len(candidates) != 59 or len(set(candidates)) != 59 or q in candidates or set(candidates) != set(protocol.candidates[q]):
                raise AssertionError(f"Candidate pool mismatch for {name}, query {q}")
        metrics = read_json(folder / "metrics.json")
        if any(value is not None and not np.isfinite(value) for value in metrics["metrics"].values()):
            raise AssertionError("Non-finite evaluation metric")
        rankings[name] = {"rows": len(rows), "query_count": len(by_query), "metrics": metrics["metrics"]}
    comparison = csv_rows(output / "comparison.csv")
    if len(comparison) != 5 or {row["method"] for row in comparison} != set(names):
        raise AssertionError("Comparison must contain all five method/budget results")
    return {"models": models, "rankings": rankings, "training_file_sha256": checksums,
            "backbone_comparability": {"same_architecture": True, "same_initial_encoder_weights": True,
                                       "encoder_spec": specs[0]},
            "handcrafted_descriptor": {**config["handcrafted"], "dimensions": expected_dim}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/mvp_mps_300.json"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue this isolated MVP after interruption")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if config["device"] != "mps":
        raise ValueError("This MVP explicitly requires MPS")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ.setdefault("MPLCONFIGDIR", str(Path(config["output_dir"]) / "matplotlib"))
    # Fail before preparing data if the process is unable to access Metal.
    if not args.prepare_only:
        import torch
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable; run this command in a Metal-enabled local terminal")
    subset = build_subset(config)
    output = Path(config["output_dir"])
    print(f"[MVP] Protocol: {subset['protocol']}", flush=True)
    if args.prepare_only:
        return
    started = time.perf_counter()
    report = {"status": "running", "device": "mps", "torch": torch.__version__, "mps_fallback": False,
              "scope": "300-map MVP only", "stage_seconds": {}}
    write_json(output / "mvp_report.json", report)
    try:
        for stage in ("audit", "prepare", "train-local", "train-global", "retrieve", "evaluate", "geometry"):
            run_stage(config_path, stage, output, report["stage_seconds"], resume=args.resume and stage.startswith("train-"))
            write_json(output / "mvp_report.json", report)
        verification = validate_outputs(config)
        run_stage(config_path, "all", output, report["stage_seconds"], resume=True)
        for path, expected in verification["training_file_sha256"].items():
            if file_hash(path) != expected:
                raise AssertionError(f"Completed training changed during --resume: {path}")
        after_resume = validate_outputs(config)
        current_protected = protected_files(ROOT / config["mvp"]["source_config"])
        if current_protected != subset["protected_sources"]:
            raise AssertionError("A protected formal experiment file changed")
        report.update(status="passed", total_seconds=time.perf_counter() - started,
                      verification=after_resume, completed_resume_keeps_training_unchanged=True,
                      formal_inputs_unchanged=True, protected_sources=current_protected)
        print(f"[MVP] PASSED in {report['total_seconds']:.1f}s; report: {output / 'mvp_report.json'}", flush=True)
    except BaseException:
        report.update(status="failed", total_seconds=time.perf_counter() - started, error=traceback.format_exc())
        raise
    finally:
        write_json(output / "mvp_report.json", report)


if __name__ == "__main__":
    main()
