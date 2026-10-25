"""G-only training, adapted from the frozen A-D loop; no shared module patches."""

from __future__ import annotations

import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .suite import RegionViews, build_model, evaluate
from wbm_retrieval.common import digest, get_device, read_json, seed_everything, write_json
from wbm_retrieval.methods.proposed import MemoryBank, nce_loss
from wbm_retrieval.training import (loader, load_checkpoint, optimization, optimizer_to_device, parameter_hash,
                       restore, save_checkpoint, validation_rng)


def run_signature(suite_signature, seed, name):
    return digest({"suite": suite_signature, "seed": seed, "variant": name})


def save_summary(output, settings, epoch, best_nce, extra, model):
    write_json(output / "training_summary.json", {"epochs_completed": epoch, "max_epochs": settings["epochs"],
        "completed": epoch == settings["epochs"], "best_nce_epoch": extra["best_nce_epoch"], "best_nce": best_nce,
        "best_retrieval_epoch": extra["best_retrieval_epoch"], "best_retrieval": extra["best_retrieval"],
        "optimizer_steps": extra["optimizer_steps"], "initial_parameter_sha256": extra["initial_parameter_sha256"],
        "final_parameter_sha256": parameter_hash(model), "opening_enabled": False, "input_spec": extra["input_spec"],
        "num_negatives": settings["num_negatives"], "aggregation": "query_region_count"})


def train_variant(config, suite, suite_signature, name, resume=False, stop_after=None):
    if name != "G" or config["local"]["num_negatives"] != 1000:
        raise ValueError("This trainer only accepts G with 1000 negatives")
    spec, settings = suite["variants"][name], config["local"]
    seed_everything(config["seed"])
    device = get_device(config["device"])
    root = Path(config["output_dir"])
    output = root / f"seed_{config['seed']}" / name
    output.mkdir(parents=True, exist_ok=True)
    signature = run_signature(suite_signature, config["seed"], name)
    model = build_model(config, spec).to(device)
    initial_hash = parameter_hash(model)
    optimizer, scheduler = optimization(model, settings)
    start, best_nce, history, checkpoint = restore(output / "last.pt", model, optimizer, scheduler, signature, resume)
    optimizer_to_device(optimizer, device)
    if checkpoint and checkpoint["initial_parameter_sha256"] != initial_hash:
        raise ValueError("Initialization changed (e.g. torch version); do not mix runs in one suite")
    manifest = {"signature": signature, "variant": name, "input": spec, "config": config,
                "initial_parameter_sha256": initial_hash, "selection_metric": suite["selection_metric"],
                "selection_note": "Uses valid labels for checkpoint selection, never for NCE optimization; no test selection",
                "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                                "numpy": np.__version__, "device": str(device), "cuda": torch.version.cuda,
                                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}}
    if not checkpoint:
        write_json(output / "run_manifest.json", manifest)
    if start >= settings["epochs"]:
        # A process may have died after committing last.pt but before writing these JSON files.
        write_json(output / "history.json", history)
        save_summary(output, settings, start, best_nce, checkpoint, model)
        print(f"{name}: already completed {start} epochs", flush=True)
        return True
    train = RegionViews(root / "cache", "train", spec, config["seed"])
    valid = RegionViews(root / "cache", "valid", spec, config["seed"]+1)
    if len(train) < 2 or not len(valid):
        raise ValueError("Need >=2 train regions and >=1 valid region")
    bank = MemoryBank(len(train), settings["projector_dim"], device, settings["memory_momentum"])
    best_retrieval = checkpoint["best_retrieval"] if checkpoint else None
    best_retrieval_epoch = checkpoint["best_retrieval_epoch"] if checkpoint else 0
    best_nce_epoch = checkpoint["best_nce_epoch"] if checkpoint else 0
    steps = checkpoint["optimizer_steps"] if checkpoint else 0
    if checkpoint:
        bank.buffer.copy_(checkpoint["memory"].to(device))
    else:
        print(f"{name}: evaluating random initialization (not eligible for best checkpoint)", flush=True)
        evaluate(config, suite, "valid", output / "valid/random", model=model, spec=spec, device=device)
        bank.initialize(model["encoder"], model["projector"], loader(train, config, config["seed"], False), device)
    for epoch in range(start, settings["epochs"]):
        started = time.monotonic()
        train.epoch = epoch
        model.train()
        total, count = 0., 0
        for batch, (indices, original, view, _) in enumerate(loader(train, config, config["seed"]+epoch, True), 1):
            indices = indices.to(device)
            z, zt = model["projector"](model["encoder"](torch.cat([original.to(device), view.to(device)]))).chunk(2)
            anchors = bank.buffer[indices].detach().clone()
            if len(indices) == len(train):
                losses = []
                for i, index in enumerate(indices):
                    negatives = bank.buffer[bank.negative_indices(index.reshape(1), settings["num_negatives"])]
                    losses.append(.5 * (nce_loss(anchors[i:i+1], z[i:i+1], negatives, settings["temperature"]) +
                                         nce_loss(anchors[i:i+1], zt[i:i+1], negatives, settings["temperature"])))
                loss = torch.stack(losses).mean()
            else:
                negatives = bank.buffer[bank.negative_indices(indices, settings["num_negatives"])]
                loss = .5 * (nce_loss(anchors, z, negatives, settings["temperature"]) +
                             nce_loss(anchors, zt, negatives, settings["temperature"]))
            value = loss.item()
            if not math.isfinite(value):
                raise FloatingPointError(f"{name}: nonfinite train loss at epoch {epoch+1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            bank.update(indices, z.detach())
            steps += 1
            total += value * len(indices)
            count += len(indices)
            if batch % 100 == 0:
                print(f"{name} epoch={epoch+1} batch={batch}: loss={value:.5f}", flush=True)
        model.eval()
        val_sum, val_count = 0., 0
        with torch.no_grad(), validation_rng(device, config["seed"]+999):
            for _, original, view, _ in loader(valid, config, config["seed"], False):
                z, zt = model["projector"](model["encoder"](torch.cat([original.to(device), view.to(device)]))).chunk(2)
                chosen = torch.randint(len(bank.buffer), (settings["num_negatives"],), device=device)
                value = nce_loss(z, zt, bank.buffer[chosen], settings["temperature"]).item()
                val_sum += value * len(original)
                val_count += len(original)
        val_loss = val_sum / val_count
        if not math.isfinite(val_loss):
            raise FloatingPointError(f"{name}: nonfinite valid loss")
        nce_improved = val_loss < best_nce
        if nce_improved:
            best_nce, best_nce_epoch = val_loss, epoch+1
        score, retrieval_improved = None, False
        if epoch == 0 or (epoch+1) % suite["evaluate_every"] == 0 or epoch+1 == settings["epochs"]:
            result = evaluate(config, suite, "valid", output / "valid" / f"epoch_{epoch+1:03d}",
                              model=model, spec=spec, device=device)
            score = result["metrics"][suite["selection_metric"]]
            if score is None:
                raise ValueError("No valid queries have defined selection metric")
            retrieval_improved = best_retrieval is None or score > best_retrieval
            if retrieval_improved:
                best_retrieval, best_retrieval_epoch = score, epoch+1
        history.append({"epoch": epoch+1, "train_loss": total/count, "valid_nce": val_loss,
                        suite["selection_metric"]: score, "learning_rate": optimizer.param_groups[0]["lr"],
                        "optimizer_steps": steps, "seconds": time.monotonic()-started})
        scheduler.step()
        extra = {"memory": bank.buffer.detach().cpu(), "initial_parameter_sha256": initial_hash,
                 "variant": name, "input_spec": spec, "optimizer_steps": steps,
                 "best_nce_epoch": best_nce_epoch, "best_retrieval": best_retrieval,
                 "best_retrieval_epoch": best_retrieval_epoch, "training_device": str(device),
                 "num_negatives": settings["num_negatives"], "aggregation": "query_region_count"}
        if nce_improved:
            save_checkpoint(output / "best_nce.pt", model, optimizer, scheduler, epoch+1, best_nce, history, signature, **extra)
        if retrieval_improved:
            save_checkpoint(output / "best_retrieval.pt", model, optimizer, scheduler, epoch+1, best_nce, history, signature, **extra)
        save_checkpoint(output / "last.pt", model, optimizer, scheduler, epoch+1, best_nce, history, signature, **extra)
        write_json(output / "history.json", history)
        save_summary(output, settings, epoch+1, best_nce, extra, model)
        print(f"{name} epoch {epoch+1}/{settings['epochs']}: train={total/count:.5f}, valid_nce={val_loss:.5f}, "
              f"NDCG@10={score}, best_epoch={best_retrieval_epoch}", flush=True)
        if stop_after is not None and epoch+1 >= stop_after:
            return epoch+1 == settings["epochs"]
    return True


def evaluate_variant(config, suite, suite_signature, name, split="valid", checkpoint_kind="both"):
    device = get_device(config["device"])
    output = Path(config["output_dir"]) / f"seed_{config['seed']}" / name
    summary = read_json(output / "training_summary.json")
    if not summary["completed"]:
        raise ValueError(f"{name}: training is incomplete; use --resume before final evaluation")
    kinds = ("best_nce", "best_retrieval") if checkpoint_kind == "both" else (checkpoint_kind,)
    for kind in kinds:
        checkpoint = load_checkpoint(output / f"{kind}.pt")
        if checkpoint["signature"] != run_signature(suite_signature, config["seed"], name):
            raise ValueError("Checkpoint does not belong to this ablation run")
        model = build_model(config, suite["variants"][name]).to(device)
        model.load_state_dict(checkpoint["model"])
        target = output / split / kind
        result = evaluate(config, suite, split, target, model=model, spec=suite["variants"][name], device=device,
                          rankings=True, bootstrap=config["evaluation"]["bootstrap_samples"])
        result.update(epoch=checkpoint["epoch"], checkpoint=str(output / f"{kind}.pt"))
        write_json(target / "metrics.json", result)
        print(f"{name}/{kind}/{split}: epoch={checkpoint['epoch']} NDCG@10={result['metrics']['ContainmentNDCG@10']:.5f}", flush=True)
