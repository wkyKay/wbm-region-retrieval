"""Whole-map BCE training with deterministic resume and classification selection."""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from wbm_retrieval.common import csv_rows, digest, file_hash, get_device, read_json, seed_everything, write_csv, write_json
from wbm_retrieval.training import (encoder_spec, load_checkpoint, optimization, optimizer_to_device,
                                    parameter_hash, restore, save_checkpoint)
from .suite import Classifier, WholeMaps, make_loader, method_name, split_protocol
from .evaluation import classification_metrics, predict, retrieval_report, save_classification
from .figures import learning_curves, classification_figures, budget_comparison


def run_signature(signature, budget):
    return digest({"suite": signature, "budget": budget})


def better_classifier(macro_f1, loss, best_f1, best_loss):
    return macro_f1 > best_f1 or (macro_f1 == best_f1 and loss < best_loss)


def save_summary(folder, config, model, history, state):
    write_json(folder / "history.json", history)
    write_json(folder / "training_summary.json", {
        "epochs_completed": len(history), "max_epochs": config["training"]["epochs"],
        "completed": len(history) >= config["training"]["epochs"],
        "selection": "highest valid macro_f1 at threshold 0.5; ties: lower valid BCE, then earlier epoch",
        "best_classifier_epoch": state["best_classifier_epoch"], "best_macro_f1": state["best_macro_f1"],
        "best_classifier_bce": state["best_classifier_bce"], "best_loss_epoch": state["best_loss_epoch"],
        "optimizer_steps": state["optimizer_steps"], "initial_parameter_sha256": state["initial_parameter_sha256"],
        "final_parameter_sha256": parameter_hash(model), "initial_encoder_sha256": state["initial_encoder_sha256"],
        "encoder_spec": encoder_spec(config), "classifier_head": "Linear(512,8)",
        "input": "whole map; raw==2; square padding; nearest resize 96", "threshold": config["threshold"]})


def train(config, protocol, signature, budget, resume=False, stop_after=None):
    root = Path(config["output_root"])
    folder = root / method_name(budget)
    models = folder / "models"
    models.mkdir(exist_ok=True)
    seed_everything(config["seed"])
    device = get_device(config["device"])
    model = Classifier().to(device)
    initial_hash, encoder_hash = parameter_hash(model), parameter_hash(model.encoder)
    optimizer, scheduler = optimization(model, config["training"])
    signature = run_signature(signature, budget)
    start, best_loss, history, checkpoint = restore(models / "last.pt", model, optimizer, scheduler, signature, resume)
    optimizer_to_device(optimizer, device)
    if checkpoint and (checkpoint["initial_parameter_sha256"] != initial_hash or checkpoint["torch_version"] != str(torch.__version__)):
        raise ValueError("Initialization or PyTorch version changed; use a new output root")
    state = checkpoint or {"best_macro_f1": -1., "best_classifier_bce": float("inf"), "best_classifier_epoch": 0,
                           "best_loss_epoch": 0, "optimizer_steps": 0, "initial_parameter_sha256": initial_hash,
                           "initial_encoder_sha256": encoder_hash}
    selected = [int(row["sample_id"]) for row in csv_rows(folder / "labeled_train_ids.csv")]
    if len(selected) < 2:
        raise ValueError("Need at least two selected training maps")
    train_set = WholeMaps(root, "train", selected, config["training"]["rotation"], config["seed"])
    diagnostic = WholeMaps(root, "train", read_json(folder / "train_diagnostic_ids.json"))
    valid_set = WholeMaps(root, "valid")
    valid_protocol = split_protocol(config, protocol, "valid")
    if checkpoint is None:
        import sklearn
        write_json(folder / "run_manifest.json", {"signature": signature, "budget": budget, "config": config,
                   "initial_parameter_sha256": initial_hash, "initial_encoder_sha256": encoder_hash,
                   "environment": {"python": sys.version, "platform": platform.platform(), "torch": str(torch.__version__),
                       "numpy": np.__version__, "sklearn": sklearn.__version__, "device": str(device),
                       "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}})
    if start >= config["training"]["epochs"]:
        save_summary(folder, config, model, history, state)
        learning_curves(folder)
        print(f"{folder.name}: training already complete at epoch {start}", flush=True)
        return
    for epoch in range(start, config["training"]["epochs"]):
        started = time.monotonic()
        train_set.epoch = epoch
        model.train()
        total, count = 0., 0
        learning_rate = optimizer.param_groups[0]["lr"]
        for _, images, targets in make_loader(train_set, config, training=True, epoch=epoch):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(images.to(device)), targets.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch+1}")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(targets)
            count += len(targets)
            state["optimizer_steps"] += 1
        _, train_y, train_p, train_bce = predict(model, diagnostic, config, device)
        train_metrics = classification_metrics(train_y, train_p, config["threshold"], groups=False)["metrics"]
        train_metrics["bce"] = train_bce
        valid_ids, valid_y, valid_p, valid_bce = predict(model, valid_set, config, device)
        valid_metrics = classification_metrics(valid_y, valid_p, config["threshold"], groups=False)["metrics"]
        selected_best = better_classifier(valid_metrics["macro_f1"], valid_bce, state["best_macro_f1"], state["best_classifier_bce"])
        if selected_best:
            state.update(best_macro_f1=valid_metrics["macro_f1"], best_classifier_bce=valid_bce, best_classifier_epoch=epoch+1)
        loss_best = valid_bce < best_loss
        if loss_best:
            best_loss, state["best_loss_epoch"] = valid_bce, epoch+1
        retrieval = None
        if epoch == 0 or (epoch+1) % config["evaluate_every"] == 0 or epoch+1 == config["training"]["epochs"]:
            result = retrieval_report(folder / "retrieval/valid_monitor" / f"epoch_{epoch+1:03d}", valid_ids, valid_p,
                                      valid_protocol, config, bootstrap=0)
            retrieval = result["metrics"]
        history.append({"epoch": epoch+1, "train_loss": total / count, "valid_loss": valid_bce,
                        "train_eval": train_metrics, "valid_metrics": valid_metrics, "valid_retrieval": retrieval,
                        "learning_rate": learning_rate, "optimizer_steps": state["optimizer_steps"],
                        "epoch_seconds": time.monotonic() - started})
        scheduler.step()
        extra = {key: state[key] for key in ("best_macro_f1", "best_classifier_bce", "best_classifier_epoch",
                 "best_loss_epoch", "optimizer_steps", "initial_parameter_sha256", "initial_encoder_sha256")}
        extra.update(torch_version=str(torch.__version__), encoder_spec=encoder_spec(config), threshold=config["threshold"],
                     budget=budget, num_labels=8)
        if device.type == "mps":
            extra["mps_rng"] = torch.mps.get_rng_state()
        for name, needed in (("best_classifier.pt", selected_best), ("best_loss.pt", loss_best), ("last.pt", True)):
            if needed:
                save_checkpoint(models / name, model, optimizer, scheduler, epoch+1, best_loss, history, signature, **extra)
        save_summary(folder, config, model, history, state)
        print(f"{folder.name} epoch {epoch+1}/{config['training']['epochs']}: "
              f"train BCE={total/count:.5f} valid BCE={valid_bce:.5f} valid macro-F1={valid_metrics['macro_f1']:.5f}", flush=True)
        if stop_after is not None and epoch+1 >= stop_after:
            break
    learning_curves(folder)


def evaluate(config, protocol, signature, budget, split):
    folder = Path(config["output_root"]) / method_name(budget)
    device = get_device(config["device"])
    path = folder / "models/best_classifier.pt"
    checkpoint = load_checkpoint(path)
    if checkpoint["signature"] != run_signature(signature, budget):
        raise ValueError("Classifier checkpoint is incompatible with this experiment")
    last = load_checkpoint(folder / "models/last.pt")
    if last["epoch"] < config["training"]["epochs"]:
        raise ValueError("Finish the configured training before final evaluation")
    del last
    model = Classifier().to(device)
    model.load_state_dict(checkpoint["model"])
    dataset = WholeMaps(config["output_root"], split)
    ids, labels, probabilities, loss = predict(model, dataset, config, device)
    classification = save_classification(folder / "classification" / split, ids, labels, probabilities, loss, config["threshold"])
    retrieval = retrieval_report(folder / "retrieval" / split, ids, probabilities, split_protocol(config, protocol, split), config)
    provenance = {"checkpoint": str(path), "checkpoint_sha256": file_hash(path), "epoch": checkpoint["epoch"],
                  "selection": "valid macro-F1; ties: lower BCE, then earlier epoch", "split": split,
                  "classification_maps": len(ids), "threshold": config["threshold"], "signature": signature}
    for group in ("classification", "retrieval"):
        write_json(folder / group / split / "provenance.json", provenance)
    classification_figures(folder, split, config["output_root"])
    print(f"{folder.name} {split}: macro-F1={classification['metrics']['macro_f1']:.5f}, "
          f"NDCG@10={retrieval['metrics']['ContainmentNDCG@10']}", flush=True)


def summarize(config, split):
    root = Path(config["output_root"])
    classifications, retrievals, combined = [], [], []
    for budget in config["budgets"]:
        name = method_name(budget)
        class_path = root / name / "classification" / split / "metrics.json"
        retrieval_path = root / name / "retrieval" / split / "metrics.json"
        if class_path.exists() and retrieval_path.exists():
            classification = {"method": name, **read_json(class_path)["metrics"]}
            retrieval = {"method": name, **read_json(retrieval_path)["metrics"]}
            classifications.append(classification)
            retrievals.append(retrieval)
            combined.append({**classification, **retrieval})
    for name, rows in (("classification", classifications), ("retrieval", retrievals)):
        if rows:
            write_csv(root / f"{split}_{name}_comparison.csv", rows, list(rows[0]))
            write_json(root / f"{split}_{name}_comparison.json", rows)
    budget_comparison(root, split, combined)
