"""Five single-augmentation runs with the unchanged frozen F training recipe."""
from copy import deepcopy
from pathlib import Path
import math
import platform
import shutil
import sys
import time

import numpy as np
import torch
import cv2

from cnn_augmentation.augmentation import AUGMENTATION_SPECS, VARIANTS, PaperViews
from experiments.proposed._shared.ef_suite import RegionViews, build_model
from experiments.proposed.main.method import MemoryBank, nce_loss
from f_region_structure.state import now, read_signed, record, seal, signed, verified
from wbm_retrieval.common import digest, file_hash, seed_everything, write_csv, write_json
from wbm_retrieval.training import (loader, optimization, optimizer_to_device, parameter_hash,
                                    restore, save_checkpoint, validation_rng)

INITIAL_HASH = "df8cfa59821feeff548ef6e963339b770ecc058fad7c9b4ac0e2727a50a57260"
SNAPSHOTS = (15, 25, 50, 75, 100)
# Every value counts anchor presentations, not distinct dataset items across epochs.
AUDIT_COUNTERS = {
    "augmentation_applied_count": "applied", "augmentation_no_op_count": "no_op",
    "empty_target_before_count": "empty_target_before", "empty_target_after_count": "empty_target_after",
    "newly_empty_target_count": "newly_empty_target", "target_before_pixels": "target_before_pixels",
    "target_after_pixels": "target_after_pixels", "target_retained_source_pixels": "target_retained_source_pixels",
    "target_retention_eligible_count": "target_retention_eligible",
    "target_retention_fraction_sum": "target_retention_fraction", "changed_positions": "changed_positions",
    "support_before_pixels": "support_before_pixels", "support_after_pixels": "support_after_pixels",
    "support_retained_source_pixels": "support_retained_source_pixels",
    "support_retention_fraction_sum": "support_retention_fraction",
    "support_retention_eligible_count": "support_retention_eligible",
    "noise_selected_pixels": "noise_selected_pixels", "noise_added_defects": "noise_added_defects",
    "noise_removed_defects": "noise_removed_defects", "noise_changed_pixels": "noise_changed_pixels",
}


def audit_rates(totals, samples):
    if samples <= 0:
        raise ValueError("Augmentation audit requires observed samples")
    eligible = totals["target_retention_eligible_count"]
    support_eligible = totals["support_retention_eligible_count"]
    return {"applied_fraction": totals["augmentation_applied_count"]/samples,
            "no_op_fraction": totals["augmentation_no_op_count"]/samples,
            "empty_target_after_fraction": totals["empty_target_after_count"]/samples,
            "newly_empty_target_fraction": totals["newly_empty_target_count"]/samples,
            "mean_target_retention_fraction": totals["target_retention_fraction_sum"]/eligible if eligible else None,
            "mean_support_retention_fraction": totals["support_retention_fraction_sum"]/support_eligible if support_eligible else None}


def audit_history(history, variant):
    totals = {key: sum(row[key] for row in history) for key in AUDIT_COUNTERS}
    samples = sum(row["sample_count"] for row in history)
    return {"variant": variant, "epochs": len(history), "samples": samples,
            "totals": totals, "rates": audit_rates(totals, samples),
            "definitions": {
                "applied": "transform executed; cutout has a 0.5 application gate",
                "no_op": "all three output planes byte-equal to original planes",
                "target_retention": "unique original target pixel IDs represented at output target pixels, using the same actual nearest resampling; averaged only over nonempty original targets",
                "target_after_pixels": "output count, may increase under crop magnification; not source-space retention",
                "empty_target": "empty targets are reported and retained without target-aware resampling",
                "noise_selected": "Bernoulli-selected valid pixels, including same-value replacements",
                "samples": "anchor presentations, including repeated tokens across epochs"}}


def _validate_recipe(inputs):
    config, spec = inputs["config"], inputs["spec"]
    expected = {"epochs": 100, "learning_rate": .01, "momentum": .9, "weight_decay": .001,
                "projector_dim": 128, "temperature": .07, "num_negatives": 5000,
                "memory_momentum": .5, "optimizer": "sgd", "scheduler": "wapirl_cosine",
                "projector_type": "linear", "negative_sampling": "with_replacement"}
    if any(config["local"].get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen 100-epoch optimization recipe changed")
    if (config["seed"] != 2026 or config["batch_size"] != 256 or config["embedding_dim"] != 512
            or config["input_size"] != 96 or config["local"].get("early_stopping", {}).get("enabled", False)
            or config["local"].get("opening", {}).get("enabled", False)
            or spec.get("preprocess") != "fixed_window" or spec.get("input_channels") != 3
            or spec.get("stem_stride") != 1 or spec.get("scale_range") is not None
            or spec.get("window_size") != 52 or spec.get("max_shift") != 0):
        raise ValueError("Frozen data/model/view recipe changed")
    if len(inputs["datasets"]["train"]["patches"]) != 99786:
        raise ValueError("Expected all 99,786 frozen deduplicated train-region tokens")


def _identity(output, inputs, variant):
    source = Path(inputs["project_root"])
    code_paths = [Path(__file__), Path(__file__).with_name("augmentation.py"),
                  source / "experiments/proposed/_shared/ef_suite.py",
                  source / "experiments/proposed/main/method.py",
                  source / "wbm_retrieval/training.py", source / "wbm_retrieval/models.py"]
    science = {"schema_version": 1, "variant": variant, "seed": 2026, "batch_size": 256,
               "refinement_context_signature": inputs.get("refinement_context_signature"),
               "plan_signature": inputs.get("plan_signature"),
               "local": inputs["config"]["local"], "input": inputs["spec"],
               "source_manifest": inputs["source_manifest"]["signature"],
               "source_training_signature": inputs["run_manifest"]["signature"],
               "source_protection": inputs["protection"],
               "train_prepared": inputs["datasets"]["train"]["prepared_marker"],
               "valid_prepared": inputs["datasets"]["valid"]["prepared_marker"],
               "initial_parameter_sha256": INITIAL_HASH,
               "augmentation_spec": AUGMENTATION_SPECS[variant],
               "runtime": {"opencv": cv2.__version__, "numpy": np.__version__, "torch": torch.__version__},
               "view_policy": "unchanged cached96 first view; single augmentation second view; full-window inference",
               "selection": "fixed_epoch_100_no_best_loss_or_test_selection",
               "validation": "unchanged_original_rotation_NCE_diagnostic_only",
               "code": {str(p.relative_to(source)): file_hash(p) for p in code_paths}}
    signature = digest(science)
    path = output / "run_manifest.json"
    if path.exists():
        manifest = read_signed(path)
        if manifest["training_signature"] != signature or manifest["science"] != science:
            raise ValueError("Training inputs/settings/code changed; use a new output root")
    else:
        instant = now()
        key = f"cnn_aug_{variant}_seed2026_epoch100"
        manifest = signed({"run_id": instant.strftime("%Y%m%dT%H%M%S%z")+"__"+key,
                           "started_at": instant.isoformat(), "training_signature": signature,
                           "science": science, "environment": {"python": sys.version,
                            "platform": platform.platform(), "torch": torch.__version__,
                            "numpy": np.__version__, "opencv": cv2.__version__, "cuda": torch.version.cuda}})
        record(path, manifest)
    return signature, manifest


def _write_history(output, history):
    write_json(output / "history.json", history)
    if history:
        temporary = output / "history.csv.tmp"
        write_csv(temporary, history, list(history[0]))
        temporary.replace(output / "history.csv")


def _snapshot_last(output, epoch):
    """Epoch commit precedes snapshots; recover a crash in between without retraining."""
    if epoch in SNAPSHOTS:
        snapshots = output / "checkpoints"
        snapshots.mkdir(exist_ok=True)
        target = snapshots / f"epoch_{epoch:03d}.pt"
        if not target.exists():
            temporary = target.with_suffix(".tmp")
            shutil.copyfile(output / "last.pt", temporary)
            temporary.replace(target)


def train_variant(output, inputs, variant, *, device="cuda:0", workers=4, resume=False):
    """Train one authorized intervention; return its sealed final checkpoint path."""
    if variant not in VARIANTS:
        raise ValueError("Exactly one named paper augmentation must be supplied")
    _validate_recipe(inputs)
    if any(not isinstance(inputs.get(key), str) or not inputs[key]
           for key in ("refinement_context_signature", "plan_signature")):
        raise ValueError("Formal training requires signed refinement context and experiment plan")
    root = Path(output)
    output = root / "training" / variant
    output.mkdir(parents=True, exist_ok=True)
    signature, manifest = _identity(output, inputs, variant)
    if verified(output, signature):
        return output / "checkpoint.pt"
    config = deepcopy(inputs["config"])
    config.update(device=str(device), num_workers=int(workers))
    settings, spec = config["local"], inputs["spec"]
    seed_everything(config["seed"])
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(config, spec).to(device)
    initial_hash = parameter_hash(model)
    if initial_hash != INITIAL_HASH:
        raise ValueError(f"Initial model differs from frozen seed2026: {initial_hash}")
    optimizer, scheduler = optimization(model, settings)
    start, _, history, checkpoint = restore(output / "last.pt", model, optimizer, scheduler,
                                             signature, resume)
    if checkpoint:
        if len(history) != start or checkpoint["optimizer_steps"] != start * 390:
            raise ValueError("Resumable epoch/history/optimizer-step counters disagree")
        _snapshot_last(output, start)
    optimizer_to_device(optimizer, device)
    cache = inputs["datasets"]["train"]["cache"]
    train = PaperViews(cache, "train", spec, config["seed"], variant)
    valid = RegionViews(inputs["datasets"]["valid"]["cache"], "valid", spec, config["seed"]+1)
    bank = MemoryBank(len(train), settings["projector_dim"], device, settings["memory_momentum"])
    steps = int(checkpoint["optimizer_steps"]) if checkpoint else 0
    if checkpoint:
        if checkpoint["initial_parameter_sha256"] != initial_hash:
            raise ValueError("Resume initialization identity changed")
        bank.buffer.copy_(checkpoint["memory"].to(device))
    else:
        # As in the original run, eval-mode bank initialization never updates BN.
        bank.initialize(model["encoder"], model["projector"], loader(train, config, config["seed"], False), device)
    for epoch in range(start, settings["epochs"]):
        started = time.monotonic()
        train.epoch = epoch
        model.train()
        total, count = 0., 0
        totals = {key: 0. if key.endswith("fraction_sum") else 0 for key in AUDIT_COUNTERS}
        for batch, (indices, original, view, stats) in enumerate(loader(train, config, config["seed"]+epoch, True), 1):
            indices = indices.to(device)
            z, zt = model["projector"](model["encoder"](torch.cat([original.to(device), view.to(device)]))).chunk(2)
            anchors = bank.buffer[indices].detach().clone()
            # Real frozen data has 390 batches/epoch and never enters the legacy
            # one-batch fallback; refuse a changed data regime instead of changing it.
            if len(indices) == len(train):
                raise ValueError("Official training requires the unchanged 99,786-token pool")
            base_ids = bank.negative_indices(indices, settings["num_negatives"])
            negatives = bank.buffer[base_ids]
            loss = .5 * (nce_loss(anchors, z, negatives, settings["temperature"]) +
                         nce_loss(anchors, zt, negatives, settings["temperature"]))
            for name, source_name in AUDIT_COUNTERS.items():
                value = stats[source_name].sum().item()
                totals[name] += float(value) if name.endswith("fraction_sum") else int(value)
            value = loss.item()
            if not math.isfinite(value):
                raise FloatingPointError(f"{variant}: nonfinite loss at epoch {epoch+1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            bank.update(indices, z.detach())
            steps += 1
            total += value * len(indices)
            count += len(indices)
            if batch % 100 == 0:
                print(f"{variant} epoch={epoch+1} batch={batch}: loss={value:.5f}", flush=True)
        model.eval()
        val_sum, val_count = 0., 0
        with torch.no_grad(), validation_rng(device, config["seed"]+999):
            for _, original, view, _ in loader(valid, config, config["seed"], False):
                z, zt = model["projector"](model["encoder"](torch.cat([original.to(device), view.to(device)]))).chunk(2)
                chosen = torch.randint(len(bank.buffer), (settings["num_negatives"],), device=device)
                value = nce_loss(z, zt, bank.buffer[chosen], settings["temperature"]).item()
                val_sum += value * len(original)
                val_count += len(original)
        valid_nce = val_sum / val_count
        if not math.isfinite(valid_nce):
            raise FloatingPointError(f"{variant}: nonfinite validation loss")
        if count != 99786 or batch != 390 or steps != (epoch+1)*390:
            raise ValueError("Training epoch did not cover the full frozen pool and budget")
        history.append({"epoch": epoch+1, "train_loss": total/count, "valid_nce": valid_nce,
                        "learning_rate": optimizer.param_groups[0]["lr"], "optimizer_steps": steps,
                        "seconds": time.monotonic()-started, "sample_count": count,
                        "negative_positions": count*settings["num_negatives"], **totals,
                        **audit_rates(totals, count)})
        scheduler.step()
        extra = {"memory": bank.buffer.detach().cpu(), "initial_parameter_sha256": initial_hash,
                 "variant": variant, "input_spec": spec, "optimizer_steps": steps,
                 "training_device": str(device), "started_at": manifest["started_at"],
                 "run_id": manifest["run_id"], "selection": "fixed_epoch_100"}
        save_checkpoint(output / "last.pt", model, optimizer, scheduler, epoch+1, valid_nce,
                        history, signature, **extra)
        _snapshot_last(output, epoch+1)
        _write_history(output, history)
        write_json(output / "status.json", {"status": "running", "epoch": epoch+1,
                                            "optimizer_steps": steps, "updated_at": now().isoformat()})
        print(f"{variant} epoch {epoch+1}/100: train={total/count:.5f}, valid_nce={valid_nce:.5f}", flush=True)
    if steps != 39000 or len(history) != 100:
        raise ValueError("Official training must complete exactly 100 epochs / 39,000 optimizer steps")
    # Compact final inference checkpoint; resumable optimizer/RNG/bank stay in last.pt.
    final = {"model": model.state_dict(), "epoch": 100, "signature": signature, "variant": variant,
             "input_spec": spec, "initial_parameter_sha256": initial_hash, "optimizer_steps": steps,
             "selection": "fixed_epoch_100", "run_id": manifest["run_id"], "started_at": manifest["started_at"]}
    temporary = output / "checkpoint.tmp"
    torch.save(final, temporary)
    temporary.replace(output / "checkpoint.pt")
    _write_history(output, history)
    augmentation_audit = audit_history(history, variant)
    write_json(output / "augmentation_audit.json", augmentation_audit)
    write_json(output / "training_summary.json", {"completed": True, "epochs_completed": 100,
        "max_epochs": 100, "optimizer_steps": steps, "initial_parameter_sha256": initial_hash,
        "final_parameter_sha256": parameter_hash(model), "checkpoint_selection": "fixed_epoch_100",
        "augmentation_audit": augmentation_audit,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "started_at": manifest["started_at"], "completed_at": now().isoformat(), "run_id": manifest["run_id"]})
    write_json(output / "status.json", {"status": "completed", "epoch": 100, "optimizer_steps": steps,
                                        "updated_at": now().isoformat()})
    seal(output, signature)
    return output / "checkpoint.pt"
