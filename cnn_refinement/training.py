"""Two isolated 100-epoch CNN interventions using the frozen F training recipe."""
from copy import deepcopy
from pathlib import Path
import math
import platform
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from cnn_refinement.augmentation import ContextViews, prepare_context_windows
from experiments.proposed._shared.ef_suite import RegionViews, build_model
from experiments.proposed.main.method import MemoryBank, nce_loss
from f_region_structure.state import now, read_signed, record, seal, signed, verified
from wbm_retrieval.common import digest, file_hash, seed_everything, write_csv, write_json
from wbm_retrieval.training import (loader, optimization, optimizer_to_device, parameter_hash,
                                    restore, save_checkpoint, validation_rng)

INITIAL_HASH = "df8cfa59821feeff548ef6e963339b770ecc058fad7c9b4ac0e2727a50a57260"
SNAPSHOTS = (15, 25, 50, 75, 100)
VARIANTS = ("context_dropout", "filtered_negatives")


class FilteredNegativeSampler:
    """Sparse replacements of original shared negatives, with an independent RNG.

    Only the graph and the historical whole-batch exclusion are applied. A
    per-batch boolean membership table avoids B x 5000 x 128 embeddings.
    """
    def __init__(self, graph, count, device, seed=2026):
        self.count, self.device = int(count), torch.device(device)
        indptr, indices = np.asarray(graph["indptr"]), np.asarray(graph["indices"])
        if (indptr.shape != (count+1,) or indptr[0] != 0 or indptr[-1] != len(indices)
                or np.any(np.diff(indptr) < 0) or np.any(indices < 0) or np.any(indices >= count)):
            raise ValueError("Invalid negative-filter CSR")
        degrees = np.diff(indptr)
        if degrees.max(initial=0) > 500:
            raise ValueError("Frozen mutual-top500 graph cannot exceed degree 500")
        padded = np.full((count, max(1, int(degrees.max(initial=0)))), count, dtype=np.int32)
        for row in range(count):
            values = indices[indptr[row]:indptr[row+1]]
            if len(values) and (np.any(np.diff(values) <= 0) or row in values):
                raise ValueError("Expected sorted unique nonself graph rows")
            padded[row, :len(values)] = values
        self.adjacency = torch.as_tensor(padded, device=self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed + 683)

    def replacements(self, anchor_ids, base_ids, excluded):
        anchor_ids, base_ids = anchor_ids.long(), base_ids.long()
        blocked = torch.zeros((len(anchor_ids), self.count+1), dtype=torch.bool, device=self.device)
        blocked.scatter_(1, self.adjacency[anchor_ids].long(), True)
        batch_excluded = torch.zeros(self.count, dtype=torch.bool, device=self.device)
        batch_excluded[excluded] = True
        if batch_excluded[base_ids].any():
            raise ValueError("Base negative sampler violated original batch exclusion")
        eligible = self.count - batch_excluded.sum() - (blocked[:, :self.count] & ~batch_excluded).sum(1)
        if (eligible <= 0).any():
            raise ValueError("No permissible negative remains for an anchor")
        rows, columns = blocked[:, base_ids].nonzero(as_tuple=True)
        anchor_hits = torch.bincount(rows, minlength=len(anchor_ids))
        chosen = torch.empty(len(rows), dtype=torch.long, device=self.device)
        pending = torch.arange(len(rows), device=self.device)
        draws = 0
        while len(pending):
            candidates = torch.randint(self.count, (len(pending),), device=self.device,
                                       generator=self.generator)
            draws += len(pending)
            reject = blocked[rows[pending], candidates] | batch_excluded[candidates]
            chosen[pending[~reject]] = candidates[~reject]
            pending = pending[reject]
        return {"rows": rows, "columns": columns, "ids": chosen,
                "rejected": len(rows), "draws": draws, "count": int(base_ids.numel()),
                "zero_filter_anchor_count": int((anchor_hits == 0).sum()),
                "eligible_anchor_count": len(anchor_ids),
                "base_duplicate_slots": int(base_ids.numel() - base_ids.unique().numel()),
                "eligible_negatives_min": int(eligible.min()),
                "eligible_negatives_sum": int(eligible.sum())}


def filtered_nce_loss(anchors, positives, buffer, base_ids, replacements, temperature):
    """Identical shared logits except graph-hit columns; never changes positives."""
    positives, anchors = F.normalize(positives, dim=1), F.normalize(anchors.detach(), dim=1)
    negatives = F.normalize(buffer[base_ids].detach(), dim=1)
    negative_logits = positives @ negatives.T
    rows, columns, ids = (replacements[key] for key in ("rows", "columns", "ids"))
    if len(rows):
        replacement_features = F.normalize(buffer[ids].detach(), dim=1)
        values = (positives[rows] * replacement_features).sum(1)
        negative_logits = negative_logits.index_put((rows, columns), values)
    positive_logits = (anchors * positives).sum(1, keepdim=True)
    logits = torch.cat((positive_logits, negative_logits), dim=1) / temperature
    return F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=logits.device))


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


def _identity(output, inputs, variant, graph):
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
               "neighbor_graph": graph["signature"] if graph is not None else None,
               "selection": "fixed_epoch_100_no_best_loss_or_test_selection",
               "validation": "unchanged_original_rotation_NCE_diagnostic_only",
               "drop_probability": .5 if variant == "context_dropout" else None,
               "code": {str(p.relative_to(source)): file_hash(p) for p in code_paths}}
    signature = digest(science)
    path = output / "run_manifest.json"
    if path.exists():
        manifest = read_signed(path)
        if manifest["training_signature"] != signature or manifest["science"] != science:
            raise ValueError("Training inputs/settings/code changed; use a new output root")
    else:
        instant = now()
        key = f"cnn_{variant}_seed2026_epoch100"
        manifest = signed({"run_id": instant.strftime("%Y%m%dT%H%M%S%z")+"__"+key,
                           "started_at": instant.isoformat(), "training_signature": signature,
                           "science": science, "environment": {"python": sys.version,
                            "platform": platform.platform(), "torch": torch.__version__,
                            "numpy": np.__version__, "cuda": torch.version.cuda}})
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


def train_variant(output, inputs, variant, *, device="cuda:0", workers=4,
                  neighbor_graph=None, resume=False):
    """Train one authorized intervention; return its sealed final checkpoint path."""
    if variant not in VARIANTS or (variant == "filtered_negatives") != (neighbor_graph is not None):
        raise ValueError("Exactly one authorized intervention and its required graph must be supplied")
    _validate_recipe(inputs)
    if any(not isinstance(inputs.get(key), str) or not inputs[key]
           for key in ("refinement_context_signature", "plan_signature")):
        raise ValueError("Formal training requires signed refinement context and experiment plan")
    root = Path(output)
    output = root / "training" / variant
    output.mkdir(parents=True, exist_ok=True)
    signature, manifest = _identity(output, inputs, variant, neighbor_graph)
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
    if variant == "context_dropout":
        windows = prepare_context_windows(inputs, root / "prepared")
        train = ContextViews(cache, "train", spec, config["seed"], windows)
    else:
        train = RegionViews(cache, "train", spec, config["seed"])
    valid = RegionViews(inputs["datasets"]["valid"]["cache"], "valid", spec, config["seed"]+1)
    bank = MemoryBank(len(train), settings["projector_dim"], device, settings["memory_momentum"])
    sampler = None
    if neighbor_graph is not None:
        if not np.array_equal(neighbor_graph["token_map_ids"], inputs["datasets"]["train"]["token_map_ids"]):
            raise ValueError("Frozen negative graph does not match train token identities/order")
        sampler = FilteredNegativeSampler(neighbor_graph, len(train), device, config["seed"])
    steps = int(checkpoint["optimizer_steps"]) if checkpoint else 0
    if checkpoint:
        if checkpoint["initial_parameter_sha256"] != initial_hash:
            raise ValueError("Resume initialization identity changed")
        bank.buffer.copy_(checkpoint["memory"].to(device))
        if sampler is not None:
            sampler.generator.set_state(checkpoint["negative_generator_rng"])
    else:
        # As in the original run, eval-mode bank initialization never updates BN.
        bank.initialize(model["encoder"], model["projector"], loader(train, config, config["seed"], False), device)
    for epoch in range(start, settings["epochs"]):
        started = time.monotonic()
        train.epoch = epoch
        model.train()
        total, count, rejected, draws, context_pixels, removed_pixels = 0., 0, 0, 0, 0, 0
        no_op_tokens = no_context_tokens = zero_filter_anchors = eligible_anchors = 0
        base_duplicate_slots = base_negative_draws = eligible_negatives_sum = 0
        eligible_negatives_min = len(train)
        for batch, (indices, original, view, stats) in enumerate(loader(train, config, config["seed"]+epoch, True), 1):
            indices = indices.to(device)
            z, zt = model["projector"](model["encoder"](torch.cat([original.to(device), view.to(device)]))).chunk(2)
            anchors = bank.buffer[indices].detach().clone()
            # Real frozen data has 390 batches/epoch and never enters the legacy
            # one-batch fallback; refuse a changed data regime instead of changing it.
            if len(indices) == len(train):
                raise ValueError("Official training requires the unchanged 99,786-token pool")
            base_ids = bank.negative_indices(indices, settings["num_negatives"])
            if sampler is None:
                negatives = bank.buffer[base_ids]
                loss = .5 * (nce_loss(anchors, z, negatives, settings["temperature"]) +
                             nce_loss(anchors, zt, negatives, settings["temperature"]))
                context_pixels += int(stats["context_pixels"].sum())
                removed_pixels += int(stats["removed_pixels"].sum())
                no_op_tokens += int(stats["no_op"].sum())
                no_context_tokens += int(stats["no_context"].sum())
            else:
                replacements = sampler.replacements(indices, base_ids, indices)
                loss = .5 * (filtered_nce_loss(anchors, z, bank.buffer, base_ids, replacements, settings["temperature"]) +
                             filtered_nce_loss(anchors, zt, bank.buffer, base_ids, replacements, settings["temperature"]))
                rejected += replacements["rejected"]
                draws += replacements["draws"]
                zero_filter_anchors += replacements["zero_filter_anchor_count"]
                eligible_anchors += replacements["eligible_anchor_count"]
                base_duplicate_slots += replacements["base_duplicate_slots"]
                base_negative_draws += replacements["count"]
                eligible_negatives_sum += replacements["eligible_negatives_sum"]
                eligible_negatives_min = min(eligible_negatives_min, replacements["eligible_negatives_min"])
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
        history.append({"epoch": epoch+1, "train_loss": total/count, "valid_nce": valid_nce,
                        "learning_rate": optimizer.param_groups[0]["lr"], "optimizer_steps": steps,
                        "seconds": time.monotonic()-started, "negative_positions": count*settings["num_negatives"],
                        "filtered_negative_positions": rejected, "supplementary_draws": draws,
                        "zero_filter_anchor_count": zero_filter_anchors if sampler else None,
                        "eligible_anchor_count": eligible_anchors if sampler else None,
                        "mean_replacements_per_anchor": rejected/count if sampler else None,
                        "filter_position_fraction": rejected/(count*settings["num_negatives"]) if sampler else None,
                        "base_duplicate_slots": base_duplicate_slots if sampler else None,
                        "base_negative_draws": base_negative_draws if sampler else None,
                        "base_duplicate_fraction": base_duplicate_slots/base_negative_draws if sampler else None,
                        "eligible_negatives_min": eligible_negatives_min if sampler else None,
                        "mean_eligible_negatives": eligible_negatives_sum/count if sampler else None,
                        "context_pixels": context_pixels, "removed_context_pixels": removed_pixels,
                        "context_no_op_count": no_op_tokens if sampler is None else None,
                        "context_no_op_fraction": no_op_tokens/count if sampler is None else None,
                        "no_context_count": no_context_tokens if sampler is None else None})
        scheduler.step()
        extra = {"memory": bank.buffer.detach().cpu(), "initial_parameter_sha256": initial_hash,
                 "variant": variant, "input_spec": spec, "optimizer_steps": steps,
                 "training_device": str(device), "started_at": manifest["started_at"],
                 "run_id": manifest["run_id"], "selection": "fixed_epoch_100",
                 "negative_generator_rng": sampler.generator.get_state() if sampler else None}
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
    write_json(output / "training_summary.json", {"completed": True, "epochs_completed": 100,
        "max_epochs": 100, "optimizer_steps": steps, "initial_parameter_sha256": initial_hash,
        "final_parameter_sha256": parameter_hash(model), "checkpoint_selection": "fixed_epoch_100",
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "started_at": manifest["started_at"], "completed_at": now().isoformat(), "run_id": manifest["run_id"]})
    write_json(output / "status.json", {"status": "completed", "epoch": 100, "optimizer_steps": steps,
                                        "updated_at": now().isoformat()})
    seal(output, signature)
    return output / "checkpoint.pt"
