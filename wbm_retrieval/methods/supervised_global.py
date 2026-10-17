"""Method two: supervised whole-map containment scoring at fixed label budgets."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from ..common import budget_name, get_device, seed_everything, write_csv, write_json
from ..data import load_regions, sample_pairs, stratified_budget
from ..models import GlobalRanker
from ..regions import augment
from ..training import (encoder_spec, loader, load_checkpoint, optimization, optimizer_to_device, parameter_hash, restore,
                        save_checkpoint, stopping_state, training_signature)


class PairDataset(Dataset):
    def __init__(self, image_path, map_ids, pairs, targets, augmentation, seed):
        self.path, self.array = str(image_path), None
        self.index = {int(sample_id): i for i, sample_id in enumerate(map_ids)}
        self.pairs, self.targets = pairs, targets
        self.augmentation, self.seed = augmentation, seed
        if any(int(sample_id) not in self.index for sample_id in pairs.ravel()):
            raise ValueError("Pair points outside its image split")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        if self.array is None:
            self.array = np.load(self.path, mmap_mode="r")
        q, c = self.pairs[index]
        query = np.array(self.array[self.index[int(q)]], dtype=np.float32)
        candidate = np.array(self.array[self.index[int(c)]], dtype=np.float32)
        if self.augmentation is not None:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
            query = augment(query, rng, **self.augmentation)
            candidate = augment(candidate, rng, **self.augmentation)
        return torch.from_numpy(query), torch.from_numpy(candidate), torch.tensor(self.targets[index])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["array"] = None
        return state


def train(config, protocol, resume=False):
    device = get_device(config["device"])
    cache = Path(config["output_dir"]) / "cache"
    labels = np.load(cache / "labels.npy", mmap_mode="r")
    settings = config["global"]
    train_meta, valid_meta = load_regions(config, "train"), load_regions(config, "valid")
    valid_pairs, valid_targets = sample_pairs(protocol.ids["valid"], labels, settings["validation_pairs"], config["seed"] + 777)
    valid_set = PairDataset(cache / "valid_whole.npy", valid_meta["map_ids"], valid_pairs, valid_targets, None, config["seed"])
    loss_function = F.mse_loss if settings["loss"] == "mse" else F.smooth_l1_loss
    for budget in settings["budgets"]:
        # Each budget starts from identical random initialization, without pretrained external weights.
        seed_everything(config["seed"])
        name = budget_name(budget)
        output = Path(config["output_dir"]) / "models" / name
        output.mkdir(parents=True, exist_ok=True)
        selected = stratified_budget(protocol.ids["train"], labels, budget, config["seed"])
        write_csv(output / "labeled_train_ids.csv", ({"sample_id": int(i)} for i in selected), ["sample_id"])
        write_json(output / "budget.json", {"requested_fraction": budget, "selected_wafers": len(selected),
                   "actual_fraction": len(selected) / len(protocol.ids["train"]),
                   "validation_labeled_wafers": len(protocol.ids["valid"]), "test_labels_used_in_training": False})
        np.savez_compressed(output / "validation_pairs.npz", pairs=valid_pairs, targets=valid_targets)
        signature = training_signature(config, name)
        model = GlobalRanker(config["embedding_dim"], settings["pair_hidden_dim"]).to(device)
        optimizer, scheduler = optimization(model, settings)
        start, best, history, checkpoint = restore(output / "last.pt", model, optimizer, scheduler, signature, resume)
        optimizer_to_device(optimizer, device)
        stopper = stopping_state(settings, "global", checkpoint)
        if stopper.stopped or start >= settings["epochs"]:
            print(f"{name} training already complete at epoch {start}; skipping optimizer updates", flush=True)
            continue
        initial_hash = checkpoint["initial_parameter_sha256"] if checkpoint else parameter_hash(model)
        initial_encoder_hash = checkpoint["initial_encoder_sha256"] if checkpoint else parameter_hash(model.encoder)
        optimizer_steps = checkpoint.get("optimizer_steps", 0) if checkpoint else 0
        best_epoch = checkpoint.get("best_epoch", 0) if checkpoint else 0
        for epoch in range(start, settings["epochs"]):
            epoch_started = time.perf_counter()
            pairs, targets = sample_pairs(selected, labels, settings["pairs_per_epoch"], config["seed"] + epoch)
            train_set = PairDataset(cache / "train_whole.npy", train_meta["map_ids"], pairs, targets,
                                    config["augmentation"], config["seed"] + epoch)
            model.train()
            total, count = 0., 0
            for batch_index, (query, candidate, target) in enumerate(loader(train_set, config, config["seed"] + epoch, True), 1):
                predictions = model(query.to(device), candidate.to(device))
                loss = loss_function(predictions, target.to(device))
                loss_value = float(loss.item())
                if not math.isfinite(loss_value):
                    raise FloatingPointError(f"Non-finite {name} loss at epoch {epoch+1}, batch {batch_index}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                optimizer_steps += 1
                total += loss_value * len(target)
                count += len(target)
                if batch_index % config.get("log_every", 100) == 0:
                    print(f"{name} epoch {epoch+1}, batch {batch_index}: loss={loss_value:.5f}, device={device}", flush=True)
            model.eval()
            val_sum, val_count = 0., 0
            with torch.no_grad():
                for query, candidate, target in loader(valid_set, config, config["seed"], False):
                    loss = loss_function(model(query.to(device), candidate.to(device)), target.to(device))
                    val_sum += loss.item() * len(target)
                    val_count += len(target)
            val_loss = val_sum / max(val_count, 1)
            improved = val_loss < best
            best = min(best, val_loss)
            if improved:
                best_epoch = epoch + 1
            should_stop = stopper.step(val_loss, epoch + 1)
            reason = "early_stopping" if should_stop else "max_epochs" if epoch + 1 == settings["epochs"] else None
            history.append({"epoch": epoch + 1, "train_loss": total / count, "valid_loss": val_loss,
                            "learning_rate": optimizer.param_groups[0]["lr"], "optimizer_steps": optimizer_steps,
                            "early_stopping_bad_epochs": stopper.bad_epochs, "epoch_seconds": time.perf_counter() - epoch_started})
            scheduler.step()
            extra = {"embedding_dim": config["embedding_dim"], "hidden_dim": settings["pair_hidden_dim"],
                     "encoder_spec": encoder_spec(config), "initial_encoder_sha256": initial_encoder_hash,
                     "labeled_train_ids": selected, "budget": budget,
                     "early_stopping": stopper.state_dict(), "best_epoch": best_epoch, "stop_reason": reason,
                     "optimizer_steps": optimizer_steps, "training_device": str(device),
                     "initial_parameter_sha256": initial_hash}
            if improved:
                save_checkpoint(output / "best.pt", model, optimizer, scheduler, epoch + 1, best, history, signature, **extra)
            save_checkpoint(output / "last.pt", model, optimizer, scheduler, epoch + 1, best, history, signature, **extra)
            write_json(output / "history.json", history)
            print(f"{name} epoch {epoch+1}/{settings['epochs']}: train={total/count:.5f} valid={val_loss:.5f}", flush=True)
            if should_stop:
                print(f"{name} early stopping at epoch {epoch+1}; best epoch={best_epoch}", flush=True)
                break
        write_json(output / "training_summary.json", {"training_device": str(device), "epochs_completed": len(history),
                   "max_epochs": settings["epochs"], "best_epoch": best_epoch, "best_valid_loss": best,
                   "stop_reason": reason, "early_stopping": stopper.state_dict(), "optimizer_steps": optimizer_steps,
                   "initial_parameter_sha256": initial_hash, "final_parameter_sha256": parameter_hash(model),
                   "initial_encoder_sha256": initial_encoder_hash, "encoder_spec": encoder_spec(config),
                   "optimizer": "SGD"})


@torch.no_grad()
def rank_global(name, model, embeddings, metadata, config, protocol, device):
    output = Path(config["output_dir"]) / "retrieval" / name
    output.mkdir(parents=True, exist_ok=True)
    index = {int(sample_id): i for i, sample_id in enumerate(metadata["map_ids"])}
    embeddings = torch.as_tensor(np.array(embeddings), device=device)
    model.eval()
    started = time.perf_counter()
    with (output / "rankings.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "rank", "candidate_id", "similarity_score"])
        for query_index, query_id in enumerate(protocol.queries, 1):
            candidates = protocol.candidates[query_id]
            scores = []
            for start in range(0, len(candidates), config["pair_score_batch_size"]):
                chunk = candidates[start:start + config["pair_score_batch_size"]]
                candidate = embeddings[[index[c] for c in chunk]]
                query = embeddings[index[query_id]].unsqueeze(0).expand(len(chunk), -1)
                scores.extend(model.scorer(query, candidate).cpu().tolist())
            scored = sorted(zip(candidates, scores), key=lambda value: (-value[1], value[0]))
            writer.writerows([query_id, rank, c, score] for rank, (c, score) in enumerate(scored, 1))
            if query_index % 100 == 0:
                print(f"{name}: {query_index}/{len(protocol.queries)} queries", flush=True)
    write_json(output / "retrieval_info.json", {"method": name, "relevance": "query_label_containment",
               "ranking_wall_seconds_excluding_encoding": time.perf_counter() - started,
               "num_queries": len(protocol.queries), "scorer": "ordered shared-encoder MLP"})


def retrieve(config, protocol, methods=None):
    """Load each requested budget model and score ordered whole-map pairs."""
    from ..retrieval import feature_cache

    cache = Path(config["output_dir"]) / "cache"
    metadata = load_regions(config, "test")
    names = methods if methods is not None else [budget_name(b) for b in config["global"]["budgets"]]
    for budget in config["global"]["budgets"]:
        name = budget_name(budget)
        if name not in names:
            continue
        device = get_device(config["device"])
        path = Path(config["output_dir"]) / "models" / name / "best.pt"
        checkpoint = load_checkpoint(path)
        if checkpoint["signature"] != training_signature(config, name):
            raise ValueError(f"Checkpoint/config mismatch for {name}")
        model = GlobalRanker(config["embedding_dim"], config["global"]["pair_hidden_dim"]).to(device)
        model.load_state_dict(checkpoint["model"])
        embeddings = feature_cache(
            model.encoder, cache / "test_whole.npy", path,
            cache / f"test_{name}_embeddings.npy", config, device,
        )
        rank_global(name, model, embeddings, metadata, config, protocol, device)
