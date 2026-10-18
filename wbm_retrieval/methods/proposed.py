"""Proposed method: token instance learning and greedy local retrieval."""

from __future__ import annotations

from pathlib import Path
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from ..common import get_device, seed_everything, write_json
from ..data import load_regions
from ..matching import group_tokens
from ..models import Encoder, Projector
from ..augmentation import opening_summary, region_view, save_view_examples
from ..training import (encoder_spec, loader, load_checkpoint, optimization, optimizer_to_device, parameter_hash, restore,
                        save_checkpoint, stopping_state, training_signature, validation_rng)


class PatchDataset(Dataset):
    def __init__(self, path, augmentation, seed, epoch=0, opening=None):
        self.path = str(path)
        self.array = None
        self.length = len(np.load(self.path, mmap_mode="r"))
        self.augmentation, self.seed, self.epoch = augmentation, seed, epoch
        self.opening = opening

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.array is None:
            self.array = np.load(self.path, mmap_mode="r")
        image = np.array(self.array[index], dtype=np.float32)
        # Epoch-aware views; the same seed and epoch are reproducible across worker counts.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index)]))
        transformed, info = region_view(image, rng, self.augmentation, self.opening)
        return int(index), torch.from_numpy(image), torch.from_numpy(transformed), info

    def __getstate__(self):
        state = self.__dict__.copy()
        state["array"] = None
        return state


class MemoryBank:
    def __init__(self, count, dim, device, momentum=0.5):
        self.buffer = torch.zeros(count, dim, device=device)
        self.momentum = momentum

    @torch.no_grad()
    def initialize(self, encoder, projector, data_loader, device):
        encoder.eval()
        projector.eval()
        for indices, image, _, _info in data_loader:
            self.buffer[indices.to(device)] = projector(encoder(image.to(device)))

    def negative_indices(self, excluded, count):
        probabilities = torch.ones(len(self.buffer), device=self.buffer.device)
        probabilities[excluded] = 0
        available = int((probabilities > 0).sum())
        if available == 0:
            raise ValueError("No negatives left outside the current batch")
        return torch.multinomial(probabilities, count, replacement=True)

    @torch.no_grad()
    def update(self, indices, values):
        self.buffer[indices] = self.momentum * self.buffer[indices] + (1 - self.momentum) * values


def nce_loss(anchors, positives, negatives, temperature):
    """Each row contrasts one positive against a shared, detached bank sample."""
    positives, anchors = F.normalize(positives, dim=1), F.normalize(anchors.detach(), dim=1)
    negatives = F.normalize(negatives.detach(), dim=1)
    positive = (anchors * positives).sum(1, keepdim=True)
    logits = torch.cat([positive, positives @ negatives.T], dim=1) / temperature
    return F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=logits.device))


def train(config, resume=False):
    seed_everything(config["seed"])
    device = get_device(config["device"])
    settings = config["local"]
    cache = Path(config["output_dir"]) / "cache"
    output = Path(config["output_dir"]) / "models/local"
    output.mkdir(parents=True, exist_ok=True)
    signature = training_signature(config, "local")
    encoder = Encoder(config["embedding_dim"]).to(device)
    projector = Projector(config["embedding_dim"], settings["projector_dim"]).to(device)
    model = nn.ModuleDict({"encoder": encoder, "projector": projector})
    optimizer, scheduler = optimization(model, settings)
    start, best, history, checkpoint = restore(output / "last.pt", model, optimizer, scheduler, signature, resume)
    optimizer_to_device(optimizer, device)
    stopper = stopping_state(settings, "local", checkpoint)
    if stopper.stopped or start >= settings["epochs"]:
        print(f"Local training already complete at epoch {start}; skipping optimizer updates", flush=True)
        return
    initial_hash = checkpoint["initial_parameter_sha256"] if checkpoint else parameter_hash(model)
    initial_encoder_hash = checkpoint["initial_encoder_sha256"] if checkpoint else parameter_hash(encoder)
    optimizer_steps = checkpoint.get("optimizer_steps", 0) if checkpoint else 0
    best_epoch = checkpoint.get("best_epoch", 0) if checkpoint else 0
    train = PatchDataset(cache / "train_patches.npy", config["augmentation"], config["seed"], opening=settings.get("opening"))
    valid = PatchDataset(cache / "valid_patches.npy", config["augmentation"], config["seed"] + 1, opening=settings.get("opening"))
    if settings.get("opening", {}).get("enabled", False):
        save_view_examples(config, valid)
    if len(train) < 2:
        raise ValueError("At least two train tokens are required")
    bank = MemoryBank(len(train), settings["projector_dim"], device, settings["memory_momentum"])
    if checkpoint:
        bank.buffer.copy_(checkpoint["memory"].to(device))
    else:
        print(f"Initializing local memory bank with {len(train)} train tokens", flush=True)
        bank.initialize(encoder, projector, loader(train, config, config["seed"], False), device)
    for epoch in range(start, settings["epochs"]):
        epoch_started = time.perf_counter()
        train.epoch = epoch
        model.train()
        total, examples = 0., 0
        audit_keys = ("attempted", "accepted", "changed", "fallback", "before_pixels", "after_pixels")
        view_audit = {name: dict.fromkeys(("views",) + audit_keys, 0) for name in ("train", "valid")}
        for batch_index, (indices, image, transformed, info) in enumerate(loader(train, config, config["seed"] + epoch, True), 1):
            view_audit["train"]["views"] += len(indices)
            for key in audit_keys:
                view_audit["train"][key] += int(info[key].sum())
            indices = indices.to(device)
            image, transformed = image.to(device), transformed.to(device)
            value = projector(encoder(torch.cat([image, transformed])))
            z, zt = value.chunk(2)
            anchors = bank.buffer[indices].detach().clone()
            # For tiny diagnostics with one full batch, exclude self per anchor instead.
            if len(indices) == len(train):
                losses = []
                for i, token_id in enumerate(indices):
                    neg = bank.buffer[bank.negative_indices(token_id.reshape(1), settings["num_negatives"])]
                    losses.append(0.5 * (nce_loss(anchors[i:i+1], z[i:i+1], neg, settings["temperature"]) +
                                         nce_loss(anchors[i:i+1], zt[i:i+1], neg, settings["temperature"])))
                loss = torch.stack(losses).mean()
            else:
                negatives = bank.buffer[bank.negative_indices(indices, settings["num_negatives"])]
                loss = 0.5 * (nce_loss(anchors, z, negatives, settings["temperature"]) +
                              nce_loss(anchors, zt, negatives, settings["temperature"]))
            loss_value = float(loss.item())
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"Non-finite local loss at epoch {epoch+1}, batch {batch_index}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            bank.update(indices, z.detach())
            total += loss_value * len(indices)
            examples += len(indices)
            if batch_index % config.get("log_every", 100) == 0:
                print(f"Local epoch {epoch+1}, batch {batch_index}: loss={loss_value:.5f}, device={device}", flush=True)
        model.eval()
        val_sum, val_count = 0., 0
        # Fixed views and fixed negative indices make epochs comparable.
        with torch.no_grad(), validation_rng(device, config["seed"] + 999):
            for _, image, transformed, info in loader(valid, config, config["seed"], False):
                view_audit["valid"]["views"] += len(image)
                for key in audit_keys:
                    view_audit["valid"][key] += int(info[key].sum())
                z, zt = projector(encoder(torch.cat([image.to(device), transformed.to(device)]))).chunk(2)
                # Validation IDs are unrelated to train-bank IDs: exclude none.
                chosen = torch.randint(len(bank.buffer), (settings["num_negatives"],), device=device)
                loss = nce_loss(z, zt, bank.buffer[chosen], settings["temperature"])
                val_sum += loss.item() * len(image)
                val_count += len(image)
        val_loss = val_sum / max(val_count, 1)
        improved = val_loss < best
        best = min(best, val_loss)
        if improved:
            best_epoch = epoch + 1
        should_stop = stopper.step(val_loss, epoch + 1)
        reason = "early_stopping" if should_stop else "max_epochs" if epoch + 1 == settings["epochs"] else None
        history.append({"epoch": epoch + 1, "train_loss": total / max(examples, 1), "valid_loss": val_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"], "optimizer_steps": optimizer_steps,
                        "early_stopping_bad_epochs": stopper.bad_epochs, "epoch_seconds": time.perf_counter() - epoch_started,
                        "opening": {name: opening_summary(values) for name, values in view_audit.items()}})
        scheduler.step()
        extra = {"memory": bank.buffer.detach().cpu(), "embedding_dim": config["embedding_dim"],
                 "encoder_spec": encoder_spec(config), "initial_encoder_sha256": initial_encoder_hash,
                 "early_stopping": stopper.state_dict(), "best_epoch": best_epoch, "stop_reason": reason,
                 "optimizer_steps": optimizer_steps, "training_device": str(device),
                 "initial_parameter_sha256": initial_hash, "opening": settings.get("opening")}
        if improved:
            save_checkpoint(output / "best.pt", model, optimizer, scheduler, epoch + 1, best, history, signature, **extra)
        save_checkpoint(output / "last.pt", model, optimizer, scheduler, epoch + 1, best, history, signature, **extra)
        write_json(output / "history.json", history)
        write_json(output / "augmentation_audit.json", {"scope": "proposed region view only, before rotation",
                   "settings": settings.get("opening"), "input_pixels": config["input_size"],
                   "epochs": [{"epoch": item["epoch"], **item["opening"]} for item in history if "opening" in item]})
        print(f"Local epoch {epoch+1}/{settings['epochs']}: train={history[-1]['train_loss']:.5f} valid={val_loss:.5f}", flush=True)
        if should_stop:
            print(f"Local early stopping at epoch {epoch+1}; best epoch={best_epoch}", flush=True)
            break
    write_json(output / "training_summary.json", {"training_device": str(device), "epochs_completed": len(history),
               "max_epochs": settings["epochs"], "best_epoch": best_epoch, "best_valid_loss": best,
               "stop_reason": reason, "early_stopping": stopper.state_dict(), "optimizer_steps": optimizer_steps,
               "initial_parameter_sha256": initial_hash, "final_parameter_sha256": parameter_hash(model),
               "initial_encoder_sha256": initial_encoder_hash, "encoder_spec": encoder_spec(config),
               "optimizer": "SGD", "negative_sampling": "with replacement, excluding training batch",
               "memory_update": "raw projected features, unnormalized EMA", "opening": settings.get("opening")})


def load_encoder(config, device):
    path = Path(config["output_dir"]) / "models/local/best.pt"
    checkpoint = load_checkpoint(path)
    if checkpoint["signature"] != training_signature(config, "local"):
        raise ValueError("Local checkpoint is incompatible with the current configuration/cache")
    state = {key[len("encoder."):]: value for key, value in checkpoint["model"].items() if key.startswith("encoder.")}
    model = Encoder(config["embedding_dim"]).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, path


def retrieve(config, protocol):
    """Extract learned token embeddings and rank every frozen query candidate pool."""
    from ..retrieval import feature_cache, rank_local

    cache = Path(config["output_dir"]) / "cache"
    metadata = load_regions(config, "test")
    device = get_device(config["device"])
    encoder, checkpoint = load_encoder(config, device)
    embeddings = feature_cache(
        encoder, cache / "test_patches.npy", checkpoint,
        cache / "test_local_embeddings.npy", config, device,
    )
    rank_local("proposed", group_tokens(metadata, embeddings), config, protocol)
