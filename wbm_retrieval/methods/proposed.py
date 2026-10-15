"""Proposed method: token instance learning and greedy local retrieval."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from ..common import get_device, seed_everything, write_json
from ..data import load_regions
from ..matching import group_tokens
from ..models import Encoder, Projector
from ..regions import augment
from ..training import loader, load_checkpoint, optimizer_to_device, restore, save_checkpoint, training_signature


class PatchDataset(Dataset):
    def __init__(self, path, augmentation, seed, epoch=0):
        self.path = str(path)
        self.array = None
        self.length = len(np.load(self.path, mmap_mode="r"))
        self.augmentation, self.seed, self.epoch = augmentation, seed, epoch

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.array is None:
            self.array = np.load(self.path, mmap_mode="r")
        image = np.array(self.array[index], dtype=np.float32)
        # Epoch-aware views; the same seed and epoch are reproducible across worker counts.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index)]))
        transformed = augment(image, rng, **self.augmentation)
        return int(index), torch.from_numpy(image), torch.from_numpy(transformed)

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
        for indices, image, _ in data_loader:
            self.buffer[indices.to(device)] = projector(encoder(image.to(device)))

    def negative_indices(self, excluded, count):
        probabilities = torch.ones(len(self.buffer), device=self.buffer.device)
        probabilities[excluded] = 0
        available = int((probabilities > 0).sum())
        if available == 0:
            raise ValueError("No negatives left outside the current batch")
        return torch.multinomial(probabilities, min(count, available), replacement=False)

    @torch.no_grad()
    def update(self, indices, values):
        updated = self.momentum * self.buffer[indices] + (1 - self.momentum) * values
        self.buffer[indices] = F.normalize(updated, dim=1)


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
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, settings["epochs"])
    start, best, history, checkpoint = restore(output / "last.pt", model, optimizer, scheduler, signature, resume)
    optimizer_to_device(optimizer, device)
    train = PatchDataset(cache / "train_patches.npy", config["augmentation"], config["seed"])
    valid = PatchDataset(cache / "valid_patches.npy", config["augmentation"], config["seed"] + 1)
    if len(train) < 2:
        raise ValueError("At least two train tokens are required")
    bank = MemoryBank(len(train), settings["projector_dim"], device, settings["memory_momentum"])
    if checkpoint:
        bank.buffer.copy_(checkpoint["memory"].to(device))
    else:
        print(f"Initializing local memory bank with {len(train)} train tokens", flush=True)
        bank.initialize(encoder, projector, loader(train, config, config["seed"], False), device)
    for epoch in range(start, settings["epochs"]):
        train.epoch = epoch
        model.train()
        total, examples = 0., 0
        for indices, image, transformed in loader(train, config, config["seed"] + epoch, True):
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
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            bank.update(indices, z.detach())
            total += loss.item() * len(indices)
            examples += len(indices)
        model.eval()
        val_sum, val_count = 0., 0
        # Fixed views and fixed negative indices make epochs comparable.
        with torch.no_grad(), torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
            torch.manual_seed(config["seed"] + 999)
            for _, image, transformed in loader(valid, config, config["seed"], False):
                z, zt = projector(encoder(torch.cat([image.to(device), transformed.to(device)]))).chunk(2)
                chosen = torch.randperm(len(bank.buffer), device=device)[:settings["num_negatives"]]
                loss = nce_loss(z, zt, bank.buffer[chosen], settings["temperature"])
                val_sum += loss.item() * len(image)
                val_count += len(image)
        val_loss = val_sum / max(val_count, 1)
        improved = val_loss < best
        best = min(best, val_loss)
        history.append({"epoch": epoch + 1, "train_loss": total / max(examples, 1), "valid_loss": val_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        scheduler.step()
        extra = {"memory": bank.buffer.detach().cpu(), "embedding_dim": config["embedding_dim"]}
        if improved:
            save_checkpoint(output / "best.pt", model, optimizer, scheduler, epoch + 1, best, history, signature, **extra)
        save_checkpoint(output / "last.pt", model, optimizer, scheduler, epoch + 1, best, history, signature, **extra)
        write_json(output / "history.json", history)
        print(f"Local epoch {epoch+1}/{settings['epochs']}: train={history[-1]['train_loss']:.5f} valid={val_loss:.5f}", flush=True)


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

