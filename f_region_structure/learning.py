"""Train-only, label-free regional feature fusion on a frozen CNN's two views.

Both families use the last epoch, never a validation/test-selected checkpoint.
The caller supplies original region geometry, which is invariant under the
rotation views used here. No CNN, raw-map loading, or evaluation is performed.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from wbm_retrieval.common import digest, file_hash, read_json, write_json


VERSION = "regional_joint_learning_v1"
FAMILIES = ("plain_mlp", "gated_mlp")
DEFAULTS = {"epochs": 20, "batch_size": 512, "seed": 2026, "learning_rate": 1e-3,
            "weight_decay": 1e-4, "temperature": 0.1, "structure_loss_weight": 0.1,
            "area_loss_weight": 0.05, "embedding_dim": 128, "hidden_dim": 256}


def _array_hash(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(json.dumps(list(value.shape)).encode())
    h.update(memoryview(value).cast("B"))
    return h.hexdigest()


def _positive_int(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _geometry(joint, relative_area, count):
    joint = np.asarray(joint)
    area = np.asarray(relative_area)
    if (joint.shape != (count, 128) or joint.dtype.kind not in "fiu"
            or not np.isfinite(joint).all() or np.any(joint < 0)):
        raise ValueError("joint must be N by 128 finite nonnegative probabilities")
    masses = joint.sum(axis=1, dtype=np.float64)
    missing = masses == 0
    if not np.all(missing | np.isclose(masses, 1., atol=1e-6, rtol=0)):
        raise ValueError("joint rows must have mass one or explicitly zero mass")
    if (area.shape != (count,) or area.dtype.kind not in "fiu"
            or not np.isfinite(area).all() or np.any(area <= 0) or np.any(area > 1)):
        raise ValueError("relative_area must be N finite values in (0, 1]")
    return (np.sqrt(joint).astype(np.float32), np.log(area).astype(np.float32),
            missing.astype(np.float32))


def _features(h, *, views=False):
    h = np.asarray(h)
    expected = (2, 512) if views else (512,)
    if (h.ndim != (3 if views else 2) or h.shape[1:] != expected
            or h.dtype.kind not in "fiu" or not np.isfinite(h).all()):
        raise ValueError("Expected finite N by 2 by 512 views" if views else "Expected finite N by 512 features")
    norms = np.linalg.norm(h, axis=-1)
    if not np.allclose(norms, 1., atol=2e-4, rtol=0):
        raise ValueError("Frozen CNN features must already be L2 normalized and nonzero")
    return np.ascontiguousarray(h, dtype=np.float32)


def _device(device):
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only the recorded CPU or CUDA backend is supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


class FusionModel(nn.Module):
    """128-D representation with an explicit structure-missing input bit."""
    def __init__(self, family, area_mean=0., area_std=1.):
        super().__init__()
        if family not in FAMILIES:
            raise ValueError("Unknown fusion family")
        if not np.isfinite([area_mean, area_std]).all() or area_std <= 0:
            raise ValueError("Invalid train-only area normalization")
        self.family = family
        self.register_buffer("area_mean", torch.tensor(float(area_mean), dtype=torch.float32))
        self.register_buffer("area_std", torch.tensor(float(area_std), dtype=torch.float32))
        if family == "plain_mlp":
            self.fusion = nn.Sequential(nn.Linear(642, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, 128))
        else:
            self.visual = nn.Sequential(nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU())
            self.geometry = nn.Sequential(nn.Linear(130, 256), nn.LayerNorm(256), nn.GELU())
            self.gate = nn.Linear(256, 256)
            self.fusion = nn.Sequential(nn.Linear(768, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, 128))
        # Decoders only anchor training; they do not participate in retrieval.
        self.structure_decoder = nn.Linear(128, 128)
        self.area_decoder = nn.Linear(128, 1)

    def forward(self, h, root_joint, log_area, missing):
        standardized_area = (log_area.reshape(-1, 1) - self.area_mean) / self.area_std
        geom = torch.cat((root_joint, standardized_area, missing.reshape(-1, 1)), dim=1)
        if self.family == "plain_mlp":
            latent = self.fusion(torch.cat((h, geom), dim=1))
        else:
            visual = self.visual(h)
            geometry = self.geometry(geom)
            interaction = visual * torch.sigmoid(self.gate(geometry))
            latent = self.fusion(torch.cat((visual, geometry, interaction), dim=1))
        return F.normalize(latent, dim=1)


def contrastive_loss(z1, z2, sample_ids, temperature=0.1):
    """Symmetric cross-view InfoNCE; exclude other regions of the same wafer.

    The diagonal alone is positive. No class labels are consumed. A row with
    no available different-wafer negatives contributes zero, explicitly counted.
    """
    if (z1.ndim != 2 or z1.shape != z2.shape or sample_ids.shape != (len(z1),)
            or not np.isfinite(temperature) or temperature <= 0):
        raise ValueError("Invalid contrastive inputs")
    count = len(z1)
    diagonal = torch.eye(count, device=z1.device, dtype=torch.bool)
    same_wafer = sample_ids[:, None] == sample_ids[None, :]
    forbidden = same_wafer & ~diagonal
    logits = (z1 @ z2.T) / temperature
    logits = logits.masked_fill(forbidden, -torch.inf)
    labels = torch.arange(count, device=z1.device)
    loss = .5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    without_negatives = int(torch.all(same_wafer, dim=1).sum().item())
    return loss, without_negatives


def _auxiliary_loss(model, z, root_joint, log_area, missing):
    root_prediction = torch.sqrt(F.softmax(model.structure_decoder(z), dim=1).clamp_min(1e-12))
    valid = missing == 0
    if valid.any():
        structure = ((root_prediction[valid] - root_joint[valid]) ** 2).sum(1).mean()
    else:
        structure = z.sum() * 0.
    normalized_area = (log_area - model.area_mean) / model.area_std
    area = F.smooth_l1_loss(model.area_decoder(z).flatten(), normalized_area)
    return structure, area


@contextmanager
def _numerics(device):
    """Disable TF32 for this fit without changing the caller's persistent settings."""
    old_matmul, old_cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


def fit(h_views, joint, relative_area, sample_ids, output_dir, family, device="cpu",
        epochs=20, batch_size=512, seed=2026, *, split="train"):
    """Fit on train only and return an eval-mode model with checkpoint metadata.

    The output directory must not already exist. Fixed final-epoch selection;
    no validation arrays, labels, early stopping, or test metrics are accepted.
    Parent provenance must bind sample IDs to the frozen train split.
    """
    if split != "train":
        raise ValueError("Fusion training accepts only split='train'")
    if family not in FAMILIES:
        raise ValueError("Unknown fusion family")
    epochs = _positive_int(epochs, "epochs")
    batch_size = _positive_int(batch_size, "batch_size", 2)
    seed = _positive_int(seed, "seed", 0)
    h_views = _features(h_views, views=True)
    count = len(h_views)
    if count < 2:
        raise ValueError("Training needs at least two regions")
    root_joint, log_area, missing = _geometry(joint, relative_area, count)
    sample_ids = np.asarray(sample_ids)
    if (sample_ids.shape != (count,) or sample_ids.dtype.kind not in "iu"
            or np.any(sample_ids < 0) or len(np.unique(sample_ids)) < 2):
        raise ValueError("sample_ids must identify at least two distinct original train wafers")
    sample_ids = sample_ids.astype(np.int64, copy=False)
    device = _device(device)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing training output")
    area_mean, area_std = float(log_area.mean(dtype=np.float64)), max(float(log_area.std(dtype=np.float64)), 1e-3)
    started = datetime.now(ZoneInfo("Asia/Shanghai")).replace(microsecond=0)
    settings = {**DEFAULTS, "family": family, "epochs": epochs, "batch_size": batch_size, "seed": seed,
                "checkpoint_rule": "last_epoch", "positive_rule": "same_original_region_across_two_CNN_rotation_views",
                "negative_rule": "other_wafer_regions_in_batch; same_wafer_other_regions_excluded",
                "geometry_view_rule": "same_full_original_mask; rotation_invariant_rho_stick_ball_area",
                "split": "train", "labels_used": False, "CNN_frozen": True,
                "area_normalization": {"mean": area_mean, "std": area_std, "fit_split": "train"}}
    inputs = {"h_views": _array_hash(h_views), "joint": _array_hash(np.asarray(joint)),
              "relative_area": _array_hash(np.asarray(relative_area)), "sample_ids": _array_hash(sample_ids)}
    identity = {"run_id": started.strftime("%Y%m%dT%H%M%S%z") + "__f_region_structure_train_" + family,
                "started_at": started.isoformat(), "kind": "training", "split": "train"}
    manifest = {"version": VERSION, **identity, "settings": settings, "inputs": inputs,
                "region_count": count, "wafer_count": len(np.unique(sample_ids)),
                "missing_structure_count": int(missing.sum()), "code_sha256": file_hash(__file__),
                "environment": {"torch": torch.__version__, "numpy": np.__version__, "device": str(device)},
                "supervision": "self_supervised", "posthoc_design": True, "test_used": False}
    manifest["signature"] = digest(manifest)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "training_manifest.json", manifest)
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices), _numerics(device):
        torch.manual_seed(seed)
        model = FusionModel(family, area_mean, area_std).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=DEFAULTS["learning_rate"], weight_decay=DEFAULTS["weight_decay"])
        rng = np.random.default_rng(seed)
        tensor_inputs = (h_views, root_joint, log_area, missing, sample_ids)
        with (output / "loss.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("epoch", "regions", "loss", "contrastive", "structure", "area", "rows_without_negatives"))
            writer.writeheader()
            for epoch in range(1, epochs + 1):
                model.train()
                indices = rng.permutation(count)
                total = np.zeros(4, dtype=np.float64)
                rows_without_negatives = 0
                # Include every region, even a final singleton (auxiliary loss only).
                for start in range(0, count, batch_size):
                    rows = indices[start:start + batch_size]
                    hv, rj, la, mi, si = (torch.as_tensor(np.array(a[rows], copy=True), device=device) for a in tensor_inputs)
                    z1, z2 = model(hv[:, 0], rj, la, mi), model(hv[:, 1], rj, la, mi)
                    contrastive, no_neg = contrastive_loss(z1, z2, si, DEFAULTS["temperature"])
                    structure, area = _auxiliary_loss(model, torch.cat((z1, z2)), rj.repeat(2, 1), la.repeat(2), mi.repeat(2))
                    loss = contrastive + DEFAULTS["structure_loss_weight"] * structure + DEFAULTS["area_loss_weight"] * area
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite fusion training loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                    optimizer.step()
                    total += len(rows) * np.array([loss.item(), contrastive.item(), structure.item(), area.item()])
                    rows_without_negatives += no_neg
                row = dict(zip(("loss", "contrastive", "structure", "area"), (total / count).tolist()))
                writer.writerow({"epoch": epoch, "regions": count, **row, "rows_without_negatives": rows_without_negatives})
                handle.flush()
                print(f"Fusion {family} epoch {epoch}/{epochs}: loss={row['loss']:.6f}, contrastive={row['contrastive']:.6f}", flush=True)
        model.eval()
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        checkpoint = {"version": VERSION, "family": family, "epoch": epochs, "settings": settings,
                      "manifest_signature": manifest["signature"], "state_dict": state}
        torch.save(checkpoint, output / "final.pt")
    write_json(output / "completed.json", {"status": "completed", "signature": manifest["signature"],
               "checkpoint_rule": "last_epoch", "epoch": epochs,
               "files": {name: file_hash(output / name) for name in ("training_manifest.json", "loss.csv", "final.pt")}})
    model.checkpoint_path = str((output / "final.pt").resolve())
    model.training_metadata = manifest
    return model


def load_model(checkpoint_path, device="cpu"):
    """Load only a completed, hash-bound final checkpoint from this package."""
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.name != "final.pt":
        raise ValueError("Expected the fixed final.pt checkpoint")
    folder = checkpoint_path.parent
    manifest = read_json(folder / "training_manifest.json")
    complete = read_json(folder / "completed.json")
    if (manifest.get("signature") != digest({k: v for k, v in manifest.items() if k != "signature"})
            or manifest.get("version") != VERSION or manifest.get("split") != "train"
            or manifest.get("code_sha256") != file_hash(__file__)
            or manifest.get("test_used") is not False or complete.get("status") != "completed"
            or complete.get("signature") != manifest["signature"]
            or complete.get("checkpoint_rule") != "last_epoch"
            or set(complete.get("files", {})) != {"training_manifest.json", "loss.csv", "final.pt"}
            or any(file_hash(folder / name) != expected for name, expected in complete["files"].items())):
        raise ValueError("Incomplete or changed fusion training artifacts")
    device = _device(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    settings = manifest["settings"]
    if (checkpoint.get("version") != VERSION or checkpoint.get("manifest_signature") != manifest["signature"]
            or checkpoint.get("settings") != settings or checkpoint.get("family") != settings["family"]
            or checkpoint.get("epoch") != settings["epochs"] or complete.get("epoch") != settings["epochs"]):
        raise ValueError("Final checkpoint does not match its frozen training manifest")
    area = settings["area_normalization"]
    model = FusionModel(checkpoint["family"], area["mean"], area["std"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if not np.allclose([model.area_mean.item(), model.area_std.item()], [area["mean"], area["std"]], rtol=1e-6, atol=1e-7):
        raise ValueError("Checkpoint area scaling differs from train-only metadata")
    model = model.to(device).eval()
    model.checkpoint_path, model.training_metadata = str(checkpoint_path.resolve()), manifest
    return model


def encode(model_or_checkpoint, h, joint, relative_area, device="cpu", batch_size=2048):
    """Encode each region once; geometry scaling is frozen from train."""
    batch_size = _positive_int(batch_size, "batch_size")
    h = _features(h)
    root_joint, log_area, missing = _geometry(joint, relative_area, len(h))
    device = _device(device)
    if isinstance(model_or_checkpoint, (str, Path)):
        model = load_model(model_or_checkpoint, device)
    elif isinstance(model_or_checkpoint, FusionModel):
        model = model_or_checkpoint.to(device).eval()
    else:
        raise TypeError("Expected a FusionModel or completed final.pt path")
    outputs = []
    with torch.inference_mode(), _numerics(device):
        for start in range(0, len(h), batch_size):
            args = [torch.as_tensor(np.array(a[start:start + batch_size], copy=True), device=device)
                    for a in (h, root_joint, log_area, missing)]
            outputs.append(model(*args).cpu().numpy())
    result = np.concatenate(outputs) if outputs else np.empty((0, 128), dtype=np.float32)
    if not np.isfinite(result).all() or not np.allclose(np.linalg.norm(result, axis=1), 1., rtol=0, atol=2e-5):
        raise FloatingPointError("Fusion produced invalid normalized embeddings")
    return result
