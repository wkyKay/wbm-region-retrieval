"""Configuration, provenance and reproducibility helpers without a Torch dependency."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def csv_rows(path):
    return list(iter_csv(path))


def iter_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        yield from csv.DictReader(handle)


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_config(path):
    path = Path(path).resolve()
    config = read_json(path)
    # Relative paths are always project-relative, independent of the invoking cwd.
    for key in ("data_file", "split_manifest", "query_manifest", "candidate_manifest", "output_dir"):
        value = Path(config[key]).expanduser()
        config[key] = str(value.resolve() if value.is_absolute() else (ROOT / value).resolve())
    p, m = config["proposal"], config["matching"]
    if p["min_area"] < 1 or p["top_k"] < 1 or p["padding"] < 0:
        raise ValueError("min_area/top_k must be positive and padding must be nonnegative")
    if config["input_size"] < 32 or config["embedding_dim"] != 512:
        raise ValueError("WaPIRL-aligned models require input_size >= 32 and embedding_dim = 512")
    from .methods.handcrafted import moment_orders
    descriptor = config["handcrafted"]
    if descriptor["type"] != "zernike" or not np.isfinite(descriptor["distance_scale"]) or descriptor["distance_scale"] <= 0:
        raise ValueError("Use a Zernike descriptor with a finite positive distance_scale")
    moment_orders(descriptor["max_order"])
    if config["batch_size"] < 2 or config["num_workers"] < 0:
        raise ValueError("batch_size >= 2 and num_workers >= 0 are required")
    if not (0 <= m["shape_threshold"] <= 1 and 0 <= m["score_threshold"] <= 1):
        raise ValueError("Matching thresholds must lie in [0, 1]")
    if m["weight"] not in ("uniform", "sqrt_area", "area"):
        raise ValueError("Unknown query token weight")
    if m["lambda_pos"] != 0 or m["lambda_scale"] != 0:
        raise ValueError("Experiment A must be shape-only; use geometry settings for supplementary examples")
    g = config["geometry"]
    if min(g["lambda_pos"], g["lambda_scale"]) < 0 or g["lambda_pos"] + g["lambda_scale"] > 0.25 + 1e-12:
        raise ValueError("Geometry lambdas must be nonnegative and sum to at most 0.25")
    if m["sigma_pos"] <= 0 or m["sigma_scale"] <= 0:
        raise ValueError("Affinity bandwidths must be positive")
    a = config["augmentation"]
    if not all(0 <= a[k] < 1 for k in ("dropout", "noise")) or a["max_shift"] < 0:
        raise ValueError("Invalid augmentation strength")
    local, global_ = config["local"], config["global"]
    for settings in (local, global_):
        if settings["optimizer"] != "sgd" or settings["scheduler"] != "wapirl_cosine":
            raise ValueError("Both learned methods use the WaPIRL SGD/cosine recipe")
        if settings["learning_rate"] <= 0 or settings["weight_decay"] < 0 or not 0 <= settings["momentum"] < 1:
            raise ValueError("Invalid SGD settings")
    if local["projector_type"] != "linear" or local["negative_sampling"] != "with_replacement":
        raise ValueError("WaPIRL alignment requires a linear projector and with-replacement negatives")
    if local["epochs"] < 1 or global_["epochs"] < 1 or local["num_negatives"] < 1:
        raise ValueError("Epochs and negative count must be positive")
    if local["temperature"] <= 0 or not 0 <= local["memory_momentum"] < 1:
        raise ValueError("Invalid temperature or memory momentum")
    if any(not 0 < b <= 1 for b in global_["budgets"]) or len(set(global_["budgets"])) != len(global_["budgets"]):
        raise ValueError("Label budgets must be unique values in (0, 1]")
    if global_["pairs_per_epoch"] < 1 or global_["validation_pairs"] < 1:
        raise ValueError("Pair counts must be positive")
    if global_["loss"] not in ("mse", "smooth_l1"):
        raise ValueError("Global loss must be mse or smooth_l1")
    from .early_stopping import EarlyStopping
    for kind, settings in (("local", local), ("global", global_)):
        EarlyStopping.from_settings(settings, kind)
    if any(int(k) < 1 for k in config["evaluation"]["ks"]):
        raise ValueError("Evaluation K must be positive")
    return config


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(name):
    import torch
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select --device cpu for forward checks")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this process; run on a Metal-enabled Mac outside a restrictive sandbox")
    return device


def budget_name(value):
    return f"global_{100 * value:g}pct"
