"""Isolated A/B/C/D inputs, frozen pools and retrieval; legacy runs stay unchanged."""

from __future__ import annotations

import copy
import csv
import fcntl
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .common import ROOT, digest, file_hash, load_config, read_json, write_csv, write_json
from .data import Protocol, containment, source_fingerprint
from .diagnostics import batched_map_scores, normalized, summarize_groups
from .evaluation import bootstrap_interval, query_metrics
from .methods.handcrafted import handcrafted_descriptor
from .models import Encoder, Projector
from .regions import extract_regions, resize_nearest, square_pad, token_patch


def merge(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_suite(path, *, output_root=None, seed=None, device=None, batch_size=None, workers=None):
    suite = read_json(path)
    config = merge(load_config(ROOT / suite["base_config"]), suite["overrides"])
    original_output = Path(config["output_dir"]).resolve()
    root = Path(output_root or suite["output_root"]).expanduser()
    root = (ROOT / root).resolve()
    if root == ROOT or root in ROOT.parents or root == original_output or root in original_output.parents or original_output in root.parents:
        raise ValueError("Ablations require a separate output directory outside the original experiment")
    config["output_dir"] = str(root)
    for key, value in (("seed", seed), ("device", device), ("batch_size", batch_size), ("num_workers", workers)):
        if value is not None:
            config[key] = value
    if config["batch_size"] < 2 or config["num_workers"] < 0 or config["inference_batch_size"] < 1:
        raise ValueError("Invalid batch size / worker count")
    if config["local"].get("opening", {}).get("enabled") or config["local"]["early_stopping"]["enabled"]:
        raise ValueError("All four experiments disable opening and early stopping")
    if config["augmentation"] != dict(rotation=True, max_shift=0, dropout=0., noise=0.):
        raise ValueError("Base augmentation must be right-angle rotations only")
    if config["input_size"] != 96 or config["proposal"] != dict(min_area=5, top_k=4, padding=4):
        raise ValueError("This suite fixes input=96, min_area=5, top_k=4, old-grid padding=4")
    if suite["selection_metric"] != "ContainmentNDCG@10" or 10 not in config["evaluation"]["ks"]:
        raise ValueError("Checkpoint selection requires ContainmentNDCG@10")
    if suite["evaluate_every"] < 1 or config["local"]["epochs"] < 1:
        raise ValueError("Epoch count and evaluation interval must be positive")
    if set(suite["variants"]) != set("ABCD"):
        raise ValueError("Expected variants A, B, C, D")
    a, b, c, d = (suite["variants"][name] for name in "ABCD")
    if not (a == {**b, "preprocess": "pad_resize"} and d == {**b, "stem_stride": 1}
            and c == {**b, "scale_range": [80, 88], "max_shift": 2}
            and b == dict(preprocess="resize_pad", content_size=88, stem_stride=2, scale_range=None, max_shift=0)):
        raise ValueError("A/B/C/D must follow the specified one-factor comparisons")
    for spec in suite["variants"].values():
        if spec["preprocess"] not in ("pad_resize", "resize_pad") or spec["stem_stride"] not in (1, 2):
            raise ValueError("Invalid preprocessing or stride")
        low, high = spec["scale_range"] or [spec["content_size"]] * 2
        if not (1 <= low <= high <= spec["content_size"] <= 96) or not 0 <= spec["max_shift"] <= (96-high)//2:
            raise ValueError("Scale / shift must fit completely within the 96-pixel canvas")
    return suite, config


@contextmanager
def exclusive(root):
    """Linux/macOS advisory lock; releases on crashes, no stale PID lock cleanup."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".ablation.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another ablation process is using {root}; run sequentially") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def code_hashes():
    names = ["ablation.py", "ablation_training.py", "regions.py", "data.py", "models.py", "training.py",
             "diagnostics.py", "evaluation.py", "matching.py", "common.py", "methods/proposed.py",
             "methods/handcrafted.py"]
    return {name: file_hash(Path(__file__).parent / name) for name in names}


def resize_then_pad(mask, size=96, content_size=88, shift=(0, 0)):
    points = np.argwhere(mask)
    if not len(points):
        raise ValueError("Empty region")
    low, high = points.min(0), points.max(0) + 1
    # Square padding here ONLY preserves aspect ratio; extra margin is added AFTER resize.
    crop = mask[low[0]:high[0], low[1]:high[1]].astype(np.uint8)
    content = resize_nearest(square_pad(crop), content_size)
    r, c = (size-content_size)//2 + int(shift[0]), (size-content_size)//2 + int(shift[1])
    if min(r, c) < 0 or max(r, c) + content_size > size:
        raise ValueError("Translation would clip the resized region")
    result = np.zeros((size, size), dtype=np.uint8)
    result[r:r+content_size, c:c+content_size] = content
    return result


class RegionViews(Dataset):
    def __init__(self, cache, split, spec, seed, epoch=0):
        self.spec, self.seed, self.epoch = spec, seed, epoch
        family = "A" if spec["preprocess"] == "pad_resize" else "B"
        self.patch_path = str(Path(cache) / f"{split}_{family}.npy")
        self.mask_path = str(Path(cache) / f"{split}_masks.npy")
        self.length = len(np.load(self.patch_path, mmap_mode="r"))
        self.patches = self.masks = None

    def __len__(self):
        return self.length

    def __getstate__(self):
        return {**self.__dict__, "patches": None, "masks": None}

    def __getitem__(self, index):
        if self.patches is None:
            self.patches = np.load(self.patch_path, mmap_mode="r")
        original = np.array(self.patches[index], dtype=np.float32)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index)]))
        turns = int(rng.integers(4))  # Same rotations across A/B/C/D for each seed/epoch/token.
        if self.spec["scale_range"] is None:
            view = original
        else:
            if self.masks is None:
                self.masks = np.load(self.mask_path, mmap_mode="r")
            low, high = self.spec["scale_range"]
            side = int(rng.integers(low, high+1))
            delta = self.spec["max_shift"]
            shift = rng.integers(-delta, delta+1, size=2)
            # Always resample the original mask once, never rescale a 96x96 patch again.
            view = resize_then_pad(self.masks[index], original.shape[-1], side, shift)[None].astype(np.float32)
        view = np.ascontiguousarray(np.rot90(view, turns, axes=(-2, -1)))
        return index, torch.from_numpy(original), torch.from_numpy(view), {}


def build_model(config, spec):
    encoder = Encoder(config["embedding_dim"])
    encoder.stem[0].stride = (spec["stem_stride"],) * 2
    return nn.ModuleDict({"encoder": encoder,
                          "projector": Projector(config["embedding_dim"], config["local"]["projector_dim"])})


def valid_pool(ids, settings):
    ids = np.asarray(sorted(ids), dtype=np.int64)
    nq, nc = settings["queries"], settings["candidates"]
    if not 0 < nq <= len(ids) or not 0 < nc < len(ids):
        raise ValueError("Valid query/candidate counts exceed available split; no silent truncation")
    rng = np.random.default_rng(settings["seed"])
    queries = sorted(rng.choice(ids, nq, replace=False).tolist())
    return {"settings": settings, "queries": queries, "candidates": {
        str(q): sorted(rng.choice(ids[ids != q], nc, replace=False).tolist()) for q in queries}}


def prepare_suite(suite, config):
    root = Path(config["output_dir"])
    sources = source_fingerprint(config)
    # Device/workers/training seed may change; all scientific settings and batch size may not.
    settings = {k: v for k, v in config.items() if k not in (
        "device", "num_workers", "seed", "output_dir", "data_file", "split_manifest", "query_manifest", "candidate_manifest")}
    identity = {"sources": sources, "settings": settings,
                "suite": {k: suite[k] for k in ("variants", "valid_pool", "selection_metric", "evaluate_every")},
                "code": code_hashes()}
    signature = digest(identity)
    manifest = root / "suite_manifest.json"
    if manifest.exists():
        if read_json(manifest)["signature"] != signature:
            raise ValueError("Suite inputs/config/code changed; choose a new --output-root")
    else:
        if any(p.name != ".ablation.lock" for p in root.iterdir()):
            raise ValueError("Nonempty output has no ablation manifest; choose a new --output-root")
        write_json(manifest, {"signature": signature, **identity})
    cache = root / "cache"
    done = cache / "prepared.json"
    if done.exists():
        status = read_json(done)
        if status["signature"] != signature:
            raise ValueError("Cache signature mismatch")
        for name, sha in status["files"].items():
            if not (root / name).is_file() or file_hash(root / name) != sha:
                raise ValueError(f"Missing or modified ablation cache: {name}")
        print("Reusing verified ablation cache and fixed valid pool", flush=True)
        return signature
    cache.mkdir(exist_ok=True)
    with np.load(config["data_file"], allow_pickle=False) as archive:
        maps, labels = archive["arr_0"], archive["arr_1"]
    if maps.ndim != 3 or labels.ndim != 2 or len(maps) != len(labels) or not np.isin(maps, [0, 1, 2, 3]).all():
        raise ValueError("Expected Mixed-WM38 NPZ encoding")
    maps = maps.astype(np.uint8)
    protocol = Protocol(config, labels)
    np.save(cache / "labels.npy", labels)
    for split, ids in protocol.ids.items():
        regions, offsets = [], [0]
        for n, sample_id in enumerate(ids, 1):
            regions.extend(extract_regions(maps[sample_id], **config["proposal"]))
            offsets.append(len(regions))
            if n % 2000 == 0:
                print(f"CC {split}: {n}/{len(ids)}", flush=True)
        count = len(regions)
        if not count:
            raise ValueError(f"No retained regions: {split}")
        masks = np.lib.format.open_memmap(cache / f"{split}_masks.npy", mode="w+", dtype=np.uint8,
                                         shape=(count, *maps.shape[1:]))
        patches = {name: np.lib.format.open_memmap(cache / f"{split}_{name}.npy", mode="w+", dtype=np.uint8,
                    shape=(count, 1, config["input_size"], config["input_size"])) for name in ("A", "B")}
        descriptors = []
        for index, region in enumerate(regions):
            masks[index] = region["mask"]
            patches["A"][index, 0] = token_patch(region["mask"], config["input_size"], config["proposal"]["padding"])
            patches["B"][index, 0] = resize_then_pad(region["mask"], config["input_size"], suite["variants"]["B"]["content_size"])
            descriptors.append(handcrafted_descriptor(region["mask"], config["handcrafted"]["max_order"]))
            if (index+1) % 2000 == 0:
                print(f"Patches/Zernike {split}: {index+1}/{count}", flush=True)
        masks.flush()
        for patch in patches.values():
            patch.flush()
        np.savez_compressed(cache / f"{split}_regions.npz", map_ids=ids, offsets=offsets,
                            area=np.array([r["area"] for r in regions]), descriptor=np.asarray(descriptors, dtype=np.float32))
        print(f"Prepared {split}: maps={len(ids)}, tokens={count}", flush=True)
    write_json(root / "valid_pool.json", valid_pool(protocol.ids["valid"], suite["valid_pool"]))
    files = list(cache.glob("*.npy")) + list(cache.glob("*.npz")) + [root / "valid_pool.json"]
    write_json(done, {"signature": signature, "files": {str(p.relative_to(root)): file_hash(p) for p in files},
                     "protocol": protocol.summary(), "test_note": "Preprocessed only; never used for training or model selection"})
    return signature


@torch.no_grad()
def embeddings(model, patch_path, config, device):
    model.eval()
    patches = np.load(patch_path, mmap_mode="r")
    result = []
    for start in range(0, len(patches), config["inference_batch_size"]):
        x = torch.from_numpy(np.array(patches[start:start+config["inference_batch_size"]], dtype=np.float32)).to(device)
        result.append(model["encoder"](x).cpu().numpy())
    return normalized(np.concatenate(result))


def evaluate(config, suite, split, output, *, model=None, spec=None, device=None, rankings=False, bootstrap=0):
    root, output = Path(config["output_dir"]), Path(output)
    cache = root / "cache"
    with np.load(cache / f"{split}_regions.npz") as archive:
        meta = dict(archive)
    labels = np.load(cache / "labels.npy", mmap_mode="r")
    if split == "valid":
        pool = read_json(root / "valid_pool.json")
        queries = pool["queries"]
        candidates = {int(q): c for q, c in pool["candidates"].items()}
    elif split == "test":
        protocol = Protocol(config)
        queries, candidates = protocol.queries, protocol.candidates
    else:
        raise ValueError("Evaluation supports only valid and explicitly requested test")
    if model is None:
        features = meta["descriptor"].astype(np.float64)
    else:
        family = "A" if spec["preprocess"] == "pad_resize" else "B"
        features = embeddings(model, cache / f"{split}_{family}.npy", config, device)
    indices = {int(i): np.arange(a, b) for i, a, b in zip(meta["map_ids"], meta["offsets"][:-1], meta["offsets"][1:])}
    width = int(np.diff(meta["offsets"]).max())
    padded = np.full((len(meta["map_ids"]), width), -1, dtype=np.int64)
    positions = {int(i): n for n, i in enumerate(meta["map_ids"])}
    for sample_id, tokens in indices.items():
        padded[positions[sample_id], :len(tokens)] = tokens
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with ((output / "rankings.csv").open("w", newline="") if rankings else nullcontext(None)) as stream:
        writer = csv.writer(stream) if stream else None
        if writer:
            writer.writerow(["query_id", "rank", "candidate_id", "similarity_score"])
        for number, q in enumerate(queries, 1):
            c = np.asarray(candidates[q], dtype=np.int64)
            tokens = indices[q]
            if model is None:
                distances = np.linalg.norm(features[tokens, None] - features[None], axis=-1)
                similarities = np.exp(-distances / config["handcrafted"]["distance_scale"])
            else:
                similarities = features[tokens] @ features.T
            scores = batched_map_scores(similarities, padded[[positions[int(i)] for i in c]], meta["area"][tokens], config["matching"])
            if not np.isfinite(scores).all():
                raise FloatingPointError("Nonfinite retrieval score")
            order = np.lexsort((c, -scores))
            if writer:
                writer.writerows((q, rank, int(c[i]), float(scores[i])) for rank, i in enumerate(order, 1))
            relevance = containment(labels[q], labels[c])
            metrics = query_metrics(relevance[order], relevance, config["evaluation"]["ks"])
            rows.append({"query_id": q, "group": "single" if labels[q].sum() == 1 else "mixed", **metrics})
            if number % 100 == 0:
                print(f"Retrieval {split}: {number}/{len(queries)}", flush=True)
    groups = summarize_groups(rows, list(metrics), bootstrap, suite["valid_pool"]["seed"])
    result = {"split": split, "groups": groups, "metrics": groups[0]["metrics"], "num_queries": len(rows),
              "representation": "h512" if model is not None else "Zernike47",
              "pool_sha256": file_hash(root / "valid_pool.json") if split == "valid" else file_hash(config["candidate_manifest"])}
    write_json(output / "metrics.json", result)
    write_json(output / "per_query.json", rows)
    write_csv(output / "per_query_metrics.csv", rows, list(rows[0]))
    return result


def summarize(config, suite, split="valid"):
    root = Path(config["output_dir"]) / f"seed_{config['seed']}"
    rows, per_variant, paired = [], {}, []
    for name in "ABCD":
        for kind in ("random", "best_nce", "best_retrieval"):
            folder = root / name / split / kind
            if not (folder / "metrics.json").exists():
                continue
            result = read_json(folder / "metrics.json")
            rows.extend({"variant": name, "checkpoint": kind, "group": g["group"],
                         "epoch": result.get("epoch", 0), "num_queries": g["num_queries"], **g["metrics"]} for g in result["groups"])
            if kind == "best_retrieval":
                per_variant[name] = {r["query_id"]: r for r in read_json(folder / "per_query.json")}
    hc = Path(config["output_dir"]) / "handcrafted" / split / "metrics.json"
    if hc.exists():
        result = read_json(hc)
        rows.extend({"variant": "handcrafted", "checkpoint": "none", "group": g["group"], "epoch": 0,
                     "num_queries": g["num_queries"], **g["metrics"]} for g in result["groups"])
    for left, right in (("B", "A"), ("C", "B"), ("D", "B")):
        if left not in per_variant or right not in per_variant:
            continue
        if set(per_variant[left]) != set(per_variant[right]):
            raise ValueError("Query sets differ")
        for metric in (suite["selection_metric"], "mAP_full", "FullCoveragePrecision@1"):
            delta = [row[metric] - per_variant[right][q][metric] for q, row in per_variant[left].items()
                     if row[metric] is not None and per_variant[right][q][metric] is not None]
            paired.append({"left": left, "right": right, "metric": metric, "num_queries": len(delta),
                           "difference": float(np.mean(delta)) if delta else None,
                           "query_bootstrap_95ci": bootstrap_interval(delta, config["evaluation"]["bootstrap_samples"],
                                                                      np.random.default_rng(suite["valid_pool"]["seed"]))})
    root.mkdir(parents=True, exist_ok=True)
    if rows:
        write_csv(root / f"comparison_{split}.csv", rows, list(rows[0]))
        write_json(root / f"comparison_{split}.json", rows)
    write_json(root / f"paired_{split}.json", {"comparisons": paired,
        "caution": "valid is used for selection; exploratory query intervals are not final test evidence and exclude training-seed variance"})
    print(f"Summary: {root / f'comparison_{split}.csv'}", flush=True)
