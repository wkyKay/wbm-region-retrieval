"""Configuration, protected output, whole-map cache and frozen protocols."""

from __future__ import annotations

import fcntl
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from wbm_retrieval.common import ROOT, digest, file_hash, read_json, write_csv, write_json
from wbm_retrieval.data import Protocol, stratified_budget
from wbm_retrieval.models import Encoder
from wbm_retrieval.regions import whole_map_patch


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def method_name(budget):
    return f"supervised_{100 * budget:g}pct"


def load_suite(path, **overrides):
    config = read_json(project_path(path))
    base_path = project_path(config["base_config"])
    base = read_json(base_path)
    config["base_config"] = str(base_path)
    for key in ("data_file", "split_manifest", "query_manifest", "candidate_manifest"):
        config[key] = str(project_path(base[key]))
    config["expected_candidates"] = base["expected_candidates"]
    config["input_size"], config["embedding_dim"], config["num_labels"] = 96, 512, 8
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    if config["encoder_reference"] != "proposed-main":
        raise ValueError("This experiment matches proposed-main (stride=2), not the G encoder")
    if config["threshold"] != 0.5:
        raise ValueError("Version 1 freezes the hard-label threshold at 0.5")
    budgets = config["budgets"]
    if not budgets or len(set(budgets)) != len(budgets) or any(not 0 < b <= 1 for b in budgets):
        raise ValueError("Budgets must be unique fractions in (0, 1]")
    settings = config["training"]
    if settings["optimizer"] != "sgd" or settings["scheduler"] != "wapirl_cosine":
        raise ValueError("Use the proposed SGD/cosine optimizer recipe")
    if settings["learning_rate"] <= 0 or settings["weight_decay"] < 0 or not 0 <= settings["momentum"] < 1:
        raise ValueError("Invalid optimizer settings")
    if settings["batch_size"] < 2 or settings["epochs"] < 1 or not isinstance(settings["rotation"], bool):
        raise ValueError("Need batch_size>=2, epochs>=1 and a boolean rotation setting")
    if min(config["inference_batch_size"], config["train_diagnostic_maps"], config["evaluate_every"],
           config["cpu_threads"], config["valid_pool"]["queries"], config["valid_pool"]["candidates"]) < 1:
        raise ValueError("Batch, diagnostic, pool and interval sizes must be positive")
    if config["num_workers"] < 0 or config["seed"] < 0 or config["evaluation"]["bootstrap_samples"] < 0:
        raise ValueError("Workers, seed and bootstrap count must be nonnegative")
    if 10 not in config["evaluation"]["ks"] or any(k < 1 for k in config["evaluation"]["ks"]):
        raise ValueError("Evaluation ks must be positive and include 10")
    root = project_path(config["output_root"])
    protected = [ROOT, project_path(base["output_dir"])]
    if config.get("budget_source"):
        config["budget_source"] = str(project_path(config["budget_source"]))
        protected.append(Path(config["budget_source"]))
    if root == ROOT or root in ROOT.parents:
        raise ValueError("Output cannot replace or contain the project")
    for source in protected[1:]:
        if root == source or root in source.parents or source in root.parents:
            raise ValueError(f"Output must be separate from existing experiment: {source}")
    for key in ("base_config", "data_file", "split_manifest", "query_manifest", "candidate_manifest"):
        if root == Path(config[key]) or root in Path(config[key]).parents:
            raise ValueError("Output must not contain source inputs")
    config["output_root"] = str(root)
    return config


@contextmanager
def exclusive(root):
    root = Path(root)
    if root.exists() and not (root / "experiment_manifest.json").exists():
        if any(p.name != ".supervised.lock" for p in root.iterdir()):
            raise ValueError(f"Refusing nonempty, unrecognized output directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".supervised.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process is using {root}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def prepare(config):
    root = Path(config["output_root"])
    source_files = [Path(config[k]) for k in ("base_config", "data_file", "split_manifest", "query_manifest", "candidate_manifest")]
    if config.get("budget_source"):
        source_files += [Path(config["budget_source"]) / "models" / f"global_{100*b:g}pct" / "labeled_train_ids.csv"
                         for b in config["budgets"]]
    sources = {str(p): file_hash(p) for p in source_files}
    code_files = sorted(Path(__file__).parent.glob("*.py"))
    code_files += [ROOT / "wbm_retrieval" / name for name in (
        "common.py", "data.py", "models.py", "regions.py", "training.py", "evaluation.py")]
    code_files += [ROOT / "scripts/run_supervised_multilabel.py"]
    scientific = {k: v for k, v in config.items() if k not in ("device", "num_workers", "cpu_threads", "output_root")}
    manifest = {"experiment": "supervised_multilabel_v1", "settings": scientific, "sources": sources,
                "code": {str(p.relative_to(ROOT)): file_hash(p) for p in code_files}}
    signature = digest(manifest)
    manifest["signature"] = signature
    marker = root / "experiment_manifest.json"
    if marker.exists() and read_json(marker) != manifest:
        raise ValueError("Experiment inputs/code/settings changed; use a new output root")
    cache_marker = root / "cache/prepared.json"
    if cache_marker.exists():
        cached = read_json(cache_marker)
        if not marker.exists() or cached["signature"] != signature:
            raise ValueError("Cache signature mismatch")
        for name, expected in cached["files"].items():
            if file_hash(root / name) != expected:
                raise ValueError(f"Prepared cache/protocol was modified: {name}")
        return signature, Protocol(config)
    with np.load(config["data_file"], allow_pickle=False) as archive:
        maps, labels = archive["arr_0"], archive["arr_1"]
    if maps.ndim != 3 or labels.shape != (len(maps), 8) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected maps=(N,H,W) and eight binary label columns")
    if not np.isin(maps, [0, 1, 2, 3]).all():
        raise ValueError("Unknown source map pixel encoding")
    protocol = Protocol(config, labels)
    if config["valid_pool"]["queries"] > len(protocol.ids["valid"]) or config["valid_pool"]["candidates"] >= len(protocol.ids["valid"]):
        raise ValueError("Valid pool exceeds available maps; configure smaller counts explicitly")
    write_json(marker, manifest)
    write_json(root / "resolved_config.json", config)
    cache = root / "cache"
    cache.mkdir(exist_ok=True)
    for split, split_ids in protocol.ids.items():
        ids = np.asarray(sorted(split_ids), dtype=np.int64)
        images = np.lib.format.open_memmap(cache / f"{split}_whole.npy", mode="w+", dtype=np.uint8,
                                          shape=(len(ids), 1, 96, 96))
        for index, sample_id in enumerate(ids):
            images[index, 0] = whole_map_patch(maps[sample_id], 96)
        images.flush()
        del images
        np.save(cache / f"{split}_ids.npy", ids)
        np.save(cache / f"{split}_labels.npy", labels[ids].astype(np.uint8))
    rng = np.random.default_rng(config["valid_pool"]["seed"])
    ids = np.array(sorted(protocol.ids["valid"]))
    queries = sorted(rng.choice(ids, config["valid_pool"]["queries"], replace=False).tolist())
    pool = {"queries": queries, "candidates": {str(q): sorted(rng.choice(ids[ids != q], config["valid_pool"]["candidates"],
                                                          replace=False).tolist()) for q in queries}}
    write_json(root / "protocol/valid_pool.json", pool)
    write_csv(root / "protocol/valid_queries.csv", ({"sample_id": q} for q in queries), ["sample_id"])
    write_csv(root / "protocol/valid_candidates.csv", ({"query_id": q, "candidate_id": c} for q in queries
              for c in pool["candidates"][str(q)]), ["query_id", "candidate_id"])
    for budget in config["budgets"]:
        selected = stratified_budget(protocol.ids["train"], labels, budget, config["seed"])
        if config.get("budget_source"):
            from wbm_retrieval.common import csv_rows
            previous = Path(config["budget_source"]) / "models" / f"global_{100*budget:g}pct/labeled_train_ids.csv"
            old = sorted(int(r["sample_id"]) for r in csv_rows(previous))
            if old != selected.tolist():
                raise ValueError(f"Existing budget IDs differ from the configured seed/protocol: {previous}")
        folder = root / method_name(budget)
        write_csv(folder / "labeled_train_ids.csv", ({"sample_id": int(i)} for i in selected), ["sample_id"])
        write_json(folder / "budget.json", {"requested_fraction": budget, "selected_wafers": len(selected),
                   "actual_fraction": len(selected) / len(protocol.ids["train"]),
                   "validation_labeled_wafers": len(protocol.ids["valid"]), "test_labels_used_in_training": False})
        diagnostic = np.sort(np.random.default_rng(config["seed"]).choice(selected,
                             min(len(selected), config["train_diagnostic_maps"]), replace=False))
        write_json(folder / "train_diagnostic_ids.json", diagnostic.tolist())
    files = list(cache.glob("*.npy")) + list((root / "protocol").glob("*"))
    for budget in config["budgets"]:
        folder = root / method_name(budget)
        files += [folder / name for name in ("labeled_train_ids.csv", "budget.json", "train_diagnostic_ids.json")]
    write_json(cache_marker, {"signature": signature, "protocol": protocol.summary(),
               "files": {str(p.relative_to(root)): file_hash(p) for p in files}})
    return signature, protocol


def split_protocol(config, protocol, split):
    if split == "test":
        return protocol
    pool = read_json(Path(config["output_root"]) / "protocol/valid_pool.json")
    return SimpleNamespace(queries=pool["queries"], candidates={int(q): cs for q, cs in pool["candidates"].items()},
                           labels=protocol.labels)


class WholeMaps(Dataset):
    def __init__(self, root, split, selected=None, rotation=False, seed=2026, epoch=0):
        cache = Path(root) / "cache"
        self.path, self.array = str(cache / f"{split}_whole.npy"), None
        ids = np.load(cache / f"{split}_ids.npy")
        self.labels = np.load(cache / f"{split}_labels.npy")
        lookup = {int(value): i for i, value in enumerate(ids)}
        self.indices = np.array([lookup[int(i)] for i in selected], dtype=np.int64) if selected is not None else np.arange(len(ids))
        self.ids = ids[self.indices]
        self.rotation, self.seed, self.epoch = rotation, seed, epoch

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if self.array is None:
            self.array = np.load(self.path, mmap_mode="r")
        row = self.indices[index]
        image = np.array(self.array[row], dtype=np.float32)
        if self.rotation:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(self.ids[index])]))
            image = np.rot90(image, int(rng.integers(4)), axes=(-2, -1)).copy()
        return int(self.ids[index]), torch.from_numpy(image), torch.from_numpy(self.labels[row].astype(np.float32))

    def __getstate__(self):
        return {**self.__dict__, "array": None}


def make_loader(dataset, config, training=False, epoch=0):
    return DataLoader(dataset, batch_size=config["training"]["batch_size"] if training else config["inference_batch_size"],
                      shuffle=training, num_workers=config["num_workers"],
                      pin_memory=config["device"].startswith("cuda"),
                      generator=torch.Generator().manual_seed(config["seed"] + epoch))


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(512)
        self.classifier = nn.Linear(512, 8)

    def forward(self, image):
        return self.classifier(self.encoder(image))
