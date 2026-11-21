"""Strict read-only loading of the frozen cc8 F training source.

This module never calls preparation, training, evaluation, or identity writers.
Only train/valid feature caches are opened. Test IDs/labels are read solely to
validate the already-frozen split and source bindings, not to measure performance.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from wbm_retrieval.common import csv_rows, digest, file_hash, read_json
from wbm_retrieval.retrieval_protocol import load_retrieval_protocol


CLASS_NAMES = ("center", "donut", "edge-loc", "edge-ring", "loc", "near-full", "scratch", "random")
EXPECTED_SPLIT_COUNTS = {"train": 25193, "valid": 3599, "test": 7199}
EXPECTED_CHECKPOINT_EPOCHS = {"best_nce": 100, "best_retrieval": 15}
_RUNTIME_KEYS = {"device", "num_workers", "output_dir", "data_file", "split_manifest",
                 "deduplication_metadata", "retrieval_protocol_dir"}


def _inside(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe relative source path: {name}")
    resolved = (Path(root) / path).resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Source path escapes its root: {name}")
    return resolved


def _bind(path, protection, expected=None):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen source is missing: {path}")
    actual = protection.get(str(path))
    if actual is None:
        actual = file_hash(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Frozen source SHA-256 mismatch: {path}")
    protection[str(path)] = actual
    return actual


def verify_protection(protection):
    """Rehash every used source after the audit; fail on any missing/change."""
    if not isinstance(protection, dict) or not protection:
        raise ValueError("A nonempty absolute-path SHA-256 protection mapping is required")
    for path, expected in protection.items():
        if not Path(path).is_absolute() or not Path(path).is_file() or file_hash(path) != expected:
            raise ValueError(f"Protected source changed: {path}")
    return {"status": "verified", "files": len(protection)}


def _signed(path, protection):
    _bind(path, protection)
    value = read_json(path)
    if value.get("signature") != digest({k: v for k, v in value.items() if k != "signature"}):
        raise ValueError(f"Invalid frozen manifest signature: {path}")
    return value


def _verify_code(project, recorded, protection):
    """Reconstruct the frozen dependency *paths*, without invoking identity()."""
    files = set()
    for name in ("wbm_retrieval", "experiments/_shared"):
        files.update(p for p in (project / name).rglob("*.py") if p.is_file())
    for name in ("experiments/catalog.json", "experiments/proposed/main/method.py",
                 "experiments/handcrafted/a/descriptor.py",
                 "experiments/proposed/_shared/ef_suite.py",
                 "experiments/proposed/_shared/ef_training.py",
                 "experiments/proposed/_shared/ef_cli.py"):
        files.add(project / name)
    files.update((project / "experiments/proposed/f_segmentation").glob("*.py"))
    expected_paths = {str(path.relative_to(project)) for path in files}
    if not isinstance(recorded, dict) or set(recorded) != expected_paths:
        raise ValueError("Frozen suite code dependency paths changed")
    for name, expected in recorded.items():
        _bind(_inside(project, name), protection, expected)


def _validate_split_rows(rows, labels, counts):
    """Preserve original NPZ IDs and exact CSV order; never relabel/reindex."""
    result = {name: [] for name in ("train", "valid", "test")}
    seen = set()
    nonempty = np.flatnonzero(labels.sum(axis=1) > 0)
    for row in rows:
        sid, split = int(row["sample_id"]), row["split"]
        if sid in seen or not 0 <= sid < len(labels) or split not in result:
            raise ValueError("Duplicate/out-of-range source ID or invalid split")
        if int(row["original_index"]) != sid:
            raise ValueError("sample_id must be original NPZ row ID")
        vector = np.asarray([int(row[f"label_{i}"]) for i in range(8)])
        if not vector.any() or not np.array_equal(vector, labels[sid]):
            raise ValueError(f"Split label vector differs from original NPZ: {sid}")
        valid_index = int(row["valid_index"])
        if not 0 <= valid_index < len(nonempty) or nonempty[valid_index] != sid:
            raise ValueError(f"Split valid_index was confused with sample_id: {sid}")
        if row.get("label_signature") != "|".join(map(str, np.flatnonzero(vector))):
            raise ValueError(f"Label signature differs from declared label order: {sid}")
        result[split].append(sid)
        seen.add(sid)
    if {name: len(values) for name, values in result.items()} != counts:
        raise ValueError("Frozen split counts differ from required protocol")
    return {name: np.asarray(values, dtype=np.int64) for name, values in result.items()}


def _load_dataset(source, split, signature, expected_ids, maps, labels, config, protection):
    if split not in ("train", "valid"):
        raise ValueError("The representation audit opens only train/valid caches")
    cache = source / "cache"
    marker_path = cache / f"{split}_prepared.json"
    _bind(marker_path, protection)
    marker = read_json(marker_path)
    required = {f"cache/{split}_{suffix}" for suffix in
                ("region_labels.npy", "regions.npz", "F.npy", "F_window_audit.csv",
                 "segmentation_audit.csv", "audit.json")} | {"cache/labels.npy", "valid_pool.json"}
    if marker.get("signature") != signature or set(marker.get("files", {})) != required:
        raise ValueError(f"Incomplete or incompatible frozen {split} prepared marker")
    for name, expected in marker["files"].items():
        _bind(_inside(source, name), protection, expected)
    cached_labels = np.load(cache / "labels.npy", mmap_mode="r", allow_pickle=False)
    if not np.array_equal(cached_labels, labels):
        raise ValueError("Source cache labels differ from original NPZ labels")
    with np.load(cache / f"{split}_regions.npz", allow_pickle=False) as archive:
        if not {"map_ids", "offsets", "area"} <= set(archive.files):
            raise ValueError("Region metadata lacks source IDs, offsets, or areas")
        ids, offsets, area = (np.array(archive[name]) for name in ("map_ids", "offsets", "area"))
    if (ids.dtype.kind not in "iu" or ids.ndim != 1
            or not np.array_equal(ids, expected_ids) or len(np.unique(ids)) != len(ids)):
        raise ValueError(f"Frozen {split} cache sample IDs/order differ from split.csv")
    if (offsets.dtype.kind not in "iu" or offsets.shape != (len(ids) + 1,) or offsets[0] != 0
            or np.any(np.diff(offsets.astype(np.int64)) < 0)
            or np.any(np.diff(offsets.astype(np.int64)) > config["proposal"]["top_k"])):
        raise ValueError(f"Invalid frozen {split} region offsets")
    if (area.dtype.kind not in "iu" or area.shape != (int(offsets[-1]),)
            or np.any(area < config["proposal"]["min_area"])):
        raise ValueError(f"Invalid frozen {split} token areas")
    masks = np.load(cache / f"{split}_region_labels.npy", mmap_mode="r", allow_pickle=False)
    patches = np.load(cache / f"{split}_F.npy", mmap_mode="r", allow_pickle=False)
    if (masks.shape != (len(ids), *maps.shape[1:]) or masks.dtype.kind not in "iu"
            or patches.shape != (len(area), 3, config["input_size"], config["input_size"])
            or patches.dtype != np.uint8):
        raise ValueError(f"Invalid frozen {split} mask/patch shape or dtype")
    for row, sid in enumerate(ids):
        start, end = map(int, offsets[row:row + 2])
        region_ids = np.asarray(masks[row])
        if np.any(region_ids > end - start) or np.any((region_ids > 0) & (maps[sid] != 2)):
            raise ValueError(f"Cached region IDs/nondefect membership invalid for source {sid}")
        counts = np.bincount(region_ids.ravel(), minlength=end - start + 1)[1:]
        if not np.array_equal(counts, area[start:end]):
            raise ValueError(f"Region area/order differs from original-grid mask for source {sid}")
        if len(counts) > 1 and np.any(np.diff(counts) > 0):
            raise ValueError("Frozen cc8 regions must be in descending original pixel area")
    sizes = np.diff(offsets).astype(np.int64)
    token_map_ids = np.repeat(ids, sizes)
    token_region_ids = np.concatenate([np.arange(1, count + 1, dtype=np.int64) for count in sizes])
    map_singletons = np.full(len(labels), -1, dtype=np.int64)
    singleton = labels.sum(axis=1) == 1
    map_singletons[singleton] = labels[singleton].argmax(axis=1)
    return {
        "cache": cache, "patches": patches, "region_masks": masks,
        "map_ids": ids, "offsets": offsets, "area": area,
        "token_map_ids": token_map_ids, "token_region_ids": token_region_ids,
        "singleton_labels": map_singletons[token_map_ids],
        "rep_indices": offsets[:-1][sizes > 0].astype(np.int64), "labels": labels,
        "prepared_marker": marker,
    }


def verify_checkpoint_metadata(checkpoint, kind, inputs):
    """Validate a caller-loaded checkpoint without importing Torch here."""
    if kind not in EXPECTED_CHECKPOINT_EPOCHS or not isinstance(checkpoint, dict):
        raise ValueError("Expected a best_nce or best_retrieval project checkpoint")
    expected_epoch = EXPECTED_CHECKPOINT_EPOCHS[kind]
    run = inputs["run_manifest"]
    if (checkpoint.get("signature") != run["signature"] or checkpoint.get("variant") != "F"
            or checkpoint.get("input_spec") != inputs["spec"]
            or checkpoint.get("epoch") != expected_epoch
            or inputs["run_summary"][f"{kind}_epoch"] != expected_epoch
            or checkpoint.get("initial_parameter_sha256") != run["initial_parameter_sha256"]):
        raise ValueError(f"Frozen {kind} checkpoint metadata differs from its source identity")
    if not isinstance(checkpoint.get("model"), dict) or not checkpoint["model"]:
        raise ValueError("Checkpoint has no model state dictionary")
    _bind(inputs["checkpoint_paths"][kind], {}, inputs["protection"][str(inputs["checkpoint_paths"][kind])])
    return {"kind": kind, "epoch": expected_epoch,
            "sha256": inputs["protection"][str(inputs["checkpoint_paths"][kind])]}


def load_inputs(project_root, source_root, data_file=None):
    """Load verified cc8 inputs; return source arrays and provenance, no writes."""
    project, source = Path(project_root).resolve(), Path(source_root).resolve()
    protection = {}
    manifest = _signed(source / "suite_manifest.json", protection)
    if manifest.get("format") != "f_segmentation_v1":
        raise ValueError("Expected the frozen f_segmentation_v1 suite")
    _verify_code(project, manifest.get("code"), protection)
    config, suite = copy.deepcopy(manifest["settings"]), copy.deepcopy(manifest["suite"])
    spec = suite["variants"]["F"]
    expected_spec = {"preprocess": "fixed_window", "content_size": 88, "stem_stride": 1,
                     "scale_range": None, "max_shift": 0, "input_channels": 3, "window_size": 52}
    if (config.get("seed") != 2026 or config.get("data_protocol") != "deduplicated_v1"
            or config.get("final_checkpoint") != "best_nce" or config.get("input_size") != 96
            or config.get("embedding_dim") != 512 or config.get("batch_size") != 256
            or config.get("proposal") != {"min_area": 5, "top_k": 4, "padding": 4}
            or config.get("segmentation") != {"method": "cc8", "parameters": {"connectivity": 8}, "version": "f_segmentation_v1"}
            or config.get("augmentation") != {"rotation": True, "max_shift": 0, "dropout": 0., "noise": 0.}
            or config["local"].get("epochs") != 100 or config["local"].get("projector_dim") != 128
            or config["local"].get("num_negatives") != 5000
            or config["local"].get("opening", {}).get("enabled") is not False
            or spec != expected_spec):
        raise ValueError("The audit requires the frozen cc8 Top-4, rotation-only F source")
    common = _signed(source.parent / "segmentation_comparison.json", protection)
    expected_common = copy.deepcopy({k: v for k, v in manifest.items() if k != "signature"})
    expected_common["settings"].pop("segmentation")
    if common != {"signature": digest(expected_common), **expected_common}:
        raise ValueError("cc8 suite differs from the common frozen comparison identity")
    run_folder = source / "seed_2026/F"
    for filename in ("run_manifest.json", "training_summary.json"):
        _bind(run_folder / filename, protection)
    run = read_json(run_folder / "run_manifest.json")
    summary = read_json(run_folder / "training_summary.json")
    signature = digest({"suite": manifest["signature"], "seed": 2026, "variant": "F"})
    if (run.get("signature") != signature or run.get("variant") != "F" or run.get("input") != spec
            or {k: v for k, v in run["config"].items() if k not in _RUNTIME_KEYS} != manifest["settings"]
            or run.get("default_test_checkpoint") != "best_nce"
            or run.get("selection_metric") != suite.get("selection_metric")):
        raise ValueError("Training manifest is incompatible with the frozen suite")
    if (summary.get("completed") is not True or summary.get("epochs_completed") != 100
            or summary.get("max_epochs") != 100 or summary.get("input_spec") != spec
            or summary.get("initial_parameter_sha256") != run.get("initial_parameter_sha256")
            or any(summary.get(f"{kind}_epoch") != epoch for kind, epoch in EXPECTED_CHECKPOINT_EPOCHS.items())):
        raise ValueError("Expected completed 100-epoch F with best_nce=100 and best_retrieval=15")
    path_defaults = {"data_file": project.parent / "data/wm38k/Wafer_Map_Datasets.npz",
                     "split_manifest": project / "protocol/deduplicated/split.csv",
                     "deduplication_metadata": project / "protocol/deduplicated/metadata.json",
                     "retrieval_protocol_dir": project / "protocol/retrieval_v1"}
    for name, fallback in path_defaults.items():
        recorded = Path(run["config"][name]).expanduser()
        config[name] = str(Path(data_file).expanduser().resolve() if name == "data_file" and data_file is not None
                           else recorded.resolve() if recorded.exists() else fallback.resolve())
    config.update(output_dir=str(source), device="cpu", num_workers=0)
    split_path = Path(config["split_manifest"])
    dedup_path = Path(config["deduplication_metadata"])
    retrieval_dir = Path(config["retrieval_protocol_dir"])
    if split_path != dedup_path.parent / "split.csv":
        raise ValueError("Split is not the split.csv bound by the deduplication metadata")
    source_paths = {"data_file": Path(config["data_file"]), "split_manifest": split_path,
                    "deduplication_metadata": dedup_path, "duplicates_manifest": dedup_path.parent / "duplicates.csv"}
    source_paths.update({f"retrieval_protocol/{name}": retrieval_dir / name for name in
                         ("metadata.json", "valid_queries.csv", "valid_gallery.csv", "test_queries.csv", "test_gallery.csv")})
    source_paths.update({f"protocol_code/{name}": project / "wbm_retrieval" / name for name in
                         ("protocol_adapter.py", "data_protocol.py", "retrieval_protocol.py")})
    if set(source_paths) != set(manifest["sources"]):
        raise ValueError("Frozen suite source binding paths changed")
    for name, path in source_paths.items():
        _bind(path, protection, manifest["sources"][name])
    dedup = read_json(dedup_path)
    if (dedup.get("class_names") != list(CLASS_NAMES) or dedup.get("split_counts") != EXPECTED_SPLIT_COUNTS
            or dedup.get("source_sha256") != manifest["sources"]["data_file"]
            or dedup.get("files") != {"split.csv": manifest["sources"]["split_manifest"],
                                      "duplicates.csv": manifest["sources"]["duplicates_manifest"]}
            or dedup.get("seed") != 2026):
        raise ValueError("Frozen protocol label order/counts/source bindings are incompatible")
    with np.load(config["data_file"], allow_pickle=False) as archive:
        maps, labels = archive["arr_0"], archive["arr_1"]
    if (maps.shape != (38015, 52, 52) or list(maps.shape) != dedup.get("source_shape")
            or maps.dtype.kind not in "iu" or maps.min() < 0 or maps.max() > 3
            or labels.shape != (len(maps), 8) or labels.dtype.kind not in "iu"
            or not np.isin(labels, [0, 1]).all()):
        raise ValueError("Unexpected MixedWM38 source maps/labels")
    splits = _validate_split_rows(csv_rows(split_path), labels, EXPECTED_SPLIT_COUNTS)
    removed = [int(row["removed_id"]) for row in csv_rows(dedup_path.parent / "duplicates.csv")]
    if len(set(removed)) != len(removed) or any(not 0 <= sid < len(labels) for sid in removed):
        raise ValueError("Invalid duplicate source IDs")
    representatives = set(np.flatnonzero(labels.sum(axis=1) > 0).tolist()) - set(removed)
    if set(np.concatenate(list(splits.values())).tolist()) != representatives:
        raise ValueError("Split does not cover the frozen nonempty representative IDs")
    retrieval = load_retrieval_protocol(retrieval_dir, data_file=config["data_file"], split_file=split_path,
                                        valid_ids=splits["valid"], test_ids=splits["test"], labels=labels)
    pool_path = source / "valid_pool.json"
    _bind(pool_path, protection)
    expected_pool = {"settings": suite["valid_pool"], "queries": retrieval.valid.queries.tolist(),
                     "gallery": retrieval.valid.gallery.tolist(), "candidate_rule": "all_gallery_except_query",
                     "protocol_signature": retrieval.valid.protocol_signature}
    if read_json(pool_path) != expected_pool:
        raise ValueError("Source valid pool differs from the frozen retrieval lists")
    datasets = {split: _load_dataset(source, split, manifest["signature"], splits[split], maps, labels,
                                      config, protection) for split in ("train", "valid")}
    for split, dataset in datasets.items():
        protocol = dataset["prepared_marker"].get("protocol", {})
        if (protocol.get("split_sizes") != EXPECTED_SPLIT_COUNTS
                or protocol.get("retrieval_protocol_signature") != retrieval.signature
                or protocol.get("label_dimensions") != 8 or protocol.get("metric_version") != "exact_v1"):
            raise ValueError(f"Frozen {split} prepared protocol differs from source bindings")
    checkpoint_paths = {kind: (run_folder / f"{kind}.pt").resolve() for kind in EXPECTED_CHECKPOINT_EPOCHS}
    for path in checkpoint_paths.values():
        _bind(path, protection)
    # These source-owned final reports bind the main checkpoint when present.
    # Auxiliary best_retrieval has no separate final report; its hash is frozen
    # for this audit and the caller must check its payload via the helper above.
    report_marker = run_folder / "valid/best_nce/completed.json"
    if report_marker.exists():
        _bind(report_marker, protection)
        marker = read_json(report_marker)
        if (marker.get("signature") != manifest["signature"] or marker.get("split") != "valid"
                or marker.get("checkpoint_kind") != "best_nce"
                or marker.get("checkpoint_sha256") != protection[str(checkpoint_paths["best_nce"])]):
            raise ValueError("Main checkpoint no longer matches its frozen valid evaluation binding")
    maps.setflags(write=False)
    labels.setflags(write=False)
    return {"maps": maps, "labels": labels, "config": config, "spec": spec, "suite": suite,
            "checkpoint_paths": checkpoint_paths, "run_summary": summary, "source_manifest": manifest,
            "run_manifest": run, "protection": dict(sorted(protection.items())), "datasets": datasets,
            "split_ids": splits, "source_root": source, "project_root": project,
            "checkpoint_payload_validation_required": True}
