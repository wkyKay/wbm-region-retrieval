"""Frozen protocol validation and portable, memory-mapped preprocessing caches."""

from __future__ import annotations

from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import sys

import numpy as np

from .common import csv_rows, digest, file_hash, iter_csv, read_json, write_json
from .protocol_adapter import is_deduplicated, frozen_source_hashes, audited_source, validate_split_assignments
from .retrieval_protocol import FullGalleryPool, load_retrieval_protocol, protocol_file_hashes


def containment(query_labels, candidate_labels):
    q = np.asarray(query_labels, dtype=np.float64)
    c = np.asarray(candidate_labels, dtype=np.float64)
    if q.sum() <= 0:
        raise ValueError("An empty-label query has no defined containment relevance")
    return np.sum(q * c, axis=-1) / q.sum()


class Protocol:
    def __init__(self, config, labels=None, maps=None):
        from .exact_evaluation import is_exact
        self.exact_evaluation = is_exact(config)
        self.retrieval = None
        self.deduplication = None
        new_protocol = is_deduplicated(config)
        if new_protocol:
            labels, unique, metadata = audited_source(config, labels, maps)
            self.deduplication = unique.summary()
        rows = csv_rows(config["split_manifest"])
        self.ids = {name: [] for name in ("train", "valid", "test")}
        self.labels = {}
        label_columns = sorted((key for key in rows[0] if key.startswith("label_") and key[6:].isdigit()),
                               key=lambda key: int(key[6:])) if rows else []
        if not label_columns:
            raise ValueError("Split manifest must contain label_0 ... label_K columns")
        valid_ids = np.flatnonzero(labels.sum(axis=1) > 0) if labels is not None else None
        for row in rows:
            sample_id = int(row["sample_id"])
            if sample_id in self.labels or row["split"] not in self.ids:
                raise ValueError(f"Duplicate ID or unknown split: {row}")
            if sample_id != int(row["original_index"]):
                raise ValueError("sample_id must equal the original NPZ row index")
            label = np.array([int(row[key]) for key in label_columns], dtype=np.uint8)
            if not np.isin(label, [0, 1]).all() or not label.any():
                raise ValueError(f"Invalid label vector for {sample_id}")
            if labels is not None:
                if sample_id < 0 or sample_id >= len(labels) or not np.array_equal(label, labels[sample_id]):
                    raise ValueError(f"NPZ/manifest label mismatch for {sample_id}")
                valid_index = int(row["valid_index"])
                if not 0 <= valid_index < len(valid_ids) or valid_ids[valid_index] != sample_id:
                    raise ValueError(f"NPZ/manifest valid_index mismatch for {sample_id}")
            self.labels[sample_id] = label
            self.ids[row["split"]].append(sample_id)
        if labels is not None:
            expected = (unique.representative_ids[labels[unique.representative_ids].sum(axis=1) > 0]
                        if new_protocol else valid_ids)
            if set(self.labels) != set(expected.tolist()):
                raise ValueError("Frozen split must cover every nonempty unique representative exactly once"
                                 if new_protocol else "Frozen split must cover every nonempty-label source wafer exactly once")
        if any(not values for values in self.ids.values()):
            raise ValueError("All three splits must be nonempty")
        if new_protocol:
            validate_split_assignments(self.ids, labels, unique, metadata)
            self.retrieval = load_retrieval_protocol(
                config["retrieval_protocol_dir"], data_file=config["data_file"],
                split_file=config["split_manifest"], valid_ids=self.ids["valid"],
                test_ids=self.ids["test"], labels=labels)
            self.queries = list(self.retrieval.test)
            self.candidates = self.retrieval.test
            expected = config.get("expected_candidates")
            if expected is not None and expected != self.candidates.candidates_per_query:
                raise ValueError("expected_candidates disagrees with the frozen full test gallery")
            return
        self.queries = [int(row["sample_id"]) for row in csv_rows(config["query_manifest"])]
        if not self.queries or len(self.queries) != len(set(self.queries)):
            raise ValueError("Query IDs must be nonempty and unique")
        test = set(self.ids["test"])
        if not set(self.queries) <= test:
            raise ValueError("All queries must belong to the frozen test split")
        self.candidates = {query_id: [] for query_id in self.queries}
        # jaccard/bucket columns in the old manifest are deliberately not read.
        for row in iter_csv(config["candidate_manifest"]):
            q, c = int(row["query_id"]), int(row["candidate_id"])
            if q not in self.candidates or c not in test or q == c:
                raise ValueError(f"Invalid candidate edge: {q} -> {c}")
            self.candidates[q].append(c)
        expected = config.get("expected_candidates")
        for q, values in self.candidates.items():
            if not values or len(values) != len(set(values)) or (expected is not None and len(values) != expected):
                raise ValueError(f"Query {q}: expected {expected} unique candidates, found {len(values)}")

    def summary(self):
        result = {
            "split_sizes": {key: len(value) for key, value in self.ids.items()},
            "num_queries": len(self.queries),
            "candidate_counts": ({self.candidates.candidates_per_query: len(self.queries)} if self.retrieval
                                 else dict(Counter(len(value) for value in self.candidates.values()))),
            "label_dimensions": len(next(iter(self.labels.values()))),
            "num_signatures": len({tuple(label) for label in self.labels.values()}),
            "relevance": ("exact equality of the entire nonempty defect-label combination"
                          if self.exact_evaluation else
                          "intersection(query_labels, candidate_labels) / len(query_labels)"),
            "metric_version": "exact_v1" if self.exact_evaluation else "containment",
        }
        if self.retrieval:
            result.update(data_protocol="deduplicated_v1", deduplication=self.deduplication,
                          retrieval_protocol_signature=self.retrieval.signature,
                          candidate_rule="all_gallery_except_query",
                          valid_queries=len(self.retrieval.valid),
                          valid_candidates_per_query=self.retrieval.valid.candidates_per_query)
        return result


def source_fingerprint(config):
    if is_deduplicated(config):
        return frozen_source_hashes(config)
    names = ("data_file", "split_manifest", "query_manifest", "candidate_manifest")
    return {key: file_hash(config[key]) for key in names}


def validation_pool(protocol, settings):
    """Persist O(Q+G) frozen-pool identity; preserve legacy diagnostic fixtures."""
    if protocol.retrieval is not None:
        pool = protocol.retrieval.valid
        frozen = protocol.retrieval.metadata["valid"]
        if (settings.get("queries") != len(pool) or
                settings.get("candidates") != pool.candidates_per_query or
                settings.get("seed") != frozen["seed"]):
            raise ValueError("Valid pool settings must match the frozen retrieval protocol")
        return {"settings": dict(settings), "queries": pool.queries.tolist(), "gallery": pool.gallery.tolist(),
                "candidate_rule": "all_gallery_except_query", "protocol_signature": pool.protocol_signature}
    ids = np.asarray(sorted(protocol.ids["valid"]), dtype=np.int64)
    nq, nc = settings["queries"], settings["candidates"]
    if not 0 < nq <= len(ids) or not 0 < nc < len(ids):
        raise ValueError("Valid query/candidate counts exceed available split; no silent truncation")
    rng = np.random.default_rng(settings["seed"])
    queries = sorted(rng.choice(ids, nq, replace=False).tolist())
    return {"settings": settings, "queries": queries, "candidates": {
        str(q): sorted(rng.choice(ids[ids != q], nc, replace=False).tolist()) for q in queries}}


def decode_pool(payload):
    if "gallery" in payload:
        if payload.get("candidate_rule") != "all_gallery_except_query" or not payload.get("protocol_signature"):
            raise ValueError("Invalid frozen full-gallery pool metadata")
        pool = FullGalleryPool(payload["queries"], payload["gallery"], "valid", payload["protocol_signature"])
        return list(pool), pool
    return payload["queries"], {int(q): c for q, c in payload["candidates"].items()}


def retrieval_pool_hash(config, split, valid_pool_path=None):
    if split == "valid":
        if valid_pool_path is None:
            raise ValueError("A saved valid pool is required")
        return file_hash(valid_pool_path)
    if split != "test":
        raise ValueError("Retrieval split must be valid or test")
    if is_deduplicated(config):
        return digest(protocol_file_hashes(config["retrieval_protocol_dir"]))
    return file_hash(config["candidate_manifest"])


def audit(config):
    with np.load(config["data_file"], allow_pickle=False) as archive:
        labels = archive["arr_1"]
    return Protocol(config, labels).summary()


def prepare(config):
    from experiments.handcrafted.a.descriptor import handcrafted_descriptor
    from .regions import extract_regions, token_patch, whole_map_patch
    output = Path(config["output_dir"])
    cache = output / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    fingerprints = source_fingerprint(config)
    signature = digest({"sources": fingerprints, "proposal": config["proposal"], "input_size": config["input_size"],
                        "handcrafted": config["handcrafted"],
                        "handcrafted_code": file_hash(Path(__file__).parent.parent / "experiments/handcrafted/a/descriptor.py"),
                        "regions_code": file_hash(Path(__file__).with_name("regions.py")),
                        "data_code": file_hash(__file__)})
    status_file = cache / "prepared.json"
    if status_file.exists():
        status = read_json(status_file)
        if status["signature"] != signature:
            raise ValueError("Preprocessing inputs changed; choose a new output_dir instead of mixing experiments")
        required = [cache / "maps.npy", cache / "labels.npy"]
        required += [cache / f"{split}_{name}" for split in ("train", "valid", "test")
                     for name in ("regions.npz", "patches.npy", "whole.npy")]
        if not all(path.is_file() for path in required):
            raise ValueError("Preprocessing completion marker exists but cache files are missing")
        print("Reusing validated preprocessing cache", flush=True)
        return status
    with np.load(config["data_file"], allow_pickle=False) as archive:
        maps, labels = archive["arr_0"], archive["arr_1"]
    if maps.ndim != 3 or labels.ndim != 2 or len(maps) != len(labels):
        raise ValueError("Expected arr_0=(N,H,W), arr_1=(N,K) in the source NPZ")
    if not np.isin(maps, [0, 1, 2, 3]).all():
        raise ValueError("Only the source 0/1/2/3 WBM encoding is supported")
    protocol = Protocol(config, labels, maps=maps)
    np.save(cache / "maps.npy", maps.astype(np.uint8))
    np.save(cache / "labels.npy", labels.astype(np.uint8))
    details = {}
    for split, split_ids in protocol.ids.items():
        ids = np.array(split_ids, dtype=np.int64)
        regions, offsets = [], [0]
        for n, sample_id in enumerate(ids, 1):
            regions.extend(extract_regions(maps[sample_id], **config["proposal"]))
            offsets.append(len(regions))
            if n % 2000 == 0:
                print(f"CC {split}: {n}/{len(ids)} maps", flush=True)
        count, size = len(regions), config["input_size"]
        if not count:
            raise ValueError(f"No retained regions in {split}; check proposal thresholds")
        patches = np.lib.format.open_memmap(cache / f"{split}_patches.npy", mode="w+", dtype=np.uint8,
                                          shape=(count, 1, size, size))
        whole = np.lib.format.open_memmap(cache / f"{split}_whole.npy", mode="w+", dtype=np.uint8,
                                        shape=(len(ids), 1, size, size))
        desc, positions, scales, areas = [], [], [], []
        for i, region in enumerate(regions):
            patches[i, 0] = token_patch(region["mask"], size, config["proposal"]["padding"])
            desc.append(handcrafted_descriptor(region["mask"], config["handcrafted"]["max_order"]))
            positions.append(region["position"])
            scales.append(region["scale"])
            areas.append(region["area"])
        for i, sample_id in enumerate(ids):
            whole[i, 0] = whole_map_patch(maps[sample_id], size)
        patches.flush()
        whole.flush()
        np.savez_compressed(cache / f"{split}_regions.npz", map_ids=ids, offsets=np.array(offsets),
                            area=np.array(areas, dtype=np.float32), position=np.array(positions, dtype=np.float32),
                            scale=np.array(scales, dtype=np.float32), descriptor=np.array(desc, dtype=np.float32))
        counts = np.diff(offsets)
        details[split] = {"maps": len(ids), "tokens": count, "maps_without_tokens": int((counts == 0).sum()),
                          "mean_tokens": float(counts.mean()), "max_tokens": int(counts.max())}
        print(f"Prepared {split}: {details[split]}", flush=True)
    status = {"signature": signature, "source_sha256": fingerprints, "protocol": protocol.summary(), "regions": details,
              "handcrafted": {**config["handcrafted"], "descriptor_dim": len(desc[0])}}
    write_json(output / "resolved_config.json", config)
    packages = {}
    for package in ("numpy", "scipy", "torch", "matplotlib"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    write_json(output / "environment.json", {"python": sys.version, "platform": platform.platform(), "packages": packages})
    write_json(status_file, status)
    return status


def load_regions(config, split):
    cache = Path(config["output_dir"]) / "cache"
    with np.load(cache / f"{split}_regions.npz", allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def stratified_budget(ids, labels, fraction, seed):
    """Nested signature-stratified label subsets; retain at least one per signature."""
    groups = {}
    for sample_id in sorted(ids):
        groups.setdefault(tuple(labels[sample_id]), []).append(sample_id)
    selected = []
    rng = np.random.default_rng(seed)
    for signature in sorted(groups):
        values = np.array(groups[signature], dtype=np.int64)
        rng.shuffle(values)
        selected.extend(values[:max(1, int(np.floor(len(values) * fraction)))].tolist())
    return np.array(sorted(selected), dtype=np.int64)


def sample_pairs(ids, labels, count, seed):
    """Balanced full/partial/zero-coverage ordered pairs, using only supplied IDs."""
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) < 2:
        raise ValueError("At least two labeled wafers are required to sample distinct pairs")
    signatures = {}
    for sample_id in ids:
        signatures.setdefault(tuple(labels[sample_id]), []).append(int(sample_id))
    groups = [np.array(value) for value in signatures.values()]
    targets = np.array(list(signatures), dtype=np.uint8)
    buckets = {}
    for key in signatures:
        rel = containment(np.array(key), targets)
        buckets[key] = [np.flatnonzero(rel == 1), np.flatnonzero((rel > 0) & (rel < 1)), np.flatnonzero(rel == 0)]
    rng = np.random.default_rng(seed)
    pairs = np.empty((count, 2), dtype=np.int64)
    relevance = np.empty(count, dtype=np.float32)
    for index in range(count):
        q = int(rng.choice(ids))
        options = [bucket for bucket in buckets[tuple(labels[q])] if len(bucket)]
        eligible = []
        for bucket in options:
            available = [g for g in bucket if len(groups[g]) > 1 or groups[g][0] != q]
            if available:
                eligible.append(available)
        chosen_group = rng.choice(eligible[int(rng.integers(len(eligible)))])
        members = groups[chosen_group]
        c = int(rng.choice(members))
        while c == q:
            c = int(rng.choice(members))
        pairs[index] = q, c
        relevance[index] = containment(labels[q], labels[c])
    return pairs, relevance
