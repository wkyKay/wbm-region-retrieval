"""Frozen query lists and complete galleries shared by all retrieval methods.

Candidate identity is represented in O(Q + G) space: each query searches its
saved gallery with only its own original source ID removed. No random candidate
sampling occurs at evaluation time.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data_protocol import deduplicate_maps


FORMAT_VERSION = 1
CANDIDATE_RULE = "all_gallery_except_query"
ALLOCATION_POLICY = "bounded_proportional_largest_remainder_minimum_per_class_v1"
PROTOCOL_FILES = ("valid_queries.csv", "valid_gallery.csv", "test_queries.csv", "test_gallery.csv")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _ids(values, name, *, nonempty=True):
    values = np.asarray(values)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must be one-dimensional integer source IDs")
    if (nonempty and not len(values)) or np.any(values < 0) or len(np.unique(values)) != len(values):
        raise ValueError(f"{name} must contain unique nonnegative source IDs")
    return np.asarray(sorted(values.tolist()), dtype=np.int64)


class FullGalleryPool(Mapping):
    """Lazy query -> full-gallery-excluding-self mapping, with no pair cache."""

    def __init__(self, queries, gallery, split_name, protocol_signature):
        if split_name not in ("valid", "test"):
            raise ValueError("Retrieval split must be valid or test")
        self.queries = _ids(queries, "queries")
        self.gallery = _ids(gallery, "gallery")
        if len(self.gallery) < 2 or not np.isin(self.queries, self.gallery).all():
            raise ValueError("Queries must belong to a gallery containing at least two samples")
        self.queries.flags.writeable = False
        self.gallery.flags.writeable = False
        self._query_set = frozenset(int(value) for value in self.queries)
        self.split_name = split_name
        self.protocol_signature = str(protocol_signature)

    def __len__(self):
        return len(self.queries)

    def __iter__(self):
        return (int(value) for value in self.queries)

    def __getitem__(self, query):
        if isinstance(query, bool) or not isinstance(query, (int, np.integer)) or query not in self._query_set:
            raise KeyError(query)
        return self.gallery[self.gallery != query]

    @property
    def candidates_per_query(self):
        return len(self.gallery) - 1


@dataclass(frozen=True)
class FrozenRetrievalProtocol:
    valid: FullGalleryPool
    test: FullGalleryPool
    metadata: dict
    signature: str


def allocate_query_counts(class_sizes, query_count, min_per_class=5):
    """Allocate a fixed budget proportionally with a lower bound per class.

    Start from each ideal quota's floor, clamped to [min(size, minimum), size].
    Add/remove one at a time according to the largest quota deficit/surplus,
    breaking ties by input class order. Integer numerators avoid roundoff ties.
    """
    query_count = _integer(query_count, "query_count", 1)
    min_per_class = _integer(min_per_class, "min_per_class", 1)
    sizes = np.asarray(class_sizes)
    if sizes.ndim != 1 or not len(sizes) or not np.issubdtype(sizes.dtype, np.integer) or np.any(sizes < 1):
        raise ValueError("class_sizes must contain positive integer class counts")
    sizes = sizes.astype(np.int64)
    total = int(sizes.sum())
    lower = np.minimum(sizes, min_per_class)
    if not int(lower.sum()) <= query_count <= total:
        raise ValueError("Query budget cannot satisfy class minima or exceeds gallery size")
    numerators = sizes * query_count
    counts = np.maximum(lower, numerators // total)
    while int(counts.sum()) > query_count:
        eligible = np.flatnonzero(counts > lower)
        chosen = max(eligible, key=lambda i: int(counts[i] * total - numerators[i]))
        counts[chosen] -= 1
    while int(counts.sum()) < query_count:
        eligible = np.flatnonzero(counts < sizes)
        chosen = max(eligible, key=lambda i: int(numerators[i] - counts[i] * total))
        counts[chosen] += 1
    return counts


def _class_groups(ids, labels):
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.shape[1] != 8 or not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected source labels with shape (N, 8), containing only 0/1")
    groups = defaultdict(list)
    for sample_id in ids:
        if sample_id >= len(labels):
            raise ValueError("Source ID outside label array")
        key = tuple(np.flatnonzero(labels[sample_id]).tolist())
        if not key:
            raise ValueError("Retrieval gallery must contain only nonempty defect labels")
        groups[key].append(int(sample_id))
    return [(key, np.asarray(values, dtype=np.int64)) for key, values in sorted(groups.items())]


def _allocation_records(groups, counts):
    return [{"label_signature": "|".join(map(str, key)), "gallery_count": len(ids),
             "query_count": int(count)} for (key, ids), count in zip(groups, counts)]


def stratified_queries(valid_ids, labels, query_count=500, seed=2026, min_per_class=5):
    ids = _ids(valid_ids, "valid IDs")
    seed = _integer(seed, "seed")
    groups = _class_groups(ids, labels)
    counts = allocate_query_counts([len(values) for _, values in groups], query_count, min_per_class)
    rng = np.random.default_rng(seed)
    chosen = np.concatenate([rng.choice(values, int(count), replace=False)
                             for (_, values), count in zip(groups, counts)])
    return np.sort(chosen), _allocation_records(groups, counts)


def _read_source_split(split_file, labels, representative_ids=None):
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.shape[1] != 8 or not np.isin(labels, [0, 1]).all():
        raise ValueError("Invalid source labels")
    nonempty = np.flatnonzero(labels.sum(axis=1) > 0)
    result = {name: [] for name in ("train", "valid", "test")}
    seen = set()
    with Path(split_file).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            sample_id, split_name = int(row["sample_id"]), row["split"]
            if sample_id in seen or not 0 <= sample_id < len(labels) or split_name not in result:
                raise ValueError("Split contains duplicate/out-of-range ID or unknown split")
            if int(row["original_index"]) != sample_id:
                raise ValueError("Split sample_id must equal original_index")
            target = labels[sample_id]
            if not target.any() or [int(row[f"label_{i}"]) for i in range(8)] != target.tolist():
                raise ValueError("Split label mismatch")
            if "valid_index" in row:
                index = int(row["valid_index"])
                if not 0 <= index < len(nonempty) or nonempty[index] != sample_id:
                    raise ValueError("Split valid_index mismatch")
            seen.add(sample_id)
            result[split_name].append(sample_id)
    if any(not values for values in result.values()):
        raise ValueError("Frozen source split must contain train, valid and test")
    if representative_ids is not None:
        representatives = np.asarray(representative_ids)
        expected = representatives[labels[representatives].sum(axis=1) > 0]
        if seen != set(expected.tolist()):
            raise ValueError("Source split must cover exactly the deduplicated defect representatives")
    taxonomy = {tuple(labels[i]) for i in result["train"]}
    if any(tuple(labels[i]) not in taxonomy for name in ("valid", "test") for i in result[name]):
        raise ValueError("Retrieval class absent from source train taxonomy")
    return {name: np.asarray(sorted(values), dtype=np.int64) for name, values in result.items()}


def _write_ids(path, ids):
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id"])
        writer.writerows((int(sample_id),) for sample_id in ids)


def _read_ids(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["sample_id"]:
            raise ValueError(f"Invalid ID-list columns: {path}")
        values = np.asarray([int(row["sample_id"]) for row in reader], dtype=np.int64)
    if len(values) < 1 or np.any(values < 0) or np.any(np.diff(values) <= 0):
        raise ValueError(f"Frozen ID list must be nonempty, sorted and unique: {path}")
    return values


def protocol_file_hashes(protocol_dir):
    """Return source-independent content hashes, rejecting changed list files."""
    directory = Path(protocol_dir)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    hashes = {name: _sha256(directory / name) for name in PROTOCOL_FILES}
    if metadata.get("files") != hashes:
        raise ValueError("Frozen retrieval protocol file hashes do not match metadata")
    return {"metadata.json": _sha256(directory / "metadata.json"), **hashes}


def build_retrieval_protocol(data_file, split_file, output_dir, seed=2026, valid_queries=500, min_per_class=5):
    """Audit inputs and create an exclusive directory; never overwrite inputs."""
    data_file, split_file, output_dir = (Path(p).expanduser().resolve() for p in (data_file, split_file, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing retrieval protocol: {output_dir}")
    source_hash, split_hash = _sha256(data_file), _sha256(split_file)
    with np.load(data_file, allow_pickle=False) as archive:
        maps, labels = archive["arr_0"], archive["arr_1"]
    unique = deduplicate_maps(maps, labels)
    splits = _read_source_split(split_file, labels, unique.representative_ids)
    queries, allocations = stratified_queries(splits["valid"], labels, valid_queries, seed, min_per_class)
    if len(splits["valid"]) < 2 or len(splits["test"]) < 2:
        raise ValueError("Retrieval galleries need at least two samples")
    lists = {"valid_queries.csv": queries, "valid_gallery.csv": splits["valid"],
             "test_queries.csv": splits["test"], "test_gallery.csv": splits["test"]}
    if _sha256(data_file) != source_hash or _sha256(split_file) != split_hash:
        raise RuntimeError("Source data/split changed while preparing retrieval protocol")
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename, ids in lists.items():
        _write_ids(output_dir / filename, ids)
    metadata = {
        "format_version": FORMAT_VERSION,
        "source_sha256": source_hash, "split_sha256": split_hash,
        "data_file": str(data_file), "split_file": str(split_file),
        "candidate_rule": CANDIDATE_RULE,
        "candidate_order": "original sample_id ascending",
        "identity": "original NPZ row IDs, globally exact-deduplicated before source split",
        "valid": {"query_count": len(queries), "gallery_count": len(splits["valid"]),
                  "candidates_per_query": len(splits["valid"]) - 1,
                  "seed": int(seed), "min_per_class": int(min_per_class),
                  "allocation_policy": ALLOCATION_POLICY,
                  "allocation_tie_break": "label_signature tuple ascending",
                  "class_allocations": allocations},
        "test": {"query_count": len(splits["test"]), "gallery_count": len(splits["test"]),
                 "candidates_per_query": len(splits["test"]) - 1,
                 "query_policy": "all_test_samples"},
        "files": {name: _sha256(output_dir / name) for name in PROTOCOL_FILES},
    }
    with (output_dir / "metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return metadata


def load_retrieval_protocol(protocol_dir, *, data_file, split_file, valid_ids, test_ids, labels):
    """Read the frozen lists and validate their binding to the complete split.

    The caller prepares/deduplicates source arrays first. This loader verifies
    file integrity, original source hashes, split membership, labels and class
    allocation; it never draws new random queries or changes persisted files.
    """
    directory = Path(protocol_dir).expanduser().resolve()
    hashes = protocol_file_hashes(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format_version") != FORMAT_VERSION or metadata.get("candidate_rule") != CANDIDATE_RULE:
        raise ValueError("Unsupported frozen retrieval protocol format/candidate rule")
    if metadata.get("candidate_order") != "original sample_id ascending":
        raise ValueError("Unsupported candidate ordering")
    if metadata.get("source_sha256") != _sha256(data_file) or metadata.get("split_sha256") != _sha256(split_file):
        raise ValueError("Frozen retrieval protocol belongs to different source data/split")
    splits = _read_source_split(split_file, labels)
    for name, provided in (("valid", valid_ids), ("test", test_ids)):
        if not np.array_equal(_ids(provided, f"{name} IDs"), splits[name]):
            raise ValueError(f"Frozen retrieval requires the complete {name} split; subsets are not allowed")
    lists = {name: _read_ids(directory / name) for name in PROTOCOL_FILES}
    for name in ("valid", "test"):
        gallery = lists[f"{name}_gallery.csv"]
        if not np.array_equal(gallery, splits[name]):
            raise ValueError(f"Frozen {name} gallery does not equal the complete source split")
        record = metadata[name]
        if (record.get("gallery_count") != len(gallery) or
                record.get("query_count") != len(lists[f"{name}_queries.csv"]) or
                record.get("candidates_per_query") != len(gallery) - 1):
            raise ValueError(f"Frozen {name} counts do not match ID lists")
    if metadata["test"].get("query_policy") != "all_test_samples" or not np.array_equal(
            lists["test_queries.csv"], lists["test_gallery.csv"]):
        raise ValueError("Frozen test queries must contain every test sample")
    settings = metadata["valid"]
    _integer(settings["seed"], "valid seed")
    if (settings.get("allocation_policy") != ALLOCATION_POLICY or
            settings.get("allocation_tie_break") != "label_signature tuple ascending"):
        raise ValueError("Unsupported stratified query allocation policy")
    groups = _class_groups(splits["valid"], labels)
    counts = allocate_query_counts([len(ids) for _, ids in groups], settings["query_count"], settings["min_per_class"])
    if settings.get("class_allocations") != _allocation_records(groups, counts):
        raise ValueError("Frozen query allocation metadata disagrees with class proportions")
    queries = lists["valid_queries.csv"]
    if not np.isin(queries, splits["valid"]).all():
        raise ValueError("Frozen valid query outside complete valid gallery")
    actual = np.asarray([np.isin(queries, ids).sum() for _, ids in groups])
    if not np.array_equal(actual, counts):
        raise ValueError("Frozen query IDs violate stratified class allocation")
    if protocol_file_hashes(directory) != hashes:
        raise RuntimeError("Retrieval protocol changed while loading")
    protocol_signature = _signature(hashes)
    return FrozenRetrievalProtocol(
        FullGalleryPool(queries, lists["valid_gallery.csv"], "valid", protocol_signature),
        FullGalleryPool(lists["test_queries.csv"], lists["test_gallery.csv"], "test", protocol_signature),
        metadata, protocol_signature,
    )
