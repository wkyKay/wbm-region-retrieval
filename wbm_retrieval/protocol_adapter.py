"""Audit the compared_methods frozen protocol without rewriting its manifests.

The adjacent data_protocol/retrieval_protocol modules and protocol manifests are
byte-identical copies from compared_methods. Relative recorded provenance paths
are informational: source identity is checked by content, not machine paths.
"""

from pathlib import Path

import numpy as np

from .common import csv_rows, file_hash, read_json
from .data_protocol import BASE_CLASS_NAMES, DEDUPLICATION_POLICY, deduplicate_maps, stratified_split
from .retrieval_protocol import protocol_file_hashes


def is_deduplicated(config):
    mode = config.get("data_protocol", "legacy")
    if mode not in ("legacy", "deduplicated_v1"):
        raise ValueError(f"Unknown data protocol: {mode}")
    return mode == "deduplicated_v1"


def frozen_source_hashes(config):
    """Validate all content bindings before permitting a cache/model to be reused."""
    split = Path(config["split_manifest"])
    metadata_path = Path(config["deduplication_metadata"])
    metadata = read_json(metadata_path)
    if (metadata.get("format_version") != 1 or
            metadata.get("deduplication_policy") != DEDUPLICATION_POLICY or
            metadata.get("representative_policy") != "smallest original NPZ row ID" or
            metadata.get("label_conflicts") != "rejected" or
            metadata.get("class_names") != list(BASE_CLASS_NAMES)):
        raise ValueError("Unsupported deduplicated split metadata or label order")
    if split.resolve() != (metadata_path.parent / "split.csv").resolve():
        raise ValueError("Frozen split must be the split.csv bound by deduplication metadata")
    files = {name: file_hash(metadata_path.parent / name) for name in ("split.csv", "duplicates.csv")}
    if metadata.get("files") != files:
        raise ValueError("Deduplicated split/duplicates hashes do not match metadata")
    source_hash = file_hash(config["data_file"])
    if source_hash != metadata.get("source_sha256"):
        raise ValueError("Deduplicated split belongs to different source data")
    retrieval_dir = Path(config["retrieval_protocol_dir"])
    retrieval_hashes = protocol_file_hashes(retrieval_dir)
    retrieval = read_json(retrieval_dir / "metadata.json")
    if (retrieval.get("source_sha256") != source_hash or
            retrieval.get("split_sha256") != files["split.csv"]):
        raise ValueError("Frozen retrieval protocol belongs to different source data/split")
    return {
        "data_file": source_hash, "split_manifest": files["split.csv"],
        "deduplication_metadata": file_hash(metadata_path), "duplicates_manifest": files["duplicates.csv"],
        **{f"retrieval_protocol/{name}": sha for name, sha in retrieval_hashes.items()},
        **{f"protocol_code/{name}": file_hash(Path(__file__).with_name(name))
           for name in ("protocol_adapter.py", "data_protocol.py", "retrieval_protocol.py")},
    }


def audited_source(config, labels=None, maps=None):
    """Recheck raw exact-image identity, including all Normal source rows."""
    before = frozen_source_hashes(config)
    with np.load(config["data_file"], allow_pickle=False) as archive:
        source_maps, source_labels = archive["arr_0"], archive["arr_1"]
    if labels is not None and not np.array_equal(labels, source_labels):
        raise ValueError("Provided labels do not match the original NPZ")
    if maps is not None and not np.array_equal(maps, source_maps):
        raise ValueError("Provided maps do not match the original NPZ")
    maps, labels = source_maps, source_labels
    unique = deduplicate_maps(maps, labels)
    metadata = read_json(config["deduplication_metadata"])
    if list(maps.shape) != metadata.get("source_shape") or unique.summary() != metadata.get("summary"):
        raise ValueError("Raw source deduplication differs from frozen metadata")
    removed = np.flatnonzero(unique.representative_of != np.arange(len(maps)))
    expected = {int(i): int(unique.representative_of[i]) for i in removed}
    records = csv_rows(Path(config["deduplication_metadata"]).parent / "duplicates.csv")
    recorded = {int(row["removed_id"]): int(row["representative_id"]) for row in records}
    if len(recorded) != len(records) or recorded != expected:
        raise ValueError("Frozen duplicates manifest differs from exact raw-pixel representatives")
    if frozen_source_hashes(config) != before:
        raise RuntimeError("Frozen source inputs changed during the deduplication audit")
    return labels, unique, metadata


def validate_split_assignments(ids, labels, unique, metadata):
    if {name: len(values) for name, values in ids.items()} != metadata.get("split_counts"):
        raise ValueError("Frozen split counts differ from metadata")
    ratios = metadata.get("split_ratios", {})
    if set(ratios) != {"train", "valid", "test"}:
        raise ValueError("Frozen split metadata needs train/valid/test ratios")
    expected = stratified_split(labels, unique.representative_ids, seed=metadata["seed"],
                                ratios=tuple(ratios[name] for name in ("train", "valid", "test")))
    for name in expected:
        if not np.array_equal(sorted(ids[name]), expected[name]):
            raise ValueError(f"Frozen {name} IDs differ from the declared deduplicated stratified split")
