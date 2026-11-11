"""Frozen deduplicated IDs, complete candidate pools and run isolation."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wbm_retrieval.common import ROOT, csv_rows, file_hash, load_config, read_json, write_csv, write_json
from wbm_retrieval.data import (Protocol, decode_pool, prepare, retrieval_pool_hash, source_fingerprint,
                                validation_pool)
from wbm_retrieval.data_protocol import (BASE_CLASS_NAMES, DEDUPLICATION_POLICY, deduplicate_maps,
                                         stratified_split)
from wbm_retrieval.retrieval_protocol import FullGalleryPool, build_retrieval_protocol


def fixture(root, per_class=30):
    """Include source-ID holes, Normal rows and duplicates of both defect/Normal."""
    root = Path(root)
    maps, labels = [], []
    defect = 0
    normal_positions = tuple(i * (per_class + 1) for i in range(4))
    for i in range(3 * per_class + 4):
        raw = np.ones((8, 8), dtype=np.uint8)
        raw[0, :4] = [(i >> (2 * digit)) & 3 for digit in range(4)]
        raw[3:6, 3:6] = 2
        label = np.zeros(8, dtype=np.uint8)
        if i not in normal_positions:
            label[defect // per_class] = 1
            defect += 1
        maps.append(raw)
        labels.append(label)
        if i in (2, per_class - 1, normal_positions[-1]):
            maps.append(raw.copy())
            labels.append(label.copy())
    maps, labels = np.asarray(maps), np.asarray(labels)
    source = root / "source.npz"
    np.savez(source, arr_0=maps, arr_1=labels)
    unique = deduplicate_maps(maps, labels)
    splits = stratified_split(labels, unique.representative_ids, seed=2026)
    nonempty = np.flatnonzero(labels.sum(axis=1))
    valid_indices = {int(sid): i for i, sid in enumerate(nonempty)}
    split_dir = root / "deduplicated"
    rows = []
    for name, ids in splits.items():
        for sid in ids:
            sid = int(sid)
            row = {"sample_id": sid, "original_index": sid, "valid_index": valid_indices[sid],
                   "split": name, "label_signature": "|".join(map(str, np.flatnonzero(labels[sid])))}
            row.update({f"label_{i}": int(labels[sid, i]) for i in range(8)})
            rows.append(row)
    write_csv(split_dir / "split.csv", rows, list(rows[0]))
    duplicates = [{"removed_id": sid, "representative_id": int(representative)}
                  for sid, representative in enumerate(unique.representative_of) if sid != representative]
    write_csv(split_dir / "duplicates.csv", duplicates, ["removed_id", "representative_id"])
    write_json(split_dir / "metadata.json", {
        "format_version": 1, "data_file": str(source), "source_sha256": file_hash(source),
        "source_shape": list(maps.shape), "deduplication_policy": DEDUPLICATION_POLICY,
        "representative_policy": "smallest original NPZ row ID",
        "identity": "all raw pixel codes 0/1/2/3; hash candidates confirmed by array_equal",
        "label_conflicts": "rejected", "seed": 2026,
        "split_ratios": {"train": .7, "valid": .1, "test": .2},
        "split_counts": {name: len(ids) for name, ids in splits.items()},
        "class_names": list(BASE_CLASS_NAMES), "summary": unique.summary(),
        "normal_policy": "unique Normal IDs excluded from split.csv; runtime uses explicit normal_split counts",
        "valid_index_policy": "index in original nonempty-label source rows, before deduplication",
        "files": {name: file_hash(split_dir / name) for name in ("split.csv", "duplicates.csv")},
    })
    retrieval = root / "retrieval"
    build_retrieval_protocol(source, split_dir / "split.csv", retrieval,
                             seed=2026, valid_queries=6, min_per_class=2)
    config = copy.deepcopy(read_json(ROOT / "experiments/_shared/base.json"))
    # These migration tests keep their historical containment evaluator. Exact
    # metric integration tests explicitly opt in after creating the fixture.
    config["evaluation"].pop("metric_version", None)
    config["evaluation"]["ks"] = [1, 5, 10, 20]
    config.update(data_protocol="deduplicated_v1", data_file=str(source),
                  split_manifest=str(split_dir / "split.csv"),
                  deduplication_metadata=str(split_dir / "metadata.json"),
                  retrieval_protocol_dir=str(retrieval), expected_candidates=len(splits["test"]) - 1,
                  output_dir=str(root / "output"), device="cpu", num_workers=0, input_size=32)
    for key in ("query_manifest", "candidate_manifest"):
        config.pop(key, None)
    return config, maps, labels, unique, splits


def refresh_hash(config, filename):
    metadata_path = Path(config["deduplication_metadata"])
    metadata = read_json(metadata_path)
    metadata["files"][filename] = file_hash(metadata_path.parent / filename)
    write_json(metadata_path, metadata)
    if filename == "split.csv":
        retrieval_path = Path(config["retrieval_protocol_dir"]) / "metadata.json"
        retrieval = read_json(retrieval_path)
        retrieval["split_sha256"] = metadata["files"][filename]
        write_json(retrieval_path, retrieval)


class DeduplicatedProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config, self.maps, self.labels, self.unique, self.splits = fixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_original_ids_and_complete_unique_representatives_are_used(self):
        before = {p: file_hash(p) for p in self.root.rglob("*") if p.is_file()}
        protocol = Protocol(self.config, self.labels, maps=self.maps)
        self.assertEqual({name: len(ids) for name, ids in protocol.ids.items()},
                         {"train": 63, "valid": 9, "test": 18})
        all_ids = set().union(*(set(ids) for ids in protocol.ids.values()))
        expected = {int(sid) for sid in self.unique.representative_ids if self.labels[sid].any()}
        self.assertEqual(all_ids, expected)
        self.assertEqual(len(all_ids), 90)
        self.assertNotEqual(all_ids, set(range(90)))
        self.assertFalse(all_ids & {int(row["removed_id"]) for row in csv_rows(self.root / "deduplicated/duplicates.csv")})
        for sid in all_ids:
            np.testing.assert_array_equal(protocol.labels[sid], self.labels[sid])
        self.assertFalse(set(protocol.ids["train"]) & set(protocol.ids["valid"]))
        self.assertFalse(set(protocol.ids["train"]) & set(protocol.ids["test"]))
        self.assertEqual(before, {p: file_hash(p) for p in before})

    def test_frozen_queries_do_not_depend_on_training_seed_or_sampling(self):
        protocol = Protocol(self.config)
        settings = {"seed": 2026, "queries": 6, "candidates": 8}
        payload = validation_pool(protocol, settings)
        changed = Protocol({**self.config, "seed": 19})
        other = validation_pool(changed, settings)
        self.assertEqual(payload, other)
        self.assertNotIn("candidates", payload)
        queries, pool = decode_pool(payload)
        self.assertIsInstance(pool, FullGalleryPool)
        self.assertEqual(list(queries), list(protocol.retrieval.valid.queries))
        for q in queries:
            self.assertEqual(set(pool[q]), set(protocol.ids["valid"]) - {q})
            self.assertFalse(set(pool[q]) & set(protocol.ids["train"]))
        self.assertEqual(list(protocol.queries), sorted(protocol.ids["test"]))
        self.assertIsInstance(protocol.candidates, FullGalleryPool)
        for q in protocol.queries:
            self.assertEqual(set(protocol.candidates[q]), set(protocol.ids["test"]) - {q})
        with self.assertRaises(ValueError):
            validation_pool(protocol, {**settings, "candidates": 3})

    def test_protocol_summary_describes_the_selected_evaluation_version(self):
        legacy = Protocol(self.config).summary()
        self.assertEqual(legacy["metric_version"], "containment")
        self.assertIn("intersection", legacy["relevance"])
        config = {**self.config, "evaluation": {**self.config["evaluation"], "metric_version": "exact_v1"}}
        exact = Protocol(config).summary()
        self.assertEqual(exact["metric_version"], "exact_v1")
        self.assertEqual(exact["relevance"], "exact equality of the entire nonempty defect-label combination")
        self.assertEqual(exact["retrieval_protocol_signature"], legacy["retrieval_protocol_signature"])

    def test_test_candidates_use_linear_storage_and_returns_are_not_cached(self):
        ids = np.arange(7199, dtype=np.int64)
        pool = FullGalleryPool(ids, ids, "test", "frozen")
        arrays = [value for value in vars(pool).values() if isinstance(value, np.ndarray)]
        self.assertEqual(sum(a.size for a in arrays), 2 * len(ids))
        candidates = pool[0]
        self.assertEqual(len(candidates), 7198)
        candidates[0] = -1
        np.testing.assert_array_equal(pool[0], np.arange(1, 7199))
        self.assertFalse(pool.gallery.flags.writeable)

    def test_missing_representative_rejected_even_if_csv_hash_is_updated(self):
        path = Path(self.config["split_manifest"])
        rows = csv_rows(path)
        write_csv(path, rows[1:], list(rows[0]))
        refresh_hash(self.config, "split.csv")
        with self.assertRaisesRegex(ValueError, "unique representative"):
            Protocol(self.config, self.labels, maps=self.maps)

    def test_retained_duplicate_rejected_even_if_source_labels_match(self):
        path = Path(self.config["split_manifest"])
        rows = csv_rows(path)
        duplicate = next(row for row in csv_rows(self.root / "deduplicated/duplicates.csv")
                         if self.labels[int(row["removed_id"])].any())
        representative, removed = int(duplicate["representative_id"]), int(duplicate["removed_id"])
        row = next(row for row in rows if int(row["sample_id"]) == representative)
        row.update(sample_id=str(removed), original_index=str(removed),
                   valid_index=str(np.flatnonzero(self.labels.sum(axis=1)).tolist().index(removed)))
        write_csv(path, rows, list(rows[0]))
        refresh_hash(self.config, "split.csv")
        with self.assertRaisesRegex(ValueError, "unique representative"):
            Protocol(self.config, self.labels, maps=self.maps)

    def test_hashes_invalid_indices_and_duplicate_map_are_checked(self):
        path = Path(self.config["split_manifest"])
        original = path.read_bytes()
        rows = csv_rows(path)
        rows[0]["valid_index"] = "-1"
        write_csv(path, rows, list(rows[0]))
        with self.assertRaises(ValueError):
            Protocol(self.config)
        refresh_hash(self.config, "split.csv")
        with self.assertRaisesRegex(ValueError, "valid_index"):
            Protocol(self.config)
        path.write_bytes(original)
        refresh_hash(self.config, "split.csv")
        duplicate_path = self.root / "deduplicated/duplicates.csv"
        duplicates = csv_rows(duplicate_path)
        duplicates[0]["representative_id"] = duplicates[0]["removed_id"]
        write_csv(duplicate_path, duplicates, list(duplicates[0]))
        refresh_hash(self.config, "duplicates.csv")
        with self.assertRaisesRegex(ValueError, "duplicates manifest"):
            Protocol(self.config)

    def test_split_membership_cannot_change_under_same_declared_seed(self):
        path = Path(self.config["split_manifest"])
        rows = csv_rows(path)
        train = next(row for row in rows if row["split"] == "train" and row["label_0"] == "1")
        valid = next(row for row in rows if row["split"] == "valid" and row["label_0"] == "1")
        train["split"], valid["split"] = valid["split"], train["split"]
        write_csv(path, rows, list(rows[0]))
        refresh_hash(self.config, "split.csv")
        with self.assertRaisesRegex(ValueError, "declared deduplicated stratified split"):
            Protocol(self.config)

    def test_preloaded_arrays_must_equal_the_hashed_source(self):
        maps = self.maps.copy()
        maps[0, 7, 7] = 3
        with self.assertRaisesRegex(ValueError, "maps.*NPZ"):
            Protocol(self.config, self.labels, maps=maps)
        labels = self.labels.copy()
        labels[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "labels.*NPZ"):
            Protocol(self.config, labels, maps=self.maps)

    def test_identical_maps_with_conflicting_labels_and_all_pixel_codes(self):
        duplicate = next(i for i, representative in enumerate(self.unique.representative_of) if i != representative)
        labels = self.labels.copy()
        labels[duplicate] = np.roll(labels[duplicate], 1)
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            deduplicate_maps(self.maps, labels)
        np.savez(self.config["data_file"], arr_0=self.maps, arr_1=labels)
        for path in (Path(self.config["deduplication_metadata"]),
                     Path(self.config["retrieval_protocol_dir"]) / "metadata.json"):
            metadata = read_json(path)
            metadata["source_sha256"] = file_hash(self.config["data_file"])
            write_json(path, metadata)
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            Protocol(self.config)
        maps = np.ones((4, 8, 8), dtype=np.uint8)
        maps[:, 0, 0] = np.arange(4)
        labels = np.zeros((4, 8), dtype=np.uint8)
        labels[:, 0] = 1
        np.testing.assert_array_equal(deduplicate_maps(maps, labels).representative_ids, np.arange(4))

    def test_frozen_retrieval_tampering_rejected_and_fingerprint_changes(self):
        before = source_fingerprint(self.config)
        directory = Path(self.config["retrieval_protocol_dir"])
        path = directory / "valid_gallery.csv"
        rows = csv_rows(path)
        rows[0]["sample_id"] = str(self.splits["train"][0])
        rows.sort(key=lambda row: int(row["sample_id"]))
        write_csv(path, rows, ["sample_id"])
        with self.assertRaises(ValueError):
            Protocol(self.config)
        metadata = read_json(directory / "metadata.json")
        metadata["files"][path.name] = file_hash(path)
        write_json(directory / "metadata.json", metadata)
        with self.assertRaises(ValueError):
            Protocol(self.config)
        # Valid changed manifests cannot retain a previous run's source identity.
        self.assertNotEqual(before, source_fingerprint(self.config))

    def test_compact_pool_rejects_self_leakage_or_unknown_rule(self):
        protocol = Protocol(self.config)
        payload = validation_pool(protocol, {"seed": 2026, "queries": 6, "candidates": 8})
        bad = copy.deepcopy(payload)
        bad["gallery"] = [int(self.splits["train"][0])]
        with self.assertRaises(ValueError):
            decode_pool(bad)
        bad = copy.deepcopy(payload)
        bad["candidate_rule"] = "include_self"
        with self.assertRaises(ValueError):
            decode_pool(bad)
        path = self.root / "valid_pool.json"
        write_json(path, payload)
        self.assertEqual(retrieval_pool_hash(self.config, "valid", path), file_hash(path))
        self.assertEqual(retrieval_pool_hash(self.config, "test"),
                         retrieval_pool_hash({**self.config, "seed": 42}, "test"))

    def test_old_cache_marker_is_rejected_without_overwriting_old_artifacts(self):
        cache = Path(self.config["output_dir"]) / "cache"
        cache.mkdir(parents=True)
        write_json(cache / "prepared.json", {"signature": "historical-undeduplicated-run"})
        sentinel = cache / "maps.npy"
        sentinel.write_bytes(b"historical cache must remain untouched")
        before = {path.name: path.read_bytes() for path in cache.iterdir()}
        with self.assertRaisesRegex(ValueError, "new output_dir"):
            prepare(self.config)
        self.assertEqual(before, {path.name: path.read_bytes() for path in cache.iterdir()})


class FrozenMirrorTests(unittest.TestCase):
    def test_frozen_files_and_actual_ids_match_compared_methods(self):
        reference = ROOT.parent / "compared_methods/protocol"
        files = [f"deduplicated/{name}" for name in ("split.csv", "duplicates.csv", "metadata.json")]
        files += [f"retrieval_v1/{name}" for name in
                  ("valid_queries.csv", "valid_gallery.csv", "test_queries.csv", "test_gallery.csv", "metadata.json")]
        for name in files:
            with self.subTest(file=name):
                self.assertTrue((ROOT / "protocol" / name).is_file())
                if reference.is_dir():
                    self.assertEqual(file_hash(ROOT / "protocol" / name), file_hash(reference / name))
        rows = csv_rows(ROOT / "protocol/deduplicated/split.csv")
        splits = {name: {int(row["sample_id"]) for row in rows if row["split"] == name}
                  for name in ("train", "valid", "test")}
        self.assertEqual({name: len(ids) for name, ids in splits.items()},
                         {"train": 25193, "valid": 3599, "test": 7199})
        self.assertEqual(len(set().union(*splits.values())), 35991)
        for split in ("valid", "test"):
            gallery = {int(row["sample_id"]) for row in csv_rows(ROOT / f"protocol/retrieval_v1/{split}_gallery.csv")}
            self.assertEqual(gallery, splits[split])
        queries = csv_rows(ROOT / "protocol/retrieval_v1/valid_queries.csv")
        self.assertEqual(len(queries), 500)
        self.assertEqual(file_hash(ROOT / "protocol/split.csv"),
                         "e1785da387a70da63b65052975872da1377d15f690d4c37c5009840d08444cda")

    def test_real_source_audit_accepts_only_unique_frozen_ids(self):
        config = load_config(ROOT / "experiments/_shared/base.json")
        if not Path(config["data_file"]).is_file():
            self.skipTest("Original MixedWM38 NPZ is not available in this checkout")
        protocol = Protocol(config)
        self.assertEqual({name: len(ids) for name, ids in protocol.ids.items()},
                         {"train": 25193, "valid": 3599, "test": 7199})
        self.assertEqual(len(protocol.retrieval.valid), 500)
        self.assertEqual(protocol.retrieval.valid.candidates_per_query, 3598)
        self.assertEqual(len(protocol.queries), 7199)
        self.assertEqual(protocol.candidates.candidates_per_query, 7198)


if __name__ == "__main__":
    unittest.main()
