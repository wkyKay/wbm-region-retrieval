"""Synthetic source provenance, original IDs, and forbidden-test-access guards."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from f_region_score_search import inputs
from f_region_structure import scoring, selection, state
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from wbm_retrieval.retrieval_protocol import FullGalleryPool


def reseal(folder, signature, files):
    write_json(Path(folder)/"completed.json", {"status": "completed", "signature": signature,
               "files": {name: file_hash(Path(folder)/name) for name in sorted(files)}})


def fixture(root):
    root = Path(root).resolve()
    source_run, original = root/"old_run", root/"old_f"
    model = original/"cc8/seed_2026/F/best_nce.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"synthetic model; the mature source loader is mocked")
    source_run.mkdir()
    ids = np.arange(4, 7202, 2, dtype=np.int64)
    labels = np.zeros((7201, 8), dtype=np.uint8)
    labels[ids, 0] = 1
    data = root/"raw.npz"
    np.savez_compressed(data, arr_0=np.zeros((1, 1), dtype=np.uint8), arr_1=labels)
    dedup, retrieval = root/"deduplicated", root/"retrieval"
    dedup.mkdir(); retrieval.mkdir()
    for name in ("split.csv", "metadata.json", "duplicates.csv"):
        (dedup/name).write_text("synthetic frozen protocol\n")
    for name in ("metadata.json", "valid_queries.csv", "valid_gallery.csv", "test_queries.csv", "test_gallery.csv"):
        (retrieval/name).write_text("synthetic frozen ID list, never a test feature/report\n")
    config = {"data_file": str(data), "split_manifest": str(dedup/"split.csv"),
              "deduplication_metadata": str(dedup/"metadata.json"), "retrieval_protocol_dir": str(retrieval)}
    fingerprint = {key: file_hash(config[key]) for key in ("data_file", "split_manifest", "deduplication_metadata")}
    fingerprint["duplicates_manifest"] = file_hash(dedup/"duplicates.csv")
    fingerprint.update({"retrieval_protocol/"+p.name: file_hash(p) for p in retrieval.iterdir()})
    fresh = {"source_files": {"seed_2026/F/best_nce.pt": file_hash(model)},
             "source_fingerprint": fingerprint, "geometry": {"fixture": True}, "checkpoint": "best_nce", "seed": 2026}
    protocol = SimpleNamespace(ids={"train": [0], "valid": ids, "test": [1]},
        retrieval=SimpleNamespace(valid=FullGalleryPool(ids[:500], ids, "valid", "synthetic_protocol")))
    context = {"source": fresh, "fixture_only": True}
    old_signature = digest(context)
    feature_root = original/"frozen_features"
    upstream = feature_root/"valid/h512.npy"
    upstream.parent.mkdir(parents=True)
    np.save(upstream, np.ones((1, 1), dtype=np.float32))
    manifest = {"version": "f_region_structure_v1", "context": context,
                "source_root": str(original), "feature_roots": [str(feature_root)]}
    write_json(source_run/"run_manifest.json", state.signed(manifest))
    grid = scoring.frozen_grid()
    evaluation_signature = digest({"context": old_signature, "split": "valid", "variant": inputs.WINNER,
                                    "spec": inputs.WINNER_SPEC})
    baseline = source_run/"valid"/inputs.WINNER
    baseline.mkdir(parents=True)
    meta = {"signature": evaluation_signature, "split": "valid", "variant": inputs.WINNER,
            "spec": inputs.WINNER_SPEC, "ks": inputs.KS, "tie_seed": 2026,
            "retrieval_protocol_signature": "synthetic_protocol"}
    exact = {"Precision@10": .4988}
    coverage = {"Coverage-nDCG@10": .84}
    for name in inputs.EVALUATION_FILES - {"verification.json"}:
        write_json(baseline/name, {"fixture_only": True})
    write_json(baseline/"evaluation_manifest.json", meta)
    for name, family, metrics in (("metrics.json", "exact", exact), ("coverage_metrics.json", "coverage", coverage)):
        write_json(baseline/name, {"split": "valid", "num_queries": 500, "ks": inputs.KS,
                   "metric_version": family+"_v1", "metadata": {"signature": evaluation_signature}, "metrics": metrics})
    evidence = {"status": "verified", "signature": evaluation_signature, "split": "valid", "variant": inputs.WINNER,
                "ks": inputs.KS, "tie_seed": 2026, "num_queries": 500, "ranking_rows": 250000,
                "full_candidates_checked": 1799000, "all_queries_full_candidate_topk_exact": True,
                "max_metric_absolute_error": 0., "max_saved_score_absolute_error": 0., "top10_hits": 2494,
                "ranking_sha256": file_hash(baseline/"rankings.csv"),
                "files": {name: file_hash(baseline/name) for name in inputs.EVALUATION_FILES - {"verification.json"}}}
    write_json(baseline/"verification.json", evidence)
    reseal(baseline, evaluation_signature, inputs.EVALUATION_FILES)
    results = {name: {"top10_hits": 2000, "num_queries": 500} for name in grid}
    results[inputs.WINNER] = {"top10_hits": 2494, "num_queries": 500, "metrics": exact, "coverage_metrics": coverage,
                             "completed_sha256": file_hash(baseline/"completed.json")}
    index = {"split": "valid", "context_signature": old_signature, "specs": grid, "results": results}
    write_json(source_run/"evaluation_index_valid.json", index)
    selected = {**selection.select_best(results), "context_signature": old_signature,
                "valid_index_sha256": file_hash(source_run/"evaluation_index_valid.json")}
    write_json(source_run/"selection.json", state.signed(selected))
    cache_folder = source_run/"cache/valid"
    cache_folder.mkdir(parents=True)
    counts = np.zeros(3599, dtype=np.int64)
    counts[[0, 1, 3]] = [1, 2, 1]
    offsets = np.r_[0, np.cumsum(counts)]
    joint = np.full((4, 128), 1/128)
    radial = np.full((4, 16), 1/16)
    arrays = {"map_ids": ids, "offsets": offsets, "area": np.full(4, 5, dtype=np.int64),
              "original_count": np.full(4, 5, dtype=np.int64), "retained_count": np.full(4, 5, dtype=np.int64),
              "owners": np.repeat(ids, counts), "histograms": radial, "radial_hist16": radial,
              "relative_area": np.full(4, .05), "joint": joint, "joint_all": joint,
              "shape": np.full((4, 16), 1/16), "shape_all": np.full((4, 16), 1/16),
              "quality": np.ones(4), "empty": np.zeros(4, dtype=bool), "empty_all": np.zeros(4, dtype=bool)}
    np.savez_compressed(cache_folder/"regions.npz", **arrays)
    h = np.zeros((4, 512), dtype=np.float32); h[:, 0] = 1
    np.save(cache_folder/"h512.npy", h)
    write_json(cache_folder/"source_binding.json", {str(upstream): file_hash(upstream)})
    write_json(cache_folder/"audit.json", {"fixture_only": True})
    cache_signature = digest({"context": old_signature, "split": "valid", "cache_version": 1})
    reseal(cache_folder, cache_signature, inputs.CACHE_FILES)
    # Any recursive verification of the old root/test would hit these sentinels.
    (source_run/"completed.json").write_text("DO NOT READ OR VERIFY OLD ROOT")
    for forbidden in (source_run/"test/secret.json", source_run/"cache/test/secret.npy"):
        forbidden.parent.mkdir(parents=True, exist_ok=True)
        forbidden.write_text("FORBIDDEN TEST ARTIFACT")
    return {"run": source_run, "source_root": original, "config": config, "protocol": protocol,
            "fresh": fresh, "old_signature": old_signature, "cache_signature": cache_signature,
            "baseline_signature": evaluation_signature, "cache_folder": cache_folder,
            "baseline": baseline, "arrays": arrays, "upstream": upstream, "selected": selected}


class InputTests(unittest.TestCase):
    def load(self, data, **kwargs):
        returned = (data["config"], {}, {}, data["protocol"], kwargs.pop("fresh", data["fresh"]))
        with patch.object(inputs.old_sources, "load", return_value=returned) as loader:
            result = inputs.load(data["run"])
            loader.assert_called_once_with(data["source_root"])
            return result

    def test_read_only_valid_source_and_complete_dependency_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = fixture(tmp)
            before = {str(p): (file_hash(p), p.stat().st_mtime_ns) for p in Path(tmp).rglob("*") if p.is_file()}
            real_hash, real_load = inputs.file_hash, np.load
            touched = []
            def guarded_hash(path):
                path = Path(path)
                self.assertFalse(path.is_relative_to(data["run"]/"test"))
                self.assertFalse(path.is_relative_to(data["run"]/"cache/test"))
                self.assertNotEqual(path, data["run"]/"completed.json")
                touched.append(path)
                return real_hash(path)
            class LabelsOnly:
                def __init__(self, archive): self.archive = archive
                def __enter__(self): return self
                def __exit__(self, *args): self.archive.close()
                def __getitem__(self, key):
                    if key != "arr_1": raise AssertionError("new loader extracted raw maps")
                    return self.archive[key]
            def guarded_load(path, *args, **kwargs):
                result = real_load(path, *args, **kwargs)
                return LabelsOnly(result) if Path(path) == Path(data["config"]["data_file"]) else result
            with patch.object(inputs, "file_hash", side_effect=guarded_hash), patch.object(inputs.np, "load", side_effect=guarded_load):
                result = self.load(data)
            self.assertEqual(set(result), {"config", "protocol", "bundle", "cache", "source_binding", "old_context_signature", "baseline_folder"})
            self.assertEqual(result["baseline_folder"], data["baseline"])
            self.assertEqual(result["old_context_signature"], data["old_signature"])
            np.testing.assert_array_equal(result["cache"]["owners"], [4, 6, 6, 10])
            for name in ("run_manifest.json", "selection.json", "evaluation_index_valid.json"):
                self.assertIn(str(data["run"]/name), result["source_binding"])
            self.assertIn(str(data["upstream"]), result["source_binding"])
            self.assertIn(str(data["config"]["data_file"]), result["source_binding"])
            self.assertEqual(set(result["source_binding"]), set(map(str, touched)))
            after = {str(p): (file_hash(p), p.stat().st_mtime_ns) for p in Path(tmp).rglob("*") if p.is_file()}
            self.assertEqual(before, after)

    def test_selection_and_fresh_model_identity_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = fixture(tmp)
            changed = deepcopy(data["fresh"]); changed["checkpoint"] = "last"
            with self.assertRaisesRegex(ValueError, "original model/data/protocol"):
                self.load(data, fresh=changed)
            changed = deepcopy(data["selected"]); changed["selected_spec"]["strength"] = 2.
            write_json(data["run"]/"selection.json", state.signed(changed))
            with self.assertRaisesRegex(ValueError, "frozen real"):
                self.load(data)

    def test_index_and_completed_cache_corruption_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = fixture(tmp)
            file = data["run"]/"evaluation_index_valid.json"
            file.write_text(file.read_text()+"\n")
            with self.assertRaisesRegex(ValueError, "index binding"):
                self.load(data)
        with tempfile.TemporaryDirectory() as tmp:
            data = fixture(tmp)
            with (data["cache_folder"]/"h512.npy").open("ab") as handle: handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                self.load(data)

    def test_test_dependency_rejected_without_reading_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = fixture(tmp)
            forbidden = data["run"]/"cache/test/secret.npy"
            write_json(data["cache_folder"]/"source_binding.json", {str(forbidden): "0"*64})
            reseal(data["cache_folder"], data["cache_signature"], inputs.CACHE_FILES)
            real = inputs.file_hash
            def guard(path):
                if Path(path) == forbidden: raise AssertionError("forbidden content was read")
                return real(path)
            with patch.object(inputs, "file_hash", side_effect=guard):
                with self.assertRaisesRegex(ValueError, "test access is forbidden"):
                    self.load(data)

    def test_resealed_invalid_region_order_and_counts_are_rejected(self):
        for mutation in ("owners", "offsets", "joint_all", "area", "map_ids"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                data = fixture(tmp)
                arrays = deepcopy(data["arrays"])
                if mutation == "owners": arrays[mutation][0] = 999
                elif mutation == "offsets": arrays[mutation][1] = 9
                elif mutation == "joint_all": arrays[mutation][0] = 0
                elif mutation == "area": arrays[mutation][0] = 4
                else: arrays[mutation][[0, 1]] = arrays[mutation][[1, 0]]
                np.savez_compressed(data["cache_folder"]/"regions.npz", **arrays)
                reseal(data["cache_folder"], data["cache_signature"], inputs.CACHE_FILES)
                with self.assertRaises(ValueError): self.load(data)

    def test_upstream_feature_and_original_data_hash_changes_are_rejected(self):
        for location in ("upstream", "data"):
            with self.subTest(location=location), tempfile.TemporaryDirectory() as tmp:
                data = fixture(tmp)
                path = data["upstream"] if location == "upstream" else Path(data["config"]["data_file"])
                with path.open("ab") as handle: handle.write(b"changed")
                with self.assertRaisesRegex(ValueError, "hash changed"): self.load(data)

    def test_missing_required_completion_file_and_wrong_query_count_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = fixture(tmp)
            marker = read_json(data["cache_folder"]/"completed.json")
            del marker["files"]["audit.json"]
            write_json(data["cache_folder"]/"completed.json", marker)
            with self.assertRaisesRegex(ValueError, "completion evidence"): self.load(data)
        with tempfile.TemporaryDirectory() as tmp:
            data = fixture(tmp)
            pool = data["protocol"].retrieval.valid
            data["protocol"].retrieval.valid = FullGalleryPool(pool.queries[:-1], pool.gallery, "valid", pool.protocol_signature)
            with self.assertRaisesRegex(ValueError, "500 queries"): self.load(data)


if __name__ == "__main__":
    unittest.main()
