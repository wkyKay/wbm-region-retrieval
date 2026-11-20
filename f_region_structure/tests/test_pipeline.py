"""Small synthetic integration, never formal model fitting or evaluation."""
from contextlib import redirect_stdout
import copy
import csv
from io import StringIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.proposed.f_segmentation_geometry.scoring import geometry as original_geometry
from experiments.proposed.f_segmentation_geometry.scoring import pair_scores as old_pairs
from f_region_structure import pipeline, scoring, selection, sources, state
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from wbm_retrieval.diagnostics import batched_map_scores
from wbm_retrieval.retrieval_protocol import FullGalleryPool


MATCHING = {"lambda_pos": 0., "lambda_scale": 0., "shape_threshold": 0.,
            "score_threshold": 0., "weight": "sqrt_area"}


def snapshot(folder):
    return {str(p.relative_to(folder)): (file_hash(p), p.stat().st_mtime_ns)
            for p in Path(folder).rglob("*") if p.is_file()}


def fixture(split="valid"):
    rng = np.random.default_rng(922)
    ids = np.arange(7, 211, 2, dtype=np.int64)
    counts = np.tile([0, 1, 2, 4, 1, 3], 17)
    offsets = np.r_[0, np.cumsum(counts)]
    n = offsets[-1]
    h = rng.uniform(.01, 1., (n, 512)).astype(np.float32)
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    radial = rng.random((n, 16))
    radial /= radial.sum(1, keepdims=True)
    joint = rng.random((n, 128))
    joint /= joint.sum(1, keepdims=True)
    retained = joint.copy()
    retained[::7] = 0
    area = rng.integers(5, 100, n)
    cache = {"map_ids": ids, "offsets": offsets, "area": area, "features": h,
             "histograms": radial, "relative_area": area/1000., "joint": retained,
             "joint_all": joint, "owners": np.repeat(ids, counts)}
    labels = np.zeros((ids[-1]+1, 2), dtype=np.int64)
    labels[ids] = np.tile([[1, 0], [0, 1], [1, 1]], (34, 1))
    protocol = SimpleNamespace(ids={"valid": ids, "test": ids})
    bundle = pipeline.bundle_for(labels, protocol)
    pool = FullGalleryPool(ids[:3], ids, split, "synthetic_full_gallery")
    learned = {}
    for name in ("plain_mlp", "gated_mlp"):
        z = rng.normal(size=(n, 128)).astype(np.float32)
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        learned[name] = z
    return cache, pool, bundle, learned


def write_source(root, split="valid"):
    """Three original-ID wafers, one with no eligible connected components."""
    ids = np.array([2, 5, 7], dtype=np.int64)
    raw = np.ones((8, 9, 9), dtype=np.uint8)
    masks = np.zeros((3, 9, 9), dtype=np.uint8)
    raw[2, 2, 1:6] = 2
    masks[0, 2, 1:6] = 1
    raw[5, 4, 4] = 2
    raw[7, 2, 1:6] = 2
    raw[7, 6, 2:7] = 2
    masks[2, 2, 1:6] = 1
    masks[2, 6, 2:7] = 2
    radial, relative, area = [], [], []
    for i, sid in enumerate(ids):
        for token in range(1, int(masks[i].max())+1):
            mask = masks[i] == token
            hist, fraction = original_geometry(mask, (raw[sid] == 1) | (raw[sid] == 2))
            radial.append(hist)
            relative.append(fraction)
            area.append(int(mask.sum()))
    meta = {"map_ids": ids, "offsets": np.array([0, 1, 1, 3]), "area": np.array(area),
            "histograms": np.array(radial), "relative_area": np.array(relative)}
    h = np.zeros((3, 512), dtype=np.float32)
    h[:, 0] = 1
    folder = Path(root)/split
    folder.mkdir(parents=True)
    np.savez_compressed(folder/"regions.npz", **meta)
    np.save(folder/"region_labels.npy", masks)
    np.save(folder/"h512.npy", h)
    geometry = {"fixture": "original_raw_masks"}
    names = ["regions.npz", "region_labels.npy", "h512.npy"]
    write_json(folder/"completed.json", {"signature": digest(geometry),
               "files": {name: file_hash(folder/name) for name in names}})
    return ids, raw, masks, meta, h, geometry


class ProviderTests(unittest.TestCase):
    def test_baseline_replays_source_pair_and_map_scores_bitwise(self):
        cache, pool, _, learned = fixture()
        provider = pipeline.Provider(cache, pool, MATCHING, learned)
        for query in pool:
            qt = provider.tokens(query)
            pair = old_pairs(cache["features"][qt], cache["features"], cache["histograms"][qt],
                             cache["histograms"], cache["relative_area"][qt], cache["relative_area"])
            candidates = pool[query]
            padded = provider.padded[[provider.positions[int(c)] for c in candidates]]
            expected = batched_map_scores(pair, padded, cache["area"][qt], MATCHING)
            got_ids, actual = provider.score(query, scoring.frozen_grid()["baseline"])
            np.testing.assert_array_equal(got_ids, candidates)
            np.testing.assert_array_equal(actual, expected)
        first = int(cache["map_ids"][0])
        self.assertEqual(len(provider.tokens(first)), 0)
        np.testing.assert_array_equal(provider.score(first, scoring.frozen_grid()["baseline"])[1], np.zeros(101))

    def test_scalar_greedy_is_query_weighted_and_is_not_optimal_assignment(self):
        pair = np.array([[.9, .8], [.85, .1]])
        self.assertAlmostEqual(pipeline._scalar_greedy(pair, [1, 4]), (.9+.1*2)/3)
        self.assertNotAlmostEqual(pipeline._scalar_greedy(pair, [1, 4]), (.8+.85*2)/3)
        self.assertEqual(pipeline._scalar_greedy(np.zeros((2, 0)), [1, 4]), 0.)
        self.assertEqual(pipeline._scalar_greedy(np.empty((0, 3)), []), 0.)

    def test_rejects_incomplete_gallery(self):
        cache, pool, _, learned = fixture()
        bad = copy.deepcopy(cache)
        bad["map_ids"][0] = 999
        with self.assertRaisesRegex(ValueError, "full fixed gallery"):
            pipeline.Provider(bad, pool, MATCHING, learned)


class IntegrationTests(unittest.TestCase):
    def test_interrupt_after_reports_preserves_attempt_and_resumes(self):
        cache, pool, bundle, learned = fixture()
        grid = scoring.frozen_grid()
        with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline, "KS", [1, 5, 10, 100]), redirect_stdout(StringIO()):
            output = Path(tmp)/"interrupted_run"
            with patch.object(pipeline.verify, "verify_evaluation", side_effect=InterruptedError("after reports")):
                with self.assertRaises(InterruptedError):
                    pipeline.evaluate(output, cache, pool, bundle, MATCHING, "fixture_context", grid, learned)
            identities = snapshot(output/"identities")
            unfinished = [(file_hash(p), p.stat().st_mtime_ns) for p in output.rglob("metrics.json")]
            self.assertTrue(unfinished)
            result = pipeline.evaluate(output, cache, pool, bundle, MATCHING, "fixture_context", grid, learned)
            self.assertEqual(set(result), set(grid))
            self.assertEqual(identities, snapshot(output/"identities"))
            after = [(file_hash(p), p.stat().st_mtime_ns) for p in output.rglob("metrics.json")]
            self.assertTrue(all(item in after for item in unfinished), "unfinished reports must remain preserved")
            self.assertTrue(all(state.verified(output/"valid"/name) for name in grid))

    def test_complete_grid_valid_selection_then_single_test_and_immutable_resume(self):
        cache, pool, bundle, learned = fixture()
        grid = scoring.frozen_grid()
        with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline, "KS", [1, 5, 10, 100]), redirect_stdout(StringIO()):
            output = Path(tmp)/"new_run"
            result = pipeline.evaluate(output, cache, pool, bundle, MATCHING, "fixture_context", grid, learned)
            self.assertEqual(set(result), set(grid))
            selected = selection.select_best(result, expected_queries=len(pool))
            self.assertTrue(selected["synthetic"])
            for name, spec in grid.items():
                folder = output/"valid"/name
                self.assertTrue(state.verified(folder))
                checked = read_json(folder/"verification.json")
                self.assertEqual(checked["status"], "verified")
                self.assertEqual(checked["num_queries"], 3)
                self.assertEqual(read_json(folder/"metrics.json")["num_queries"], 3)
                with (folder/"rankings.csv").open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 300)
                self.assertTrue(all(row["query_id"] != row["candidate_id"] for row in rows))
            before = snapshot(output)
            with patch.object(pipeline.Provider, "parts", side_effect=AssertionError("sealed rankings must not be rerun")):
                again = pipeline.evaluate(output, cache, pool, bundle, MATCHING, "fixture_context", grid, learned)
            self.assertEqual(result, again)
            self.assertEqual(before, snapshot(output))
            test_pool = FullGalleryPool(pool.queries, pool.gallery, "test", "synthetic_full_gallery")
            test_specs = {selected["winner"]: selected["selected_spec"]}
            test_result = pipeline.evaluate(output, cache, test_pool, bundle, MATCHING,
                                            "fixture_context", test_specs, learned)
            self.assertEqual(set(test_result), {selected["winner"]})
            self.assertEqual(len(list((output/"test").iterdir())), 1)
            with self.assertRaises(ValueError):
                selection.select_best(test_result, split="test", expected_queries=3)
            corrupt = output/"valid"/selected["winner"]/"rankings.csv"
            corrupt.write_text(corrupt.read_text()+"\n")
            with self.assertRaisesRegex(ValueError, "Changed or missing"):
                pipeline.evaluate(output, cache, pool, bundle, MATCHING, "fixture_context", grid, learned)

    def test_direct_pipeline_refuses_test_grid(self):
        cache, pool, bundle, learned = fixture("test")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"forbidden"
            with self.assertRaisesRegex(ValueError, "test permits only one"):
                pipeline.evaluate(output, cache, pool, bundle, MATCHING,
                                  "fixture", scoring.frozen_grid(), learned)
            self.assertFalse(output.exists(), "test grid rejection must precede output creation")


class SourceTests(unittest.TestCase):
    def test_source_binding_and_new_cache_preserve_original_ids_and_empty_maps(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            root = Path(tmp)
            old = root/"old_features"
            ids, maps, labels, meta, h, geometry = write_source(old)
            before = snapshot(old)
            protocol = SimpleNamespace(ids={"valid": ids})
            output = root/"new_experiment"
            cache = sources.prepare(output, "valid", maps, {}, {}, {}, protocol,
                                    {"geometry": geometry}, [old], "fixture", workers=0)
            np.testing.assert_array_equal(cache["map_ids"], ids)
            np.testing.assert_array_equal(cache["owners"], [2, 7, 7])
            np.testing.assert_array_equal(cache["offsets"], meta["offsets"])
            np.testing.assert_array_equal(cache["features"], h)
            np.testing.assert_allclose(cache["relative_area"], meta["relative_area"], rtol=0, atol=0)
            audit = read_json(cache["folder"]/"audit.json")
            self.assertEqual(audit["empty_maps"], 1)
            self.assertEqual(before, snapshot(old))
            new_snapshot = snapshot(output)
            with patch.object(sources, "feature_source", side_effect=AssertionError("sealed cache should be reused")):
                again = sources.prepare(output, "valid", maps, {}, {}, {}, protocol,
                                        {"geometry": geometry}, [old], "fixture", workers=0)
            self.assertEqual(new_snapshot, snapshot(output))
            np.testing.assert_array_equal(again["joint"], cache["joint"])
            source_marker = old/"valid"/"completed.json"
            changed = read_json(source_marker)
            changed["extra"] = "concurrent_source_change"
            write_json(source_marker, changed)
            with self.assertRaisesRegex(ValueError, "Frozen source changed"):
                sources.prepare(output, "valid", maps, {}, {}, {}, protocol,
                                {"geometry": geometry}, [old], "fixture", workers=0)

    def test_feature_source_rejects_checkpoint_geometry_signature_and_corrupt_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp)/"old"
            _, _, _, _, _, geometry = write_source(old)
            with self.assertRaises(ValueError):
                sources.feature_source("valid", {}, {"different_checkpoint": True}, [old])
            np.save(old/"valid"/"h512.npy", np.zeros((3, 512), dtype=np.float32))
            with self.assertRaises(ValueError):
                sources.feature_source("valid", {}, geometry, [old])

    def test_train_source_binds_preparation_marker_and_suite_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/"old_training"
            cache = folder/"cache"
            cache.mkdir(parents=True)
            np.savez_compressed(cache/"train_regions.npz", map_ids=np.array([7]), offsets=np.array([0, 1]), area=np.array([5]))
            np.save(cache/"train_region_labels.npy", np.zeros((1, 9, 9), dtype=np.uint8))
            np.save(cache/"train_F.npy", np.zeros((1, 3, 8, 8), dtype=np.uint8))
            names = ["cache/train_regions.npz", "cache/train_region_labels.npy", "cache/train_F.npy"]
            write_json(folder/"suite_manifest.json", {"signature": "original_training"})
            write_json(cache/"train_prepared.json", {"signature": "original_training",
                       "files": {name: file_hash(folder/name) for name in names}})
            *_, binding = sources.feature_source("train", {"output_dir": str(folder)}, {}, [])
            for p in (cache/"train_prepared.json", folder/"suite_manifest.json"):
                self.assertEqual(binding[str(p.resolve())], file_hash(p))
            marker = read_json(cache/"train_prepared.json")
            marker["signature"] = "different_training"
            write_json(cache/"train_prepared.json", marker)
            with self.assertRaisesRegex(ValueError, "different source signature"):
                sources.feature_source("train", {"output_dir": str(folder)}, {}, [])

    def test_frozen_CNN_views_use_checkpoint_and_rotation_without_updates(self):
        torch.set_num_threads(1)
        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(27, 512, bias=False)
            def forward(self, x):
                return self.linear(x.reshape(len(x), -1))
        model = torch.nn.ModuleDict({"encoder": TinyEncoder(), "projector": torch.nn.Identity()})
        with torch.no_grad():
            model["encoder"].linear.weight.copy_(torch.arange(512*27).reshape(512, 27)/10000.)
        original = {k: v.clone() for k, v in model.state_dict().items()}
        checkpoint = {"model": original}
        created = []
        def build(*_):
            fresh = copy.deepcopy(model)
            created.append(fresh)
            return fresh
        with tempfile.TemporaryDirectory() as tmp, patch.object(sources.ef_suite, "build_model", side_effect=build), redirect_stdout(StringIO()):
            folder = Path(tmp)
            patches = np.arange(2*27, dtype=np.float32).reshape(2, 3, 3, 3)
            np.save(folder/"patches.npy", patches)
            before = file_hash(folder/"patches.npy")
            sources.encode_train_views(folder/"patches.npy", {}, {"variants": {"F": {}}}, checkpoint, folder, "cpu")
            h = np.load(folder/"h_views.npy")
            self.assertEqual(h.shape, (2, 2, 512))
            with torch.no_grad():
                x = torch.from_numpy(patches)
                for view in range(2):
                    expected = torch.nn.functional.normalize(model["encoder"](torch.rot90(x, view, dims=(-2, -1))), dim=1).numpy()
                    np.testing.assert_array_equal(h[:, view], expected)
            self.assertEqual(before, file_hash(folder/"patches.npy"))
            self.assertFalse(created[0].training)
            self.assertTrue(all(not p.requires_grad for p in created[0].parameters()))
            for key in original:
                torch.testing.assert_close(original[key], model.state_dict()[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
