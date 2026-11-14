"""Fusion endpoint parity, full-gallery scoring and frozen-valid selection."""
import hashlib
import json
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_f_segmentation as legacy
from test_f_segmentation_geometry import snapshot
from experiments.proposed._shared import ef_suite, ef_training
from experiments.proposed.f_segmentation import run as train_run
from experiments.proposed.f_segmentation_geometry import pipeline as geo
from experiments.proposed.f_global_fusion import pipeline as fusion, run
from wbm_retrieval.common import csv_rows, digest, file_hash, read_json, write_json
from wbm_retrieval.data import Protocol, source_fingerprint
from wbm_retrieval.retrieval_protocol import FullGalleryPool


def sign_global(payload):
    value = dict(payload)
    value["signature"] = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                                   allow_nan=False, separators=(",", ":")).encode()).hexdigest()
    return value


def global_fixture(root, split, pool, sources, checkpoint="frozen_global"):
    root.mkdir(parents=True)
    prepared = root / split / "prepared"
    prepared.mkdir(parents=True)
    rng = np.random.default_rng(42)
    ids = np.asarray(pool.gallery)[::-1].copy()  # Alignment must use sample IDs, not row indices.
    features = rng.random((len(ids), 512)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    np.savez(prepared / "features.npz", sample_ids=ids, features=features)
    write_json(prepared / "prepared_metadata.json", {"checkpoint_sha256": checkpoint,
                                                    "similarity": "clip(h512 cosine,0,1)"})
    expected = {"data_file": sources["data_file"], "split_file": sources["split_manifest"],
                "retrieval_protocol": {k.split("/", 1)[1]: v for k, v in sources.items()
                                       if k.startswith("retrieval_protocol/")}}
    manifest = sign_global({"method": "wapirl", "variant": "global", "split": split, "smoke": False,
                            "settings": {"selection": "minimum_valid_nce"}, "code": {"encoder": "fixture"},
                            "candidate_rule": "all_gallery_except_query", "derived_protocol": None,
                            "sources": expected, "query_count": len(pool), "gallery_count": len(pool.gallery),
                            "candidates_per_query": pool.candidates_per_query,
                            "training_source": {"files": {"best_nce.pt": checkpoint}, "training_complete": True,
                                                "smoke_training": False, "selection": "minimum_valid_nce",
                                                "sources": expected}})
    write_json(root / "evaluation_manifest.json", manifest)
    write_json(root / "completed.json", {"evaluation_signature": manifest["signature"], "split": split,
                                        "smoke": False, "files": {str(p.relative_to(root)): file_hash(p)
                                        for p in prepared.iterdir()}})
    return ids, features


class FusionScoringTests(unittest.TestCase):
    def test_empty_query_and_multiregion_match_and_global_id_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = np.array([10, 20, 30])
            np.save(root / "h512.npy", np.array([[1., 0.], [0., 1.], [1., 0.]]))
            np.savez(root / "regions.npz", map_ids=ids, offsets=[0, 0, 2, 3],
                     area=[9, 16, 9], relative_area=[.09, .16, .09], histograms=np.ones((3, 16))/16)
            matching = {"lambda_pos": 0, "lambda_scale": 0, "weight": "sqrt_area",
                        "shape_threshold": 0, "score_threshold": 0}
            scorer = fusion.BranchScores(root, ids[::-1], np.array([[1., 0.], [.6, .8], [0., 1.]]), ids, matching)
            local, whole = scorer.score(20, np.array([10, 30]))
            np.testing.assert_allclose(local, [0, 3/7])
            np.testing.assert_allclose(whole, [.8, .6])
            local, whole = scorer.score(10, np.array([20, 30]))
            np.testing.assert_array_equal(local, [0, 0])
            np.testing.assert_allclose(whole, [.8, 0])

    def test_full_pool_fusion_can_promote_candidate_outside_both_top1(self):
        class Scores:
            def score(self, q, candidates):
                return np.array([1., 0., .7]), np.array([0., 1., .7])
        labels = np.array([[1, 0], [0, 1], [0, 1], [1, 1]])
        pool = FullGalleryPool([0], np.arange(4), "valid", "fixture")
        protocol = type("Protocol", (), {"ids": {"valid": np.arange(4), "test": np.arange(4)}})()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fusion.evaluate(root, "valid", [0., .5, 1.], "sig", Scores(), pool,
                            fusion.bundle_for(protocol, labels), [1], 2026)
            best = lambda a: int(csv_rows(root / "valid" / fusion.alpha_name(a) / "rankings.csv")[0]["candidate_id"])
            self.assertEqual([best(a) for a in (0, .5, 1)], [1, 3, 2])
            exact = read_json(root / "valid/alpha_0.5/metrics.json")
            coverage = read_json(root / "valid/alpha_0.5/coverage_metrics.json")
            self.assertEqual(exact["metrics"]["Precision@1"], 0)
            self.assertEqual(coverage["metrics"]["FullCoveragePrecision@1"], 1)
            selected = fusion.select(root, [0., .5, 1.], "sig", "Precision@1")
            self.assertEqual(selected["alpha"], 0.)  # Deterministic tie -> more F.
            selected = fusion.select(root, [0., .5, 1.], "sig", "FullCoveragePrecision@1", "coverage")
            self.assertEqual(selected["alpha"], .5)
            before = snapshot(root)
            with patch.object(Scores, "score", side_effect=AssertionError("should reuse")):
                fusion.evaluate(root, "valid", [0., .5, 1.], "sig", Scores(), pool,
                                fusion.bundle_for(protocol, labels), [1], 2026)
            self.assertEqual(snapshot(root), before)
            with self.assertRaises(ValueError):
                fusion.evaluate(root, "valid", [0.], "changed", Scores(), pool,
                                fusion.bundle_for(protocol, labels), [1], 2026)

    def test_alpha_and_output_validation(self):
        for values in ([0, 1, float("nan")], [0, 1, -1], [0, 1, 1], [.1, .2], []):
            with self.assertRaises(ValueError):
                fusion.validate_alphas(values)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for output in (root, root / "nested", root.parent):
                with self.assertRaises(ValueError):
                    run.separate_output(output, [root])
            protected = root / "keep"
            protected.mkdir()
            (protected / "run.log").write_text("untouched")
            before = snapshot(protected)
            with self.assertRaisesRegex(ValueError, "empty"):
                run.main(["--output-root", str(protected)])
            self.assertEqual(snapshot(protected), before)
            with self.assertRaisesRegex(ValueError, "stage valid"):
                run.main(["--stage", "test", "--output-root", str(root / "missing")])
            self.assertFalse((root / "missing").exists())

    def test_global_cache_hash_protocol_model_and_ids_are_checked(self):
        pool = FullGalleryPool([5], [5, 8, 10], "valid", "fixture")
        sources = {"data_file": "data", "split_manifest": "split", "retrieval_protocol/metadata.json": "meta"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "global"
            ids, features = global_fixture(root, "valid", pool, sources)
            actual_ids, actual, _ = fusion.load_global(root, "valid", pool, sources)
            np.testing.assert_array_equal(actual_ids, ids)
            np.testing.assert_array_equal(actual, features)
            with self.assertRaisesRegex(ValueError, "hashes"):
                fusion.load_global(root, "valid", pool, {**sources, "data_file": "other"})
            with self.assertRaisesRegex(ValueError, "same completed"):
                fusion.load_global(root, "valid", pool, sources, "other_model")
            with self.assertRaisesRegex(ValueError, "inference"):
                fusion.load_global(root, "valid", pool, sources, expected_recipe="changed")
            with self.assertRaises(ValueError):
                fusion.validate_features(ids, features*2, pool.gallery)
            with self.assertRaises(ValueError):
                fusion.validate_features(np.array([5, 5, 8]), features, pool.gallery)
            with (root / "valid/prepared/features.npz").open("ab") as handle:
                handle.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "modified"):
                fusion.load_global(root, "valid", pool, sources)


class FusionPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_valid_selection_test_controls_source_readonly_and_restart(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(ef_training, "build_model", legacy.tiny_model), \
                patch.object(ef_suite, "build_model", legacy.tiny_model):
            root = Path(tmp)
            legacy.PipelineTests().make_config(root)
            train_run.main(["--config", str(root / "suite.json"), "--methods", "cc8", "--stage", "train",
                            "--device", "cpu", "--num-workers", "0"])
            config, _, _, _, _ = geo.load_source(root / "runs", "cc8", device="cpu")
            protocol, sources = Protocol(config), source_fingerprint(config)
            global_fixture(root / "global_valid", "valid", protocol.retrieval.valid, sources)
            global_fixture(root / "global_test", "test", protocol.retrieval.test, sources)
            protected = {name: snapshot(root / name) for name in ("runs", "global_valid", "global_test", "deduplicated", "retrieval")}
            args = ["--source-root", str(root / "runs"), "--f-feature-root", str(root / "geometry"),
                    "--wapirl-valid-root", str(root / "global_valid"), "--wapirl-test-root", str(root / "global_test"),
                    "--output-root", str(root / "fusion"), "--alphas", "0", "0.3", "1"]
            run.main(args)
            output = root / "fusion"
            self.assertFalse((output / "test").exists())
            selection = read_json(output / "selection.json")
            before_selection = snapshot(output)["selection.json"]
            valid_before = snapshot(output / "valid")
            run.main(args + ["--stage", "test"])
            weights = sorted({0., 1., selection["alpha"]})
            self.assertEqual(sorted(p.name for p in (output / "test").iterdir()), sorted(fusion.alpha_name(a) for a in weights))
            # alpha=0 reproduces the existing geometric evaluator, not a reimplementation.
            cfg, suite, checkpoint, kind, inputs = geo.load_source(root / "runs", "cc8", device="cpu")
            geometry_signature = digest(geo.new_identity(cfg, inputs))
            with np.load(cfg["data_file"]) as data:
                labels = data["arr_1"]
            for split in ("valid", "test"):
                geo.evaluate_split(root / "reference", cfg, protocol, labels,
                                   output / "f_cache/features" / split, split, kind, checkpoint["epoch"], geometry_signature)
                reference = csv_rows(root / "reference" / split / kind / "rankings.csv")
                actual = csv_rows(output / split / "alpha_0.0/rankings.csv")
                self.assertEqual([(r["query_id"], r["candidate_id"], r["similarity_score"]) for r in reference],
                                 [(r["query_id"], r["candidate_id"], r["similarity_score"]) for r in actual])
                # alpha=1 uses the exact original whole-wafer score and SHA tie policy.
                ids, features, _ = fusion.load_global(root / f"global_{split}", split, getattr(protocol.retrieval, split), sources)
                rows = {int(s): i for i, s in enumerate(ids)}
                for row in csv_rows(output / split / "alpha_1.0/rankings.csv"):
                    q, c = int(row["query_id"]), int(row["candidate_id"])
                    expected = float(np.clip(features[rows[c]] @ features[rows[q]], 0, 1))
                    self.assertAlmostEqual(float(row["similarity_score"]), expected, places=6)
            run.main(args + ["--stage", "all"])
            self.assertEqual(snapshot(output / "valid"), valid_before)
            self.assertEqual(snapshot(output)["selection.json"], before_selection)
            self.assertEqual(protected, {name: snapshot(root / name) for name in protected})
            self.assertFalse((root / "geometry").exists())
            # A relocated complete geometry cache is reused without writing to it.
            shutil.copytree(output / "f_cache/features", root / "geometry/cc8/features")
            cache_before = snapshot(root / "geometry")
            with patch.object(geo, "build_features", side_effect=AssertionError("must reuse")):
                run.main(args + ["--stage", "all"])
            self.assertEqual(snapshot(root / "geometry"), cache_before)
            with self.assertRaisesRegex(ValueError, "changed"):
                run.main(args + ["--selection-metric", "mAP@10"])
            bad_selection = read_json(output / "selection.json")
            bad_selection["alpha"] = .99
            write_json(output / "selection.json", bad_selection)
            with self.assertRaisesRegex(ValueError, "selection changed"):
                run.main(args + ["--stage", "test"])

    def test_interrupted_metrics_resume_in_new_attempt(self):
        class Scores:
            def score(self, q, candidates):
                return np.array([.7, .5]), np.array([.4, .8])
        labels = np.array([[1, 0], [1, 0], [1, 1]])
        pool = FullGalleryPool([0], np.arange(3), "valid", "fixture")
        protocol = type("Protocol", (), {"ids": {"valid": np.arange(3), "test": np.arange(3)}})()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            bundle = fusion.bundle_for(protocol, labels)
            with patch.object(geo, "evaluate_rankings_file", side_effect=InterruptedError("after exact")):
                with self.assertRaises(InterruptedError):
                    fusion.evaluate(output, "valid", [0., 1.], "sig", Scores(), pool, bundle, [1], 2026)
            attempts = list((output / "valid").glob(".*attempt_*"))
            self.assertEqual(len(attempts), 2)
            before = {str(p): snapshot(p) for p in attempts}
            fusion.evaluate(output, "valid", [0., 1.], "sig", Scores(), pool, bundle, [1], 2026)
            self.assertEqual(before, {str(p): snapshot(p) for p in attempts})
            self.assertTrue(geo.completed(output / "valid/alpha_0.0", "sig"))
            self.assertTrue(geo.completed(output / "valid/alpha_1.0", "sig"))


if __name__ == "__main__":
    unittest.main()
