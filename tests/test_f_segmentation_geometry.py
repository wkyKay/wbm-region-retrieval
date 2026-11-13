"""Old-model compatibility, immutable outputs, scoring and dual metric checks."""
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import test_f_segmentation as legacy
from experiments.proposed._shared import ef_suite, ef_training
from experiments.proposed.f_segmentation import run as training_run
from experiments.proposed.f_segmentation.proposals import METHODS
from experiments.proposed.f_segmentation_geometry import pipeline, run
from experiments.proposed.f_segmentation_geometry.scoring import geometry, pair_scores
from experiments.proposed.f_segmentation_geometry.coverage_metrics import evaluate_rankings_file, query_metrics
from wbm_retrieval.common import csv_rows, file_hash, read_json, write_json
from wbm_retrieval.data import Protocol
from wbm_retrieval.exact_metrics import evaluate_rankings, order_scores
from wbm_retrieval.retrieval_protocol import FullGalleryPool


def snapshot(root):
    return {str(p.relative_to(root)): (file_hash(p), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


class GeometryScoringTests(unittest.TestCase):
    def test_empty_and_multi_region_maps_remain_in_the_complete_pool(self):
        labels = np.array([[1, 0], [1, 1], [0, 1], [1, 0]], dtype=np.uint8)
        pool = FullGalleryPool(np.arange(4), np.arange(4), "valid", "fixture")
        protocol = SimpleNamespace(ids={"valid": np.arange(4), "test": np.arange(4)},
                                   retrieval=SimpleNamespace(valid=pool))
        config = {"segmentation": {"method": "cc8"}, "evaluation": {"ks": [1, 3], "tie_seed": 2026},
                  "matching": {"lambda_pos": 0., "lambda_scale": 0., "weight": "sqrt_area",
                               "shape_threshold": 0., "score_threshold": 0.}}
        features = np.array([[1., 0], [.6, .8], [0, 1.], [1., 0]])
        hist = np.ones((4, 16))/16
        area = np.array([5, 20, 10, 8])
        relative = area/100
        offsets = [0, 0, 2, 3, 4]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            feature_dir = folder / "features"
            feature_dir.mkdir()
            np.save(feature_dir / "h512.npy", features)
            np.savez(feature_dir / "regions.npz", map_ids=np.arange(4), offsets=offsets,
                     area=area, relative_area=relative, histograms=hist)
            pipeline.evaluate_split(folder, config, protocol, labels, feature_dir, "valid", "best_nce", 1, "fixture")
            rows = csv_rows(folder / "valid/best_nce/rankings.csv")
            self.assertEqual(len(rows), 12)
            for q in pool:
                candidates = pool[q]
                expected_scores = []
                for c in candidates:
                    qt, ct = list(range(offsets[q], offsets[q+1])), list(range(offsets[c], offsets[c+1]))
                    pairs = []
                    for i in qt:
                        for j in ct:
                            value = np.dot(features[i], features[j]) * (.85+.15*np.exp(-.5*abs(np.log(relative[i]/relative[j]))/1.5))
                            pairs.append((value, i, j))
                    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
                    used_q, used_c, score = set(), set(), 0.
                    for value, i, j in pairs:
                        if i not in used_q and j not in used_c and value > 0:
                            score += np.sqrt(area[i])*value
                            used_q.add(i)
                            used_c.add(j)
                    expected_scores.append(score / np.sqrt(area[qt]).sum() if qt else 0.)
                actual = [r for r in rows if int(r["query_id"]) == q]
                expected_order = order_scores(candidates, expected_scores, q, 2026)
                self.assertEqual([int(r["candidate_id"]) for r in actual], candidates[expected_order].tolist())
                np.testing.assert_allclose([float(r["similarity_score"]) for r in actual], np.asarray(expected_scores)[expected_order], atol=1e-14)
            coverage = read_json(folder / "valid/best_nce/coverage_metrics.json")
            self.assertEqual(coverage["num_queries"], 4)
            self.assertEqual(coverage["groups"]["mixed_label"]["num_queries"], 1)

    def test_pair_score_against_independent_scalar_formula(self):
        rng = np.random.default_rng(25)
        features = rng.normal(size=(12, 9))
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        hist = rng.dirichlet(np.ones(16), size=12)
        areas = rng.uniform(.01, .8, 12)
        actual = pair_scores(features[:4], features, hist[:4], hist, areas[:4], areas)
        for i in range(4):
            for j in range(12):
                radial = sum(abs(sum(hist[i, :b+1])-sum(hist[j, :b+1])) for b in range(16))/16
                scale = abs(np.log(np.sqrt(areas[i])/np.sqrt(areas[j])))
                expected = max(0, min(1, float(np.dot(features[i], features[j]))))*np.exp(-radial)*(.85+.15*np.exp(-scale/1.5))
                self.assertAlmostEqual(actual[i, j], expected, places=14)
        np.testing.assert_allclose(np.diag(actual[:, :4]), 1, atol=1e-14)
        self.assertEqual(pair_scores(features[:0], features, hist[:0], hist, areas[:0], areas).shape, (0, 12))

    def test_geometry_matches_radial_definition_and_rotation(self):
        support = np.zeros((11, 11), bool)
        support[1:10, 1:10] = True
        mask = np.zeros_like(support)
        mask[2:5, 5:8] = True
        hist, area = geometry(mask, support)
        points = np.argwhere(mask)
        radii = np.linalg.norm(points-5, axis=1)/np.sqrt(32)
        expected = np.histogram(radii, bins=np.linspace(0, 1, 17))[0]/9
        np.testing.assert_array_equal(hist, expected)
        self.assertEqual(area, 9/81)
        rotated, fraction = geometry(np.rot90(mask), support)
        np.testing.assert_array_equal(hist, rotated)
        self.assertEqual(area, fraction)
        with self.assertRaises(ValueError):
            geometry(np.zeros_like(mask), support)

    def test_coverage_uses_complete_pool_and_differs_from_exact(self):
        labels = np.array([[1, 0], [1, 0], [1, 1], [0, 1], [1, 0]], dtype=np.uint8)
        pool = {0: np.array([1, 2, 3, 4])}
        bundle = SimpleNamespace(labels=labels, class_ids=np.array([0, 0, 1, 2, 0]), valid_ids=np.arange(5), reserved_ids=np.arange(5))
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with (folder / "rankings.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["query_id", "rank", "candidate_id"])
                writer.writerows([(0, 1, 2), (0, 2, 1)])
            exact = evaluate_rankings(bundle, pool, {0: [2, 1]}, [1, 2], folder)
            coverage = evaluate_rankings_file(bundle, pool, folder / "rankings.csv", [1, 2], folder)
            self.assertEqual(exact["metrics"]["Hit@1"], 0)
            self.assertEqual(exact["metrics"]["mAP@2"], .25)
            self.assertEqual(coverage["metrics"]["FullCoverage_mAP@2"], 2/3)
            self.assertEqual(coverage["metrics"]["FullCoverageRecall@2"], 2/3)
            self.assertEqual(coverage["metrics"]["Coverage-nDCG@2"], 1)
        # Graded relevance, zero-relevant and short-return behavior.
        values = query_metrics([1, 1], labels[[1, 2, 3]], labels[[1, 2]], [2])
        self.assertAlmostEqual(values["Coverage-nDCG@2"], (.5+1/np.log2(3))/(1+.5/np.log2(3)))
        empty = query_metrics([1, 1], labels[[1, 3]], np.empty((0, 2)), [1])
        self.assertIsNone(empty["FullCoverageAP@1"])
        self.assertEqual(empty["Coverage-nDCG@1"], 0)


class GeometryPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def train_fixture(self, root, methods=METHODS, stage="all"):
        legacy.PipelineTests().make_config(root)
        args = ["--config", str(root / "suite.json"), "--device", "cpu", "--num-workers", "0",
                "--methods", *methods, "--stage", stage]
        training_run.main(args)
        return args

    def eval_args(self, root, methods=METHODS):
        return ["--source-root", str(root / "runs"), "--output-root", str(root / "geometry"),
                "--methods", *methods, "--device", "cpu"]

    def test_all_six_existing_models_two_splits_both_metrics_and_resume(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(ef_training, "build_model", legacy.tiny_model), patch.object(ef_suite, "build_model", legacy.tiny_model):
            root = Path(directory)
            train_args = self.train_fixture(root)
            before = snapshot(root / "runs")
            protected = {p: file_hash(p) for d in ("deduplicated", "retrieval") for p in (root / d).iterdir()}
            run.main(self.eval_args(root))
            self.assertEqual(before, snapshot(root / "runs"))
            self.assertEqual(protected, {p: file_hash(p) for p in protected})
            for method in METHODS:
                folder = root / "geometry" / method
                config, suite, checkpoint, kind, _ = pipeline.load_source(root / "runs", method, device="cpu")
                protocol = Protocol(config)
                for split in ("valid", "test"):
                    target = folder / split / "best_nce"
                    exact, coverage = (read_json(target / n) for n in ("metrics.json", "coverage_metrics.json"))
                    self.assertEqual(exact["num_queries"], 6 if split == "valid" else 36)
                    self.assertEqual(coverage["num_queries"], exact["num_queries"])
                    self.assertEqual(len(exact["metrics"]), 12)
                    self.assertEqual(len(coverage["metrics"]), 18)
                    ranks = csv_rows(target / "rankings.csv")
                    self.assertTrue(all(r["query_id"] != r["candidate_id"] for r in ranks))
                    features = np.load(folder / f"features/{split}/h512.npy")
                    meta = np.load(folder / f"features/{split}/regions.npz")
                    # Independent scalar re-score the first query against its entire pool.
                    pool = getattr(protocol.retrieval, split)
                    q = int(next(iter(pool)))
                    positions = {int(s): i for i, s in enumerate(meta["map_ids"])}
                    qi = positions[q]
                    qt = range(meta["offsets"][qi], meta["offsets"][qi+1])
                    self.assertEqual(len(qt), 1)  # Fixture has one retained region per wafer.
                    scores = []
                    for c in pool[q]:
                        ci = positions[int(c)]
                        ct = range(meta["offsets"][ci], meta["offsets"][ci+1])
                        self.assertEqual(len(ct), 1)
                        i, j = qt.start, ct.start
                        radial = np.abs(np.cumsum(meta["histograms"][i])-np.cumsum(meta["histograms"][j])).sum()/16
                        scale = .5*abs(np.log(meta["relative_area"][i]/meta["relative_area"][j]))
                        scores.append(np.clip(features[i]@features[j], 0, 1)*np.exp(-radial)*(.85+.15*np.exp(-scale/1.5)))
                    candidates = pool[q]
                    # Score tolerances avoid BLAS dot-vs-matmul ulp tie changes.
                    by_id = dict(zip(candidates.tolist(), scores))
                    for row in ranks:
                        if int(row["query_id"]) == q:
                            self.assertAlmostEqual(float(row["similarity_score"]), by_id[int(row["candidate_id"])], places=6)
            output_before = snapshot(root / "geometry")
            run.main(self.eval_args(root))
            for name, value in output_before.items():
                if name not in ("run.log", "status.json", "comparison.csv", "comparison.json"):
                    self.assertEqual(snapshot(root / "geometry")[name], value)
            self.assertEqual(before, snapshot(root / "runs"))
            # Existing CLI still recognizes its own signatures after adding new scoring.
            training_run.main(train_args)
            self.assertEqual(file_hash(root / "runs/cc8/seed_2026/F/best_nce.pt"), before["cc8/seed_2026/F/best_nce.pt"][0])

    def test_missing_test_cache_rebuilt_only_in_new_output_and_partial_retry(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(ef_training, "build_model", legacy.tiny_model), patch.object(ef_suite, "build_model", legacy.tiny_model):
            root = Path(directory)
            self.train_fixture(root, ("cc4",), stage="train")
            before = snapshot(root / "runs")
            args = self.eval_args(root, ("cc4",))
            with patch.object(pipeline, "evaluate_rankings_file", side_effect=InterruptedError("after exact")):
                with self.assertRaises(InterruptedError):
                    run.main(args)
            partial = list((root / "geometry/cc4/valid").glob(".best_nce_attempt_*/metrics.json"))
            self.assertEqual(len(partial), 1)
            partial_hash = file_hash(partial[0])
            run.main(args)
            self.assertEqual(file_hash(partial[0]), partial_hash)
            self.assertEqual(before, snapshot(root / "runs"))
            self.assertFalse((root / "runs/cc4/cache/test_prepared.json").exists())
            provenance = read_json(root / "geometry/cc4/features/test/provenance.json")
            self.assertEqual(provenance["region_source"], "recomputed_same_segmentation")
            self.assertTrue((root / "geometry/cc4/test/best_nce/coverage_metrics.json").exists())
            damaged = root / "geometry/cc4/features/test/h512.npy"
            with damaged.open("r+b") as f:
                f.seek(-1, 2)
                f.write(b"\xff")
            with self.assertRaisesRegex(ValueError, "modified"):
                run.main(args)

    def test_rejects_overlap_changed_sources_checkpoint_rule_and_running_source(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(ef_training, "build_model", legacy.tiny_model), patch.object(ef_suite, "build_model", legacy.tiny_model):
            root = Path(directory)
            self.train_fixture(root, ("cc8",), stage="train")
            args = self.eval_args(root, ("cc8",))
            with self.assertRaises(SystemExit):
                run.main(args + ["--output-root", str(root / "runs/cc8/new")])
            with self.assertRaisesRegex(ValueError, "checkpoint rule"):
                run.main(args + ["--checkpoint", "best_retrieval"])
            with pipeline.source_lock(root / "runs"):
                with self.assertRaisesRegex(RuntimeError, "training is running"):
                    run.main(args)
            manifest = root / "runs/cc8/suite_manifest.json"
            data = read_json(manifest)
            data["settings"]["proposal"]["top_k"] = 5
            write_json(manifest, data)
            with self.assertRaisesRegex(ValueError, "manifest signature"):
                run.main(args)


if __name__ == "__main__":
    unittest.main()
