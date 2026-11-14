"""Mathematical and end-to-end tests for the valid-only four-group experiment."""
import itertools
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_f_segmentation as legacy
from test_f_global_fusion import global_fixture
from test_f_segmentation_geometry import snapshot
from experiments.proposed._shared import ef_suite, ef_training
from experiments.proposed.f_segmentation import run as train_run
from experiments.proposed.f_segmentation_geometry import pipeline as geo
from experiments.proposed.f_global_fusion import pipeline as fusion
from experiments.proposed.f_fusion_ablation import scoring, pipeline, run
from wbm_retrieval.common import csv_rows, read_json, write_json
from wbm_retrieval.data import Protocol, source_fingerprint
from wbm_retrieval.diagnostics import batched_map_scores
from wbm_retrieval.retrieval_protocol import FullGalleryPool


MATCHING = {"lambda_pos": 0, "lambda_scale": 0, "shape_threshold": 0,
            "score_threshold": 0, "weight": "sqrt_area"}


class MathematicsTests(unittest.TestCase):
    def test_original_mask_density_and_empty_outer_policy(self):
        raw = np.ones((11, 11), dtype=np.uint8)
        raw[0, 0] = 0
        raw[0, 1] = 3
        raw[5, 5] = 2
        density, moments, count = scoring.describe(raw)
        self.assertEqual(density, 1/119)
        self.assertEqual(count, 0)
        np.testing.assert_array_equal(moments, np.zeros(4))
        raw[0, 10] = 2  # Included despite being an isolated small component.
        density, moments, count = scoring.describe(raw)
        self.assertEqual(density, 2/119)
        self.assertEqual(count, 1)
        np.testing.assert_allclose(moments, 1)
        empty = np.ones((5, 5))
        self.assertEqual(scoring.describe(empty)[0], 0)
        with self.assertRaises(ValueError):
            scoring.describe(np.zeros((5, 5)))

    def test_continuous_rotation_and_raw_quarter_turn_invariance(self):
        rng = np.random.default_rng(2026)
        points = rng.normal(size=(32, 2)) + [2, 1]
        expected = scoring.moment_magnitudes(points, [0, 0])
        for theta in (.17, .71, 1.92):
            matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            actual = scoring.moment_magnitudes(points @ matrix.T, [0, 0])
            np.testing.assert_allclose(actual, expected, atol=1e-14)
        raw = np.ones((17, 17), dtype=np.uint8)
        raw[1:5, 10:14] = 2
        base = scoring.describe(raw)
        for k in (1, 2, 3):
            actual = scoring.describe(np.rot90(raw, k))
            self.assertEqual(actual[0], base[0])
            self.assertEqual(actual[2], base[2])
            np.testing.assert_allclose(actual[1], base[1], atol=1e-14)
        angles = np.linspace(0, 2*np.pi, 64, endpoint=False)
        ring = np.column_stack([np.sin(angles), np.cos(angles)])
        np.testing.assert_allclose(scoring.moment_magnitudes(ring, [0, 0]), 0, atol=1e-14)
        self.assertGreater(scoring.moment_magnitudes(ring[:8], [0, 0])[0], .9)

    def test_exact_assignment_against_independent_partial_enumeration(self):
        rng = np.random.default_rng(25)
        for nq in range(5):
            for nc in range(5):
                pairs = rng.uniform(-.2, 1.2, (nq, max(nc, 1)))
                # Holes in padded indices must not create real matches.
                ids = np.full((1, 4), -1, dtype=int)
                ids[0, :nc] = np.arange(nc)
                area = rng.integers(5, 80, nq)
                actual = scoring.optimal_map_scores(pairs, ids, area)[0]
                best = 0.
                for choices in itertools.product(range(-1, nc), repeat=nq):
                    real = [j for j in choices if j >= 0]
                    if len(set(real)) != len(real):
                        continue
                    value = sum(np.sqrt(area[i])*np.clip(pairs[i, j], 0, 1)
                                for i, j in enumerate(choices) if j >= 0)
                    best = max(best, value)
                self.assertAlmostEqual(actual, best / np.sqrt(area).sum() if nq else 0, places=13)
                greedy = batched_map_scores(pairs, ids, area, MATCHING)[0]
                self.assertGreaterEqual(actual+1e-14, greedy)
        pairs = np.array([[.9, .8], [.85, 0.]])
        ids = np.array([[0, 1]])
        self.assertAlmostEqual(scoring.optimal_map_scores(pairs, ids, [9, 9])[0], .825)
        self.assertAlmostEqual(batched_map_scores(pairs, ids, [9, 9], MATCHING)[0], .45)

    def test_baseline_and_zero_strength_are_exact_reversions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = np.array([10, 30, 50])
            features = np.array([[1., 0.], [.6, .8], [0., 1.]])
            np.save(root / "h512.npy", features)
            np.savez(root / "regions.npz", map_ids=ids, offsets=[0, 0, 2, 3], area=[5, 8, 10],
                     relative_area=[.05, .08, .1], histograms=np.ones((3, 16))/16)
            global_ids = ids[::-1]
            global_features = features[::-1]
            descriptions = {"sample_ids": ids, "density": np.array([.1, .2, .3]),
                            "moments": np.array([[0.]*4, [.9]*4, [.2]*4]), "outer_count": np.array([0, 2, 5])}
            old = fusion.BranchScores(root, global_ids, global_features, ids, MATCHING)
            new = scoring.Scores(root, global_ids, global_features, ids, MATCHING, descriptions)
            for q in ids:
                candidates = ids[ids != q]
                local, whole = old.score(q, candidates)
                values = new.components(q, candidates)
                np.testing.assert_array_equal(values["baseline"], .5*local+.5*whole)
                for family in ("density", "angular"):
                    np.testing.assert_array_equal(scoring.variant_score(values, family, 0), values["baseline"])
                    expected = values["baseline"]*np.exp(-2*values[family+"_distance"])
                    np.testing.assert_allclose(scoring.variant_score(values, family, 2), expected)
                if q == 10:
                    np.testing.assert_array_equal(values["angular_distance"], 0)

    def test_reject_test_all_alpha_overrides_and_output_collisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            for extra in (["--stage", "test"], ["--stage", "all"], ["--alpha", "0.3"]):
                with self.assertRaises(SystemExit):
                    run.main(["--output-root", str(output), *extra])
            self.assertFalse(output.exists())
            pool = FullGalleryPool([0], [0, 1], "test", "fixture")
            with self.assertRaisesRegex(ValueError, "only permits"):
                pipeline.evaluate(output, "sig", None, pool, None, [1], 2026)
            output.mkdir()
            (output / "keep").write_text("existing")
            before = snapshot(output)
            with self.assertRaisesRegex(ValueError, "empty"):
                run.main(["--output-root", str(output)])
            self.assertEqual(snapshot(output), before)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_end_to_end_valid_only_resume_selection_baseline_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(ef_training, "build_model", legacy.tiny_model), \
                patch.object(ef_suite, "build_model", legacy.tiny_model):
            root = Path(tmp)
            legacy.PipelineTests().make_config(root)
            train_run.main(["--config", str(root / "suite.json"), "--methods", "cc8", "--stage", "train",
                            "--device", "cpu", "--num-workers", "0"])
            config, _, _, _, _ = geo.load_source(root / "runs", "cc8", device="cpu")
            protocol = Protocol(config)
            global_fixture(root / "global_valid", "valid", protocol.retrieval.valid, source_fingerprint(config))
            protected = {name: snapshot(root / name) for name in ("runs", "global_valid", "deduplicated", "retrieval")}
            args = ["--source-root", str(root / "runs"), "--f-feature-root", str(root / "geometry"),
                    "--fusion-root", str(root / "previous"), "--wapirl-valid-root", str(root / "global_valid"),
                    "--output-root", str(root / "ablation")]
            build = geo.build_features
            def valid_only(*a, **kw):
                self.assertEqual(a[6], "valid")
                return build(*a, **kw)
            with patch.object(geo, "build_features", side_effect=valid_only):
                run.main(args)
            output = root / "ablation"
            self.assertEqual(read_json(output / "status.json")["status"], "completed")
            self.assertFalse(read_json(output / "status.json")["test_evaluated"])
            self.assertFalse(any(p.name == "test" for p in output.rglob("*")))
            self.assertEqual(len(list((output / "valid").glob("*/completed.json"))), 10)
            selected = read_json(output / "selection.json")
            for family in ("density", "angular"):
                entries = selected["selections"][family]
                expected = max(entries["candidates"], key=lambda c: (c["value"], -c["strength"]))
                self.assertEqual(expected, entries["selected"])
                self.assertEqual(len(entries["candidates"]), 5)
            # Compare the baseline against the unchanged previous fusion evaluator.
            gids, gf, _ = fusion.load_global(root / "global_valid", "valid", protocol.retrieval.valid, source_fingerprint(config))
            scorer = fusion.BranchScores(output / "f_cache/features/valid", gids, gf,
                                         protocol.retrieval.valid.gallery, config["matching"])
            with np.load(config["data_file"]) as archive:
                labels = archive["arr_1"]
            fusion.evaluate(root / "reference", "valid", [.5], "reference", scorer, protocol.retrieval.valid,
                            fusion.bundle_for(protocol, labels), config["evaluation"]["ks"], 2026)
            actual = csv_rows(output / "valid/baseline/rankings.csv")
            expected = csv_rows(root / "reference/valid/alpha_0.5/rankings.csv")
            keys = ("query_id", "candidate_id", "similarity_score")
            self.assertEqual([tuple(r[k] for k in keys) for r in actual], [tuple(r[k] for k in keys) for r in expected])
            self.assertTrue(all(r["query_id"] != r["candidate_id"] for r in actual))
            summary = csv_rows(output / "four_groups.csv")
            self.assertEqual({r["experiment"] for r in summary}, {"baseline", "density", "angular", "optimal"})
            self.assertTrue(all(r["split"] == "valid" and r["alpha_global"] == "0.5" for r in summary))
            before = snapshot(output / "valid")
            selection_before = snapshot(output)["selection.json"]
            with patch.object(scoring.Scores, "components", side_effect=AssertionError("must reuse")):
                run.main(args)
            self.assertEqual(snapshot(output / "valid"), before)
            self.assertEqual(snapshot(output)["selection.json"], selection_before)
            # Reuse a relocated previous fusion cache without modifying it.
            shutil.copytree(output / "f_cache/features", root / "previous/f_cache/features")
            previous = snapshot(root / "previous")
            with patch.object(geo, "build_features", side_effect=AssertionError("must reuse external cache")):
                run.main(args)
            self.assertEqual(snapshot(root / "previous"), previous)
            self.assertEqual(protected, {name: snapshot(root / name) for name in protected})
            # Scientific setting changes cannot silently reuse the current run.
            with patch.dict(scoring.SETTINGS, {"outer_radius_min": .5}):
                with self.assertRaisesRegex(ValueError, "changed"):
                    run.main(args)
            with (root / "previous/f_cache/features/valid/h512.npy").open("ab") as handle:
                handle.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "modified"):
                run.main(args)

    def test_metrics_interruption_and_zero_tie_selection(self):
        class ConstantScores:
            def components(self, q, candidates):
                return {k: np.full(len(candidates), .5 if "distance" not in k else 0.) for k in
                        ("baseline", "optimal", "f_score", "global_score", "optimal_f_score",
                         "density_distance", "angular_distance")}
        ids = np.arange(12)
        labels = np.eye(2, dtype=np.uint8)[ids % 2]
        pool = FullGalleryPool([0, 1], ids, "valid", "fixture")
        protocol = type("Protocol", (), {"ids": {"valid": ids, "test": ids}})()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            bundle = fusion.bundle_for(protocol, labels)
            with patch.object(geo, "evaluate_rankings_file", side_effect=InterruptedError("after exact")):
                with self.assertRaises(InterruptedError):
                    pipeline.evaluate(output, "sig", ConstantScores(), pool, bundle, [1, 10], 2026)
            attempts = list((output / "valid").glob(".*attempt_*"))
            before = {str(p): snapshot(p) for p in attempts}
            pipeline.evaluate(output, "sig", ConstantScores(), pool, bundle, [1, 10], 2026)
            self.assertEqual(before, {str(p): snapshot(p) for p in attempts})
            selected = pipeline.summarize(output, "sig")
            self.assertEqual(selected["density"], "baseline")
            self.assertEqual(selected["angular"], "baseline")
            selection = read_json(output / "selection.json")
            selection["alpha_global"] = .9
            write_json(output / "selection.json", selection)
            with self.assertRaisesRegex(ValueError, "selection changed"):
                pipeline.summarize(output, "sig")


if __name__ == "__main__":
    unittest.main()
