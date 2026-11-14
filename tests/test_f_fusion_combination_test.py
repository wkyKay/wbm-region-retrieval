"""Fixed combination, complete candidates, provenance and historical-result preservation."""
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
from experiments.proposed.f_fusion_ablation import scoring
from experiments.proposed.f_fusion_ablation_test import pipeline as previous_pipeline
from experiments.proposed.f_fusion_combination_test import pipeline, run
from wbm_retrieval.common import ROOT, csv_rows, digest, file_hash, read_json, write_json
from wbm_retrieval.data import Protocol, source_fingerprint
from wbm_retrieval.retrieval_protocol import FullGalleryPool


def signed(payload):
    return {**payload, "signature": digest(payload)}


class CombinationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_formula_reversions_and_no_local_only_penalty(self):
        local, whole, distance = np.array([.6, .2, 0.]), np.array([.2, .8, .5]), np.array([.8, 0., 1.])
        values = {"optimal": .5*local+.5*whole, "angular_distance": distance}
        expected = (.5*local+.5*whole)*np.exp(-.5*distance)
        np.testing.assert_array_equal(pipeline.combined_score(values), expected)
        self.assertNotEqual(expected[0], .5*local[0]*np.exp(-.5*distance[0])+.5*whole[0])
        zero = {**values, "angular_distance": np.zeros(3)}
        np.testing.assert_array_equal(pipeline.combined_score(zero), values["optimal"])
        # If assignments coincide, the combination is exactly the existing angular-only score.
        same = {**values, "baseline": values["optimal"]}
        np.testing.assert_array_equal(pipeline.combined_score(same), scoring.variant_score(same, "angular", .5))

    def test_full_gallery_promotion_beyond_both_reference_top1_and_resume(self):
        testcase = self
        class Scores:
            def components(self, query, candidates):
                testcase.assertEqual(query, 0)
                np.testing.assert_array_equal(candidates, [1, 2, 3])
                return {"baseline": np.array([.9, .6, .7]), "optimal": np.array([.91, 1., .95]),
                        "angular_distance": np.array([0., 1., .05]), "f_score": np.array([.9, .6, .7]),
                        "optimal_f_score": np.array([.91, 1., .95]), "global_score": np.ones(3)}
        labels = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
        ids = np.arange(4)
        protocol = type("Protocol", (), {"ids": {"valid": ids, "test": ids}})()
        pool = FullGalleryPool([0], ids, "test", "fixture")
        values = Scores().components(0, np.array([1, 2, 3]))
        self.assertEqual(np.argmax(scoring.variant_score(values, "angular", .5))+1, 1)
        self.assertEqual(np.argmax(values["optimal"])+1, 2)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            bundle = fusion.bundle_for(protocol, labels)
            with patch.object(geo, "evaluate_rankings_file", side_effect=InterruptedError("after exact")):
                with self.assertRaises(InterruptedError):
                    pipeline.evaluate(output, "sig", Scores(), pool, bundle, [1], 2026)
            attempts = list((output / "test").glob(".*attempt_*"))
            before = {str(p): snapshot(p) for p in attempts}
            pipeline.evaluate(output, "sig", Scores(), pool, bundle, [1], 2026)
            self.assertEqual(before, {str(p): snapshot(p) for p in attempts})
            target = output / "test" / pipeline.VARIANT
            rows = csv_rows(target / "rankings.csv")
            self.assertEqual([r["candidate_id"] for r in rows], ["3"])
            self.assertEqual(read_json(target / "metrics.json")["metrics"]["Precision@1"], 1.)
            self.assertEqual(read_json(target / "coverage_metrics.json")["metrics"]["FullCoveragePrecision@1"], 1.)
            done = snapshot(target)
            with patch.object(Scores, "components", side_effect=AssertionError("must reuse")):
                pipeline.evaluate(output, "sig", Scores(), pool, bundle, [1], 2026)
            self.assertEqual(snapshot(target), done)

    def test_reject_search_wrong_split_and_collisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            for extra in (["--stage", "valid"], ["--stage", "all"], ["--alpha", ".4"], ["--gamma", "1"]):
                with self.assertRaises(SystemExit):
                    run.main(["--output-root", str(output), *extra])
            run.main(["--output-root", str(output), "--dry-run"])
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError, "separate"):
                run.main(["--output-root", str(output), "--previous-test-root", str(output), "--dry-run"])
            pool = FullGalleryPool([0], [0, 1], "valid", "fixture")
            with self.assertRaisesRegex(ValueError, "only permits"):
                pipeline.evaluate(output, "sig", None, pool, None, [1], 2026)
            output.mkdir()
            (output / "keep").write_text("history")
            before = snapshot(output)
            with self.assertRaisesRegex(ValueError, "empty"):
                run.main(["--output-root", str(output)])
            self.assertEqual(snapshot(output), before)

    def test_complete_prior_test_handoff_comparison_readonly_and_cache_guards(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(ef_training, "build_model", legacy.tiny_model), \
                patch.object(ef_suite, "build_model", legacy.tiny_model):
            root = Path(tmp)
            legacy.PipelineTests().make_config(root)
            train_run.main(["--config", str(root / "suite.json"), "--methods", "cc8", "--stage", "train",
                            "--device", "cpu", "--num-workers", "0"])
            config, suite, checkpoint, _, inputs = geo.load_source(root / "runs", "cc8", device="cpu")
            protocol, sources = Protocol(config), source_fingerprint(config)
            pool = protocol.retrieval.test
            global_fixture(root / "global", "test", pool, sources)
            gids, gf, global_id = fusion.load_global(root / "global", "test", pool, sources)
            previous = root / "previous_test"
            previous.mkdir()
            (previous / ".ablation.lock").touch()
            geometry = geo.new_identity(config, inputs)
            with np.load(config["data_file"]) as archive:
                maps, labels = archive["arr_0"], archive["arr_1"]
            local = geo.build_features(previous / "f_cache", config, suite, checkpoint, protocol,
                                       maps, "test", digest(geometry), 0)
            selection = signed({"alpha_global": .5, "split": "valid", "test_evaluated": False,
                         "selections": {"angular": {"selected": {"variant": "angular_0.5", "strength": .5}},
                                        "density": {"selected": {"variant": "baseline", "strength": 0.}}}})
            # Synthetic upstream metadata with real old-evaluator reports and feature hashes.
            selected = {"baseline": "baseline", "density": "baseline", "angular": "angular_0.5", "optimal": "optimal"}
            code = lambda folder: {str(p.relative_to(ROOT)): file_hash(p) for p in folder.glob("*.py")}
            manifest = signed({"version": "f_fusion_four_test_v1", "alpha_global": .5,
                       "selected_variants": selected, "selection_signature": selection["signature"],
                       "geometry": geometry, "sources": sources, "global": global_id,
                       "ks": config["evaluation"]["ks"], "tie_seed": 2026,
                       "code": code(Path(previous_pipeline.__file__).parent),
                       "valid_code": code(Path(scoring.__file__).parent)})
            descriptions = previous_pipeline.descriptors(previous, maps, pool.gallery, manifest["signature"])
            evaluation = signed({"run_signature": manifest["signature"], "f_features": file_hash(local / "completed.json"),
                                  "descriptors": file_hash(previous / "descriptors/completed.json")})
            for name, payload in [("test_manifest.json", manifest), ("evaluation_manifest.json", evaluation),
                                  ("frozen_selection.json", selection)]:
                write_json(previous / name, payload)
            scorer = scoring.Scores(local, gids, gf, pool.gallery, config["matching"], descriptions)
            previous_pipeline.evaluate(previous, evaluation["signature"], scorer, pool,
                                       fusion.bundle_for(protocol, labels), config["evaluation"]["ks"], 2026, selected)
            previous_pipeline.summarize(previous, evaluation["signature"], selected)
            protected = {n: snapshot(root / n) for n in ("previous_test", "runs", "global", "deduplicated", "retrieval")}
            output = root / "combination"
            args = ["--source-root", str(root / "runs"), "--f-feature-root", str(root / "geometry"),
                    "--fusion-root", str(root / "old_fusion"), "--previous-test-root", str(previous),
                    "--wapirl-test-root", str(root / "global"), "--output-root", str(output)]
            global_fixture(root / "wrong", "test", pool, sources, checkpoint="different")
            with self.assertRaisesRegex(ValueError, "same completed"):
                run.main([*args, "--wapirl-test-root", str(root / "wrong")])
            self.assertFalse(output.exists())
            with patch.object(geo, "build_features", side_effect=AssertionError("must not extract")):
                run.main(args)
            self.assertEqual(read_json(output / "status.json")["status"], "completed")
            self.assertTrue(read_json(output / "plan_before_evaluation.json")["posthoc_test_combination"])
            self.assertEqual({p.name for p in (output / "test").iterdir()}, {pipeline.VARIANT})
            self.assertFalse((output / "selection.json").exists())
            self.assertFalse((output / "valid").exists())
            actual = csv_rows(output / "test" / pipeline.VARIANT / "rankings.csv")
            self.assertEqual({int(r["query_id"]) for r in actual}, set(pool))
            for r in actual:
                self.assertNotEqual(r["query_id"], r["candidate_id"])
                expected = (.5*float(r["optimal_f_score"])+.5*float(r["global_score"]))*np.exp(-.5*float(r["angular_distance"]))
                self.assertAlmostEqual(float(r["similarity_score"]), expected, places=14)
            summary = csv_rows(output / "comparison.csv")
            index = {(r["variant"], r["group"], r["metric"]): (float(r["value"]) if r["value"] else None) for r in summary}
            self.assertEqual({r["variant"] for r in summary}, {*pipeline.REFERENCES, pipeline.VARIANT})
            for row in csv_rows(output / "combination.csv"):
                for ref, col in zip(pipeline.REFERENCES, ("baseline", "angular", "optimal")):
                    if not row["value"]:
                        self.assertEqual(row[f"delta_vs_{col}"], "")
                    else:
                        self.assertAlmostEqual(float(row[f"delta_vs_{col}"]), float(row["value"])-index[ref,row["group"],row["metric"]])
            before = snapshot(output / "test")
            with patch.object(scoring.Scores, "components", side_effect=AssertionError("must reuse")):
                run.main(args)
            self.assertEqual(snapshot(output / "test"), before)
            self.assertEqual(protected, {n: snapshot(root / n) for n in protected})
            # Signed but scientifically altered upstream choice is rejected.
            changed = {k:v for k,v in manifest.items() if k != "signature"}
            changed["selected_variants"] = {**selected, "angular": "angular_1.0"}
            write_json(previous / "test_manifest.json", signed(changed))
            with self.assertRaisesRegex(ValueError, "Expected previous"):
                pipeline.load_previous(previous)
            write_json(previous / "test_manifest.json", manifest)
            # A cache with a valid geometry marker but different content must not replace prior features.
            alternative = root / "geometry/cc8/features/test"
            shutil.copytree(local, alternative)
            with (alternative / "h512.npy").open("ab") as handle:
                handle.write(b"changed")
            geo.commit_marker(alternative, digest(geometry))
            with self.assertRaisesRegex(ValueError, "cache differs"):
                run.main(args)
            # Corruption of any comparison report must also be detected.
            with (previous / "test/angular_0.5/metrics.json").open("a") as handle:
                handle.write(" ")
            with self.assertRaisesRegex(ValueError, "modified"):
                pipeline.load_previous(previous)


if __name__ == "__main__":
    unittest.main()
