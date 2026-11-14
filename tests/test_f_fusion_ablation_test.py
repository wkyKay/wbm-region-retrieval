"""Frozen valid-to-test handoff, baseline parity, full-gallery evaluation and resume."""
from pathlib import Path
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
from experiments.proposed.f_fusion_ablation import run as valid_run, scoring
from experiments.proposed.f_fusion_ablation_test import pipeline, run
from wbm_retrieval.common import csv_rows, digest, read_json, write_json
from wbm_retrieval.data import Protocol, source_fingerprint
from wbm_retrieval.retrieval_protocol import FullGalleryPool


class FrozenTestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_reject_tuning_wrong_split_and_output_collisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            for extra in (["--stage", "valid"], ["--stage", "all"], ["--alpha", ".3"], ["--gamma", "2"]):
                with self.assertRaises(SystemExit):
                    run.main(["--output-root", str(output), *extra])
            self.assertFalse(output.exists())
            run.main(["--output-root", str(output), "--dry-run"])
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError, "separate"):
                run.main(["--output-root", str(output), "--valid-root", str(output), "--dry-run"])
            pool = FullGalleryPool([0], [0, 1], "valid", "fixture")
            with self.assertRaisesRegex(ValueError, "only permits"):
                pipeline.evaluate(output, "sig", None, pool, None, [1], 2026, {})
            output.mkdir()
            (output / "keep").write_text("history")
            before = snapshot(output)
            with self.assertRaisesRegex(ValueError, "empty"):
                run.main(["--output-root", str(output)])
            self.assertEqual(snapshot(output), before)

    def test_end_to_end_frozen_selection_baseline_parity_readonly_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(ef_training, "build_model", legacy.tiny_model), \
                patch.object(ef_suite, "build_model", legacy.tiny_model):
            root = Path(tmp)
            legacy.PipelineTests().make_config(root)
            train_run.main(["--config", str(root / "suite.json"), "--methods", "cc8", "--stage", "train",
                            "--device", "cpu", "--num-workers", "0"])
            config, _, _, _, _ = geo.load_source(root / "runs", "cc8", device="cpu")
            protocol = Protocol(config)
            sources = source_fingerprint(config)
            global_fixture(root / "global_valid", "valid", protocol.retrieval.valid, sources)
            gids, gf = global_fixture(root / "global_test", "test", protocol.retrieval.test, sources)
            common = ["--source-root", str(root / "runs"), "--f-feature-root", str(root / "geometry"),
                      "--fusion-root", str(root / "previous")]
            valid = root / "valid_run"
            valid_run.main([*common, "--wapirl-valid-root", str(root / "global_valid"), "--output-root", str(valid)])
            manifest, selection, selected = pipeline.load_selection(valid)
            protected_names = ("runs", "valid_run", "global_valid", "global_test", "deduplicated", "retrieval")
            protected = {name: snapshot(root / name) for name in protected_names}
            output = root / "test_run"
            args = [*common, "--valid-root", str(valid), "--wapirl-test-root", str(root / "global_test"),
                    "--output-root", str(output)]
            # A different test checkpoint cannot be silently accepted, even with valid cache signatures.
            global_fixture(root / "wrong_global", "test", protocol.retrieval.test, sources, checkpoint="different")
            with self.assertRaisesRegex(ValueError, "same completed"):
                run.main([*args, "--wapirl-test-root", str(root / "wrong_global")])
            self.assertFalse(output.exists())
            # A re-signed manual selection still must match the frozen valid metrics.
            edited = read_json(valid / "selection.json")
            edited["selections"]["angular"]["selected"]["value"] += .01
            edited.pop("signature")
            write_json(valid / "selection.json", {**edited, "signature": digest(edited)})
            with self.assertRaisesRegex(ValueError, "does not match"):
                pipeline.load_selection(valid)
            write_json(valid / "selection.json", selection)
            protected["valid_run"] = snapshot(valid)
            build = geo.build_features
            def test_only(*a, **kw):
                self.assertEqual(a[6], "test")
                return build(*a, **kw)
            with patch.object(geo, "build_features", side_effect=test_only):
                run.main(args)
            self.assertEqual(read_json(output / "status.json")["status"], "completed")
            self.assertEqual(read_json(output / "frozen_selection.json"), selection)
            self.assertEqual(read_json(output / "test_manifest.json")["selected_variants"], selected)
            self.assertFalse((output / "valid").exists())
            self.assertFalse((output / "selection.json").exists())
            self.assertEqual({p.name for p in (output / "test").iterdir()}, set(selected.values()))
            with np.load(output / "descriptors/test.npz") as data:
                np.testing.assert_array_equal(data["sample_ids"], protocol.retrieval.test.gallery)
            # Independent existing fusion evaluator verifies the baseline's IDs AND scores.
            local_dir = output / "f_cache/features/test"
            old = fusion.BranchScores(local_dir, gids, gf, protocol.retrieval.test.gallery, config["matching"])
            with np.load(config["data_file"]) as archive:
                labels = archive["arr_1"]
            fusion.evaluate(root / "reference", "test", [.5], "reference", old, protocol.retrieval.test,
                            fusion.bundle_for(protocol, labels), config["evaluation"]["ks"], 2026)
            actual = csv_rows(output / "test/baseline/rankings.csv")
            expected = csv_rows(root / "reference/test/alpha_0.5/rankings.csv")
            keys = ("query_id", "candidate_id", "similarity_score")
            self.assertEqual([tuple(r[k] for k in keys) for r in actual], [tuple(r[k] for k in keys) for r in expected])
            self.assertEqual({int(r["query_id"]) for r in actual}, set(protocol.retrieval.test))
            self.assertTrue(all(r["query_id"] != r["candidate_id"] for r in actual))
            summary = csv_rows(output / "four_groups.csv")
            self.assertEqual({r["experiment"] for r in summary}, set(selected))
            self.assertTrue(all(r["split"] == "test" and r["alpha_global"] == "0.5" for r in summary))
            for name in set(selected.values()):
                self.assertTrue((output / "test" / name / "coverage_metrics.json").exists())
            before = snapshot(output / "test")
            with patch.object(scoring.Scores, "components", side_effect=AssertionError("must reuse")):
                run.main(args)
            self.assertEqual(snapshot(output / "test"), before)
            self.assertEqual(protected, {name: snapshot(root / name) for name in protected})
            with (local_dir / "h512.npy").open("ab") as handle:
                handle.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "modified"):
                run.main(args)

    def test_selected_variants_only_duplicate_zero_group_and_interrupted_metrics(self):
        class ConstantScores:
            def components(self, q, candidates):
                return {k: np.full(len(candidates), .5 if "distance" not in k else 0.) for k in
                        ("baseline", "optimal", "f_score", "global_score", "optimal_f_score",
                         "density_distance", "angular_distance")}
        ids = np.arange(12)
        labels = np.eye(2, dtype=np.uint8)[ids % 2]
        pool = FullGalleryPool(ids, ids, "test", "fixture")
        protocol = type("Protocol", (), {"ids": {"valid": ids, "test": ids}})()
        selected = {"baseline": "baseline", "density": "baseline", "angular": "angular_0.5", "optimal": "optimal"}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            bundle = fusion.bundle_for(protocol, labels)
            with patch.object(geo, "evaluate_rankings_file", side_effect=InterruptedError("after exact")):
                with self.assertRaises(InterruptedError):
                    pipeline.evaluate(output, "sig", ConstantScores(), pool, bundle, [1, 10], 2026, selected)
            attempts = list((output / "test").glob(".*attempt_*"))
            before = {str(p): snapshot(p) for p in attempts}
            pipeline.evaluate(output, "sig", ConstantScores(), pool, bundle, [1, 10], 2026, selected)
            pipeline.summarize(output, "sig", selected)
            self.assertEqual(before, {str(p): snapshot(p) for p in attempts})
            self.assertEqual({p.parent.name for p in (output / "test").glob("*/completed.json")}, set(selected.values()))
            summary = csv_rows(output / "four_groups.csv")
            self.assertEqual({r["experiment"] for r in summary}, set(selected))
            self.assertEqual({r["variant"] for r in summary if r["experiment"] == "density"}, {"baseline"})
            self.assertEqual({r["variant"] for r in summary if r["experiment"] == "angular"}, {"angular_0.5"})


if __name__ == "__main__":
    unittest.main()
