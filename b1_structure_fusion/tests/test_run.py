"""CLI/source handoff tests using the existing tiny frozen-model fixture.

Only synthetic source data and tiny CPU models are created in temporary folders.
The real B1 cache, scoring, evaluators, selection and verification run end to end.
Formal registration is mocked at its boundary: these small synthetic pools must
never pass the production 500/7199-query gate. Registration has separate tests.
"""
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "tests"))

import test_f_segmentation as legacy
from test_f_global_fusion import global_fixture
from test_f_segmentation_geometry import snapshot
from b1_structure_fusion import descriptors, pipeline, register, run
from experiments.proposed._shared import ef_suite, ef_training
from experiments.proposed.f_segmentation import run as training_run
from experiments.proposed.f_segmentation_geometry import pipeline as geo
from experiments.proposed.f_fusion_ablation.scoring import Scores
from experiments.proposed.f_fusion_combination_test.pipeline import combined_score
from wbm_retrieval.common import csv_rows, digest, file_hash, read_json
from wbm_retrieval.data import Protocol, source_fingerprint
from wbm_retrieval.exact_metrics import order_scores


class RunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.settings = descriptors.load_settings()
        cls.settings["ks"] = [1, 5, 10]
        with redirect_stdout(io.StringIO()), patch.object(ef_training, "build_model", legacy.tiny_model), \
                patch.object(ef_suite, "build_model", legacy.tiny_model):
            legacy.PipelineTests().make_config(cls.root)
            training_run.main(["--config", str(cls.root / "suite.json"), "--methods", "cc8", "--stage", "train",
                               "--device", "cpu", "--num-workers", "0"])
            config, suite, checkpoint, _, inputs = geo.load_source(cls.root / "runs", "cc8", device="cpu")
            cls.config, cls.protocol = config, Protocol(config)
            with np.load(config["data_file"], allow_pickle=False) as archive:
                cls.maps, cls.labels = archive["arr_0"], archive["arr_1"]
            signature = digest(geo.new_identity(config, inputs))
            sources = source_fingerprint(config)
            cls.locals, cls.globals = {}, {}
            for split in ("valid", "test"):
                cls.locals[split] = geo.build_features(cls.root / "features/cc8", config, suite, checkpoint,
                    cls.protocol, cls.maps, split, signature, 0)
                cls.globals[split] = global_fixture(cls.root / ("global_"+split), split,
                    getattr(cls.protocol.retrieval, split), sources)
        cls.protected = [cls.root / name for name in ("runs", "features", "global_valid", "global_test",
                                                     "deduplicated", "retrieval")]
        cls.source_snapshot = {str(path): snapshot(path) for path in cls.protected}
        cls.arguments = ["--source-root", str(cls.root / "runs"),
                         "--f-feature-root", str(cls.root / "features"),
                         "--fusion-root", str(cls.root / "no_fusion"),
                         "--previous-test-root", str(cls.root / "no_previous"),
                         "--wapirl-valid-root", str(cls.root / "global_valid"),
                         "--wapirl-test-root", str(cls.root / "global_test"),
                         "--data-file", str(cls.root / "source.npz"),
                         "--device", "cpu", "--prepare-workers", "0"]
        (cls.root / "code_fixture.py").write_text("# immutable fixture source\n")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @contextmanager
    def environment(self, settings=None):
        chosen = deepcopy(self.settings if settings is None else settings)
        original_choose = run.choose_output
        captured = io.StringIO()
        def reference(split, _sources):
            return [{"split": split, "group": "all", "metric_version": "exact_v1",
                     "metric": "Precision", "K": 10, "value": .6}], {"key": "wbbs_"+split, "fixture": True}
        with patch.object(run, "ROOT", self.root), \
                patch.object(run, "choose_output", side_effect=lambda args: original_choose(args, root=self.root)), \
                patch.object(run.descriptors, "load_settings", side_effect=lambda path=None: deepcopy(chosen)), \
                patch.object(run, "checked_reference", side_effect=reference), \
                patch.object(register, "write_proposal", return_value=None), \
                patch.object(run, "code_identity", return_value={"code_fixture.py": file_hash(self.root / "code_fixture.py")}), \
                redirect_stdout(captured):
            yield captured

    def assert_sources_unchanged(self):
        self.assertEqual(self.source_snapshot, {str(path): snapshot(path) for path in self.protected})

    def test_valid_then_frozen_test_and_resume_match_original_b0(self):
        valid = self.root / "artificial/b1_valid_20261117T120000+0800"
        test = self.root / "artificial/b1_test_20261117T120100+0800"
        with self.environment() as captured:
            self.assertEqual(run.main([*self.arguments, "--stage", "valid", "--output-root", str(valid)]), 0)
            selection = pipeline.load_selection(valid / "selection.json", self.settings)
            self.assertEqual(set(selection["selections"]), {"R", "T", "I", "J"})
            self.assertEqual(len(selection["completed_sha256"]), 18)
            self.assertEqual(read_json(valid / "status.json")["split"], "valid")
            preserved = {str(path): file_hash(path) for path in valid.rglob("*")
                         if path.is_file() and path.name in ("completed.json", "rankings.csv", "selection.json", "run_identity.json")}
            self.assertEqual(run.main([*self.arguments, "--stage", "valid", "--resume-root", str(valid)]), 0)
            self.assertEqual(preserved, {str(path): file_hash(path) for path in map(Path, preserved)})
            self.assertEqual(run.main([*self.arguments, "--stage", "test", "--output-root", str(test),
                                      "--selection-file", str(valid / "selection.json")]), 0)
            self.assertEqual(read_json(test / "status.json")["split"], "test")
            expected_names = set(pipeline.test_variants(selection, self.settings))
            self.assertEqual({path.name for path in (test / "test").iterdir() if not path.name.startswith(".")}, expected_names)
            self.assertEqual(set(read_json(test / "verification_test.json")), expected_names)
        self.assertIn("exact", captured.getvalue().lower())
        self.assertIn("coverage", captured.getvalue().lower())
        self.assertIn("WBBS", captured.getvalue())
        # Cross-check the unchanged old combination scorer on the same real frozen fixture.
        desc_path = next((test / "descriptors").rglob("test.npz"))
        with np.load(desc_path, allow_pickle=False) as archive:
            description = dict(archive)
        ids, global_features = self.globals["test"]
        pool = self.protocol.retrieval.test
        old = Scores(self.locals["test"], ids, global_features, pool.gallery, self.config["matching"], description)
        query = int(next(iter(pool)))
        candidates = pool[query]
        scores = combined_score(old.components(query, candidates))
        order = order_scores(candidates, scores, query, 2026)[:10]
        rows = [row for row in csv_rows(test / "test/B0/rankings.csv") if int(row["query_id"]) == query]
        np.testing.assert_array_equal([int(row["candidate_id"]) for row in rows], candidates[order])
        np.testing.assert_array_equal([float(row["similarity_score"]) for row in rows], scores[order])
        self.assert_sources_unchanged()

    def test_all_zero_primary_stops_before_test_and_resumes(self):
        output = self.root / "artificial/b1_zero_20261117T120200+0800"
        settings = deepcopy(self.settings)
        settings["lambda_grid"] = [0.]
        loaded_splits = []
        real_loader = run.fusion.load_global
        def load_global(root, split, *args, **kwargs):
            loaded_splits.append(split)
            return real_loader(root, split, *args, **kwargs)
        with self.environment(settings), patch.object(run.fusion, "load_global", side_effect=load_global):
            self.assertEqual(run.main([*self.arguments, "--stage", "all", "--output-root", str(output)]), 0)
            self.assertFalse((output / "test").exists())
            self.assertEqual(loaded_splits, ["valid"])
            self.assertFalse(read_json(output / "status.json")["test_evaluated"])
            self.assertEqual(run.main([*self.arguments, "--stage", "all", "--resume-root", str(output)]), 0)
            self.assertFalse((output / "test").exists())
        self.assert_sources_unchanged()

    def test_dry_run_and_input_check_do_not_create_outputs(self):
        dry = self.root / "artificial/b1_dry_20261117T120300+0800"
        check = self.root / "artificial/b1_check_20261117T120400+0800"
        with self.environment():
            with patch.object(run.geo, "load_source", side_effect=AssertionError("dry-run must not load model")):
                self.assertEqual(run.main([*self.arguments, "--output-root", str(dry), "--dry-run"]), 0)
            self.assertFalse(dry.exists())
            with patch.object(run.pipeline, "cache_descriptors", side_effect=AssertionError("input check must not prepare")):
                self.assertEqual(run.main([*self.arguments, "--stage", "all", "--output-root", str(check), "--check-inputs"]), 0)
            self.assertFalse(check.exists())
        self.assert_sources_unchanged()

    def test_output_collision_missing_cache_and_unsafe_selection(self):
        with self.environment():
            for path in (self.root / "artificial", self.root / "runs", self.root / "artifacts/new_run"):
                with self.assertRaisesRegex(ValueError, "artificial"):
                    run.main([*self.arguments, "--output-root", str(path), "--dry-run"])
            existing = self.root / "artificial/b1_existing_20261117T120500+0800"
            existing.mkdir(parents=True)
            (existing / "keep.txt").write_text("historical")
            before = snapshot(existing)
            with self.assertRaisesRegex(ValueError, "Existing output"):
                run.main([*self.arguments, "--output-root", str(existing)])
            self.assertEqual(before, snapshot(existing))
            previous = self.root / "artificial/b1_previous_20261117T120510+0800"
            previous.mkdir()
            (previous / "run_manifest.json").write_text("{\"historical\": true}\n")
            previous_before = snapshot(previous)
            with self.assertRaisesRegex(ValueError, "nest"):
                run.main([*self.arguments, "--output-root", str(previous / "new_child"), "--dry-run"])
            self.assertEqual(previous_before, snapshot(previous))
            # Selection sources are protected even before loading their signed content.
            selection_parent = self.root / "artificial/selection_source"
            selection_parent.mkdir()
            (selection_parent / "selection.json").write_text("{}\n")
            selection_before = snapshot(selection_parent)
            with self.assertRaisesRegex(ValueError, "separate"):
                run.main([*self.arguments, "--stage", "test", "--selection-file", str(selection_parent / "selection.json"),
                          "--output-root", str(selection_parent / "new_test")])
            self.assertEqual(selection_before, snapshot(selection_parent))
            with self.assertRaisesRegex(ValueError, "selection-file"):
                run.main([*self.arguments, "--stage", "test"])
            missing = self.root / "artificial/b1_missing_20261117T120600+0800"
            with self.assertRaisesRegex(FileNotFoundError, "frozen F features"):
                run.main([*self.arguments, "--f-feature-root", str(self.root / "absent_features"),
                          "--output-root", str(missing), "--check-inputs"])
            self.assertFalse(missing.exists())
        self.assert_sources_unchanged()


if __name__ == "__main__":
    unittest.main()
