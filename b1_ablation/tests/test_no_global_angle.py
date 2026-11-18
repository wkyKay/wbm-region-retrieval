"""The isolated F-optimal times J recipe, including complete evaluation routing."""
from contextlib import ExitStack, nullcontext, redirect_stdout
from copy import deepcopy
import csv
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np

from b1_ablation import pipeline, run, scoring, verify
from b1_ablation.tests.test_pipeline import Components, fixture, snapshot
from wbm_retrieval.common import read_json, write_json


class NoGlobalAngleTests(unittest.TestCase):
    def setUp(self):
        self.settings = deepcopy(scoring.NO_GLOBAL_ANGLE_SETTINGS)
        self.spec = scoring.specs("valid", self.settings)["no_global_angle"]
        self.values = {"f_score": np.array([.8, .1, .4, .3]),
                       "optimal_f_score": np.array([.2, .7, 0., 1.]),
                       "global_score": np.array([.9, .2, 1., 0.]),
                       "angular_distance": np.array([.8, .2, 1., 0.])}
        self.distances = {"J": np.array([0., .25, .5, 1.]),
                          "I": np.array([1., .5, .25, 0.])}

    def test_exact_f_optimal_times_joint_formula(self):
        expected = self.values["optimal_f_score"] * np.exp(-2*self.distances["J"])
        actual = scoring.score_variant(self.values, self.distances, self.spec)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(self.spec, {"family": "J", "strength": 2., "local_mode": "optimal",
                                     "local_weight": 1., "global_weight": 0., "angular_gamma": 0.})

    def test_global_and_angle_changes_do_not_change_scores(self):
        before = scoring.score_variant(self.values, self.distances, self.spec)
        altered = deepcopy(self.values)
        altered["global_score"] = 1-altered["global_score"]
        altered["angular_distance"] = 1-altered["angular_distance"]
        after = scoring.score_variant(altered, self.distances, self.spec)
        np.testing.assert_array_equal(before, after)
        # The removed whole-image branch and angle must not affect ranking, either.
        np.testing.assert_array_equal(np.argsort(-before), np.argsort(-after))

    def test_greedy_and_independent_distances_cannot_leak_into_new_recipe(self):
        before = scoring.score_variant(self.values, self.distances, self.spec)
        altered = deepcopy(self.values)
        altered["f_score"] = np.zeros(4)
        d = {**self.distances, "I": np.zeros(4)}
        np.testing.assert_array_equal(before, scoring.score_variant(altered, d, self.spec))
        self.assertFalse(np.array_equal(before, self.values["f_score"]*np.exp(-2*self.distances["J"])))
        with self.assertRaisesRegex(ValueError, "Unknown or modified"):
            scoring.score_variant(self.values, self.distances, {**self.spec, "local_mode": "greedy"})

    def test_unit_interval_zero_and_one_boundaries(self):
        v = deepcopy(self.values)
        v["optimal_f_score"] = np.array([0., 1., 1., .5])
        d = {"J": np.array([0., 0., 1., 1.])}
        actual = scoring.score_variant(v, d, self.spec)
        np.testing.assert_array_equal(actual, np.array([0., 1., np.exp(-2), .5*np.exp(-2)]))
        self.assertTrue(np.all(actual >= 0))
        self.assertTrue(np.all(actual <= v["optimal_f_score"]))
        for bad in (np.nan, -.01, 1.01):
            invalid = deepcopy(v)
            invalid["optimal_f_score"][0] = bad
            with self.assertRaises(ValueError):
                scoring.score_variant(invalid, d, self.spec)

    def test_config_is_frozen_and_old_default_matrix_is_unchanged(self):
        config_path = Path(scoring.__file__).with_name("config_no_global_angle.json")
        self.assertEqual(scoring.load_settings(config_path), self.settings)
        self.assertEqual(self.settings["recipe"], "b1_no_global_angle_v1")
        self.assertEqual(self.settings["valid_variants"], ["no_global_angle"])
        self.assertEqual(self.settings["test_additional_variants"], [])
        self.assertEqual(set(scoring.specs("valid")),
                         {"no_local", "no_global", "no_angle", "greedy", "structure_only"})
        self.assertEqual(set(scoring.specs("test")), set(scoring.specs("valid")) | {"I_2"})
        for split in ("valid", "test"):
            self.assertEqual(set(scoring.specs(split, self.settings)), {"no_global_angle"})
        for key, changed in (("structure_strength", 1.), ("angular_gamma", 1.),
                             ("fusion_weight", .25), ("tie_seed", 7),
                             ("test_parameter_search", True),
                             ("valid_variants", ["no_global_angle", "no_global"])):
            with self.subTest(key=key):
                invalid = deepcopy(self.settings)
                invalid[key] = changed
                with self.assertRaises(ValueError):
                    scoring.validate_settings(invalid)
        with self.assertRaisesRegex(ValueError, "Unknown or modified"):
            scoring.score_variant(self.values, self.distances, {**self.spec, "strength": 1.})

    def test_pipeline_only_new_variant_in_each_split_and_independent_verification(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            output = Path(tmp)/"new_recipe"
            for split in ("valid", "test"):
                pool, bundle, descriptors, old_settings = fixture(split)
                settings = deepcopy(self.settings)
                settings["ks"] = old_settings["ks"]
                # Nonidentical structure distributions exercise the J multiplier end to end.
                descriptors["joint128"][1:, 0] = .5
                descriptors["joint128"][1:, 1] = .5
                scorer = Components()
                reports = pipeline.evaluate(output, "new_recipe_"+split, scorer, pool, bundle,
                                            settings["ks"], 2026, descriptors, settings, device="cpu")
                self.assertEqual(set(reports), {"no_global_angle"})
                self.assertEqual(len(scorer.calls), len(pool))
                self.assertEqual(scorer.calls[0][1], tuple(range(1, 12)))
                index = read_json(output/f"evaluation_index_{split}.json")
                self.assertEqual(set(index["variants"]), {"no_global_angle"})
                folder = output/split/"no_global_angle"
                result = verify.verify_evaluation(folder, pool, bundle, settings["ks"],
                                                  index["variant_signatures"]["no_global_angle"])
                self.assertEqual(result["status"], "verified")
                self.assertEqual(result["num_queries"], len(pool))
                manifest = read_json(folder/"evaluation_manifest.json")
                self.assertEqual(manifest["settings"]["recipe"], "b1_no_global_angle_v1")
                self.assertEqual(manifest["spec"], self.spec)
                with (folder/"rankings.csv").open() as handle:
                    for row in csv.DictReader(handle):
                        expected = float(row["optimal_f_score"])*np.exp(-2*float(row["joint_distance"]))
                        self.assertAlmostEqual(float(row["similarity_score"]), expected, places=14)
                before = snapshot(folder)
                with patch.object(Components, "components", side_effect=AssertionError("reuse completed new recipe")):
                    pipeline.evaluate(output, "new_recipe_"+split, Components(), pool, bundle,
                                      settings["ks"], 2026, descriptors, settings, device="cpu")
                self.assertEqual(before, snapshot(folder))
                with self.assertRaisesRegex(ValueError, "signature changed"):
                    pipeline.evaluate(output, "different_recipe_source", Components(), pool, bundle,
                                      settings["ks"], 2026, descriptors, settings, device="cpu")
            with (output/"comparison.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["split"] for row in rows}, {"valid", "test"})
            self.assertEqual({row["variant"] for row in rows}, {"no_global_angle"})

    def test_runner_keeps_wbbs_rows_separate_from_reference_directory(self):
        # Exercise main through both split loops; heavy computation is replaced,
        # but metadata, audit signatures and filesystem routing are real.
        from b1_ablation import analysis, register
        for recipe in ("no_global_angle", "components"):
            with self.subTest(recipe=recipe), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root/"artificial"/"source_b1"
                reference = root/"artificial"/"previous_components" if recipe == "no_global_angle" else None
                output = root/"artificial"/"b1_ablation_mock_runner"
                write_json(source/"run_manifest.json", {"mock_source": True})
                write_json(source/"selection.json", {"mock_selection": True})
                if reference is not None:
                    write_json(reference/"run_manifest.json", run.signed({"context": {"plan": {
                        "source_root": str(source), "settings": scoring.DEFAULT_SETTINGS}}}))
                    for split in ("valid", "test"):
                        write_json(reference/f"evaluation_index_{split}.json", {"mock_index": split})
                contexts, wbbs_rows = {}, {}
                for split in ("valid", "test"):
                    pool, bundle, descriptors, _ = fixture(split)
                    wbbs_rows[split] = [{"source": "WBBS", "split": split, "value": .5}]
                    contexts[split] = (pool, Components(), descriptors, wbbs_rows[split],
                                       {"reference": {"reports": {}}, "protocol_signature": "mock_protocol"})
                observed = {"evaluate": [], "analysis": [], "inventory": []}

                def evaluate_stub(out, signature, scorer, pool, bundle, ks, tie_seed,
                                  descriptor_values, settings, device="cuda", reference=None):
                    split = pool.split_name
                    self.assertIs(reference, wbbs_rows[split])
                    observed["evaluate"].append(split)
                    variants = scoring.specs(split, settings)
                    write_json(Path(out)/f"evaluation_index_{split}.json", {
                        "variant_signatures": {name: "mock_"+split+"_"+name for name in variants}})
                    return {}

                def analysis_sources_stub(out, old_source, split, provenance, settings=None, ref=None):
                    self.assertEqual(old_source, source)
                    self.assertEqual(ref, reference)
                    if recipe == "no_global_angle":
                        self.assertIsInstance(ref, Path)
                    else:
                        self.assertIsNone(ref)
                    observed["analysis"].append(split)
                    return {"mock_source": {"exact_dir": Path(out)/split}}

                def inventory_stub(old_source, ref=None):
                    self.assertEqual(old_source, source)
                    self.assertEqual(ref, reference)
                    if recipe == "no_global_angle":
                        self.assertIsInstance(ref, Path)
                    else:
                        self.assertIsNone(ref)
                    observed["inventory"].append(ref)
                    return {"mock_protected_inventory": "unchanged"}

                with ExitStack() as stack:
                    stack.enter_context(redirect_stdout(StringIO()))
                    stack.enter_context(patch.object(run, "ROOT", root))
                    stack.enter_context(patch.object(run, "load_inputs", return_value=(bundle, contexts)))
                    stack.enter_context(patch.object(run, "identity", return_value={}))
                    stack.enter_context(patch.object(run, "inventory", side_effect=inventory_stub))
                    stack.enter_context(patch.object(run, "analysis_sources", side_effect=analysis_sources_stub))
                    stack.enter_context(patch.object(run, "benchmark", return_value={"mock_timing": True}))
                    stack.enter_context(patch.object(run, "diagnostic_log", side_effect=lambda _: nullcontext()))
                    stack.enter_context(patch.object(run.subprocess, "check_output", return_value="mock_commit\n"))
                    stack.enter_context(patch.object(run.fusion, "checked_files", return_value=None))
                    stack.enter_context(patch.object(pipeline, "evaluate", side_effect=evaluate_stub))
                    stack.enter_context(patch.object(verify, "audit_source_sample", return_value={"status": "verified"}))
                    stack.enter_context(patch.object(verify, "verify_evaluation", return_value={"status": "verified"}))
                    stack.enter_context(patch.object(register, "proposal", return_value={"runs": []}))
                    registered = stack.enter_context(patch.object(register, "write_proposal", return_value=output/"catalog_sources.json"))
                    postprocessed = stack.enter_context(patch.object(analysis, "postprocess", return_value={}))
                    args = ["--recipe", recipe, "--source-root", str(source), "--output-root", str(output), "--device", "cpu"]
                    if reference is not None:
                        args += ["--reference-ablation-root", str(reference)]
                    self.assertEqual(run.main(args), 0)
                    registered.assert_called_once_with(output)
                    self.assertEqual(postprocessed.call_count, 2)
                    expected_anchor = "no_global_angle" if recipe == "no_global_angle" else "J_2"
                    self.assertEqual([call.kwargs["anchor"] for call in postprocessed.call_args_list],
                                     [expected_anchor, expected_anchor])
                    self.assertEqual([call.kwargs["split"] for call in postprocessed.call_args_list], ["valid", "test"])
                self.assertEqual(observed["evaluate"], ["valid", "test"])
                self.assertEqual(observed["analysis"], ["valid", "test"])
                self.assertEqual(observed["inventory"], [reference, reference])
                self.assertEqual(read_json(output/"status.json")["status"], "completed")
                self.assertEqual(read_json(output/"source_protection_after.json")["status"], "unchanged")

    def test_dry_run_selects_only_new_matrix_and_creates_no_output(self):
        output = run.ROOT/"artificial"/("b1_ablation_dry_no_global_angle_"+uuid4().hex)
        self.assertFalse(output.exists())
        with redirect_stdout(StringIO()) as console, patch.object(
                run, "load_inputs", side_effect=AssertionError("dry run must not read formal data")):
            exit_code = run.main(["--recipe", "no_global_angle", "--dry-run", "--output-root", str(output)])
        self.assertEqual(exit_code, 0)
        plan = json.loads(console.getvalue())
        self.assertEqual(plan["settings"]["recipe"], "b1_no_global_angle_v1")
        self.assertEqual(set(plan["valid"]), {"no_global_angle"})
        self.assertEqual(set(plan["test"]), {"no_global_angle"})
        self.assertFalse(plan["parameter_search"])
        self.assertFalse(plan["training_required"])
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
