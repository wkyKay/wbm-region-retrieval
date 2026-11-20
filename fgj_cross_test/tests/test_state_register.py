"""Synthetic completion receipts test gates and catalogue transactions.

These fixtures contain no real benchmark rankings. Metric/formula verification
is independently covered by the six-recipe verifier's integration tests.
"""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fgj_cross_test import state, register
from fgj_cross_test.verify import REPORT_FILES
from wbm_retrieval.common import digest, file_hash, read_json, write_json


CODE = {"synthetic_scoring_dependency.py": "a" * 64}


def reseal(folder, signature):
    (folder / "completed.json").unlink(missing_ok=True)
    state.seal(folder, signature)


def fixture(root):
    output = root / "artificial/fgj_cross_test_20261120T200000+0800"
    output.mkdir(parents=True)
    sources = {}
    for name in ("F_ONLY", "WBBS"):
        p = root / "artificial/synthetic_original" / name / "rankings.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("synthetic control receipt only\n")
        sources[name] = p
    binding = {str(p): file_hash(p) for p in sources.values()}
    derived = dict(protocol_version=state.PROTOCOL, candidate_rule=state.CANDIDATE_RULE,
        base_protocol_signature=state.BASE_PROTOCOL_SIGNATURE,
        candidate_identity_sha256=state.CANDIDATE_IDENTITY, filter_uses_ground_truth=True,
        filter_scope="benchmark eligibility only; method scores never receive labels",
        candidate_count_min=6999, candidate_count_max=7170)
    frozen = state.fixed_variants()
    context = dict(code=CODE, source_binding=binding, settings=deepcopy(state.SETTINGS),
        specs=deepcopy(state.SPECS), variants=list(state.VARIANTS), split="test",
        derived_protocol=derived, retrieval_protocol_signature=state.derived_signature(derived),
        primary_variant=None, selection_rule=state.SELECTION_RULE, test_parameter_search=False,
        formal_protocol=True, f_version="cc8_optimal_radial_area", structure_strength=2.0,
        training_required=False, posthoc_research=True, protocol=state.PROTOCOL,
        candidate_rule=state.CANDIDATE_RULE, frozen_variants_signature=frozen["signature"],
        normalization_scope="none")
    signature = digest(context)
    write_json(output / "run_manifest.json", state.signed(dict(version=state.VERSION,
        stage="test", split="test", context=context, started_at="2026-11-20T20:00:00+08:00",
        run_id="20261120T200000+0800__fgj_cross_test")))
    for name, data in (("frozen_variants.json", frozen), ("settings.json", state.SETTINGS),
                       ("candidate_protocol.json", derived), ("source_protection_before.json", {}),
                       ("source_protection_after.json", {"status": "unchanged", "files": 0})):
        write_json(output / name, data)
    audit_file = output / "protocol/cross_combination_candidates.csv"
    audit_file.parent.mkdir(); audit_file.write_text("synthetic candidate audit receipt\n")
    shared = dict(status="verified", split="test", context_signature=signature,
        ks=state.SETTINGS["ks"], tie_seed=2026, variants=state.VARIANTS,
        num_queries=7199, components_calls=7199, tie_key_computations=7199,
        full_candidates_checked=state.FULL_PAIRS, candidate_identity_sha256=state.CANDIDATE_IDENTITY,
        candidate_count_min=6999, candidate_count_max=7170,
        source_binding_sha256=digest(binding), source_file_count=len(binding))
    receipts, results, counts = {}, {}, state.expected_metric_counts()
    for name in state.VARIANTS:
        folder = state.evaluation_folder(output, name); folder.mkdir(parents=True)
        sig = state.variant_signature(signature, name)
        with patch.object(state, "now", return_value=datetime(2026, 11, 20, 20, 0, 0, tzinfo=state.TZ)):
            ident = state.identity(output, name, sig)
        ev = dict(**ident, context_signature=signature, signature=sig, spec=state.SPECS[name],
            settings=state.SETTINGS, stage="test", derived_protocol=derived, protocol=state.PROTOCOL,
            candidate_rule=state.CANDIDATE_RULE, retrieval_protocol_signature=state.derived_signature(derived),
            ks=state.SETTINGS["ks"], tie_seed=2026, num_queries=7199, gallery_count=7199,
            formal_protocol=True, structure_strength=2.0, f_version="cc8_optimal_radial_area",
            training_required=False, posthoc_research=True, test_parameter_search=False,
            selection_rule=state.SELECTION_RULE, primary_variant=None, normalization_scope="none",
            source_binding_signature=digest(binding))
        write_json(folder / "run_identity.json", ident)
        write_json(folder / "evaluation_manifest.json", ev)
        metrics = {key: .4 for key in counts}
        report = dict(split="test", num_queries=7199, ks=state.SETTINGS["ks"],
            metric_version="coverage_v1", metadata=ev, metrics=metrics,
            metric_query_counts=counts, queries_without_full_coverage_candidates=999)
        write_json(folder / "coverage_metrics.json", report)
        for filename in ("rankings.csv", "coverage_per_query.csv", "coverage_per_class.csv"):
            (folder / filename).write_text("synthetic metadata receipt only\n")
        files = {filename: file_hash(folder / filename) for filename in REPORT_FILES}
        checked = dict(**shared, signature=sig, variant=name, spec=state.SPECS[name],
            ranking_rows=7199*500, all_queries_full_candidate_topk_exact=True,
            independent_formulas=True, exact_reports=False,
            max_metric_absolute_error=0., max_saved_score_absolute_error=0.,
            max_saved_component_absolute_error=0., aggregation_values_checked=2000,
            metric_query_counts=counts, ranking_sha256=files["rankings.csv"], files=files)
        receipts[name] = checked
        write_json(folder / "verification.json", checked)
        write_json(folder / "timing.json", {"synthetic": True})
        state.seal(folder, sig)
        results[name] = dict(coverage_metrics=metrics, metric_query_counts=counts,
                             signature=sig, completed_sha256=file_hash(folder / "completed.json"))
    write_json(output / "shared_verification.json", dict(status="verified", shared=shared, variants=receipts))
    write_json(output / "evaluation_index_test.json", dict(context_signature=signature,
        split="test", variants=state.VARIANTS, specs=state.SPECS, results=results))
    write_json(output / "component_audit.json", dict(status="verified", context_signature=signature,
        split="test", query_ids=list(range(37)), max_candidates_per_query=20,
        full_global_cache_bitwise_identical=True,
        num_queries=37, candidate_pairs=740, full_global_candidates_checked=266326,
        max_global_absolute_error=1e-7, max_joint_distance_absolute_error=0., max_full_global_cache_error=0.))
    controls = {name: dict(num_queries=7199, top500_identical=True, source=str(source),
        source_sha256=file_hash(source), new_sha256=file_hash(output / "test" / name / "rankings.csv"),
        max_score_error=0.) for name, source in sources.items()}
    write_json(output / "baseline_replay.json", dict(status="verified", context_signature=signature, controls=controls))
    write_json(output / "status.json", dict(status="completed", context_signature=signature))
    state.seal(output, signature)
    return output, signature


class StateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "wbm-region-retrieval"
        self.output, self.signature = fixture(self.root)
        p = patch.object(state, "_current_code", return_value=CODE)
        p.start(); self.addCleanup(p.stop)

    def mutate_root(self, filename, change):
        path = self.output / filename
        value = read_json(path); change(value); write_json(path, value)
        reseal(self.output, self.signature)

    def test_fixed_choices_are_fresh_and_not_test_selected(self):
        frozen = state.fixed_variants()
        self.assertEqual(frozen["variants"], ["GJ_ADD", "FJ_ADD", "F_ONLY", "G_ONLY", "J_ONLY", "WBBS"])
        self.assertIsNone(frozen["primary_variant"])
        self.assertFalse(frozen["test_parameter_search"])
        frozen["specs"]["F_ONLY"]["formula"] = "changed"
        self.assertEqual(state.fixed_variants()["specs"]["F_ONLY"], {"formula": "F"})

    def test_identity_and_scientific_signature_are_persistent(self):
        name = "F_ONLY"; sig = state.variant_signature(self.signature, name)
        first = state.identity(self.output, name, sig)
        self.assertEqual(first, state.identity(self.output, name, sig))
        self.assertTrue(first["run_id"].startswith("20261120T200000+0800__"))
        with self.assertRaises(ValueError):
            state.identity(self.output, name, "different_scientific_signature")
        with self.assertRaises(ValueError):
            state.variant_signature(self.signature, "unexpected")

    def test_accepts_complete_six_role_receipts(self):
        manifest, frozen = state.verify_run(self.output)
        self.assertEqual(digest(manifest["context"]), self.signature)
        self.assertEqual(frozen, state.fixed_variants())

    def test_root_requires_context_signature_not_manifest_signature(self):
        manifest = state.read_signed(self.output / "run_manifest.json")
        reseal(self.output, manifest["signature"])
        with self.assertRaises(ValueError):
            state.verify_run(self.output)

    def test_rejects_root_identity_with_wrong_time(self):
        manifest = state.read_signed(self.output / "run_manifest.json")
        manifest.pop("signature")
        manifest["run_id"] = "20261120T200001+0800__fgj_cross_test"
        write_json(self.output / "run_manifest.json", state.signed(manifest))
        reseal(self.output, self.signature)
        with self.assertRaisesRegex(ValueError, "persisted start time"):
            state.verify_run(self.output)

    def test_rejects_repeated_component_query_audit(self):
        self.mutate_root("component_audit.json", lambda x: x.update(query_ids=[0] * 37))
        with self.assertRaisesRegex(ValueError, "source-component audit"):
            state.verify_run(self.output)

    def test_rejects_partial_index(self):
        self.mutate_root("evaluation_index_test.json", lambda x: x["results"].pop("WBBS"))
        with self.assertRaises(ValueError):
            state.verify_run(self.output)

    def test_rejects_changed_source_and_code(self):
        binding = state.read_signed(self.output / "run_manifest.json")["context"]["source_binding"]
        Path(next(iter(binding))).write_text("changed old source")
        with self.assertRaises(ValueError):
            state.verify_run(self.output)
        # No source read on this explicit mode; root uses it only after its own source check.
        state.verify_run(self.output, check_source=False)
        with patch.object(state, "_current_code", return_value={"changed.py": "b"*64}):
            with self.assertRaises(ValueError):
                state.verify_run(self.output, check_source=False)

    def test_rejects_shared_candidate_count_or_identity_change(self):
        for key, value in (("full_candidates_checked", 51818402), ("candidate_identity_sha256", "wrong")):
            with self.subTest(key=key):
                original = read_json(self.output / "shared_verification.json")
                self.mutate_root("shared_verification.json", lambda x: x["shared"].update({key:value}))
                with self.assertRaises(ValueError):
                    state.verify_run(self.output)
                write_json(self.output / "shared_verification.json", original); reseal(self.output, self.signature)

    def test_rejects_invalid_audit_and_historical_replay(self):
        original = read_json(self.output / "component_audit.json")
        self.mutate_root("component_audit.json", lambda x: x.update(max_full_global_cache_error=1e-9))
        with self.assertRaises(ValueError):
            state.verify_run(self.output)
        write_json(self.output / "component_audit.json", original); reseal(self.output, self.signature)
        self.mutate_root("baseline_replay.json", lambda x: x["controls"]["F_ONLY"].update(top500_identical=False))
        with self.assertRaises(ValueError):
            state.verify_run(self.output)

    def test_rejects_extra_exact_report_even_if_resealed(self):
        folder = self.output / "test/GJ_ADD"
        write_json(folder / "metrics.json", {"synthetic_exact": True})
        reseal(folder, state.variant_signature(self.signature, "GJ_ADD")); reseal(self.output, self.signature)
        with self.assertRaises(ValueError):
            state.verify_run(self.output)

    def test_rejects_modified_or_missing_report(self):
        (self.output / "test/GJ_ADD/coverage_metrics.json").write_text("{}")
        with self.assertRaises(ValueError):
            state.verify_run(self.output)

    def test_metric_counts_preserve_undefined_denominators(self):
        counts = state.expected_metric_counts()
        self.assertEqual(len(counts), 42)
        for k in state.SETTINGS["ks"]:
            self.assertEqual(counts[f"Coverage-nDCG@{k}"], 7000)
            self.assertEqual(counts[f"FullCoverage_mAP@{k}"], 6200)
            self.assertEqual(counts[f"FullCoverageRecall@{k}"], 6200)
            self.assertEqual(counts[f"MeanCoverage@{k}"], 7199)


class RegisterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name).resolve(); self.root = self.workspace / "wbm-region-retrieval"
        self.output, self.signature = fixture(self.root)
        p = patch.object(state, "_current_code", return_value=CODE); p.start(); self.addCleanup(p.stop)
        self.folder = self.root / "results_catalog"
        self.sources = {"schema_version": 1, "runs": [{"key": "old_run", "display": "Original"}]}
        self.old = {"key": "old_run", "metric": .123, "identity": {"run_id": "unchanged"}}
        self.previous = {"source_spec_sha256": self.fingerprint(self.sources), "runs": [self.old]}
        write_json(self.folder / "sources.json", self.sources)
        write_json(self.folder / "snapshot.json", self.previous)
        for name, content in self.render(self.previous).items():
            (self.folder / name).write_text(content)
        # Archives are independent lossless artifacts; raw transaction must not overwrite them.
        (self.folder / "snapshot.json.gz").write_bytes(b"historical archive sentinel")
        (self.folder / "metrics.csv.gz").write_bytes(b"historical table archive sentinel")

    @staticmethod
    def fingerprint(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def render(snapshot):
        return {"registry.csv": "key\n" + "\n".join(row["key"] for row in snapshot["runs"]) + "\n"}

    def collect(self, workspace, sources, previous):
        old = {row["key"]: row for row in previous["runs"]}
        return {"source_spec_sha256": self.fingerprint(sources),
                "runs": [deepcopy(old.get(row["key"], row)) for row in sources["runs"]]}

    def catalogue_bytes(self):
        return {p.name:p.read_bytes() for p in self.folder.iterdir() if p.is_file() and not p.name.startswith(".")}

    def test_proposal_has_exactly_six_correctly_classified_coverage_roles(self):
        proposed = register.proposal(self.output, root=self.root)
        self.assertEqual(len(proposed["runs"]), 6)
        self.assertEqual(len({r["key"] for r in proposed["runs"]}), 6)
        for name, row in zip(state.VARIANTS, proposed["runs"]):
            self.assertEqual(row["required_families"], ["coverage"])
            self.assertEqual(row["protocol"], state.PROTOCOL)
            self.assertEqual(row["category"], "cross_combination")
            self.assertFalse(row["valid_selected_primary"])
            self.assertEqual(row["selection_rule"], state.SELECTION_RULE)
            self.assertEqual(row["supervision"], "untrained" if name=="J_ONLY" else "none" if name=="WBBS" else "self_supervised")

    def test_atomic_addition_preserves_old_record_and_is_idempotent(self):
        before_archives = {p.name:p.read_bytes() for p in self.folder.glob("*.gz")}
        with patch.object(register.catalog, "render", side_effect=self.render), patch.object(register.catalog, "collect", side_effect=self.collect):
            result = register.apply_proposal(self.output, workspace=self.workspace, root=self.root)
            self.assertEqual(result["new_results"], 6)
            updated = read_json(self.folder / "snapshot.json")
            self.assertEqual(updated["runs"][0], self.old)
            self.assertEqual(len(updated["runs"]), 7)
            second = register.apply_proposal(self.output, workspace=self.workspace, root=self.root)
            self.assertEqual(second, {"status":"already_registered", "new_results":0})
        self.assertEqual({p.name:p.read_bytes() for p in self.folder.glob("*.gz")}, before_archives)

    def test_refuses_collector_mutation_of_prior_result(self):
        before = self.catalogue_bytes()
        def mutate(workspace, sources, previous):
            result = self.collect(workspace, sources, previous)
            result["runs"][0]["metric"] = .9
            return result
        with patch.object(register.catalog, "render", side_effect=self.render), patch.object(register.catalog, "collect", side_effect=mutate):
            with self.assertRaises(ValueError):
                register.apply_proposal(self.output, workspace=self.workspace, root=self.root)
        self.assertEqual(self.catalogue_bytes(), before)

    def test_refuses_collector_omission_or_unexpected_identity(self):
        for extra in (False, True):
            before = self.catalogue_bytes()
            def mutate(workspace, sources, previous):
                result = self.collect(workspace, sources, previous)
                if extra: result["runs"].append({"key":"unexpected"})
                else: result["runs"].pop()
                return result
            with patch.object(register.catalog, "render", side_effect=self.render), patch.object(register.catalog, "collect", side_effect=mutate):
                with self.assertRaises(ValueError):
                    register.apply_proposal(self.output, workspace=self.workspace, root=self.root)
            self.assertEqual(self.catalogue_bytes(), before)

    def test_failed_write_rolls_back_raw_catalogue(self):
        before = self.catalogue_bytes()
        original_write = register._atomic_write
        failed = [False]
        def fail_once(path, data):
            if Path(path) == self.folder / "registry.csv" and not failed[0]:
                failed[0] = True
                raise OSError("synthetic interrupted write")
            return original_write(path, data)
        with patch.object(register.catalog, "render", side_effect=self.render), patch.object(register.catalog, "collect", side_effect=self.collect), patch.object(register, "_atomic_write", side_effect=fail_once):
            with self.assertRaises(OSError):
                register.apply_proposal(self.output, workspace=self.workspace, root=self.root)
        self.assertEqual(self.catalogue_bytes(), before)

    def test_inconsistent_prior_source_snapshot_keys_are_rejected(self):
        previous = read_json(self.folder / "snapshot.json")
        previous["runs"][0]["key"] = "different_old_key"
        write_json(self.folder / "snapshot.json", previous)
        for name, content in self.render(previous).items(): (self.folder/name).write_text(content)
        with patch.object(register.catalog, "render", side_effect=self.render):
            with self.assertRaises(ValueError):
                register.apply_proposal(self.output, workspace=self.workspace, root=self.root)


if __name__ == "__main__":
    unittest.main()
