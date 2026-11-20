"""Synthetic receipts test registration gates and transactional preservation.

Fixtures deliberately contain no real rankings or experimental results. The
independent verifier has separate real dual-evaluator integration tests.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gj_fusion import register, run, scoring, selection, state
from gj_fusion.verify import REPORT_FILES
from wbm_retrieval.common import digest, file_hash, read_json, write_json


CODE = {"synthetic_scientific_dependency.py": "a"*64}


def reseal(folder, signature):
    (folder/"completed.json").unlink(missing_ok=True)
    state.seal(folder, signature)


def fixture(root):
    output = root/"artificial/gj_fusion_valid_20261120T180000+0800"
    output.mkdir(parents=True)
    source = root/"artificial/synthetic_original_source.json"
    write_json(source, {"synthetic": True})
    control_paths = {}
    binding = {str(source): file_hash(source)}
    for name in ("G_only", "J_only", "product"):
        path = root/"artificial/synthetic_control"/name/"rankings.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic metadata receipt only\n")
        control_paths[name] = path
        binding[str(path)] = file_hash(path)
    context = {"version": register.VERSION, "split": "valid", "code": CODE,
               "grid": scoring.frozen_grid(), "full_grid": scoring.frozen_grid(),
               "source_binding": binding, "selection_rule": selection.RULE,
               "structure_strength": 2., "angular_component": False, "local_component": False,
               "training_required": False, "posthoc_research": True, "test_parameter_search": False}
    signature = digest(context)
    manifest = state.signed({"version": register.VERSION, "split": "valid", "context": context,
                            "source_run": str(source), "started_at": "2026-11-20T18:00:00+08:00"})
    write_json(output/"run_manifest.json", manifest)
    write_json(output/"inputs.json", {"source_binding": context["source_binding"]})
    write_json(output/"source_protection_after.json", {"status": "unchanged"})
    results = {}
    for i, (name, spec) in enumerate(scoring.frozen_grid().items()):
        folder = output/"valid"/name
        folder.mkdir(parents=True)
        sig = register.variant_signature(signature, "valid", name, spec)
        key = "gj_fusion_fixture_valid_"+name
        ident = {"schema_version": 1, "key": key, "split": "valid", "variant": name,
                 "evaluation_signature": sig, "started_at": "2026-11-20T18:00:00+08:00",
                 "run_id": "20261120T180000+0800__"+key}
        evaluation = {**ident, "signature": sig, "context_signature": signature, "spec": spec,
                      "ks": register.KS, "tie_seed": 2026, "num_queries": 500, "gallery_count": 3599,
                      "candidate_rule": "all_gallery_except_query",
                      "retrieval_protocol_signature": register.catalog.BASE_PROTOCOL_SIGNATURE,
                      "structure_strength": 2., "angular_component": False, "local_component": False,
                      "training_required": False, "test_parameter_search": False, "posthoc_research": True,
                      "test_evaluated": False, "source_seed": 2026, "checkpoint_kind": "best_nce"}
        write_json(folder/"run_identity.json", ident)
        write_json(folder/"evaluation_manifest.json", evaluation)
        hits = 2500+i
        exact = {f"{metric}@{k}": .1 for metric in ("Hit", "Precision", "Recall", "mAP") for k in register.KS}
        exact["Precision@10"] = hits/5000
        coverage = {f"{metric}@{k}": .8 for metric in ("Coverage-nDCG", "MeanCoverage", "FullCoverageHit",
                    "FullCoveragePrecision", "FullCoverageRecall", "FullCoverage_mAP") for k in register.KS}
        for family, filename, metrics in (("exact", "metrics.json", exact), ("coverage", "coverage_metrics.json", coverage)):
            write_json(folder/filename, {"split": "valid", "num_queries": 500, "ks": register.KS,
                       "metric_version": family+"_v1", "metrics": metrics, "metadata": evaluation,
                       "test_evaluated": False, "retrieval_protocol_signature": register.catalog.BASE_PROTOCOL_SIGNATURE})
        for filename in ("rankings.csv", "per_query.csv", "per_class.csv", "coverage_per_query.csv", "coverage_per_class.csv"):
            (folder/filename).write_text("synthetic metadata receipt only\n")
        files = {name: file_hash(folder/name) for name in (*REPORT_FILES, "run_identity.json")}
        checked = {"status": "verified", "signature": sig, "split": "valid", "variant": name,
                   "ks": register.KS, "tie_seed": 2026, "num_queries": 500, "ranking_rows": 250000,
                   "full_candidates_checked": 500*3598, "all_queries_full_candidate_topk_exact": True,
                   "independent_fusion_all_candidates": True, "top10_hits": hits,
                   "max_metric_absolute_error": 0., "max_saved_score_absolute_error": 0.,
                   "max_component_absolute_error": 0., "aggregation_values_checked": 3150,
                   "ranking_sha256": files["rankings.csv"], "files": files}
        write_json(folder/"verification.json", checked)
        state.seal(folder, sig)
        results[name] = {"split": "valid", "num_queries": 500, "top10_hits": hits, "spec": spec,
                         "signature": sig, "completed_sha256": file_hash(folder/"completed.json"),
                         "metrics": exact, "coverage_metrics": coverage}
    write_json(output/"evaluation_index_valid.json", {"context_signature": signature, "split": "valid",
               "variants": scoring.frozen_grid(), "results": results})
    frozen = state.signed({**selection.select_candidates(results), "context_signature": signature,
              "valid_index_sha256": file_hash(output/"evaluation_index_valid.json"),
              "frozen_at": "2026-11-20T18:10:00+08:00"})
    write_json(output/"selection.json", frozen)
    controls = {}
    for name in ("G_only", "J_only", "product"):
        path = output/"valid"/name/"rankings.csv"
        controls[name] = {"num_queries": 500, "top500_identical": True, "source": str(control_paths[name]),
                          "source_sha256": file_hash(control_paths[name]), "new_sha256": file_hash(path),
                          "max_score_error": None if name == "J_only" else 0.}
    write_json(output/"baseline_replay.json", {"status": "verified", "split": "valid", "controls": controls})
    write_json(output/"component_audit_valid.json", {"status": "verified", "split": "valid",
               "context_signature": signature, "query_ids": list(range(37)), "num_queries": 37,
               "candidate_pairs": 740, "max_candidates_per_query": 20,
               "max_global_absolute_error": 0., "max_joint_distance_absolute_error": 0.,
               "global_tolerance": 2e-7, "joint_distance_tolerance": 1e-12,
               "candidate_ids": {str(i): list(range(40, 60)) for i in range(37)}})
    write_json(output/"status.json", {"status": "completed", "context_signature": signature, "test_evaluated": False})
    state.seal(output, signature)
    return output, signature


class RegisterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name).resolve()
        self.root = self.workspace/"wbm-region-retrieval"
        self.output, self.signature = fixture(self.root)
        for module in (register, run):
            patcher = patch.object(module, "code_identity", return_value=CODE)
            patcher.start(); self.addCleanup(patcher.stop)
        self.folder = self.root/"results_catalog"
        self.sources = {"schema_version": 1, "runs": [{"key": "old_run", "display": "Original"}]}
        self.old = {"key": "old_run", "metric": .123, "identity": {"run_id": "unchanged"}}
        self.previous = {"source_spec_sha256": self.fingerprint(self.sources), "runs": [self.old]}
        write_json(self.folder/"sources.json", self.sources)
        write_json(self.folder/"snapshot.json", self.previous)
        for name, content in self.render(self.previous).items():
            (self.folder/name).write_text(content)

    @staticmethod
    def fingerprint(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def render(snapshot):
        return {"registry.csv": "key\n"+"\n".join(row["key"] for row in snapshot["runs"])+"\n"}

    def snapshot(self):
        return {p.name: p.read_bytes() for p in self.folder.iterdir() if p.is_file() and not p.name.startswith(".")}

    def collect(self, workspace, sources, previous):
        old = {row["key"]: row for row in previous["runs"]}
        return {"source_spec_sha256": self.fingerprint(sources),
                "runs": [deepcopy(old.get(row["key"], row)) for row in sources["runs"]]}

    def reseal_evaluation(self, name):
        folder = self.output/"valid"/name
        reseal(folder, read_json(folder/"evaluation_manifest.json")["signature"])
        index = read_json(self.output/"evaluation_index_valid.json")
        index["results"][name]["completed_sha256"] = file_hash(folder/"completed.json")
        write_json(self.output/"evaluation_index_valid.json", index)
        frozen = read_json(self.output/"selection.json")
        frozen.pop("signature")
        frozen["valid_index_sha256"] = file_hash(self.output/"evaluation_index_valid.json")
        write_json(self.output/"selection.json", state.signed(frozen))
        reseal(self.output, self.signature)

    def test_proposal_is_read_only_and_has_eight_unique_identities(self):
        before = self.snapshot()
        proposed = register.proposal(self.output, root=self.root)
        self.assertEqual(len(proposed["runs"]), 8)
        self.assertEqual(len({row["key"] for row in proposed["runs"]}), 8)
        self.assertEqual(next(r for r in proposed["runs"] if r["variant"] == "GJ/J_only")["supervision"], "untrained")
        self.assertEqual(self.snapshot(), before)

    def test_missing_root_completion_rejected(self):
        (self.output/"completed.json").unlink()
        with self.assertRaisesRegex(ValueError, "completed"):
            register.proposal(self.output, root=self.root)

    def test_changed_completed_report_rejected(self):
        path = self.output/"valid/J_only/metrics.json"
        value = read_json(path)
        value["metrics"]["Precision@10"] = .99
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, "Changed|changed|hash"):
            register.proposal(self.output, root=self.root)

    def test_signed_selection_change_rejected_even_after_root_reseal(self):
        path = self.output/"selection.json"
        value = read_json(path)
        value.pop("signature")
        value["winner"] = "J_only"
        write_json(path, state.signed(value))
        reseal(self.output, self.signature)
        with self.assertRaisesRegex(ValueError, "Selection differs"):
            register.proposal(self.output, root=self.root)

    def test_independent_fusion_receipt_is_required(self):
        path = self.output/"valid/product/verification.json"
        checked = read_json(path)
        del checked["independent_fusion_all_candidates"]
        write_json(path, checked)
        self.reseal_evaluation("product")
        with self.assertRaises(ValueError):
            register.proposal(self.output, root=self.root)

    def test_verified_report_hash_set_cannot_be_empty(self):
        path = self.output/"valid/product/verification.json"
        checked = read_json(path)
        checked["files"] = {}
        write_json(path, checked)
        self.reseal_evaluation("product")
        with self.assertRaisesRegex(ValueError, "verification is incomplete"):
            register.proposal(self.output, root=self.root)

    def test_component_audit_cannot_be_omitted(self):
        (self.output/"component_audit_valid.json").unlink()
        reseal(self.output, self.signature)
        with self.assertRaises((ValueError, FileNotFoundError)):
            register.proposal(self.output, root=self.root)

    def test_component_audit_count_and_error_are_binding(self):
        path = self.output/"component_audit_valid.json"
        original = read_json(path)
        for key, value in (("candidate_pairs", 739), ("num_queries", 36),
                           ("max_global_absolute_error", 2.1e-7),
                           ("max_joint_distance_absolute_error", 1.1e-12)):
            write_json(path, dict(original, **{key: value}))
            reseal(self.output, self.signature)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "component audit"):
                register.proposal(self.output, root=self.root)

    def test_failed_control_rank_replay_is_rejected(self):
        path = self.output/"baseline_replay.json"
        replay = read_json(path)
        replay["controls"]["product"]["top500_identical"] = False
        write_json(path, replay)
        reseal(self.output, self.signature)
        with self.assertRaises(ValueError):
            register.proposal(self.output, root=self.root)

    def test_upstream_source_mutation_is_rejected(self):
        source = self.root/"artificial/synthetic_original_source.json"
        write_json(source, {"synthetic": "changed after completion"})
        with self.assertRaises(ValueError):
            register.proposal(self.output, root=self.root)

    def test_transaction_preserves_existing_records_and_is_idempotent(self):
        with patch.object(register.catalog, "collect", side_effect=self.collect), patch.object(register.catalog, "render", side_effect=self.render):
            result = register.apply_proposal(self.output, root=self.root)
            self.assertEqual(result["new_results"], 8)
            self.assertEqual(read_json(self.folder/"snapshot.json")["runs"][0], self.old)
            before = self.snapshot()
            self.assertEqual(register.apply_proposal(self.output, root=self.root)["status"], "already_registered")
            self.assertEqual(self.snapshot(), before)

    def test_collection_cannot_silently_change_old_records(self):
        before = self.snapshot()
        def corrupt(*args):
            value = self.collect(*args)
            value["runs"][0]["metric"] = .999
            return value
        with patch.object(register.catalog, "collect", side_effect=corrupt), patch.object(register.catalog, "render", side_effect=self.render):
            with self.assertRaises(ValueError):
                register.apply_proposal(self.output, root=self.root)
        self.assertEqual(self.snapshot(), before)

    def test_write_failure_rolls_back_all_shared_files(self):
        before = self.snapshot()
        original = register._atomic_write
        failed = False
        def fail_once(path, data):
            nonlocal failed
            if path == self.folder/"registry.csv" and not failed:
                failed = True
                raise OSError("injected catalogue write failure")
            return original(path, data)
        with patch.object(register.catalog, "collect", side_effect=self.collect), patch.object(register.catalog, "render", side_effect=self.render), patch.object(register, "_atomic_write", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "injected catalogue"):
                register.apply_proposal(self.output, root=self.root)
        self.assertEqual(self.snapshot(), before)
        backups = list((self.output/"catalog_backup").glob("*/manifest.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(read_json(backups[0])["status"], "rolled_back")


if __name__ == "__main__":
    unittest.main()
