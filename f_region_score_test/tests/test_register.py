"""Synthetic frozen-valid-source receipts and one-test registration transactions."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from f_region_score_search import selection as valid_selection
from f_region_score_search.tests.test_register import fixture as valid_fixture
from f_region_score_test import register
from f_region_structure import state
from f_region_structure.verify import REPORT_FILES
from wbm_retrieval.common import digest, file_hash, read_json, write_json


def reseal(folder, signature):
    (folder / "completed.json").unlink(missing_ok=True)
    state.seal(folder, signature)


def fixture(root):
    """Small bound files model receipts, not generated formal rankings or results."""
    source, source_signature = valid_fixture(root)
    selected_folder = source / "valid" / register.NAME
    metrics = read_json(selected_folder / "metrics.json")
    metrics["metrics"]["Precision@10"] = .6
    write_json(selected_folder / "metrics.json", metrics)
    checked = read_json(selected_folder / "verification.json")
    checked["top10_hits"] = 3000
    checked["files"]["metrics.json"] = file_hash(selected_folder / "metrics.json")
    write_json(selected_folder / "verification.json", checked)
    reseal(selected_folder, read_json(selected_folder / "evaluation_manifest.json")["signature"])
    index = read_json(source / "evaluation_index_valid.json")
    index["results"][register.NAME].update(top10_hits=3000, metrics=metrics["metrics"],
        completed_sha256=file_hash(selected_folder / "completed.json"))
    write_json(source / "evaluation_index_valid.json", index)
    old_selection = read_json(source / "selection.json")
    old_selection.pop("signature")
    old_selection.update(valid_selection.select_best(index["results"]))
    old_selection["valid_index_sha256"] = file_hash(source / "evaluation_index_valid.json")
    selected = state.signed(old_selection)
    write_json(source / "selection.json", selected)
    status = read_json(source / "status.json")
    status["winner"] = register.NAME
    write_json(source / "status.json", status)
    reseal(source, source_signature)

    output = root / "artificial/f_region_score_test_20261120T160000+0800"
    output.mkdir(parents=True)
    binding = {str(source / name): file_hash(source / name)
               for name in ("selection.json", "completed.json", "evaluation_index_valid.json")}
    frozen = state.signed({"schema_version": 1, "source_run": str(source),
        "source_selection_sha256": file_hash(source / "selection.json"),
        "source_completed_sha256": file_hash(source / "completed.json"),
        "source_valid_index_sha256": file_hash(source / "evaluation_index_valid.json"),
        "selection": selected, "winner": register.NAME, "selected_spec": register.SPEC,
        "selection_policy": register.POLICY})
    write_json(output / "frozen_selection.json", frozen)
    cache_source = str(root / "artificial/old_regional")
    context = {"version": "f_region_score_test_v1", "split": "test", "source_binding": binding,
               "frozen_variant": register.NAME, "frozen_spec": register.SPEC,
               "selection_policy": register.POLICY, "posthoc_design": True, "test_parameter_search": False,
               "training": False, "frozen_selection_sha256": file_hash(output / "frozen_selection.json"),
               "source_run": str(source), "cache_source_run": cache_source, "code": {"fixture.py": "a" * 64}}
    signature = digest(context)
    write_json(output / "run_manifest.json", state.signed({"version": "f_region_score_test_v1", "context": context}))
    write_json(output / "inputs.json", {"source_binding": binding, "context_signature": signature,
               "source_run": str(source), "cache_source_run": cache_source})
    write_json(output / "source_protection_before.json", {})
    write_json(output / "source_protection_after.json", {"status": "unchanged", "files": 0})
    folder = output / "test" / register.NAME
    folder.mkdir(parents=True)
    science = digest({"context": signature, "split": "test", "variant": register.NAME, "spec": register.SPEC})
    key = "f_region_score_test_fixture_test_" + register.NAME
    identity = {"schema_version": 1, "key": key, "split": "test", "variant": register.NAME,
                "started_at": "2026-11-20T16:00:00+08:00", "run_id": "20261120T160000+0800__" + key,
                "evaluation_signature": science}
    manifest = {**identity, "signature": science, "spec": register.SPEC, "ks": register.KS, "tie_seed": 2026,
                "retrieval_protocol_signature": register.catalog.BASE_PROTOCOL_SIGNATURE,
                "num_queries": 7199, "gallery_count": 7199, "candidate_rule": "all_gallery_except_query",
                "checkpoint_kind": "best_nce", "source_seed": 2026, "matching": "original_F_greedy_sqrt_area",
                "posthoc_research": True, "test_parameter_search": False, "test_evaluated": True,
                "training_required": False, "selection_split": "valid"}
    write_json(folder / "run_identity.json", identity)
    write_json(folder / "evaluation_manifest.json", manifest)
    metrics = {f"{metric}@{k}": .1 for metric in ("Hit", "Precision", "Recall", "mAP") for k in register.KS}
    metrics["Precision@10"] = 40000 / 71990
    coverage = {f"{metric}@{k}": .8 for metric in ("Coverage-nDCG", "MeanCoverage", "FullCoverageHit",
                "FullCoveragePrecision", "FullCoverageRecall", "FullCoverage_mAP") for k in register.KS}
    for family, filename, values in (("exact", "metrics.json", metrics), ("coverage", "coverage_metrics.json", coverage)):
        write_json(folder / filename, {"split": "test", "num_queries": 7199, "ks": register.KS,
            "metric_version": family + "_v1", "test_evaluated": True, "metadata": manifest,
            "metrics": values, "retrieval_protocol_signature": register.catalog.BASE_PROTOCOL_SIGNATURE})
    for filename in ("rankings.csv", "per_query.csv", "per_class.csv", "coverage_per_query.csv", "coverage_per_class.csv"):
        (folder / filename).write_text("synthetic_metadata_fixture\n")
    files = {filename: file_hash(folder / filename) for filename in (*REPORT_FILES, "run_identity.json")}
    checked = {"status": "verified", "signature": science, "split": "test", "variant": register.NAME,
               "ks": register.KS, "tie_seed": 2026, "num_queries": 7199, "ranking_rows": 3599500,
               "full_candidates_checked": 51818402, "top10_hits": 40000,
               "all_queries_full_candidate_topk_exact": True, "max_metric_absolute_error": 0.,
               "max_saved_score_absolute_error": 0., "aggregation_values_checked": 1000,
               "ranking_sha256": files["rankings.csv"], "files": files}
    write_json(folder / "verification.json", checked)
    state.seal(folder, science)
    result = {"top10_hits": 40000, "num_queries": 7199, "metrics": metrics, "coverage_metrics": coverage,
              "completed_sha256": file_hash(folder / "completed.json")}
    write_json(output / "evaluation_index_test.json", {"split": "test", "specs": {register.NAME: register.SPEC},
               "results": {register.NAME: result}, "context_signature": signature})
    write_json(output / "formula_audit_test.json", {"status": "verified", "split": "test", "variant": register.NAME,
               "context_signature": signature, "query_ids": list(range(37)), "candidate_pairs": 740, "max_absolute_error": 0.})
    write_json(output / "status.json", {"status": "completed", "split": "test", "winner": register.NAME,
               "test_evaluated": True, "context_signature": signature})
    state.seal(output, signature)
    return output, signature


class RegisterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name).resolve()
        self.root = self.workspace / "wbm-region-retrieval"
        self.output, self.signature = fixture(self.root)
        self.folder = self.root / "results_catalog"
        self.sources = {"schema_version": 1, "runs": [{"key": f"old_{i}"} for i in range(363)]}
        self.old = [{"key": f"old_{i}", "metric": i / 363, "identity": {"run_id": f"original_{i}"}} for i in range(363)]
        self.previous = {"source_spec_sha256": register._fingerprint(self.sources), "runs": self.old}
        write_json(self.folder / "sources.json", self.sources)
        write_json(self.folder / "snapshot.json", self.previous)
        for name, content in self.render(self.previous).items():
            (self.folder / name).write_text(content)

    @staticmethod
    def render(snapshot):
        return {"registry.csv": "key\n" + "\n".join(row["key"] for row in snapshot["runs"]) + "\n"}

    def snapshot(self):
        return {p.name: p.read_bytes() for p in self.folder.iterdir() if p.is_file() and not p.name.startswith(".")}

    def collect(self, workspace, sources, previous):
        return {"source_spec_sha256": register._fingerprint(sources),
                "runs": [*self.old, *[dict(row) for row in sources["runs"] if not row["key"].startswith("old_")]]}

    def reseal(self):
        reseal(self.output, self.signature)

    def reseal_eval(self):
        folder = self.output / "test" / register.NAME
        reseal(folder, read_json(folder / "evaluation_manifest.json")["signature"])
        path = self.output / "evaluation_index_test.json"
        index = read_json(path)
        index["results"][register.NAME]["completed_sha256"] = file_hash(folder / "completed.json")
        write_json(path, index)
        self.reseal()

    def test_unique_frozen_test_proposal_and_idempotence(self):
        before = self.snapshot()
        path = register.write_proposal(self.output, root=self.root)
        proposal = read_json(path)
        self.assertEqual(len(proposal["runs"]), 1)
        row = proposal["runs"][0]
        self.assertEqual(row["split"], "test")
        self.assertEqual(row["variant"], "RegionalJScore/" + register.NAME)
        self.assertEqual(row["supervision"], "self_supervised")
        self.assertTrue(row["selection"].endswith("frozen_selection.json"))
        stamp = path.stat().st_mtime_ns
        register.write_proposal(self.output, root=self.root)
        self.assertEqual(path.stat().st_mtime_ns, stamp)
        self.assertEqual(before, self.snapshot())

    def test_incomplete_run_rejected(self):
        (self.output / "completed.json").unlink()
        with self.assertRaisesRegex(ValueError, "completed"):
            register.proposal(self.output, root=self.root)

    def test_other_test_variant_rejected(self):
        (self.output / "test/extra_variant").mkdir()
        self.reseal()
        with self.assertRaisesRegex(ValueError, "extra"):
            register.proposal(self.output, root=self.root)

    def test_new_selection_or_training_rejected(self):
        for name in ("valid", "training"):
            with self.subTest(name=name):
                (self.output / name).mkdir()
                self.reseal()
                with self.assertRaisesRegex(ValueError, "Unexpected selection"):
                    register.proposal(self.output, root=self.root)
                (self.output / name).rmdir()
                self.reseal()

    def test_full_candidate_count_required(self):
        path = self.output / "test" / register.NAME / "verification.json"
        value = read_json(path)
        value["full_candidates_checked"] -= 1
        write_json(path, value)
        self.reseal_eval()
        with self.assertRaisesRegex(ValueError, "Full-test"):
            register.proposal(self.output, root=self.root)

    def test_valid_completed_sha_change_rejected(self):
        source = Path(read_json(self.output / "frozen_selection.json")["source_run"])
        done = read_json(source / "completed.json")
        done["signature"] = "f" * 64
        write_json(source / "completed.json", done)
        with self.assertRaisesRegex(ValueError, "input source"):
            register.proposal(self.output, root=self.root)

    def test_altered_frozen_wrapper_rejected(self):
        path = self.output / "frozen_selection.json"
        wrapper = read_json(path)
        wrapper.pop("signature")
        wrapper["winner"] = "j_only"
        write_json(path, state.signed(wrapper))
        self.reseal()
        with self.assertRaisesRegex(ValueError, "selection wrapper"):
            register.proposal(self.output, root=self.root)

    def test_missing_metric_rejected_after_rebinding_receipt(self):
        folder = self.output / "test" / register.NAME
        path = folder / "metrics.json"
        report = read_json(path)
        del report["metrics"]["Recall@1"]
        write_json(path, report)
        checked = read_json(folder / "verification.json")
        checked["files"]["metrics.json"] = file_hash(path)
        write_json(folder / "verification.json", checked)
        self.reseal_eval()
        with self.assertRaisesRegex(ValueError, "metric grid"):
            register.proposal(self.output, root=self.root)

    def test_protection_receipt_required(self):
        write_json(self.output / "source_protection_after.json", {"status": "changed", "files": 0})
        self.reseal()
        with self.assertRaisesRegex(ValueError, "protection"):
            register.proposal(self.output, root=self.root)

    def test_transaction_adds_one_preserves_363_and_idempotent(self):
        with patch.object(register.catalog, "collect", side_effect=self.collect), patch.object(register.catalog, "render", side_effect=self.render):
            result = register.apply_proposal(self.output, root=self.root)
            self.assertEqual(result["new_results"], 1)
            self.assertEqual(result["old_results_unchanged"], 363)
            self.assertEqual(read_json(self.folder / "snapshot.json")["runs"][:363], self.old)
            self.assertEqual(register.apply_proposal(self.output, root=self.root)["status"], "already_registered")
        self.assertTrue(state.verified(self.output, self.signature))

    def test_existing_result_cannot_change(self):
        before = self.snapshot()
        def bad_collect(*args):
            value = self.collect(*args)
            value["runs"][20] = {**self.old[20], "metric": .999}
            return value
        with patch.object(register.catalog, "collect", side_effect=bad_collect), patch.object(register.catalog, "render", side_effect=self.render):
            with self.assertRaisesRegex(ValueError, "previously registered"):
                register.apply_proposal(self.output, root=self.root)
        self.assertEqual(before, self.snapshot())

    def test_write_failure_rolls_back(self):
        before = self.snapshot()
        atomic, failed = register._atomic_write, False
        def fail_once(path, content):
            nonlocal failed
            if Path(path).name == "registry.csv" and not failed:
                failed = True
                raise OSError("injected failure")
            atomic(path, content)
        with patch.object(register.catalog, "collect", side_effect=self.collect), patch.object(register.catalog, "render", side_effect=self.render), patch.object(register, "_atomic_write", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "injected"):
                register.apply_proposal(self.output, root=self.root)
        self.assertEqual(before, self.snapshot())
        receipts = list((self.output / "catalog_backup").glob("*/manifest.json"))
        self.assertEqual(len(receipts), 1)
        self.assertEqual(read_json(receipts[0])["status"], "rolled_back")


if __name__ == "__main__":
    unittest.main()
