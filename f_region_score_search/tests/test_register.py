"""Synthetic registration receipts and temporary catalogue transactions only."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from f_region_score_search import register, scoring, selection
from f_region_structure import state
from f_region_structure.verify import REPORT_FILES
from wbm_retrieval.common import digest, file_hash, read_json, write_json


def reseal(folder, signature):
    (folder / "completed.json").unlink(missing_ok=True)
    state.seal(folder, signature)


def fixture(root):
    """Small bound files model verification receipts; numerical QA has separate tests."""
    output = root / "artificial/f_region_score_search_20261120T150000+0800"
    output.mkdir(parents=True)
    old = root / "artificial/old_regional/valid/replace_radial_all_points_l0p5/rankings.csv"
    old.parent.mkdir(parents=True)
    old.write_text("synthetic_metadata_fixture\n")
    binding = {str(old): file_hash(old)}
    context = {"version": "f_region_score_search_v1", "split": "valid", "grid": scoring.frozen_grid(),
               "selection_rule": selection.RULE, "posthoc_design": True, "test_evaluated": False,
               "training": False, "source_binding": binding, "code": {"testfixture.py": "a" * 64}}
    signature = digest(context)
    write_json(output / "run_manifest.json", state.signed({"version": "f_region_score_search_v1", "context": context}))
    write_json(output / "inputs.json", {"source_binding": binding, "context_signature": signature,
               "source_run": str(old.parents[2]), "baseline_folder": str(old.parent)})
    write_json(output / "source_protection_before.json", {})
    write_json(output / "source_protection_after.json", {"status": "unchanged", "files": 0})
    write_json(output / "baseline_replay.json", {"status": "verified", "rows": 250000, "max_score_error": 0.,
               "source": str(old), "source_sha256": file_hash(old), "ranking_byte_identical": True})
    results = {}
    for i, (name, spec) in enumerate(scoring.frozen_grid().items()):
        folder = output / "valid" / name
        folder.mkdir(parents=True)
        science = digest({"context": signature, "split": "valid", "variant": name, "spec": spec})
        key = "f_region_score_search_fixture_valid_" + name
        identity = {"schema_version": 1, "key": key, "split": "valid", "variant": name,
                    "started_at": "2026-11-20T15:00:00+08:00", "run_id": "20261120T150000+0800__" + key,
                    "evaluation_signature": science}
        manifest = {**identity, "signature": science, "spec": spec, "ks": register.KS, "tie_seed": 2026,
                    "retrieval_protocol_signature": register.catalog.BASE_PROTOCOL_SIGNATURE,
                    "num_queries": 500, "gallery_count": 3599, "candidate_rule": "all_gallery_except_query",
                    "checkpoint_kind": "best_nce", "source_seed": 2026, "matching": "original_F_greedy_sqrt_area",
                    "posthoc_research": True, "test_parameter_search": False, "test_evaluated": False,
                    "training_required": False}
        write_json(folder / "run_identity.json", identity)
        write_json(folder / "evaluation_manifest.json", manifest)
        hits = 2494 if name == "product" else 2500 + i
        metrics = {f"{metric}@{k}": .1 for metric in ("Hit", "Precision", "Recall", "mAP") for k in register.KS}
        metrics["Precision@10"] = hits / 5000
        coverage = {f"{metric}@{k}": .8 for metric in ("Coverage-nDCG", "MeanCoverage", "FullCoverageHit",
                    "FullCoveragePrecision", "FullCoverageRecall", "FullCoverage_mAP") for k in register.KS}
        for family, filename, values in (("exact", "metrics.json", metrics), ("coverage", "coverage_metrics.json", coverage)):
            write_json(folder / filename, {"split": "valid", "num_queries": 500, "ks": register.KS,
                "metric_version": family + "_v1", "test_evaluated": False, "metadata": manifest,
                "metrics": values, "retrieval_protocol_signature": register.catalog.BASE_PROTOCOL_SIGNATURE})
        for filename in ("rankings.csv", "per_query.csv", "per_class.csv", "coverage_per_query.csv", "coverage_per_class.csv"):
            (folder / filename).write_text("synthetic_metadata_fixture\n")
        files = {filename: file_hash(folder / filename) for filename in (*REPORT_FILES, "run_identity.json")}
        verification = {"status": "verified", "signature": science, "split": "valid", "variant": name,
                        "ks": register.KS, "tie_seed": 2026, "num_queries": 500, "ranking_rows": 250000,
                        "full_candidates_checked": 1799000, "top10_hits": hits,
                        "all_queries_full_candidate_topk_exact": True, "max_metric_absolute_error": 0.,
                        "max_saved_score_absolute_error": 0., "aggregation_values_checked": 1000,
                        "ranking_sha256": files["rankings.csv"], "files": files}
        write_json(folder / "verification.json", verification)
        state.seal(folder, science)
        results[name] = {"top10_hits": hits, "num_queries": 500, "metrics": metrics, "coverage_metrics": coverage,
                         "completed_sha256": file_hash(folder / "completed.json")}
    write_json(output / "evaluation_index_valid.json", {"split": "valid", "specs": scoring.frozen_grid(),
               "results": results, "context_signature": signature})
    chosen = selection.select_best(results)
    write_json(output / "selection.json", state.signed({**chosen, "context_signature": signature,
               "valid_index_sha256": file_hash(output / "evaluation_index_valid.json"),
               "frozen_at": "2026-11-20T15:01:00+08:00"}))
    write_json(output / "formula_audit_valid.json", {"status": "verified", "query_ids": list(range(37)),
               "candidate_pairs": 11100, "max_absolute_error": 0.})
    write_json(output / "status.json", {"status": "completed", "split": "valid", "winner": chosen["winner"],
               "test_evaluated": False, "context_signature": signature})
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
        self.sources = {"schema_version": 1, "runs": [{"key": f"old_{i}"} for i in range(348)]}
        self.old = [{"key": f"old_{i}", "metric": i / 348, "identity": {"run_id": f"original_{i}"}} for i in range(348)]
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

    def reseal_eval(self, name):
        folder = self.output / "valid" / name
        science = read_json(folder / "evaluation_manifest.json")["signature"]
        reseal(folder, science)
        path = self.output / "evaluation_index_valid.json"
        index = read_json(path)
        index["results"][name]["completed_sha256"] = file_hash(folder / "completed.json")
        write_json(path, index)
        selected = read_json(self.output / "selection.json")
        selected.pop("signature")
        selected["valid_index_sha256"] = file_hash(path)
        write_json(self.output / "selection.json", state.signed(selected))
        self.reseal()

    def test_proposal_fifteen_valid_only_and_idempotent(self):
        before = self.snapshot()
        path = register.write_proposal(self.output, root=self.root)
        result = read_json(path)
        self.assertEqual(len(result["runs"]), 15)
        self.assertTrue(all(row["split"] == "valid" and row["variant"].startswith("RegionalJScore/") for row in result["runs"]))
        self.assertTrue(all(row["category"] == "fusion_ablation" for row in result["runs"]))
        self.assertTrue(all(row["supervision"] == ("untrained" if row["variant"] == "RegionalJScore/j_only" else "self_supervised")
                            for row in result["runs"]))
        structure = next(row for row in result["runs"] if row["variant"] == "RegionalJScore/j_only")
        self.assertEqual(structure["display"], "区域J＋面积（无CNN评分）")
        self.assertEqual(read_json(self.output / "selection.json")["selection_rule"]["test_policy"], "valid_only_no_test")
        stamp = path.stat().st_mtime_ns
        register.write_proposal(self.output, root=self.root)
        self.assertEqual(path.stat().st_mtime_ns, stamp)
        self.assertEqual(before, self.snapshot())
        self.assertTrue(state.verified(self.output, self.signature))

    def test_incomplete_run_rejected(self):
        (self.output / "completed.json").unlink()
        with self.assertRaisesRegex(ValueError, "completed"):
            register.proposal(self.output, root=self.root)

    def test_test_or_training_artifacts_rejected(self):
        for name in ("test", "training"):
            with self.subTest(name=name):
                (self.output / name).mkdir()
                self.reseal()
                with self.assertRaisesRegex(ValueError, "Unexpected test or training"):
                    register.proposal(self.output, root=self.root)
                (self.output / name).rmdir()
                self.reseal()

    def test_partial_grid_rejected(self):
        index = read_json(self.output / "evaluation_index_valid.json")
        del index["specs"]["product"]
        write_json(self.output / "evaluation_index_valid.json", index)
        self.reseal()
        with self.assertRaisesRegex(ValueError, "full predefined grid"):
            register.proposal(self.output, root=self.root)

    def test_winner_change_rejected(self):
        chosen = read_json(self.output / "selection.json")
        chosen.pop("signature")
        chosen["winner"] = "product"
        write_json(self.output / "selection.json", state.signed(chosen))
        self.reseal()
        with self.assertRaisesRegex(ValueError, "Frozen winner"):
            register.proposal(self.output, root=self.root)

    def test_full_candidate_verification_required(self):
        path = self.output / "valid/product/verification.json"
        checked = read_json(path)
        checked["full_candidates_checked"] -= 1
        write_json(path, checked)
        self.reseal_eval("product")
        with self.assertRaisesRegex(ValueError, "verification"):
            register.proposal(self.output, root=self.root)

    def test_input_and_baseline_binding_required(self):
        path = self.output / "inputs.json"
        inputs = read_json(path)
        inputs["source_binding"] = {}
        write_json(path, inputs)
        self.reseal()
        with self.assertRaisesRegex(ValueError, "source binding"):
            register.proposal(self.output, root=self.root)

    def test_changed_old_source_rejected(self):
        original = Path(read_json(self.output / "baseline_replay.json")["source"])
        original.write_text("changed")
        with self.assertRaisesRegex(ValueError, "input source"):
            register.proposal(self.output, root=self.root)

    def test_missing_metric_rejected_even_with_updated_receipt(self):
        folder = self.output / "valid/cnn_only"
        path = folder / "metrics.json"
        report = read_json(path)
        del report["metrics"]["Recall@1"]
        write_json(path, report)
        checked = read_json(folder / "verification.json")
        checked["files"]["metrics.json"] = file_hash(path)
        write_json(folder / "verification.json", checked)
        self.reseal_eval("cnn_only")
        with self.assertRaisesRegex(ValueError, "metric grid"):
            register.proposal(self.output, root=self.root)

    def test_historical_protection_required(self):
        write_json(self.output / "source_protection_after.json", {"status": "changed", "files": 0})
        self.reseal()
        with self.assertRaisesRegex(ValueError, "protection"):
            register.proposal(self.output, root=self.root)

    def test_apply_preserves_all_348_results_and_is_idempotent(self):
        with patch.object(register.catalog, "collect", side_effect=self.collect), patch.object(register.catalog, "render", side_effect=self.render):
            result = register.apply_proposal(self.output, root=self.root)
            self.assertEqual(result["new_results"], 15)
            self.assertEqual(result["old_results_unchanged"], 348)
            self.assertEqual(read_json(self.folder / "snapshot.json")["runs"][:348], self.old)
            self.assertEqual(read_json(Path(result["backup"]) / "manifest.json")["status"], "committed")
            self.assertEqual(register.apply_proposal(self.output, root=self.root)["status"], "already_registered")
        self.assertTrue(state.verified(self.output, self.signature))

    def test_old_result_mutation_aborts_before_shared_writes(self):
        before = self.snapshot()
        def bad_collect(*args):
            value = self.collect(*args)
            value["runs"][30] = {**self.old[30], "metric": .99}
            return value
        with patch.object(register.catalog, "collect", side_effect=bad_collect), patch.object(register.catalog, "render", side_effect=self.render):
            with self.assertRaisesRegex(ValueError, "previously registered"):
                register.apply_proposal(self.output, root=self.root)
        self.assertEqual(before, self.snapshot())

    def test_write_failure_rolls_back_shared_files(self):
        before = self.snapshot()
        atomic = register._atomic_write
        failed = False
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
