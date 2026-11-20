"""Synthetic metadata receipts exercise selection gates and atomic registration."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from f_region_area_fusion import register, scoring, selection
from f_region_structure import state
from f_region_structure.verify import REPORT_FILES
from wbm_retrieval.common import digest, file_hash, read_json, write_json


def reseal(folder, signature):
    (folder / "completed.json").unlink(missing_ok=True)
    state.seal(folder, signature)


def fixture(root, *, source=None, baseline_selected=True):
    """Model small verification receipts, never fabricate full production rankings."""
    split = "valid" if source is None else "test"
    output = root / f"artificial/f_region_area_fusion_{split}_20261120T180000+0800"
    output.mkdir(parents=True)
    grid = scoring.frozen_grid()
    cache_source = root / "artificial/original_regional"
    cache_source.mkdir(parents=True, exist_ok=True)
    payload = cache_source / "input.json"
    write_json(payload, {"synthetic_input": True})
    binding = {str(payload): file_hash(payload)}
    specs, frozen = grid, None
    if source is not None:
        chosen = read_json(source / "selection.json")
        specs = chosen["selected_specs"]
        original_files = {"selection.json": "source_selection_sha256", "completed.json": "source_completed_sha256",
                          "evaluation_index_valid.json": "source_valid_index_sha256"}
        frozen = state.signed({"schema_version": 1, "source_run": str(source),
            **{field: file_hash(source / name) for name, field in original_files.items()},
            "selection": chosen, "selected_variants": chosen["selected_variants"],
            "selected_specs": specs, "selection_policy": register.POLICY})
        write_json(output / "frozen_selection.json", frozen)
        binding.update({str(source / name): file_hash(source / name) for name in original_files})
    context = {"version": register.VERSION, "split": split, "grid": specs, "full_grid": grid, "selection_rule": selection.RULE,
               "source_binding": binding, "source_run": str(source or cache_source),
               "cache_source_run": str(cache_source), "training": False, "posthoc_design": True,
               "test_parameter_search": False,
               "code": {"synthetic.py": "a" * 64}, "matching": "original_F_greedy_sqrt_area"}
    if frozen is not None:
        context["frozen_selection_signature"] = frozen["signature"]
    signature = digest(context)
    write_json(output / "run_manifest.json", state.signed({"version": register.VERSION, "context": context}))
    write_json(output / "inputs.json", {"source_binding": binding, "context_signature": signature,
               "source_run": context["source_run"], "cache_source_run": str(cache_source)})
    write_json(output / "source_protection_before.json", {})
    write_json(output / "source_protection_after.json", {"status": "unchanged", "files": 0})
    nquery, ngallery = (500, 3599) if split == "valid" else (7199, 7199)
    results = {}
    for index, (name, spec) in enumerate(specs.items()):
        folder = output / split / name
        folder.mkdir(parents=True)
        science = digest({"context": signature, "split": split, "variant": name, "spec": spec})
        key = "f_region_area_fusion_fixture_" + split + "_" + name
        identity = {"schema_version": 1, "key": key, "split": split, "variant": name,
                    "started_at": "2026-11-20T18:00:00+08:00", "run_id": "20261120T180000+0800__" + key,
                    "evaluation_signature": science}
        manifest = {**identity, "signature": science, "spec": spec, "ks": register.KS, "tie_seed": 2026,
                    "retrieval_protocol_signature": register.catalog.BASE_PROTOCOL_SIGNATURE,
                    "num_queries": nquery, "gallery_count": ngallery, "candidate_rule": "all_gallery_except_query",
                    "checkpoint_kind": "best_nce", "source_seed": 2026, "matching": "original_F_greedy_sqrt_area",
                    "posthoc_research": True, "test_parameter_search": False, "test_evaluated": split == "test",
                    "training_required": False, "selection_split": "valid"}
        write_json(folder / "run_identity.json", identity)
        write_json(folder / "evaluation_manifest.json", manifest)
        if split == "test":
            hits = 40000 + index
        elif name == register.BASELINE:
            hits = 2585
        else:
            hits = (2500 if baseline_selected else 2600) + index
        metrics = {f"{metric}@{k}": .1 for metric in ("Hit", "Precision", "Recall", "mAP") for k in register.KS}
        metrics["Precision@10"] = hits / (nquery * 10)
        coverage = {f"{metric}@{k}": .8 for metric in ("Coverage-nDCG", "MeanCoverage", "FullCoverageHit",
                    "FullCoveragePrecision", "FullCoverageRecall", "FullCoverage_mAP") for k in register.KS}
        for family, filename, values in (("exact", "metrics.json", metrics), ("coverage", "coverage_metrics.json", coverage)):
            write_json(folder / filename, {"split": split, "num_queries": nquery, "ks": register.KS,
                "metric_version": family + "_v1", "test_evaluated": split == "test", "metadata": manifest,
                "metrics": values, "retrieval_protocol_signature": register.catalog.BASE_PROTOCOL_SIGNATURE})
        for filename in ("rankings.csv", "per_query.csv", "per_class.csv", "coverage_per_query.csv", "coverage_per_class.csv"):
            (folder / filename).write_text("synthetic_metadata_fixture\n")
        files = {filename: file_hash(folder / filename) for filename in (*REPORT_FILES, "run_identity.json")}
        checked = {"status": "verified", "signature": science, "split": split, "variant": name,
                   "ks": register.KS, "tie_seed": 2026, "num_queries": nquery, "ranking_rows": nquery * 500,
                   "full_candidates_checked": nquery * (ngallery - 1), "top10_hits": hits,
                   "all_queries_full_candidate_topk_exact": True, "max_metric_absolute_error": 0.,
                   "max_saved_score_absolute_error": 0., "aggregation_values_checked": 1000,
                   "ranking_sha256": files["rankings.csv"], "files": files}
        write_json(folder / "verification.json", checked)
        state.seal(folder, science)
        results[name] = {"top10_hits": hits, "num_queries": nquery, "metrics": metrics, "coverage_metrics": coverage,
                         "completed_sha256": file_hash(folder / "completed.json")}
    write_json(output / f"evaluation_index_{split}.json", {"split": split, "specs": specs,
               "results": results, "context_signature": signature})
    if split == "valid":
        chosen = state.signed({**selection.select_top10(results), "context_signature": signature,
            "valid_index_sha256": file_hash(output / "evaluation_index_valid.json"),
            "frozen_at": "2026-11-20T18:10:00+08:00"})
        write_json(output / "selection.json", chosen)
    if register.BASELINE in specs:
        original = root / "artificial/old_equal_weight" / split / "rankings.csv"
        original.parent.mkdir(parents=True, exist_ok=True)
        original.write_bytes((output / split / register.BASELINE / "rankings.csv").read_bytes())
        write_json(output / "baseline_replay.json", {"status": "verified", "split": split,
            "variant": register.BASELINE, "rows": nquery * 500, "max_score_error": 0,
            "source": str(original), "source_sha256": file_hash(original), "ranking_byte_identical": True})
    write_json(output / f"formula_audit_{split}.json", {"status": "verified", "split": split,
               "context_signature": signature, "specs": specs, "query_ids": list(range(37)),
               "candidate_pairs": 37 * 20 * len(specs), "max_absolute_error": 0.})
    write_json(output / "status.json", {"status": "completed", "split": split, "winner": chosen["winner"],
               "selected_variants": chosen["selected_variants"],
               "test_evaluated": split == "test", "context_signature": signature})
    state.seal(output, signature)
    return output, signature


class RegisterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name).resolve()
        self.root = self.workspace / "wbm-region-retrieval"
        self.valid, self.valid_signature = fixture(self.root)
        self.test, self.test_signature = fixture(self.root, source=self.valid)
        self.folder = self.root / "results_catalog"
        self.sources = {"schema_version": 1, "runs": [{"key": f"old_{i}"} for i in range(364)]}
        self.old = [{"key": f"old_{i}", "metric": i / 364, "identity": {"run_id": f"original_{i}"}} for i in range(364)]
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
        existing = {row["key"]: row for row in previous["runs"]}
        return {"source_spec_sha256": register._fingerprint(sources),
                "runs": [existing.get(row["key"], dict(row)) for row in sources["runs"]]}

    def reseal_test(self):
        reseal(self.test, self.test_signature)

    def reseal_test_evaluation(self, name):
        folder = self.test / "test" / name
        reseal(folder, read_json(folder / "evaluation_manifest.json")["signature"])
        path = self.test / "evaluation_index_test.json"
        index = read_json(path)
        index["results"][name]["completed_sha256"] = file_hash(folder / "completed.json")
        write_json(path, index)
        self.reseal_test()

    def test_valid29_test10_proposals_and_idempotence(self):
        before = self.snapshot()
        valid = register.proposal(self.valid, root=self.root)
        path = register.write_proposal(self.test, root=self.root)
        test = read_json(path)
        self.assertEqual(len(valid["runs"]), 29)
        self.assertEqual(len(test["runs"]), 10)
        self.assertEqual({row["variant"].split("/")[1] for row in test["runs"]},
                         set(read_json(self.valid / "selection.json")["selected_variants"]))
        self.assertTrue(all(row["supervision"] == "self_supervised" for row in valid["runs"] + test["runs"]))
        stamp = path.stat().st_mtime_ns
        register.write_proposal(self.test, root=self.root)
        self.assertEqual(path.stat().st_mtime_ns, stamp)
        self.assertEqual(before, self.snapshot())

    def test_incomplete_test_rejected(self):
        (self.test / "completed.json").unlink()
        with self.assertRaisesRegex(ValueError, "completed"):
            register.proposal(self.test, root=self.root)

    def test_unselected_test_variant_rejected(self):
        (self.test / "test/unselected").mkdir()
        self.reseal_test()
        with self.assertRaisesRegex(ValueError, "extra"):
            register.proposal(self.test, root=self.root)

    def test_changed_valid_source_rejected(self):
        path = self.valid / "selection.json"
        value = read_json(path)
        value["winner"] = "entropy"
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, "input source"):
            register.proposal(self.test, root=self.root)

    def test_resigned_valid_ranking_change_rejected(self):
        path = self.valid / "selection.json"
        value = read_json(path)
        value.pop("signature")
        value["selected_variants"] = list(reversed(value["selected_variants"]))
        write_json(path, state.signed(value))
        reseal(self.valid, self.valid_signature)
        with self.assertRaisesRegex(ValueError, "integer valid"):
            register.proposal(self.valid, root=self.root)

    def test_full_candidate_verification_required(self):
        name = next(iter(read_json(self.test / "evaluation_index_test.json")["specs"]))
        path = self.test / "test" / name / "verification.json"
        value = read_json(path)
        value["full_candidates_checked"] -= 1
        write_json(path, value)
        self.reseal_test_evaluation(name)
        with self.assertRaisesRegex(ValueError, "full-candidate"):
            register.proposal(self.test, root=self.root)

    def test_missing_coverage_metric_rejected_after_reseal(self):
        name = next(iter(read_json(self.test / "evaluation_index_test.json")["specs"]))
        folder = self.test / "test" / name
        path = folder / "coverage_metrics.json"
        report = read_json(path)
        del report["metrics"]["FullCoverageRecall@1"]
        write_json(path, report)
        checked = read_json(folder / "verification.json")
        checked["files"]["coverage_metrics.json"] = file_hash(path)
        write_json(folder / "verification.json", checked)
        self.reseal_test_evaluation(name)
        with self.assertRaisesRegex(ValueError, "metric grid"):
            register.proposal(self.test, root=self.root)

    def test_formula_audit_grid_count_required(self):
        path = self.test / "formula_audit_test.json"
        value = read_json(path)
        value["candidate_pairs"] -= 1
        write_json(path, value)
        self.reseal_test()
        with self.assertRaisesRegex(ValueError, "formula/greedy"):
            register.proposal(self.test, root=self.root)

    def test_formula_audit_specs_must_match_selected_grid(self):
        path = self.test / "formula_audit_test.json"
        value = read_json(path)
        value["specs"] = {}
        write_json(path, value)
        self.reseal_test()
        with self.assertRaisesRegex(ValueError, "formula/greedy"):
            register.proposal(self.test, root=self.root)

    def test_selection_needs_timezone_aware_freeze_time(self):
        path = self.valid / "selection.json"
        value = read_json(path)
        value.pop("signature")
        value["frozen_at"] = "2026-11-20T18:10:00"
        write_json(path, state.signed(value))
        reseal(self.valid, self.valid_signature)
        with self.assertRaisesRegex(ValueError, "frozen_at"):
            register.proposal(self.valid, root=self.root)

    def test_source_protection_receipt_required(self):
        write_json(self.test / "source_protection_after.json", {"status": "changed", "files": 0})
        self.reseal_test()
        with self.assertRaisesRegex(ValueError, "protection"):
            register.proposal(self.test, root=self.root)

    def test_no_baseline_test_when_baseline_not_selected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve() / "wbm-region-retrieval"
            valid, _ = fixture(root, baseline_selected=False)
            test, _ = fixture(root, source=valid)
            self.assertNotIn(register.BASELINE, read_json(valid / "selection.json")["selected_variants"])
            self.assertFalse((test / "baseline_replay.json").exists())
            self.assertEqual(len(register.proposal(test, root=root)["runs"]), 10)

    def test_transaction_adds29_then10_preserves364_and_idempotent(self):
        with patch.object(register.catalog, "collect", side_effect=self.collect), patch.object(register.catalog, "render", side_effect=self.render):
            first = register.apply_proposal(self.valid, root=self.root)
            valid_rows = read_json(self.folder / "snapshot.json")["runs"]
            second = register.apply_proposal(self.test, root=self.root)
            self.assertEqual((first["new_results"], first["old_results_unchanged"]), (29, 364))
            self.assertEqual((second["new_results"], second["old_results_unchanged"]), (10, 393))
            self.assertEqual(read_json(self.folder / "snapshot.json")["runs"][:393], valid_rows)
            self.assertEqual(valid_rows[:364], self.old)
            self.assertEqual(register.apply_proposal(self.test, root=self.root)["status"], "already_registered")
        self.assertTrue(state.verified(self.valid, self.valid_signature))
        self.assertTrue(state.verified(self.test, self.test_signature))

    def test_collection_cannot_change_old_result(self):
        before = self.snapshot()
        def bad_collect(*args):
            value = self.collect(*args)
            value["runs"][20] = {**self.old[20], "metric": .999}
            return value
        with patch.object(register.catalog, "collect", side_effect=bad_collect), patch.object(register.catalog, "render", side_effect=self.render):
            with self.assertRaisesRegex(ValueError, "previously registered"):
                register.apply_proposal(self.test, root=self.root)
        self.assertEqual(before, self.snapshot())

    def test_failed_write_rolls_back(self):
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
                register.apply_proposal(self.test, root=self.root)
        self.assertEqual(before, self.snapshot())
        receipts = list((self.test / "catalog_backup").glob("*/manifest.json"))
        self.assertEqual(len(receipts), 1)
        self.assertEqual(read_json(receipts[0])["status"], "rolled_back")


if __name__ == "__main__":
    unittest.main()
