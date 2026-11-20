"""Registration metadata gates and transactional rollback; no formal evaluation."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from f_region_structure import register, scoring, selection, state
from f_region_structure.verify import REPORT_FILES
from wbm_retrieval.common import digest, file_hash, read_json, write_json


def reseal(folder, signature):
    (folder / "completed.json").unlink(missing_ok=True)
    state.seal(folder, signature)


def fixture(root):
    """Small files model verification receipts; metrics themselves are tested elsewhere."""
    output = root / "artificial" / "f_region_structure_20261120T120000+0800"
    output.mkdir(parents=True)
    context = {"version": "f_region_structure_v1", "grid": scoring.frozen_grid(),
               "selection_rule": selection.RULE, "posthoc_design": True, "test_parameter_search": False,
               "code": {"f_region_structure/learning.py": "a" * 64}}
    signature = digest(context)
    write_json(output / "run_manifest.json", state.signed({"version": "f_region_structure_v1", "context": context}))
    model_binding = {}
    for family in register.FAMILIES:
        folder = output / "training" / family
        folder.mkdir(parents=True)
        settings = {"family": family, "split": "train", "epochs": 20, "batch_size": 512,
                    "seed": 2026, "checkpoint_rule": "last_epoch", "labels_used": False,
                    "CNN_frozen": True, "area_normalization": {"fit_split": "train"}}
        manifest = state.signed({"version": "regional_joint_learning_v1", "split": "train", "test_used": False,
                                 "settings": settings, "code_sha256": "a" * 64})
        write_json(folder / "training_manifest.json", manifest)
        (folder / "final.pt").write_bytes(family.encode())
        (folder / "loss.csv").write_text("epoch,loss\n20,1.0\n")
        state.seal(folder, manifest["signature"])
        done = read_json(folder / "completed.json")
        done.update(epoch=20, checkpoint_rule="last_epoch")
        write_json(folder / "completed.json", done)
        model_binding[family] = file_hash(folder / "final.pt")
    write_json(output / "model_binding.json", model_binding)
    write_json(output / "training_inputs.json", {"labels_used": False, "fixed_epochs": 20,
               "checkpoint_rule": "final_epoch", "seed": 2026})
    write_json(output / "source_protection_before.json", {})
    write_json(output / "source_protection_after.json", {"status": "unchanged", "files": 0})
    write_json(output / "baseline_replay.json", {"status": "verified", "rows": 250000, "max_score_error": 0.})

    def evaluation(split, name, spec, hits):
        folder = output / split / name
        folder.mkdir(parents=True)
        nquery, ngallery = (500, 3599) if split == "valid" else (7199, 7199)
        science = digest({"context": signature, "split": split, "variant": name, "spec": spec})
        key = "f_region_structure_fixture_" + split + "_" + name
        identity = {"schema_version": 1, "key": key, "split": split, "variant": name,
                    "started_at": "2026-11-20T12:00:00+08:00", "run_id": "20261120T120000+0800__" + key,
                    "evaluation_signature": science}
        manifest = {**identity, "signature": science, "spec": spec, "ks": register.KS, "tie_seed": 2026,
                    "retrieval_protocol_signature": register.catalog.BASE_PROTOCOL_SIGNATURE,
                    "num_queries": nquery, "gallery_count": ngallery, "candidate_rule": "all_gallery_except_query",
                    "checkpoint_kind": "best_nce", "source_seed": 2026, "matching": "original_F_greedy_sqrt_area",
                    "posthoc_research": True, "test_parameter_search": False,
                    "training_required": spec["family"] in register.FAMILIES}
        write_json(folder / "run_identity.json", identity)
        write_json(folder / "evaluation_manifest.json", manifest)
        metrics = {f"{metric}@{k}": .1 for metric in ("Hit", "Precision", "Recall", "mAP") for k in register.KS}
        metrics["Precision@10"] = hits / (nquery * 10)
        coverage = {f"{metric}@{k}": .8 for metric in ("Coverage-nDCG", "MeanCoverage", "FullCoverageHit",
                    "FullCoveragePrecision", "FullCoverageRecall", "FullCoverage_mAP") for k in register.KS}
        for family, namefile, values in (("exact", "metrics.json", metrics), ("coverage", "coverage_metrics.json", coverage)):
            write_json(folder / namefile, {"split": split, "num_queries": nquery, "ks": register.KS,
                "metric_version": family + "_v1", "test_evaluated": split == "test", "metadata": manifest,
                "metrics": values, "retrieval_protocol_signature": register.catalog.BASE_PROTOCOL_SIGNATURE})
        for filename in ("rankings.csv", "per_query.csv", "per_class.csv", "coverage_per_query.csv", "coverage_per_class.csv"):
            (folder / filename).write_text("synthetic_metadata_fixture\n")
        files = {name: file_hash(folder / name) for name in (*REPORT_FILES, "run_identity.json")}
        verification = {"status": "verified", "signature": science, "split": split, "variant": name,
                        "ks": register.KS, "tie_seed": 2026, "num_queries": nquery, "ranking_rows": nquery * 500,
                        "full_candidates_checked": nquery * (ngallery - 1), "top10_hits": hits,
                        "all_queries_full_candidate_topk_exact": True, "max_metric_absolute_error": 0.,
                        "max_saved_score_absolute_error": 0., "aggregation_values_checked": 1000,
                        "ranking_sha256": files["rankings.csv"], "files": files}
        write_json(folder / "verification.json", verification)
        state.seal(folder, science)
        return {"top10_hits": hits, "num_queries": nquery, "metrics": metrics, "coverage_metrics": coverage,
                "completed_sha256": file_hash(folder / "completed.json")}

    grid = scoring.frozen_grid()
    valid = {name: evaluation("valid", name, spec, 1000 + i) for i, (name, spec) in enumerate(grid.items())}
    write_json(output / "evaluation_index_valid.json", {"split": "valid", "specs": grid, "results": valid,
               "context_signature": signature})
    chosen = selection.select_best(valid)
    write_json(output / "selection.json", state.signed({**chosen, "context_signature": signature,
               "valid_index_sha256": file_hash(output / "evaluation_index_valid.json"), "models": model_binding,
               "frozen_at": "2026-11-20T12:01:00+08:00"}))
    specs = {chosen["winner"]: chosen["selected_spec"]}
    test = {name: evaluation("test", name, spec, 20000) for name, spec in specs.items()}
    write_json(output / "evaluation_index_test.json", {"split": "test", "specs": specs, "results": test,
               "context_signature": signature})
    for split, total in (("valid", 25), ("test", 1)):
        write_json(output / f"formula_audit_{split}.json", {"status": "verified", "query_ids": list(range(37)),
                   "candidate_pairs": 37 * 20 * total, "max_absolute_error": 0.})
    write_json(output / "status.json", {"status": "completed", "stage": "all", "winner": chosen["winner"],
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
        self.sources = {"schema_version": 1, "runs": [{"key": "old"}]}
        self.old = {"key": "old", "metric": 0.42, "identity": {"run_id": "original"}}
        self.previous = {"source_spec_sha256": register._fingerprint(self.sources), "runs": [self.old]}
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
                "runs": [self.old, *[dict(row) for row in sources["runs"] if row["key"] != "old"]]}

    def reseal(self):
        reseal(self.output, self.signature)

    def reseal_eval(self, split, name):
        folder = self.output / split / name
        science = read_json(folder / "evaluation_manifest.json")["signature"]
        reseal(folder, science)
        path = self.output / f"evaluation_index_{split}.json"
        index = read_json(path)
        index["results"][name]["completed_sha256"] = file_hash(folder / "completed.json")
        write_json(path, index)
        if split == "valid":
            selected = read_json(self.output / "selection.json")
            selected.pop("signature")
            selected["valid_index_sha256"] = file_hash(path)
            write_json(self.output / "selection.json", state.signed(selected))
        self.reseal()

    def test_proposal_complete_grid_single_test_and_stable_identity(self):
        before = self.snapshot()
        path = register.write_proposal(self.output, root=self.root)
        value = read_json(path)
        self.assertEqual(len(value["runs"]), 26)
        self.assertEqual(sum(row["split"] == "test" for row in value["runs"]), 1)
        self.assertTrue(all(row["category"] == "fusion_ablation" for row in value["runs"]))
        self.assertTrue(all(row["supervision"] == "self_supervised" for row in value["runs"]))
        stamp = path.stat().st_mtime_ns
        register.write_proposal(self.output, root=self.root)
        self.assertEqual(path.stat().st_mtime_ns, stamp)
        self.assertEqual(before, self.snapshot())
        self.assertTrue(state.verified(self.output, self.signature))

    def test_missing_root_seal_and_valid_only_rejected(self):
        (self.output / "completed.json").unlink()
        with self.assertRaisesRegex(ValueError, "completed"):
            register.proposal(self.output, root=self.root)
        status = read_json(self.output / "status.json")
        status.update(stage="valid", test_evaluated=False)
        write_json(self.output / "status.json", status)
        self.reseal()
        with self.assertRaisesRegex(ValueError, "single test"):
            register.proposal(self.output, root=self.root)

    def test_partial_grid_rejected_even_if_resealed(self):
        index = read_json(self.output / "evaluation_index_valid.json")
        del index["specs"]["baseline"]
        write_json(self.output / "evaluation_index_valid.json", index)
        self.reseal()
        with self.assertRaisesRegex(ValueError, "full predefined grid"):
            register.proposal(self.output, root=self.root)

    def test_winner_cannot_be_changed_after_validation(self):
        chosen = read_json(self.output / "selection.json")
        chosen.pop("signature")
        chosen["winner"] = "baseline"
        write_json(self.output / "selection.json", state.signed(chosen))
        self.reseal()
        with self.assertRaisesRegex(ValueError, "Frozen winner"):
            register.proposal(self.output, root=self.root)

    def test_missing_verification_coverage_rejected(self):
        path = self.output / "valid/baseline/verification.json"
        checked = read_json(path)
        checked["full_candidates_checked"] -= 1
        write_json(path, checked)
        self.reseal_eval("valid", "baseline")
        with self.assertRaisesRegex(ValueError, "verification"):
            register.proposal(self.output, root=self.root)

    def test_model_and_historical_protection_gates(self):
        path = self.output / "source_protection_after.json"
        after = read_json(path)
        after["status"] = "changed"
        write_json(path, after)
        self.reseal()
        with self.assertRaisesRegex(ValueError, "protection"):
            register.proposal(self.output, root=self.root)
        after["status"] = "unchanged"
        write_json(path, after)
        path = self.output / "model_binding.json"
        binding = read_json(path)
        binding["plain_mlp"] = "f" * 64
        write_json(path, binding)
        self.reseal()
        with self.assertRaisesRegex(ValueError, "model bindings"):
            register.proposal(self.output, root=self.root)

    def test_extra_test_directory_rejected(self):
        (self.output / "test/unselected").mkdir()
        self.reseal()
        with self.assertRaisesRegex(ValueError, "extra"):
            register.proposal(self.output, root=self.root)

    def test_registration_preserves_old_results_and_is_idempotent(self):
        with patch.object(register.catalog, "collect", side_effect=self.collect), patch.object(register.catalog, "render", side_effect=self.render):
            result = register.apply_proposal(self.output, root=self.root)
            self.assertEqual(result["new_results"], 26)
            self.assertEqual(result["old_results_unchanged"], 1)
            self.assertEqual(read_json(self.folder / "snapshot.json")["runs"][0], self.old)
            self.assertEqual(read_json(Path(result["backup"]) / "manifest.json")["status"], "committed")
            second = register.apply_proposal(self.output, root=self.root)
            self.assertEqual(second["status"], "already_registered")
        self.assertTrue(state.verified(self.output, self.signature))

    def test_collection_cannot_change_old_result(self):
        before = self.snapshot()
        def bad_collect(*args):
            value = self.collect(*args)
            value["runs"][0] = {**self.old, "metric": .99}
            return value
        with patch.object(register.catalog, "collect", side_effect=bad_collect), patch.object(register.catalog, "render", side_effect=self.render):
            with self.assertRaisesRegex(ValueError, "previously registered"):
                register.apply_proposal(self.output, root=self.root)
        self.assertEqual(before, self.snapshot())

    def test_write_failure_rolls_back_all_shared_files(self):
        before = self.snapshot()
        original_atomic = register._atomic_write
        failed = False
        def fail_once(path, content):
            nonlocal failed
            if Path(path).name == "registry.csv" and not failed:
                failed = True
                raise OSError("injected write failure")
            original_atomic(path, content)
        with patch.object(register.catalog, "collect", side_effect=self.collect), patch.object(register.catalog, "render", side_effect=self.render), patch.object(register, "_atomic_write", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "injected"):
                register.apply_proposal(self.output, root=self.root)
        self.assertEqual(before, self.snapshot())
        receipts = list((self.output / "catalog_backup").glob("*/manifest.json"))
        self.assertEqual(len(receipts), 1)
        self.assertEqual(read_json(receipts[0])["status"], "rolled_back")


if __name__ == "__main__":
    unittest.main()
