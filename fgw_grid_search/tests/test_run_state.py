"""Root orchestration and sealed metadata tests, exclusively in temporary trees.

The sealed fixture below exercises metadata/hash gates, not retrieval accuracy;
the real synthetic ranking and metric checks live in test_pipeline/test_verify.
"""
from contextlib import contextmanager, ExitStack, redirect_stdout
from copy import deepcopy
import csv
from datetime import datetime
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from fgw_grid_search import cache, pipeline, register, run, scoring, sources, state
from results_catalog.build import BASE_PROTOCOL_SIGNATURE
from wbm_retrieval.common import digest, file_hash, read_json, write_json


def snapshot(folder):
    return {str(p.relative_to(folder)): (file_hash(p), p.stat().st_mtime_ns)
            for p in Path(folder).rglob("*") if p.is_file()}


@contextmanager
def isolated_run(root):
    """Patch imported path resolvers too: their default ROOT is definition-bound."""
    with ExitStack() as stack:
        stack.enter_context(patch.object(run, "ROOT", root))
        stack.enter_context(patch.object(run, "choose_output", side_effect=lambda *a, **kw: state.choose_output(*a, root=root, **kw)))
        stack.enter_context(patch.object(run, "resolve", side_effect=lambda value: state.resolve(value, root)))
        yield


def grid_rows(specs, count=500, hits=2500):
    return [{"variant": name, "family": spec["family"], "split": "valid", "num_queries": count,
             "precision10_hits": hits, "Precision@10": hits / (10 * count),
             "mAP@100": .2, "Coverage-nDCG@10": .8} for name, spec in specs.items()]


def seal_root(output, manifest, count):
    write_json(output / "completed.json", {"status": "completed", "version": state.VERSION,
        "stage": manifest["stage"], "signature": manifest["signature"], "new_evaluations": count,
        "files": {str(p.relative_to(output)): file_hash(p) for p in output.rglob("*")
                  if p.is_file() and p != output / "completed.json"}})


def sealed_fixture(root):
    """One-variant metadata fixture with formal count fields; no real data/cache."""
    output = root / "artificial/fgw_grid_synthetic"
    settings = scoring.validate_settings({"denominator": 1, "families": ["sum_raw"], "include_reference": False})
    specs = scoring.specs(settings)
    context = {"settings": settings, "specs": specs, "split": "valid", "source_run": str(root / "source"),
               "selection_source": None, "code": {}, "environment": {}, "provenance": {}}
    manifest = state.signed({"version": state.VERSION, "stage": "search", "context": context,
                             "output": str(output), "started_at": "2026-11-19T18:00:00+08:00"})
    write_json(output / "run_manifest.json", manifest)
    write_json(output / "settings.json", settings)
    inp = state.signed({"parent_signature": manifest["signature"], "split": "valid", "context_digest": digest(context)})
    write_json(output / "inputs.json", inp)
    write_json(output / "source_audit.json", state.signed({"status": "verified", "context_digest": digest(context)}))
    write_json(output / "source_protection_before.json", {})
    write_json(output / "source_protection_after.json", {"status": "unchanged", "files": 0, "inventory_digest": digest({})})
    marker = output / "component_cache/valid/completed/completed.json"
    cache_signature = digest({"inputs": inp["signature"], "operation": "full_F_G_W_scores"})
    write_json(marker, {"signature": cache_signature, "synthetic_metadata_fixture": True})
    binding = {"signature": cache_signature, "completed_sha256": file_hash(marker)}
    index = {"signature": inp["signature"], "split": "valid", "stage": "search", "settings": settings,
             "variants": specs, "provider_source": binding, "retrieval_protocol_signature": BASE_PROTOCOL_SIGNATURE,
             "variant_signatures": {}, "completed_sha256": {}, "summary_rows": grid_rows(specs)}
    for name, spec in specs.items():
        folder = output / "valid" / name
        signature = pipeline.variant_signature(inp["signature"], name, spec, "valid", settings, "search", binding, BASE_PROTOCOL_SIGNATURE)
        identity = {"schema_version": 1, "key": "fgw_grid_synthetic_valid_" + name,
                    "split": "valid", "variant": name, "evaluation_signature": signature,
                    "started_at": "2026-11-19T18:00:00+08:00",
                    "run_id": "20261119T180000+0800__fgw_grid_synthetic_valid_" + name}
        write_json(folder / "run_identity.json", identity)
        write_json(folder / "evaluation_manifest.json", {**identity, "signature": signature,
            "parent_signature": inp["signature"], "stage": "search", "spec": spec, "settings": settings,
            "ks": settings["ks"], "tie_seed": settings["tie_seed"], "provider_source": binding,
            "retrieval_protocol_signature": BASE_PROTOCOL_SIGNATURE})
        metadata = {"signature": signature, "retrieval_protocol_signature": BASE_PROTOCOL_SIGNATURE}
        report = {"split": "valid", "num_queries": 500, "ks": settings["ks"], "metadata": metadata}
        write_json(folder / "metrics.json", {**report, "metric_version": "exact_v1", "metrics": {"Precision@10": .5, "mAP@100": .2}})
        write_json(folder / "coverage_metrics.json", {**report, "metric_version": "coverage_v1", "metrics": {"Coverage-nDCG@10": .8}})
        with (folder / "per_query.csv").open("w", newline="") as handle:
            writer = csv.writer(handle); writer.writerow(["query_id", "Precision@10"])
            writer.writerows((q, .5) for q in range(500))
        for filename in ("rankings.csv", "per_class.csv", "coverage_per_query.csv", "coverage_per_class.csv"):
            (folder / filename).write_text("synthetic metadata integrity fixture only\n")
        verification = {"status": "verified", "signature": signature, "provider_source": binding,
            "num_queries": 500, "ks": settings["ks"], "ranking_rows": 250000,
            "full_candidates_checked": 500 * 3598, "all_queries_full_candidate_topk_exact": True,
            "ranking_sha256": file_hash(folder / "rankings.csv"),
            "files": {p: file_hash(folder / p) for p in pipeline.REPORT_FILES}}
        write_json(folder / "verification.json", verification)
        write_json(folder / "timing.json", {})
        write_json(folder / "completed.json", {"status": "completed", "signature": signature,
            "files": {p.name: file_hash(p) for p in folder.iterdir()}})
        index["variant_signatures"][name] = signature
        index["completed_sha256"][name] = file_hash(folder / "completed.json")
    index["query_ids_digest"] = digest(list(range(500)))
    index["gallery_ids_digest"] = digest(list(range(3599)))
    write_json(output / "evaluation_request_valid.json", {k: v for k, v in index.items()
        if k not in ("variant_signatures", "completed_sha256", "summary_rows")})
    index_path = output / "evaluation_index_valid.json"
    write_json(index_path, index)
    run.write_grid_table(output, index)
    chosen = state.select_payload(manifest, index, index_path)
    write_json(output / "selection.json", chosen)
    write_json(output / "status.json", {"status": "completed", "stage": "search", "test_evaluated": False})
    seal_root(output, manifest, len(specs))
    return output, manifest, index, chosen


def metadata_provider(output, index):
    return SimpleNamespace(signature=index["provider_source"]["signature"],
        marker=output / "component_cache/valid/completed/completed.json",
        split="valid", protocol_signature=BASE_PROTOCOL_SIGNATURE,
        query_ids=np.arange(500, dtype=np.int64), gallery_ids=np.arange(3599, dtype=np.int64))


class PlanTests(unittest.TestCase):
    def test_dry_run_default_166_is_valid_only_without_loading_or_creating(self):
        with tempfile.TemporaryDirectory() as tmp, isolated_run(Path(tmp).resolve()), ExitStack() as stack:
            for module, name in ((sources, "load_inputs"), (cache, "build_cache"), (pipeline, "evaluate"), (run, "code_identity")):
                stack.enter_context(patch.object(module, name, side_effect=AssertionError("dry-run side effect")))
            with redirect_stdout(StringIO()) as capture:
                self.assertEqual(run.main(["--dry-run"]), 0)
            plan = json.loads(capture.getvalue())
            self.assertEqual(plan["evaluations"], 166)
            self.assertEqual(plan["stage"], "search")
            self.assertEqual(plan["split"], "valid")
            self.assertFalse(plan["test_grid_search"])
            self.assertEqual(len(plan["specs"]), 166)
            self.assertTrue(all(v["weights"][0] > 0 for v in plan["specs"].values()))
            self.assertEqual(list(Path(tmp).resolve().iterdir()), [])

    def test_output_isolation_existing_unsigned_and_nested_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for path in ("artifacts/fgw_grid_x", "protocol/fgw_grid_x", "artificial/wrong_prefix", "artificial"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    state.choose_output(path, root=root)
            parent = root / "artificial/fgw_grid_old"
            parent.mkdir(parents=True); write_json(parent / "run_manifest.json", {})
            with self.assertRaises(FileExistsError): state.choose_output(str(parent), root=root)
            with self.assertRaisesRegex(ValueError, "Cannot nest"):
                state.choose_output(str(parent / "fgw_grid_child"), root=root)
            with self.assertRaisesRegex(ValueError, "original signed manifest"):
                state.choose_output(resume_root="artificial/fgw_grid_missing", root=root)
            with self.assertRaises(ValueError): state.separate(parent / "new", [parent])
            with self.assertRaises(ValueError): state.separate(parent, [parent / "nested_source"])
            outside = root / "outside"
            outside.mkdir()
            (root / "artificial/link").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                state.choose_output("artificial/link/fgw_grid_escape", root=root)

    def test_explicit_resume_rejects_grid_source_stage_and_unsigned_changes(self):
        with tempfile.TemporaryDirectory() as tmp, isolated_run(Path(tmp).resolve()):
            root = Path(tmp).resolve()
            args = run.parser().parse_args(["--output-root", "artificial/fgw_grid_resume"])
            output, plan, _, _ = run.prepare_plan(args)
            manifest = state.signed({"version": state.VERSION, "stage": "search", "context": plan})
            write_json(output / "run_manifest.json", manifest)
            resume = run.parser().parse_args(["--resume-root", str(output)])
            self.assertEqual(run.prepare_plan(resume)[1], plan)
            before = snapshot(output)
            config = root / "changed.json"; write_json(config, {"denominator": 2})
            for options in (("--config", str(config)), ("--source-run", "artificial/different_source"), ("--stage", "test")):
                with self.subTest(options=options), self.assertRaisesRegex(ValueError, "change|changed"):
                    run.prepare_plan(run.parser().parse_args(["--resume-root", str(output), *options]))
            self.assertEqual(snapshot(output), before)
            broken = deepcopy(manifest); broken["context"]["settings"]["denominator"] = 2
            write_json(output / "run_manifest.json", broken)
            with self.assertRaisesRegex(ValueError, "Invalid signed metadata"):
                run.prepare_plan(resume)

    def test_test_requires_completed_search_and_rejects_config_or_source_override(self):
        with tempfile.TemporaryDirectory() as tmp, isolated_run(Path(tmp).resolve()):
            root = Path(tmp).resolve()
            with self.assertRaisesRegex(ValueError, "explicit completed"):
                run.prepare_plan(run.parser().parse_args(["--stage", "test"]))
            parent = root / "artificial/fgw_grid_parent"
            write_json(parent / "run_manifest.json", state.signed({"version": state.VERSION, "stage": "search",
                "context": {"settings": scoring.SETTINGS, "split": "valid"}}))
            base = ["--stage", "test", "--search-root", str(parent), "--output-root", "artificial/fgw_grid_test"]
            with self.assertRaises(FileNotFoundError):
                run.prepare_plan(run.parser().parse_args(base))
            with patch.object(run, "verify_run", side_effect=AssertionError("test config reached parent evaluation")):
                with self.assertRaisesRegex(ValueError, "Test cannot change"):
                    run.prepare_plan(run.parser().parse_args([*base, "--config", str(root / "irrelevant.json")]))
            settings = scoring.validate_settings({"denominator": 1})
            specs = scoring.specs(settings); name = next(iter(specs))
            manifest = {"context": {"settings": settings, "source_run": str(root / "frozen_source")}}
            chosen = {"winner": name, "selected_spec": specs[name]}
            for filename in ("completed.json", "selection.json"): write_json(parent / filename, {})
            with patch.object(run, "verify_run", return_value=(manifest, chosen)) as checked:
                output, plan, frozen, _ = run.prepare_plan(run.parser().parse_args(base))
                checked.assert_called_once_with(parent, required_stage="search")
                self.assertEqual(plan["specs"], {name: specs[name]})
                self.assertEqual(plan["settings"], settings)
                self.assertEqual(frozen, chosen)
                self.assertEqual(plan["split"], "test")
                self.assertFalse(output.exists())
                with self.assertRaisesRegex(ValueError, "retain"):
                    run.prepare_plan(run.parser().parse_args([*base, "--source-run", "different_source"]))

    def test_worker_bounds_are_checked_without_loading(self):
        with tempfile.TemporaryDirectory() as tmp, isolated_run(Path(tmp).resolve()), patch.object(sources, "load_inputs", side_effect=AssertionError("loaded")):
            for number in ("0", "9", "-1"):
                with self.subTest(number=number), self.assertRaisesRegex(ValueError, "bounded"):
                    run.main(["--workers", number, "--dry-run"])

    def test_postseal_failure_preserves_status_and_resume_only_restores_missing_proposal(self):
        class Pool:
            gallery = np.arange(3599)
            candidates_per_query = 3598
            def __len__(self): return 500
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp).resolve() / "wbm-region-retrieval"
            root.mkdir()
            stack.enter_context(isolated_run(root))
            config = root / "small_grid.json"
            write_json(config, {"denominator": 1, "families": ["sum_raw"], "include_reference": False})
            output = root / "artificial/fgw_grid_recovery"
            stack.enter_context(patch.object(run, "code_identity", return_value={}))
            stack.enter_context(patch.object(run.subprocess, "check_output", return_value="synthetic_commit"))
            stack.enter_context(patch.object(sources, "load_inputs", return_value=(object(), Pool(), object(), object(), {})))
            inventory = stack.enter_context(patch.object(sources, "inventory", return_value={}))
            stack.enter_context(patch.object(sources, "audit_components", return_value={"status": "verified"}))
            build = stack.enter_context(patch.object(cache, "build_cache", return_value=(object(), {})))
            def fake_evaluate(folder, signature, provider, pool, bundle, specs, settings, **kwargs):
                index = {"signature": signature, "split": "valid", "variants": specs, "summary_rows": grid_rows(specs)}
                write_json(folder / "evaluation_index_valid.json", index)
                return index
            evaluation = stack.enter_context(patch.object(pipeline, "evaluate", side_effect=fake_evaluate))
            stack.enter_context(patch.object(run, "verify_run", return_value=({}, {})))
            with patch.object(register, "write_proposal", side_effect=OSError("after-seal proposal failure")), redirect_stdout(StringIO()):
                with self.assertRaisesRegex(OSError, "after-seal"):
                    run.main(["--config", str(config), "--output-root", str(output), "--workers", "1"])
            self.assertEqual(read_json(output / "status.json")["status"], "completed")
            marker = read_json(output / "completed.json")
            self.assertEqual(marker["files"]["status.json"], file_hash(output / "status.json"))
            before = snapshot(output)
            def restore_proposal(folder):
                path = folder / "catalog_sources.json"
                if not path.exists(): write_json(path, {"synthetic_recovery_proposal": True})
                return path
            with patch.object(register, "write_proposal", side_effect=restore_proposal), redirect_stdout(StringIO()):
                self.assertEqual(run.main(["--resume-root", str(output), "--workers", "1"]), 0)
                after = snapshot(output)
                self.assertEqual({k: after[k] for k in before}, before)
                self.assertEqual(set(after) - set(before), {"catalog_sources.json"})
                self.assertEqual(run.main(["--resume-root", str(output), "--workers", "1"]), 0)
                self.assertEqual(snapshot(output), after)
            self.assertEqual(evaluation.call_count, 1)
            self.assertEqual(build.call_count, 1)
            self.assertEqual(inventory.call_count, 2)


class StateTests(unittest.TestCase):
    def test_signed_metadata_and_bound_paths_detect_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); path = root / "metadata.json"
            value = state.signed({"settings": {"F": .5}, "stage": "search"})
            state.record(path, value); before = snapshot(root)
            self.assertEqual(state.read_signed(path), value)
            state.record(path, value); self.assertEqual(snapshot(root), before)
            with self.assertRaises(ValueError): state.record(path, state.signed({"stage": "test"}))
            state.checked_files(root, {"metadata.json": file_hash(path)}, {"metadata.json"})
            for files in ({}, {"../outside": "bad"}, {str(path): file_hash(path)}):
                with self.assertRaises(ValueError): state.checked_files(root, files, {"metadata.json"})
            changed = deepcopy(value); changed["settings"]["F"] = 1.
            write_json(path, changed)
            with self.assertRaisesRegex(ValueError, "Invalid signed"):
                state.read_signed(path)

    def test_grid_table_is_stable_and_different_values_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); settings = scoring.validate_settings({"denominator": 2})
            specs = scoring.specs(settings); index = {"variants": specs, "summary_rows": grid_rows(specs)}
            run.write_grid_table(root, index); before = snapshot(root)
            run.write_grid_table(root, index); self.assertEqual(snapshot(root), before)
            with (root / "grid_results.csv").open(newline="") as handle: rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(specs))
            self.assertEqual([row["variant"] for row in rows], list(specs))
            changed = deepcopy(index); changed["summary_rows"][0]["mAP@100"] = .3
            with self.assertRaisesRegex(ValueError, "summary changed"): run.write_grid_table(root, changed)
            self.assertEqual(snapshot(root), before)

    def test_interrupted_grid_table_publish_can_be_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); specs = scoring.specs(scoring.validate_settings({"denominator": 1}))
            index = {"variants": specs, "summary_rows": grid_rows(specs)}
            original = Path.write_text
            def interrupted(path, text, *args, **kwargs):
                if text.startswith("variant,family,split,"):
                    original(path, text[:31], *args, **kwargs)
                    raise InterruptedError("interrupted summary publication")
                return original(path, text, *args, **kwargs)
            with patch.object(Path, "write_text", new=interrupted), self.assertRaises(InterruptedError):
                run.write_grid_table(root, index)
            run.write_grid_table(root, index)
            with (root / "grid_results.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), len(specs))

    def test_select_payload_binds_complete_grid_integer_hit_winner_and_index_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = scoring.validate_settings({"denominator": 2}); specs = scoring.specs(settings)
            manifest = state.signed({"stage": "search", "context": {"settings": settings}})
            index = {"split": "valid", "variants": specs, "summary_rows": grid_rows(specs)}
            path = Path(tmp).resolve() / "index.json"; write_json(path, index)
            selected = state.select_payload(manifest, index, path)
            pure = next(k for k, v in specs.items() if v["family"] == "sum_raw" and v.get("ticks") == [2, 0, 0])
            self.assertEqual(selected["winner"], pure)
            self.assertEqual(selected["evaluation_index_sha256"], file_hash(path))
            self.assertEqual(selected["parent_signature"], manifest["signature"])
            self.assertFalse(selected["test_used_for_selection"])
            alternate = next(k for k, v in specs.items() if v["family"] == "sum_minmax" and v.get("ticks") == [1, 0, 1])
            changed = deepcopy(index)
            row = next(r for r in changed["summary_rows"] if r["variant"] == alternate)
            row.update(precision10_hits=2501, **{"Precision@10": .5002, "mAP@100": 0., "Coverage-nDCG@10": 0.})
            write_json(path, changed)
            self.assertEqual(state.select_payload(manifest, changed, path)["winner"], alternate)
            for damaged in (dict(index, split="test"), dict(index, variants={}), dict(index, summary_rows=index["summary_rows"][:-1])):
                with self.assertRaises(ValueError): state.select_payload(manifest, damaged, path)

    def test_verify_sealed_search_is_read_only_and_missing_or_stale_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, manifest, index, chosen = sealed_fixture(Path(tmp).resolve() / "wbm-region-retrieval")
            before = snapshot(output)
            with patch.object(cache, "FullScoreCache", return_value=metadata_provider(output, index)) as cache_constructor:
                self.assertEqual(state.verify_run(output, required_stage="search"), (manifest, chosen))
                cache_constructor.assert_called_once_with(output / "component_cache/valid")
            self.assertEqual(snapshot(output), before)
            with self.assertRaisesRegex(ValueError, "Unexpected run stage"):
                state.verify_run(output, required_stage="test", check_cache=False)
            path = output / "source_audit.json"
            original = read_json(path)
            write_json(path, state.signed({"status": "verified", "context_digest": "stale"}))
            seal_root(output, manifest, len(index["variants"]))
            with self.assertRaisesRegex(ValueError, "provenance differs"):
                state.verify_run(output, check_cache=False)
            write_json(path, original); seal_root(output, manifest, len(index["variants"]))
            done = read_json(output / "completed.json"); del done["files"]["selection.json"]
            write_json(output / "completed.json", done)
            with self.assertRaisesRegex(ValueError, "omits required"):
                state.verify_run(output, check_cache=False)

    def test_cache_query_identity_and_missing_grid_table_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, manifest, index, chosen = sealed_fixture(Path(tmp).resolve() / "wbm-region-retrieval")
            provider = metadata_provider(output, index)
            provider.query_ids = provider.query_ids[::-1]
            with patch.object(cache, "FullScoreCache", return_value=provider):
                with self.assertRaisesRegex(ValueError, "query/gallery identity"):
                    state.verify_run(output)
            (output / "grid_results.csv").unlink()
            seal_root(output, manifest, len(index["variants"]))
            with self.assertRaisesRegex(ValueError, "omits required"):
                state.verify_run(output, check_cache=False)


if __name__ == "__main__":
    unittest.main()
