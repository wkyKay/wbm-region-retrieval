#!/usr/bin/env python3
"""Register sealed regional-J searches plus their single frozen test winner."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from b1_structure_fusion.register import _atomic_write, _json_bytes, _lock
from results_catalog import build as catalog
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from . import scoring, selection
from .state import TZ, read_signed, verified
from .verify import REPORT_FILES

KS = [1, 5, 10, 20, 50, 100, 500]
FAMILIES = ("plain_mlp", "gated_mlp")


def _relative(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    try:
        return str(path.relative_to(root))
    except ValueError as error:
        raise ValueError("Registration source escapes its repository") from error


def _complete(folder, signature, required):
    if not verified(folder, signature):
        raise ValueError("Registration requires a completed sealed output")
    done = read_json(Path(folder) / "completed.json")
    if not set(required) <= set(done.get("files", {})):
        raise ValueError("Completion marker omits required evidence")
    return done


def _finite_error(value, maximum, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= maximum:
        raise ValueError("Invalid verification error: " + name)


def _metric_values(values, family):
    names = (("Hit", "Precision", "Recall", "mAP") if family == "exact" else
             ("Coverage-nDCG", "MeanCoverage", "FullCoverageHit", "FullCoveragePrecision",
              "FullCoverageRecall", "FullCoverage_mAP"))
    if not isinstance(values, dict) or set(values) != {f"{name}@{k}" for name in names for k in KS}:
        raise ValueError("Incomplete metric grid")
    for value in values.values():
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or not 0 <= value <= 1 + 1e-12):
            raise ValueError("Invalid metric value")


def _models(output, context, selected):
    models = read_json(output / "model_binding.json")
    if models != selected.get("models") or set(models) != set(FAMILIES):
        raise ValueError("Selected fusion model bindings differ")
    binding = read_json(output / "training_inputs.json")
    if (binding.get("labels_used") is not False or binding.get("fixed_epochs") != 20
            or binding.get("checkpoint_rule") != "final_epoch" or binding.get("seed") != 2026):
        raise ValueError("Fusion training input policy differs")
    for family in FAMILIES:
        folder = output / "training" / family
        manifest = read_signed(folder / "training_manifest.json")
        done = _complete(folder, manifest["signature"], ("training_manifest.json", "final.pt", "loss.csv"))
        settings = manifest.get("settings", {})
        if (file_hash(folder / "final.pt") != models[family]
                or manifest.get("version") != "regional_joint_learning_v1"
                or manifest.get("split") != "train" or manifest.get("test_used") is not False
                or settings.get("family") != family or settings.get("split") != "train"
                or settings.get("epochs") != 20 or settings.get("batch_size") != 512
                or settings.get("seed") != 2026 or settings.get("checkpoint_rule") != "last_epoch"
                or settings.get("labels_used") is not False or settings.get("CNN_frozen") is not True
                or settings.get("area_normalization", {}).get("fit_split") != "train"
                or done.get("epoch") != 20 or done.get("checkpoint_rule") != "last_epoch"
                or manifest.get("code_sha256") != context.get("code", {}).get("f_region_structure/learning.py")):
            raise ValueError("Fusion model is not the bound train-only final checkpoint")


def _evaluation(output, root, split, name, spec, entry, context_signature):
    folder = output / split / name
    nquery, ngallery = (500, 3599) if split == "valid" else (7199, 7199)
    signature = digest({"context": context_signature, "split": split, "variant": name, "spec": spec})
    required = (*REPORT_FILES, "run_identity.json", "verification.json")
    _complete(folder, signature, required)
    marker = folder / "completed.json"
    if entry.get("completed_sha256") != file_hash(marker) or entry.get("num_queries") != nquery:
        raise ValueError("Evaluation index completion binding differs")
    identity = read_json(folder / "run_identity.json")
    evaluation = read_json(folder / "evaluation_manifest.json")
    if (evaluation.get("signature") != signature or evaluation.get("spec") != spec
            or evaluation.get("variant") != name or evaluation.get("split") != split
            or evaluation.get("ks") != KS or evaluation.get("tie_seed") != 2026
            or evaluation.get("num_queries") != nquery or evaluation.get("gallery_count") != ngallery
            or evaluation.get("candidate_rule") != "all_gallery_except_query"
            or evaluation.get("retrieval_protocol_signature") != catalog.BASE_PROTOCOL_SIGNATURE
            or evaluation.get("checkpoint_kind") != "best_nce" or evaluation.get("source_seed") != 2026
            or evaluation.get("matching") != "original_F_greedy_sqrt_area"
            or evaluation.get("test_parameter_search") is not False
            or evaluation.get("posthoc_research") is not True
            or evaluation.get("training_required") != (spec["family"] in FAMILIES)
            or identity.get("evaluation_signature") != signature
            or any(evaluation.get(k) != v for k, v in identity.items())):
        raise ValueError("Regional evaluation manifest identity or protocol differs")
    checked = read_json(folder / "verification.json")
    files = checked.get("files", {})
    if (checked.get("status") != "verified" or checked.get("signature") != signature
            or checked.get("split") != split or checked.get("variant") != name
            or checked.get("ks") != KS or checked.get("tie_seed") != 2026
            or checked.get("num_queries") != nquery or checked.get("ranking_rows") != nquery * 500
            or checked.get("full_candidates_checked") != nquery * (ngallery - 1)
            or checked.get("all_queries_full_candidate_topk_exact") is not True
            or not isinstance(checked.get("aggregation_values_checked"), int)
            or checked["aggregation_values_checked"] <= 0
            or set(files) != set(REPORT_FILES) | {"run_identity.json"}
            or any(file_hash(folder / filename) != sha for filename, sha in files.items())
            or checked.get("ranking_sha256") != file_hash(folder / "rankings.csv")):
        raise ValueError("Independent verification is missing, incomplete or stale")
    _finite_error(checked.get("max_metric_absolute_error"), 1e-12, "metric")
    _finite_error(checked.get("max_saved_score_absolute_error"), 1e-12, "saved_score")
    hits = checked.get("top10_hits")
    if type(hits) is not int or not 0 <= hits <= nquery * 10 or entry.get("top10_hits") != hits:
        raise ValueError("Integer Top-10 selection objective differs from verified ranking")
    for family, filename in (("exact", "metrics.json"), ("coverage", "coverage_metrics.json")):
        report = read_json(folder / filename)
        _metric_values(report.get("metrics"), family)
        metadata = report.get("metadata", {})
        entry_name = "metrics" if family == "exact" else "coverage_metrics"
        if (report.get("split") != split or report.get("num_queries") != nquery or report.get("ks") != KS
                or report.get("metric_version") != family + "_v1"
                or report.get("test_evaluated") != (split == "test")
                or metadata.get("signature") != signature or metadata.get("smoke")
                or metadata.get("derived_protocol")
                or report.get("retrieval_protocol_signature", metadata.get("retrieval_protocol_signature")) != catalog.BASE_PROTOCOL_SIGNATURE
                or entry.get(entry_name) != report.get("metrics")):
            raise ValueError("Formal metric report identity or index differs")
        if family == "exact" and abs(report["metrics"].get("Precision@10", -1) - hits / (nquery * 10)) > 1e-12:
            raise ValueError("Precision@10 differs from the verified integer hit total")
    item = {"key": identity["key"], "repository": "wbm-region-retrieval", "path": _relative(folder, root),
            "category": "fusion_ablation", "method": "F", "variant": "RegionalJ/" + name,
            "split": split, "manifest": _relative(folder / "evaluation_manifest.json", root),
            "marker": _relative(marker, root), "identity_schema_version": 1,
            "identity": _relative(folder / "run_identity.json", root),
            "selection": _relative(output / "selection.json", root), "coverage": None,
            "supervision": "self_supervised", "display": "区域结构 " + name,
            "required_families": ["exact", "coverage"], "protocol": "retrieval_v1", "posthoc": True,
            "supporting_manifests": [_relative(output / p, root) for p in
                ("run_manifest.json", "completed.json", "model_binding.json", "training_inputs.json",
                 "source_protection_after.json", "baseline_replay.json", f"formula_audit_{split}.json",
                 f"evaluation_index_{split}.json", f"{split}/{name}/verification.json")],
            "design_note": "Regional rho/stick/ball descriptors pooled after original-map tensor voting; no whole-wafer late score fusion. Frozen cc8 Top-4 h512 best_nce seed2026, original greedy matching and query sqrt(area) aggregation. Twenty-five predeclared valid candidates, including two train-only self-supervised 20-epoch fusion heads with final checkpoints. Integer valid exact P@10 and fixed tie rules select one recipe for test. Research follows historical test observations; finite-grid optimum only. Historical outputs remain unchanged."}
    catalog.execution_identity(root, item, marker, evaluation)
    return item


def proposal(output, *, root=ROOT):
    """Only a complete all/test run is registrable, including its entire valid grid."""
    output, root = Path(output).resolve(), Path(root).resolve()
    _relative(output, root / "artificial")
    if output == root / "artificial":
        raise ValueError("Expected an isolated experiment output")
    manifest = read_signed(output / "run_manifest.json")
    context = manifest.get("context", {})
    context_signature = digest(context)
    required = ("run_manifest.json", "status.json", "selection.json", "evaluation_index_valid.json",
                "evaluation_index_test.json", "formula_audit_valid.json", "formula_audit_test.json",
                "source_protection_before.json", "source_protection_after.json", "baseline_replay.json",
                "model_binding.json", "training_inputs.json")
    _complete(output, context_signature, required)
    status = read_json(output / "status.json")
    grid = scoring.frozen_grid()
    if (manifest.get("version") != "f_region_structure_v1" or context.get("version") != "f_region_structure_v1"
            or context.get("grid") != grid or context.get("selection_rule") != selection.RULE
            or context.get("test_parameter_search") is not False or context.get("posthoc_design") is not True
            or status.get("status") != "completed" or status.get("stage") not in ("all", "test")
            or status.get("test_evaluated") is not True or status.get("context_signature") != context_signature):
        raise ValueError("Registration requires a completed valid search and single test evaluation")
    before = read_json(output / "source_protection_before.json")
    after = read_json(output / "source_protection_after.json")
    if after.get("status") != "unchanged" or after.get("files") != len(before):
        raise ValueError("Historical source protection evidence failed")
    baseline = read_json(output / "baseline_replay.json")
    if baseline.get("status") != "verified" or baseline.get("rows") != 500 * 500:
        raise ValueError("Frozen original F baseline replay is incomplete")
    _finite_error(baseline.get("max_score_error"), 1e-12, "baseline")
    selected = read_signed(output / "selection.json")
    valid_index = read_json(output / "evaluation_index_valid.json")
    if (selected.get("context_signature") != context_signature or selected.get("synthetic") is not False
            or selected.get("valid_index_sha256") != file_hash(output / "evaluation_index_valid.json")
            or valid_index.get("split") != "valid" or valid_index.get("specs") != grid
            or valid_index.get("context_signature") != context_signature):
        raise ValueError("Frozen valid selection does not bind the full predefined grid")
    expected = selection.select_best(valid_index.get("results"), split="valid")
    if any(selected.get(k) != v for k, v in expected.items()) or status.get("winner") != selected["winner"]:
        raise ValueError("Frozen winner differs from integer valid objective and tie rules")
    _models(output, context, selected)
    rows = []
    for split, specs in (("valid", grid), ("test", {selected["winner"]: selected["selected_spec"]})):
        index = read_json(output / f"evaluation_index_{split}.json")
        if (index.get("split") != split or index.get("context_signature") != context_signature
                or index.get("specs") != specs or set(index.get("results", {})) != set(specs)):
            raise ValueError("Evaluation index differs from full valid grid or frozen test winner")
        actual = {p.name for p in (output / split).iterdir() if p.is_dir()}
        if actual != set(specs):
            raise ValueError("Unexpected extra or missing evaluation directories")
        audit = read_json(output / f"formula_audit_{split}.json")
        representatives = audit.get("query_ids", [])
        if (audit.get("status") != "verified" or len(representatives) != 37
                or len(set(representatives)) != 37 or audit.get("candidate_pairs") != 37 * 20 * len(specs)):
            raise ValueError("Independent regional formula/greedy audit is incomplete")
        _finite_error(audit.get("max_absolute_error"), 2e-6, "regional_formula")
        for name, spec in specs.items():
            rows.append(_evaluation(output, root, split, name, spec, index["results"][name], context_signature))
    if len(rows) != 26 or len({row["key"] for row in rows}) != 26:
        raise ValueError("Expected 25 valid plus one test evaluation with unique identities")
    return {"schema_version": 1, "runs": rows}


def write_proposal(output, *, root=ROOT):
    value = proposal(output, root=root)
    path = Path(output) / "catalog_sources.json"
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Registration proposal changed")
    else:
        write_json(path, value)
    return path


def _fingerprint(sources):
    return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()


def _check_catalog(folder, sources, snapshot):
    """The same source fingerprint and render comparisons used by catalog check."""
    if snapshot.get("source_spec_sha256") != _fingerprint(sources):
        raise ValueError("Catalogue sources and snapshot differ")
    generated = catalog.render(snapshot)
    for name, content in generated.items():
        path = folder / name
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            raise ValueError("Stale generated catalogue: " + name)
    return generated


def apply_proposal(output, *, workspace=None, root=ROOT):
    """Collect before writes; preserve every previous run deeply; rollback on failure."""
    output, root = Path(output).resolve(), Path(root).resolve()
    workspace = root.parent if workspace is None else Path(workspace).resolve()
    if (workspace / "wbm-region-retrieval").resolve() != root:
        raise ValueError("Workspace must contain this wbm-region-retrieval repository")
    proposed = read_json(write_proposal(output, root=root))
    folder = root / "results_catalog"
    with _lock(folder):
        original = {name: (folder / name).read_bytes() for name in ("sources.json", "snapshot.json")}
        sources, previous = (json.loads(original[name]) for name in ("sources.json", "snapshot.json"))
        generated = _check_catalog(folder, sources, previous)
        for name in generated:
            original[name] = (folder / name).read_bytes()
        entries = {row["key"]: row for row in sources["runs"]}
        old_runs = {row["key"]: row for row in previous["runs"]}
        if len(entries) != len(sources["runs"]) or set(old_runs) != set(entries) or len(old_runs) != len(previous["runs"]):
            raise ValueError("Existing catalogue keys are duplicated or inconsistent")
        additions = []
        for item in proposed["runs"]:
            if item["key"] in entries:
                if entries[item["key"]] != item:
                    raise ValueError("Conflicting registered key: " + item["key"])
            else:
                additions.append(item)
        if not additions:
            return {"status": "already_registered", "new_results": 0, "old_results_unchanged": len(old_runs)}
        merged = {**sources, "runs": [*sources["runs"], *additions]}
        updated = catalog.collect(workspace, merged, previous)
        updated_rows = {row["key"]: row for row in updated["runs"]}
        if any(updated_rows.get(key) != value for key, value in old_runs.items()):
            raise ValueError("Collection changed a previously registered result")
        if len(updated_rows) != len(updated["runs"]) or set(updated_rows) != set(entries) | {row["key"] for row in additions}:
            raise ValueError("Collected result identities differ from merged sources")
        if updated.get("source_spec_sha256") != _fingerprint(merged):
            raise ValueError("Collected source fingerprint differs")
        generated = catalog.render(updated)
        if any(not (folder / name).is_file() or (folder / name).read_bytes() != content for name, content in original.items()):
            raise ValueError("Catalogue changed concurrently during collection")
        contents = {"snapshot.json": (json.dumps(updated, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"),
                    **{name: content.encode("utf-8") for name, content in generated.items()},
                    "sources.json": _json_bytes(merged)}
        if any(name not in original and (folder / name).exists() for name in contents):
            raise ValueError("New generated file would overwrite an untracked catalogue file")
        before = {name: original.get(name) for name in contents}
        parent = output / "catalog_backup"
        parent.mkdir(exist_ok=True)
        backup = Path(tempfile.mkdtemp(prefix=datetime.now(TZ).strftime("%Y%m%dT%H%M%S%z") + "_", dir=parent))
        for name, content in before.items():
            if content is not None:
                path = backup / "files" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
        audit = {"schema_version": 1, "status": "prepared", "old_results_unchanged": len(old_runs),
                 "new_keys": [row["key"] for row in additions],
                 "before_sha256": {name: hashlib.sha256(value).hexdigest() if value is not None else None for name, value in before.items()},
                 "after_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()}}
        write_json(backup / "manifest.json", audit)
        written = []
        try:
            for name, content in contents.items():
                if before[name] != ((folder / name).read_bytes() if (folder / name).exists() else None):
                    raise ValueError("Catalogue changed concurrently: " + name)
                written.append(name)
                _atomic_write(folder / name, content)
            for name, content in contents.items():
                if (folder / name).read_bytes() != content:
                    raise ValueError("Catalogue write verification failed: " + name)
            _check_catalog(folder, read_json(folder / "sources.json"), read_json(folder / "snapshot.json"))
        except BaseException as error:
            failures = []
            for name in reversed(written):
                try:
                    if before[name] is None:
                        (folder / name).unlink(missing_ok=True)
                    else:
                        _atomic_write(folder / name, before[name])
                except BaseException as restoration:
                    failures.append(f"{name}: {restoration}")
            audit.update(status="rollback_failed" if failures else "rolled_back", error=str(error), rollback_errors=failures)
            write_json(backup / "manifest.json", audit)
            if failures:
                raise RuntimeError(f"Catalogue rollback incomplete; backups at {backup}: {failures}") from error
            raise
        audit["status"] = "committed"
        write_json(backup / "manifest.json", audit)
        return {"status": "registered", "new_results": len(additions), "old_results_unchanged": len(old_runs), "backup": str(backup)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--workspace")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    output = (ROOT / args.run_root).resolve()
    result = apply_proposal(output, workspace=args.workspace) if args.apply else str(write_proposal(output))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
