#!/usr/bin/env python3
"""Register the complete, independently verified valid-only regional score grid."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from b1_structure_fusion.register import _atomic_write, _json_bytes, _lock
from f_region_structure.register import (_check_catalog, _complete, _finite_error,
                                         _fingerprint, _metric_values, _relative)
from f_region_structure.state import TZ, read_signed
from f_region_structure.verify import REPORT_FILES
from results_catalog import build as catalog
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from . import scoring, selection

KS = [1, 5, 10, 20, 50, 100, 500]


def _evaluation(output, root, name, spec, entry, context_signature):
    folder = output / "valid" / name
    signature = digest({"context": context_signature, "split": "valid", "variant": name, "spec": spec})
    _complete(folder, signature, (*REPORT_FILES, "run_identity.json", "verification.json"))
    marker = folder / "completed.json"
    if entry.get("completed_sha256") != file_hash(marker) or entry.get("num_queries") != 500:
        raise ValueError("Evaluation index completion binding differs")
    identity = read_json(folder / "run_identity.json")
    evaluation = read_json(folder / "evaluation_manifest.json")
    if (evaluation.get("signature") != signature or evaluation.get("spec") != spec
            or evaluation.get("variant") != name or evaluation.get("split") != "valid"
            or evaluation.get("ks") != KS or evaluation.get("tie_seed") != 2026
            or evaluation.get("num_queries") != 500 or evaluation.get("gallery_count") != 3599
            or evaluation.get("candidate_rule") != "all_gallery_except_query"
            or evaluation.get("retrieval_protocol_signature") != catalog.BASE_PROTOCOL_SIGNATURE
            or evaluation.get("checkpoint_kind") != "best_nce" or evaluation.get("source_seed") != 2026
            or evaluation.get("matching") != "original_F_greedy_sqrt_area"
            or evaluation.get("test_parameter_search") is not False
            or evaluation.get("test_evaluated") is not False
            or evaluation.get("posthoc_research") is not True
            or evaluation.get("training_required") is not False
            or identity.get("evaluation_signature") != signature
            or not identity.get("key", "").startswith("f_region_score_search_")
            or any(evaluation.get(k) != v for k, v in identity.items())):
        raise ValueError("Valid-only score manifest identity or protocol differs")
    checked = read_json(folder / "verification.json")
    files = checked.get("files", {})
    if (checked.get("status") != "verified" or checked.get("signature") != signature
            or checked.get("split") != "valid" or checked.get("variant") != name
            or checked.get("ks") != KS or checked.get("tie_seed") != 2026
            or checked.get("num_queries") != 500 or checked.get("ranking_rows") != 250000
            or checked.get("full_candidates_checked") != 1799000
            or checked.get("all_queries_full_candidate_topk_exact") is not True
            or type(checked.get("aggregation_values_checked")) is not int or checked["aggregation_values_checked"] <= 0
            or set(files) != set(REPORT_FILES) | {"run_identity.json"}
            or any(file_hash(folder / filename) != sha for filename, sha in files.items())
            or checked.get("ranking_sha256") != file_hash(folder / "rankings.csv")):
        raise ValueError("Independent verification is missing, incomplete or stale")
    _finite_error(checked.get("max_metric_absolute_error"), 1e-12, "metric")
    _finite_error(checked.get("max_saved_score_absolute_error"), 1e-12, "saved_score")
    hits = checked.get("top10_hits")
    if type(hits) is not int or not 0 <= hits <= 5000 or entry.get("top10_hits") != hits:
        raise ValueError("Integer Top-10 selection objective differs from verified ranking")
    for family, filename in (("exact", "metrics.json"), ("coverage", "coverage_metrics.json")):
        report = read_json(folder / filename)
        _metric_values(report.get("metrics"), family)
        metadata = report.get("metadata", {})
        index_name = "metrics" if family == "exact" else "coverage_metrics"
        if (report.get("split") != "valid" or report.get("num_queries") != 500 or report.get("ks") != KS
                or report.get("metric_version") != family + "_v1" or report.get("test_evaluated") is not False
                or metadata.get("signature") != signature or metadata.get("smoke") or metadata.get("derived_protocol")
                or report.get("retrieval_protocol_signature", metadata.get("retrieval_protocol_signature")) != catalog.BASE_PROTOCOL_SIGNATURE
                or entry.get(index_name) != report.get("metrics")):
            raise ValueError("Formal metric report identity or index differs")
        if family == "exact" and abs(report["metrics"].get("Precision@10", -1) - hits / 5000) > 1e-12:
            raise ValueError("Precision@10 differs from verified integer hits")
    item = {"key": identity["key"], "repository": "wbm-region-retrieval", "path": _relative(folder, root),
            "category": "fusion_ablation", "method": "F", "variant": "RegionalJScore/" + name,
            "split": "valid", "manifest": _relative(folder / "evaluation_manifest.json", root),
            "marker": _relative(marker, root), "identity_schema_version": 1,
            "identity": _relative(folder / "run_identity.json", root),
            "selection": _relative(output / "selection.json", root), "coverage": None,
            "supervision": "untrained" if name == "j_only" else "self_supervised",
            "display": "区域J＋面积（无CNN评分）" if name == "j_only" else "区域结构评分 " + name,
            "required_families": ["exact", "coverage"], "protocol": "retrieval_v1", "posthoc": True,
            "supporting_manifests": [_relative(output / p, root) for p in
                ("run_manifest.json", "completed.json", "inputs.json", "source_protection_after.json",
                 "baseline_replay.json", "formula_audit_valid.json", "evaluation_index_valid.json",
                 f"valid/{name}/verification.json")],
            "design_note": "Valid-only frozen 15-candidate comparison of multiplicative, mixed arithmetic, zero-gated arithmetic and CNN/structure endpoint regional scores. Reuses the previous all-point regional joint descriptors, frozen cc8 Top-4 h512 best_nce seed2026, area factor, original greedy matching and query sqrt(area) aggregation. The j_only endpoint uses regional J plus area without CNN scoring, and is untrained; other variants reuse the frozen self-supervised CNN. No training and no test evaluation. Integer valid exact P@10 with a fixed tie rule selects the finite-grid winner. Product baseline exactly replays the previously selected regional-J formula. Research follows historical test observations; no test-generalization claim."}
    catalog.execution_identity(root, item, marker, evaluation)
    return item


def proposal(output, *, root=ROOT):
    output, root = Path(output).resolve(), Path(root).resolve()
    _relative(output, root / "artificial")
    if output == root / "artificial":
        raise ValueError("Expected an isolated experiment output")
    manifest = read_signed(output / "run_manifest.json")
    context = manifest.get("context", {})
    context_signature = digest(context)
    required = ("run_manifest.json", "status.json", "selection.json", "evaluation_index_valid.json",
                "inputs.json", "formula_audit_valid.json", "source_protection_before.json",
                "source_protection_after.json", "baseline_replay.json")
    _complete(output, context_signature, required)
    status = read_json(output / "status.json")
    grid = scoring.frozen_grid()
    if (manifest.get("version") != "f_region_score_search_v1" or context.get("version") != "f_region_score_search_v1"
            or context.get("grid") != grid or context.get("selection_rule") != selection.RULE
            or context.get("split") != "valid" or context.get("test_evaluated") is not False
            or context.get("training") is not False or context.get("posthoc_design") is not True
            or status.get("status") != "completed" or status.get("split") != "valid"
            or status.get("test_evaluated") is not False or status.get("context_signature") != context_signature):
        raise ValueError("Registration requires the complete valid-only score search without training")
    if any((output / name).exists() for name in ("test", "training", "evaluation_index_test.json")):
        raise ValueError("Unexpected test or training artifacts in a valid-only search")
    source_binding = context.get("source_binding")
    inputs = read_json(output / "inputs.json")
    if (not isinstance(source_binding, dict) or not source_binding or inputs.get("source_binding") != source_binding
            or inputs.get("context_signature") != context_signature):
        raise ValueError("Frozen input source binding differs from the scientific context")
    for name, expected_hash in source_binding.items():
        path = Path(name)
        if (not path.is_absolute() or path.is_symlink() or not path.is_file()
                or file_hash(path) != expected_hash):
            raise ValueError("Frozen input source is missing or changed: " + name)
    before = read_json(output / "source_protection_before.json")
    after = read_json(output / "source_protection_after.json")
    if after.get("status") != "unchanged" or after.get("files") != len(before):
        raise ValueError("Historical source protection evidence failed")
    baseline = read_json(output / "baseline_replay.json")
    if (baseline.get("status") != "verified" or baseline.get("rows") != 250000
            or baseline.get("max_score_error") != 0 or baseline.get("ranking_byte_identical") is not True):
        raise ValueError("Frozen regional-J product baseline replay is incomplete or inexact")
    if not isinstance(baseline.get("source"), str) or not baseline["source"]:
        raise ValueError("Missing baseline ranking source")
    original = Path(baseline["source"])
    original = original if original.is_absolute() else root / original
    _relative(original, root)
    if (not original.is_file() or original.resolve().is_relative_to(output)
            or file_hash(original) != baseline.get("source_sha256")
            or file_hash(output / "valid/product/rankings.csv") != baseline.get("source_sha256")):
        raise ValueError("Original baseline ranking changed or is missing")
    selected = read_signed(output / "selection.json")
    index = read_json(output / "evaluation_index_valid.json")
    if (selected.get("context_signature") != context_signature or selected.get("synthetic") is not False
            or selected.get("valid_index_sha256") != file_hash(output / "evaluation_index_valid.json")
            or index.get("split") != "valid" or index.get("specs") != grid
            or index.get("context_signature") != context_signature):
        raise ValueError("Frozen valid selection does not bind the full predefined grid")
    expected = selection.select_best(index.get("results"), split="valid")
    if any(selected.get(k) != v for k, v in expected.items()) or status.get("winner") != selected["winner"]:
        raise ValueError("Frozen winner differs from integer valid objective and tie rules")
    if index["results"]["product"]["top10_hits"] != 2494:
        raise ValueError("Product baseline must replay the old 2494/5000 valid Top-10 hits")
    actual = {p.name for p in (output / "valid").iterdir() if p.is_dir()}
    if actual != set(grid):
        raise ValueError("Unexpected extra or missing valid evaluation directories")
    audit = read_json(output / "formula_audit_valid.json")
    representatives = audit.get("query_ids", [])
    if (audit.get("status") != "verified" or len(representatives) != 37
            or len(set(representatives)) != 37 or audit.get("candidate_pairs") != 37 * 20 * len(grid)):
        raise ValueError("Independent regional formula/greedy audit is incomplete")
    _finite_error(audit.get("max_absolute_error"), 2e-6, "regional_formula")
    rows = [_evaluation(output, root, name, spec, index["results"][name], context_signature) for name, spec in grid.items()]
    if len(rows) != 15 or len({row["key"] for row in rows}) != 15:
        raise ValueError("Expected 15 valid evaluations with unique identities")
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


def apply_proposal(output, *, workspace=None, root=ROOT):
    """Collect/render first, keep every old run identical, atomically commit or rollback."""
    output, root = Path(output).resolve(), Path(root).resolve()
    workspace = root.parent if workspace is None else Path(workspace).resolve()
    if (workspace / "wbm-region-retrieval").resolve() != root:
        raise ValueError("Workspace must contain this wbm-region-retrieval repository")
    proposed = read_json(write_proposal(output, root=root))
    folder = root / "results_catalog"
    with _lock(folder):
        original = {name: (folder / name).read_bytes() for name in ("sources.json", "snapshot.json")}
        sources, previous = (json.loads(original[name]) for name in ("sources.json", "snapshot.json"))
        for name in _check_catalog(folder, sources, previous):
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
        compact = (json.dumps(updated, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        contents = {"snapshot.json": compact, **{name: value.encode("utf-8") for name, value in generated.items()},
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
