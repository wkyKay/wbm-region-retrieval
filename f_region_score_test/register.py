#!/usr/bin/env python3
"""Register one full test evaluation of the immutable valid-selected regional score."""
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
from f_region_score_search import register as valid_registration
from f_region_structure.register import (_check_catalog, _complete, _finite_error,
                                         _fingerprint, _metric_values, _relative)
from f_region_structure.state import TZ, read_signed
from f_region_structure.verify import REPORT_FILES
from results_catalog import build as catalog
from wbm_retrieval.common import digest, file_hash, read_json, write_json

KS = [1, 5, 10, 20, 50, 100, 500]
NAME = "mixed_a0p5_e1p0"
SPEC = {"family": "mixed", "descriptor": "all_points", "strength": .5, "area": True,
        "alpha": .5, "eta": 1., "gate_zero": False}
POLICY = "frozen_valid_winner_only"


def _frozen_selection(output, root, context, binding):
    path = output / "frozen_selection.json"
    frozen = read_signed(path)
    if (frozen.get("schema_version") != 1 or frozen.get("winner") != NAME
            or frozen.get("selected_spec") != SPEC or frozen.get("selection_policy") != POLICY
            or context.get("frozen_selection_sha256") != file_hash(path)
            or frozen.get("source_run") != context.get("source_run")):
        raise ValueError("Frozen valid selection wrapper differs from the test context")
    source = Path(frozen["source_run"]).resolve()
    _relative(source, root / "artificial")
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Test output must be isolated from its valid source")
    expected_files = {"selection.json": "source_selection_sha256", "completed.json": "source_completed_sha256",
                      "evaluation_index_valid.json": "source_valid_index_sha256"}
    for name, field in expected_files.items():
        original = source / name
        expected = frozen.get(field)
        if not expected or file_hash(original) != expected or binding.get(str(original)) != expected:
            raise ValueError("Valid source selection/completion/index SHA binding differs")
    original_selection = read_signed(source / "selection.json")
    if (frozen.get("selection") != original_selection or original_selection.get("winner") != NAME
            or original_selection.get("selected_spec") != SPEC or original_selection.get("synthetic") is not False):
        raise ValueError("Copied selection is not the original valid winner")
    # Pure read-only validation: never call the old writer or registrar here.
    # It rechecks all 15 valid results, original protocol and integer selection.
    proposed = valid_registration.proposal(source, root=root)
    if len(proposed["runs"]) != 15 or any(row["split"] != "valid" for row in proposed["runs"]):
        raise ValueError("Frozen source must be the complete valid-only grid")
    for name, field in expected_files.items():
        if file_hash(source / name) != frozen[field]:
            raise ValueError("Frozen valid source changed during registration validation")
    return frozen


def _evaluation(output, root, entry, context_signature):
    folder = output / "test" / NAME
    signature = digest({"context": context_signature, "split": "test", "variant": NAME, "spec": SPEC})
    _complete(folder, signature, (*REPORT_FILES, "run_identity.json", "verification.json"))
    marker = folder / "completed.json"
    if entry.get("completed_sha256") != file_hash(marker) or entry.get("num_queries") != 7199:
        raise ValueError("Test index completion binding differs")
    identity = read_json(folder / "run_identity.json")
    evaluation = read_json(folder / "evaluation_manifest.json")
    if (evaluation.get("signature") != signature or evaluation.get("spec") != SPEC
            or evaluation.get("variant") != NAME or evaluation.get("split") != "test"
            or evaluation.get("ks") != KS or evaluation.get("tie_seed") != 2026
            or evaluation.get("num_queries") != 7199 or evaluation.get("gallery_count") != 7199
            or evaluation.get("candidate_rule") != "all_gallery_except_query"
            or evaluation.get("retrieval_protocol_signature") != catalog.BASE_PROTOCOL_SIGNATURE
            or evaluation.get("checkpoint_kind") != "best_nce" or evaluation.get("source_seed") != 2026
            or evaluation.get("matching") != "original_F_greedy_sqrt_area"
            or evaluation.get("test_parameter_search") is not False or evaluation.get("test_evaluated") is not True
            or evaluation.get("selection_split") != "valid" or evaluation.get("posthoc_research") is not True
            or evaluation.get("training_required") is not False or identity.get("evaluation_signature") != signature
            or not identity.get("key", "").startswith("f_region_score_test_")
            or any(evaluation.get(k) != v for k, v in identity.items())):
        raise ValueError("Frozen-winner test manifest identity or protocol differs")
    checked = read_json(folder / "verification.json")
    files = checked.get("files", {})
    if (checked.get("status") != "verified" or checked.get("signature") != signature
            or checked.get("split") != "test" or checked.get("variant") != NAME
            or checked.get("ks") != KS or checked.get("tie_seed") != 2026
            or checked.get("num_queries") != 7199 or checked.get("ranking_rows") != 3599500
            or checked.get("full_candidates_checked") != 51818402
            or checked.get("all_queries_full_candidate_topk_exact") is not True
            or type(checked.get("aggregation_values_checked")) is not int or checked["aggregation_values_checked"] <= 0
            or set(files) != set(REPORT_FILES) | {"run_identity.json"}
            or any(file_hash(folder / filename) != sha for filename, sha in files.items())
            or checked.get("ranking_sha256") != file_hash(folder / "rankings.csv")):
        raise ValueError("Full-test independent verification is missing, incomplete or stale")
    _finite_error(checked.get("max_metric_absolute_error"), 1e-12, "metric")
    _finite_error(checked.get("max_saved_score_absolute_error"), 1e-12, "saved_score")
    hits = checked.get("top10_hits")
    if type(hits) is not int or not 0 <= hits <= 71990 or entry.get("top10_hits") != hits:
        raise ValueError("Reported test Top-10 hits differ from verified ranking")
    for family, filename in (("exact", "metrics.json"), ("coverage", "coverage_metrics.json")):
        report = read_json(folder / filename)
        _metric_values(report.get("metrics"), family)
        metadata = report.get("metadata", {})
        index_name = "metrics" if family == "exact" else "coverage_metrics"
        if (report.get("split") != "test" or report.get("num_queries") != 7199 or report.get("ks") != KS
                or report.get("metric_version") != family + "_v1" or report.get("test_evaluated") is not True
                or metadata.get("signature") != signature or metadata.get("smoke") or metadata.get("derived_protocol")
                or report.get("retrieval_protocol_signature", metadata.get("retrieval_protocol_signature")) != catalog.BASE_PROTOCOL_SIGNATURE
                or entry.get(index_name) != report.get("metrics")):
            raise ValueError("Formal test metric report identity or index differs")
        if family == "exact" and abs(report["metrics"].get("Precision@10", -1) - hits / 71990) > 1e-12:
            raise ValueError("Test Precision@10 differs from verified integer hits")
    item = {"key": identity["key"], "repository": "wbm-region-retrieval", "path": _relative(folder, root),
            "category": "fusion_ablation", "method": "F", "variant": "RegionalJScore/" + NAME,
            "split": "test", "manifest": _relative(folder / "evaluation_manifest.json", root),
            "marker": _relative(marker, root), "identity_schema_version": 1,
            "identity": _relative(folder / "run_identity.json", root),
            "selection": _relative(output / "frozen_selection.json", root), "coverage": None,
            "supervision": "self_supervised", "display": "区域J等权加法（valid冻结胜者）",
            "required_families": ["exact", "coverage"], "protocol": "retrieval_v1", "posthoc": True,
            "supporting_manifests": [_relative(output / p, root) for p in
                ("run_manifest.json", "completed.json", "inputs.json", "source_protection_after.json",
                 "formula_audit_test.json", "evaluation_index_test.json", f"test/{NAME}/verification.json")],
            "design_note": "One complete test evaluation of the previously frozen valid-only 15-candidate winner mixed_a0p5_e1p0: region score B*(0.5*C+0.5*exp(-0.5*D_J)), without the zero-CNN gate. Frozen cc8 Top-4 h512 best_nce seed2026, original all-point joint descriptors, area factor, greedy matching and query sqrt(area) aggregation. Original valid selection/completion/index SHA256 are bound before test. No training, test parameter selection or extra test variants. Research follows historical test observations; this does not replace the project main method."}
    catalog.execution_identity(root, item, marker, evaluation)
    return item


def proposal(output, *, root=ROOT):
    output, root = Path(output).resolve(), Path(root).resolve()
    _relative(output, root / "artificial")
    if output == root / "artificial":
        raise ValueError("Expected an isolated test experiment output")
    manifest = read_signed(output / "run_manifest.json")
    context = manifest.get("context", {})
    context_signature = digest(context)
    required = ("run_manifest.json", "status.json", "frozen_selection.json", "evaluation_index_test.json",
                "inputs.json", "formula_audit_test.json", "source_protection_before.json", "source_protection_after.json")
    _complete(output, context_signature, required)
    status = read_json(output / "status.json")
    if (manifest.get("version") != "f_region_score_test_v1" or context.get("version") != "f_region_score_test_v1"
            or context.get("split") != "test" or context.get("frozen_variant") != NAME
            or context.get("frozen_spec") != SPEC or context.get("selection_policy") != POLICY
            or context.get("training") is not False or context.get("test_parameter_search") is not False
            or context.get("posthoc_design") is not True or status.get("status") != "completed"
            or status.get("split") != "test" or status.get("test_evaluated") is not True
            or status.get("winner") != NAME or status.get("context_signature") != context_signature):
        raise ValueError("Registration requires one complete frozen-valid-winner test without training")
    if any((output / name).exists() for name in ("valid", "training", "selection.json", "evaluation_index_valid.json")):
        raise ValueError("Unexpected selection, valid search or training artifacts in a test-only run")
    source_binding = context.get("source_binding")
    inputs = read_json(output / "inputs.json")
    if (not isinstance(source_binding, dict) or not source_binding or inputs.get("source_binding") != source_binding
            or inputs.get("context_signature") != context_signature
            or inputs.get("source_run") != context.get("source_run")
            or inputs.get("cache_source_run") != context.get("cache_source_run")):
        raise ValueError("Frozen input source binding differs from the test context")
    for name, expected in source_binding.items():
        path = Path(name)
        if not path.is_absolute() or path.is_symlink() or not path.is_file() or file_hash(path) != expected:
            raise ValueError("Frozen test input source is missing or changed: " + name)
    _frozen_selection(output, root, context, source_binding)
    before = read_json(output / "source_protection_before.json")
    after = read_json(output / "source_protection_after.json")
    if after.get("status") != "unchanged" or after.get("files") != len(before):
        raise ValueError("Historical source protection evidence failed")
    index = read_json(output / "evaluation_index_test.json")
    if (index.get("split") != "test" or index.get("specs") != {NAME: SPEC}
            or set(index.get("results", {})) != {NAME} or index.get("context_signature") != context_signature):
        raise ValueError("Test index must contain only the frozen valid winner")
    if {p.name for p in (output / "test").iterdir() if p.is_dir()} != {NAME}:
        raise ValueError("Unexpected extra or missing test evaluation directories")
    audit = read_json(output / "formula_audit_test.json")
    representatives = audit.get("query_ids", [])
    if (audit.get("status") != "verified" or audit.get("split") != "test" or audit.get("variant") != NAME
            or audit.get("context_signature") != context_signature or len(representatives) != 37
            or len(set(representatives)) != 37 or audit.get("candidate_pairs") != 740):
        raise ValueError("Independent test regional formula/greedy audit is incomplete")
    _finite_error(audit.get("max_absolute_error"), 2e-6, "regional_formula")
    return {"schema_version": 1, "runs": [_evaluation(output, root, index["results"][NAME], context_signature)]}


def write_proposal(output, *, root=ROOT):
    value = proposal(output, root=root)
    path = Path(output) / "catalog_sources.json"
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Test registration proposal changed")
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
