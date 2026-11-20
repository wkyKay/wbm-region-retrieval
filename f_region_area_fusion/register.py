#!/usr/bin/env python3
"""Register a sealed 29-way valid search or its ten frozen test evaluations."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
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
VERSION = "f_region_area_fusion_v1"
BASELINE = "mul_a0p5_b0p15"
POLICY = "frozen_valid_top10_only"


def _evaluation(output, root, split, name, spec, entry, context_signature):
    folder = output / split / name
    nquery, ngallery = (500, 3599) if split == "valid" else (7199, 7199)
    signature = digest({"context": context_signature, "split": split, "variant": name, "spec": spec})
    _complete(folder, signature, (*REPORT_FILES, "run_identity.json", "verification.json"))
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
            or evaluation.get("test_evaluated") != (split == "test")
            or evaluation.get("posthoc_research") is not True
            or evaluation.get("training_required") is not False
            or identity.get("evaluation_signature") != signature
            or not identity.get("key", "").startswith("f_region_area_fusion_")
            or any(evaluation.get(k) != v for k, v in identity.items())):
        raise ValueError("Area-fusion evaluation manifest identity or protocol differs")
    checked = read_json(folder / "verification.json")
    files = checked.get("files", {})
    if (checked.get("status") != "verified" or checked.get("signature") != signature
            or checked.get("split") != split or checked.get("variant") != name
            or checked.get("ks") != KS or checked.get("tie_seed") != 2026
            or checked.get("num_queries") != nquery or checked.get("ranking_rows") != nquery * 500
            or checked.get("full_candidates_checked") != nquery * (ngallery - 1)
            or checked.get("all_queries_full_candidate_topk_exact") is not True
            or type(checked.get("aggregation_values_checked")) is not int
            or checked["aggregation_values_checked"] <= 0
            or set(files) != set(REPORT_FILES) | {"run_identity.json"}
            or any(file_hash(folder / filename) != sha for filename, sha in files.items())
            or checked.get("ranking_sha256") != file_hash(folder / "rankings.csv")):
        raise ValueError("Independent full-candidate verification is missing, incomplete or stale")
    _finite_error(checked.get("max_metric_absolute_error"), 1e-12, "metric")
    _finite_error(checked.get("max_saved_score_absolute_error"), 1e-12, "saved_score")
    hits = checked.get("top10_hits")
    if type(hits) is not int or not 0 <= hits <= nquery * 10 or entry.get("top10_hits") != hits:
        raise ValueError("Integer Top-10 hit total differs from verified ranking")
    for family, filename in (("exact", "metrics.json"), ("coverage", "coverage_metrics.json")):
        report = read_json(folder / filename)
        _metric_values(report.get("metrics"), family)
        metadata = report.get("metadata", {})
        index_name = "metrics" if family == "exact" else "coverage_metrics"
        if (report.get("split") != split or report.get("num_queries") != nquery or report.get("ks") != KS
                or report.get("metric_version") != family + "_v1"
                or report.get("test_evaluated") != (split == "test")
                or metadata.get("signature") != signature or metadata.get("smoke") or metadata.get("derived_protocol")
                or report.get("retrieval_protocol_signature", metadata.get("retrieval_protocol_signature")) != catalog.BASE_PROTOCOL_SIGNATURE
                or entry.get(index_name) != report.get("metrics")):
            raise ValueError("Formal metric report identity or index differs")
        if family == "exact" and abs(report["metrics"].get("Precision@10", -1) - hits / (nquery * 10)) > 1e-12:
            raise ValueError("Precision@10 differs from verified integer hits")
    choice = "selection.json" if split == "valid" else "frozen_selection.json"
    evidence = ["run_manifest.json", "completed.json", "inputs.json", "source_protection_after.json",
                f"formula_audit_{split}.json", f"evaluation_index_{split}.json", f"{split}/{name}/verification.json"]
    if (output / "baseline_replay.json").exists():
        evidence.append("baseline_replay.json")
    item = {"key": identity["key"], "repository": "wbm-region-retrieval", "path": _relative(folder, root),
            "category": "fusion_ablation", "method": "F", "variant": "RegionalJArea/" + name,
            "split": split, "manifest": _relative(folder / "evaluation_manifest.json", root),
            "marker": _relative(marker, root), "identity_schema_version": 1,
            "identity": _relative(folder / "run_identity.json", root),
            "selection": _relative(output / choice, root), "coverage": None,
            "supervision": "self_supervised", "display": "区域J与面积融合 " + name,
            "required_families": ["exact", "coverage"], "protocol": "retrieval_v1", "posthoc": True,
            "supporting_manifests": [_relative(output / p, root) for p in evidence],
            "design_note": "Prespecified 29-candidate regional CNN/J/area score study; 29 valid evaluations select exactly ten by integer exact Top-10 hits, with ties preferring the original equal-weight multiplicative-area baseline and then lexicographic names. Test evaluates only these ten frozen choices, without test parameter selection. Reuses frozen self-supervised cc8 Top-4 h512 best_nce seed2026, all-point regional joint descriptors, original score scales, greedy matching and query sqrt(area) aggregation. Entropy weighting is a regional adaptation over every non-self gallery region, not a reproduction of whole-wafer literature scores. No training; posthoc research does not replace the project main method."}
    catalog.execution_identity(root, item, marker, evaluation)
    return item


def _selected(output, root, context, signature, binding):
    """Read the immutable valid decision, recursively verifying its complete grid."""
    if context["split"] == "valid":
        selected = read_signed(output / "selection.json")
        index = read_json(output / "evaluation_index_valid.json")
        expected = selection.select_top10(index.get("results"), split="valid")
        if (selected.get("context_signature") != signature or selected.get("synthetic") is not False
                or selected.get("valid_index_sha256") != file_hash(output / "evaluation_index_valid.json")
                or any(selected.get(k) != v for k, v in expected.items())):
            raise ValueError("Frozen selection differs from complete-grid integer valid Top-10 selection")
        stamp = selected.get("frozen_at")
        if not isinstance(stamp, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00", stamp):
            raise ValueError("Valid selection must have its actual timezone-aware frozen_at")
        datetime.fromisoformat(stamp)
        return selected
    frozen = read_signed(output / "frozen_selection.json")
    if (frozen.get("schema_version") != 1 or frozen.get("selection_policy") != POLICY
            or frozen.get("source_run") != context.get("source_run")
            or frozen.get("signature") != context.get("frozen_selection_signature")):
        raise ValueError("Frozen valid Top-10 wrapper differs from the test context")
    source = Path(frozen["source_run"]).resolve()
    _relative(source, root / "artificial")
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Test output must be isolated from its valid source")
    expected_files = {"selection.json": "source_selection_sha256", "completed.json": "source_completed_sha256",
                      "evaluation_index_valid.json": "source_valid_index_sha256"}
    for name, field in expected_files.items():
        path = source / name
        if not frozen.get(field) or file_hash(path) != frozen[field] or binding.get(str(path)) != frozen[field]:
            raise ValueError("Frozen valid source selection/completion/index SHA differs")
    manifest = read_signed(source / "run_manifest.json")
    if manifest.get("context", {}).get("split") != "valid":
        raise ValueError("Test selection source must be a valid search")
    original = read_signed(source / "selection.json")
    if (frozen.get("selection") != original
            or frozen.get("selected_variants") != original.get("selected_variants")
            or frozen.get("selected_specs") != original.get("selected_specs")):
        raise ValueError("Copied selection is not the original valid Top-10 decision")
    proposed = proposal(source, root=root)
    if len(proposed["runs"]) != 29 or any(row["split"] != "valid" for row in proposed["runs"]):
        raise ValueError("Frozen source must contain all 29 valid evaluations")
    for name, field in expected_files.items():
        if file_hash(source / name) != frozen[field]:
            raise ValueError("Frozen valid source changed during registration validation")
    return original


def _baseline(output, root, split, specs):
    path = output / "baseline_replay.json"
    if BASELINE not in specs:
        if path.exists():
            raise ValueError("Unexpected unselected baseline replay")
        return
    value = read_json(path)
    nquery = 500 if split == "valid" else 7199
    if (value.get("status") != "verified" or value.get("split") != split or value.get("variant") != BASELINE
            or value.get("rows") != nquery * 500 or value.get("max_score_error") != 0
            or value.get("ranking_byte_identical") is not True or not isinstance(value.get("source"), str)
            or not value["source"]):
        raise ValueError("Equal-weight baseline replay is incomplete or inexact")
    original = Path(value["source"])
    original = original if original.is_absolute() else root / original
    _relative(original, root)
    if (not original.is_file() or original.resolve().is_relative_to(output)
            or file_hash(original) != value.get("source_sha256")
            or file_hash(output / split / BASELINE / "rankings.csv") != value.get("source_sha256")):
        raise ValueError("Original baseline ranking changed or is missing")


def proposal(output, *, root=ROOT):
    output, root = Path(output).resolve(), Path(root).resolve()
    _relative(output, root / "artificial")
    if output == root / "artificial":
        raise ValueError("Expected an isolated experiment output")
    manifest = read_signed(output / "run_manifest.json")
    context = manifest.get("context", {})
    signature = digest(context)
    split = context.get("split")
    grid = scoring.frozen_grid()
    if (manifest.get("version") != VERSION or context.get("version") != VERSION or split not in ("valid", "test")
            or context.get("full_grid") != grid or context.get("selection_rule") != selection.RULE
            or context.get("training") is not False or context.get("posthoc_design") is not True
            or context.get("test_parameter_search") is not False):
        raise ValueError("Registration requires a fixed 29-candidate area-fusion run without training")
    choice = "selection.json" if split == "valid" else "frozen_selection.json"
    required = ["run_manifest.json", "status.json", choice, f"evaluation_index_{split}.json", "inputs.json",
                f"formula_audit_{split}.json", "source_protection_before.json", "source_protection_after.json"]
    if BASELINE in context.get("grid", {}):
        required.append("baseline_replay.json")
    _complete(output, signature, required)
    status = read_json(output / "status.json")
    if (status.get("status") != "completed" or status.get("split") != split
            or status.get("test_evaluated") != (split == "test")
            or status.get("context_signature") != signature):
        raise ValueError("Registration requires a completely sealed formal evaluation")
    opposite = "test" if split == "valid" else "valid"
    if any((output / name).exists() for name in (opposite, "training", f"evaluation_index_{opposite}.json")):
        raise ValueError("Unexpected training or opposite-split artifacts")
    binding = context.get("source_binding")
    inputs = read_json(output / "inputs.json")
    if (not isinstance(binding, dict) or not binding or inputs.get("source_binding") != binding
            or inputs.get("context_signature") != signature
            or inputs.get("source_run") != context.get("source_run")
            or inputs.get("cache_source_run") != context.get("cache_source_run")):
        raise ValueError("Frozen input source binding differs from the scientific context")
    for name, expected in binding.items():
        path = Path(name)
        if not path.is_absolute() or path.is_symlink() or not path.is_file() or file_hash(path) != expected:
            raise ValueError("Frozen input source is missing or changed: " + name)
    before = read_json(output / "source_protection_before.json")
    after = read_json(output / "source_protection_after.json")
    if after.get("status") != "unchanged" or after.get("files") != len(before):
        raise ValueError("Historical source protection evidence failed")
    selected = _selected(output, root, context, signature, binding)
    specs = grid if split == "valid" else selected["selected_specs"]
    index = read_json(output / f"evaluation_index_{split}.json")
    if (context.get("grid") != specs or index.get("split") != split or index.get("specs") != specs
            or index.get("context_signature") != signature or set(index.get("results", {})) != set(specs)):
        raise ValueError("Evaluation index must contain exactly the frozen valid grid or selected test Top-10")
    if (status.get("winner") != selected.get("winner")
            or status.get("selected_variants") != selected.get("selected_variants")):
        raise ValueError("Run status winner differs from the frozen valid decision")
    if {p.name for p in (output / split).iterdir() if p.is_dir()} != set(specs):
        raise ValueError("Unexpected extra or missing evaluation directories")
    _baseline(output, root, split, specs)
    if split == "valid" and index["results"][BASELINE].get("top10_hits") != 2585:
        raise ValueError("Baseline must replay the old 2585/5000 valid Top-10 hits")
    audit = read_json(output / f"formula_audit_{split}.json")
    representatives = audit.get("query_ids", [])
    if (audit.get("status") != "verified" or audit.get("split") != split
            or audit.get("context_signature") != signature or audit.get("specs") != specs or len(representatives) != 37
            or len(set(representatives)) != 37 or audit.get("candidate_pairs") != 37 * 20 * len(specs)):
        raise ValueError("Independent regional formula/greedy audit is incomplete")
    _finite_error(audit.get("max_absolute_error"), 2e-6, "regional_formula")
    rows = [_evaluation(output, root, split, name, spec, index["results"][name], signature)
            for name, spec in specs.items()]
    expected_count = 29 if split == "valid" else 10
    if len(rows) != expected_count or len({row["key"] for row in rows}) != expected_count:
        raise ValueError("Expected 29 valid or ten selected test evaluations with unique identities")
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
