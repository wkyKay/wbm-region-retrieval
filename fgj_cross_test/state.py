"""Frozen user-requested cross-combination recipes and completed-run evidence."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

from f_region_structure.state import (TZ, now, record, signed, read_signed,
                                      protection, seal, verified)
from results_catalog.build import BASE_PROTOCOL_SIGNATURE
from wbm_retrieval.common import digest, file_hash, read_json


ROOT = Path(__file__).resolve().parents[1]
VERSION = "fgj_cross_test_v1"
PROTOCOL = "coverage_cross_combination_v1"
CANDIDATE_RULE = "all_gallery_except_query_and_exact_label_combination"
CANDIDATE_IDENTITY = "68c6875330ed70841586c654da776f8d0bc72bcd4cd341b79cea29705543f0b8"
VARIANTS = ["GJ_ADD", "FJ_ADD", "F_ONLY", "G_ONLY", "J_ONLY", "WBBS"]
SPECS = dict(zip(VARIANTS, ({"formula": f} for f in
             ("0.5*G+0.5*J", "0.5*F+0.5*J", "F", "G", "J", "WBBS"))))
SETTINGS = {"ks": [1, 5, 10, 20, 50, 100, 500], "tie_seed": 2026}
SELECTION_RULE = "user_specified_posthoc_components"
FULL_PAIRS = 50395860
QUERY_COUNT = GALLERY_COUNT = 7199


def fixed_variants():
    return signed(dict(schema_version=1, version=VERSION, variants=list(VARIANTS),
        specs=deepcopy(SPECS), settings=deepcopy(SETTINGS), selection_rule=SELECTION_RULE,
        primary_variant=None, test_parameter_search=False,
        reporting_order="Coverage-nDCG@10 descending, with metric-specific descriptive ranks; no method selection"))


def variant_signature(context_signature, name):
    if name not in SPECS or not isinstance(context_signature, str) or not context_signature:
        raise ValueError("Expected a frozen variant and nonempty context signature")
    return digest({"context_signature": context_signature, "variant": name, "spec": SPECS[name]})


def evaluation_folder(output, name):
    if name not in VARIANTS:
        raise ValueError("Unknown fixed cross-combination variant")
    return Path(output) / "test" / name


def evaluation_paths(output, names=None):
    names = VARIANTS if names is None else list(names)
    if len(names) != len(set(names)) or any(name not in VARIANTS for name in names):
        raise ValueError("Invalid or repeated fixed evaluation variant")
    return {name: str(evaluation_folder(output, name).relative_to(output)) for name in names}


def identity(output, name, signature):
    if name not in VARIANTS:
        raise ValueError("Unknown fixed cross-combination variant")
    output = Path(output)
    key = "fgj_cross_" + re.sub(r"[^A-Za-z0-9_.-]", "_", output.name) + "_test_" + name
    science = dict(schema_version=1, key=key, split="test", variant=name,
                   evaluation_signature=signature)
    path = output / "identities" / (name + ".json")
    if path.exists():
        value = read_json(path)
        stamp = value.get("started_at", "")
        if not isinstance(stamp, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00", stamp):
            raise ValueError("Invalid persisted +08:00 evaluation start time")
        started = datetime.fromisoformat(stamp)
        expected = dict(**science, started_at=stamp,
                        run_id=started.strftime("%Y%m%dT%H%M%S%z") + "__" + key)
        if value != expected:
            raise ValueError("Persisted cross-combination evaluation identity changed")
        return value
    started = now()
    value = dict(**science, started_at=started.isoformat(),
                 run_id=started.strftime("%Y%m%dT%H%M%S%z") + "__" + key)
    record(path, value)
    return value


def derived_signature(metadata):
    """Preserve the original cross-protocol serialization, including spaces."""
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()


def expected_metric_counts():
    return {f"{metric}@{k}": count for k in SETTINGS["ks"] for metric, count in
            (("Coverage-nDCG", 7000), ("MeanCoverage", 7199), ("FullCoverageHit", 7199),
             ("FullCoveragePrecision", 7199), ("FullCoverageRecall", 6200), ("FullCoverage_mAP", 6200))}


def _current_code():
    from .run import code_identity
    return code_identity()


def _require(value, message):
    if not value:
        raise ValueError(message)


def _finite_error(value, tolerance):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= tolerance


def verify_run(output, *, check_source=True):
    """Read-only verification of the six fixed coverage-only test evaluations."""
    from .verify import REPORT_FILES

    output = Path(output).resolve()
    manifest = read_signed(output / "run_manifest.json")
    context = manifest["context"]
    signature = digest(context)
    frozen = read_signed(output / "frozen_variants.json")
    started_at = manifest.get("started_at", "")
    _require(isinstance(started_at, str) and re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00", started_at),
        "Root run has no real persisted +08:00 start time")
    started = datetime.fromisoformat(started_at)
    _require(manifest.get("run_id") == started.strftime("%Y%m%dT%H%M%S%z") + "__fgj_cross_test",
             "Root run ID differs from its persisted start time")
    _require(manifest.get("version") == VERSION and manifest.get("stage") == "test"
             and manifest.get("split") == "test" and context.get("split") == "test"
             and context.get("code") == _current_code()
             and context.get("settings") == SETTINGS and context.get("specs") == SPECS
             and context.get("variants") == VARIANTS and frozen == fixed_variants()
             and context.get("frozen_variants_signature") == frozen["signature"]
             and context.get("protocol") == PROTOCOL and context.get("candidate_rule") == CANDIDATE_RULE
             and context.get("normalization_scope") == "none"
             and context.get("primary_variant", "missing") is None
             and context.get("selection_rule") == SELECTION_RULE
             and context.get("test_parameter_search") is False
             and context.get("formal_protocol") is True
             and context.get("f_version") == "cc8_optimal_radial_area"
             and context.get("structure_strength") == 2.0
             and context.get("training_required") is False
             and context.get("posthoc_research") is True,
             "Run differs from the fixed six-formula cross-combination plan")
    _require(verified(output, signature), "Cross-combination root is not fully sealed")
    status = read_json(output / "status.json")
    _require(status.get("status") == "completed" and status.get("context_signature") == signature,
             "Completed run status does not match its scientific context")
    _require(read_json(output / "settings.json") == SETTINGS, "Evaluation settings changed")
    derived = context.get("derived_protocol", {})
    _require(read_json(output / "candidate_protocol.json") == derived
             and derived.get("protocol_version") == PROTOCOL
             and derived.get("candidate_rule") == CANDIDATE_RULE
             and derived.get("base_protocol_signature") == BASE_PROTOCOL_SIGNATURE
             and derived.get("candidate_identity_sha256") == CANDIDATE_IDENTITY
             and derived.get("candidate_count_min") == 6999
             and derived.get("candidate_count_max") == 7170
             and derived.get("filter_uses_ground_truth") is True
             and context.get("retrieval_protocol_signature") == derived_signature(derived),
             "Cross-combination candidate identities or eligibility rule changed")
    root_files = read_json(output / "completed.json")["files"]
    required_root = {"run_manifest.json", "frozen_variants.json", "settings.json", "status.json",
        "candidate_protocol.json", "protocol/cross_combination_candidates.csv",
        "evaluation_index_test.json", "shared_verification.json", "component_audit.json",
        "baseline_replay.json", "source_protection_before.json", "source_protection_after.json"}
    _require(required_root <= set(root_files), "Root seal omits required evaluation evidence")
    binding = context.get("source_binding")
    _require(isinstance(binding, dict) and bool(binding)
             and all(isinstance(p, str) and Path(p).is_absolute() and isinstance(h, str)
                     and re.fullmatch(r"[0-9a-f]{64}", h) for p, h in binding.items()),
             "Frozen source binding is missing or malformed")
    if check_source:
        for path, expected in binding.items():
            _require(Path(path).is_file() and file_hash(path) == expected,
                     "Frozen source changed: " + path)
    before, after = (read_json(output / ("source_protection_" + phase + ".json"))
                     for phase in ("before", "after"))
    _require(after.get("status") == "unchanged" and after.get("files") == len(before),
             "Historical source protection is incomplete")
    shared_file = read_json(output / "shared_verification.json")
    shared = shared_file.get("shared", {})
    _require(shared_file.get("status") == "verified" and shared.get("status") == "verified"
             and shared.get("split") == "test" and shared.get("context_signature") == signature
             and shared.get("ks") == SETTINGS["ks"] and shared.get("tie_seed") == SETTINGS["tie_seed"]
             and shared.get("variants") == VARIANTS and shared.get("num_queries") == QUERY_COUNT
             and shared.get("components_calls") == QUERY_COUNT
             and shared.get("tie_key_computations") == QUERY_COUNT
             and shared.get("full_candidates_checked") == FULL_PAIRS
             and shared.get("candidate_identity_sha256") == CANDIDATE_IDENTITY
             and shared.get("candidate_count_min") == 6999 and shared.get("candidate_count_max") == 7170
             and shared.get("source_binding_sha256") == digest(binding)
             and shared.get("source_file_count") == len(binding)
             and set(shared_file.get("variants", {})) == set(VARIANTS),
             "Shared full-candidate independent verification is incomplete")
    audit = read_json(output / "component_audit.json")
    _require(audit.get("status") == "verified" and audit.get("context_signature") == signature
             and audit.get("split") == "test"
             and audit.get("num_queries") == 37 and audit.get("candidate_pairs") == 740
             and len(audit.get("query_ids", [])) == 37 and len(set(audit["query_ids"])) == 37
             and audit.get("max_candidates_per_query") == 20
             and audit.get("full_global_candidates_checked") == 266326
             and audit.get("full_global_cache_bitwise_identical") is True
             and _finite_error(audit.get("max_global_absolute_error"), 2e-7)
             and _finite_error(audit.get("max_joint_distance_absolute_error"), 1e-12)
             and _finite_error(audit.get("max_full_global_cache_error"), 0.0),
             "Independent F/G/J/W source-component audit is incomplete")
    replay = read_json(output / "baseline_replay.json")
    _require(replay.get("status") == "verified" and replay.get("context_signature") == signature
             and set(replay.get("controls", {})) == {"F_ONLY", "WBBS"},
             "Required historical cross-combination controls were not replayed")
    for name, item in replay["controls"].items():
        source = item.get("source")
        _require(item.get("num_queries") == QUERY_COUNT and item.get("top500_identical") is True
                 and isinstance(source, str) and source in binding
                 and item.get("source_sha256") == binding[source]
                 and item.get("new_sha256") == file_hash(evaluation_folder(output, name) / "rankings.csv")
                 and _finite_error(item.get("max_score_error"), 1e-12),
                 "Historical cross-combination control differs: " + name)
    index = read_json(output / "evaluation_index_test.json")
    _require(index.get("context_signature") == signature and index.get("split") == "test"
             and index.get("variants") == VARIANTS and index.get("specs") == SPECS
             and set(index.get("results", {})) == set(VARIANTS),
             "Evaluation index differs from the complete six-formula test")
    _require({p.name for p in (output / "test").iterdir() if not p.name.startswith(".")} == set(VARIANTS),
             "Unexpected or missing child evaluation")
    counts = expected_metric_counts()
    for name in VARIANTS:
        folder = evaluation_folder(output, name)
        sig = variant_signature(signature, name)
        _require(verified(folder, sig), "Child evaluation is not sealed: " + name)
        marker = folder / "completed.json"
        child_files = read_json(marker)["files"]
        _require(set(REPORT_FILES) | {"verification.json", "timing.json"} == set(child_files),
                 "Child completion file set differs")
        _require((output / "identities" / (name + ".json")).is_file(), "Persistent child identity is missing")
        ident = identity(output, name, sig)
        ev = read_json(folder / "evaluation_manifest.json")
        _require(read_json(folder / "run_identity.json") == ident and all(ev.get(k) == v for k, v in ident.items())
                 and ev.get("context_signature") == signature and ev.get("signature") == sig
                 and ev.get("spec") == SPECS[name] and ev.get("settings") == SETTINGS
                 and ev.get("stage") == "test" and ev.get("split") == "test" and ev.get("variant") == name
                 and ev.get("derived_protocol") == derived
                 and ev.get("protocol") == PROTOCOL and ev.get("candidate_rule") == CANDIDATE_RULE
                 and ev.get("retrieval_protocol_signature") == derived_signature(derived)
                 and ev.get("ks") == SETTINGS["ks"] and ev.get("tie_seed") == SETTINGS["tie_seed"]
                 and ev.get("num_queries") == QUERY_COUNT and ev.get("gallery_count") == GALLERY_COUNT
                 and ev.get("formal_protocol") is True and ev.get("structure_strength") == 2.0
                 and ev.get("f_version") == "cc8_optimal_radial_area"
                 and ev.get("training_required") is False and ev.get("posthoc_research") is True
                 and ev.get("test_parameter_search") is False
                 and ev.get("selection_rule") == SELECTION_RULE and ev.get("primary_variant", "missing") is None
                 and ev.get("normalization_scope") == "none"
                 and ev.get("source_binding_signature") == digest(binding),
                 "Child identity, formula or protocol differs: " + name)
        _require(not any((folder / p).exists() for p in ("metrics.json", "per_query.csv", "per_class.csv")),
                 "Cross-combination results must remain coverage-only")
        checked = read_json(folder / "verification.json")
        _require(checked == shared_file["variants"][name]
                 and checked.get("status") == "verified" and checked.get("signature") == sig
                 and checked.get("context_signature") == signature and checked.get("split") == "test"
                 and checked.get("variant") == name and checked.get("spec") == SPECS[name]
                 and checked.get("ks") == SETTINGS["ks"] and checked.get("tie_seed") == SETTINGS["tie_seed"]
                 and checked.get("num_queries") == QUERY_COUNT and checked.get("ranking_rows") == QUERY_COUNT * 500
                 and checked.get("full_candidates_checked") == FULL_PAIRS
                 and checked.get("candidate_identity_sha256") == CANDIDATE_IDENTITY
                 and checked.get("candidate_count_min") == 6999 and checked.get("candidate_count_max") == 7170
                 and checked.get("all_queries_full_candidate_topk_exact") is True
                 and checked.get("independent_formulas") is True and checked.get("exact_reports") is False
                 and checked.get("source_binding_sha256") == digest(binding)
                 and checked.get("source_file_count") == len(binding)
                 and checked.get("metric_query_counts") == counts
                 and checked.get("ranking_sha256") == file_hash(folder / "rankings.csv")
                 and set(checked.get("files", {})) == set(REPORT_FILES),
                 "Per-variant full-candidate independent verification is incomplete: " + name)
        for key in ("max_metric_absolute_error", "max_saved_score_absolute_error", "max_saved_component_absolute_error"):
            _require(_finite_error(checked.get(key), 1e-12), "Verification tolerance exceeded: " + key)
        for filename, expected in checked["files"].items():
            _require(file_hash(folder / filename) == expected, "Verified report changed: " + filename)
        report = read_json(folder / "coverage_metrics.json")
        metadata = report.get("metadata", {})
        _require(report.get("split") == "test" and report.get("num_queries") == QUERY_COUNT
                 and report.get("ks") == SETTINGS["ks"] and report.get("metric_version") == "coverage_v1"
                 and all(metadata.get(k) == v for k, v in ev.items())
                 and metadata.get("signature") == sig and metadata.get("derived_protocol") == derived
                 and not metadata.get("smoke") and report.get("metric_query_counts") == counts
                 and set(report.get("metrics", {})) == set(counts)
                 and report.get("queries_without_full_coverage_candidates") == 999,
                 "Coverage report or unchanged denominator rules differ")
        row = index["results"][name]
        _require(row.get("signature") == sig and row.get("completed_sha256") == file_hash(marker)
                 and row.get("coverage_metrics") == report["metrics"] and row.get("metric_query_counts") == counts,
                 "Indexed child evidence differs from sealed coverage reports")
    return manifest, frozen
