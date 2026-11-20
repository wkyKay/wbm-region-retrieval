"""Immutable provenance for one frozen regional-J/area cross-combination test."""
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
VERSION = "f_region_j_cross_test_v1"
VARIANT = "add_a0p4_b0p25"
SPEC = {"family": "add", "alpha": .4, "beta": .25}
SETTINGS = {"ks": [1, 5, 10, 20, 50, 100, 500], "tie_seed": 2026}
PROTOCOL = "coverage_cross_combination_v1"
CANDIDATE_RULE = "all_gallery_except_query_and_exact_label_combination"
CANDIDATE_IDENTITY = "68c6875330ed70841586c654da776f8d0bc72bcd4cd341b79cea29705543f0b8"
SELECTION_RULE = "user_requested_cross_evaluation_of_frozen_valid_winner"
REFERENCE_VARIANTS = ["GJ_ADD", "FJ_ADD", "F_ONLY", "G_ONLY", "J_ONLY", "WBBS"]
REPORT_FILES = ("run_identity.json", "evaluation_manifest.json", "rankings.csv",
                "coverage_metrics.json", "coverage_per_query.csv", "coverage_per_class.csv")
FACTS = dict(f_version="regional_J_area_greedy", matching="original_F_greedy_sqrt_area",
    structure_strength=.5, origin_valid_winner=True, valid_selection_rank=1,
    training_required=False, posthoc_research=True, test_parameter_search=False,
    normalization_scope="none", primary_variant=None, selection_rule=SELECTION_RULE)


def fixed_variants():
    return signed(dict(schema_version=1, version=VERSION, variants=[VARIANT],
        specs={VARIANT: deepcopy(SPEC)}, settings=deepcopy(SETTINGS),
        selection_rule=SELECTION_RULE, primary_variant=None, test_parameter_search=False,
        origin_valid_winner=True, valid_selection_rank=1))


def variant_signature(context_signature):
    if not isinstance(context_signature, str) or not context_signature:
        raise ValueError("Expected nonempty scientific context signature")
    return digest(dict(context_signature=context_signature, variant=VARIANT, spec=SPEC))


def cache_signature(context_signature):
    return digest(dict(context_signature=context_signature, cache_version=1, variant=VARIANT))


def evaluation_folder(output):
    return Path(output) / "test" / VARIANT


def identity(output, signature):
    output = Path(output)
    key = "f_region_j_cross_" + re.sub(r"[^A-Za-z0-9_.-]", "_", output.name) + "_test_" + VARIANT
    science = dict(schema_version=1, key=key, split="test", variant=VARIANT,
                   evaluation_signature=signature)
    path = output / "identities" / (VARIANT + ".json")
    if path.exists():
        value = read_json(path)
        started = _time(value.get("started_at"))
        expected = dict(**science, started_at=started.isoformat(),
                        run_id=started.strftime("%Y%m%dT%H%M%S%z") + "__" + key)
        _require(value == expected, "Persisted evaluation identity changed")
        return value
    started = now()
    value = dict(**science, started_at=started.isoformat(),
                 run_id=started.strftime("%Y%m%dT%H%M%S%z") + "__" + key)
    record(path, value)
    return value


def expected_metric_counts():
    return {f"{metric}@{k}": count for k in SETTINGS["ks"] for metric, count in
            (("Coverage-nDCG", 7000), ("MeanCoverage", 7199), ("FullCoverageHit", 7199),
             ("FullCoveragePrecision", 7199), ("FullCoverageRecall", 6200), ("FullCoverage_mAP", 6200))}


def derived_signature(metadata):
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _time(value):
    _require(isinstance(value, str) and re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00", value), "Missing actual +08:00 timestamp")
    return datetime.fromisoformat(value)


def _error(value, limit):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= limit


def _current_code():
    from .run import code_identity
    return code_identity()


def _check_cache(folder, context_signature):
    from .cache import ScoreCache
    return ScoreCache(folder, context_signature)


def _bound(path, binding, *, signed_json=False):
    """Recheck small source evidence even when the caller already hashed all sources."""
    path = Path(path).resolve()
    _require(str(path) in binding and path.is_file() and file_hash(path) == binding[str(path)],
             "Source evidence is not bound or changed: " + str(path))
    return read_signed(path) if signed_json else read_json(path)


def _source_chain(context, binding):
    from f_region_area_fusion.selection import select_top10

    meta = context["source_metadata"]
    for key in ("source_valid", "source_test", "source_winner_folder", "reference_root"):
        _require(isinstance(meta.get(key), str) and Path(meta[key]).is_absolute(), "Missing source path: " + key)
    valid, test = Path(meta["source_valid"]), Path(meta["source_test"])
    vm = _bound(valid / "run_manifest.json", binding, signed_json=True)
    tm = _bound(test / "run_manifest.json", binding, signed_json=True)
    vs, ts = digest(vm["context"]), digest(tm["context"])
    vd, td = (_bound(p / "completed.json", binding) for p in (valid, test))
    chosen = _bound(valid / "selection.json", binding, signed_json=True)
    index = _bound(valid / "evaluation_index_valid.json", binding)
    expected = select_top10(index["results"], split="valid")
    _require(vm.get("version") == "f_region_area_fusion_v1"
             and tm.get("version") == "f_region_area_fusion_v1"
             and vm["context"].get("split") == "valid" and tm["context"].get("split") == "test"
             and vd.get("status") == td.get("status") == "completed"
             and vd.get("signature") == vs and td.get("signature") == ts
             and chosen.get("context_signature") == vs and index.get("context_signature") == vs
             and chosen.get("valid_index_sha256") == binding[str(valid / "evaluation_index_valid.json")]
             and all(chosen.get(k) == value for k, value in expected.items())
             and chosen.get("winner") == VARIANT and chosen["selected_specs"][VARIANT] == SPEC,
             "Original complete valid-grid winner differs")
    _time(chosen.get("frozen_at"))
    frozen = _bound(test / "frozen_selection.json", binding, signed_json=True)
    _require(frozen.get("selection") == chosen and frozen.get("source_run") == str(valid)
             and tm["context"].get("source_run") == str(valid)
             and tm["context"].get("frozen_selection_signature") == frozen["signature"]
             and frozen.get("selection_policy") == "frozen_valid_top10_only"
             and frozen.get("selected_variants") == chosen["selected_variants"]
             and frozen.get("selected_specs") == chosen["selected_specs"],
             "Original test does not preserve the frozen valid selection")
    for name, field in (("selection.json", "source_selection_sha256"),
                        ("completed.json", "source_completed_sha256"),
                        ("evaluation_index_valid.json", "source_valid_index_sha256")):
        _require(frozen.get(field) == binding[str(valid / name)], "Source selection SHA changed")
    folder = test / "test" / VARIANT
    _require(meta["source_winner_folder"] == str(folder), "Source winner folder changed")
    ev = _bound(folder / "evaluation_manifest.json", binding)
    done = _bound(folder / "completed.json", binding)
    sig = digest(dict(context=ts, split="test", variant=VARIANT, spec=SPEC))
    _require(done.get("status") == "completed" and done.get("signature") == sig
             and ev.get("signature") == sig and ev.get("variant") == VARIANT
             and ev.get("spec") == SPEC and ev.get("split") == "test"
             and ev.get("matching") == FACTS["matching"] and ev.get("valid_selection_rank") == 1
             and ev.get("num_queries") == ev.get("gallery_count") == 7199
             and ev.get("training_required") is False and ev.get("test_parameter_search") is False
             and ev.get("retrieval_protocol_signature") == BASE_PROTOCOL_SIGNATURE
             and meta.get("source_selection_signature") == chosen["signature"]
             and meta.get("source_valid_context_signature") == vs
             and meta.get("source_test_context_signature") == ts
             and meta.get("source_winner_signature") == sig
             and meta.get("origin_valid_winner") is True and meta.get("valid_selection_rank") == 1,
             "Original regional-J/area test identity or selection facts changed")
    _require(td.get("files", {}).get(f"test/{VARIANT}/completed.json") == binding.get(str(folder / "completed.json")),
             "Original test root does not bind the winner completion")
    for parent, filename in ((vd, "run_manifest.json"), (vd, "selection.json"),
                             (vd, "evaluation_index_valid.json"), (td, "run_manifest.json"),
                             (td, "frozen_selection.json"), (done, "evaluation_manifest.json"),
                             (done, "rankings.csv")):
        base = valid if parent is vd else test if parent is td else folder
        _require(parent.get("files", {}).get(filename) == binding.get(str(base / filename)),
                 "Source completion does not bind required evidence: " + filename)
    return frozen


def _references(output, context, binding, signature):
    meta = context["source_metadata"]
    folders = meta.get("reference_folders", {})
    _require(set(folders) == set(REFERENCE_VARIANTS), "Expected exactly six existing comparison references")
    root = Path(meta["reference_root"])
    manifest = _bound(root / "run_manifest.json", binding, signed_json=True)
    done = _bound(root / "completed.json", binding)
    _require(manifest.get("version") == "fgj_cross_test_v1"
             and manifest["context"].get("derived_protocol") == context["derived_protocol"]
             and done.get("status") == "completed" and done.get("signature") == digest(manifest["context"]),
             "Original six-method comparison root differs")
    recorded = read_json(Path(output) / "comparison_references.json")
    _require(recorded.get("status") == "verified" and recorded.get("context_signature") == signature
             and set(recorded.get("references", {})) == set(REFERENCE_VARIANTS),
             "Missing read-only comparison references")
    for name in REFERENCE_VARIANTS:
        folder = root / "test" / name
        _require(folders[name] == str(folder), "Historical comparison folder changed")
        marker = _bound(folder / "completed.json", binding)
        report = _bound(folder / "coverage_metrics.json", binding)
        evidence = recorded["references"][name]
        _require(evidence.get("folder") == str(folder)
                 and evidence.get("completed_sha256") == binding[str(folder / "completed.json")]
                 and evidence.get("coverage_metrics_sha256") == binding[str(folder / "coverage_metrics.json")]
                 and marker.get("status") == "completed"
                 and marker.get("files", {}).get("coverage_metrics.json") == binding[str(folder / "coverage_metrics.json")]
                 and done.get("files", {}).get(f"test/{name}/completed.json") == binding[str(folder / "completed.json")]
                 and report.get("split") == "test" and report.get("metric_version") == "coverage_v1"
                 and report.get("num_queries") == 7199 and report.get("ks") == SETTINGS["ks"]
                 and report.get("metadata", {}).get("derived_protocol") == context["derived_protocol"]
                 and report.get("metric_query_counts") == expected_metric_counts(),
                 "Historical comparison evidence changed: " + name)


def verify_run(output, *, check_source=True):
    """Accept only the sealed one-result evaluation and unchanged historical references."""
    output = Path(output).resolve()
    manifest = read_signed(output / "run_manifest.json")
    context = manifest["context"]
    signature = digest(context)
    frozen = read_signed(output / "frozen_variants.json")
    started = _time(manifest.get("started_at"))
    _require(manifest.get("run_id") == started.strftime("%Y%m%dT%H%M%S%z") + "__f_region_j_cross_test"
             and manifest.get("version") == VERSION and manifest.get("stage") == "test"
             and manifest.get("split") == context.get("split") == "test"
             and context.get("code") == _current_code() and context.get("settings") == SETTINGS
             and context.get("variant") == VARIANT and context.get("spec") == SPEC
             and all(context.get(k, "missing") == v for k, v in FACTS.items())
             and context.get("formal_protocol") is True and frozen == fixed_variants()
             and context.get("frozen_variants_signature") == frozen["signature"], "Frozen single-variant plan differs")
    _require(verified(output, signature), "New cross run is not sealed")
    status = read_json(output / "status.json")
    _require(status.get("status") == "completed" and status.get("context_signature") == signature
             and status.get("test_evaluated") is True and status.get("new_evaluations") == 1
             and _time(status.get("finished_at")) >= started, "Run completion or actual time differs")
    required = {"run_manifest.json", "frozen_variants.json", "frozen_selection.json", "settings.json", "inputs.json", "status.json",
        "candidate_protocol.json", "protocol/cross_combination_candidates.csv", "evaluation_index_test.json",
        "formula_audit_test.json", "historical_replay.json", "comparison_references.json",
        "source_protection_before.json", "source_protection_after.json", "score_cache/completed.json"}
    _require(required <= set(read_json(output / "completed.json")["files"]), "Root seal omits required evidence")
    derived = context["derived_protocol"]
    _require(read_json(output / "settings.json") == SETTINGS
             and read_json(output / "candidate_protocol.json") == derived
             and context.get("protocol") == PROTOCOL and context.get("candidate_rule") == CANDIDATE_RULE
             and derived.get("protocol_version") == PROTOCOL and derived.get("candidate_rule") == CANDIDATE_RULE
             and derived.get("base_protocol_signature") == BASE_PROTOCOL_SIGNATURE
             and derived.get("candidate_identity_sha256") == CANDIDATE_IDENTITY
             and derived.get("candidate_count_min") == 6999 and derived.get("candidate_count_max") == 7170
             and derived.get("filter_uses_ground_truth") is True
             and context.get("retrieval_protocol_signature") == derived_signature(derived), "Cross candidate protocol changed")
    binding = context.get("source_binding")
    _require(isinstance(binding, dict) and bool(binding) and all(isinstance(p, str) and Path(p).is_absolute()
             and isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) for p, h in binding.items()), "Invalid source binding")
    inp = read_json(output / "inputs.json")
    _require(inp.get("source_binding") == binding and inp.get("source_metadata") == context.get("source_metadata"),
             "Input source identity changed")
    if check_source:
        for path, h in binding.items():
            _require(Path(path).is_file() and file_hash(path) == h, "Frozen source changed: " + path)
    original_frozen = _source_chain(context, binding)
    _require(read_signed(output / "frozen_selection.json") == original_frozen
             and _time(original_frozen["selection"]["frozen_at"]) <= started,
             "Original frozen valid selection copy or time changed")
    _references(output, context, binding, signature)
    before, after = (read_json(output / ("source_protection_" + phase + ".json")) for phase in ("before", "after"))
    _require(after.get("status") == "unchanged" and after.get("files") == len(before), "Historical protection failed")
    cache = output / "score_cache"
    _check_cache(cache, signature)
    cache_done, cache_meta = read_json(cache / "completed.json"), read_json(cache / "metadata.json")
    _require(cache_done.get("signature") == cache_signature(signature)
             and set(cache_done.get("files", {})) == {"ids.npy", "scores.npy", "metadata.json"}
             and all(cache_meta.get(k) == v for k, v in dict(context_signature=signature, variant=VARIANT,
                 spec=SPEC, base_protocol_signature=BASE_PROTOCOL_SIGNATURE, query_count=7199,
                 gallery_count=7199, dtype="float64", self_score="NaN", full_pairs=51818402).items()),
             "Complete original-pool score cache differs")
    provider_source = dict(signature=cache_signature(signature), completed_sha256=file_hash(cache / "completed.json"))
    audit = read_json(output / "formula_audit_test.json")
    _require(audit.get("status") == "verified" and audit.get("split") == "test"
             and audit.get("context_signature") == signature and audit.get("specs") == {VARIANT: SPEC}
             and len(audit.get("query_ids", [])) == 37 and len(set(audit["query_ids"])) == 37
             and audit.get("candidate_pairs") == 740 and _error(audit.get("max_absolute_error"), 2e-6),
             "Independent regional formula/greedy audit is incomplete")
    replay = read_json(output / "historical_replay.json")
    source = str(Path(context["source_metadata"]["source_winner_folder"]) / "rankings.csv")
    _require(replay.get("status") == "verified" and replay.get("context_signature") == signature
             and replay.get("num_queries") == 7199 and replay.get("ranking_rows") == 3599500
             and replay.get("top500_identical") is True and _error(replay.get("max_score_error"), 0.)
             and replay.get("source") == source and replay.get("source_sha256") == binding.get(source)
             and replay.get("score_cache_completed_sha256") == provider_source["completed_sha256"],
             "Complete original-pool scores did not reproduce the historical test")
    _require({p.name for p in (output / "test").iterdir() if not p.name.startswith(".")} == {VARIANT},
             "Exactly one new evaluation is permitted")
    folder, sig = evaluation_folder(output), variant_signature(signature)
    _require(verified(folder, sig), "Child evaluation is not sealed")
    _require(set(read_json(folder / "completed.json")["files"]) == set(REPORT_FILES) | {"verification.json", "timing.json"},
             "Child completion file set differs")
    _require((output / "identities" / (VARIANT + ".json")).is_file(), "Persistent evaluation identity is missing")
    ident = identity(output, sig)
    ev = read_json(folder / "evaluation_manifest.json")
    _require(read_json(folder / "run_identity.json") == ident and all(ev.get(k) == v for k, v in ident.items())
             and ev.get("signature") == sig and ev.get("context_signature") == signature
             and ev.get("settings") == SETTINGS and ev.get("spec") == SPEC and ev.get("stage") == "test"
             and ev.get("protocol") == PROTOCOL and ev.get("candidate_rule") == CANDIDATE_RULE
             and ev.get("derived_protocol") == derived and ev.get("retrieval_protocol_signature") == derived_signature(derived)
             and ev.get("num_queries") == ev.get("gallery_count") == 7199 and ev.get("ks") == SETTINGS["ks"]
             and ev.get("formal_protocol") is True
             and ev.get("tie_seed") == 2026 and ev.get("provider_source") == provider_source
             and ev.get("source_binding_sha256") == digest(binding)
             and all(ev.get(k, "missing") == v for k, v in FACTS.items()), "Child scientific metadata differs")
    _require(not any((folder / name).exists() for name in ("metrics.json", "per_query.csv", "per_class.csv")),
             "Cross evaluation must remain coverage-only")
    receipt = read_json(folder / "verification.json")
    expected = dict(status="verified", signature=sig, context_signature=signature, split="test", variant=VARIANT,
        spec=SPEC, ks=SETTINGS["ks"], tie_seed=2026, provider_source=provider_source,
        source_binding_sha256=digest(binding), source_file_count=len(binding), num_queries=7199,
        ranking_rows=3599500, full_candidates_checked=50395860, candidate_identity_sha256=CANDIDATE_IDENTITY,
        candidate_count_min=6999, candidate_count_max=7170, all_queries_full_candidate_topk_exact=True,
        exact_reports=False, metric_query_counts=expected_metric_counts(), ranking_sha256=file_hash(folder / "rankings.csv"))
    _require(all(receipt.get(k) == v for k, v in expected.items()) and set(receipt.get("files", {})) == set(REPORT_FILES)
             and _error(receipt.get("max_metric_absolute_error"), 1e-12)
             and _error(receipt.get("max_saved_score_absolute_error"), 1e-12)
             and isinstance(receipt.get("scope"), str) and bool(receipt["scope"]), "Independent cached-score ranking/metric receipt differs")
    for filename, h in receipt["files"].items():
        _require(file_hash(folder / filename) == h, "Verified report changed: " + filename)
    report = read_json(folder / "coverage_metrics.json")
    _require(report.get("split") == "test" and report.get("num_queries") == 7199
             and report.get("ks") == SETTINGS["ks"] and report.get("metric_version") == "coverage_v1"
             and all(report.get("metadata", {}).get(k) == v for k, v in ev.items())
             and not report.get("metadata", {}).get("smoke") and set(report.get("metrics", {})) == set(expected_metric_counts())
             and report.get("metric_query_counts") == expected_metric_counts()
             and report.get("queries_without_full_coverage_candidates") == 999, "Coverage contract changed")
    index = read_json(output / "evaluation_index_test.json")
    _require(all(index.get(k) == v for k, v in dict(context_signature=signature, split="test", variant=VARIANT,
        spec=SPEC, coverage_metrics=report["metrics"], metric_query_counts=expected_metric_counts(),
        completed_sha256=file_hash(folder / "completed.json")).items()), "Indexed evidence differs")
    return manifest, frozen
