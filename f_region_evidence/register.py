"""Register a sealed twelve-valid, three-ordinary and three-cross evidence study."""
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
from f_region_structure.register import _complete, _finite_error, _metric_values, _relative
from f_region_structure.state import TZ, read_signed
from results_catalog import build as catalog
from wbm_retrieval.common import digest, file_hash, read_json
from . import proposals, selection, state, verify


def _bound_files(binding):
    if (not isinstance(binding, dict) or not binding or any(
            not isinstance(p, str) or not Path(p).is_absolute()
            or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha)
            for p, sha in binding.items())):
        raise ValueError("Nonempty absolute source-file hash binding required")
    for path, expected in binding.items():
        if not Path(path).is_file() or file_hash(path) != expected:
            raise ValueError("Frozen input source changed: " + path)


def _stamp(value, label):
    if not isinstance(value,str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00",value):
        raise ValueError("Invalid timezone-aware time: " + label)
    return datetime.fromisoformat(value)


def _nested_seal(output, folder, signature, required, root_files):
    """The root hash pass already checked the bytes; bind each nested seal to it."""
    folder = Path(folder)
    marker = read_json(folder / "completed.json")
    relative = _relative(folder, output)
    files = marker.get("files", {})
    if (marker.get("status") != "completed" or marker.get("signature") != signature
            or not set(required) <= set(files)
            or root_files.get(relative + "/completed.json") != file_hash(folder / "completed.json")):
        raise ValueError("Nested completion identity or required evidence differs")
    for name, sha in files.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or root_files.get(relative + "/" + name) != sha:
            raise ValueError("Nested completion is not bound by the root seal")
    return marker


def _audits(output, split, specs, binding):
    formula = read_json(output / f"formula_audit_{split}.json")
    if (formula.get("status") != "verified" or set(formula.get("variants", {})) != set(specs)
            or formula.get("candidate_pairs") != 740 * len(specs)):
        raise ValueError("Sampled regional formula audit is incomplete")
    for detail in formula["variants"].values():
        if len(set(detail.get("query_ids", []))) != 37 or detail.get("candidate_pairs") != 740:
            raise ValueError("Formula audit must cover all 37 classes and 740 candidate pairs")
        _finite_error(detail.get("max_raw_formula_absolute_error"), 2e-6, "raw regional formula")
        _finite_error(detail.get("max_matching_absolute_error"), 1e-12, "rejection/greedy")
    _finite_error(formula.get("max_raw_formula_absolute_error"), 2e-6, "regional formula maximum")
    _finite_error(formula.get("max_matching_absolute_error"), 1e-12, "greedy maximum")
    calibrated = read_json(output / f"calibration_audit_{split}.json")
    reference = read_json(output / "reference.json")
    if (calibrated.get("status") != "verified" or calibrated.get("reference_size") != 4096
            or calibrated.get("reference_identity_sha256") != reference.get("reference_identity_sha256")
            or calibrated.get("train_query_disjoint") is not True
            or set(calibrated.get("proposals", {})) != {"base4", "top8", "residual8"}):
        raise ValueError("Independent train-reference calibration audit is incomplete")
    for detail in calibrated["proposals"].values():
        if (not 1 <= detail.get("regions_checked", 0) <= 37
                or len(set(detail.get("query_ids", []))) != detail["regions_checked"]
                or len(set(detail.get("query_tokens", []))) != detail["regions_checked"]
                or detail.get("region_reference_pairs") != detail["regions_checked"] * 4096):
            raise ValueError("Calibration audit query/reference counts differ")
        _finite_error(detail.get("max_threshold_absolute_error"), 2e-6, "background quantile")
    _finite_error(calibrated.get("max_threshold_absolute_error"), 2e-6, "background quantile maximum")
    preserved = read_json(output / f"preserved_regions_{split}.json")
    if preserved.get("status") != "verified" or set(preserved.get("proposals", {})) != {"base4", "top8", "residual8"}:
        raise ValueError("Original-region cache preservation audit is incomplete")
    for detail in preserved["proposals"].values():
        if (detail.get("all_original_regions_byte_identical") is not True
                or detail.get("num_maps") != (3599 if split == "valid" else 7199)
                or type(detail.get("original_regions")) is not int or detail["original_regions"] <= 0
                or detail.get("new_regions", -1) < detail["original_regions"]
                or not {"features", "area", "relative_area", "joint_all", "owners"} <= set(detail.get("preserved_fields", []))
                or set(detail.get("preserved_sha256", {})) != set(detail.get("preserved_fields", []))):
            raise ValueError("Every original token must have a byte-preservation receipt")
    replay = read_json(output / f"baseline_replay_{split}.json")
    expected_queries = 500 if split == "valid" else 7199
    source = Path(replay.get("source", ""))
    if (replay.get("status") != "verified" or replay.get("split") != split
            or replay.get("num_queries") != expected_queries or replay.get("ranking_rows") != expected_queries * 500
            or replay.get("top500_identical") is not True or replay.get("max_score_error") != 0.
            or not source.is_absolute() or not source.is_file() or source.resolve().is_relative_to(output)
            or file_hash(source) != replay.get("source_sha256")
            or binding.get(str(source)) != replay.get("source_sha256")
            or replay.get("score_cache_sha256") != file_hash(output / "score_cache" / split / state.BASELINE / "completed.json")):
        raise ValueError("Historical baseline replay is incomplete, changed or inexact")


def _evaluation(output, root, stage, name, context_signature, binding, chosen, index, root_files):
    cross = stage == "cross_test"
    split = "valid" if stage == "valid" else "test"
    nqueries, ngallery = (500, 3599) if split == "valid" else (7199, 7199)
    spec = state.grid()[name]
    signature = state.evaluation_signature(context_signature, stage, name)
    folder = output / stage / name
    files = verify.CROSS_REPORT_FILES if cross else verify.REPORT_FILES
    _nested_seal(output, folder, signature, (*files, "verification.json"), root_files)
    manifest, ident, checked = (read_json(folder / file) for file in
                               ("evaluation_manifest.json", "run_identity.json", "verification.json"))
    marker = folder / "completed.json"
    score_folder = output / "score_cache" / split / name
    cache_meta = read_json(score_folder / "metadata.json")
    cache_signature = digest({"context": context_signature, "split": split, "variant": name,
                              "spec": spec, "score_cache_version": 1})
    _nested_seal(output, score_folder, cache_signature, ("scores.npy", "identities.npz", "metadata.json"), root_files)
    provider_source = {"signature": cache_signature, "completed_sha256": file_hash(score_folder / "completed.json")}
    if (cache_meta.get("signature") != cache_signature or cache_meta.get("variant") != name
            or cache_meta.get("spec") != spec or cache_meta.get("split") != split
            or cache_meta.get("context_signature") != context_signature
            or cache_meta.get("retrieval_protocol_signature") != catalog.BASE_PROTOCOL_SIGNATURE
            or cache_meta.get("num_queries") != nqueries or cache_meta.get("gallery_count") != ngallery
            or cache_meta.get("dtype") != "float64"):
        raise ValueError("Full score cache identity differs")
    if (cache_meta.get("source_cache_sha256") != file_hash(output/"cache"/split/spec["proposal"]/"completed.json")
            or cache_meta.get("calibration_sha256") != file_hash(output/"calibration"/split/spec["proposal"]/"completed.json")):
        raise ValueError("Full scores are not bound to the sealed region/calibration caches")
    expected_protocol = verify.PROTOCOL if cross else "retrieval_v1"
    if (manifest.get("signature") != signature or manifest.get("context_signature") != context_signature
            or manifest.get("spec") != spec or manifest.get("variant") != name or manifest.get("stage") != stage
            or manifest.get("split") != split or manifest.get("protocol") != expected_protocol
            or manifest.get("ks") != state.KS or manifest.get("tie_seed") != 2026
            or manifest.get("num_queries") != nqueries or manifest.get("gallery_count") != ngallery
            or manifest.get("candidate_rule") != (verify.RULE if cross else "all_gallery_except_query")
            or manifest.get("provider_source") != provider_source
            or manifest.get("source_binding_sha256") != digest(binding)
            or manifest.get("formal_protocol") is not True
            or manifest.get("training_required") is not False or manifest.get("posthoc_research") is not True
            or manifest.get("test_parameter_search") is not False or manifest.get("selection_split") != "valid"
            or ident.get("evaluation_signature") != signature
            or not ident.get("key", "").startswith("f_region_evidence_")
            or any(manifest.get(key) != value for key, value in ident.items())):
        raise ValueError("Evaluation frozen identity, source or selection policy differs")
    if cross:
        protocol = manifest.get("derived_protocol", {})
        if (protocol.get("protocol_version") != verify.PROTOCOL or protocol.get("candidate_rule") != verify.RULE
                or protocol.get("base_protocol_signature") != catalog.BASE_PROTOCOL_SIGNATURE
                or protocol.get("candidate_identity_sha256") != verify._FORMAL_CANDIDATE_HASH
                or hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest() != manifest.get("retrieval_protocol_signature")
                or any((folder / filename).exists() for filename in ("metrics.json", "per_query.csv", "per_class.csv"))):
            raise ValueError("Cross-combination candidate protocol differs")
    elif manifest.get("derived_protocol") or manifest.get("retrieval_protocol_signature") != catalog.BASE_PROTOCOL_SIGNATURE:
        raise ValueError("Ordinary protocol differs")
    if split == "test":
        if (manifest.get("primary_variant") != chosen["winner"] or manifest.get("family_winners") != chosen["family_winners"]
                or manifest.get("frozen_selection_sha256") != file_hash(output / "selection.json")
                or datetime.fromisoformat(ident["started_at"]) < datetime.fromisoformat(chosen["frozen_at"])):
            raise ValueError("Test evaluation is not bound to the prior frozen valid selection")
    expected_count = 50395860 if cross else nqueries * (ngallery - 1)
    if (checked.get("status") != "verified" or checked.get("signature") != signature
            or checked.get("context_signature") != context_signature or checked.get("stage") != stage
            or checked.get("split") != split or checked.get("variant") != name or checked.get("spec") != spec
            or checked.get("ks") != state.KS or checked.get("tie_seed") != 2026
            or checked.get("num_queries") != nqueries or checked.get("ranking_rows") != nqueries * 500
            or checked.get("full_candidates_checked") != expected_count
            or checked.get("all_queries_full_candidate_topk_exact") is not True
            or checked.get("exact_reports") is not (not cross)
            or checked.get("provider_source") != provider_source
            or checked.get("source_binding_sha256") != digest(binding)
            or checked.get("source_file_count") != len(binding)
            or type(checked.get("aggregation_values_checked")) is not int or checked["aggregation_values_checked"] <= 0
            or set(checked.get("files", {})) != set(files)
            or any(file_hash(folder / filename) != sha for filename, sha in checked["files"].items())
            or checked.get("ranking_sha256") != file_hash(folder / "rankings.csv")):
        raise ValueError("Independent complete-cache ranking/metric receipt is missing or stale")
    _finite_error(checked.get("max_metric_absolute_error"), 1e-12, "metric")
    _finite_error(checked.get("max_saved_score_absolute_error"), 1e-12, "cached score")
    if (checked.get("candidate_count_min") != (6999 if cross else ngallery-1)
            or checked.get("candidate_count_max") != (7170 if cross else ngallery-1)
            or (cross and checked.get("candidate_identity_sha256") != verify._FORMAL_CANDIDATE_HASH)):
        raise ValueError("Full eligible-candidate count/hash differs")
    entry = index["results"][name]
    if entry.get("completed_sha256") != file_hash(marker):
        raise ValueError("Evaluation index completion binding differs")
    families = ["coverage"] if cross else ["exact", "coverage"]
    if set(checked.get("metric_query_counts", {})) != set(families):
        raise ValueError("Metric effective-query receipt families differ")
    for family in families:
        filename = "metrics.json" if family == "exact" else "coverage_metrics.json"
        report = read_json(folder / filename)
        _metric_values(report.get("metrics"), family)
        if (report.get("split") != split or report.get("num_queries") != nqueries or report.get("ks") != state.KS
                or report.get("metric_version") != family + "_v1" or report.get("test_evaluated") != (split == "test")
                or report.get("num_classes") != 37
                or report.get("metadata") != manifest or entry.get(filename[:-5]) != report.get("metrics")
                or checked["metric_query_counts"][family] != report.get("metric_query_counts")):
            raise ValueError("Metric report, effective denominator or index differs")
        for metric, count in report["metric_query_counts"].items():
            expected = (7000 if metric.startswith("Coverage-nDCG@") else 6200
                        if metric.startswith(("FullCoverageRecall@", "FullCoverage_mAP@")) else 7199) if cross else nqueries
            if count != expected:
                raise ValueError("Formal effective-query counts differ")
    if not cross:
        hits = checked.get("top10_hits")
        if (type(hits) is not int or not 0 <= hits <= nqueries*10 or entry.get("top10_hits") != hits
                or abs(read_json(folder/"metrics.json")["metrics"]["Precision@10"] - hits/(nqueries*10)) > 1e-12):
            raise ValueError("Verified integer exact Top-10 hit total differs")
    evidence = ["run_manifest.json", "completed.json", "selection.json", "valid_completed.json",
                "reference.json", f"inputs_{split}.json", f"formula_audit_{split}.json",
                f"calibration_audit_{split}.json", f"preserved_regions_{split}.json",
                f"baseline_replay_{split}.json", f"evaluation_index_{stage}.json", "source_protection_after.json",
                f"{stage}/{name}/verification.json"]
    item = dict(key=ident["key"], repository="wbm-region-retrieval", path=_relative(folder, root),
        category="cross_combination" if cross else "fusion_ablation", method="F",
        variant=("regional-evidence-cross/" if cross else "regional-evidence/") + name,
        split=split, manifest=_relative(folder/"evaluation_manifest.json", root), marker=_relative(marker, root),
        identity_schema_version=1, identity=_relative(folder/"run_identity.json", root),
        selection=_relative(output/"selection.json", root), coverage=None, supervision="self_supervised",
        display="区域证据F " + name + ("（跨组合）" if cross else ""),
        required_families=families, protocol=expected_protocol, posthoc=True,
        selection_rule=selection.RULE, valid_selected_primary=name == chosen["winner"],
        repeat_validation_of_existing_result=name == state.BASELINE,
        supporting_manifests=[_relative(output/name, root) for name in evidence],
        design_note=("Prespecified 3 proposal families (original Top-4, add remaining CC8 regions up to eight, "
            "or residual-pixel grouping up to eight) by 4 train-background rejection settings. All 12 valid "
            "recipes select the main winner and one frozen winner per nonbaseline family by integer exact "
            "Top-10 hits; ties prefer baseline, fewer maximum regions, weaker rejection then name. Only "
            "the three frozen family winners receive ordinary and cross-combination test; test does not "
            "select parameters. Original regional C/T/A score 0.75*(0.4*C+0.6*exp(-0.5*H))+0.25*A, "
            "frozen self-supervised cc8 h512 best_nce seed2026, original tokens and greedy query sqrt-area "
            "denominator preserved. Optional per-query-region net evidence subtracts a q75/q90/q95 "
            "linear quantile against the same 4096 distinct-wafer original-train region bank. No new "
            "CNN training; train-only calibration uses no labels. New proposals encode additional regions "
            "with the frozen encoder. Baseline valid is a repeat-validation control, not an independent "
            "replication; historical ordinary test baseline is replayed exactly and only referenced. "
            "Independent complete-cache ordering/metrics and sampled regional/calibration audits have "
            "distinct scopes. Cross protocol excludes self and identical full-label combinations and "
            "reports only the original six coverage_v1 metrics. Posthoc diagnostic research; no claim "
            "of untouched test design, seed uncertainty, or replacement of the designated main method."))
    catalog.execution_identity(root, item, marker, manifest)
    return item


def proposal(output, *, root=ROOT):
    output, root = Path(output).resolve(), Path(root).resolve()
    _relative(output, root / "artificial")
    if output == root / "artificial":
        raise ValueError("Expected an isolated new run")
    manifest = read_signed(output / "run_manifest.json")
    context = manifest.get("context", {})
    signature = digest(context)
    if (manifest.get("version") != state.VERSION or context.get("version") != state.VERSION
            or context.get("grid") != verify.independent_grid() or context.get("ks") != state.KS
            or context.get("selection_rule") != selection.RULE or context.get("cnn_training") is not False
            or context.get("test_parameter_search") is not False or context.get("posthoc_research") is not True
            or context.get("seed") != 2026 or context.get("reference_design") != state.REFERENCE
            or context.get("proposals") != proposals.PROPOSAL_SPECS
            or context.get("reference") != read_json(output / "reference.json")):
        raise ValueError("Registration requires the complete prespecified evidence study")
    required = {"run_manifest.json", "selection.json", "valid_completed.json", "reference.json", "status.json",
                "source_protection_before.json", "source_protection_after.json"}
    for split in ("valid", "test"):
        required.update({f"inputs_{split}.json", f"formula_audit_{split}.json", f"calibration_audit_{split}.json",
                         f"preserved_regions_{split}.json", f"baseline_replay_{split}.json"})
    required.update(f"evaluation_index_{stage}.json" for stage in ("valid", "test", "cross_test"))
    root_files = _complete(output, signature, required)["files"]
    status = read_json(output / "status.json")
    protection = read_json(output / "source_protection_after.json")
    if status.get("status") != "completed" or protection.get("status") != "unchanged":
        raise ValueError("Registration requires completed evaluation and source protection")
    started = _stamp(manifest.get("started_at"),"root started_at")
    completed = _stamp(status.get("completed_at"),"root completed_at")
    if manifest.get("run_id") != started.strftime("%Y%m%dT%H%M%S%z")+"__f_region_evidence" or started > completed:
        raise ValueError("Root time identity or completion order differs")
    code = context.get("code")
    if not isinstance(code, dict) or not code:
        raise ValueError("Scientific source-code hashes are missing")
    for name, expected in code.items():
        path = root / name
        if Path(name).is_absolute() or ".." in Path(name).parts or not path.is_file() or file_hash(path) != expected:
            raise ValueError("Scientific code changed: " + name)
    from .run import code_identity
    if code != code_identity() or context.get("matching") != {
            "lambda_pos":0.,"lambda_scale":0.,"shape_threshold":0.,"score_threshold":0.,"weight":"sqrt_area"}:
        raise ValueError("Full scientific dependency set or frozen matcher differs")
    bindings = {}
    for split in ("valid", "test"):
        inputs = read_json(output / f"inputs_{split}.json")
        binding = inputs.get("source_binding")
        _bound_files(binding)
        if inputs.get("context_signature") != signature or (split == "valid" and binding != context.get("source_binding")):
            raise ValueError("Frozen input binding differs from scientific context")
        if split == "test" and inputs.get("selection_sha256") != file_hash(output / "selection.json"):
            raise ValueError("Test inputs are not bound to the valid decision")
        bindings[split] = binding
    chosen = read_signed(output / "selection.json")
    valid_marker = read_json(output / "valid_completed.json")
    valid_index = read_json(output / "evaluation_index_valid.json")
    expected = selection.select(valid_index.get("results", {}))
    if (chosen.get("context_signature") != signature or any(chosen.get(k) != v for k,v in expected.items())
            or chosen.get("valid_index_sha256") != file_hash(output / "evaluation_index_valid.json")
            or valid_marker.get("status") != "completed" or valid_marker.get("context_signature") != signature
            or valid_marker.get("selection_sha256") != file_hash(output / "selection.json")
            or valid_marker.get("evaluation_index_sha256") != chosen["valid_index_sha256"]
            or set(valid_marker.get("evaluation_completions", {})) != set(state.grid())
            or any(valid_marker["evaluation_completions"][n] != file_hash(output/"valid"/n/"completed.json") for n in state.grid())
            or status.get("winner") != chosen["winner"] or status.get("family_winners") != chosen["family_winners"]):
        raise ValueError("Frozen complete-valid selection identity differs")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00", chosen.get("frozen_at", "")):
        raise ValueError("Selection requires the actual timezone-aware frozen time")
    frozen = _stamp(chosen["frozen_at"],"valid frozen_at")
    if not started <= frozen <= completed:
        raise ValueError("Valid selection falls outside the real run interval")
    examples = {f"explanations/valid/{name}/examples.json" for name in (state.BASELINE,*chosen["selected_variants"])}
    examples.update(f"explanations/test/{name}/examples.json" for name in chosen["selected_variants"])
    if not examples <= set(root_files):
        raise ValueError("Root completion omits required regional explanations")
    items = []
    for split in ("valid", "test"):
        _audits(output, split, state.grid() if split == "valid" else
                {state.BASELINE: state.grid()[state.BASELINE], **chosen["selected_specs"]}, bindings[split])
        for name in ("base4", "top8", "residual8"):
            region_folder = output/"cache"/split/name
            region_meta = read_json(region_folder/"metadata.json")
            region_signature = digest({"context_signature":signature,"split":split,"proposal":name,
                "spec":context["proposals"][name],"cache_version":1})
            _nested_seal(output, region_folder, region_signature,
                ("regions.npz","h512.npy","region_labels.npy","region_trace.jsonl","audit.json","metadata.json","source_binding.json"),root_files)
            if (region_meta.get("signature") != region_signature or region_meta.get("split") != split
                    or region_meta.get("proposal") != name or region_meta.get("context_signature") != signature
                    or region_meta.get("spec") != context["proposals"][name]):
                raise ValueError("Region cache scientific identity differs")
            region_binding=read_json(region_folder/"source_binding.json")
            if (not isinstance(region_binding,dict) or not region_binding
                    or any(bindings[split].get(path)!=sha for path,sha in region_binding.items())
                    or region_meta.get("source_binding_sha256") != digest(region_binding)):
                raise ValueError("Region cache source binding differs from the frozen inputs")
            cal_folder = output/"calibration"/split/name
            cal_signature = digest({"context":signature,"split":split,"proposal":name,
                "reference":context["reference"]["reference_identity_sha256"],"version":1})
            _nested_seal(output,cal_folder,cal_signature,("thresholds.npz","metadata.json"),root_files)
            cal_meta=read_json(cal_folder/"metadata.json")
            if (cal_meta.get("signature") != cal_signature or cal_meta.get("context_signature") != signature
                    or cal_meta.get("split") != split or cal_meta.get("proposal") != name
                    or cal_meta.get("reference") != context["reference"] or cal_meta.get("labels_used") is not False
                    or cal_meta.get("train_query_disjoint") is not True
                    or cal_meta.get("quantiles") != {"q75":.75,"q90":.9,"q95":.95}
                    or cal_meta.get("quantile_method") != "linear"
                    or cal_meta.get("tau_equal_one_policy") != "zero_evidence_when_background_quantile_equals_one"
                    or cal_meta.get("cache_sha256") != file_hash(region_folder/"completed.json")):
                raise ValueError("Calibration cache/reference binding differs")
    for stage in ("valid", "test", "cross_test"):
        specs = state.grid() if stage == "valid" else chosen["selected_specs"]
        index = read_json(output / f"evaluation_index_{stage}.json")
        split = "valid" if stage == "valid" else "test"
        if (index.get("context_signature") != signature or index.get("stage") != stage
                or index.get("split") != split or index.get("specs") != specs
                or set(index.get("results", {})) != set(specs)
                or {p.name for p in (output/stage).iterdir() if p.is_dir()} != set(specs)):
            raise ValueError("Evaluation stage is incomplete or contains unselected test variants")
        for name in specs:
            timestamp = _stamp(read_json(output/stage/name/"run_identity.json").get("started_at"),"evaluation started_at")
            if not started <= timestamp <= completed or (stage == "valid" and timestamp > frozen):
                raise ValueError("Evaluation start lies outside its valid/test time interval")
            items.append(_evaluation(output, root, stage, name, signature, bindings[split], chosen, index, root_files))
    # The historical test control is deliberately not a registered evaluation,
    # but its complete cached scores still require the same scientific seal.
    baseline = output/"score_cache"/"test"/state.BASELINE
    baseline_spec = state.grid()[state.BASELINE]
    baseline_signature = digest({"context":signature,"split":"test","variant":state.BASELINE,
                                 "spec":baseline_spec,"score_cache_version":1})
    _nested_seal(output,baseline,baseline_signature,("scores.npy","identities.npz","metadata.json"),root_files)
    bm=read_json(baseline/"metadata.json")
    if (bm.get("signature") != baseline_signature or bm.get("context_signature") != signature
            or bm.get("variant") != state.BASELINE or bm.get("spec") != baseline_spec or bm.get("split") != "test"
            or bm.get("num_queries") != 7199 or bm.get("gallery_count") != 7199 or bm.get("dtype") != "float64"
            or bm.get("retrieval_protocol_signature") != catalog.BASE_PROTOCOL_SIGNATURE
            or bm.get("source_cache_sha256") != file_hash(output/"cache/test/base4/completed.json")
            or bm.get("calibration_sha256") != file_hash(output/"calibration/test/base4/completed.json")):
        raise ValueError("Historical test baseline full-score control identity differs")
    if len(items) != 18 or len({item["key"] for item in items}) != 18:
        raise ValueError("Expected exactly 12 valid and 3+3 frozen test identities")
    return {"schema_version": 1, "runs": items}


def write_proposal(output, *, root=ROOT):
    value = proposal(output, root=root)
    path = Path(output) / "catalog_sources.json"
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Registration proposal changed")
    else:
        _atomic_write(path, _json_bytes(value))
    return path


def apply_proposal(output, *, workspace=None, root=ROOT):
    """Verify all sources and render in memory, then back up and replace catalogue files."""
    output, root = Path(output).resolve(), Path(root).resolve()
    workspace = root.parent if workspace is None else Path(workspace).resolve()
    if (workspace / "wbm-region-retrieval").resolve() != root:
        raise ValueError("Workspace must contain this wbm-region-retrieval repository")
    proposed = catalog.read(write_proposal(output, root=root))
    folder = root / "results_catalog"
    with _lock(folder):
        original = {name: (folder / name).read_bytes() for name in ("sources.json", "snapshot.json")}
        sources = json.loads(original["sources.json"])
        previous = json.loads(original["snapshot.json"])
        fingerprint = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
        if previous.get("source_spec_sha256") != fingerprint:
            raise ValueError("Existing catalogue sources and snapshot differ; collect/check before registration")
        old_generated = catalog.render(previous)
        for name, content in old_generated.items():
            path = folder / name
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                raise ValueError(f"Existing generated catalogue is stale: {name}")
            original[name] = path.read_bytes()
        entries = {item["key"]: item for item in sources["runs"]}
        old_runs = {item["key"]: item for item in previous["runs"]}
        if (len(entries) != len(sources["runs"]) or len(old_runs) != len(previous["runs"])
                or set(entries) != set(old_runs)):
            raise ValueError("Existing catalogue contains duplicate or inconsistent keys")
        additions = []
        for item in proposed["runs"]:
            if item["key"] in entries:
                if entries[item["key"]] != item:
                    raise ValueError(f"Conflicting catalogue key: {item['key']}")
            else:
                additions.append(item)
        if not additions:
            return {"status": "already_registered", "new_results": 0}
        merged = {**sources, "runs": [*sources["runs"], *additions]}
        # Collection verifies every report/identity before any shared data file is changed.
        updated = catalog.collect(workspace, merged, previous)
        new_runs = {item["key"]: item for item in updated["runs"]}
        expected_keys = set(old_runs) | {item["key"] for item in additions}
        if (len(new_runs) != len(updated["runs"]) or set(new_runs) != expected_keys
                or any(new_runs.get(key) != value for key, value in old_runs.items())):
            raise ValueError("Collection changed a prior result or omitted/added an unexpected identity")
        generated = catalog.render(updated)
        if any(not (folder / name).is_file() or (folder / name).read_bytes() != content
               for name, content in original.items()):
            raise ValueError("Catalogue changed concurrently during collection")
        contents = {"snapshot.json": _json_bytes(updated),
                    **{name: text.encode("utf-8") for name, text in generated.items()},
                    "sources.json": _json_bytes(merged)}
        if any(name not in original and (folder / name).exists() for name in contents):
            raise ValueError("New generated output would replace an untracked catalogue file")
        before = {name: original.get(name) for name in contents}
        backup_parent = output / "catalog_backup"
        backup_parent.mkdir(exist_ok=True)
        backup = Path(tempfile.mkdtemp(prefix=datetime.now(TZ).strftime("%Y%m%dT%H%M%S%z") + "_", dir=backup_parent))
        for name, data in before.items():
            if data is not None:
                path = backup / "files" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        audit = {"schema_version": 1, "status": "prepared", "catalog": str(folder),
                 "new_keys": [item["key"] for item in additions],
                 "before_sha256": {name: hashlib.sha256(data).hexdigest() if data is not None else None
                                   for name, data in before.items()},
                 "after_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}}
        (backup / "manifest.json").write_bytes(_json_bytes(audit))
        written = []
        try:
            for name, content in contents.items():
                if before[name] != ((folder / name).read_bytes() if (folder / name).exists() else None):
                    raise ValueError(f"Catalogue changed concurrently: {name}")
                written.append(name)
                _atomic_write(folder / name, content)
            for name, content in contents.items():
                if (folder / name).read_bytes() != content:
                    raise ValueError(f"Catalogue write verification failed: {name}")
        except BaseException as error:
            failures = []
            for name in reversed(written):
                try:
                    current = (folder/name).read_bytes() if (folder/name).exists() else None
                    if current == before[name]:
                        continue
                    if current != contents[name]:
                        raise ValueError("Concurrent external change preserved during rollback: " + name)
                    if before[name] is None:
                        (folder / name).unlink(missing_ok=True)
                    else:
                        _atomic_write(folder / name, before[name])
                except BaseException as restore_error:
                    failures.append(f"{name}: {restore_error}")
            audit.update(status="rollback_failed" if failures else "rolled_back", error=str(error), rollback_errors=failures)
            (backup / "manifest.json").write_bytes(_json_bytes(audit))
            if failures:
                raise RuntimeError(f"Catalogue rollback incomplete; backups at {backup}: {failures}") from error
            raise
        audit["status"] = "committed"
        (backup / "manifest.json").write_bytes(_json_bytes(audit))
        return {"status": "registered", "new_results": len(additions), "backup": str(backup)}



def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--apply',action='store_true')
    args=p.parse_args(argv)
    output=(ROOT/args.run_root).resolve()
    print(json.dumps(apply_proposal(output) if args.apply else {'proposal':str(write_proposal(output))},ensure_ascii=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
