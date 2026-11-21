"""Register complete frozen regional-WBBS studies without changing old results."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from b1_structure_fusion.register import _atomic_write, _json_bytes, _lock
from f_region_structure.register import _complete, _finite_error, _metric_values, _relative
from f_region_structure.state import TZ, read_signed
from f_region_evidence.register import _bound_files, _nested_seal, _stamp
from results_catalog import build as catalog
from wbm_retrieval.common import digest, file_hash, read_json
from . import protocol, selection, state, verify


def _score_cache(output,split,name,spec,ctx,root_files):
    folder=output/"score_cache"/split/name
    signature=digest({"context":ctx,"split":split,"variant":name,"spec":spec,"score_cache_version":1})
    _nested_seal(output,folder,signature,("scores.npy","identities.npz","metadata.json"),root_files)
    m=read_json(folder/"metadata.json");frozen=protocol.FROZEN[split]
    if (m.get("signature")!=signature or m.get("context_signature")!=ctx or m.get("variant")!=name
            or m.get("split")!=split or m.get("spec")!=spec or m.get("dtype")!="float64"
            or m.get("self_scores")!="zero; never ranked"
            or m.get("num_queries")!=frozen["num_queries"] or m.get("gallery_count")!=frozen["gallery_count"]
            or m.get("retrieval_protocol_signature")!=protocol.BASE_PROTOCOL_SIGNATURE):
        raise ValueError("Full score-cache scientific identity differs")
    if name!=state.GLOBAL:
        if (m.get("source_cache_sha256")!=file_hash(output/"cache"/split/"completed.json")
                or m.get("counts_sha256")!=file_hash(output/"counts"/split/"completed.json")):
            raise ValueError("Full regional scores are detached from point/count caches")
    else:
        copy=read_json(output/"global_wbbs_copy_valid.json")
        source=m.get("source_receipt",{})
        if (split!="valid" or source.get("status")!="verified" or source.get("component")!="WBBS"
                or source.get("split")!="valid" or source.get("num_queries")!=500
                or source.get("gallery_count")!=3599 or source.get("nonself_pairs")!=1799000
                or source.get("retrieval_protocol_signature")!=protocol.BASE_PROTOCOL_SIGNATURE
                or copy.get("source_receipt")!=source or copy.get("status")!="verified"
                or copy.get("context_signature")!=ctx or copy.get("variant")!=state.GLOBAL
                or copy.get("split")!="valid" or copy.get("num_queries")!=500 or copy.get("gallery_count")!=3599
                or copy.get("self_storage_conversion")!="source NaN to zero; excluded before ranking"
                or copy.get("score_cache_signature")!=signature
                or copy.get("score_cache_completed_sha256")!=file_hash(folder/"completed.json")
                or copy.get("full_nonself_pairs_checked")!=1799000
                or copy.get("nonself_scores_bitwise_identical") is not True or copy.get("max_score_absolute_error")!=0.):
            raise ValueError("Global WBBS benchmark lacks its complete byte-identical source replay")
    return {"signature":signature,"completed_sha256":file_hash(folder/"completed.json")}


def _evaluation(output,root,stage,name,spec,ctx,binding,chosen,index,root_files,audit):
    split="valid" if stage.endswith("valid") else "test";cross=stage.startswith("cross_")
    nq,ng=(500,3599) if split=="valid" else (7199,7199)
    folder=output/stage/name;signature=state.evaluation_signature(ctx,stage,name,spec)
    files=verify.CROSS_REPORT_FILES if cross else verify.REPORT_FILES
    _nested_seal(output,folder,signature,(*files,"verification.json"),root_files)
    manifest=read_json(folder/"evaluation_manifest.json");identity=read_json(folder/"run_identity.json")
    checked=read_json(folder/"verification.json");entry=index["results"][name]
    source=_score_cache(output,split,name,spec,ctx,root_files)
    expected_protocol=protocol.PROTOCOL if cross else "retrieval_v1"
    if (manifest.get("signature")!=signature or manifest.get("context_signature")!=ctx
            or manifest.get("variant")!=name or manifest.get("spec")!=spec
            or manifest.get("stage")!=stage or manifest.get("split")!=split
            or manifest.get("protocol")!=expected_protocol or manifest.get("ks")!=state.KS
            or manifest.get("tie_seed")!=2026 or manifest.get("num_queries")!=nq or manifest.get("gallery_count")!=ng
            or manifest.get("candidate_rule")!=audit["candidate_rule"]
            or manifest.get("retrieval_protocol_signature")!=audit["retrieval_protocol_signature"]
            or manifest.get("provider_source")!=source or manifest.get("source_binding_sha256")!=digest(binding)
            or manifest.get("formal_protocol") is not True or manifest.get("training_required") is not False
            or manifest.get("posthoc_research") is not True or manifest.get("test_parameter_search") is not False
            or manifest.get("selection_split")!="valid" or manifest.get("selection_protocol")!=protocol.PROTOCOL
            or identity.get("evaluation_signature")!=signature
            or not identity.get("key","").startswith("f_region_wbbs_")
            or any(manifest.get(k)!=v for k,v in identity.items())):
        raise ValueError("Evaluation identity, protocol, source or scientific settings differ")
    if cross:
        derived=manifest.get("derived_protocol",{})
        if (derived.get("candidate_identity_sha256")!=audit["candidate_identity_sha256"]
                or derived.get("candidate_rule")!=protocol.RULE or derived.get("protocol_version")!=protocol.PROTOCOL
                or derived.get("base_protocol_signature")!=protocol.BASE_PROTOCOL_SIGNATURE
                or hashlib.sha256(json.dumps(derived,sort_keys=True).encode()).hexdigest()!=manifest["retrieval_protocol_signature"]
                or any((folder/f).exists() for f in ("metrics.json","per_query.csv","per_class.csv"))):
            raise ValueError("Derived cross-combination protocol differs")
    elif manifest.get("derived_protocol"):
        raise ValueError("Ordinary evaluation cannot carry a derived candidate pool")
    if split=="test" and (manifest.get("primary_variant")!=chosen["winner"]
            or manifest.get("best_regional_wbbs")!=chosen["best_regional_wbbs"]
            or manifest.get("frozen_selection_sha256")!=file_hash(output/"selection.json")
            or _stamp(identity.get("started_at"),"evaluation")<_stamp(chosen["frozen_at"],"selection")):
        raise ValueError("Test evaluation is detached from its prior frozen valid selection")
    if (checked.get("status")!="verified" or checked.get("signature")!=signature
            or checked.get("context_signature")!=ctx or checked.get("stage")!=stage
            or checked.get("split")!=split or checked.get("variant")!=name or checked.get("spec")!=spec
            or checked.get("ks")!=state.KS or checked.get("tie_seed")!=2026
            or checked.get("num_queries")!=nq or checked.get("ranking_rows")!=nq*500
            or checked.get("full_candidates_checked")!=audit["full_candidates_checked"]
            or checked.get("all_queries_full_candidate_topk_exact") is not True
            or checked.get("exact_reports") is not (not cross)
            or checked.get("provider_source")!=source or checked.get("source_binding_sha256")!=digest(binding)
            or checked.get("source_file_count")!=len(binding)
            or checked.get("candidate_protocol_audit")!=audit
            or any(checked.get(k)!=audit[k] for k in ("candidate_identity_sha256","candidate_count_min",
                                                    "candidate_count_max","query_ids_sha256","gallery_ids_sha256"))
            or type(checked.get("aggregation_values_checked")) is not int or checked["aggregation_values_checked"]<=0
            or set(checked.get("files",{}))!=set(files)
            or any(file_hash(folder/f)!=sha for f,sha in checked["files"].items())
            or checked.get("ranking_sha256")!=file_hash(folder/"rankings.csv")
            or entry.get("completed_sha256")!=file_hash(folder/"completed.json")):
        raise ValueError("Independent complete-candidate ranking/metric receipt is incomplete or stale")
    _finite_error(checked.get("max_metric_absolute_error"),1e-12,"metric")
    _finite_error(checked.get("max_saved_score_absolute_error"),1e-12,"cached score")
    families=["coverage"] if cross else ["exact","coverage"]
    if set(checked.get("metric_query_counts",{}))!=set(families):
        raise ValueError("Metric receipt families differ")
    for family in families:
        stem="metrics" if family=="exact" else "coverage_metrics";report=read_json(folder/(stem+".json"))
        _metric_values(report.get("metrics"),family)
        if (report.get("split")!=split or report.get("num_queries")!=nq or report.get("num_classes")!=37
                or report.get("metric_version")!=family+"_v1" or report.get("ks")!=state.KS
                or report.get("test_evaluated")!=(split=="test") or report.get("metadata")!=manifest
                or entry.get(stem)!=report.get("metrics")
                or checked["metric_query_counts"][family]!=report.get("metric_query_counts")):
            raise ValueError("Full metric report/index/count identity differs")
        wanted=(audit["metric_query_counts"] if family=="coverage" else {k:nq for k in report["metrics"]})
        if report["metric_query_counts"]!=wanted:
            raise ValueError("Formal effective-query denominators differ")
    if not cross:
        hits=checked.get("top10_hits")
        if (type(hits) is not int or not 0<=hits<=nq*10 or entry.get("top10_hits")!=hits
                or abs(entry["metrics"]["Precision@10"]-hits/(nq*10))>1e-12):
            raise ValueError("Exact integer Top-10 hits disagree")
    evidence=["run_manifest.json","completed.json","settings.json","selection.json","valid_completed.json",
        f"inputs_{split}.json",f"source_compatibility_{split}.json",f"formula_audit_{split}.json",f"protocol_audit_{'cross_' if cross else ''}{split}.json",
        f"evaluation_index_{stage}.json",f"cache/{split}/audit.json",f"counts/{split}/metadata.json",
        f"score_cache/{split}/{name}/metadata.json",f"{stage}/{name}/verification.json","source_protection_after.json"]
    evidence += (["global_wbbs_copy_valid.json"] if name==state.GLOBAL else [f"baseline_replay_{split}.json"])
    if split=="test":evidence += ["test_context.json","historical_test_controls.json","baseline_replay_cross_test.json"]
    item={"key":identity["key"],"repository":"wbm-region-retrieval","path":_relative(folder,root),
        "category":"cross_combination" if cross else "fusion_ablation","method":"wbbs" if name==state.GLOBAL else "F",
        "variant":("regional-WBBS-cross/" if cross else "regional-WBBS/")+name,"split":split,
        "manifest":_relative(folder/"evaluation_manifest.json",root),"marker":_relative(folder/"completed.json",root),
        "identity_schema_version":1,"identity":_relative(folder/"run_identity.json",root),
        "selection":_relative(output/"selection.json",root),"coverage":None,
        "supervision":"untrained" if name==state.GLOBAL else "self_supervised",
        "display":("整图WBBS基准" if name==state.GLOBAL else "区域WBBS F "+name)+("（跨组合）" if cross else ""),
        "required_families":families,"protocol":expected_protocol,"posthoc":True,
        "valid_selected_primary":name==chosen["winner"],"benchmark_only":name==state.GLOBAL,
        "repeat_validation_of_existing_result":name==state.BASELINE and not cross,
        "selection_rule":state.SELECTION_RULE,
        "supporting_manifests":[_relative(output/p,root) for p in evidence],
        "design_note":"Frozen Top-4 h512 CNN and original greedy query sqrt-area matching. Regional WBBS changes only the structure term: 0.75*(a*C+(1-a)*W)+0.25*A; all original points or original full-wafer-denoised subsets, count penalty 0/5, CNN weight .25/.4/.6, total_dies=2061. Original regional J is a control. Complete cross-valid Coverage-nDCG@10 selects the primary and best regional WBBS; 1e-12 ties follow the frozen simplicity rule. Whole-map WBBS is benchmark only. Four prespecified weight-.4 controls plus the best regional WBBS, deduplicated, receive ordinary/cross test; no test reselection. No training, new CNN forward passes, Top-8 completion or score calibration. Full cached-score ordering and all denominator/metric aggregations are independently checked; original WBBS regional formulas/counts and greedy are separately sampled. Existing J ordinary/cross test scores are replayed exactly and referenced, not re-registered. Posthoc research, not a claim of untouched test design or cross-seed significance."}
    catalog.execution_identity(root,item,folder/"completed.json",manifest)
    return item


def _protocol_audit(output,stage):
    split="valid" if stage.endswith("valid") else "test";cross=stage.startswith("cross_")
    audit=read_json(output/f"protocol_audit_{stage}.json");f=protocol.FROZEN[split]
    nq,ng=f["num_queries"],f["gallery_count"]
    expected_counts={f"{stem}@{k}":(f["ndcg_valid"] if stem=="Coverage-nDCG" else
        f["full_valid"] if stem in ("FullCoverageRecall","FullCoverage_mAP") else nq) if cross else nq
        for stem in protocol.STEMS for k in state.KS}
    if (audit.get("status")!="verified" or audit.get("split")!=split
            or audit.get("protocol")!=(protocol.PROTOCOL if cross else "retrieval_v1")
            or audit.get("base_protocol_signature")!=protocol.BASE_PROTOCOL_SIGNATURE
            or audit.get("retrieval_protocol_signature")!=(f["derived_signature"] if cross else protocol.BASE_PROTOCOL_SIGNATURE)
            or audit.get("candidate_rule")!=(protocol.RULE if cross else "all_gallery_except_query")
            or audit.get("num_queries")!=nq or audit.get("gallery_count")!=ng
            or audit.get("num_classes")!=37 or audit.get("ks")!=state.KS
            or any(audit.get(k)!=f[k] for k in ("query_ids_sha256","gallery_ids_sha256"))
            or audit.get("candidate_count_min")!=(f["candidate_count_min"] if cross else ng-1)
            or audit.get("candidate_count_max")!=(f["candidate_count_max"] if cross else ng-1)
            or audit.get("full_candidates_checked")!=(f["candidate_pairs"] if cross else nq*(ng-1))
            or audit.get("metric_query_counts")!=expected_counts
            or audit.get("queries_without_full_coverage_candidates")!=(nq-f["full_valid"] if cross else 0)
            or audit.get("queries_without_positive_coverage_candidates")!=(nq-f["ndcg_valid"] if cross else 0)
            or not re.fullmatch(r"[0-9a-f]{64}",audit.get("candidate_identity_sha256",""))
            or (cross and audit["candidate_identity_sha256"]!=f["candidate_identity_sha256"])):
        raise ValueError("Independent formal candidate protocol/effective-query audit differs")
    groups=audit.get("groups",{})
    if (set(groups)!={"single_label","mixed_label","by_label_count"}
            or set(groups["by_label_count"])!={"1","2","3","4"}
            or groups["single_label"]!=groups["by_label_count"]["1"]):
        raise ValueError("Candidate audit label groups are incomplete")
    for children in ([groups["single_label"],groups["mixed_label"]],list(groups["by_label_count"].values())):
        for field in ("num_queries","queries_without_full_coverage_candidates","queries_without_positive_coverage_candidates"):
            if any(type(g.get(field)) is not int or g[field]<0 for g in children) or sum(g[field] for g in children)!=audit[field]:
                raise ValueError("Candidate audit group denominators differ")
        for metric,total in expected_counts.items():
            if any(type(g.get("metric_query_counts",{}).get(metric)) is not int for g in children) or sum(g["metric_query_counts"][metric] for g in children)!=total:
                raise ValueError("Candidate audit grouped effective-query counts differ")
    return audit


def _audits(output,split,specs,ctx,binding,root_files):
    from . import counts, prepare
    formula=read_json(output/f"formula_audit_{split}.json")
    queries=formula.get("query_ids",[])
    if (formula.get("status")!="verified" or formula.get("num_queries")!=37 or len(set(queries))!=37
            or any(type(q) is not int or q<0 for q in queries)
            or set(formula.get("variants",{}))!=set(specs)
            or formula.get("candidate_pairs")!=740*len(specs) or formula.get("bb_counts_exact") is not True
            or type(formula.get("bb_region_pairs_checked")) is not int or formula["bb_region_pairs_checked"]<=0):
        raise ValueError("Independent sampled regional WBBS formula audit is incomplete")
    for row in [formula,*formula["variants"].values()]:
        if row.get("query_ids")!=queries:
            raise ValueError("Regional formula audits disagree on their fixed representative queries")
        _finite_error(row.get("max_raw_formula_absolute_error"),2e-6,"regional formula")
        _finite_error(row.get("max_matching_absolute_error"),1e-12,"greedy matching")
    _finite_error(formula.get("max_wbbs_absolute_error"),1e-15,"original WBBS formula")
    if any(row.get("candidate_pairs")!=740 for row in formula["variants"].values()):
        raise ValueError("Each regional variant requires all 740 sampled map pairs")
    folder=output/"cache"/split;signature=prepare.cache_signature(ctx,split)
    marker=_nested_seal(output,folder,signature,prepare.CACHE_FILES,root_files)
    meta=read_json(folder/"metadata.json");audit=read_json(folder/"audit.json")
    cachebinding=read_json(folder/"source_binding.json")
    if (set(marker["files"])!=prepare.CACHE_FILES or meta.get("signature")!=signature
            or meta.get("context_signature")!=ctx or meta.get("split")!=split or meta.get("cache_version")!=1
            or meta.get("point_modes")!=["all_points","wbbs_denoised"]
            or meta.get("descriptor_settings")!=prepare._settings()
            or meta.get("point_axis_order")!=["stick","ball","rho"]
            or meta.get("training_required") is not False or meta.get("cnn_forward_passes")!=0
            or not isinstance(cachebinding,dict) or not cachebinding
            or any(binding.get(p)!=sha for p,sha in cachebinding.items())
            or meta.get("source_binding_sha256")!=digest(cachebinding)
            or set(meta.get("audit_ids",[]))!=set(queries)
            or audit.get("status")!="verified" or audit.get("split")!=split
            or audit.get("maps")!=protocol.FROZEN[split]["gallery_count"]
            or any(audit.get(k) is not True for k in ("old_features_bitwise_identical","old_descriptors_bitwise_identical","old_masks_bitwise_identical"))
            or audit.get("original_wbbs_audit_maps")!=37
            or len(audit.get("original_wbbs_audits",[]))!=37
            or {r.get("sample_id") for r in audit["original_wbbs_audits"]}!=set(queries)
            or audit.get("descriptor_tolerance")!=1e-14):
        raise ValueError("Point cache provenance/original region preservation audit differs")
    for field in ("max_joint_reconstruction_error","max_geometry_reconstruction_error","max_original_wbbs_descriptor_error"):
        _finite_error(audit.get(field),1e-14,field)
    for row in audit["original_wbbs_audits"]:
        _finite_error(row.get("max_descriptor_error"),1e-14,"original tensor voting")
    countfolder=output/"counts"/split;cm=read_signed(countfolder/"metadata.json")
    _nested_seal(output,countfolder,cm["signature"],("metadata.json","query_tokens.npy","identities.npz","timing.json","all_points.npy","wbbs_denoised.npy"),root_files)
    if (cm.get("context_signature")!=ctx or cm.get("split")!=split
            or cm.get("source_cache")!=str(folder.resolve())
            or cm.get("source_cache_completed_sha256")!=file_hash(folder/"completed.json")
            or cm.get("retrieval_protocol_signature")!=protocol.BASE_PROTOCOL_SIGNATURE
            or cm.get("upper_triangle") is not (split=="test")
            or cm.get("diagonal")!="original exact mutual-nearest-neighbour count; never forced"):
        raise ValueError("BB count cache is detached from the fixed regional point cache")
    # The root seal has checked every byte. This validates shapes, identity arrays,
    # and the exact row-receipt inventory without hashing the large matrices again.
    counts.read_counts(countfolder,check=False)


def _baseline(output,stage,ctx,binding,root_files):
    split="valid" if stage=="valid" else "test";nq=protocol.FROZEN[split]["num_queries"]
    source=_score_cache(output,split,state.BASELINE,state.grid()[state.BASELINE],ctx,root_files)
    replay=read_json(output/f"baseline_replay_{stage}.json")
    if (replay.get("status")!="verified" or replay.get("split")!=stage or replay.get("num_queries")!=nq
            or replay.get("ranking_rows")!=nq*500 or replay.get("top500_identical") is not True
            or replay.get("max_score_error")!=0. or replay.get("score_cache_sha256")!=source["completed_sha256"]
            or binding.get(replay.get("source"))!=replay.get("source_sha256")
            or not isinstance(replay.get("source"),str) or not Path(replay["source"]).is_absolute()
            or Path(replay["source"]).name!="rankings.csv"
            or file_hash(replay["source"])!=replay["source_sha256"]):
        raise ValueError("Historical regional-J baseline does not have a complete byte-identical replay")
    return replay


def _compatibility(output,root,split,ctx,binding,context):
    """Admit exactly the documented administrative builder-version transition.

    Both actual byte sources are retained; the current canonical file is never
    represented as having its historical hash. No scoring-source substitution
    is admitted by this rule.
    """
    old="13d294548056ff6905968f39d0cd8f793672485b29cd8caf4674028f80f1bd6b"
    current="aca87c654e2c4bf1b4576695f32376cb4272593e1b3175ce2135faedc14b65cf"
    archive=root/"f_region_wbbs/provenance/results_catalog_build_13d29454.py.txt"
    canonical=root/"results_catalog/build.py"
    expected=[{"historical_path":"results_catalog/build.py","expected_sha256":old,
        "archived_path":str(archive.resolve()),"archived_sha256":old,
        "current_path":str(canonical.resolve()),"current_sha256":current,"scope":"catalog administration only"}]
    receipt=read_json(output/f"source_compatibility_{split}.json")
    inputs=read_json(output/f"inputs_{split}.json")
    if (receipt.get("context_signature")!=ctx or receipt.get("source_compatibility")!=expected
            or inputs.get("source_compatibility")!=expected or context.get("source_compatibility")!=expected
            or (split=="test" and receipt.get("selection_sha256")!=file_hash(output/"selection.json"))
            or any(binding.get(str(path.resolve()))!=sha or not path.is_file() or path.is_symlink()
                   or file_hash(path)!=sha for path,sha in ((archive,old),(canonical,current)))):
        raise ValueError("Administrative source compatibility must bind unchanged historical bytes and actual current builder separately")
    return expected


def verify_run(output,*,root=ROOT):
    """Verify a complete study and return only its 35 or 37 new result identities."""
    output,root=Path(output).resolve(),Path(root).resolve()
    _relative(output,root/"artificial")
    if output==root/"artificial":raise ValueError("Expected an isolated new run")
    manifest=read_signed(output/"run_manifest.json");context=manifest.get("context",{});ctx=digest(context)
    if (manifest.get("version")!=state.VERSION or context.get("version")!=state.VERSION
            or context.get("grid")!=verify.independent_grid() or context.get("settings")!=state.SETTINGS
            or context.get("selection_rule")!=state.SELECTION_RULE or context.get("ks")!=state.KS
            or context.get("cnn_training") is not False or context.get("cnn_forward_passes")!=0
            or context.get("test_parameter_search") is not False or context.get("posthoc_research") is not True
            or context.get("seed")!=2026 or context.get("region_count")!="unchanged Top-4"
            or context.get("matching")!={"lambda_pos":0.,"lambda_scale":0.,"shape_threshold":0.,"score_threshold":0.,"weight":"sqrt_area"}
            or read_json(output/"settings.json")!=state.SETTINGS):
        raise ValueError("Registration requires the complete frozen regional-WBBS design")
    required={"run_manifest.json","settings.json","status.json","selection.json","valid_completed.json","test_context.json",
        "source_protection_before.json","source_protection_after.json","historical_test_controls.json","global_wbbs_copy_valid.json"}
    for split in ("valid","test"):
        required.update({f"inputs_{split}.json",f"formula_audit_{split}.json",f"baseline_replay_{split}.json",f"source_compatibility_{split}.json"})
    required.add("baseline_replay_cross_test.json")
    for stage in ("valid","cross_valid","test","cross_test"):
        required.update({f"evaluation_index_{stage}.json",f"protocol_audit_{stage}.json",f"comparison_{stage}.csv"})
    root_files=_complete(output,ctx,required)["files"]
    status=read_json(output/"status.json");protection=read_json(output/"source_protection_after.json")
    if (status.get("status")!="completed" or protection.get("status")!="unchanged"
            or protection.get("files")!=len(read_json(output/"source_protection_before.json"))):
        raise ValueError("Study is incomplete or historical source protection failed")
    started=_stamp(manifest.get("started_at"),"root start");completed=_stamp(status.get("completed_at"),"root completion")
    if started>completed or manifest.get("run_id")!=started.strftime("%Y%m%dT%H%M%S%z")+"__f_region_wbbs":
        raise ValueError("Root run identity or time order differs")
    from .run import code_identity
    code=context.get("code")
    if not isinstance(code,dict) or not code or code!=code_identity():
        raise ValueError("Full scientific source dependency set changed")
    for name,sha in code.items():
        path=(root/name).resolve()
        if (Path(name).is_absolute() or not (path.is_relative_to(root) or path.is_relative_to(root.parent/"compared_methods"))
                or not path.is_file() or file_hash(path)!=sha):
            raise ValueError("Scientific source code changed: "+name)
    bindings={}
    for split in ("valid","test"):
        inputs=read_json(output/f"inputs_{split}.json");binding=inputs.get("source_binding");_bound_files(binding)
        if (inputs.get("context_signature")!=ctx or (split=="valid" and binding!=context.get("source_binding"))
                or (split=="test" and inputs.get("selection_sha256")!=file_hash(output/"selection.json"))):
            raise ValueError("Stage inputs differ from the frozen context/valid selection")
        _compatibility(output,root,split,ctx,binding,context)
        bindings[split]=binding
    indexes={stage:read_json(output/f"evaluation_index_{stage}.json") for stage in ("valid","cross_valid","test","cross_test")}
    chosen=read_signed(output/"selection.json");expected=selection.select(indexes["cross_valid"].get("results",{}))
    vm=read_json(output/"valid_completed.json")
    if (chosen.get("context_signature")!=ctx or any(chosen.get(k)!=v for k,v in expected.items())
            or vm.get("status")!="completed" or vm.get("context_signature")!=ctx
            or vm.get("selection_sha256")!=file_hash(output/"selection.json")
            or set(vm.get("indexes",{}))!={"valid","cross_valid"}
            or set(vm.get("evaluations",{}))!={"valid","cross_valid"}
            or status.get("winner")!=chosen["winner"] or status.get("best_regional_wbbs")!=chosen["best_regional_wbbs"]):
        raise ValueError("Primary/control selection differs from the complete cross-valid rule")
    frozen=_stamp(chosen.get("frozen_at"),"selection freeze")
    if not started<=frozen<=completed:raise ValueError("Selection freeze lies outside the study interval")
    tc=read_signed(output/"test_context.json")
    if (tc.get("context_signature")!=ctx or tc.get("selection_sha256")!=file_hash(output/"selection.json")
            or tc.get("source_binding_sha256")!=digest(bindings["test"]) or tc.get("specs")!=chosen["selected_specs"]):
        raise ValueError("Test context differs from the previously frozen valid selection")
    for stage in ("valid","cross_valid"):
        sha=file_hash(output/f"evaluation_index_{stage}.json")
        if (chosen.get(stage+"_index_sha256")!=sha or vm["indexes"][stage]!=sha
                or set(vm["evaluations"][stage])!=set(state.evaluation_specs(stage))
                or any(vm["evaluations"][stage][n]!=file_hash(output/stage/n/"completed.json") for n in state.evaluation_specs(stage))):
            raise ValueError("Frozen valid stage is incomplete or its child/index seal changed")
    examples=set()
    for stage in indexes:
        names=([state.BASELINE,*chosen["selected_variants"]] if stage.endswith("valid") else chosen["selected_variants"])
        examples.update(f"explanations/{stage}/{name}/examples.json" for name in names)
    if not examples<=set(root_files):raise ValueError("Root completion omits required regional explanations")
    audits={stage:_protocol_audit(output,stage) for stage in indexes}
    for split in ("valid","test"):
        specs=state.grid() if split=="valid" else {state.BASELINE:state.grid()[state.BASELINE],**chosen["selected_specs"]}
        _audits(output,split,specs,ctx,bindings[split],root_files)
        cachemeta=read_json(output/"cache"/split/"metadata.json")
        cache_start=_stamp(cachemeta.get("started_at"),"point-cache start")
        cache_end=_stamp(cachemeta.get("finished_at"),"point-cache completion")
        if (not started<=cache_start<=cache_end<=completed or (split=="valid" and cache_end>frozen)
                or (split=="test" and cache_start<frozen)):
            raise ValueError("Point-cache preparation violates the frozen valid/test time order")
        expected_scores=set(specs)|({state.GLOBAL} if split=="valid" else set())
        if {p.name for p in (output/"score_cache"/split).iterdir() if p.is_dir()}!=expected_scores:
            raise ValueError("Full score caches include missing or unselected recipes")
    replays={stage:_baseline(output,stage,ctx,bindings["valid" if stage=="valid" else "test"],root_files)
             for stage in ("valid","test","cross_test")}
    controls=read_json(output/"historical_test_controls.json")
    from .run import CROSS_J,GLOBAL_TEST
    if (controls.get("regional_j")!=str(Path(replays["test"]["source"]).parent)
            or controls.get("regional_j_cross")!=str(CROSS_J) or controls.get("global_wbbs_cross")!=str(GLOBAL_TEST)
            or Path(replays["cross_test"]["source"]).parent!=CROSS_J
            or controls.get("regional_j_replays")!=["baseline_replay_test.json","baseline_replay_cross_test.json"]
            or any(str(p/"completed.json") not in bindings["test"] for p in (CROSS_J,GLOBAL_TEST))):
        raise ValueError("Historical held-out comparison controls differ from the source-bound reports")
    gs=read_json(output/"score_cache/valid/global_wbbs/metadata.json")["source_receipt"]
    paths={"source_completed_sha256":Path(gs["source_root"])/"completed.json",
        "source_cache_completed_sha256":Path(gs["source_cache"])/"completed/completed.json",
        "source_scores_sha256":Path(gs["source_cache"])/"completed/WBBS.npy",
        "source_query_ids_sha256":Path(gs["source_cache"])/"completed/query_ids.npy",
        "source_gallery_ids_sha256":Path(gs["source_cache"])/"completed/gallery_ids.npy"}
    if any(bindings["valid"].get(str(p))!=gs.get(field) for field,p in paths.items()):
        raise ValueError("Global WBBS benchmark source receipt is detached from frozen source bytes")
    items=[]
    for stage,index in indexes.items():
        specs=state.evaluation_specs(stage,chosen);split="valid" if stage.endswith("valid") else "test"
        if (index.get("context_signature")!=ctx or index.get("stage")!=stage or index.get("split")!=split
                or index.get("specs")!=specs or set(index.get("results",{}))!=set(specs)
                or {p.name for p in (output/stage).iterdir() if p.is_dir()}!=set(specs)):
            raise ValueError("Evaluation stage is incomplete or contains unselected test variants")
        for name,spec in specs.items():
            stamp=_stamp(read_json(output/stage/name/"run_identity.json").get("started_at"),"evaluation start")
            if not started<=stamp<=completed or (split=="valid" and stamp>frozen) or (split=="test" and stamp<frozen):
                raise ValueError("Evaluation start violates the frozen valid/test time order")
            items.append(_evaluation(output,root,stage,name,spec,ctx,bindings[split],chosen,index,root_files,audits[stage]))
    expected_count=27+2*len(chosen["selected_variants"])
    if expected_count not in (35,37) or len(items)!=expected_count or len({i["key"] for i in items})!=expected_count:
        raise ValueError("Expected 27 valid and 8 or 10 frozen-test result identities")
    return {"schema_version":1,"runs":items}


def proposal(output,*,root=ROOT):
    return verify_run(output,root=root)


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
