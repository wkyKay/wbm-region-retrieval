#!/usr/bin/env python3
"""Frozen B1 ablation runner. All source artifacts remain read-only."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from zoneinfo import ZoneInfo
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from b1_structure_fusion import run as oldrun, register as oldregister
from b1_structure_fusion.scoring import StructuralDistances
from experiments.proposed.f_fusion_ablation.scoring import Scores
from experiments.proposed.f_global_fusion import pipeline as fusion
from experiments.proposed.f_global_fusion.run import separate_output
from experiments.proposed.f_segmentation_geometry import pipeline as geo
from experiments.proposed._shared.ef_suite import exclusive
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from wbm_retrieval.data import Protocol, source_fingerprint
from wbm_retrieval.diagnostics import diagnostic_log
from b1_ablation import scoring, pipeline
TZ = ZoneInfo("Asia/Shanghai")
VERSION = "b1_ablation_v1"
DEFAULT_SOURCE = "artificial/b1_structure_all_20261117T193244+0800"
DEFAULT_REFERENCE_ABLATION = "artificial/b1_ablation_20261118T150805+0800"


def path(value):
    return (ROOT / Path(value).expanduser()).resolve()


def signed(value):
    return {**value, "signature": digest(value)}


def identity():
    result = oldrun.code_identity()
    for item in sorted(Path(__file__).parent.glob("*.py")) + sorted(Path(__file__).parent.glob("config*.json")):
        result[str(item.relative_to(ROOT))] = file_hash(item)
    return result


def inventory(source, reference=None, protected_roots=()):
    files = sorted(p for p in source.rglob("*") if p.is_file())
    if reference is not None:
        files += sorted(p for p in reference.rglob("*") if p.is_file())
    for protected in protected_roots:
        files += sorted(p for p in protected.rglob("*") if p.is_file())
    files += sorted(p for p in (ROOT / "b1_structure_fusion").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return {str(p): {"sha256": file_hash(p), "size": p.stat().st_size,
                    "mtime_ns": p.stat().st_mtime_ns} for p in files}


def choose_output(args):
    if args.output_root and args.resume_root:
        raise ValueError("Choose either new output or resume")
    target = path(args.resume_root or args.output_root or
                  ("artificial/b1_ablation_" + datetime.now(TZ).strftime("%Y%m%dT%H%M%S%z")))
    allowed = (ROOT / "artificial").resolve()
    if allowed not in target.parents or not target.name.startswith("b1_ablation_"):
        raise ValueError("Use a new artificial/b1_ablation_<actual-time> directory")
    for parent in target.parents:
        if parent == allowed:
            break
        if (parent / "run_manifest.json").exists():
            raise ValueError("Cannot nest output inside another experiment")
    if args.resume_root:
        if not (target / "run_manifest.json").is_file():
            raise ValueError("Resume requires an existing manifest")
    elif target.exists() and any(p.name != "launch.log" for p in target.iterdir()):
        raise ValueError("Existing output: explicit resume or new directory required")
    return target


def load_inputs(source, device):
    print("Verifying the completed original B1 and all protected hashes", flush=True)
    oldregister.proposal(source)
    manifest = fusion.signed_manifest(source / "run_manifest.json")
    context = manifest["context"]
    if context["code"] != oldrun.code_identity() or device != context["device"]:
        raise ValueError("Original B1 code/device changed; no silent source substitution")
    if read_json(source / "selection.json")["selections"]["J"]["selected"]["variant"] != "J_2":
        raise ValueError("This fixed ablation requires the frozen J_2 source")
    defaults = oldrun.parser().parse_args([])
    config, suite, checkpoint, kind, training_inputs = geo.load_source(path(defaults.source_root), "cc8", None, "cpu")
    del checkpoint
    oldrun.assert_frozen(config, suite, kind, context["settings"])
    protocol = Protocol(config)
    sources = source_fingerprint(config)
    geometry = geo.new_identity(config, training_inputs)
    if sources != context["sources"] or geometry != context["geometry"]:
        raise ValueError("Data, protocol or F training source differs from original B1")
    with np.load(config["data_file"], allow_pickle=False) as z:
        maps, labels = z["arr_0"], z["arr_1"]
    bundle = fusion.bundle_for(protocol, labels)
    bundle.maps = maps
    contexts = {}
    for split in ("valid", "test"):
        pool = getattr(protocol.retrieval, split)
        local = oldrun.frozen_local_cache(defaults, split, digest(geometry))
        global_root = path(getattr(defaults, "wapirl_" + split + "_root"))
        gids, gf, global_id = fusion.load_global(global_root, split, pool, sources,
                    context["global"]["checkpoint"], context["global"]["inference_recipe"])
        source_inputs = fusion.signed_manifest(source / f"inputs_{split}.json")
        if file_hash(local / "completed.json") != source_inputs["f_features"] or global_id != source_inputs["global_features"]:
            raise ValueError("Frozen features differ from source B1")
        desc_dir = source / "descriptors" / split
        signature = digest({"sources": sources, "split": split, "settings": context["settings"], "code": context["code"]})
        if not geo.completed(desc_dir, signature):
            raise ValueError("Missing original descriptor cache")
        with np.load(desc_dir / (split + ".npz"), allow_pickle=False) as z:
            descriptors = dict(z)
        if not np.array_equal(descriptors["sample_ids"], pool.gallery):
            raise ValueError("Original descriptor IDs differ from frozen gallery")
        scorer = Scores(local, gids, gf, pool.gallery, config["matching"], descriptors)
        reference, ref_id = oldrun.checked_reference(split, sources)
        provenance = {"source_inputs_sha256": file_hash(source / f"inputs_{split}.json"),
                      "source_evaluation_index_sha256": file_hash(source / f"evaluation_index_{split}.json"),
                      "descriptor_completed_sha256": file_hash(desc_dir / "completed.json"),
                      "f_features": file_hash(local / "completed.json"), "global_features": global_id,
                      "reference": ref_id, "protocol_signature": pool.protocol_signature}
        contexts[split] = (pool, scorer, descriptors, reference, provenance)
        print(f"Verified {split}: {len(pool)} queries, {pool.candidates_per_query} candidates each", flush=True)
    return bundle, contexts


def analysis_sources(output, source, split, provenance, settings=None, reference=None):
    index = read_json(source / f"evaluation_index_{split}.json")
    selected = ({"B0", "J_2", "J_only"} if reference is not None else
                {"B0", "R_0.25", "T_0.25", "I_1", "J_2", "J_only", "I_2"})
    result = {name: {"exact_dir": source / split / name} for name in index["variants"] if name in selected}
    result.update({name: {"exact_dir": output / split / name} for name in scoring.specs(split, settings)})
    if reference is not None:
        for name in ("no_global", "no_angle", "no_local"):
            result[name] = {"exact_dir": reference / split / name}
    reports = provenance["reference"]["reports"]
    if "coverage" in reports:
        exact = ROOT.parent / "compared_methods" / reports["exact"]["path"]
        coverage = ROOT.parent / "compared_methods" / reports["coverage"]["path"]
        result["WBBS"] = {"exact_dir": exact.parent, "coverage_dir": coverage.parent,
                          "ranking_path": exact.parent / "rankings.csv"}
    return result


def benchmark(scorer, descriptors, pool, device, ks, settings=None):
    from wbm_retrieval.exact_metrics import order_scores
    # The audit records per-class IDs; caller supplies them using all pool rows below.
    ids = sorted(map(int, pool))[:10]
    distances = StructuralDistances(descriptors["sample_ids"], descriptors["joint128"], descriptors["empty"], device)
    timings = []
    for q in ids:
        c = pool[q]
        start = time.perf_counter(); values = scorer.components(q, c); component_s = time.perf_counter()-start
        start = time.perf_counter(); d = distances.for_query(q, c); distance_s = time.perf_counter()-start
        variants = {}
        for name, spec in scoring.specs(pool.split_name, settings).items():
            start = time.perf_counter(); score = scoring.score_variant(values, d, spec)
            order_scores(c, score, q, 2026)[:max(ks)]
            variants[name] = time.perf_counter()-start
        timings.append({"query_id": q, "shared_components_seconds": component_s,
                        "shared_structural_distances_seconds": distance_s,
                        "formula_and_sort_seconds": variants})
    return {"queries": timings, "sample_rule": "first ten original query IDs", "cache": "warm frozen features",
            "scope": "shared F/G/angle components, shared all structural families, per-formula sorting; excludes encoding, TV and IO",
            "warning": "Not independent end-to-end variant latency, and not a WBBS speed comparison"}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", default=DEFAULT_SOURCE)
    p.add_argument("--recipe", choices=("components", "no_global_angle", "no_local_angle"), default="components")
    p.add_argument("--reference-ablation-root", help="Completed component ablation used read-only for joint-deletion comparisons")
    p.add_argument("--protected-root", action="append", default=[],
                   help="Additional completed run to protect read-only; does not add an analysis comparison")
    out = p.add_mutually_exclusive_group()
    out.add_argument("--output-root"); out.add_argument("--resume-root")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--check-inputs", action="store_true")
    p.add_argument("--audit-only", action="store_true", help="Persist real-source independent audits without new rankings")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    config = Path(__file__).with_name(f"config_{args.recipe}.json") if args.recipe != "components" else None
    settings = scoring.load_settings(config)
    reference = (path(args.reference_ablation_root or DEFAULT_REFERENCE_ABLATION)
                 if args.recipe != "components" else None)
    if args.recipe == "components" and args.reference_ablation_root:
        raise ValueError("Reference ablation is only used by the joint-deletion recipe")
    source, output = path(args.source_root), choose_output(args)
    protected = sorted({path(p) for p in args.protected_root} - {source, reference}, key=str)
    separate_output(output, [source, ROOT / "artifacts", ROOT / "protocol", ROOT / "results_catalog", *([reference] if reference else []), *protected])
    plan = {"version": VERSION, "source_root": str(source), "settings": settings,
            "valid": scoring.specs("valid", settings), "test": scoring.specs("test", settings),
            "reuse": {"valid/I_2": str(source / "valid/I_2"), "baseline": str(source)},
            "parameter_search": False, "posthoc_design": True, "training_required": False,
            "audit_queries": "minimum original ID within each frozen complete label combination in each split",
            "bootstrap": {"replicates": 10000, "seed": 2026, "strata": "37 frozen multi-hot combinations"},
            "mode": "audit_only" if args.audit_only else "all", "device": args.device}
    if reference is not None:
        plan["reference_ablation_root"] = str(reference)
        plan["reuse"]["component_controls"] = ["no_global", "no_angle", "no_local"]
    if protected:
        plan["protected_roots"] = [str(p) for p in protected]
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2)); return 0
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Original PyTorch CUDA environment required")
    protected_identities = {}
    for protected_root in protected:
        if (ROOT / "artificial") not in protected_root.parents:
            raise ValueError("Extra protection requires an isolated artificial/ run")
        if read_json(protected_root / "status.json").get("status") != "completed":
            raise ValueError("Additional protected run must already be completed")
        fusion.signed_manifest(protected_root / "run_manifest.json")
        protected_identities[str(protected_root)] = file_hash(protected_root / "run_manifest.json")
    bundle, contexts = load_inputs(source, args.device)
    reference_identity = None
    if reference is not None:
        from b1_ablation.register import proposal
        previous = fusion.signed_manifest(reference / "run_manifest.json")
        if (Path(previous["context"]["plan"]["source_root"]).resolve() != source or
            previous["context"]["plan"]["settings"] != scoring.DEFAULT_SETTINGS):
            raise ValueError("Reference must be the frozen component ablation of the same B1 source")
        proposal(reference)
        reference_identity = {"manifest_sha256": file_hash(reference / "run_manifest.json"),
                              "indexes": {split: file_hash(reference / f"evaluation_index_{split}.json")
                                          for split in ("valid", "test")}}
        print("Verified read-only component controls:", reference, flush=True)
    if args.check_inputs:
        print("Inputs verified; no output or evaluation created."); return 0
    code = identity()
    context = {"plan": plan, "code": code, "source_manifest_sha256": file_hash(source / "run_manifest.json"),
               "source_selection_sha256": file_hash(source / "selection.json"),
               "inputs": {s: v[4] for s, v in contexts.items()}}
    if reference_identity is not None:
        context["reference_ablation"] = reference_identity
    if protected_identities:
        context["protected_run_manifests"] = protected_identities
    def protected_inventory():
        return inventory(source, reference, protected) if protected else inventory(source, reference)
    with exclusive(output):
        manifest_file = output / "run_manifest.json"
        if manifest_file.exists():
            manifest = fusion.signed_manifest(manifest_file)
            if manifest["context"] != context:
                raise ValueError("Resume scientific settings or source changed")
            before = read_json(output / "source_protection_before.json")
        else:
            stamp = datetime.now(TZ)
            manifest = signed({"version": VERSION, "context": context, "started_at": stamp.isoformat(timespec="seconds"),
                               "batch_id": stamp.strftime("%Y%m%dT%H%M%S%z") + "__b1_ablation",
                               "output": str(output), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
            write_json(manifest_file, manifest); write_json(output / "plan_before_evaluation.json", plan)
            before = protected_inventory(); write_json(output / "source_protection_before.json", before)
        with diagnostic_log(output):
            try:
                from b1_ablation.verify import audit_source_sample, verify_evaluation
                from b1_ablation.analysis import postprocess
                for split, (pool, scorer, descriptors, wbbs_reference, provenance) in contexts.items():
                    write_json(output / "status.json", {"status": "running", "phase": "source_audit", "split": split})
                    audit_file = output / f"source_audit_{split}.json"
                    if not audit_file.exists():
                        audit = audit_source_sample(source, pool, bundle, scorer, descriptors, args.device)
                        write_json(audit_file, signed({**audit, "input_context_signature": digest(context)}))
                    audit = fusion.signed_manifest(audit_file)
                    if audit.get("input_context_signature") != digest(context):
                        raise ValueError("Source audit belongs to different inputs")
                    if audit.get("status") != "verified":
                        raise ValueError("Independent source audit failed: " + split + "/" + str(audit.get("status")))
                    if args.audit_only:
                        continue
                    inputs = signed({"parent_signature": manifest["signature"], "split": split, "provenance": provenance,
                                     "specs": scoring.specs(split, settings), "code": code})
                    oldrun.pipeline.record(output / f"inputs_{split}.json", inputs)
                    pipeline.evaluate(output, inputs["signature"], scorer, pool, bundle, settings["ks"], 2026,
                                      descriptors, settings, device=args.device, reference=wbbs_reference)
                    verification = {}
                    evaluation_index = read_json(output / f"evaluation_index_{split}.json")
                    for name, spec in scoring.specs(split, settings).items():
                        variant_signature = evaluation_index["variant_signatures"][name]
                        verification[name] = verify_evaluation(output / split / name, pool, bundle, settings["ks"], variant_signature, 2026)
                    write_json(output / f"verification_{split}.json", verification)
                    postprocess(output / "analysis" / split,
                                analysis_sources(output, source, split, provenance, settings, reference), bundle.labels,
                                list(map(int, pool)), list(map(int, pool.gallery)), split=split,
                                ks=settings["ks"], bootstrap_replicates=10000, seed=2026,
                                anchor=args.recipe if reference is not None else "J_2")
                    write_json(output / f"benchmark_{split}.json", benchmark(scorer, descriptors, pool, args.device, settings["ks"], settings))
                fusion.checked_files(ROOT, code)
                after = protected_inventory()
                if before != after:
                    raise ValueError("Protected B1 or reference ablation files changed during evaluation")
                write_json(output / "source_protection_after.json", {"status": "unchanged", "files": len(after), "inventory_digest": digest(after)})
                write_json(output / "status.json", {"status": "audited" if args.audit_only else "completed",
                           "mode": plan["mode"], "output": str(output), "test_evaluated": not args.audit_only})
                if not args.audit_only:
                    from b1_ablation.register import write_proposal
                    write_proposal(output)
                print("Completed:", output, flush=True)
            except BaseException as exc:
                write_json(output / "status.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
                raise
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
