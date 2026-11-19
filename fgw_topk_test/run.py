#!/usr/bin/env python3
"""Evaluate exactly ten recipes frozen by a completed valid search, without tuning test."""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import argparse
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fgw_grid_search import scoring
from fgw_grid_search.state import verify_run as verify_parent
from fgw_topk_test.state import (PARENT_FILES, TOP_K, TZ, VERSION, checked_files, choose_output,
    evaluation_paths, exclusive, freeze_top10, read_signed, record, separate, signed,
    variant_output, verify_run)
from wbm_retrieval.common import digest, file_hash, read_json, write_json

_WORK = None


def _evaluate_one(item):
    from fgw_grid_search import pipeline
    name, spec = item
    output, signature, provider, pool, bundle, settings = _WORK
    return pipeline.evaluate(variant_output(output, name), signature, provider, pool, bundle,
                             {name: spec}, settings, workers=1, stage="test")


def evaluate_fixed(output, signature, provider, pool, bundle, specs, settings, *, workers=1):
    """Use one shared cache and ten isolated singleton calls to the frozen pipeline.

    Order is supplied by the caller's frozen valid prefix, never by test scores.
    The formal runner/verify_run enforce 7,199 queries; synthetic pools are useful
    here for checking the real ranking and verification path without real data.
    """
    from fgw_grid_search import pipeline
    global _WORK
    settings = scoring.validate_settings(settings)
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("workers must be an integer in [1,8]")
    grid = scoring.specs(settings)
    if (not isinstance(specs, dict) or len(specs) != TOP_K
            or any(name not in grid or scoring.validate_spec(spec) != grid[name] for name, spec in specs.items())):
        raise ValueError("Exactly ten unchanged valid-grid recipes are required")
    if not isinstance(signature, str) or not signature:
        raise ValueError("A signed parent input identity is required")
    if pool.split_name != "test" or provider.split != "test":
        raise ValueError("Fixed top-ten evaluation requires a test pool and provider")
    if workers > 1 and "fork" not in multiprocessing.get_all_start_methods():
        raise ValueError("Parallel fixed evaluation requires the fork context")
    output = Path(output).resolve()
    names = list(specs)
    paths = evaluation_paths(output, names)
    children_dir = output / "evaluations"
    if children_dir.exists() and not {p.name for p in children_dir.iterdir()} <= {Path(p).name for p in paths.values()}:
        raise ValueError("Unexpected child output in fixed top-ten test")
    binding = pipeline.source_binding(provider)
    request = {"signature": signature, "stage": "test", "split": "test",
               "variants": specs, "ranked_variants": names, "evaluation_paths": paths,
               "settings": settings, "provider_source": binding,
               "retrieval_protocol_signature": pool.protocol_signature}
    pipeline.record(output / "evaluation_request_test.json", request)
    _WORK = (output, signature, provider, pool, bundle, settings)
    try:
        if workers == 1:
            children = [_evaluate_one(item) for item in specs.items()]
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as executor:
                children = list(executor.map(_evaluate_one, specs.items()))
    finally:
        _WORK = None
    if pipeline.source_binding(provider) != binding:
        raise ValueError("Full-score source changed during fixed evaluation")
    index = {**request, "children": {}, "variant_signatures": {}, "completed_sha256": {}, "summary_rows": []}
    for name, child in zip(names, children):
        child_index = variant_output(output, name) / "evaluation_index_test.json"
        if (child.get("variants") != {name: specs[name]} or len(child.get("summary_rows", [])) != 1
                or child["summary_rows"][0]["variant"] != name or child.get("signature") != signature
                or child.get("provider_source") != binding):
            raise ValueError("Singleton test child returned a different recipe/source")
        index["children"][name] = {"path": paths[name], "index": str(child_index.relative_to(output)),
                                   "sha256": file_hash(child_index)}
        index["variant_signatures"][name] = child["variant_signatures"][name]
        index["completed_sha256"][name] = child["completed_sha256"][name]
        index["summary_rows"].append(child["summary_rows"][0])
    pipeline.record(output / "evaluation_index_test.json", index)
    return index


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--search-root", help="Completed valid grid; inferred from original manifest when resuming")
    out = p.add_mutually_exclusive_group()
    out.add_argument("--output-root")
    out.add_argument("--resume-root")
    p.add_argument("--workers", type=int, default=8, help="Independent recipe workers, 1..8")
    p.add_argument("--dry-run", action="store_true", help="Read-only frozen plan; no cache or evaluation")
    p.add_argument("--check-inputs", action="store_true", help="Read-only source verification; no score computation")
    return p


def code_identity():
    return {str(p.relative_to(ROOT)): file_hash(p) for p in
            sorted(Path(__file__).parent.glob("*.py")) + sorted(Path(__file__).parent.glob("*.json"))}


def prepare_plan(args, *, now=None):
    if type(args.workers) is not int or not 1 <= args.workers <= 8:
        raise ValueError("Use 1..8 bounded recipe workers")
    output = choose_output(args.output_root, args.resume_root, now=now)
    previous = read_signed(output / "run_manifest.json") if args.resume_root else None
    if previous and (previous.get("version") != VERSION or previous.get("stage") != "test"):
        raise ValueError("Resume requires this exact top-ten test version")
    source_name = args.search_root or (previous["context"]["selection_source"]["root"] if previous else None)
    if not source_name:
        raise ValueError("A new fixed test requires an explicit completed --search-root")
    parent = (ROOT / Path(source_name).expanduser()).resolve()
    if (ROOT / "artificial").resolve() not in parent.parents:
        raise ValueError("The frozen valid search must belong to this repository")
    parent_manifest, original = verify_parent(parent, required_stage="search")
    frozen = freeze_top10(original)
    origin = {"root": str(parent), "files": {name: file_hash(parent / name) for name in PARENT_FILES}}
    settings = scoring.validate_settings(parent_manifest["context"]["settings"])
    source = Path(parent_manifest["context"]["source_run"]).resolve()
    separate(output, [source, parent, ROOT / "artifacts", ROOT / "protocol", ROOT / "results_catalog"])
    plan = {"version": VERSION, "stage": "test", "split": "test", "top_k": TOP_K,
            "source_run": str(source), "settings": settings, "specs": frozen["specs"],
            "ranked_variants": frozen["ranked_variants"],
            "evaluation_paths": evaluation_paths(output, frozen["ranked_variants"]),
            "selection_source": origin, "frozen_selection_signature": frozen["signature"],
            "training_required": False, "posthoc_research": True, "test_parameter_search": False,
            "primary_variant": frozen["primary_variant"],
            "selection_rule": "fixed first ten of original valid ranking; no test winner selection"}
    if previous and any(previous["context"].get(k) != v for k, v in plan.items()):
        raise ValueError("Resume frozen recipe list/source/settings changed")
    return output, plan, frozen, previous, parent_manifest


def _inventory(plan, provenance):
    from fgw_grid_search import sources
    before = sources.inventory(Path(plan["source_run"]), provenance)
    parent = Path(plan["selection_source"]["root"])
    for name, sha in plan["selection_source"]["files"].items():
        p = parent / name
        if file_hash(p) != sha:
            raise ValueError("Original valid selection changed")
        before[str(p)] = {"sha256": sha, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
    return before


def main(argv=None):
    args = parser().parse_args(argv)
    started = datetime.now(TZ).replace(microsecond=0)
    output, plan, frozen, previous, parent = prepare_plan(args, now=started)
    if args.dry_run:
        print(json.dumps({**plan, "output": str(output), "workers": args.workers}, ensure_ascii=False, indent=2))
        return 0
    import numpy as np
    import scipy
    import torch
    from fgw_grid_search import cache, sources
    from fgw_grid_search.run import code_identity as original_code_identity
    torch.set_num_threads(1)
    code, grid_code = code_identity(), original_code_identity()
    environment = {"numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__, "python": sys.version}
    if grid_code != parent["context"]["code"] or environment != parent["context"]["environment"]:
        raise ValueError("Frozen grid scientific code/environment changed after valid selection")
    bundle, pool, scorer, wbbs, provenance = sources.load_inputs(Path(plan["source_run"]), "test")
    if (len(pool), len(pool.gallery), pool.candidates_per_query) != (7199, 7199, 7198):
        raise ValueError("Formal test requires the complete 7,199-query frozen protocol")
    context = {**plan, "code": code, "grid_code": grid_code, "environment": environment, "provenance": provenance}
    if previous and previous["context"] != context:
        raise ValueError("Resume code/source/environment identity changed")
    if args.check_inputs:
        print("Frozen top ten and complete test sources checked; no scores or output created.", flush=True)
        return 0
    if previous and (output / "completed.json").exists():
        verify_run(output)
        from fgw_topk_test.register import write_proposal
        write_proposal(output)
        print("Verified completed fixed test:", output, flush=True)
        return 0
    with exclusive(output):
        if previous:
            manifest, before = previous, read_json(output / "source_protection_before.json")
        else:
            manifest = signed({"version": VERSION, "stage": "test", "context": context,
                "batch_id": started.strftime("%Y%m%dT%H%M%S%z") + "__fgw_topk_test", "started_at": started.isoformat(),
                "output": str(output), "workers": args.workers,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "launch_command": [str(x) for x in (sys.argv if argv is None else [__file__, *argv])]})
            before = _inventory(plan, provenance)
            record(output / "source_protection_before.json", before)
            record(output / "run_manifest.json", manifest)
        record(output / "settings.json", plan["settings"])
        record(output / "frozen_selection.json", frozen)
        inp = signed({"parent_signature": manifest["signature"], "split": "test",
                      "context_digest": digest(context), "provenance": provenance})
        record(output / "inputs.json", inp)
        try:
            write_json(output / "status.json", {"status": "running", "phase": "source_audit", "split": "test"})
            audit_path = output / "source_audit.json"
            if audit_path.exists():
                audit = read_signed(audit_path)
                if audit.get("context_digest") != digest(context) or audit.get("status") != "verified":
                    raise ValueError("Source audit is stale")
            else:
                evidence = sources.audit_components(pool, bundle, scorer, wbbs)
                if evidence.get("status") != "verified":
                    raise ValueError("Source component audit failed")
                record(audit_path, signed({"status": "verified", "context_digest": digest(context), "evidence": evidence}))
            write_json(output / "status.json", {"status": "running", "phase": "full_candidate_components", "split": "test"})
            provider, _ = cache.build_cache(output / "component_cache/test",
                digest({"inputs": inp["signature"], "operation": "full_F_G_W_scores"}), pool, scorer, wbbs)
            write_json(output / "status.json", {"status": "running", "phase": "fixed_test_evaluation", "split": "test"})
            evaluate_fixed(output, inp["signature"], provider, pool, bundle, plan["specs"], plan["settings"], workers=args.workers)
            checked_files(ROOT, code)
            checked_files(ROOT, grid_code)
            after = _inventory(plan, provenance)
            if before != after:
                raise ValueError("Protected old sources changed")
            record(output / "source_protection_after.json", {"status": "unchanged", "files": len(after), "inventory_digest": digest(after)})
            write_json(output / "status.json", {"status": "completed", "stage": "test", "test_evaluated": True,
                                                "new_evaluations": TOP_K})
            files = {}
            for p in sorted(output.rglob("*")):
                rel = p.relative_to(output)
                if (not p.is_file() or p == output / "completed.json" or "rows" in rel.parts
                        or any(part.startswith(".") for part in rel.parts)
                        or p.suffix in (".npy", ".npz", ".log") or p.name == "rankings.csv"):
                    continue
                files[str(rel)] = file_hash(p)
            write_json(output / "completed.json", {"status": "completed", "version": VERSION,
                "signature": manifest["signature"], "stage": "test", "batch_id": manifest["batch_id"],
                "new_evaluations": TOP_K, "files": files})
            verify_run(output)
            from fgw_topk_test.register import write_proposal
            write_proposal(output)
            print("Completed and verified fixed valid top ten:", output, flush=True)
        except BaseException as exc:
            if not (output / "completed.json").exists():
                write_json(output / "status.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
