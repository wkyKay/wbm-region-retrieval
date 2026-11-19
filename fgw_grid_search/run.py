#!/usr/bin/env python3
"""Valid-only grid search; an explicit separate test consumes one frozen winner."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wbm_retrieval.common import digest, file_hash, read_json, write_json
from fgw_grid_search import scoring
from fgw_grid_search.state import (DEFAULT_SOURCE, TZ, VERSION, checked_files, choose_output,
    exclusive, read_signed, record, resolve, select_payload, separate, signed, verify_run)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("search", "test"), default=None,
                   help="Default search runs valid only; test requires a completed --search-root")
    p.add_argument("--config", type=Path, help="Search grid JSON; never accepted for test")
    p.add_argument("--source-run", help="Completed frozen FGW source; sources remain read only")
    p.add_argument("--search-root", help="Completed valid grid that supplies the single test winner")
    out = p.add_mutually_exclusive_group()
    out.add_argument("--output-root")
    out.add_argument("--resume-root")
    p.add_argument("--workers", type=int, default=4, help="Independent grid evaluation workers, 1..8")
    p.add_argument("--dry-run", action="store_true", help="Show plan; no output/cache/evaluation writes")
    p.add_argument("--check-inputs", action="store_true", help="Read-only source checks; no scoring/search")
    return p


def code_identity():
    from fgw_fusion.run import identity
    result = identity()
    for p in sorted(Path(__file__).parent.glob("*.py")) + sorted(Path(__file__).parent.glob("*.json")):
        result[str(p.relative_to(ROOT))] = file_hash(p)
    return result


def prepare_plan(args, *, now=None):
    if not 1 <= args.workers <= 8:
        raise ValueError("Use 1..8 bounded evaluation workers")
    now = now or datetime.now(TZ).replace(microsecond=0)
    output = choose_output(args.output_root, args.resume_root, now=now)
    previous = read_signed(output / "run_manifest.json") if args.resume_root else None
    if previous is not None and previous.get("version") != VERSION:
        raise ValueError("Resume is limited to this search version")
    stage = args.stage or (previous["stage"] if previous else "search")
    if previous and previous["stage"] != stage:
        raise ValueError("Cannot change stage in an existing output")
    origin, frozen = None, None
    if stage == "test":
        if args.config is not None:
            raise ValueError("Test cannot change the valid-selected search settings")
        parent_name = args.search_root or (previous["context"]["selection_source"]["root"] if previous else None)
        if not parent_name:
            raise ValueError("Test requires an explicit completed --search-root")
        parent = resolve(parent_name)
        if ROOT / "artificial" not in parent.parents:
            raise ValueError("Frozen search must belong to this repository's artificial directory")
        parent_manifest, frozen = verify_run(parent, required_stage="search")
        settings = scoring.validate_settings(parent_manifest["context"]["settings"])
        source = resolve(parent_manifest["context"]["source_run"])
        if args.source_run and resolve(args.source_run) != source:
            raise ValueError("Test must retain the valid search's frozen source")
        specs = {frozen["winner"]: frozen["selected_spec"]}
        origin = {"root": str(parent), "files": {name: file_hash(parent / name) for name in
                  ("run_manifest.json", "completed.json", "selection.json")}}
    else:
        if args.search_root:
            raise ValueError("--search-root is only used by the separate test stage")
        settings = (scoring.load_settings(args.config) if args.config is not None or previous is None
                    else scoring.validate_settings(previous["context"]["settings"]))
        source = resolve(args.source_run or (previous["context"]["source_run"] if previous else DEFAULT_SOURCE))
        specs = scoring.specs(settings)
    separate(output, [source, ROOT / "artifacts", ROOT / "protocol", ROOT / "results_catalog",
                      *([Path(origin["root"])] if origin else [])])
    plan = {"version": VERSION, "stage": stage, "split": "valid" if stage == "search" else "test",
            "source_run": str(source), "settings": settings, "specs": specs,
            "selection_source": origin, "training_required": False, "posthoc_design": True,
            "F_component": "frozen cc8 optimal regional matching; no angular A or J",
            "full_candidate_rule": "all_gallery_except_query", "parameter_search_split": "valid",
            "test_grid_search": False, "best_scope": "best among this finite frozen grid and included reference",
            "normalization": "per-query complete candidate score ranges only; no candidate labels"}
    if previous:
        for key, value in plan.items():
            if previous["context"].get(key) != value:
                raise ValueError("Resume plan/source/settings changed: " + key)
    return output, plan, frozen, previous


def write_grid_table(output, index):
    path = Path(output) / "grid_results.csv"
    rows = index["summary_rows"]
    fields = ["variant", "family", "split", "num_queries", "precision10_hits",
              "Precision@10", "mAP@100", "Coverage-nDCG@10", "F_weight", "G_weight", "WBBS_weight"]
    import io
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        spec = index["variants"][row["variant"]]
        writer.writerow({**{k: row[k] for k in fields[:8]},
                         **dict(zip(fields[8:], spec["weights"]))})
    text = handle.getvalue()
    if path.exists():
        if path.read_text() != text:
            raise ValueError("Completed grid summary changed")
    else:
        temporary = path.with_suffix(".csv.tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)


def _extra_selection_inventory(before, plan):
    if plan["selection_source"]:
        folder = Path(plan["selection_source"]["root"])
        for name, sha in plan["selection_source"]["files"].items():
            p = folder / name
            if file_hash(p) != sha:
                raise ValueError("Frozen valid selection changed")
            before[str(p)] = {"sha256": sha, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
    return before


def main(argv=None):
    args = parser().parse_args(argv)
    started = datetime.now(TZ).replace(microsecond=0)
    output, plan, frozen, previous = prepare_plan(args, now=started)
    if args.dry_run:
        print(json.dumps({**plan, "output": str(output), "evaluations": len(plan["specs"]),
                          "workers": args.workers}, ensure_ascii=False, indent=2))
        return 0
    # Imports below deliberately keep --help/--dry-run independent of model loading.
    import numpy as np
    import scipy
    import torch
    from fgw_grid_search import sources, cache, pipeline
    torch.set_num_threads(1)
    code = code_identity()
    if plan["selection_source"]:
        parent = read_signed(Path(plan["selection_source"]["root"]) / "run_manifest.json")
        if parent["context"]["code"] != code:
            raise ValueError("Search code changed after selection; freeze the original code or run a new search")
    bundle, pool, scorer, wbbs, provenance = sources.load_inputs(Path(plan["source_run"]), plan["split"])
    expected = (500, 3599) if plan["split"] == "valid" else (7199, 7199)
    if (len(pool), len(pool.gallery)) != expected or pool.candidates_per_query != expected[1] - 1:
        raise ValueError("Formal entry requires the complete frozen retrieval protocol")
    if args.check_inputs:
        print("All frozen sources checked; no scores, search or output were created.", flush=True)
        return 0
    environment = {"numpy": np.__version__, "scipy": scipy.__version__,
                   "torch": torch.__version__, "python": sys.version}
    if plan["selection_source"] and parent["context"]["environment"] != environment:
        raise ValueError("Test environment must match the frozen valid search")
    context = {**plan, "code": code, "environment": environment, "provenance": provenance}
    if previous and previous["context"] != context:
        raise ValueError("Resume code/source/environment signature changed")
    if previous and (output / "completed.json").exists():
        verify_run(output)
        from fgw_grid_search.register import write_proposal
        missing_proposal = not (output / "catalog_sources.json").exists()
        write_proposal(output)
        print("Verified completed run; " + ("restored missing registration proposal:" if missing_proposal
              else "no files changed:"), output, flush=True)
        return 0
    with exclusive(output):
        if previous:
            manifest = previous
            before = read_json(output / "source_protection_before.json")
        else:
            manifest = signed({"version": VERSION, "stage": plan["stage"], "context": context,
                "batch_id": started.strftime("%Y%m%dT%H%M%S%z") + "__fgw_grid_" + plan["stage"],
                "started_at": started.isoformat(), "output": str(output), "workers": args.workers,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "launch_command": [str(x) for x in (sys.argv if argv is None else [__file__, *argv])]})
            # Protect sources before publishing a resumable manifest.
            before = _extra_selection_inventory(sources.inventory(Path(plan["source_run"]), provenance), plan)
            record(output / "source_protection_before.json", before)
            record(output / "run_manifest.json", manifest)
        record(output / "settings.json", plan["settings"])
        if frozen is not None:
            record(output / "frozen_selection.json", frozen)
        inp = signed({"parent_signature": manifest["signature"], "split": plan["split"],
                      "context_digest": digest(context), "provenance": provenance})
        record(output / "inputs.json", inp)
        try:
            audit_path = output / "source_audit.json"
            write_json(output / "status.json", {"status": "running", "phase": "source_audit", "split": plan["split"]})
            if audit_path.exists():
                audit = read_signed(audit_path)
                if audit.get("context_digest") != digest(context) or audit.get("status") != "verified":
                    raise ValueError("Source audit belongs to different inputs")
            else:
                evidence = sources.audit_components(pool, bundle, scorer, wbbs)
                if evidence.get("status") != "verified":
                    raise ValueError("Source component audit did not pass")
                record(audit_path, signed({"status": "verified", "context_digest": digest(context), "evidence": evidence}))
            write_json(output / "status.json", {"status": "running", "phase": "full_candidate_components", "split": plan["split"]})
            provider, _ = cache.build_cache(output / "component_cache" / plan["split"],
                digest({"inputs": inp["signature"], "operation": "full_F_G_W_scores"}), pool, scorer, wbbs)
            write_json(output / "status.json", {"status": "running", "phase": "grid_evaluation", "split": plan["split"]})
            index = pipeline.evaluate(output, inp["signature"], provider, pool, bundle, plan["specs"],
                plan["settings"], workers=args.workers, stage=plan["stage"])
            if plan["stage"] == "search":
                write_grid_table(output, index)
                record(output / "selection.json", select_payload(manifest, index, output / "evaluation_index_valid.json"))
            checked_files(ROOT, code)
            after = _extra_selection_inventory(sources.inventory(Path(plan["source_run"]), provenance), plan)
            if before != after:
                raise ValueError("A protected source file changed")
            record(output / "source_protection_after.json", {"status": "unchanged", "files": len(after), "inventory_digest": digest(after)})
            write_json(output / "status.json", {"status": "completed", "stage": plan["stage"],
                "test_evaluated": plan["stage"] == "test", "new_evaluations": len(plan["specs"])})
            files = {}
            for p in sorted(output.rglob("*")):
                rel = p.relative_to(output)
                if (not p.is_file() or p == output / "completed.json" or
                        rel.parts[0] == "identities" or "rows" in rel.parts or
                        any(part.startswith(".") for part in rel.parts) or
                        p.suffix in (".npy", ".npz", ".log") or p.name == "rankings.csv"):
                    continue
                files[str(rel)] = file_hash(p)
            write_json(output / "completed.json", {"status": "completed", "version": VERSION,
                "signature": manifest["signature"], "stage": plan["stage"], "batch_id": manifest["batch_id"],
                "new_evaluations": len(plan["specs"]), "files": files})
            verify_run(output)
            from fgw_grid_search.register import write_proposal
            write_proposal(output)
            print("Completed and independently verified:", output, flush=True)
        except BaseException as exc:
            # Once published, the completion marker binds status.json too.
            # A later validation/proposal failure must not rewrite sealed evidence.
            if not (output / "completed.json").exists():
                write_json(output / "status.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
