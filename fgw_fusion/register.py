#!/usr/bin/env python3
"""Validate and register completed F/G/WBBS experiments without changing old sources."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from zoneinfo import ZoneInfo
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from b1_structure_fusion import register as original
from fgw_fusion import scoring
from results_catalog import build as catalog
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from experiments.proposed.f_global_fusion.pipeline import signed_manifest
TZ = ZoneInfo("Asia/Shanghai")
_json_bytes = original._json_bytes
_lock = original._lock
_atomic_write = original._atomic_write


def _bound_files(folder, marker, required=()):
    """All declared files must exist inside the immutable completed directory."""
    value = read_json(marker)
    files = value.get("files")
    if not isinstance(files, dict) or not set(required) <= set(files):
        raise ValueError("Completion marker omits required files: " + str(marker))
    for name, expected in files.items():
        relative = Path(name)
        path = (folder / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Completion file escapes output")
        try:
            path.relative_to(folder.resolve())
        except ValueError as error:
            raise ValueError("Completion symlink escapes output") from error
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError("Completed file hash mismatch: " + str(path))
    return value


def _analysis_checked(folder, split, settings):
    completed = _bound_files(folder, folder / "completed.json",
                             ("analysis_plan.json", "metrics.csv", "paired_bootstrap.csv", "sources.json"))
    count = 500 if split == "valid" else 7199
    plan = read_json(folder / "analysis_plan.json")
    if (completed.get("status") != "complete" or completed.get("split") != split or
            completed.get("num_queries") != count or completed.get("num_classes") != 37 or
            not set(scoring.specs()) <= set(completed.get("methods", [])) or
            plan.get("anchor") != "FGW" or plan.get("ks") != settings["ks"] or
            plan.get("bootstrap_replicates") != 10000 or plan.get("seed") != 2026):
        raise ValueError("Analysis is incomplete or differs from frozen F/G/WBBS plan")
    sources = read_json(folder / "sources.json")
    if set(sources) != set(completed.get("methods", [])):
        raise ValueError("Analysis methods and source references differ")
    for refs in sources.values():
        if not isinstance(refs, dict) or not refs:
            raise ValueError("Invalid analysis source references")
        for name, expected in refs.items():
            path = Path(name)
            if not path.is_file() or file_hash(path) != expected:
                raise ValueError("Analysis source changed: " + name)


def proposal(output, *, root=ROOT):
    """Require complete six-evaluation matrix, independent checks and source protection."""
    output, root = Path(output).resolve(), Path(root).resolve()
    if root / "artificial" not in output.parents:
        raise ValueError("Expected an isolated artificial/ output")
    if read_json(output / "status.json").get("status") != "completed":
        raise ValueError("Only completed and independently verified runs can be registered")
    manifest = signed_manifest(output / "run_manifest.json")
    context = manifest["context"]
    plan = context["plan"]
    settings = dict(scoring.SETTINGS)
    specs = scoring.specs()
    if (manifest.get("version") != "fgw_fusion_v1" or plan.get("mode") != "all" or
            plan.get("splits") != ["valid", "test"] or plan.get("settings") != settings):
        raise ValueError("Not a formal frozen F/G/WBBS experiment")
    source = Path(plan["source_root"]).resolve()
    if file_hash(source / "run_manifest.json") != context.get("source_manifest_sha256"):
        raise ValueError("Original frozen model source manifest changed")
    required = {"run_manifest.json", "status.json", "source_protection_before.json", "source_protection_after.json"}
    for split in ("valid", "test"):
        required.update(f"{prefix}_{split}.json" for prefix in
                        ("inputs", "evaluation_index", "verification", "source_audit"))
        required.add(f"analysis/{split}/completed.json")
        required.update(f"{split}/{name}/completed.json" for name in specs)
    root_completed = _bound_files(output, output / "completed.json", required)
    if (root_completed.get("signature") != manifest["signature"] or
            root_completed.get("status") not in ("completed", "verified", "complete")):
        raise ValueError("Root completion does not bind the experiment signature")
    before = read_json(output / "source_protection_before.json")
    protection = read_json(output / "source_protection_after.json")
    if (protection.get("status") != "unchanged" or protection.get("files") != len(before) or
            protection.get("inventory_digest") != digest(before)):
        raise ValueError("Source protection verification is required")
    runs = []
    for split in ("valid", "test"):
        inp = signed_manifest(output / f"inputs_{split}.json")
        if (inp.get("parent_signature") != manifest["signature"] or inp.get("split") != split or
                inp.get("specs") != specs or inp.get("code") != context.get("code") or
                inp.get("provenance", {}).get("protocol_signature") != catalog.BASE_PROTOCOL_SIGNATURE):
            raise ValueError("Invalid frozen F/G/WBBS input specification")
        index = read_json(output / f"evaluation_index_{split}.json")
        verify = read_json(output / f"verification_{split}.json")
        if (index.get("signature") != inp["signature"] or index.get("split") != split or
                index.get("variants") != specs or set(verify) != set(specs) or
                set(index.get("completed_sha256", {})) != set(specs) or
                set(index.get("variant_signatures", {})) != set(specs)):
            raise ValueError("Incomplete F/G/WBBS evaluation matrix")
        audit = signed_manifest(output / f"source_audit_{split}.json")
        if audit.get("status") != "verified" or audit.get("input_context_signature") != digest(context):
            raise ValueError("Missing or stale successful source audit")
        _analysis_checked(output / "analysis" / split, split, settings)
        count = 500 if split == "valid" else 7199
        for name, spec in specs.items():
            folder = output / split / name
            signature = digest({"evaluation": inp["signature"], "variant": name, "spec": spec,
                                "split": split, "settings": settings, "device": "cpu",
                                "retrieval_protocol_signature": inp["provenance"]["protocol_signature"]})
            marker = original._complete(folder, signature)
            identity = read_json(folder / "run_identity.json")
            evaluation = read_json(folder / "evaluation_manifest.json")
            v = verify[name]
            if (index["variant_signatures"][name] != signature or
                    v.get("status") != "verified" or v.get("split") != split or v.get("variant") != name or
                    v.get("completion_sha256") != file_hash(marker) or
                    v.get("ranking_sha256") != file_hash(folder / "rankings.csv") or
                    v.get("num_queries") != count or v.get("ranking_rows") != count * max(settings["ks"]) or
                    index["completed_sha256"][name] != file_hash(marker) or evaluation.get("spec") != spec or
                    evaluation.get("parent_signature") != inp["signature"] or evaluation.get("signature") != signature or
                    evaluation.get("ks") != settings["ks"] or evaluation.get("tie_seed") != settings["tie_seed"] or
                    any(evaluation.get(k) != value for k, value in identity.items())):
                raise ValueError("Incomplete or stale independent verification")
            for family, filename in (("exact", "metrics.json"), ("coverage", "coverage_metrics.json")):
                report = read_json(folder / filename)
                metadata = report.get("metadata", {})
                protocol_signature = report.get("retrieval_protocol_signature", metadata.get("retrieval_protocol_signature"))
                if (report.get("split") != split or report.get("num_queries") != count or
                        report.get("metric_version") != family + "_v1" or report.get("ks") != settings["ks"] or
                        metadata.get("signature") != signature or metadata.get("smoke") or
                        protocol_signature != catalog.BASE_PROTOCOL_SIGNATURE):
                    raise ValueError("Report differs from the frozen formal evaluation")
            rel = lambda path: str(Path(path).relative_to(root))
            item = {"key": identity["key"], "repository": "wbm-region-retrieval", "path": rel(folder),
                    "category": "fusion_ablation", "method": "F", "variant": "FGW/" + name,
                    "split": split, "manifest": rel(folder / "evaluation_manifest.json"),
                    "marker": rel(marker), "identity_schema_version": 1, "identity": rel(folder / "run_identity.json"),
                    "selection": None, "coverage": None, "supervision": "self_supervised",
                    "display": "F/G/WBBS " + name, "required_families": ["exact", "coverage"],
                    "protocol": "retrieval_v1", "posthoc": True,
                    "supporting_manifests": [rel(output / path) for path in
                        ("run_manifest.json", "completed.json", f"inputs_{split}.json", f"evaluation_index_{split}.json",
                         f"source_audit_{split}.json", f"verification_{split}.json", f"analysis/{split}/completed.json")],
                    "design_note": "Fixed (0.5 F_optimal + 0.5 G)*WBBS with GW and FW deletions; remaining branch weight 1. No angle or J. Frozen self-supervised cc8/WaPIRL sources and original WBBS settings, no retraining or parameter search. Designed after prior B1/test observations; raw WBBS and exponential J differ in scale."}
            catalog.execution_identity(root, item, marker, evaluation)
            runs.append(item)
    if len(runs) != 6 or len({r["key"] for r in runs}) != len(runs):
        raise ValueError("Expected six unique F/G/WBBS evaluation keys")
    return {"schema_version": 1, "runs": runs}


def write_proposal(output, *, root=ROOT):
    value = proposal(output, root=root)
    target = Path(output) / "catalog_sources.json"
    if target.exists() and read_json(target) != value:
        raise ValueError("Existing registration proposal differs")
    if not target.exists():
        write_json(target, value)
    return target


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
        if len(entries) != len(sources["runs"]):
            raise ValueError("Existing catalogue contains duplicate keys")
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    output = (ROOT / args.run_root).resolve()
    print(apply_proposal(output) if args.apply else write_proposal(output))

if __name__ == "__main__":
    main()
