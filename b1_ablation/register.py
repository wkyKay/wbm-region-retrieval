#!/usr/bin/env python3
"""Validate and register completed ablations, preserving prior result identities."""
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
from b1_ablation import scoring
from results_catalog import build as catalog
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from experiments.proposed.f_global_fusion.pipeline import signed_manifest
TZ = ZoneInfo("Asia/Shanghai")
_json_bytes = original._json_bytes
_lock = original._lock
_atomic_write = original._atomic_write


def proposal(output, *, root=ROOT):
    output, root = Path(output).resolve(), Path(root).resolve()
    if root / "artificial" not in output.parents:
        raise ValueError("Expected an isolated artificial/ output")
    if read_json(output / "status.json").get("status") != "completed":
        raise ValueError("Only completed and independently verified runs can be registered")
    manifest = signed_manifest(output / "run_manifest.json")
    if manifest["version"] != "b1_ablation_v1" or manifest["context"]["plan"]["mode"] != "all":
        raise ValueError("Not a formal completed ablation")
    source = Path(manifest["context"]["plan"]["source_root"]).resolve()
    if file_hash(source / "run_manifest.json") != manifest["context"]["source_manifest_sha256"]:
        raise ValueError("Original B1 manifest changed")
    if read_json(output / "source_protection_after.json").get("status") != "unchanged":
        raise ValueError("Source protection verification is required")
    # Register the original completed B1 once, without moving or editing its outputs.
    runs = original.proposal(source, root=root)["runs"]
    for split in ("valid", "test"):
        inp = signed_manifest(output / f"inputs_{split}.json")
        if inp.get("parent_signature") != manifest["signature"] or inp.get("specs") != scoring.specs(split):
            raise ValueError("Invalid fixed ablation input specification")
        index = read_json(output / f"evaluation_index_{split}.json")
        verify = read_json(output / f"verification_{split}.json")
        if index["variants"] != scoring.specs(split) or set(verify) != set(scoring.specs(split)):
            raise ValueError("Incomplete fixed matrix or verification")
        if not (output / "analysis" / split).is_dir() or not (output / f"source_audit_{split}.json").is_file():
            raise ValueError("Missing analysis or independent source audit")
        audit = signed_manifest(output / f"source_audit_{split}.json")
        if audit.get("status") != "verified" or audit.get("input_context_signature") != digest(manifest["context"]):
            raise ValueError("Missing or stale successful source audit")
        for name, spec in scoring.specs(split).items():
            folder = output / split / name
            settings = manifest["context"]["plan"]["settings"]
            signature = digest({"evaluation": inp["signature"], "variant": name, "spec": spec,
                                "split": split, "settings": settings, "device": manifest["context"]["plan"]["device"],
                                "retrieval_protocol_signature": inp["provenance"]["protocol_signature"]})
            marker = original._complete(folder, signature)
            identity = read_json(folder / "run_identity.json")
            evaluation = read_json(folder / "evaluation_manifest.json")
            v = verify[name]
            if (v.get("status") != "verified" or v.get("completion_sha256") != file_hash(marker)
                or v.get("ranking_sha256") != file_hash(folder / "rankings.csv")
                or v.get("num_queries") != (500 if split == "valid" else 7199)
                or index["completed_sha256"][name] != file_hash(marker)
                or evaluation.get("spec") != spec
                or any(evaluation.get(k) != value for k,value in identity.items())):
                raise ValueError("Incomplete or stale independent verification")
            rel = lambda path: str(Path(path).relative_to(root))
            item = {"key": identity["key"], "repository": "wbm-region-retrieval", "path": rel(folder),
                    "category": "fusion_ablation", "method": "F", "variant": "B1-ablation/" + name,
                    "split": split, "manifest": rel(folder / "evaluation_manifest.json"),
                    "marker": rel(marker), "identity_schema_version": 1, "identity": rel(folder / "run_identity.json"),
                    "selection": None, "coverage": None, "supervision": "untrained" if name == "structure_only" else "self_supervised",
                    "display": "B1 ablation " + name, "required_families": ["exact", "coverage"],
                    "protocol": "retrieval_v1", "posthoc": True,
                    "supporting_manifests": [rel(output / "run_manifest.json"), rel(output / f"inputs_{split}.json"),
                                             rel(output / f"verification_{split}.json")],
                    "design_note": "Fixed-parameter component deletion or I_2 control; source J lambda=2, angle gamma=.5. No retraining or new parameter search. Designed after observing B1 test; conditional contribution, not optimal simplified model."}
            catalog.execution_identity(root, item, marker, evaluation)
            runs.append(item)
    if len({r["key"] for r in runs}) != len(runs):
        raise ValueError("Duplicate run keys")
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
