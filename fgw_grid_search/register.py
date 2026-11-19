#!/usr/bin/env python3
"""Register only completed, independently verified grid or frozen-winner evaluations."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tempfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wbm_retrieval.common import file_hash, read_json, write_json
from results_catalog import build as catalog
from fgw_fusion.register import _json_bytes, _lock, _atomic_write
from fgw_grid_search.state import TZ, verify_run


def proposal(output, *, root=ROOT):
    output, root = Path(output).resolve(), Path(root).resolve()
    if root / "artificial" not in output.parents:
        raise ValueError("Expected an isolated grid output in artificial/")
    manifest, selection = verify_run(output)
    context = manifest["context"]
    split = context["split"]
    settings = context["settings"]
    specs = context["specs"]
    if manifest["stage"] == "search":
        selection_path = output / "selection.json"
    else:
        selection_path = output / "frozen_selection.json"
    def relative(path):
        try:
            return str(Path(path).resolve().relative_to(root))
        except ValueError as error:
            raise ValueError("A registered source escapes this repository") from error
    rows = []
    for name, spec in specs.items():
        folder = output / split / name
        identity = read_json(folder / "run_identity.json")
        evaluation = read_json(folder / "evaluation_manifest.json")
        marker = folder / "completed.json"
        if (evaluation.get("stage") != manifest["stage"] or evaluation.get("spec") != spec or
                evaluation.get("settings") != settings or
                any(evaluation.get(k) != v for k, v in identity.items())):
            raise ValueError("Evaluation identity/specification differs from its grid")
        for family, filename in (("exact", "metrics.json"), ("coverage", "coverage_metrics.json")):
            report = read_json(folder / filename)
            metadata = report.get("metadata", {})
            if (report.get("split") != split or report.get("num_queries") != (500 if split == "valid" else 7199) or
                    report.get("ks") != settings["ks"] or report.get("metric_version") != family + "_v1" or
                    metadata.get("signature") != evaluation["signature"] or metadata.get("smoke") or
                    report.get("retrieval_protocol_signature", metadata.get("retrieval_protocol_signature")) != catalog.BASE_PROTOCOL_SIGNATURE):
                raise ValueError("Grid reports do not use the formal frozen protocol")
        item = {
            "key": identity["key"], "repository": "wbm-region-retrieval", "path": relative(folder),
            "category": "fusion", "method": "F", "variant": "FGW-grid/" + name,
            "split": split, "manifest": relative(folder / "evaluation_manifest.json"),
            "marker": relative(marker), "identity_schema_version": 1,
            "identity": relative(folder / "run_identity.json"), "selection": relative(selection_path),
            "coverage": None, "supervision": "self_supervised", "display": "F/G/WBBS grid " + name,
            "required_families": ["exact", "coverage"], "protocol": "retrieval_v1", "posthoc": True,
            "supporting_manifests": [relative(output / p) for p in ("run_manifest.json", "completed.json",
                "inputs.json", "source_audit.json", "source_protection_after.json", f"evaluation_index_{split}.json")],
            "design_note": "Finite valid-only grid of additive/raw, additive/query-minmax and weighted geometric scores; F coefficient/exponent strictly positive. Frozen optimal cc8 F, WaPIRL G and original WBBS. No angle A or J, no retraining. Winner selected only by valid exact P@10 integer hits with a fixed label-independent tie policy. Test evaluates only the frozen global winner. Designed after prior test observations; best means best in the declared finite grid. Duplicate/equivalent endpoints are comparison roles, not independent repeats."
        }
        catalog.execution_identity(root, item, marker, evaluation)
        rows.append(item)
    if not rows or len({r["key"] for r in rows}) != len(rows):
        raise ValueError("Empty or duplicate grid registration identities")
    if split == "test" and (len(rows) != 1 or set(specs) != {selection["winner"]}):
        raise ValueError("Only the frozen valid winner may be registered as test")
    return {"schema_version": 1, "runs": rows}


def write_proposal(output, *, root=ROOT):
    value = proposal(output, root=root)
    path = Path(output) / "catalog_sources.json"
    if path.exists() and read_json(path) != value:
        raise ValueError("Registration proposal changed")
    if not path.exists():
        write_json(path, value)
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
