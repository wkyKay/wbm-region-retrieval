#!/usr/bin/env python3
"""Prepare B1 catalogue sources; shared catalogue writes require explicit --apply."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from results_catalog import build as catalog
from wbm_retrieval.common import digest
from b1_structure_fusion import pipeline, scoring

TZ = ZoneInfo("Asia/Shanghai")


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _signed(path):
    value = catalog.read(path)
    if not isinstance(value, dict) or value.get("signature") != digest(
            {key: item for key, item in value.items() if key != "signature"}):
        raise ValueError(f"Invalid signed source metadata: {path}")
    return value


def _relative(path, root):
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError as error:
        raise ValueError(f"Catalogue source must remain inside its repository: {path}") from error


def _complete(folder, expected_signature):
    marker = folder / "completed.json"
    if not marker.is_file():
        raise ValueError(f"Incomplete B1 evaluation: {folder}")
    value = catalog.read(marker)
    if value.get("signature") != expected_signature:
        raise ValueError(f"Completed B1 signature mismatch: {folder}")
    files = value.get("files")
    required = {"run_identity.json", "evaluation_manifest.json", "metrics.json", "coverage_metrics.json",
                "per_query.csv", "coverage_per_query.csv", "rankings.csv"}
    if not isinstance(files, dict) or not required.issubset(files):
        raise ValueError(f"Completion marker is missing required B1 files: {folder}")
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Completion file path escapes evaluation")
        path = folder / relative
        _relative(path, folder)
        if not path.is_file() or catalog.sha(path) != expected:
            raise ValueError(f"B1 completed file hash mismatch: {path}")
    return marker


def proposal(output, *, root=ROOT):
    """Validate completed outputs and produce sources without touching the shared catalogue."""
    output, root = Path(output).resolve(), Path(root).resolve()
    _relative(output, root / "artificial")
    status = catalog.read(output / "status.json")
    if status.get("status") != "completed":
        raise ValueError("B1 catalogue proposal requires a completed, verified run")
    manifest_path = output / "run_manifest.json"
    manifest = _signed(manifest_path)
    if manifest.get("version") != "b1_structure_fusion_v1" or manifest.get("stage") not in ("valid", "test", "all"):
        raise ValueError("Unsupported B1 run manifest")
    context = manifest["context"]
    settings = context["settings"]
    if manifest["stage"] == "test":
        value = manifest.get("selection_file")
        if not value:
            raise ValueError("Test run has no frozen valid selection source")
        selection_path = Path(value)
        if not selection_path.is_absolute():
            selection_path = root / selection_path
    else:
        selection_path = output / "selection.json"
    selection_path = selection_path.resolve()
    _relative(selection_path, root)
    selection = pipeline.load_selection(selection_path, settings, context)
    splits = ["test"] if manifest["stage"] == "test" else ["valid"]
    if manifest["stage"] == "all" and selection["selections"]["J"]["selected"]["strength"] != 0:
        splits.append("test")
    if bool(status.get("test_evaluated")) != ("test" in splits):
        raise ValueError("Completed status disagrees with the frozen stage/selection")
    if "test" not in splits and (output / "test").exists():
        raise ValueError("Unexpected test outputs in a valid-only/zero-selected B1 run")
    runs = []
    for split in splits:
        specs = scoring.valid_variants(settings) if split == "valid" else pipeline.test_variants(selection, settings)
        inputs_path = output / f"inputs_{split}.json"
        inputs = _signed(inputs_path)
        if inputs.get("context") != context or inputs.get("split") != split:
            raise ValueError("B1 input provenance differs from the frozen context")
        index_path = output / f"evaluation_index_{split}.json"
        index = catalog.read(index_path)
        if (index.get("signature") != inputs["signature"] or index.get("split") != split or
                index.get("variants") != specs or set(index.get("completed_sha256", {})) != set(specs)):
            raise ValueError("B1 evaluation index does not match its complete frozen configuration set")
        verification_path = output / f"verification_{split}.json"
        verification = catalog.read(verification_path)
        if set(verification) != set(specs):
            raise ValueError("B1 verification does not cover every completed variant")
        for name, variant in specs.items():
            folder = output / split / name
            expected_signature = digest({"evaluation": inputs["signature"], "variant": name, "spec": variant})
            if index.get("variant_signatures", {}).get(name) != expected_signature:
                raise ValueError("Variant signature differs from frozen inputs and score specification")
            marker = _complete(folder, expected_signature)
            if catalog.sha(marker) != index["completed_sha256"][name]:
                raise ValueError("B1 completed marker changed after evaluation")
            evaluation = catalog.read(folder / "evaluation_manifest.json")
            identity = catalog.read(folder / "run_identity.json")
            if (evaluation.get("signature") != expected_signature or evaluation.get("parent_signature") != inputs["signature"] or
                    evaluation.get("spec") != variant or evaluation.get("split") != split or
                    any(evaluation.get(key) != value for key, value in identity.items())):
                raise ValueError("B1 evaluation manifest disagrees with its identity or inputs")
            checked = verification[name]
            query_count = 500 if split == "valid" else 7199
            if (checked.get("status") != "verified" or checked.get("split") != split or checked.get("variant") != name or
                    checked.get("num_queries") != query_count or
                    checked.get("ranking_rows") != query_count * max(settings["ks"]) or
                    checked.get("completion_sha256") != catalog.sha(marker) or
                    checked.get("ranking_sha256") != catalog.sha(folder / "rankings.csv")):
                raise ValueError("B1 independent verification is missing, stale or not successful")
            for family, filename in (("exact", "metrics.json"), ("coverage", "coverage_metrics.json")):
                report = catalog.read(folder / filename)
                metadata = report.get("metadata", {})
                protocol_signature = report.get("retrieval_protocol_signature", metadata.get("retrieval_protocol_signature"))
                if (report.get("split") != split or report.get("num_queries") != query_count or
                        report.get("metric_version") != family + "_v1" or report.get("ks") != settings["ks"] or
                        metadata.get("signature") != expected_signature or metadata.get("smoke") or
                        protocol_signature != catalog.BASE_PROTOCOL_SIGNATURE):
                    raise ValueError("B1 report differs from the formal frozen evaluation")
            spec = {"key": identity["key"], "repository": "wbm-region-retrieval", "path": _relative(folder, root),
                    "category": "fusion_ablation", "method": "F", "variant": f"B1/{name}", "split": split,
                    "manifest": _relative(folder / "evaluation_manifest.json", root), "marker": _relative(marker, root),
                    "identity_schema_version": 1, "identity": _relative(folder / "run_identity.json", root),
                    "selection": _relative(selection_path, root), "coverage": None, "supervision": "self_supervised",
                    "display": f"B1 {name}", "required_families": ["exact", "coverage"], "protocol": "retrieval_v1",
                    "posthoc": True, "supporting_manifests": [_relative(path, root) for path in
                        (manifest_path, inputs_path, index_path, verification_path)],
                    "design_note": "Frozen cc8/WaPIRL models; no retraining. TV structure score strength selected only on valid. Research direction follows historical test observations; J is the primary hypothesis."}
            catalog.execution_identity(root, spec, marker, evaluation)
            runs.append(spec)
    if len({item["key"] for item in runs}) != len(runs):
        raise ValueError("Duplicate B1 evaluation identities")
    return {"schema_version": 1, "runs": runs}


def _atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_proposal(output, *, root=ROOT):
    """Called after successful evaluation; this writes only inside this B1 run."""
    output = Path(output).resolve()
    value = proposal(output, root=root)
    path = output / "catalog_sources.json"
    if path.exists():
        if catalog.read(path) != value:
            raise ValueError("Existing registration proposal differs from completed B1 outputs")
    else:
        _atomic_write(path, _json_bytes(value))
    return path


@contextmanager
def _lock(folder):
    path = folder / ".registration.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ValueError(f"Catalogue registration lock exists: {path}") from error
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        path.unlink()


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
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=ROOT.parent)
    parser.add_argument("--apply", action="store_true", help="Explicitly register verified results in the shared catalogue")
    args = parser.parse_args(argv)
    output = (ROOT / args.run_root.expanduser()).resolve()
    if args.apply:
        print(json.dumps(apply_proposal(output, workspace=args.workspace), ensure_ascii=False, indent=2))
    else:
        print(f"Verified proposal only; shared catalogue unchanged: {write_proposal(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
