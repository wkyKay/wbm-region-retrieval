"""Register the six sealed user-specified cross-combination coverage results.

The transaction preserves every prior catalogue result. Large raw snapshot/CSV
files remain at their original paths; after official collect/check, pack/verify
both snapshot_archive.py and generated_archive.py before repository sync.
"""
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

from b1_structure_fusion.register import _atomic_write, _json_bytes, _lock, _relative
from results_catalog import build as catalog
from wbm_retrieval.common import file_hash, read_json
from .state import (TZ, PROTOCOL, VARIANTS, SPECS, SELECTION_RULE, evaluation_folder,
                    verify_run)


DISPLAY = {
    "GJ_ADD": "整图G/J等权加法（跨组合）",
    "FJ_ADD": "最优匹配F/J等权加法（跨组合）",
    "F_ONLY": "仅局部F最优匹配（跨组合）",
    "G_ONLY": "仅整图深度G（跨组合）",
    "J_ONLY": "仅联合结构J（跨组合）",
    "WBBS": "WBBS同源重放（跨组合）",
}
METHOD = {"GJ_ADD": "GJ", "FJ_ADD": "FJ", "F_ONLY": "F",
          "G_ONLY": "G", "J_ONLY": "J", "WBBS": "wbbs"}


def proposal(output, *, root=ROOT):
    output, root = Path(output).resolve(), Path(root).resolve()
    if root / "artificial" not in output.parents:
        raise ValueError("Expected an isolated new run inside artificial/")
    _relative(output, root / "artificial")
    manifest, frozen = verify_run(output)
    rows = []
    for name in VARIANTS:
        folder = evaluation_folder(output, name)
        marker = folder / "completed.json"
        ev = read_json(folder / "evaluation_manifest.json")
        ident = read_json(folder / "run_identity.json")
        supervision = ("untrained" if name == "J_ONLY" else "none" if name == "WBBS"
                       else "self_supervised")
        evidence = ["run_manifest.json", "completed.json", "frozen_variants.json",
                    "candidate_protocol.json", "evaluation_index_test.json",
                    "shared_verification.json", "component_audit.json", "baseline_replay.json",
                    "source_protection_after.json", "test/" + name + "/verification.json"]
        item = dict(
            key=ident["key"], repository="wbm-region-retrieval", path=_relative(folder, root),
            category="cross_combination", method=METHOD[name], variant="F/G/J-cross/" + name,
            split="test", manifest=_relative(folder / "evaluation_manifest.json", root),
            marker=_relative(marker, root), identity_schema_version=1,
            identity=_relative(folder / "run_identity.json", root),
            selection=_relative(output / "frozen_variants.json", root),
            coverage=None, supervision=supervision, display=DISPLAY[name],
            selection_rule=SELECTION_RULE, valid_selected_primary=False,
            required_families=["coverage"], protocol=PROTOCOL, posthoc=True,
            supporting_manifests=[_relative(output / filename, root) for filename in evidence],
            design_note=(
                "User-specified six-recipe cross-combination coverage diagnostic; fixed formula "
                + SPECS[name]["formula"] + ". F is frozen cc8 optimal radial-area matching, "
                "G is the original frozen self-supervised whole-map score, and J=exp(-2D_H) "
                "uses the original 128-dimensional whole-map joint descriptor. WBBS is the same "
                "frozen raw score. No angle term, score calibration, new training, valid search "
                "or test winner selection. Exclude self and identical complete label combinations "
                "before complete-candidate scoring; original coverage_v1 metrics, K, groups, "
                "tie rules and undefined-denominator handling are unchanged. Ground-truth labels "
                "filter benchmark eligibility only and never enter score formulas. Descending "
                "Coverage-nDCG@10 and metric-specific ranks are descriptive reporting, not method "
                "selection. Replayed endpoints are control roles, not independent training repeats. "
                "This posthoc diagnostic does not establish unseen-training-combination generalization "
                "or replace the designated cc8 primary method."))
        catalog.execution_identity(root, item, marker, ev)
        rows.append(item)
    if len(rows) != 6 or len({row["key"] for row in rows}) != 6:
        raise ValueError("Expected exactly six unique new cross-combination identities")
    return {"schema_version": 1, "runs": rows}


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
