#!/usr/bin/env python3
"""Register frozen FGW cross-combination coverage evaluations atomically."""
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

from wbm_retrieval.common import read_json, write_json
from results_catalog import build as catalog
from fgw_fusion.register import _json_bytes, _lock, _atomic_write
from fgw_grid_search.state import TZ
from fgw_cross_test.state import evaluation_folder, evaluation_paths, verify_run

MAX_CATALOG_FILE_BYTES = 100 * 1024 ** 2


CROSS_PROTOCOL = "coverage_cross_combination_v1"
CANDIDATE_RULE = "all_gallery_except_query_and_exact_label_combination"
CANDIDATE_IDENTITY = "68c6875330ed70841586c654da776f8d0bc72bcd4cd341b79cea29705543f0b8"


def proposal(output, *, root=ROOT):
    """Validate the original valid prefix and unchanged coverage metric contract."""
    output, root = Path(output).resolve(), Path(root).resolve()
    if root / "artificial" not in output.parents:
        raise ValueError("Expected an isolated cross-combination output in artificial/")
    manifest, frozen = verify_run(output)
    context = manifest["context"]
    settings, specs = context["settings"], context["specs"]
    selection = frozen["valid_selection"]
    ranking = selection["ranking"][:10]
    names = [row["variant"] for row in ranking]
    if (manifest.get("stage") != "test" or context.get("split") != "test" or
            frozen.get("top_k") != 10 or len(names) != 10 or len(set(names)) != 10 or
            frozen.get("ranked_variants") != names or list(specs) != names or
            [row.get("rank") for row in ranking] != list(range(1, 11)) or
            any(row.get("split") != "valid" for row in ranking) or
            specs != {name: selection["specifications"][name] for name in names} or
            frozen.get("specs") != specs or frozen.get("selection_split") != "valid" or
            frozen.get("test_parameter_search") is not False or
            frozen.get("primary_variant") != selection["winner"] or names[0] != selection["winner"] or
            settings != selection["settings"]):
        raise ValueError("Registration requires exactly the frozen valid Top 10 and original settings")
    derived = context.get("derived_protocol", {})
    if (derived.get("protocol_version") != CROSS_PROTOCOL or
            derived.get("candidate_rule") != CANDIDATE_RULE or
            derived.get("candidate_identity_sha256") != CANDIDATE_IDENTITY or
            derived.get("base_protocol_signature") != catalog.BASE_PROTOCOL_SIGNATURE or
            derived.get("candidate_count_min") != 6999 or derived.get("candidate_count_max") != 7170 or
            derived.get("filter_uses_ground_truth") is not True):
        raise ValueError("Cross-combination candidate protocol differs from the frozen test pool")

    def relative(path):
        try:
            return str(Path(path).resolve().relative_to(root))
        except ValueError as error:
            raise ValueError("A registered source escapes this repository") from error

    paths = context["evaluation_paths"]
    if paths != evaluation_paths(output, names):
        raise ValueError("Missing or unexpected cross-combination evaluation paths")
    rows = []
    for rank, name in enumerate(names, 1):
        spec = specs[name]
        folder = evaluation_folder(output, name).resolve()
        if output not in folder.parents:
            raise ValueError("Evaluation path escapes its cross-combination run")
        identity = read_json(folder / "run_identity.json")
        evaluation = read_json(folder / "evaluation_manifest.json")
        marker = folder / "completed.json"
        if (evaluation.get("stage") != "test" or evaluation.get("spec") != spec or
                evaluation.get("settings") != settings or identity.get("variant") != name or
                evaluation.get("protocol") != CROSS_PROTOCOL or evaluation.get("derived_protocol") != derived or
                any(evaluation.get(k) != v for k, v in identity.items())):
            raise ValueError("Evaluation identity/specification differs from its frozen cross-combination plan")
        if any((folder / filename).exists() for filename in ("metrics.json", "per_query.csv", "per_class.csv")):
            raise ValueError("Cross-combination evaluation must contain coverage only, without exact reports")
        report = read_json(folder / "coverage_metrics.json")
        metadata = report.get("metadata", {})
        if (report.get("split") != "test" or report.get("num_queries") != 7199 or
                report.get("ks") != settings["ks"] or report.get("metric_version") != "coverage_v1" or
                metadata.get("signature") != evaluation["signature"] or metadata.get("smoke") or
                metadata.get("derived_protocol") != derived):
            raise ValueError("Coverage report differs from the formal cross-combination test protocol")
        expected_counts = {f"{metric}@{k}": count for k in settings["ks"] for metric, count in (
            ("Coverage-nDCG", 7000), ("MeanCoverage", 7199), ("FullCoverageHit", 7199),
            ("FullCoveragePrecision", 7199), ("FullCoverageRecall", 6200), ("FullCoverage_mAP", 6200))}
        if (report.get("queries_without_full_coverage_candidates") != 999 or
                report.get("metric_query_counts") != expected_counts):
            raise ValueError("Coverage valid-query counts differ from the unchanged zero-denominator rules")
        item = {
            "key": identity["key"], "repository": "wbm-region-retrieval", "path": relative(folder),
            "category": "cross_combination", "method": "F", "variant": "FGW-top10-cross/" + name,
            "split": "test", "manifest": relative(folder / "evaluation_manifest.json"),
            "marker": relative(marker), "identity_schema_version": 1,
            "identity": relative(folder / "run_identity.json"),
            "selection": relative(output / "frozen_selection.json"),
            "coverage": None, "supervision": "self_supervised",
            "display": f"F/G/WBBS cross-combination valid rank {rank}: {name}" +
                (" (original valid-selected primary)" if rank == 1 else ""),
            "valid_rank": rank, "valid_selected_primary": rank == 1,
            "required_families": ["coverage"], "protocol": CROSS_PROTOCOL, "posthoc": True,
            "supporting_manifests": [relative(output / p) for p in ("run_manifest.json", "completed.json",
                "source_protection_after.json", "evaluation_index_test.json")],
            "design_note": "User-authorized supplementary coverage test of the original frozen valid Top 10. "
                "Only candidate eligibility changes: exclude self and identical complete label combinations "
                "before scoring and min-max normalization. Reuse verified complete F/G/WBBS component scores "
                "read-only, with unchanged coverage_v1 formulas, K, groups, tie policy and zero-denominator rules. "
                "No exact reports, new metrics, training, parameter search or test winner selection. "
                "The original valid winner remains the designated primary configuration. "
                "Ground-truth labels filter benchmark eligibility only and never enter method scores. "
                "This is a posthoc diagnostic, not unseen-combination training generalization."
        }
        catalog.execution_identity(root, item, marker, evaluation)
        rows.append(item)
    if len({row["key"] for row in rows}) != 10:
        raise ValueError("Duplicate cross-combination registration identities")
    return {"schema_version": 1, "runs": rows}


def write_proposal(output, *, root=ROOT):
    value = proposal(output, root=root)
    path = Path(output) / "catalog_sources.json"
    if path.exists() and read_json(path) != value:
        raise ValueError("Registration proposal changed")
    if not path.exists():
        write_json(path, value)
    return path


def _compact_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")


def apply_proposal(output, *, workspace=None, root=ROOT):
    """Validate and preserve every old record, then publish with atomic rollback."""
    output, root = Path(output).resolve(), Path(root).resolve()
    workspace = root.parent if workspace is None else Path(workspace).resolve()
    if (workspace / "wbm-region-retrieval").resolve() != root:
        raise ValueError("Workspace must contain this wbm-region-retrieval repository")
    proposed = catalog.read(write_proposal(output, root=root))
    folder = root / "results_catalog"
    with _lock(folder):
        original = {name: (folder / name).read_bytes() for name in ("sources.json", "snapshot.json")}
        sources, previous = (json.loads(original[name]) for name in ("sources.json", "snapshot.json"))
        fingerprint = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
        if previous.get("source_spec_sha256") != fingerprint:
            raise ValueError("Existing catalogue sources and snapshot differ; collect/check before registration")
        for name, content in catalog.render(previous).items():
            path = folder / name
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                raise ValueError(f"Existing generated catalogue is stale: {name}")
            original[name] = path.read_bytes()
        entries = {item["key"]: item for item in sources["runs"]}
        old_runs = {item["key"]: item for item in previous["runs"]}
        if len(entries) != len(sources["runs"]) or len(old_runs) != len(previous["runs"]) or set(entries) != set(old_runs):
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
        updated = catalog.collect(workspace, merged, previous)
        updated_runs = {item["key"]: item for item in updated["runs"]}
        expected_keys = set(entries) | {item["key"] for item in additions}
        if len(updated_runs) != len(updated["runs"]) or set(updated_runs) != expected_keys:
            raise ValueError("Collected catalogue omitted or duplicated an evaluation")
        if any(updated_runs[key] != row for key, row in old_runs.items()):
            raise ValueError("An existing catalogue record changed during collection")
        generated = catalog.render(updated)
        if any(not (folder / name).is_file() or (folder / name).read_bytes() != content
               for name, content in original.items()):
            raise ValueError("Catalogue changed concurrently during collection")
        contents = {"snapshot.json": _compact_bytes(updated),
                    **{name: text.encode("utf-8") for name, text in generated.items()},
                    "sources.json": _json_bytes(merged)}
        oversized = [name for name, content in contents.items() if len(content) >= MAX_CATALOG_FILE_BYTES]
        if oversized:
            raise ValueError("Generated catalogue exceeds the 100 MiB file limit: " + ", ".join(oversized))
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
                 "before_sha256": {name: hashlib.sha256(data).hexdigest() if data is not None else None for name, data in before.items()},
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
