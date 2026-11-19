"""Immutable run metadata and validation for the isolated F/G/WBBS search."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import csv
import fcntl
from pathlib import Path
from zoneinfo import ZoneInfo

from wbm_retrieval.common import digest, file_hash, read_json, write_json
from . import scoring, selection

ROOT = Path(__file__).resolve().parents[1]
TZ = ZoneInfo("Asia/Shanghai")
VERSION = "fgw_grid_search_v1"
DEFAULT_SOURCE = "artificial/fgw_fusion_20261119T174724+0800"


def signed(value):
    if "signature" in value:
        raise ValueError("Sign an unsigned payload")
    return {**value, "signature": digest(value)}


def read_signed(path):
    value = read_json(path)
    if not isinstance(value, dict) or value.get("signature") != digest(
            {k: v for k, v in value.items() if k != "signature"}):
        raise ValueError("Invalid signed metadata: " + str(path))
    return value


def record(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Immutable run metadata differs: " + str(path))
    else:
        write_json(path, value)


def checked_files(folder, files, required=()):
    folder = Path(folder).resolve()
    if not isinstance(files, dict) or not set(required) <= set(files):
        raise ValueError("Completion omits required artifacts: " + str(folder))
    for name, sha in files.items():
        rel = Path(name)
        path = (folder / rel).resolve()
        if rel.is_absolute() or ".." in rel.parts or folder not in path.parents:
            raise ValueError("Artifact path escapes its completed directory")
        if not path.is_file() or file_hash(path) != sha:
            raise ValueError("Completed artifact changed: " + str(path))


@contextmanager
def exclusive(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".grid.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another grid process owns this output") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def resolve(value, root=ROOT):
    return (Path(root) / Path(value).expanduser()).resolve()


def choose_output(output_root=None, resume_root=None, *, now=None, root=ROOT):
    if output_root and resume_root:
        raise ValueError("Choose new output or explicit resume")
    now = now or datetime.now(TZ).replace(microsecond=0)
    target = resolve(resume_root or output_root or
                     ("artificial/fgw_grid_" + now.strftime("%Y%m%dT%H%M%S%z")), root)
    allowed = (Path(root) / "artificial").resolve()
    if allowed not in target.parents or not target.name.startswith("fgw_grid_"):
        raise ValueError("Use a new artificial/fgw_grid_<actual-time> directory")
    for parent in target.parents:
        if parent == allowed:
            break
        if (parent / "run_manifest.json").exists():
            raise ValueError("Cannot nest output inside another run")
    if resume_root:
        if not (target / "run_manifest.json").is_file():
            raise ValueError("Resume requires the original signed manifest")
    elif target.exists():
        raise FileExistsError("Output exists; use explicit signed resume")
    return target


def separate(output, roots):
    output = Path(output).resolve()
    for item in roots:
        source = Path(item).resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("New output must be separate from every source: " + str(source))


def select_payload(manifest, index, index_path):
    context = manifest["context"]
    settings = scoring.validate_settings(context["settings"])
    specs = scoring.specs(settings)
    if manifest["stage"] != "search" or index.get("split") != "valid" or index.get("variants") != specs:
        raise ValueError("Selection requires the complete valid grid")
    decision = selection.choose_winner(index["summary_rows"], specs, settings)
    return signed({"version": VERSION, "parent_signature": manifest["signature"],
                   "settings_sha256": digest(settings),
                   "evaluation_index_sha256": file_hash(index_path),
                   "selected_spec": specs[decision["winner"]], **decision})


def _check_summary_row(folder, row, variant):
    exact = read_json(folder / "metrics.json")
    coverage = read_json(folder / "coverage_metrics.json")
    with (folder / "per_query.csv").open(newline="") as handle:
        records = list(csv.DictReader(handle))
    hits = []
    for r in records:
        value = float(r["Precision@10"]) * 10
        if abs(value - round(value)) > 1e-10:
            raise ValueError("Precision@10 does not represent integer hits")
        hits.append(round(value))
    expected = {"variant": variant, "split": exact["split"], "num_queries": exact["num_queries"],
                "precision10_hits": sum(hits), "Precision@10": exact["metrics"]["Precision@10"],
                "mAP@100": exact["metrics"]["mAP@100"],
                "Coverage-nDCG@10": coverage["metrics"]["Coverage-nDCG@10"]}
    if len(records) != exact["num_queries"] or any(row.get(k) != v for k, v in expected.items()):
        raise ValueError("Search summary differs from completed query reports")


def verify_run(output, *, required_stage=None, check_cache=True):
    """Read only. Verify all reports and selection; never repair incomplete output."""
    from .cache import FullScoreCache
    from .pipeline import REPORT_FILES, source_binding, variant_signature
    from results_catalog.build import BASE_PROTOCOL_SIGNATURE
    output = Path(output).resolve()
    manifest = read_signed(output / "run_manifest.json")
    stage = manifest.get("stage")
    if manifest.get("version") != VERSION or stage not in ("search", "test"):
        raise ValueError("Not a supported F/G/WBBS grid run")
    if required_stage is not None and stage != required_stage:
        raise ValueError("Unexpected run stage")
    context = manifest["context"]
    settings = scoring.validate_settings(context["settings"])
    split = "valid" if stage == "search" else "test"
    if context.get("split") != split:
        raise ValueError("Stage and input split differ")
    done = read_json(output / "completed.json")
    status = read_json(output / "status.json")
    required = {"run_manifest.json", "settings.json", "status.json", "inputs.json", "source_audit.json",
                "source_protection_before.json", "source_protection_after.json",
                f"component_cache/{split}/completed/completed.json", f"evaluation_index_{split}.json"}
    required.add("selection.json" if stage == "search" else "frozen_selection.json")
    required.add(f"evaluation_request_{split}.json")
    if stage == "search":
        required.add("grid_results.csv")
    if (done.get("status") != "completed" or status.get("status") != "completed" or
            done.get("signature") != manifest["signature"] or done.get("stage") != stage or
            status.get("test_evaluated") != (stage == "test")):
        raise ValueError("Run is incomplete or its completion identity differs")
    checked_files(output, done.get("files"), required)
    if (output / ("test" if stage == "search" else "valid")).exists():
        raise ValueError("Unexpected evaluation of the other split")
    if read_json(output / "settings.json") != settings:
        raise ValueError("Settings snapshot changed")
    before = read_json(output / "source_protection_before.json")
    after = read_json(output / "source_protection_after.json")
    if after != {"status": "unchanged", "files": len(before), "inventory_digest": digest(before)}:
        raise ValueError("Missing source protection evidence")
    inp = read_signed(output / "inputs.json")
    audit = read_signed(output / "source_audit.json")
    if (inp.get("parent_signature") != manifest["signature"] or inp.get("split") != split or
            inp.get("context_digest") != digest(context) or audit.get("status") != "verified" or
            audit.get("context_digest") != digest(context)):
        raise ValueError("Input/source audit provenance differs")
    cache = FullScoreCache(output / "component_cache" / split) if check_cache else None
    if cache is not None and cache.signature != digest({"inputs": inp["signature"], "operation": "full_F_G_W_scores"}):
        raise ValueError("Component cache is not bound to this input")
    index_path = output / f"evaluation_index_{split}.json"
    index = read_json(index_path)
    specs = context["specs"]
    binding = source_binding(cache) if cache is not None else index.get("provider_source")
    if (index.get("stage") != stage or index.get("settings") != settings or
            index.get("provider_source") != binding or
            index.get("retrieval_protocol_signature") != BASE_PROTOCOL_SIGNATURE):
        raise ValueError("Evaluation index has changed stage/settings/cache/protocol")
    if cache is not None and (index.get("query_ids_digest") != digest(cache.query_ids.tolist()) or
            index.get("gallery_ids_digest") != digest(cache.gallery_ids.tolist()) or
            cache.split != split or cache.protocol_signature != BASE_PROTOCOL_SIGNATURE or
            len(cache.query_ids) != (500 if split == "valid" else 7199) or
            len(cache.gallery_ids) != (3599 if split == "valid" else 7199)):
        raise ValueError("Evaluation/cache does not retain the full frozen query/gallery identity")
    if stage == "search" and specs != scoring.specs(settings):
        raise ValueError("Completed search omits or changes grid candidates")
    if stage == "test" and len(specs) != 1:
        raise ValueError("Test is limited to the frozen single winner")
    if index.get("signature") != inp["signature"] or index.get("split") != split or index.get("variants") != specs:
        raise ValueError("Evaluation index and declared grid differ")
    for field in ("variant_signatures", "completed_sha256"):
        if set(index.get(field, {})) != set(specs):
            raise ValueError("Incomplete grid evaluation matrix")
    summaries = {row["variant"]: row for row in index["summary_rows"]}
    if len(summaries) != len(index["summary_rows"]) or set(summaries) != set(specs):
        raise ValueError("Duplicate or missing search summary rows")
    for name, spec in specs.items():
        scoring.validate_spec(spec)
        folder = output / split / name
        marker = read_json(folder / "completed.json")
        signature = index["variant_signatures"][name]
        if signature != variant_signature(inp["signature"], name, spec, split, settings,
                                          stage, binding, BASE_PROTOCOL_SIGNATURE):
            raise ValueError("Variant signature does not bind the declared formula and full-score cache")
        checked_files(folder, marker.get("files"), {"run_identity.json", "evaluation_manifest.json",
            "rankings.csv", "metrics.json", "coverage_metrics.json", "per_query.csv",
            "coverage_per_query.csv", "per_class.csv", "coverage_per_class.csv", "verification.json"})
        if marker.get("signature") != signature or file_hash(folder / "completed.json") != index["completed_sha256"][name]:
            raise ValueError("Variant completion signature/hash mismatch")
        evaluation = read_json(folder / "evaluation_manifest.json")
        verification = read_json(folder / "verification.json")
        count = 500 if split == "valid" else 7199
        if (evaluation.get("spec") != spec or evaluation.get("signature") != signature or
                evaluation.get("parent_signature") != inp["signature"] or evaluation.get("stage") != stage or
                evaluation.get("settings") != settings or evaluation.get("provider_source") != binding or
                evaluation.get("ks") != settings["ks"] or evaluation.get("tie_seed") != settings["tie_seed"] or
                evaluation.get("retrieval_protocol_signature") != BASE_PROTOCOL_SIGNATURE or
                verification.get("status") != "verified" or verification.get("num_queries") != count or
                verification.get("signature") != signature or verification.get("provider_source") != binding or
                verification.get("ks") != settings["ks"] or
                verification.get("full_candidates_checked") != count * (3598 if split == "valid" else 7198) or
                verification.get("all_queries_full_candidate_topk_exact") is not True or
                verification.get("files") != {p: file_hash(folder / p) for p in REPORT_FILES} or
                verification.get("ranking_rows") != count * 500 or
                verification.get("ranking_sha256") != file_hash(folder / "rankings.csv")):
            raise ValueError("Variant is not independently verified on the complete frozen pool")
        _check_summary_row(folder, summaries[name], name)
    if stage == "search":
        chosen = read_signed(output / "selection.json")
        if chosen != select_payload(manifest, index, index_path):
            raise ValueError("Frozen winner disagrees with completed valid grid")
    else:
        origin = context["selection_source"]
        parent = Path(origin["root"])
        for name in ("run_manifest.json", "completed.json", "selection.json"):
            if file_hash(parent / name) != origin["files"][name]:
                raise ValueError("Frozen valid selection source changed")
        _, chosen = verify_run(parent, required_stage="search", check_cache=check_cache)
        if (read_json(output / "frozen_selection.json") != chosen or
                specs != {chosen["winner"]: chosen["selected_spec"]} or
                settings != read_json(parent / "settings.json")):
            raise ValueError("Test did not use the frozen valid winner")
    if done.get("new_evaluations") != len(specs):
        raise ValueError("Completion count does not match the grid")
    return manifest, chosen
