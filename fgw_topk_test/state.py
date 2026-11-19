"""Immutable validation-prefix selection and formal test completion checks."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import re

from fgw_grid_search import scoring, selection
from fgw_grid_search.state import (TZ, checked_files, exclusive, read_signed, record,
                                  separate, signed, _check_summary_row)
from wbm_retrieval.common import digest, file_hash, read_json

ROOT = Path(__file__).resolve().parents[1]
VERSION = "fgw_topk_test_v1"
TOP_K = 10
PARENT_FILES = ("run_manifest.json", "completed.json", "selection.json",
                "evaluation_index_valid.json", "settings.json")
_ROW_FIELDS = ("variant", "family", "split", "num_queries", "precision10_hits",
               "Precision@10", "mAP@100", "Coverage-nDCG@10")


def freeze_top10(verified_selection):
    """Recheck the complete valid ordering and freeze its first ten unchanged.

    This pure function makes no use of test metrics. The caller separately
    verifies the source run and binds its paths/hashes in the parent manifest.
    """
    if not isinstance(verified_selection, dict):
        raise ValueError("Expected the signed completed valid selection")
    original = deepcopy(verified_selection)
    if original.get("signature") != digest({k: v for k, v in original.items() if k != "signature"}):
        raise ValueError("Invalid original valid-selection signature")
    settings = scoring.validate_settings(original["settings"])
    specs = scoring.specs(settings)
    rows = [{k: row[k] for k in _ROW_FIELDS} for row in original["ranking"]]
    expected = selection.choose_winner(rows, specs, settings)
    if any(original.get(k) != value for k, value in expected.items()):
        raise ValueError("Original valid ranking/winner differs from the fixed selection rule")
    if original.get("selected_spec") != specs[expected["winner"]]:
        raise ValueError("Original valid winner specification differs")
    if len(expected["ranking"]) < TOP_K:
        raise ValueError("A top-ten test requires at least ten completed valid recipes")
    names = [row["variant"] for row in expected["ranking"][:TOP_K]]
    return signed({"schema_version": 1, "version": VERSION, "top_k": TOP_K,
                   "ranked_variants": names, "specs": {name: specs[name] for name in names},
                   "valid_selection": original, "primary_variant": expected["winner"],
                   "selection_split": "valid", "test_parameter_search": False,
                   "ranking_rule": "prefix_of_frozen_valid_ranking_no_test_selection"})


def evaluation_paths(output, names):
    batch = Path(output).name
    if not batch or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_+.-]*", batch):
        raise ValueError("Unsafe test batch directory name")
    result = {}
    for name in names:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError("Unsafe fixed recipe ID")
        if name in result:
            raise ValueError("Duplicate fixed recipe ID")
        result[name] = "evaluations/" + batch + "__" + name
    return result


def variant_output(output, name):
    return Path(output) / evaluation_paths(output, [name])[name]


def evaluation_folder(output, name):
    return variant_output(output, name) / "test" / name


def choose_output(output_root=None, resume_root=None, *, now=None, root=ROOT):
    if output_root and resume_root:
        raise ValueError("Choose new output or explicit resume")
    now = now or datetime.now(TZ).replace(microsecond=0)
    output = (Path(root) / Path(resume_root or output_root or
              ("artificial/fgw_topk_test_" + now.strftime("%Y%m%dT%H%M%S%z"))).expanduser()).resolve()
    allowed = (Path(root) / "artificial").resolve()
    if allowed not in output.parents or not output.name.startswith("fgw_topk_test_"):
        raise ValueError("Use a new artificial/fgw_topk_test_<actual-time> directory")
    for parent in output.parents:
        if parent == allowed:
            break
        if (parent / "run_manifest.json").exists():
            raise ValueError("Cannot nest output in another run")
    if resume_root:
        if not (output / "run_manifest.json").is_file():
            raise ValueError("Resume requires the original signed manifest")
    elif output.exists():
        raise FileExistsError("Output exists; use explicit resume")
    return output


def verify_run(output, *, check_cache=True):
    """Read only; require the exact frozen top ten and ten complete test reports."""
    from fgw_grid_search.cache import FullScoreCache
    from fgw_grid_search.pipeline import REPORT_FILES, source_binding, variant_signature
    from fgw_grid_search.state import verify_run as verify_parent
    from results_catalog.build import BASE_PROTOCOL_SIGNATURE

    output = Path(output).resolve()
    manifest = read_signed(output / "run_manifest.json")
    if manifest.get("version") != VERSION or manifest.get("stage") != "test":
        raise ValueError("Not a fixed validation-top-ten test run")
    context = manifest["context"]
    if context.get("split") != "test" or context.get("stage") != "test":
        raise ValueError("Top-ten run must evaluate only test")
    origin = context["selection_source"]
    parent = Path(origin["root"])
    if set(origin["files"]) != set(PARENT_FILES):
        raise ValueError("Incomplete original valid-selection provenance")
    for name in PARENT_FILES:
        if file_hash(parent / name) != origin["files"][name]:
            raise ValueError("Original frozen valid source changed: " + name)
    parent_manifest, chosen = verify_parent(parent, required_stage="search", check_cache=check_cache)
    frozen = read_signed(output / "frozen_selection.json")
    if frozen != freeze_top10(chosen):
        raise ValueError("Frozen top ten differs from the completed valid prefix")
    settings = scoring.validate_settings(context["settings"])
    specs, names = context["specs"], frozen["ranked_variants"]
    paths = evaluation_paths(output, names)
    if (list(specs) != names or specs != frozen["specs"] or context.get("ranked_variants") != names
            or context.get("evaluation_paths") != paths or context.get("top_k") != TOP_K
            or context.get("frozen_selection_signature") != frozen["signature"]
            or context.get("primary_variant") != frozen["primary_variant"]
            or context.get("source_run") != parent_manifest["context"]["source_run"]
            or settings != parent_manifest["context"]["settings"]
            or context.get("grid_code") != parent_manifest["context"]["code"]
            or context.get("environment") != parent_manifest["context"]["environment"]):
        raise ValueError("Test plan/source/code/environment differs from the frozen valid selection")
    done, status = read_json(output / "completed.json"), read_json(output / "status.json")
    if (done.get("status") != "completed" or status.get("status") != "completed"
            or done.get("signature") != manifest["signature"] or done.get("stage") != "test"
            or done.get("new_evaluations") != TOP_K or status.get("test_evaluated") is not True
            or status.get("new_evaluations") != TOP_K):
        raise ValueError("Top-ten test is incomplete")
    required = {"run_manifest.json", "settings.json", "frozen_selection.json", "status.json",
                "inputs.json", "source_audit.json", "source_protection_before.json",
                "source_protection_after.json", "component_cache/test/plan.json",
                "component_cache/test/completed/completed.json",
                "evaluation_index_test.json", "evaluation_request_test.json"}
    checked_files(output, done.get("files"), required)
    if (output / "valid").exists() or (output / "selection.json").exists():
        raise ValueError("A test run must not create a fresh selection or valid evaluation")
    if read_json(output / "settings.json") != settings:
        raise ValueError("Test settings snapshot differs")
    before, after = read_json(output / "source_protection_before.json"), read_json(output / "source_protection_after.json")
    if after != {"status": "unchanged", "files": len(before), "inventory_digest": digest(before)}:
        raise ValueError("Missing old-source protection evidence")
    inp, audit = read_signed(output / "inputs.json"), read_signed(output / "source_audit.json")
    if (inp.get("parent_signature") != manifest["signature"] or inp.get("split") != "test"
            or inp.get("context_digest") != digest(context) or audit.get("status") != "verified"
            or audit.get("context_digest") != digest(context)):
        raise ValueError("Test inputs/source audit are not bound to this plan")
    index = read_json(output / "evaluation_index_test.json")
    cache_plan = read_json(output / "component_cache/test/plan.json")
    query_ids, gallery_ids = cache_plan.get("query_ids"), cache_plan.get("gallery_ids")
    expected_cache_signature = digest({"inputs": inp["signature"], "operation": "full_F_G_W_scores"})
    if (cache_plan.get("signature") != expected_cache_signature or cache_plan.get("split") != "test"
            or cache_plan.get("protocol_signature") != BASE_PROTOCOL_SIGNATURE
            or any(not isinstance(ids, list) or len(ids) != 7199
                   or any(type(value) is not int or value < 0 for value in ids)
                   or len(set(ids)) != 7199 for ids in (query_ids, gallery_ids))
            or set(query_ids) != set(gallery_ids)):
        raise ValueError("Full-score cache does not declare the complete frozen test identities")
    cache_marker = read_json(output / "component_cache/test/completed/completed.json")
    if (cache_marker.get("plan_sha256") != file_hash(output / "component_cache/test/plan.json")
            or cache_marker.get("plan_digest") != digest(cache_plan)
            or cache_marker.get("signature") != expected_cache_signature):
        raise ValueError("Completed cache is not bound to its declared query/gallery plan")
    query_digest, gallery_digest = digest(query_ids), digest(gallery_ids)
    cache = FullScoreCache(output / "component_cache/test") if check_cache else None
    if cache is not None and (cache.signature != digest({"inputs": inp["signature"], "operation": "full_F_G_W_scores"})
            or cache.split != "test" or cache.protocol_signature != BASE_PROTOCOL_SIGNATURE
            or len(cache.query_ids) != 7199 or len(cache.gallery_ids) != 7199):
        raise ValueError("Test full-score cache identity differs")
    binding = (source_binding(cache) if cache is not None else
               {"signature": expected_cache_signature,
                "completed_sha256": file_hash(output / "component_cache/test/completed/completed.json")})
    expected_request = {"signature": inp["signature"], "stage": "test", "split": "test",
                        "variants": specs, "ranked_variants": names, "evaluation_paths": paths,
                        "settings": settings, "provider_source": binding,
                        "retrieval_protocol_signature": BASE_PROTOCOL_SIGNATURE}
    if (read_json(output / "evaluation_request_test.json") != expected_request
            or any(index.get(k) != v for k, v in expected_request.items())
            or any(set(index.get(k, {})) != set(names) for k in
                   ("children", "variant_signatures", "completed_sha256"))):
        raise ValueError("Aggregate test index differs from the fixed request")
    children_dir = output / "evaluations"
    if set(p.name for p in children_dir.iterdir()) != {Path(p).name for p in paths.values()}:
        raise ValueError("Missing or extra test child directory")
    if len(index["summary_rows"]) != TOP_K or [r["variant"] for r in index["summary_rows"]] != names:
        raise ValueError("Test summary order differs from the frozen valid order")
    for name, summary in zip(names, index["summary_rows"]):
        spec, child, folder = specs[name], variant_output(output, name), evaluation_folder(output, name)
        if child.is_symlink() or (child / "valid").exists():
            raise ValueError("Unexpected child alias or split")
        published = {p.name for p in (child / "test").iterdir() if not p.name.startswith(".")}
        if published != {name}:
            raise ValueError("Child contains extra/missing test recipes")
        child_index_path = child / "evaluation_index_test.json"
        child_index = read_json(child_index_path)
        child_request = read_json(child / "evaluation_request_test.json")
        if child_request != {k: v for k, v in child_index.items()
                             if k not in ("variant_signatures", "completed_sha256", "summary_rows")}:
            raise ValueError("Child request differs from its completed evaluation index")
        expected_child = {"path": str(child.relative_to(output)),
                          "index": str(child_index_path.relative_to(output)), "sha256": file_hash(child_index_path)}
        signature = variant_signature(inp["signature"], name, spec, "test", settings,
                                      "test", binding, BASE_PROTOCOL_SIGNATURE)
        if (index["children"][name] != expected_child or child_index.get("signature") != inp["signature"]
                or child_index.get("stage") != "test" or child_index.get("split") != "test"
                or child_index.get("variants") != {name: spec} or child_index.get("settings") != settings
                or child_index.get("provider_source") != binding
                or child_index.get("retrieval_protocol_signature") != BASE_PROTOCOL_SIGNATURE
                or child_index.get("query_ids_digest") != query_digest
                or child_index.get("gallery_ids_digest") != gallery_digest
                or child_index.get("summary_rows") != [summary]
                or child_index.get("variant_signatures") != {name: signature}
                or index["variant_signatures"][name] != signature):
            raise ValueError("Child signature/specification/index differs")
        marker = read_json(folder / "completed.json")
        required_reports = set(REPORT_FILES) | {"verification.json", "timing.json"}
        if set(marker.get("files", {})) != required_reports or marker.get("status") != "completed":
            raise ValueError("Child completion omits required reports")
        checked_files(folder, marker["files"], required_reports)
        marker_hash = file_hash(folder / "completed.json")
        if (marker.get("signature") != signature or index["completed_sha256"][name] != marker_hash
                or child_index.get("completed_sha256") != {name: marker_hash}):
            raise ValueError("Child completion signature/hash differs")
        identity, evaluation = read_json(folder / "run_identity.json"), read_json(folder / "evaluation_manifest.json")
        key = "fgw_grid_" + re.sub(r"[^a-zA-Z0-9_]+", "_", child.name) + "_test_" + name
        stamp = identity.get("started_at", "")
        if (identity.get("schema_version") != 1 or identity.get("key") != key
                or identity.get("variant") != name or identity.get("split") != "test"
                or identity.get("evaluation_signature") != signature
                or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00", stamp)
                or identity.get("run_id") != datetime.fromisoformat(stamp).strftime("%Y%m%dT%H%M%S%z") + "__" + key):
            raise ValueError("Child identity lacks its original batch-specific test timestamp/key")
        if (identity != read_json(child / "identities" / ("test__" + name + ".json"))
                or any(evaluation.get(k) != v for k, v in identity.items())
                or evaluation.get("spec") != spec or evaluation.get("signature") != signature
                or evaluation.get("parent_signature") != inp["signature"] or evaluation.get("stage") != "test"
                or evaluation.get("settings") != settings or evaluation.get("provider_source") != binding
                or evaluation.get("num_queries") != 7199 or evaluation.get("gallery_count") != 7199
                or evaluation.get("ks") != settings["ks"] or evaluation.get("tie_seed") != settings["tie_seed"]
                or evaluation.get("candidate_rule") != "all_gallery_except_query"
                or evaluation.get("retrieval_protocol_signature") != BASE_PROTOCOL_SIGNATURE):
            raise ValueError("Child evaluation manifest/identity differs")
        verification = read_json(folder / "verification.json")
        if (verification.get("status") != "verified" or verification.get("num_queries") != 7199
                or verification.get("signature") != signature or verification.get("provider_source") != binding
                or verification.get("ks") != settings["ks"]
                or verification.get("full_candidates_checked") != 7199 * 7198
                or verification.get("all_queries_full_candidate_topk_exact") is not True
                or verification.get("files") != {p: file_hash(folder / p) for p in REPORT_FILES}
                or verification.get("ranking_rows") != 7199 * 500
                or verification.get("ranking_sha256") != file_hash(folder / "rankings.csv")):
            raise ValueError("Child lacks independent all-query/full-candidate verification")
        _check_summary_row(folder, summary, name)
        if summary.get("split") != "test" or summary.get("family") != spec["family"]:
            raise ValueError("Child summary is not a test result for this recipe")
    return manifest, frozen
