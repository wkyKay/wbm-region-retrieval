"""Provenance for the fixed 0.5F+0.5WBBS cross-combination diagnostic."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import re

from fgw_grid_search import scoring
from fgw_grid_search.state import TZ, checked_files, exclusive, read_signed, separate, signed
from fgw_grid_search.pipeline import record
from fgw_topk_test.state import freeze_top10
from fgw_cross_test.state import (CANDIDATE_RULE, CROSS_SOURCE, PROTOCOL, REPORT_FILES,
    cross_pool, derived_signature, evaluation_folder, evaluation_paths, load_bundle,
    )
from wbm_retrieval.common import digest, file_hash, read_json

ROOT = Path(__file__).resolve().parents[1]
VERSION = "fw_add_cross_test_v1"
VARIANTS = ["FW_ADD"]
SPECS = {"FW_ADD": {"family": "sum_raw", "weights": [.5, 0., .5],
                    "ticks": [5, 0, 5], "denominator": 10}}

SELECTION_RULE = "user_specified_posthoc_components"
from fw_cross_test.state import (cross_reference, protected_inventory as previous_inventory,
                                 EXPECTED_CANDIDATE_HASH)
REFERENCE_FW_ROOT = ROOT / "artificial/fw_cross_test_20261120T104447+0800"


def fw_reference(metadata):
    """Bind the additional formula to the completed F/FW diagnostic pool."""
    folder = REFERENCE_FW_ROOT
    manifest = read_signed(folder / "run_manifest.json")
    marker = read_json(folder / "completed.json")
    if (marker.get("status") != "completed" or marker.get("signature") != manifest["signature"] or
            manifest.get("version") != "fw_cross_test_v1"):
        raise ValueError("The preceding F/FW cross-combination test is not completed")
    files = {name: file_hash(folder / name) for name in
             ("run_manifest.json", "candidate_protocol.json", "completed.json")}
    if any(marker["files"].get(name) != files[name]
           for name in ("run_manifest.json", "candidate_protocol.json")):
        raise ValueError("Preceding F/FW candidate metadata is not completion-bound")
    if (read_json(folder / "candidate_protocol.json") != metadata or
            metadata.get("candidate_identity_sha256") != EXPECTED_CANDIDATE_HASH):
        raise ValueError("Candidate identities differ from the preceding F/FW test")
    return {"root": str(folder), "signature": manifest["signature"], "files": files}


def protected_inventory(source):
    result = previous_inventory(source)
    folder = REFERENCE_FW_ROOT
    marker = read_json(folder / "completed.json")
    # Preceding lightweight reports and markers are read-only; large rankings
    # are never read or changed by this additional evaluation.
    for path in [folder / "completed.json", *(folder / name for name in marker["files"])]:
        stat = path.stat()
        result[str(path.resolve())] = {"sha256": file_hash(path), "size": stat.st_size,
                                     "mtime_ns": stat.st_mtime_ns}
    return result


def fixed_variants(settings):
    # Historical grid settings are retained solely as source/evaluation metadata;
    # the explicit user request, not its valid objective, selects this fixed formula.
    return signed({"schema_version": 1, "version": VERSION, "variants": list(VARIANTS),
        "specs": deepcopy(SPECS), "settings": scoring.validate_settings(settings),
        "selection_rule": SELECTION_RULE, "primary_variant": None,
        "test_parameter_search": False})


def validate_spec(spec):
    if spec not in SPECS.values():
        raise ValueError("Only the user-specified 0.5F+0.5WBBS formula is allowed")
    return deepcopy(spec)


def score_variant(components, spec):
    return scoring.score_variant(components, validate_spec(spec))


def choose_output(output_root=None, resume_root=None, *, now=None, root=ROOT):
    if output_root and resume_root:
        raise ValueError("Choose a new output or an explicit resume")
    now = now or datetime.now(TZ).replace(microsecond=0)
    output = (Path(root) / Path(resume_root or output_root or
              ("artificial/fw_add_cross_test_" + now.strftime("%Y%m%dT%H%M%S%z"))).expanduser()).resolve()
    allowed = (Path(root) / "artificial").resolve()
    if allowed not in output.parents or not output.name.startswith("fw_add_cross_test_"):
        raise ValueError("Use a new artificial/fw_add_cross_test_<actual-time> output")
    for parent in output.parents:
        if parent == allowed:
            break
        if (parent / "run_manifest.json").exists():
            raise ValueError("Cannot nest outputs inside an existing run")
    if resume_root:
        if not (output / "run_manifest.json").is_file():
            raise ValueError("Resume requires its original manifest")
    elif output.exists():
        raise FileExistsError("Output exists; use explicit signed resume")
    return output


def identity(output, name, signature):
    output = Path(output)
    path = output / "identities" / (name + ".json")
    key = "fw_add_cross_" + re.sub(r"[^a-zA-Z0-9_]+", "_", output.name) + "_test_" + name
    science = {"schema_version": 1, "key": key, "evaluation_signature": signature,
               "variant": name, "split": "test"}
    if path.exists():
        value = read_json(path)
        if any(value.get(k) != v for k, v in science.items()):
            raise ValueError("Resume evaluation identity differs")
        stamp = value["started_at"]
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00", stamp):
            raise ValueError("Missing original +08:00 identity")
        if value["run_id"] != datetime.fromisoformat(stamp).strftime("%Y%m%dT%H%M%S%z") + "__" + key:
            raise ValueError("Original run timestamp differs")
        return value
    now = datetime.now(TZ).replace(microsecond=0)
    value = {**science, "run_id": now.strftime("%Y%m%dT%H%M%S%z") + "__" + key,
             "started_at": now.isoformat(timespec="seconds")}
    record(path, value)
    return value


def verify_run(output, *, check_source=True):
    """Verify published reports, user-specified formulas, derived pool and signatures."""
    output = Path(output).resolve()
    manifest = read_signed(output / "run_manifest.json")
    context = manifest["context"]
    frozen = read_signed(output / "frozen_variants.json")
    names, specs = frozen["variants"], frozen["specs"]
    if (manifest.get("version") != VERSION or manifest.get("stage") != "test" or
            context.get("protocol") != PROTOCOL or context.get("formal_protocol") is not True or
            frozen != fixed_variants(context["settings"]) or names != VARIANTS or
            context.get("variants") != names or context.get("specs") != specs or
            context.get("frozen_variants_signature") != frozen["signature"] or
            context.get("primary_variant") is not None or
            context.get("selection_rule") != frozen["selection_rule"] or
            context.get("evaluation_paths") != evaluation_paths(output, names)):
        raise ValueError("Cross evaluation does not preserve user-specified 0.5F+0.5WBBS formula")
    settings = scoring.validate_settings(context["settings"])
    if settings != frozen["settings"] or read_json(output / "settings.json") != settings:
        raise ValueError("Cross evaluation changed original settings/metrics")
    if check_source:
        from fgw_topk_test.state import verify_run as verify_original
        original, original_frozen = verify_original(context["source_root"], check_cache=False)
        if (original["signature"] != context["source_signature"] or
                original["context"]["settings"] != settings or
                file_hash(Path(context["source_root"]) / "completed.json") != context["source_completed_sha256"]):
            raise ValueError("Original completed selection changed")
        if cross_reference(context["derived_protocol"]) != context.get("cross_reference"):
            raise ValueError("Preceding cross-combination source changed")
        if fw_reference(context["derived_protocol"]) != context.get("fw_reference"):
            raise ValueError("Preceding F/FW source changed")
    done, status = read_json(output / "completed.json"), read_json(output / "status.json")
    required = {"run_manifest.json", "frozen_variants.json", "settings.json", "status.json",
                "evaluation_index_test.json", "candidate_protocol.json", "protocol/cross_combination_candidates.csv",
                "source_protection_before.json", "source_protection_after.json"}
    if (done.get("status") != "completed" or done.get("signature") != manifest["signature"] or
            done.get("new_evaluations") != len(VARIANTS) or status.get("status") != "completed"):
        raise ValueError("Cross test is not fully complete")
    checked_files(output, done.get("files"), required)
    before, after = read_json(output / "source_protection_before.json"), read_json(output / "source_protection_after.json")
    if after != {"status": "unchanged", "files": len(before), "inventory_digest": digest(before)}:
        raise ValueError("Missing source protection evidence")
    derived = context["derived_protocol"]
    if (read_json(output / "candidate_protocol.json") != derived or
            derived.get("protocol_version") != PROTOCOL or derived.get("candidate_rule") != CANDIDATE_RULE or
            (derived.get("candidate_count_min"), derived.get("candidate_count_max")) != (6999, 7170)):
        raise ValueError("Cross candidate protocol differs")
    protocol_signature = derived_signature(derived)
    if context.get("retrieval_protocol_signature") != protocol_signature:
        raise ValueError("Cross protocol signature differs from its original serialization")
    index = read_json(output / "evaluation_index_test.json")
    if (index.get("variants") != names or index.get("specs") != specs or
            index.get("signature") != manifest["signature"] or
            index.get("paths") != evaluation_paths(output, names) or
            [r["variant"] for r in index.get("summary_rows", [])] != names or
            set(index.get("completed_sha256", {})) != set(names)):
        raise ValueError("Cross completion index differs from fixed user order")
    if {p.name for p in (output / "test").iterdir() if not p.name.startswith(".")} != set(names):
        raise ValueError("Unexpected or missing cross test variants")
    for name in names:
        folder = evaluation_folder(output, name)
        marker, ev = read_json(folder / "completed.json"), read_json(folder / "evaluation_manifest.json")
        signature = digest({"parent_signature": manifest["signature"], "variant": name,
                            "spec": specs[name], "protocol_signature": protocol_signature})
        if (marker.get("status") != "completed" or marker.get("signature") != signature or
                index["completed_sha256"][name] != file_hash(folder / "completed.json") or
                ev.get("signature") != signature or ev.get("parent_signature") != manifest["signature"] or
                ev.get("spec") != specs[name] or ev.get("settings") != settings or
                ev.get("derived_protocol") != derived or ev.get("provider_source") != context["provider_source"] or
                ev.get("num_queries") != 7199 or ev.get("ks") != settings["ks"] or
                ev.get("retrieval_protocol_signature") != protocol_signature):
            raise ValueError("Child cross evaluation provenance differs")
        expected_reports = set(REPORT_FILES) | {"verification.json", "timing.json"}
        if set(marker.get("files", {})) != expected_reports:
            raise ValueError("Child completion omits original coverage reports")
        checked_files(folder, marker["files"], expected_reports)
        if not (output / "identities" / (name + ".json")).is_file():
            raise ValueError("Child lacks its original persistent identity")
        ident = identity(output, name, signature)
        if read_json(folder / "run_identity.json") != ident or any(ev.get(k) != v for k, v in ident.items()):
            raise ValueError("Child actual launch identity differs")
        verification = read_json(folder / "verification.json")
        if (verification.get("status") != "verified" or verification.get("signature") != signature or
                verification.get("num_queries") != 7199 or
                verification.get("ranking_rows") != 7199 * max(settings["ks"]) or
                verification.get("files") != {p: file_hash(folder / p) for p in REPORT_FILES}):
            raise ValueError("Missing all-query independent verification")
        if (folder / "metrics.json").exists():
            raise ValueError("Cross task must not publish exact metrics")
    return manifest, frozen
