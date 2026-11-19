"""Immutable provenance for candidate-only, frozen-top-ten coverage evaluation."""
from datetime import datetime
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np

from fgw_grid_search import scoring
from fgw_grid_search.state import TZ, checked_files, exclusive, read_signed, separate, signed
from fgw_grid_search.pipeline import record, source_binding
from fgw_topk_test.state import freeze_top10
from wbm_retrieval.common import digest, file_hash, read_json
from wbm_retrieval.retrieval_protocol import FullGalleryPool, protocol_file_hashes

ROOT = Path(__file__).resolve().parents[1]
VERSION = "fgw_cross_test_v1"
PROTOCOL = "coverage_cross_combination_v1"
CANDIDATE_RULE = "all_gallery_except_query_and_exact_label_combination"
TOP_K = 10
REPORT_FILES = ("run_identity.json", "evaluation_manifest.json", "rankings.csv",
                "coverage_metrics.json", "coverage_per_query.csv", "coverage_per_class.csv")
CROSS_SOURCE = ROOT.parent / "compared_methods/shared/cross_combination.py"


def cross_pool(base, labels):
    """Reuse the existing compared_methods candidate rule without modifying it."""
    spec = importlib.util.spec_from_file_location("fgw_cross_protocol_source", CROSS_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.PROTOCOL_VERSION != PROTOCOL or module.CANDIDATE_RULE != CANDIDATE_RULE:
        raise ValueError("Existing cross-combination candidate definition changed")
    return module.CrossCombinationPool(base, labels)


def derived_signature(metadata):
    """Match the existing protocol's JSON serialization, including spaces."""
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()


def evaluation_paths(output, names):
    paths = {}
    for name in names:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or name in paths:
            raise ValueError("Invalid or duplicate frozen variant")
        paths[name] = "test/" + name
    return paths


def evaluation_folder(output, name):
    return Path(output) / evaluation_paths(output, [name])[name]


def choose_output(output_root=None, resume_root=None, *, now=None, root=ROOT):
    if output_root and resume_root:
        raise ValueError("Choose a new output or an explicit resume")
    now = now or datetime.now(TZ).replace(microsecond=0)
    output = (Path(root) / Path(resume_root or output_root or
              ("artificial/fgw_cross_test_" + now.strftime("%Y%m%dT%H%M%S%z"))).expanduser()).resolve()
    allowed = (Path(root) / "artificial").resolve()
    if allowed not in output.parents or not output.name.startswith("fgw_cross_test_"):
        raise ValueError("Use a new artificial/fgw_cross_test_<actual-time> output")
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


def load_bundle(source, provider, *, root=ROOT):
    """Read labels and original class IDs only; no raw-map/model inference.

    Frozen split/list hashes identify labels. Original class IDs are verified
    against the source's bound per-query report instead of being renumbered.
    """
    source, root = Path(source), Path(root)
    protocol = root / "protocol/retrieval_v1"
    split = root / "protocol/deduplicated/split.csv"
    hashes = protocol_file_hashes(protocol)
    metadata = read_json(protocol / "metadata.json")
    if (digest(hashes) != provider.protocol_signature or
            file_hash(split) != metadata["split_sha256"]):
        raise ValueError("Current frozen split/query lists differ from source cache")
    with split.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    sample_ids = [int(row["sample_id"]) for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate source IDs in frozen split")
    labels = np.zeros((max(sample_ids) + 1, 8), dtype=np.uint8)
    for row, sample_id in zip(rows, sample_ids):
        values = [int(row[f"label_{i}"]) for i in range(8)]
        if any(v not in (0, 1) for v in values) or not any(values):
            raise ValueError("Frozen split contains invalid defect labels")
        if row["label_signature"] != "|".join(str(i) for i, v in enumerate(values) if v):
            raise ValueError("Frozen split signature differs from label columns")
        labels[sample_id] = values
    test_ids = np.array(sorted(int(r["sample_id"]) for r in rows if r["split"] == "test"), dtype=np.int64)
    for filename, expected in (("test_queries.csv", provider.query_ids), ("test_gallery.csv", provider.gallery_ids)):
        with (protocol / filename).open(newline="") as handle:
            actual = np.array([int(row["sample_id"]) for row in csv.DictReader(handle)], dtype=np.int64)
        if not np.array_equal(actual, expected) or not np.array_equal(actual, test_ids):
            raise ValueError("Original test identities differ from frozen full source")
    frozen = read_signed(source / "frozen_selection.json")
    source_manifest = read_signed(source / "run_manifest.json")
    first = frozen["ranked_variants"][0]
    source_folder = source / source_manifest["context"]["evaluation_paths"][first] / "test" / first
    report_path = source_folder / "coverage_per_query.csv"
    marker = read_json(source_folder / "completed.json")
    if marker["files"].get(report_path.name) != file_hash(report_path):
        raise ValueError("Source class-ID report is no longer bound")
    classes = np.full(len(labels), -1, dtype=np.int64)
    with report_path.open(newline="") as handle:
        original_rows = list(csv.DictReader(handle))
    if [int(row["query_id"]) for row in original_rows] != test_ids.tolist():
        raise ValueError("Source per-query label report misses test identities")
    forward, backward = {}, {}
    for row in original_rows:
        q, class_id = int(row["query_id"]), int(row["class_id"])
        label = tuple(map(int, labels[q]))
        if row["label_signature"] != "".join(map(str, label)) or int(row["label_count"]) != sum(label):
            raise ValueError("Frozen labels differ from source evaluated labels")
        if ((class_id in forward and forward[class_id] != label) or
                (label in backward and backward[label] != class_id)):
            raise ValueError("Source class mapping is not one-to-one")
        forward[class_id], backward[label], classes[q] = label, class_id, class_id
    if len(forward) != 37 or len(test_ids) != 7199:
        raise ValueError("Formal cross test requires 7,199 queries and 37 classes")
    labels.flags.writeable = classes.flags.writeable = False
    base = FullGalleryPool(provider.query_ids, provider.gallery_ids, "test", provider.protocol_signature)
    bundle = SimpleNamespace(labels=labels, class_ids=classes, valid_ids=test_ids, reserved_ids=test_ids)
    return bundle, base, {"split_sha256": file_hash(split), "retrieval_files": hashes,
                          "source_label_report": str(report_path), "source_label_report_sha256": file_hash(report_path),
                          "source_data_sha256": metadata["source_sha256"]}


def protected_inventory(source, *, verify_historical=True):
    """Protect old models/data/results and the complete source component cache."""
    source = Path(source).resolve()
    previous = read_json(source / "source_protection_before.json")
    paths = {Path(name) for name in previous}
    done = read_json(source / "completed.json")
    paths.update(source / name for name in done["files"])
    paths.add(source / "completed.json")
    cache = source / "component_cache/test/completed"
    paths.update(cache / name for name in read_json(cache / "completed.json")["files"])
    index = read_json(source / "evaluation_index_test.json")
    for item in index["children"].values():
        child = source / item["path"]
        child_index = read_json(child / "evaluation_index_test.json")
        for name in child_index["variants"]:
            folder = child / "test" / name
            paths.update(folder / rel for rel in read_json(folder / "completed.json")["files"])
    paths.add(CROSS_SOURCE)
    paths.update(p for p in (ROOT / "protocol").rglob("*") if p.is_file())
    result = {}
    for path in sorted(paths):
        stat = path.stat()
        value = {"sha256": file_hash(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if verify_historical and str(path) in previous and value["sha256"] != previous[str(path)]["sha256"]:
            raise ValueError("Protected historical source changed: " + str(path))
        result[str(path.resolve())] = value
    return result


def identity(output, name, signature):
    output = Path(output)
    path = output / "identities" / (name + ".json")
    key = "fgw_cross_" + re.sub(r"[^a-zA-Z0-9_]+", "_", output.name) + "_test_" + name
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
    """Verify published reports, fixed selection, derived pool and signatures."""
    output = Path(output).resolve()
    manifest = read_signed(output / "run_manifest.json")
    context = manifest["context"]
    frozen = read_signed(output / "frozen_selection.json")
    names, specs = frozen["ranked_variants"], frozen["specs"]
    if (manifest.get("version") != VERSION or manifest.get("stage") != "test" or
            context.get("protocol") != PROTOCOL or context.get("formal_protocol") is not True or
            frozen != freeze_top10(frozen["valid_selection"]) or len(names) != TOP_K or
            context.get("ranked_variants") != names or context.get("specs") != specs or
            context.get("frozen_selection_signature") != frozen["signature"] or
            context.get("primary_variant") != frozen["primary_variant"] or
            context.get("evaluation_paths") != evaluation_paths(output, names)):
        raise ValueError("Cross evaluation does not preserve frozen valid top ten")
    settings = scoring.validate_settings(context["settings"])
    if settings != frozen["valid_selection"]["settings"] or read_json(output / "settings.json") != settings:
        raise ValueError("Cross evaluation changed original settings/metrics")
    if check_source:
        from fgw_topk_test.state import verify_run as verify_original
        original, original_frozen = verify_original(context["source_root"], check_cache=False)
        if original_frozen != frozen or original["signature"] != context["source_signature"]:
            raise ValueError("Original completed selection changed")
    done, status = read_json(output / "completed.json"), read_json(output / "status.json")
    required = {"run_manifest.json", "frozen_selection.json", "settings.json", "status.json",
                "evaluation_index_test.json", "candidate_protocol.json", "protocol/cross_combination_candidates.csv",
                "source_protection_before.json", "source_protection_after.json"}
    if (done.get("status") != "completed" or done.get("signature") != manifest["signature"] or
            done.get("new_evaluations") != TOP_K or status.get("status") != "completed"):
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
    if (index.get("ranked_variants") != names or index.get("specs") != specs or
            index.get("signature") != manifest["signature"] or
            index.get("paths") != evaluation_paths(output, names) or
            [r["variant"] for r in index.get("summary_rows", [])] != names or
            set(index.get("completed_sha256", {})) != set(names)):
        raise ValueError("Cross completion index differs from valid order")
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
