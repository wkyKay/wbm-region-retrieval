"""Read-only G/J source loading, restricted to the requested evaluation split.

The caller must freeze its new valid selection before requesting ``split=test``.
Only the original B1 valid selection and the requested split's caches/results are
opened here. No F feature matrix, regional scorer, encoder or tensor voting runs.
"""
from pathlib import Path

import numpy as np

from b1_structure_fusion import pipeline as old_pipeline, run as old_run
from experiments.proposed.f_global_fusion import pipeline as fusion
from experiments.proposed.f_segmentation_geometry import pipeline as geo
from wbm_retrieval.common import digest, file_hash, read_json
from wbm_retrieval.data import Protocol, source_fingerprint

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "artificial/b1_structure_all_20261117T193244+0800"
PRODUCT_SOURCE = ROOT / "artificial/b1_ablation_20261118T190927+0800"
GLOBAL_CONTROL_SOURCE = ROOT / "artifacts/f_global_fusion_v1"
J_STRENGTH = 2.0


def _path(value):
    return (ROOT / Path(value).expanduser()).resolve()


def _bind(path, binding, expected=None):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"Missing frozen source file: {path}")
    actual = file_hash(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Modified frozen source file: {path}")
    if str(path) in binding and binding[str(path)] != actual:
        raise ValueError(f"Source changed while being loaded: {path}")
    binding[str(path)] = actual
    return actual


def _sealed(folder, signature, required, binding):
    """Check and bind a completion marker without accepting escaping paths."""
    folder = Path(folder).resolve()
    marker = folder / "completed.json"
    _bind(marker, binding)
    done = read_json(marker)
    files = done.get("files")
    if (done.get("signature") != signature or not isinstance(files, dict)
            or not set(required).issubset(files)):
        raise ValueError(f"Incomplete or incompatible frozen source: {folder}")
    for name, expected in files.items():
        relative = Path(name)
        item = (folder / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or folder not in item.parents:
            raise ValueError("Source completion path escapes its directory")
        _bind(item, binding, expected)
    return done


def validate_arrays(global_ids, global_features, descriptors, gallery):
    """Validate identity and numerical integrity without changing saved arrays."""
    ids, features, gallery = np.asarray(global_ids), np.asarray(global_features), np.asarray(gallery)
    fusion.validate_features(ids, features, gallery)
    if features.shape[1] != 512:
        raise ValueError("Expected the frozen WaPIRL h512 representation")
    required = {"sample_ids", "joint128", "empty"}
    if not required.issubset(descriptors):
        raise ValueError("Missing original whole-wafer J descriptor arrays")
    dids = np.asarray(descriptors["sample_ids"])
    joint, empty = np.asarray(descriptors["joint128"]), np.asarray(descriptors["empty"])
    if (dids.dtype.kind not in "iu" or not np.array_equal(dids, gallery)
            or joint.shape != (len(gallery), 128) or joint.dtype.kind != "f"
            or not np.isfinite(joint).all() or np.any(joint < 0)
            or empty.shape != (len(gallery),) or empty.dtype != np.bool_):
        raise ValueError("Invalid J shapes, IDs, probabilities or empty flags")
    mass = joint.sum(axis=1)
    if (not np.array_equal(mass == 0., empty)
            or not np.allclose(mass[~empty], 1., rtol=0., atol=1e-10)):
        raise ValueError("J probability mass and empty flags disagree")


def _source(source, split, device, binding):
    _bind(source / "run_manifest.json", binding)
    manifest = fusion.signed_manifest(source / "run_manifest.json")
    _bind(source / "status.json", binding)
    status = read_json(source / "status.json")
    if (manifest.get("version") != "b1_structure_fusion_v1" or manifest.get("stage") != "all"
            or status.get("status") != "completed" or status.get("test_evaluated") is not True):
        raise ValueError("Expected the completed original all-stage B1 source")
    context = manifest["context"]
    if context.get("code") != old_run.code_identity() or context.get("device") != device:
        raise ValueError("Original B1 scientific code or structural backend changed")
    for name, expected in context["code"].items():
        _bind(ROOT / name, binding, expected)
    _bind(source / "selection.json", binding)
    selection = old_pipeline.load_selection(source / "selection.json", context["settings"], context)
    selected = selection["selections"]["J"]["selected"]
    if selected["variant"] != "J_2" or selected["strength"] != J_STRENGTH:
        raise ValueError("G/J fusion requires the original frozen strength-2 J source")
    # load_selection verifies all source valid artifacts; retain their hashes in
    # the new provenance without reading any historical test artifact on valid.
    for name, expected in selection["completed_sha256"].items():
        marker = source / "valid" / name / "completed.json"
        _bind(marker, binding, expected)
        for relative, sha in read_json(marker)["files"].items():
            binding[str((marker.parent / relative).resolve())] = sha
    input_path = source / f"inputs_{split}.json"
    _bind(input_path, binding)
    source_inputs = fusion.signed_manifest(input_path)
    expected_selection = selection["signature"] if split == "test" else None
    if (source_inputs.get("context") != context or source_inputs.get("split") != split
            or source_inputs.get("selection_signature") != expected_selection):
        raise ValueError("Source split inputs differ from frozen B1 context/selection")
    index_path, verification_path = source / f"evaluation_index_{split}.json", source / f"verification_{split}.json"
    _bind(index_path, binding)
    _bind(verification_path, binding)
    index, verification = read_json(index_path), read_json(verification_path)
    if index.get("signature") != source_inputs["signature"] or index.get("split") != split:
        raise ValueError("Original B1 split index is incompatible with its inputs")
    # Verify the relevant original controls, not merely a running-status string.
    for name, spec in (("J_2", {"family": "J", "strength": 2.}),
                       ("J_only", {"family": "J_only", "strength": 1.})):
        signature = digest({"evaluation": source_inputs["signature"], "variant": name, "spec": spec})
        if (index.get("variants", {}).get(name) != spec
                or index.get("variant_signatures", {}).get(name) != signature):
            raise ValueError("Original B1 control has an incompatible scoring specification")
        folder = source / split / name
        _sealed(folder, signature, {"run_identity.json", "evaluation_manifest.json", "metrics.json",
                "coverage_metrics.json", "per_query.csv", "coverage_per_query.csv", "rankings.csv"}, binding)
        marker_hash = binding[str((folder / "completed.json").resolve())]
        checked = verification.get(name, {})
        count = 500 if split == "valid" else 7199
        if (index.get("completed_sha256", {}).get(name) != marker_hash
                or checked.get("status") != "verified" or checked.get("split") != split
                or checked.get("variant") != name or checked.get("num_queries") != count
                or checked.get("ranking_rows") != count * max(context["settings"]["ks"])
                or checked.get("completion_sha256") != marker_hash
                or checked.get("ranking_sha256") != binding[str((folder / "rankings.csv").resolve())]):
            raise ValueError("Source B1 controls lack matching completed verification")
    return manifest, source_inputs, selection


def _product_control(source, split, pool, device, binding):
    """Bind the completed historical G*exp(-2D_J) result for replay."""
    from b1_ablation import scoring as ablation_scoring
    root = _path(PRODUCT_SOURCE)
    for name in ("run_manifest.json", "status.json", "source_protection_after.json",
                 f"inputs_{split}.json", f"evaluation_index_{split}.json", f"verification_{split}.json"):
        _bind(root / name, binding)
    manifest = fusion.signed_manifest(root / "run_manifest.json")
    context = manifest["context"]
    settings = ablation_scoring.validate_settings(context["plan"]["settings"])
    if (manifest.get("version") != "b1_ablation_v1" or context["plan"].get("mode") != "all"
            or settings != ablation_scoring.NO_LOCAL_ANGLE_SETTINGS
            or _path(context["plan"]["source_root"]) != source
            or context.get("source_manifest_sha256") != binding[str((source / "run_manifest.json").resolve())]
            or context["plan"].get("device") != device
            or read_json(root / "status.json").get("status") != "completed"
            or read_json(root / "source_protection_after.json").get("status") != "unchanged"):
        raise ValueError("Historical product control is incomplete or belongs to another B1 source")
    inp = fusion.signed_manifest(root / f"inputs_{split}.json")
    specs = ablation_scoring.specs(split, settings)
    if (inp.get("parent_signature") != manifest["signature"] or inp.get("specs") != specs
            or inp.get("provenance", {}).get("protocol_signature") != pool.protocol_signature):
        raise ValueError("Historical product control has different inputs/protocol")
    name = "no_local_angle"
    spec = specs[name]
    signature = digest({"evaluation": inp["signature"], "variant": name, "spec": spec,
                        "split": split, "settings": settings, "device": device,
                        "retrieval_protocol_signature": pool.protocol_signature})
    folder = root / split / name
    _sealed(folder, signature, {"evaluation_manifest.json", "run_identity.json", "metrics.json",
            "coverage_metrics.json", "per_query.csv", "coverage_per_query.csv", "rankings.csv"}, binding)
    index = read_json(root / f"evaluation_index_{split}.json")
    verification = read_json(root / f"verification_{split}.json").get(name, {})
    evaluation = read_json(folder / "evaluation_manifest.json")
    marker_hash = binding[str((folder / "completed.json").resolve())]
    if (index.get("variants") != specs or index.get("completed_sha256", {}).get(name) != marker_hash
            or evaluation.get("spec") != spec or evaluation.get("split") != split
            or evaluation.get("parent_signature") != inp["signature"]
            or verification.get("status") != "verified" or verification.get("num_queries") != len(pool)
            or verification.get("completion_sha256") != marker_hash
            or verification.get("ranking_sha256") != binding[str((folder / "rankings.csv").resolve())]):
        raise ValueError("Historical product control lacks matching completed verification")
    return folder


def _global_control(split, pool, global_identity, sources, binding):
    """Use the historical alpha=1 F/G fusion endpoint for original G arithmetic."""
    root = _path(GLOBAL_CONTROL_SOURCE)
    for name in ("fusion_manifest.json", "status.json", f"{split}_manifest.json"):
        _bind(root / name, binding)
    manifest = fusion.signed_manifest(root / "fusion_manifest.json")
    split_manifest = fusion.signed_manifest(root / f"{split}_manifest.json")
    if (manifest.get("version") != "cc8_wapirl_score_fusion_v1"
            or manifest.get("sources") != sources
            or manifest.get("formula") != "(1-alpha)*F+alpha*WaPIRL"
            or manifest.get("tie_seed") != 2026
            or 1. not in manifest.get("alphas", [])
            or read_json(root / "status.json").get("status") != "completed"
            or split_manifest.get("run_signature") != manifest["signature"]
            or split_manifest.get("split") != split
            or split_manifest.get("global") != global_identity
            or 1. not in split_manifest.get("weights", [])):
        raise ValueError("Historical pure-G fusion endpoint has incompatible sources or settings")
    for name, expected in manifest["code"].items():
        _bind(ROOT / name, binding, expected)
    if split == "valid":
        if manifest.get("wapirl_valid") != global_identity or split_manifest.get("selection_signature") is not None:
            raise ValueError("Historical valid G endpoint has different global features")
    else:
        _bind(root / "selection.json", binding)
        choice = fusion.signed_manifest(root / "selection.json")
        _bind(root / "valid_manifest.json", binding)
        valid = fusion.signed_manifest(root / "valid_manifest.json")
        if (split_manifest.get("selection_signature") != choice["signature"]
                or choice.get("selection_split") != "valid"
                or choice.get("valid_signature") != valid["signature"]
                or valid.get("run_signature") != manifest["signature"]
                or valid.get("global") != manifest.get("wapirl_valid")):
            raise ValueError("Historical test G endpoint has no matching frozen valid selection")
    folder = root / split / "alpha_1.0"
    _sealed(folder, split_manifest["signature"], {"metrics.json", "coverage_metrics.json", "per_query.csv",
            "coverage_per_query.csv", "rankings.csv"}, binding)
    for family, filename in (("exact", "metrics.json"), ("coverage", "coverage_metrics.json")):
        report = read_json(folder / filename)
        metadata = report.get("metadata", {})
        protocol = report.get("retrieval_protocol_signature", metadata.get("retrieval_protocol_signature"))
        if (report.get("split") != split or report.get("num_queries") != len(pool)
                or report.get("metric_version") != family + "_v1"
                or metadata.get("signature") != split_manifest["signature"]
                or metadata.get("alpha_global") != 1.
                or metadata.get("candidates_scored") != "full_pool"
                or metadata.get("tie_seed") != 2026
                or (protocol != pool.protocol_signature if family == "exact"
                    else protocol not in (None, pool.protocol_signature))):
            raise ValueError("Historical pure-G report differs from the frozen protocol/endpoint")
    return folder


def load_inputs(source=DEFAULT_SOURCE, split="valid", device="cuda"):
    """Load one split, preserving saved G arithmetic and original whole-map J.

    Test gating belongs to the new runner: it must verify its sealed valid
    selection before invoking this function for test. The source B1 valid choice
    is independently checked here for both splits.
    """
    if split not in ("valid", "test") or device not in ("cpu", "cuda"):
        raise ValueError("Expected valid/test and an explicit cpu/cuda backend")
    source, binding = _path(source), {}
    manifest, source_inputs, selection = _source(source, split, device, binding)
    context = manifest["context"]
    defaults = old_run.parser().parse_args([])
    training_root = _path(defaults.source_root)
    config, suite, checkpoint, kind, training_inputs = geo.load_source(training_root, "cc8", None, "cpu")
    del checkpoint
    old_run.assert_frozen(config, suite, kind, context["settings"])
    sources = source_fingerprint(config)
    if sources != context["sources"] or geo.new_identity(config, training_inputs) != context["geometry"]:
        raise ValueError("Data/protocol/training provenance differs from original B1")
    for name, expected in training_inputs.items():
        _bind(training_root / "cc8" / name, binding, expected)
    for name, expected in sources.items():
        if name.startswith("retrieval_protocol/"):
            item = Path(config["retrieval_protocol_dir"]) / name.split("/", 1)[1]
        elif name.startswith("protocol_code/"):
            item = ROOT / "wbm_retrieval" / name.split("/", 1)[1]
        elif name == "duplicates_manifest":
            item = Path(config["deduplication_metadata"]).parent / "duplicates.csv"
        else:
            item = Path(config[name])
        _bind(item, binding, expected)
    protocol = Protocol(config)
    pool = getattr(protocol.retrieval, split)
    expected_size = (500, 3599, 3598) if split == "valid" else (7199, 7199, 7198)
    if (len(pool), len(pool.gallery), pool.candidates_per_query) != expected_size:
        raise ValueError("Expected the complete frozen retrieval_v1 pool")
    with np.load(config["data_file"], allow_pickle=False) as archive:
        labels = archive["arr_1"]
    bundle = fusion.bundle_for(protocol, labels)
    global_root = _path(getattr(defaults, f"wapirl_{split}_root"))
    ids, features, global_identity = fusion.load_global(global_root, split, pool, sources,
                context["global"]["checkpoint"], context["global"]["inference_recipe"])
    if global_identity != source_inputs["global_features"]:
        raise ValueError("Frozen global features differ from source B1")
    _bind(global_root / "evaluation_manifest.json", binding, global_identity["manifest_sha256"])
    _bind(global_root / "completed.json", binding, global_identity["completed_sha256"])
    for name, expected in read_json(global_root / "completed.json")["files"].items():
        # load_global already validated containment and every file's hash.
        binding[str((global_root / name).resolve())] = expected
    descriptor_root = source / "descriptors" / split
    descriptor_signature = digest({"sources": sources, "split": split,
                                   "settings": context["settings"], "code": context["code"]})
    _sealed(descriptor_root, descriptor_signature, {f"{split}.npz", "preparation.json"}, binding)
    with np.load(descriptor_root / f"{split}.npz", allow_pickle=False) as archive:
        descriptors = dict(archive)
    validate_arrays(ids, features, descriptors, pool.gallery)
    product_folder = _product_control(source, split, pool, device, binding)
    global_control_folder = _global_control(split, pool, global_identity, sources, binding)
    control_folders = {"product": str(product_folder), "J_only": str(source / split / "J_only"),
                       "G_only": str(global_control_folder)}
    for folder in control_folders.values():
        for name in ("metrics.json", "coverage_metrics.json", "per_query.csv", "coverage_per_query.csv", "rankings.csv"):
            if str((Path(folder) / name).resolve()) not in binding:
                raise ValueError(f"Historical control report/ranking is not bound: {folder}/{name}")
    return {"bundle": bundle, "protocol": protocol, "pool": pool, "global_ids": ids,
            "global_features": features, "descriptors": descriptors, "config": config,
            "source_run": source, "source_binding": dict(sorted(binding.items())),
            "source_metadata": {"split": split, "source_manifest_sha256": file_hash(source / "run_manifest.json"),
                "source_inputs_sha256": file_hash(source / f"inputs_{split}.json"),
                "source_selection_sha256": file_hash(source / "selection.json"),
                "source_selection_signature": selection["signature"],
                "descriptor_signature": descriptor_signature, "global_features": global_identity,
                "global_root": str(global_root), "descriptor_root": str(descriptor_root),
                "wapirl_reference": str(global_root / split),
                "g_control_definition": "alpha_global=1 endpoint; (1-alpha)*F+alpha*G; original F/G fusion arithmetic",
                "control_folders": control_folders, **control_folders,
                "protocol_signature": pool.protocol_signature, "j_strength": J_STRENGTH,
                "device": device, "regional_scoring_used": False, "training_required": False}}
