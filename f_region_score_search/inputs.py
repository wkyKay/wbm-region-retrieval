"""Strictly read-only reuse of the frozen regional-J valid winner and cache.

Never verify the old run root recursively: that would read its test artifacts.
The mature source loader may read the complete original NPZ solely to audit
frozen deduplication/protocol identities. This module only extracts arr_1 for
valid labels, and never loads old test reports/features or trains a model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from f_region_structure import pipeline as old_pipeline
from f_region_structure import scoring as old_scoring
from f_region_structure import selection as old_selection
from f_region_structure import sources as old_sources
from f_region_structure.state import read_signed
from f_region_structure.verify import REPORT_FILES
from wbm_retrieval.common import digest, file_hash, read_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "artificial/f_region_structure_20261120T120331+0800"
WINNER = "replace_radial_all_points_l0p5"
WINNER_SPEC = {"family": "replace_radial", "descriptor": "all_points",
               "distribution": "joint", "strength": .5, "area": True}
KS = [1, 5, 10, 20, 50, 100, 500]
CACHE_FILES = {"audit.json", "h512.npy", "regions.npz", "source_binding.json"}
EVALUATION_FILES = set(REPORT_FILES) | {"run_identity.json", "verification.json"}


def _bind(path, binding, expected=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or symlinked frozen source: {path}")
    value = file_hash(path)
    if expected is not None and value != expected:
        raise ValueError(f"Frozen source hash changed: {path}")
    key = str(path.resolve())
    if key in binding and binding[key] != value:
        raise ValueError(f"Source changed during input loading: {path}")
    binding[key] = value
    return value


def _sealed(folder, signature, expected_files, binding):
    folder = Path(folder)
    marker_path = folder / "completed.json"
    _bind(marker_path, binding)
    marker = read_json(marker_path)
    if (marker.get("status") != "completed" or marker.get("signature") != signature or
            not isinstance(marker.get("files"), dict) or set(marker["files"]) != set(expected_files)):
        raise ValueError(f"Incomplete or different frozen completion evidence: {folder}")
    # Exact expected basenames prevent traversal or a hidden test dependency.
    for name, expected in marker["files"].items():
        if Path(name).name != name:
            raise ValueError("Completion file must be an allowed local basename")
        _bind(folder / name, binding, expected)
    return marker


def _integer_array(value, shape, name, *, unique=False):
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind not in "iu" or np.any(value < 0):
        raise ValueError(f"Invalid original-ID/region integer array: {name}")
    if unique and len(np.unique(value)) != len(value):
        raise ValueError(f"Duplicate source identities: {name}")
    return value


def _probabilities(value, shape, name, *, empty=False):
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError(f"Invalid probability descriptor: {name}")
    masses = value.sum(1)
    valid = np.isclose(masses, 1., rtol=0, atol=1e-10)
    if empty:
        valid |= masses == 0
    if not valid.all():
        raise ValueError(f"Invalid probability mass: {name}")
    return value


def _check_cache(cache, protocol):
    ids = _integer_array(cache["map_ids"], (3599,), "map_ids", unique=True)
    valid_ids = np.asarray(protocol.ids["valid"])
    pool = protocol.retrieval.valid
    if (pool.split_name != "valid" or len(pool) != 500 or len(pool.gallery) != 3599
            or pool.candidates_per_query != 3598 or not np.array_equal(ids, valid_ids)
            or set(map(int, pool.gallery)) != set(map(int, ids)) or not set(map(int, pool)) <= set(map(int, ids))):
        raise ValueError("Cache/protocol must be frozen valid 500 queries by 3598 candidates")
    offsets = _integer_array(cache["offsets"], (3600,), "offsets")
    counts = np.diff(offsets.astype(np.int64))
    if offsets[0] != 0 or np.any(counts < 0) or np.any(counts > 4):
        raise ValueError("Invalid frozen Top-4 offsets")
    n = int(offsets[-1])
    areas = _integer_array(cache["area"], (n,), "area")
    original = _integer_array(cache["original_count"], (n,), "original_count")
    retained = _integer_array(cache["retained_count"], (n,), "retained_count")
    owners = _integer_array(cache["owners"], (n,), "owners")
    if (not n or np.any(areas < 5) or not np.array_equal(areas, original) or np.any(retained > areas)
            or not np.array_equal(owners, np.repeat(ids, counts))):
        raise ValueError("Region areas/owners differ from original frozen token ordering")
    h = np.asarray(cache["features"])
    if h.shape != (n, 512) or h.dtype.kind != "f" or not np.isfinite(h).all():
        raise ValueError("Frozen h512 features do not align with region offsets")
    norms = np.linalg.norm(h.astype(np.float64), axis=1)
    if np.any((norms != 0) & ~np.isclose(norms, 1., rtol=1e-5, atol=1e-6)):
        raise ValueError("Frozen h512 features are not normalized")
    radial = _probabilities(cache["histograms"], (n, 16), "radial")
    if not np.array_equal(radial, cache["radial_hist16"]):
        raise ValueError("Duplicated radial descriptors disagree")
    relative = np.asarray(cache["relative_area"])
    if relative.shape != (n,) or not np.isfinite(relative).all() or np.any((relative <= 0) | (relative > 1)):
        raise ValueError("Invalid original-mask relative area")
    for suffix, count, may_empty in (("", retained, True), ("_all", original, False)):
        joint = _probabilities(cache["joint"+suffix], (n, 128), "joint"+suffix, empty=may_empty)
        shape = _probabilities(cache["shape"+suffix], (n, 16), "shape"+suffix, empty=may_empty)
        flags = np.asarray(cache["empty"+suffix])
        if (flags.shape != (n,) or flags.dtype != bool or not np.array_equal(flags, count == 0)
                or not np.array_equal(joint.sum(1) == 0, flags)
                or not np.allclose(shape, joint.reshape(n, 8, 4, 4).sum(1).reshape(n, 16), rtol=0, atol=1e-12)):
            raise ValueError("Structure marginal/count/empty flags disagree")
    quality = np.asarray(cache["quality"])
    if quality.shape != (n,) or not np.isfinite(quality).all() or not np.allclose(quality, retained/areas, rtol=0, atol=1e-12):
        raise ValueError("Region quality differs from frozen point counts")


def _bind_model_and_protocol(config, source_root, identity, binding):
    for name, expected in identity["source_files"].items():
        path = (source_root / "cc8" / name).resolve()
        if not path.is_relative_to(source_root):
            raise ValueError("Frozen model dependency escaped its training source")
        _bind(path, binding, expected)
    direct = {"data_file": Path(config["data_file"]), "split_manifest": Path(config["split_manifest"]),
              "deduplication_metadata": Path(config["deduplication_metadata"]),
              "duplicates_manifest": Path(config["deduplication_metadata"]).parent / "duplicates.csv"}
    for name, expected in identity["source_fingerprint"].items():
        if name in direct:
            path = direct[name]
        elif name.startswith("retrieval_protocol/"):
            leaf = name.split("/", 1)[1]
            if Path(leaf).name != leaf:
                raise ValueError("Invalid protocol dependency path")
            path = Path(config["retrieval_protocol_dir"]) / leaf
        elif name.startswith("protocol_code/"):
            leaf = name.split("/", 1)[1]
            if leaf not in {"protocol_adapter.py", "data_protocol.py", "retrieval_protocol.py"}:
                raise ValueError("Unexpected protocol code dependency")
            path = ROOT / "wbm_retrieval" / leaf
        else:
            raise ValueError("Unexpected frozen source fingerprint: " + name)
        _bind(path, binding, expected)


def load(source_run: Path = DEFAULT_SOURCE) -> dict:
    source_run = Path(source_run).expanduser().resolve()
    binding = {}
    for name in ("run_manifest.json", "selection.json", "evaluation_index_valid.json"):
        _bind(source_run / name, binding)
    manifest = read_signed(source_run / "run_manifest.json")
    selected = read_signed(source_run / "selection.json")
    context = manifest.get("context", {})
    old_signature = digest(context)
    if (manifest.get("version") != "f_region_structure_v1" or
            selected.get("context_signature") != old_signature or selected.get("synthetic") is not False or
            selected.get("winner") != WINNER or selected.get("selected_spec") != WINNER_SPEC):
        raise ValueError("Expected the frozen real regional-J valid winner and context")
    index = read_json(source_run / "evaluation_index_valid.json")
    if (selected.get("valid_index_sha256") != binding[str((source_run / "evaluation_index_valid.json").resolve())] or
            index.get("context_signature") != old_signature or index.get("split") != "valid" or
            index.get("specs") != old_scoring.frozen_grid()):
        raise ValueError("Selected valid index binding differs from the frozen full grid")
    expected_selection = old_selection.select_best(index["results"], split="valid")
    if any(selected.get(key) != value for key, value in expected_selection.items()):
        raise ValueError("Frozen valid winner does not follow its integer selection rule")

    source_root = Path(manifest["source_root"]).expanduser().resolve()
    config, _suite, _checkpoint, protocol, fresh = old_sources.load(source_root)
    if fresh != context.get("source"):
        raise ValueError("Frozen original model/data/protocol differ from the source run")
    _bind_model_and_protocol(config, source_root, fresh, binding)

    baseline = source_run / "valid" / WINNER
    evaluation_signature = digest({"context": old_signature, "split": "valid", "variant": WINNER, "spec": WINNER_SPEC})
    marker = _sealed(baseline, evaluation_signature, EVALUATION_FILES, binding)
    old_result = index["results"][WINNER]
    if old_result["completed_sha256"] != binding[str((baseline / "completed.json").resolve())]:
        raise ValueError("Valid winner completion differs from the selected index")
    evidence = read_json(baseline / "verification.json")
    eval_manifest = read_json(baseline / "evaluation_manifest.json")
    if (eval_manifest.get("signature") != evaluation_signature or eval_manifest.get("split") != "valid" or
            eval_manifest.get("variant") != WINNER or eval_manifest.get("spec") != WINNER_SPEC or
            eval_manifest.get("ks") != KS or eval_manifest.get("tie_seed") != 2026 or
            eval_manifest.get("retrieval_protocol_signature") != protocol.retrieval.valid.protocol_signature):
        raise ValueError("Valid winner evaluation identity differs")
    fixed = {"status": "verified", "signature": evaluation_signature, "split": "valid", "variant": WINNER,
             "ks": KS, "tie_seed": 2026, "num_queries": 500, "ranking_rows": 250000,
             "full_candidates_checked": 1799000, "all_queries_full_candidate_topk_exact": True}
    if any(evidence.get(key) != value for key, value in fixed.items()):
        raise ValueError("Valid winner verification is incomplete or different")
    for field in ("max_metric_absolute_error", "max_saved_score_absolute_error"):
        value = evidence.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1e-12:
            raise ValueError("Invalid valid winner verification error")
    if (evidence.get("top10_hits") != old_result["top10_hits"] or
            evidence.get("ranking_sha256") != marker["files"]["rankings.csv"] or
            evidence.get("files") != {name: value for name, value in marker["files"].items() if name != "verification.json"}):
        raise ValueError("Valid winner verification/report hashes disagree")
    for family, filename, index_key in (("exact", "metrics.json", "metrics"),
                                        ("coverage", "coverage_metrics.json", "coverage_metrics")):
        report = read_json(baseline / filename)
        if (report.get("split") != "valid" or report.get("num_queries") != 500 or report.get("ks") != KS or
                report.get("metric_version") != family + "_v1" or
                report.get("metadata", {}).get("signature") != evaluation_signature or
                report.get("metrics") != old_result[index_key]):
            raise ValueError("Valid winner metric report differs from its selected index")
    if not np.isclose(old_result["metrics"]["Precision@10"], evidence["top10_hits"]/5000, rtol=0, atol=1e-12):
        raise ValueError("Valid winner P@10 does not agree with integer hits")

    cache_folder = source_run / "cache" / "valid"
    cache_signature = digest({"context": old_signature, "split": "valid", "cache_version": 1})
    _sealed(cache_folder, cache_signature, CACHE_FILES, binding)
    if (cache_folder / "h_views.npy").exists():
        raise ValueError("A valid cache must not contain unbound training views")
    upstream = read_json(cache_folder / "source_binding.json")
    valid_roots = [(Path(path).expanduser().resolve() / "valid") for path in manifest["feature_roots"]]
    if not isinstance(upstream, dict) or not upstream:
        raise ValueError("Valid cache lacks frozen upstream bindings")
    for path, expected in upstream.items():
        actual = Path(path)
        if not actual.is_absolute() or not any(actual.resolve().is_relative_to(root) for root in valid_roots):
            raise ValueError("Cache dependency is outside the declared valid feature roots; test access is forbidden")
        _bind(actual, binding, expected)
    cache = old_sources.read_cache(cache_folder)
    _check_cache(cache, protocol)
    with np.load(config["data_file"], allow_pickle=False) as archive:
        labels = archive["arr_1"]
    bundle = old_pipeline.bundle_for(labels, protocol)
    return {"config": config, "protocol": protocol, "bundle": bundle, "cache": cache,
            "source_binding": dict(sorted(binding.items())), "old_context_signature": old_signature,
            "baseline_folder": baseline}
