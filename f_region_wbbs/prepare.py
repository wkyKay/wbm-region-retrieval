"""Frozen Top-4 point sets for regional WBBS, without a new CNN forward pass.

One production tensor vote uses every original defect pixel of each wafer.
The two point modes differ only in the existing global denoising mask. Separate
reference audits may repeat voting on the caller's fixed representative IDs.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import tempfile

import numpy as np

from b1_structure_fusion.descriptors import load_settings, soft_histogram
from experiments.proposed.f_segmentation.proposals import segment
from experiments.proposed.f_segmentation.tensor_voting import retained_points, tensor_saliencies, tensor_vote
from f_region_structure.state import now, record, seal, verified
from fgw_fusion.wbbs_backend import original_module
from wbm_retrieval.common import digest, file_hash, read_json

POINT_MODES = ("all_points", "wbbs_denoised")
DESCRIPTOR_FIELDS = ("joint", "joint_all", "shape", "shape_all", "original_count",
                     "retained_count", "quality", "empty", "empty_all",
                     "radial_hist16", "relative_area")
REQUIRED_FIELDS = (*DESCRIPTOR_FIELDS, "map_ids", "offsets", "owners", "area", "histograms")
POINT_FILES = {f"{mode}_{field}.npy" for mode in POINT_MODES
               for field in ("offsets", "descriptors", "coordinates")}
CACHE_FILES = POINT_FILES | {"regions.npz", "h512.npy", "region_labels.npy",
    "all_points_global_keep.npy", "region_trace.jsonl", "audit.json", "metadata.json", "source_binding.json"}
TOLERANCE = 1e-14


def _settings():
    settings = load_settings()
    if any(settings[k] != v for k, v in {"sigma_d": 9.0, "noise_threshold": .32,
                                       "block_size": 256, "bins": [8, 4, 4]}.items()):
        raise ValueError("Regional WBBS requires the unchanged global tensor-voting recipe")
    return settings


def cache_signature(context_signature, split):
    if not isinstance(context_signature, str) or not context_signature or split not in ("valid", "test"):
        raise ValueError("Expected a context signature and valid or selection-guarded test split")
    return digest({"context_signature": context_signature, "split": split, "cache_version": 1,
                   "point_modes": list(POINT_MODES), "descriptor_settings": _settings()})


def _source_check(binding):
    if not isinstance(binding, dict) or not binding:
        raise ValueError("Frozen source binding must be nonempty")
    for name, expected in binding.items():
        path = Path(name)
        if not path.is_absolute() or path.is_symlink() or not path.is_file() or file_hash(path) != expected:
            raise ValueError("Frozen source changed or is invalid: " + str(name))


def _fingerprint(value):
    array = np.asarray(value)
    return {"dtype": array.dtype.str, "shape": list(array.shape),
            "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}


def _validate_inputs(maps, protocol, old_cache, labels, split):
    if split not in ("valid", "test"):
        raise ValueError("Only valid or selection-guarded test preparation is supported")
    maps, labels = np.asarray(maps), np.asarray(labels)
    ids, offsets = np.asarray(protocol.ids[split]), np.asarray(old_cache["offsets"])
    if (maps.ndim != 3 or not all(maps.shape) or ids.ndim != 1 or ids.dtype.kind not in "iu"
            or not len(ids) or np.any(ids < 0) or np.any(ids >= len(maps))
            or len(np.unique(ids)) != len(ids) or not np.array_equal(old_cache["map_ids"], ids)):
        raise ValueError("Original map IDs must match the requested frozen split")
    if (offsets.shape != (len(ids)+1,) or offsets.dtype.kind not in "iu" or offsets[0] != 0
            or np.any(np.diff(offsets.astype(np.int64)) < 0)
            or np.any(np.diff(offsets.astype(np.int64)) > 4)):
        raise ValueError("Invalid frozen Top-4 region offsets")
    n = int(offsets[-1])
    if (labels.shape != (len(ids), *maps.shape[1:]) or labels.dtype.kind not in "iu"
            or np.any(labels < 0) or np.any(labels > 4)):
        raise ValueError("Frozen labels must be original-grid integer Top-4 masks")
    features = np.asarray(old_cache["features"])
    if features.shape != (n, 512) or features.dtype.kind != "f" or not np.isfinite(features).all():
        raise ValueError("Frozen h512 feature shape or values are invalid")
    for key in REQUIRED_FIELDS:
        value = np.asarray(old_cache[key])
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError("Invalid frozen regional field: " + key)
        if key not in ("map_ids", "offsets") and len(value) != n:
            raise ValueError("Frozen regional field row count differs: " + key)
    for key, columns in (("joint", 128), ("joint_all", 128), ("shape", 16),
                         ("shape_all", 16), ("radial_hist16", 16), ("histograms", 16)):
        if np.asarray(old_cache[key]).shape != (n, columns):
            raise ValueError("Frozen descriptor dimensions differ: " + key)
    area, kept = np.asarray(old_cache["area"]), np.asarray(old_cache["retained_count"])
    if (area.dtype.kind not in "iu" or kept.dtype.kind not in "iu" or np.any(area < 5)
            or np.any(kept < 0) or np.any(kept > area)
            or not np.array_equal(area, old_cache["original_count"])
            or not np.array_equal(old_cache["owners"], np.repeat(ids, np.diff(offsets)))
            or not np.array_equal(old_cache["histograms"], old_cache["radial_hist16"])):
        raise ValueError("Frozen area, ownership, or radial aliases disagree")
    return ids, offsets


def _point_fields(raw, labels, settings):
    """Production fields; no region-local voting, normalization, or threshold."""
    raw, labels = np.asarray(raw), np.asarray(labels)
    if (raw.ndim != 2 or not all(raw.shape) or not np.isin(raw, [0, 1, 2, 3]).all()
            or not np.any((raw == 1) | (raw == 2))):
        raise ValueError("Expected an original wafer with valid support")
    expected, _ = segment(raw, "cc8", min_area=5, top_k=4)
    if labels.dtype.kind not in "iu" or labels.shape != raw.shape or not np.array_equal(labels, expected):
        raise ValueError("Frozen masks differ from original cc8 min-area-5 Top-4")
    coordinates = np.argwhere(raw == 2).astype(np.int64)
    points = coordinates.astype(np.float64)
    support = np.argwhere((raw == 1) | (raw == 2)).astype(np.float64)
    center = (support.min(0) + support.max(0)) / 2
    radius = float(np.linalg.norm(support-center, axis=1).max())
    if len(points):
        saliencies = tensor_saliencies(tensor_vote(points, settings["sigma_d"], settings["block_size"]))
        maxima = saliencies.max(0)
        scaled = np.divide(saliencies, maxima, out=np.zeros_like(saliencies), where=maxima > 0)
        rho = np.linalg.norm(points-center, axis=1)/radius if radius > 0 else np.zeros(len(points))
        descriptors = np.column_stack((scaled, rho))
        keep = retained_points(saliencies, settings["noise_threshold"])
    else:
        maxima, descriptors, keep = np.zeros(2), np.empty((0, 3)), np.zeros(0, dtype=bool)
    return {"coordinates": coordinates, "descriptors": descriptors, "keep": keep,
            "tokens": labels[tuple(coordinates.T)], "saliency_max": maxima,
            "wafer_area": len(support), "regions": int(labels.max())}


def _reference_audit(raw, labels, fields):
    """Independent original WBBS encoding, only for explicitly supplied IDs."""
    expected = original_module().encode_wafer(raw)
    keep = fields["keep"]
    actual_points, actual = fields["coordinates"][keep], fields["descriptors"][keep]
    if not np.array_equal(actual_points, expected.points) or expected.original_count != len(keep):
        raise ValueError("Original WBBS retained-point identity differs")
    error = float(np.max(np.abs(actual-expected.descriptors), initial=0.))
    if error > TOLERANCE:
        raise ValueError("Original WBBS global normalized descriptors differ")
    # Explicitly re-slice the independent reference using the original masks.
    reference_tokens = labels[tuple(expected.points.astype(np.int64).T)]
    for token in range(1, fields["regions"]+1):
        wanted = reference_tokens == token
        member = (fields["tokens"] == token) & keep
        if not np.array_equal(fields["coordinates"][member], expected.points[wanted]):
            raise ValueError("Original WBBS regional subset identity differs")
        if not np.allclose(fields["descriptors"][member], expected.descriptors[wanted], rtol=0, atol=TOLERANCE):
            raise ValueError("Original WBBS regional subset descriptor differs")
    return {"max_descriptor_error": error, "points": int(keep.sum()),
            "regions": fields["regions"], "bitwise_descriptors_identical": actual.tobytes() == expected.descriptors.tobytes()}


def _job(job):
    raw, labels, settings, audit = job
    fields = _point_fields(raw, labels, settings)
    rows = []
    for token in range(1, fields["regions"]+1):
        member = fields["tokens"] == token
        descriptors, coordinates = fields["descriptors"][member], fields["coordinates"][member]
        keep = fields["keep"][member]
        jvalues = descriptors[:, [2, 0, 1]]
        all_joint = soft_histogram(jvalues, tuple(settings["bins"])).reshape(-1)
        joint = soft_histogram(jvalues[keep], tuple(settings["bins"])).reshape(-1)
        area = len(descriptors)
        radial = np.histogram(np.clip(descriptors[:, 2], 0, 1), bins=np.linspace(0, 1, 17))[0]/area
        row = {"descriptors": descriptors, "coordinates": coordinates, "keep": keep,
               "source_point_indices": np.flatnonzero(member), "joint_all": all_joint, "joint": joint,
               "radial_hist16": radial, "relative_area": area/fields["wafer_area"]}
        rows.append(row)
    return {"rows": rows, "global_original_count": len(fields["keep"]),
            "global_retained_count": int(fields["keep"].sum()), "saliency_max": fields["saliency_max"],
            "reference_audit": _reference_audit(raw, labels, fields) if audit else None}


def _map_jobs(jobs, workers):
    if workers == 0:
        yield from map(_job, jobs)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        pending = deque()
        for _ in range(2*workers):
            job = next(jobs, None)
            if job is not None:
                pending.append(executor.submit(_job, job))
        while pending:
            yield pending.popleft().result()
            job = next(jobs, None)
            if job is not None:
                pending.append(executor.submit(_job, job))


def _arrays(cache):
    return {k: np.asarray(v) for k, v in cache.items()
            if k not in ("features", "region_labels") and isinstance(v, np.ndarray)}


def read_cache(folder, signature=None):
    folder = Path(folder)
    if not verified(folder, signature):
        raise ValueError("Regional WBBS point cache is not sealed")
    marker, metadata = read_json(folder/"completed.json"), read_json(folder/"metadata.json")
    if set(marker["files"]) != CACHE_FILES:
        raise ValueError("Regional WBBS cache has incomplete or unexpected sealed files")
    expected = cache_signature(metadata["context_signature"], metadata["split"])
    if (metadata.get("signature") != expected or marker["signature"] != expected
            or metadata.get("point_modes") != list(POINT_MODES)
            or metadata.get("descriptor_settings") != _settings()
            or metadata.get("source_binding_sha256") != digest(read_json(folder/"source_binding.json"))):
        raise ValueError("Regional WBBS cache metadata identity differs")
    with np.load(folder/"regions.npz", allow_pickle=False) as archive:
        cache = dict(archive)
    if set(cache) != set(metadata["source_arrays"])-{"features", "region_labels"}:
        raise ValueError("Frozen source array inventory differs")
    cache.update(features=np.load(folder/"h512.npy", mmap_mode="r"),
                 region_labels=np.load(folder/"region_labels.npy", mmap_mode="r"))
    for key, expected_array in metadata["source_arrays"].items():
        if _fingerprint(cache[key]) != expected_array:
            raise ValueError("Frozen source array bytes differ: " + key)
    n = len(cache["features"])
    keep = np.load(folder/"all_points_global_keep.npy", mmap_mode="r")
    cache["point_sets"] = {}
    for mode in POINT_MODES:
        pointset = {field: np.load(folder/f"{mode}_{field}.npy", mmap_mode="r")
                    for field in ("offsets", "descriptors", "coordinates")}
        offsets, values, coordinates = (pointset[k] for k in ("offsets", "descriptors", "coordinates"))
        if (offsets.dtype != np.int64 or offsets.shape != (n+1,) or offsets[0] != 0
                or np.any(np.diff(offsets) < 0) or values.dtype != np.float64
                or values.shape != (int(offsets[-1]), 3) or coordinates.dtype != np.int64
                or coordinates.shape != (len(values), 2) or not np.isfinite(values).all()
                or np.any(values < 0) or np.any(values > 1+TOLERANCE)):
            raise ValueError("Invalid regional point offsets, coordinates or descriptors")
        wanted = cache["area"] if mode == "all_points" else cache["retained_count"]
        if not np.array_equal(np.diff(offsets), wanted):
            raise ValueError("Regional point counts differ from frozen descriptor counts")
        cache["point_sets"][mode] = pointset
    allpoints, denoised = (cache["point_sets"][mode] for mode in POINT_MODES)
    if keep.dtype != bool or keep.shape != (len(allpoints["descriptors"]),):
        raise ValueError("Invalid point-level global keep mask")
    region_owners = np.repeat(np.arange(len(cache["map_ids"])), np.diff(cache["offsets"]))
    for region in range(n):
        x, y = allpoints["offsets"][region:region+2]
        u, v = denoised["offsets"][region:region+2]
        coordinates = allpoints["coordinates"][x:y]
        image_index = int(region_owners[region])
        token = region-int(cache["offsets"][image_index])+1
        expected_coordinates = np.argwhere(cache["region_labels"][image_index] == token)
        if not np.array_equal(coordinates, expected_coordinates):
            raise ValueError("Point coordinates differ from the exact original region mask")
        if len(coordinates)>1 and not np.all((np.diff(coordinates[:, 0])>0) | ((np.diff(coordinates[:, 0])==0)&(np.diff(coordinates[:, 1])>0))):
            raise ValueError("Point coordinates must preserve strict original raster order")
        for field in ("coordinates", "descriptors"):
            if not np.array_equal(denoised[field][u:v], allpoints[field][x:y][keep[x:y]]):
                raise ValueError("Denoised points are not the unchanged global-keep subset")
    cache.update(folder=folder, point_keep=keep)
    return cache


def prepare(output, maps, protocol, old_cache, old_labels, context_signature, source_binding,
            workers=4, split="valid", audit_ids=None):
    """Build immutable two-mode point sets; test access is guarded by the runner."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError("workers must be a nonnegative integer")
    ids, offsets = _validate_inputs(maps, protocol, old_cache, old_labels, split)
    audit_ids = np.asarray([] if audit_ids is None else audit_ids)
    if (audit_ids.ndim != 1 or (len(audit_ids) and audit_ids.dtype.kind not in "iu")
            or len(np.unique(audit_ids)) != len(audit_ids) or not np.isin(audit_ids, ids).all()):
        raise ValueError("Audit IDs must be a unique subset of the requested split")
    audit_set = set(map(int, audit_ids))
    _source_check(source_binding)
    output = Path(output).resolve()
    if "folder" in old_cache and output.is_relative_to(Path(old_cache["folder"]).resolve()):
        raise ValueError("New output must not be within a historical cache")
    target = output/"cache"/split
    signature = cache_signature(context_signature, split)
    arrays = _arrays(old_cache)
    source_arrays = {k: _fingerprint(v) for k, v in {**arrays, "features": old_cache["features"],
                                                   "region_labels": old_labels}.items()}
    if verified(target, signature):
        metadata = read_json(target/"metadata.json")
        if (read_json(target/"source_binding.json") != source_binding
                or metadata["source_arrays"] != source_arrays
                or metadata["audit_ids"] != list(map(int, audit_ids))):
            raise ValueError("Requested cache inputs or audit identities changed")
        return read_cache(target, signature)
    if target.exists():
        raise ValueError("Refuse to overwrite an incomplete cache: " + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="."+split+"_attempt_", dir=target.parent))
    started = now().isoformat()
    np.savez_compressed(attempt/"regions.npz", **arrays)
    np.save(attempt/"h512.npy", np.asarray(old_cache["features"]))
    np.save(attempt/"region_labels.npy", np.asarray(old_labels))
    pointsets = {}
    for mode, counts in (("all_points", old_cache["area"]), ("wbbs_denoised", old_cache["retained_count"])):
        point_offsets = np.r_[np.int64(0), np.cumsum(counts, dtype=np.int64)]
        np.save(attempt/f"{mode}_offsets.npy", point_offsets)
        pointsets[mode] = {"offsets": point_offsets}
        for field, columns, dtype in (("descriptors", 3, np.float64), ("coordinates", 2, np.int64)):
            pointsets[mode][field] = np.lib.format.open_memmap(attempt/f"{mode}_{field}.npy", mode="w+",
                                        dtype=dtype, shape=(int(point_offsets[-1]), columns))
    keep_values = np.lib.format.open_memmap(attempt/"all_points_global_keep.npy", mode="w+",
                        dtype=bool, shape=(int(pointsets["all_points"]["offsets"][-1]),))
    diagnostic = {mode: {"points": 0, "empty_regions": 0, "singleton_regions": 0,
                        "duplicate_descriptor_rows": 0, "regions_with_duplicate_descriptors": 0}
                  for mode in POINT_MODES}
    max_joint_error, max_geometry_error, max_reference_error = 0., 0., 0.
    references, global_original, global_retained, nonempty_maps = [], 0, 0, 0
    settings = _settings()
    jobs = ((np.asarray(maps[int(sid)]), np.asarray(old_labels[i]), settings, int(sid) in audit_set)
            for i, sid in enumerate(ids))
    try:
        with (attempt/"region_trace.jsonl").open("w", encoding="utf-8") as trace:
            for image_index, result in enumerate(_map_jobs(jobs, workers)):
                start, stop = map(int, offsets[image_index:image_index+2])
                if len(result["rows"]) != stop-start:
                    raise ValueError("Original mask and frozen cache region counts differ")
                trace_rows = []
                global_original += result["global_original_count"]
                global_retained += result["global_retained_count"]
                nonempty_maps += int(result["global_original_count"] > 0)
                if result["reference_audit"] is not None:
                    reference = {"sample_id": int(ids[image_index]), **result["reference_audit"]}
                    references.append(reference)
                    max_reference_error = max(max_reference_error, reference["max_descriptor_error"])
                for local, row in enumerate(result["rows"]):
                    region = start+local
                    if len(row["descriptors"]) != int(old_cache["area"][region]) or int(row["keep"].sum()) != int(old_cache["retained_count"][region]):
                        raise ValueError("Point counts disagree with the original regional cache")
                    for key in ("joint_all", "joint", "radial_hist16", "relative_area"):
                        error = float(np.max(np.abs(row[key]-old_cache[key][region]), initial=0.))
                        if error > TOLERANCE:
                            raise ValueError("Original descriptor recomputation differs: " + key)
                        if key.startswith("joint"):
                            max_joint_error = max(max_joint_error, error)
                        else:
                            max_geometry_error = max(max_geometry_error, error)
                    for mode in POINT_MODES:
                        chosen = np.ones(len(row["keep"]), dtype=bool) if mode == "all_points" else row["keep"]
                        x, y = pointsets[mode]["offsets"][region:region+2]
                        for field in ("descriptors", "coordinates"):
                            pointsets[mode][field][x:y] = row[field][chosen]
                        values = row["descriptors"][chosen]
                        duplicates = len(values)-len(np.unique(values, axis=0))
                        diagnostics = diagnostic[mode]
                        diagnostics["points"] += len(values)
                        diagnostics["empty_regions"] += int(len(values)==0)
                        diagnostics["singleton_regions"] += int(len(values)==1)
                        diagnostics["duplicate_descriptor_rows"] += duplicates
                        diagnostics["regions_with_duplicate_descriptors"] += int(duplicates>0)
                    x, y = pointsets["all_points"]["offsets"][region:region+2]
                    keep_values[x:y] = row["keep"]
                    trace_rows.append({"region_index": region, "token": local+1,
                        "all_points_offset": [int(x), int(y)],
                        "wbbs_denoised_offset": pointsets["wbbs_denoised"]["offsets"][region:region+2].tolist(),
                        "source_point_indices": row["source_point_indices"].tolist(),
                        "selection_reason": "unchanged original cc8 area>=5 Top-4"})
                trace.write(json.dumps({"sample_id": int(ids[image_index]), "regions": trace_rows}, separators=(",", ":"))+"\n")
                if (image_index+1)%100==0 or image_index+1==len(ids):
                    print(f"Regional WBBS points {split}: {image_index+1}/{len(ids)}", flush=True)
    finally:
        for pointset in pointsets.values():
            for field in ("descriptors", "coordinates"):
                pointset[field].flush()
        keep_values.flush()
    if [row["sample_id"] for row in references] != [int(sid) for sid in ids if int(sid) in audit_set]:
        raise ValueError("Missing requested original WBBS audits")
    _source_check(source_binding)
    record(attempt/"source_binding.json", source_binding)
    record(attempt/"audit.json", {"status": "verified", "split": split, "maps": len(ids),
        "regions": int(offsets[-1]), "empty_maps": int((np.diff(offsets)==0).sum()),
        "production_tensor_vote_calls": nonempty_maps, "global_original_points": global_original,
        "global_retained_points": global_retained, "point_sets": diagnostic,
        "old_features_bitwise_identical": True, "old_descriptors_bitwise_identical": True,
        "old_masks_bitwise_identical": True, "max_joint_reconstruction_error": max_joint_error,
        "max_geometry_reconstruction_error": max_geometry_error,
        "original_wbbs_audit_maps": len(references), "original_wbbs_audits": references,
        "max_original_wbbs_descriptor_error": max_reference_error,
        "descriptor_tolerance": TOLERANCE,
        "scope": "one production TV per nonempty original wafer; global normalization and keep; independent original WBBS encode audits only on fixed audit IDs"})
    record(attempt/"metadata.json", {"cache_version": 1, "context_signature": context_signature,
        "signature": signature, "split": split, "point_modes": list(POINT_MODES),
        "descriptor_settings": settings, "point_axis_order": ["stick", "ball", "rho"],
        "point_order": "original wafer raster order within each frozen region",
        "started_at": started, "finished_at": now().isoformat(), "training_required": False,
        "cnn_forward_passes": 0, "workers": workers, "source_arrays": source_arrays,
        "source_binding_sha256": digest(source_binding), "audit_ids": list(map(int, audit_ids))})
    seal(attempt, signature)
    if target.exists():
        raise ValueError("Another process created the target; preserve both attempts")
    attempt.rename(target)
    return read_cache(target, signature)
