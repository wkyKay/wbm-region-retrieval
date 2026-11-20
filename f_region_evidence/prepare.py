"""Isolated frozen-encoder caches for three auditable region proposals.

Only the requested split is touched. The calling runner guards test access
with its sealed valid selection. Source masks/features/descriptors stay read-only.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import tempfile

import numpy as np
import torch

from b1_structure_fusion.descriptors import load_settings, soft_histogram
from experiments.proposed._shared import ef_suite
from experiments.proposed.f_segmentation.tensor_voting import retained_points, tensor_saliencies, tensor_vote
from f_region_structure.state import now, record, seal, verified
from wbm_retrieval.common import digest, file_hash, read_json
from wbm_retrieval.diagnostics import normalized
from .proposals import PROPOSALS, PROPOSAL_SPECS, propose_all

DESCRIPTOR_FIELDS = ("joint", "joint_all", "shape", "shape_all", "original_count",
                     "retained_count", "quality", "empty", "empty_all", "radial_hist16",
                     "relative_area")
CACHE_FILES = {"regions.npz", "h512.npy", "region_labels.npy", "region_trace.jsonl",
               "audit.json", "metadata.json", "source_binding.json"}


def cache_signature(context_signature, split, proposal):
    if split not in ("valid", "test") or proposal not in PROPOSALS:
        raise ValueError("Invalid region cache split/proposal")
    return digest({"context_signature": context_signature, "split": split,
                   "proposal": proposal, "spec": PROPOSAL_SPECS[proposal], "cache_version": 1})


def _point_features(raw, settings):
    """One vote on the original wafer, shared by all proposals and all masks."""
    support = np.argwhere((raw == 1) | (raw == 2)).astype(np.float64)
    points_int = np.argwhere(raw == 2)
    points = points_int.astype(np.float64)
    center = (support.min(0) + support.max(0)) / 2
    radius = float(np.linalg.norm(support - center, axis=1).max())
    if len(points):
        saliencies = tensor_saliencies(tensor_vote(points, settings["sigma_d"], settings["block_size"]))
        maxima = saliencies.max(0)
        scaled = np.divide(saliencies, maxima, out=np.zeros_like(saliencies), where=maxima > 0)
        rho = np.linalg.norm(points - center, axis=1) / radius if radius > 0 else np.zeros(len(points))
        values = np.column_stack((rho, scaled))
        keep = retained_points(saliencies, settings["noise_threshold"])
    else:
        values, rho, keep = np.empty((0, 3)), np.empty(0), np.zeros(0, dtype=bool)
    return {"points": points_int, "values": values, "rho": rho, "keep": keep,
            "wafer_area": len(support)}


def _describe(labels, fields, settings):
    count = int(labels.max())
    tokens = labels[tuple(fields["points"].T)]
    area = np.bincount(labels.ravel(), minlength=count + 1)[1:].astype(np.int64)
    joint, joint_all = np.zeros((count, 128)), np.zeros((count, 128))
    radial = np.zeros((count, 16))
    retained = np.zeros(count, dtype=np.int64)
    for i in range(count):
        member = tokens == i + 1
        selected = member & fields["keep"]
        retained[i] = int(selected.sum())
        joint[i] = soft_histogram(fields["values"][selected], tuple(settings["bins"])).reshape(-1)
        joint_all[i] = soft_histogram(fields["values"][member], tuple(settings["bins"])).reshape(-1)
        radial[i] = np.histogram(np.clip(fields["rho"][member], 0, 1), bins=np.linspace(0, 1, 17))[0] / area[i]
    return {"joint": joint, "joint_all": joint_all,
            "shape": joint.reshape(count, 8, 4, 4).sum(1).reshape(count, 16),
            "shape_all": joint_all.reshape(count, 8, 4, 4).sum(1).reshape(count, 16),
            "original_count": area, "retained_count": retained,
            "quality": np.divide(retained, area, out=np.zeros(count), where=area > 0),
            "empty": retained == 0, "empty_all": np.zeros(count, dtype=bool),
            "radial_hist16": radial, "relative_area": area.astype(np.float64) / fields["wafer_area"]}


def _job(job):
    raw, old_labels, settings = job
    proposals = propose_all(raw, old_labels)
    fields = _point_features(raw, settings)
    return {name: (labels, audit, _describe(labels, fields, settings))
            for name, (labels, audit) in proposals.items()}


def _map_jobs(jobs, workers):
    if workers == 0:
        yield from map(_job, jobs)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        pending = deque()
        for _ in range(2 * workers):
            job = next(jobs, None)
            if job is not None:
                pending.append(executor.submit(_job, job))
        while pending:
            yield pending.popleft().result()
            job = next(jobs, None)
            if job is not None:
                pending.append(executor.submit(_job, job))


def _source_check(binding):
    if not isinstance(binding, dict) or not binding:
        raise ValueError("Frozen source binding must be nonempty")
    for name, expected in binding.items():
        path = Path(name)
        if not path.is_absolute() or path.is_symlink() or not path.is_file() or file_hash(path) != expected:
            raise ValueError("Frozen source changed or is invalid: " + name)


def _validate_inputs(maps, protocol, cache, labels, config, suite, split):
    if split not in ("valid", "test"):
        raise ValueError("Only valid or selection-guarded test region caches are supported")
    ids = np.asarray(protocol.ids[split])
    offsets = np.asarray(cache["offsets"])
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids) or len(np.unique(ids)) != len(ids)
            or np.any(ids < 0) or np.any(ids >= len(maps)) or not np.array_equal(cache["map_ids"], ids)):
        raise ValueError("Source original map IDs differ from requested frozen split")
    if (offsets.shape != (len(ids) + 1,) or offsets.dtype.kind not in "iu" or offsets[0] != 0
            or np.any(np.diff(offsets.astype(np.int64)) < 0) or np.any(np.diff(offsets.astype(np.int64)) > 4)):
        raise ValueError("Frozen Top-4 offsets are invalid")
    n = int(offsets[-1])
    if (np.asarray(maps).ndim != 3 or labels.shape != (len(ids), *maps.shape[1:])
            or np.asarray(cache["features"]).shape != (n, 512)
            or np.asarray(cache["features"]).dtype.kind != "f"
            or not np.isfinite(cache["features"]).all()):
        raise ValueError("Source masks or frozen h512 features have invalid dimensions")
    if (config.get("input_size") != 96 or config.get("embedding_dim") != 512
            or suite["variants"]["F"].get("window_size") != 52
            or config.get("proposal") != {"min_area": 5, "top_k": 4, "padding": 4}
            or suite["variants"]["F"].get("input_channels") != 3):
        raise ValueError("New regions require original min5/Top4, three-channel 52-to-96 h512 encoder")
    for name in DESCRIPTOR_FIELDS:
        if len(cache[name]) != n or not np.isfinite(cache[name]).all():
            raise ValueError("Frozen regional descriptor rows are invalid: " + name)
    if not np.array_equal(cache["original_count"], cache["area"]):
        raise ValueError("Frozen area aliases disagree")
    return ids.astype(np.int64, copy=False), offsets.astype(np.int64, copy=False)


def _preserve_base(row, old_cache, a, b):
    n = b - a
    maximum = 0.
    for name in DESCRIPTOR_FIELDS:
        actual, expected = row[name][:n], np.asarray(old_cache[name][a:b])
        if actual.shape != expected.shape:
            raise ValueError("Original Top-4 region descriptor shape changed")
        if expected.dtype.kind in "biu":
            if not np.array_equal(actual, expected):
                raise ValueError("Original Top-4 integer descriptor changed: " + name)
        else:
            error = float(np.max(np.abs(actual - expected))) if actual.size else 0.
            maximum = max(maximum, error)
            if error > 1e-14:
                raise ValueError("Original Top-4 geometry/J recomputation changed: " + name)
        # Preserve stored bit patterns after independently checking the masks
        # and one-pass descriptor computation against their historical values.
        row[name] = row[name].astype(expected.dtype, copy=False)
        row[name][:n] = expected
    return maximum


def read_cache(folder, signature=None):
    folder = Path(folder)
    if not verified(folder, signature):
        raise ValueError("Region evidence cache is not sealed")
    if set(read_json(folder / "completed.json")["files"]) != CACHE_FILES:
        raise ValueError("Region evidence cache is incomplete or contains unexpected files")
    with np.load(folder / "regions.npz", allow_pickle=False) as archive:
        data = dict(archive)
    data.update(features=np.load(folder / "h512.npy", mmap_mode="r"),
                region_labels=np.load(folder / "region_labels.npy", mmap_mode="r"), folder=folder)
    return data


def prepare(output, maps, protocol, old_cache, old_labels, config, suite, checkpoint,
            context_signature, source_binding, workers=4, device="cuda", split="valid"):
    """Build/cache all three proposals; encode only added regions, never train.

    The caller supplies a verified frozen regional-J cache and its original F
    label maps. Test selection authorization belongs to the caller. This API
    never loads any other split, labels, evaluation reports, or checkpoints.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError("workers must be a nonnegative integer")
    batch_size = config.get("inference_batch_size", 128)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Invalid frozen encoder inference batch size")
    ids, old_offsets = _validate_inputs(maps, protocol, old_cache, old_labels, config, suite, split)
    _source_check(source_binding)
    output = Path(output).resolve()
    if "folder" in old_cache and output.is_relative_to(Path(old_cache["folder"]).resolve()):
        raise ValueError("New output must not be inside a historical cache")
    targets = {name: output / "cache" / split / name for name in PROPOSALS}
    signatures = {name: cache_signature(context_signature, split, name) for name in PROPOSALS}
    result, pending = {}, []
    for name, target in targets.items():
        if verified(target, signatures[name]):
            if read_json(target / "source_binding.json") != source_binding:
                raise ValueError("Cached source binding changed")
            result[name] = read_cache(target, signatures[name])
        else:
            if target.exists():
                raise ValueError("Refuse to overwrite an incomplete region cache: " + str(target))
            pending.append(name)
    if not pending:
        return {name: result[name] for name in PROPOSALS}
    targets["base4"].parent.mkdir(parents=True, exist_ok=True)
    attempts, stored, label_arrays, trace_handles = {}, {}, {}, {}
    started = now().isoformat()
    for name in pending:
        attempts[name] = Path(tempfile.mkdtemp(prefix="." + name + "_attempt_", dir=targets[name].parent))
        stored[name] = {"rows": [], "features": [], "offsets": [0], "base_token_indices": [],
                        "audit_rows": [], "window_audits": [], "maximum_old_descriptor_error": 0.}
        label_arrays[name] = np.lib.format.open_memmap(attempts[name] / "region_labels.npy", mode="w+",
                                dtype=np.uint8, shape=(len(ids), *maps.shape[1:]))
        trace_handles[name] = (attempts[name] / "region_trace.jsonl").open("w", encoding="utf-8")
    model, patches, destinations = None, [], []

    def flush():
        nonlocal model
        if not patches:
            return
        if model is None:
            model = ef_suite.build_model(config, suite["variants"]["F"]).to(device).eval()
            model.load_state_dict(checkpoint["model"])
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        with torch.inference_mode():
            batch = torch.from_numpy(np.asarray(patches, dtype=np.float32)).to(device)
            values = normalized(model["encoder"](batch).cpu().numpy())
        if values.shape != (len(patches), 512) or not np.isfinite(values).all():
            raise ValueError("Frozen encoder returned invalid h512 features")
        for (destination, row), value in zip(destinations, values):
            destination[row] = value
        patches.clear()
        destinations.clear()

    settings = load_settings()
    jobs = ((np.asarray(maps[int(sid)]), np.array(old_labels[i]), settings) for i, sid in enumerate(ids))
    try:
        for i, proposals in enumerate(_map_jobs(jobs, workers)):
            a, b = map(int, old_offsets[i:i+2])
            nbase = b - a
            for name in pending:
                labels, audit, row = proposals[name]
                if audit["base_regions"] != nbase:
                    raise ValueError("Frozen token counts differ from original Top-4 masks")
                error = _preserve_base(row, old_cache, a, b)
                data = stored[name]
                data["maximum_old_descriptor_error"] = max(data["maximum_old_descriptor_error"], error)
                label_arrays[name][i] = labels
                feature = np.zeros((audit["regions"], 512), dtype=np.asarray(old_cache["features"]).dtype)
                feature[:nbase] = old_cache["features"][a:b]
                start = data["offsets"][-1]
                data["base_token_indices"].extend(range(start, start + nbase))
                data["features"].append(feature)
                data["offsets"].append(start + len(feature))
                data["rows"].append(row)
                trace = audit.pop("region_trace")
                trace_handles[name].write(json.dumps({"sample_id": int(ids[i]), "regions": trace}, separators=(",", ":")) + "\n")
                data["audit_rows"].append({"sample_id": int(ids[i]), **audit})
                for token in range(nbase + 1, audit["regions"] + 1):
                    patch, window = ef_suite.fixed_window_patch(maps[int(ids[i])], labels == token, 96, 52)
                    data["window_audits"].append({"sample_id": int(ids[i]), "token": token, **window})
                    patches.append(patch)
                    destinations.append((feature, token - 1))
                    if len(patches) >= batch_size:
                        flush()
            if (i + 1) % 100 == 0 or i + 1 == len(ids):
                print(f"Evidence regions {split}: {i+1}/{len(ids)}", flush=True)
        flush()
    finally:
        for handle in trace_handles.values():
            handle.close()
        for array in label_arrays.values():
            array.flush()
    _source_check(source_binding)
    for name in pending:
        data, attempt = stored[name], attempts[name]
        rows = data["rows"]
        values = {key: np.concatenate([row[key] for row in rows], axis=0) for key in DESCRIPTOR_FIELDS}
        features = np.concatenate(data["features"], axis=0)
        offsets = np.asarray(data["offsets"], dtype=np.int64)
        base_indices = np.asarray(data["base_token_indices"], dtype=np.int64)
        if (features[base_indices].dtype != np.asarray(old_cache["features"]).dtype
                or features[base_indices].tobytes() != np.asarray(old_cache["features"]).tobytes()):
            raise ValueError("Original Top-4 features lost byte identity")
        for key in DESCRIPTOR_FIELDS:
            if values[key][base_indices].tobytes() != np.asarray(old_cache[key]).tobytes():
                raise ValueError("Original Top-4 descriptors lost byte identity: " + key)
        values.update(map_ids=ids, offsets=offsets, area=values["original_count"],
                      histograms=values["radial_hist16"], owners=np.repeat(ids, np.diff(offsets)),
                      base_token_indices=base_indices,
                      base_region_counts=np.diff(old_offsets).astype(np.int64))
        np.savez_compressed(attempt / "regions.npz", **values)
        np.save(attempt / "h512.npy", features)
        original_pixels = sum(row["original_pixels"] for row in data["audit_rows"])
        retained_pixels = sum(row["retained_pixels"] for row in data["audit_rows"])
        record(attempt / "source_binding.json", source_binding)
        record(attempt / "audit.json", {"split": split, "proposal": name, "maps": len(ids),
               "regions": int(offsets[-1]), "original_pixels": original_pixels,
               "retained_pixels": retained_pixels,
               "total_pixel_retention": retained_pixels / original_pixels if original_pixels else 0.,
               "old_features_bitwise_identical": True, "old_descriptors_bitwise_identical": True,
               "max_old_descriptor_recomputation_error": data["maximum_old_descriptor_error"],
               "new_regions_encoded": len(data["window_audits"]),
               "map_audits": data["audit_rows"], "new_region_window_audits": data["window_audits"],
               "voting": "once per original wafer; all original defect points; shared across proposals",
               "mask_scope": "original pixels only; base masks preserved; additions disjoint"})
        record(attempt / "metadata.json", {"context_signature": context_signature,
               "signature": signatures[name], "split": split, "proposal": name,
               "spec": PROPOSAL_SPECS[name], "cache_version": 1, "started_at": started,
               "finished_at": now().isoformat(), "training_required": False,
               "source_binding_sha256": digest(source_binding), "descriptor_settings": settings,
               "encoding": "frozen original h512; three channels; original 52 window resized to 96",
               "normalization": "original NumPy float64 L2 normalization",
               "device": str(device), "workers": workers, "inference_batch_size": batch_size})
        seal(attempt, signatures[name])
        attempt.rename(targets[name])
        result[name] = read_cache(targets[name], signatures[name])
    return {name: result[name] for name in PROPOSALS}
