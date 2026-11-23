"""Immutable train-only, two-view mutual nearest-neighbor exclusion graph.

Neighbors are exclusions from the negative pool, never additional positives.
The caller supplies only frozen train h512 features in identical token order.
No dataset, label, valid feature, or test feature is opened by this module.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from f_region_structure.state import now, read_signed, record, seal, signed, verified
from wbm_retrieval.common import digest, file_hash, read_json


def _array(value):
    return np.load(value, mmap_mode="r", allow_pickle=False) if isinstance(value, (str, Path)) else np.asarray(value)


def _array_binding(value, array):
    """Hash logical row-major bytes and their dtype/shape, without a full copy."""
    hasher = hashlib.sha256()
    for start in range(0, len(array), 4096):
        hasher.update(np.ascontiguousarray(array[start:start + 4096]).tobytes())
    result = {"shape": list(array.shape), "dtype": array.dtype.str, "sha256": hasher.hexdigest()}
    if isinstance(value, (str, Path)):
        result.update(source_path=str(Path(value).resolve()), file_sha256=file_hash(value))
    return result


def _normalize(features):
    """Compute norms/division in float64, then round each component to float32."""
    result = np.empty(features.shape, dtype=np.float32)
    for start in range(0, len(features), 4096):
        block = np.asarray(features[start:start + 4096], dtype=np.float64)
        norms = np.sqrt(np.sum(block * block, axis=1, dtype=np.float64))
        if not np.isfinite(block).all() or not np.isfinite(norms).all() or np.any(norms <= 0):
            raise ValueError("Teacher features must be finite and have nonzero L2 norm")
        result[start:start + len(block)] = block / norms[:, None]
    return result


def _validate_inputs(original, rotated, map_ids, k, block_size):
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if original.ndim != 2 or original.shape[0] < 1 or original.shape[1] != 512 or rotated.shape != original.shape:
        raise ValueError("Both teacher views must have identical nonempty (N, 512) shape")
    if original.dtype.kind != "f" or rotated.dtype.kind != "f":
        raise ValueError("Teacher features must have floating dtype")
    if map_ids.ndim != 1 or len(map_ids) != len(original) or map_ids.dtype.kind not in "iu":
        raise ValueError("token_map_ids must be an integer vector in teacher token order")
    if np.any(map_ids < 0) or np.any(map_ids > np.iinfo(np.int64).max):
        raise ValueError("token_map_ids must be nonnegative int64-compatible raw wafer IDs")
    if len(original) > np.iinfo(np.int32).max:
        raise ValueError("Graph token indices exceed int32 storage")
    # Fail before creating output for malformed features.
    for features in (original, rotated):
        for start in range(0, len(features), 4096):
            block = np.asarray(features[start:start + 4096], dtype=np.float64)
            norms = np.sum(block * block, axis=1, dtype=np.float64)
            if not np.isfinite(block).all() or not np.isfinite(norms).all() or np.any(norms <= 0):
                raise ValueError("Teacher features must be finite and have nonzero L2 norm")


def _backend(device):
    if str(device) == "cpu":
        return {"device": "cpu", "numpy_version": np.__version__, "dot_product": "numpy_float32_blas"}
    import torch
    selected = torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("device must be cpu or an available CUDA device")
    return {"device": str(selected), "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(selected),
            "dot_product": "torch_float32_gemm_tf32_disabled"}


@contextmanager
def _full_float32_torch():
    import torch
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def _cpu_topk(scores, width):
    """Exact score ordering, with ascending original token ID for boundary ties."""
    result = np.full((len(scores), width), -1, dtype=np.int32)
    for row, values in enumerate(scores):
        finite = np.flatnonzero(np.isfinite(values))
        count = min(width, len(finite))
        if not count:
            continue
        if count == len(finite):
            selected = finite
        else:
            threshold = np.partition(values[finite], len(finite) - count)[len(finite) - count]
            strict = np.flatnonzero(values > threshold)
            tied = np.flatnonzero(values == threshold)
            selected = np.concatenate((strict, tied[:count - len(strict)]))
        # The graph needs membership, not score rank. Sorted indices permit binary search.
        result[row, :count] = np.sort(selected)
    return result


def _cuda_topk(scores, width):
    """Repair arbitrary CUDA topk boundary ties using exact equality of FP32 scores."""
    import torch
    thresholds = torch.topk(scores, width, dim=1, sorted=False).values.min(dim=1).values
    strict = scores > thresholds[:, None]
    required = width - strict.sum(dim=1)
    tied = (scores == thresholds[:, None]) & torch.isfinite(scores)
    ordinal = torch.cumsum(tied, dim=1, dtype=torch.int32)
    selected = strict | (tied & (ordinal <= required[:, None]))
    pairs = torch.nonzero(selected, as_tuple=False).cpu().numpy()
    result = np.full((len(scores), width), -1, dtype=np.int32)
    if len(pairs):
        counts = np.bincount(pairs[:, 0], minlength=len(scores))
        offset = np.r_[0, counts.cumsum()[:-1]]
        positions = np.arange(len(pairs)) - np.repeat(offset, counts)
        result[pairs[:, 0], positions] = pairs[:, 1]
    return result


def _directed_topk(features, map_ids, k, device, block_size, path):
    n, width = len(features), min(k, len(features))
    normalized = _normalize(features)
    temporary = path.with_suffix(".partial.npy")
    top = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.int32, shape=(n, width))
    if str(device) == "cpu":
        for start in range(0, n, block_size):
            stop = min(start + block_size, n)
            scores = normalized[start:stop] @ normalized.T
            scores[map_ids[start:stop, None] == map_ids[None, :]] = -np.inf
            top[start:stop] = _cpu_topk(scores, width)
    else:
        import torch
        with _full_float32_torch(), torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            gallery = torch.from_numpy(normalized).to(device)
            wafer = torch.as_tensor(map_ids, dtype=torch.int64, device=device)
            for start in range(0, n, block_size):
                stop = min(start + block_size, n)
                scores = gallery[start:stop] @ gallery.T
                scores.masked_fill_(wafer[start:stop, None] == wafer[None, :], -torch.inf)
                top[start:stop] = _cuda_topk(scores, width)
                if start == 0 or stop == n or start // 10000 != stop // 10000:
                    print(f"[neighbor graph] {path.stem}: {stop}/{n}", flush=True)
    top.flush()
    del top
    temporary.replace(path)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _row_contains(top, counts, rows, target):
    """Vectorized binary search through a small set of sorted, padded rows."""
    lower = np.zeros(len(rows), dtype=np.int64)
    upper = counts[rows].astype(np.int64, copy=True)
    while np.any(lower < upper):
        active = np.flatnonzero(lower < upper)
        middle = (lower[active] + upper[active]) // 2
        below = top[rows[active], middle] < target
        lower[active[below]] = middle[below] + 1
        upper[active[~below]] = middle[~below]
    present = lower < counts[rows]
    present[present] &= top[rows[present], lower[present]] == target
    return present


def _mutual_intersection(original, rotated):
    counts0 = np.sum(original >= 0, axis=1)
    counts1 = np.sum(rotated >= 0, axis=1)
    rows = []
    for query in range(len(original)):
        candidates = np.intersect1d(original[query, :counts0[query]], rotated[query, :counts1[query]], assume_unique=True)
        accepted = _row_contains(original, counts0, candidates, query)
        accepted &= _row_contains(rotated, counts1, candidates, query)
        rows.append(candidates[accepted].astype(np.int32, copy=False))
    degree = np.fromiter((len(row) for row in rows), dtype=np.int64, count=len(rows))
    indptr = np.r_[np.int64(0), np.cumsum(degree, dtype=np.int64)]
    indices = np.concatenate(rows).astype(np.int32, copy=False)
    return indptr, indices, counts0, counts1


def _save_array(path, value):
    temporary = path.with_suffix(".partial.npy")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def _statistics(indptr, indices, map_ids, counts0, counts1):
    degrees = np.diff(indptr)
    return {"tokens": len(map_ids), "wafers": len(np.unique(map_ids)), "directed_edges_original": int(counts0.sum()),
            "directed_edges_rot90": int(counts1.sum()), "exclusion_directed_edges": len(indices),
            "exclusion_undirected_pairs": len(indices) // 2, "degree_min": int(degrees.min()),
            "degree_max": int(degrees.max()), "degree_mean": float(degrees.mean()),
            "degree_median": float(np.median(degrees)), "degree_p90": float(np.percentile(degrees, 90)),
            "degree_p99": float(np.percentile(degrees, 99)), "degree_zero_count": int(np.count_nonzero(degrees == 0)),
            "degree_zero_fraction": float(np.mean(degrees == 0)),
            "degree_histogram": {str(int(k)): int(v) for k, v in zip(*np.unique(degrees, return_counts=True))},
            "same_wafer_edges": 0, "self_edges": 0, "symmetric_by_construction": True}


def build_neighbor_graph(output, features_original, features_rot90, token_map_ids, *, k=500,
                         device="cuda:0", block_size=256, binding=None):
    """Build exhaustive two-view mutual Top-k graph, or verify/reuse sealed output.

    FP64 normalization is rounded to FP32 before exhaustive FP32 dot products.
    Exact here means all train candidates are scored (no ANN), not real-number
    arithmetic or bitwise equivalence across different BLAS/CUDA environments.
    Same-wafer candidates are excluded *only for graph construction*.
    Existing incomplete output is deliberately refused; use a fresh output path.
    """
    output = Path(output)
    original, rotated, map_ids = _array(features_original), _array(features_rot90), _array(token_map_ids)
    _validate_inputs(original, rotated, map_ids, k, block_size)
    # Reject non-JSON/NaN provenance before touching output.
    binding = json.loads(json.dumps(binding or {}, sort_keys=True, allow_nan=False))
    identity = {"schema_version": 1, "algorithm": "two_view_mutual_topk_v1", "scope": "train_only_no_labels",
                "usage": "negative_exclusion_only_not_positive_attraction", "k": int(k), "block_size": int(block_size),
                "original": _array_binding(features_original, original), "rot90": _array_binding(features_rot90, rotated),
                "token_map_ids": _array_binding(token_map_ids, map_ids), "binding": binding,
                "backend": _backend(device), "implementation_sha256": file_hash(__file__),
                "normalization": "float64_l2_division_then_round_to_float32",
                "score_precision": "float32_exhaustive_dot_product", "tie_break": "ascending_original_token_index",
                "same_wafer_policy": "excluded_in_each_directed_topk_before_selection",
                "graph_rule": "q_in_topk0(c) AND c_in_topk0(q) AND q_in_topk90(c) AND c_in_topk90(q)",
                "resume_policy": "verify_complete_only_refuse_incomplete"}
    signature = digest(identity)
    if output.exists() and any(output.iterdir()):
        if not (output / "completed.json").is_file():
            raise ValueError(f"Incomplete graph output will not be overwritten; choose a new output directory: {output}")
        graph = load_graph(output)
        if graph["signature"] != signature:
            raise ValueError(f"Frozen graph input/settings signature changed: {output}")
        return graph
    output.mkdir(parents=True, exist_ok=True)
    started = now()
    record(output / "metadata.json", signed({"identity": identity, "graph_signature": signature,
           "run_id": started.strftime("%Y%m%dT%H%M%S%z") + "__cnn_refinement_train_neighbor_graph",
           "started_at": started.isoformat()}))
    timer = time.monotonic()
    ids64 = np.asarray(map_ids, dtype=np.int64)
    top0 = _directed_topk(original, ids64, k, device, block_size, output / "topk_original.npy")
    top1 = _directed_topk(rotated, ids64, k, device, block_size, output / "topk_rot90.npy")
    indptr, indices, counts0, counts1 = _mutual_intersection(top0, top1)
    _save_array(output / "indptr.npy", indptr)
    _save_array(output / "indices.npy", indices)
    _save_array(output / "token_map_ids.npy", ids64)
    record(output / "statistics.json", {**_statistics(indptr, indices, ids64, counts0, counts1),
           "elapsed_seconds": time.monotonic() - timer, "completed_at": now().isoformat()})
    seal(output, signature)
    return load_graph(output)


def load_graph(path):
    """Verify the seal and CSR invariants, then expose read-only memory maps."""
    path = Path(path)
    metadata = read_signed(path / "metadata.json")
    signature = metadata["graph_signature"]
    if signature != digest(metadata["identity"]) or not verified(path, signature):
        raise ValueError(f"Unsealed or invalid graph: {path}")
    marker = read_json(path / "completed.json")
    required = {"metadata.json", "statistics.json", "indptr.npy", "indices.npy", "token_map_ids.npy", "topk_original.npy", "topk_rot90.npy"}
    if not required.issubset(marker["files"]):
        raise ValueError("Graph seal does not bind every required artifact")
    indptr = np.load(path / "indptr.npy", mmap_mode="r", allow_pickle=False)
    indices = np.load(path / "indices.npy", mmap_mode="r", allow_pickle=False)
    ids = np.load(path / "token_map_ids.npy", mmap_mode="r", allow_pickle=False)
    n = metadata["identity"]["original"]["shape"][0]
    if (indptr.dtype != np.int64 or indices.dtype != np.int32 or ids.dtype != np.int64
            or indptr.shape != (n + 1,) or ids.shape != (n,) or indices.ndim != 1
            or indptr[0] != 0 or indptr[-1] != len(indices) or np.any(np.diff(indptr) < 0)
            or np.any(np.diff(indptr) > metadata["identity"]["k"])
            or np.any(indices < 0) or np.any(indices >= n)):
        raise ValueError("Invalid neighbor graph CSR structure")
    for row in range(n):
        neighbors = indices[indptr[row]:indptr[row + 1]]
        if np.any(np.diff(neighbors) <= 0) or np.any(ids[neighbors] == ids[row]):
            raise ValueError("Graph has duplicate, unsorted, or same-wafer neighbors")
    return {"path": path, "signature": signature, "metadata": metadata, "indptr": indptr,
            "indices": indices, "token_map_ids": ids, "statistics": read_json(path / "statistics.json")}
