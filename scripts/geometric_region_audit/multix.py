"""Multi-X line/circle adaptation: model-space modes + original GCO + CV.

The outer loop is a NumPy port of the public Multi-X framework; graph-cut
alpha-expansion and label costs run in its original C++ GCoptimization code.
See MULTIX_NOTES.md for the paper/code differences and wafer-specific choices.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

PARAMETERS = {
    "method": "Multi-X line+circle adaptation with model validation",
    "gamma_pixels": 2.0,
    "spatial_weight": 0.3,
    "neighbor_radius_pixels": 1.5,
    "initial_hypotheses": "2*N total, N lines and N circles",
    "sampler": "uniform without replacement",
    "mode_seeking": "per-class point-representation mean-shift (public implementation)",
    "mode_bandwidth_pixels": 2.0,
    "hmax_for_label_cost": 10,
    "label_cost": "MSS * log(N) / hmax",
    "max_iterations": 10,
    "cv_trials": 100,
    "cv_degenerate_resampling_limit": 100,
    "circle_radius_bounds": "0.5 to 2*image_diagonal pixels (adaptation)",
    "native_min_area": "model minimum sample size only (line 2; circle 3)",
    "output": "exclusive labels on original defective pixels only",
    "implementation_status": "Python outer-loop port + upstream C++ GCO; added circle estimator; fixed mode bandwidth",
}
AUTHOR_COMMIT = "60f71708a0e02a2a9192f59b6509556137b1fa61"
_LIB = None


def _gco():
    global _LIB
    if _LIB is not None:
        return _LIB
    directory = Path(__file__).parent / "author_multi_x"
    library = directory / "local_adapter/libmultix_gco.dylib"
    if not library.exists():
        subprocess.run([
            "clang++", "-std=c++14", "-O2", "-shared", "-fPIC", "-include", "cfloat",
            "-Ilocal_adapter", "-Iinclude", "local_adapter/gco_bridge.cpp",
            "include/GCoptimization.cpp", "include/LinkedBlockList.cpp", "-o", str(library),
        ], cwd=directory, check=True, capture_output=True)
    _LIB = ctypes.CDLL(str(library))
    array = np.ctypeslib.ndpointer
    _LIB.multix_label.argtypes = [
        ctypes.c_int, ctypes.c_int, array(np.float32, flags="C_CONTIGUOUS"),
        ctypes.c_int, array(np.int32, flags="C_CONTIGUOUS"), ctypes.c_float,
        array(np.float32, flags="C_CONTIGUOUS"), array(np.int32, flags="C_CONTIGUOUS"),
        ctypes.POINTER(ctypes.c_double), ctypes.c_char_p,
    ]
    _LIB.multix_label.restype = ctypes.c_int
    return _LIB


def _fit(points, kind, radius_max, nonminimal=False):
    if len(points) < (2 if kind == 0 else 3):
        return None
    if kind == 0:
        center = points.mean(axis=0)
        _, singular, vectors = np.linalg.svd(points - center, full_matrices=False)
        if singular[0] < 1e-9:
            return None
        normal = vectors[-1]
        if normal[0] < 0 or (abs(normal[0]) < 1e-12 and normal[1] < 0):
            normal = -normal
        return np.r_[normal, -normal @ center]
    a = np.c_[2 * points, np.ones(len(points))]
    if np.linalg.matrix_rank(a) < 3:
        return None
    solution = np.linalg.lstsq(a, np.sum(points**2, axis=1), rcond=None)[0]
    radius2 = solution[2] + solution[:2] @ solution[:2]
    if radius2 <= 0:
        return None
    fitted = np.r_[solution[:2], np.sqrt(radius2)]
    if nonminimal:
        result = least_squares(
            lambda p: np.linalg.norm(points - p[:2], axis=1) - p[2], fitted,
            max_nfev=50, ftol=1e-8, xtol=1e-8, gtol=1e-8,
        )
        fitted = result.x
    if not np.all(np.isfinite(fitted)) or not 0.5 <= fitted[2] <= radius_max:
        return None
    return fitted


def _residual(points, model):
    kind, parameters, _ = model
    if kind == 0:
        return np.abs(points @ parameters[:2] + parameters[2])
    return np.abs(np.linalg.norm(points - parameters[:2], axis=1) - parameters[2])


def _representation(model, extent):
    kind, p, _ = model
    if kind == 0:
        origin = -p[2] * p[:2]
        return np.array([origin, origin + extent * np.array([-p[1], p[0]])])
    # A fixed ordering of four cardinal circle points, in the same pixel units.
    return p[:2] + p[2] * np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])


def _mean_shift_groups(representations, rng, bandwidth):
    """Port upstream mean_shift.h pointwise-mean metric and cluster votes."""
    count = len(representations)
    if count <= 1:
        return [np.arange(count)] if count else []
    visited = np.zeros(count, bool)
    centers, votes = [], []
    while not visited.all():
        remaining = np.flatnonzero(~visited)
        current = representations[int(rng.choice(remaining))].copy()
        current_votes = np.zeros(count, np.int32)
        for _ in range(100):
            distance = np.linalg.norm(representations - current, axis=2).mean(axis=1)
            members = np.flatnonzero(distance < bandwidth)
            if not len(members):
                members = np.array([remaining[0]])
            current_votes[members] += 1
            visited[members] = True
            next_mean = representations[members].mean(axis=0)
            converged = np.linalg.norm(next_mean - current) < 1e-3 * bandwidth
            current = next_mean
            if converged:
                break
        merge = next((i for i, center in enumerate(centers)
                      if np.linalg.norm(center - current) < bandwidth / 2), None)
        if merge is None:
            centers.append(current)
            votes.append(current_votes)
        else:
            centers[merge] = (centers[merge] + current) / 2
            votes[merge] += current_votes
    assignment = np.argmax(np.stack(votes), axis=0)
    return [np.flatnonzero(assignment == j) for j in range(len(votes))
            if np.any(assignment == j)]


def _mode_move(models, points, rng, extent, radius_max):
    proposed = []
    for kind in (0, 1):
        indices = [i for i, model in enumerate(models) if model[0] == kind]
        if not indices:
            continue
        representations = np.stack([_representation(models[i], extent) for i in indices])
        groups = _mean_shift_groups(representations, rng, PARAMETERS["mode_bandwidth_pixels"])
        for group in groups:
            members = [models[indices[i]] for i in group]
            if len(members) == 1:
                proposed.append(members[0])
                continue
            inliers = np.unique(np.concatenate([m[2] for m in members]))
            fit = _fit(points[inliers], kind, radius_max, nonminimal=True)
            if fit is None:
                proposed.append(members[0])
            else:
                proposed.append((kind, fit, inliers))
    return proposed


def _label(points, models, edges):
    if not models:
        return np.zeros(len(points), np.int32), float(len(points))
    gamma = PARAMETERS["gamma_pixels"]
    distance = np.stack([_residual(points, model) for model in models], axis=1)
    # This is the actual public MultiX.h data_energy, not an invented RANSAC score.
    normalized = distance**2 / (gamma**2 * 9 / 4)
    costs = np.where(normalized > 1, 2, normalized)
    unary = np.ascontiguousarray(np.c_[np.ones(len(points)), costs], dtype=np.float32)
    penalty = np.asarray([0] + [(2 if m[0] == 0 else 3) * np.log(max(len(points), 2)) /
                                PARAMETERS["hmax_for_label_cost"] for m in models], dtype=np.float32)
    labels = np.zeros(len(points), np.int32)
    energy = ctypes.c_double()
    error = ctypes.create_string_buffer(1000)
    status = _gco().multix_label(len(points), len(models)+1, unary, len(edges), edges,
                               PARAMETERS["spatial_weight"], penalty, labels,
                               ctypes.byref(energy), error)
    if status:
        raise RuntimeError(error.value.decode())
    return labels, energy.value


def _validate(points, model, rng, radius_max):
    """Paper Algorithm 2: mean residual over all inliers and 100 MSS trials."""
    kind = model[0]
    size = 2 if kind == 0 else 3
    if len(points) < size:
        return False, None, 0
    trial_errors = []
    degenerates = 0
    for _ in range(PARAMETERS["cv_trials"]):
        fitted = None
        for _ in range(PARAMETERS["cv_degenerate_resampling_limit"]):
            sample = rng.choice(len(points), size, replace=False)
            fitted = _fit(points[sample], kind, radius_max)
            if fitted is not None:
                break
            degenerates += 1
        if fitted is None:
            return False, None, degenerates
        trial_errors.append(float(_residual(points, (kind, fitted, None)).mean()))
    average = float(np.mean(trial_errors))
    return average < PARAMETERS["gamma_pixels"], average, degenerates


def multix_regions(raw, seed=2026):
    start = time.perf_counter()
    raw = np.asarray(raw)
    rc = np.argwhere(raw == 2)
    points = rc[:, ::-1].astype(float)
    output = np.zeros(raw.shape, np.int32)
    diag = {"parameters": dict(PARAMETERS), "author_commit": AUTHOR_COMMIT,
            "n_defects": len(points), "seed": int(seed), "iterations": [], "validation": []}
    if len(points) < 2:
        diag.update(elapsed_seconds=time.perf_counter()-start, n_regions=0)
        return output, diag
    rng = np.random.default_rng(seed)
    extent = float(max(raw.shape))
    radius_max = 2 * float(np.linalg.norm(raw.shape))
    edges = np.ascontiguousarray(cKDTree(points).query_pairs(PARAMETERS["neighbor_radius_pixels"],
                                                          output_type="ndarray"), dtype=np.int32)
    models = []
    for kind in (0, 1):
        size = 2 if kind == 0 else 3
        if len(points) < size:
            continue
        for _ in range(len(points)):
            for _ in range(100):
                sample = rng.choice(len(points), size, replace=False)
                fitted = _fit(points[sample], kind, radius_max)
                if fitted is not None:
                    candidate = (kind, fitted, None)
                    inliers = np.flatnonzero(_residual(points, candidate) < PARAMETERS["gamma_pixels"])
                    if len(inliers) > size:
                        models.append((kind, fitted, inliers))
                        break
    diag["initial_models"] = len(models)
    last_energy = np.inf
    labels = np.zeros(len(points), np.int32)
    for iteration in range(PARAMETERS["max_iterations"]):
        if iteration:
            # Eq. 5 compares the proposed model set with this CURRENT model set,
            # whose parameters changed in the previous refitting step.
            baseline_labels, baseline_energy = _label(points, models, edges)
        else:
            baseline_labels, baseline_energy = None, np.inf
        proposed = _mode_move(models, points, rng, extent, radius_max)
        candidate_labels, candidate_energy = _label(points, proposed, edges)
        # Eq. 5: subsequent mode moves are accepted only if energy does not rise.
        accepted = iteration == 0 or candidate_energy <= baseline_energy + 1e-5
        if accepted:
            models, labels, energy = proposed, candidate_labels, candidate_energy
        else:
            labels, energy = baseline_labels, baseline_energy
        refined = []
        for index, model in enumerate(models, 1):
            inliers = np.flatnonzero(labels == index)
            fit = _fit(points[inliers], model[0], radius_max, nonminimal=True)
            if fit is not None:
                candidate = (model[0], fit, inliers)
                # Retain old parameters if geometric L2 refinement makes the
                # public implementation's truncated data term worse.
                denominator = PARAMETERS["gamma_pixels"]**2 * 9 / 4
                old_terms = _residual(points[inliers], model)**2 / denominator
                new_terms = _residual(points[inliers], candidate)**2 / denominator
                old_error = np.where(old_terms > 1, 2, old_terms).sum()
                new_error = np.where(new_terms > 1, 2, new_terms).sum()
                refined.append(candidate if new_error <= old_error + 1e-6 else (model[0], model[1], inliers))
        diag["iterations"].append({"iteration": iteration+1, "models_before": len(models),
                                   "models_after_refit": len(refined), "energy": energy,
                                   "baseline_energy": baseline_energy if iteration else None,
                                   "mode_move_accepted": bool(accepted)})
        converged = abs(last_energy - energy) < 1e-5
        models = refined
        last_energy = energy
        if not models or converged:
            break
    # Always relabel after model deletion/refitting so indices match descriptors.
    labels, energy = _label(points, models, edges)
    retained = []
    for index, model in enumerate(models, 1):
        inliers = np.flatnonzero(labels == index)
        valid, mean_error, degenerates = _validate(points[inliers], model, rng, radius_max)
        diag["validation"].append({"original_label": index, "kind": "line" if model[0] == 0 else "circle",
                                    "area": len(inliers), "mean_cv_distance": mean_error,
                                    "accepted": bool(valid), "degenerate_trials_resampled": degenerates,
                                    "parameters": model[1].tolist()})
        if valid:
            retained.append((inliers, model))
    # Algorithm 2 deletes unstable models; it does not reassign their pixels.
    retained.sort(key=lambda item: (-len(item[0]), int(item[0][0])))
    diag["regions"] = []
    for index, (inliers, model) in enumerate(retained, 1):
        output[tuple(rc[inliers].T)] = index
        diag["regions"].append({"label": index, "kind": "line" if model[0] == 0 else "circle",
                                "area": len(inliers), "model_parameters": model[1].tolist()})
    diag.update(n_regions=len(retained), elapsed_seconds=time.perf_counter()-start,
                assigned_pixels=int(np.count_nonzero(output)), energy_before_validation=energy,
                includes_paper_model_validation=True)
    return output, diag
