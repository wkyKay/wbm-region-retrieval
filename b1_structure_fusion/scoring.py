"""B0-exact fusion and float64 structural distances on CPU or CUDA.

CUDA applies to the structural distance kernels. The frozen F/WaPIRL and angle
score implementations remain unchanged; no GPU rounding or new tie rule is used.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from experiments.proposed.f_fusion_combination_test.pipeline import combined_score

from .descriptors import FAMILIES, distributions, load_settings, validate_settings


def valid_variants(settings=None):
    """B0 + four nonzero strengths for each family + J-only = 18 by default."""
    settings = load_settings() if settings is None else validate_settings(settings)
    specs = {"B0": {"family": "B0", "strength": 0.}}
    for family in FAMILIES:
        for strength in settings["lambda_grid"]:
            if strength > 0:
                specs[f"{family}_{strength:g}"] = {"family": family, "strength": float(strength)}
    specs["J_only"] = {"family": "J_only", "strength": 1.}
    return specs


def baseline_score(values):
    """Call the existing frozen B0 formula, without changing operation order."""
    return combined_score(values)


def score_variant(base_values, distances, spec):
    """Score an entire candidate vector; no retrieval candidates are truncated."""
    base = baseline_score(base_values) if isinstance(base_values, Mapping) else np.asarray(base_values)
    if base.ndim != 1 or not np.isfinite(base).all() or np.any(base < 0) or np.any(base > 1):
        raise ValueError("Expected finite B0 scores in [0,1]")
    family, strength = spec["family"], spec["strength"]
    if isinstance(strength, bool) or not np.isscalar(strength) or not np.isfinite(strength) or strength < 0:
        raise ValueError("Invalid structure strength")
    if family not in (*FAMILIES, "B0", "J_only"):
        raise ValueError("Unknown B1 score family")
    if family == "B0":
        if strength != 0:
            raise ValueError("B0 must have zero structure strength")
        return base
    if family != "J_only" and strength == 0:
        return base
    if family == "J_only" and strength != 1:
        raise ValueError("J-only is a fixed monotonic distance score; no strength selection")
    distance = np.asarray(distances["J" if family == "J_only" else family], dtype=np.float64)
    if distance.shape != base.shape or not np.isfinite(distance).all() or np.any(distance < 0) or np.any(distance > 1):
        raise ValueError("Expected one finite structural distance in [0,1] per candidate")
    if family == "J_only":
        return np.exp(-distance)
    return base * np.exp(-float(strength)*distance)


class StructuralDistances:
    """Cache square-root distributions once, then score one full query pool.

    CUDA calculations use torch.float64 and return NumPy float64 vectors. CPU and
    CUDA are numerically checked within tolerance, not claimed bit-identical;
    the selected backend belongs in the run signature. Exact ties retain the
    project's existing SHA tie ordering in the downstream ranking evaluator.
    """

    def __init__(self, sample_ids, joint128, empty, device="cpu"):
        ids = np.asarray(sample_ids)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != len(ids) or np.any(ids < 0):
            raise ValueError("Sample IDs must be unique nonnegative original integers")
        joint128 = np.asarray(joint128, dtype=np.float64)
        empty = np.asarray(empty)
        if joint128.shape != (len(ids), 128) or empty.dtype != bool or empty.shape != (len(ids),):
            raise ValueError("Descriptor shapes/empty flags do not match sample IDs")
        self.distributions = distributions(joint128)
        if not np.array_equal(joint128.sum(axis=1) == 0, empty):
            raise ValueError("Empty flags disagree with histogram mass")
        self.rows = {int(sample): i for i, sample in enumerate(ids)}
        self.empty = empty.copy()
        self.device = str(device)
        if self.device != "cpu" and self.device != "cuda" and not self.device.startswith("cuda:"):
            raise ValueError("Structural scoring supports only cpu or cuda[:index]")
        self.roots = {family: np.sqrt(values) for family, values in self.distributions.items()}
        self._torch = None
        if self.device != "cpu":
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available; explicitly use --device cpu")
            target = torch.device(self.device)
            if target.index is not None and target.index >= torch.cuda.device_count():
                raise ValueError("Requested CUDA device does not exist")
            self._torch = torch
            self.roots = {name: torch.as_tensor(values, dtype=torch.float64, device=target)
                          for name, values in self.roots.items()}

    def for_query(self, query, candidates):
        ids = np.asarray(candidates)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("Candidate IDs must be a one-dimensional integer array")
        try:
            qi = self.rows[int(query)]
            rows = np.array([self.rows[int(sample)] for sample in ids], dtype=np.int64)
        except KeyError as error:
            raise ValueError(f"Missing structure descriptor for sample {error.args[0]}") from error
        q_empty, c_empty = bool(self.empty[qi]), self.empty[rows]
        results = {}
        if self._torch is not None:
            torch = self._torch
            indices = torch.as_tensor(rows, device=self.device, dtype=torch.long)
            with torch.no_grad():
                for family, roots in self.roots.items():
                    difference = roots[indices] - roots[qi]
                    value = (.5 * difference.square().sum(dim=1)).clamp_min(0).sqrt().clamp(0, 1)
                    results[family] = value.cpu().numpy()
        else:
            for family, roots in self.roots.items():
                difference = roots[rows] - roots[qi]
                results[family] = np.clip(np.sqrt(np.maximum(.5*np.sum(difference*difference, axis=1), 0.)), 0., 1.)
        for result in results.values():
            result[np.logical_xor(q_empty, c_empty)] = 1.
            result[np.logical_and(q_empty, c_empty)] = 0.
        return results

    distances = for_query


# Descriptive aliases retained for callers using the initial design vocabulary.
StructureDistances = StructuralDistances
variant_score = score_variant
