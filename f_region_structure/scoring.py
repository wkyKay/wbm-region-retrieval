"""Frozen, region-level structure experiments; no whole-wafer score fusion.

All scores feed the existing cc8 one-to-one greedy matcher. Concatenation is
explicitly a weighted feature kernel, not a learned cross-feature interaction.
An empty region histogram is missing evidence and contributes no structure
penalty or bonus, including when both regions are empty after denoising.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np

from experiments.proposed.f_segmentation_geometry.scoring import pair_scores as original_pair_scores


EMPTY_POLICY = "pairwise_missing_structure_falls_back_to_active_CNN_geometry"


def frozen_grid():
    """Exactly 25 predeclared candidates, including the original F baseline."""
    grid = {"baseline": {"family": "baseline", "descriptor": None, "area": True}}
    for family in ("add_j", "replace_radial"):
        for source in ("retained", "all_points"):
            for strength in (.5, 1., 2.):
                name = f"{family}_{source}_l{str(strength).replace('.', 'p')}"
                grid[name] = {"family": family, "descriptor": source, "distribution": "joint",
                              "strength": strength, "area": True}
    for source in ("retained", "all_points"):
        for weight in (.25, .5, .75):
            name = f"concat_{source}_w{str(weight).replace('.', 'p')}"
            grid[name] = {"family": "concat", "descriptor": source, "distribution": "joint",
                          "structure_weight": weight, "area": True}
    for weight in (.25, .5, .75):
        name = f"concat_all_points_no_area_w{str(weight).replace('.', 'p')}"
        grid[name] = {"family": "concat", "descriptor": "all_points", "distribution": "joint",
                      "structure_weight": weight, "area": False}
    grid["replace_radial_independent_all_points_l1p0"] = {
        "family": "replace_radial", "descriptor": "all_points", "distribution": "independent",
        "strength": 1., "area": True}
    for family in ("plain_mlp", "gated_mlp"):
        grid[family] = {"family": family, "descriptor": "all_points", "area": False}
    return grid


def checked_spec(spec):
    if not isinstance(spec, Mapping) or not any(dict(spec) == s for s in frozen_grid().values()):
        raise ValueError("Score specification is outside the frozen 25-candidate grid")
    return dict(spec)


def independent_distribution(joint):
    """128D rho marginal times the full 16D stick/ball joint marginal."""
    joint = np.asarray(joint, dtype=np.float64)
    if joint.ndim != 2 or joint.shape[1] != 128:
        raise ValueError("Expected N by 128 distributions")
    shaped = joint.reshape(-1, 8, 4, 4)
    return (shaped.sum(axis=(2, 3))[:, :, None, None]
            * shaped.sum(axis=1)[:, None, :, :]).reshape(-1, 128)


def _features(value, n=None, width=None, name="features"):
    value = np.asarray(value)
    if (value.ndim != 2 or (n is not None and len(value) != n)
            or (width is not None and value.shape[1] != width)
            or not value.shape[1] or not np.issubdtype(value.dtype, np.floating)
            or not np.isfinite(value).all()):
        raise ValueError(f"Invalid {name} matrix")
    norms = np.linalg.norm(value.astype(np.float64), axis=1)
    if np.any((norms != 0) & ~np.isclose(norms, 1., rtol=1e-5, atol=1e-6)):
        raise ValueError(f"{name} must contain normalized (or zero) rows")
    return value


def _probabilities(value, n, width, name, allow_empty):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (n, width) or not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError(f"Invalid {name}")
    mass = value.sum(axis=1)
    good = np.isclose(mass, 1., rtol=0, atol=1e-10)
    if allow_empty:
        good |= mass == 0
    if not good.all():
        raise ValueError(f"Invalid probability mass in {name}")
    return value


def _indices(value, n):
    value = np.asarray(value)
    if value.ndim != 1 or value.dtype.kind not in "iu" or np.any(value < 0) or np.any(value >= n):
        raise ValueError("Query token indices must be a valid integer vector")
    if len(np.unique(value)) != len(value):
        raise ValueError("Duplicate query token index")
    return value.astype(np.int64, copy=False)


class RegionPairScorer:
    """Validate once and share query-pair matrices among all frozen variants."""

    def __init__(self, features, geometry, descriptors, learned_features=None):
        self.features = _features(features, width=512)
        self.n = len(self.features)
        self.histograms = _probabilities(geometry["histograms"], self.n, 16, "radial histograms", False)
        self.relative_area = np.asarray(geometry["relative_area"], dtype=np.float64)
        if (self.relative_area.shape != (self.n,) or not np.isfinite(self.relative_area).all()
                or np.any(self.relative_area <= 0) or np.any(self.relative_area > 1)):
            raise ValueError("Relative areas must be in (0,1]")
        self.probabilities = {
            "retained": _probabilities(descriptors["joint"], self.n, 128, "retained J", True),
            "all_points": _probabilities(descriptors["joint_all"], self.n, 128, "all-point J", True),
        }
        self.probabilities["independent"] = independent_distribution(self.probabilities["all_points"])
        self.roots = {key: np.sqrt(p) for key, p in self.probabilities.items()}
        self.valid = {key: p.sum(axis=1) > 0 for key, p in self.probabilities.items()}
        if learned_features is None:
            self.learned = {}
        elif isinstance(learned_features, Mapping):
            if not set(learned_features).issubset({"plain_mlp", "gated_mlp"}):
                raise ValueError("Unknown learned fusion family")
            self.learned = {name: _features(z, self.n, 128, name) for name, z in learned_features.items()}
        else:
            z = _features(learned_features, self.n, 128, "learned features")
            self.learned = {name: z for name in ("plain_mlp", "gated_mlp")}

    def components(self, query_indices, specs=None):
        """Compute one query's shared matrices, optionally only a selected subset.

        specs is a name-to-spec mapping (for example the one frozen test winner).
        Restricting it changes only which unused matrices are skipped.
        """
        q = _indices(query_indices, self.n)
        requested = frozen_grid() if specs is None else specs
        if not isinstance(requested, Mapping) or not requested:
            raise ValueError("Expected a nonempty name-to-spec mapping")
        active = [checked_spec(spec) for spec in requested.values()]
        structure_keys = {"independent" if spec["distribution"] == "independent" else spec["descriptor"]
                          for spec in active if spec["family"] in ("add_j", "replace_radial", "concat")}
        learned_keys = {spec["family"] for spec in active if spec["family"] in ("plain_mlp", "gated_mlp")}
        # Preserve the source h dtype and multiplication order for exact baseline replay.
        raw_cosine = self.features[q] @ self.features.T
        cosine = np.clip(raw_cosine, 0, 1)
        radial = np.abs(np.cumsum(self.histograms[q], axis=-1)[:, None]
                        - np.cumsum(self.histograms, axis=-1)[None]).sum(-1) / 16
        area_distance = .5 * np.abs(np.log(self.relative_area[q])[:, None] - np.log(self.relative_area)[None])
        area = .85 + .15 * np.exp(-area_distance / 1.5)
        result = {"raw_cosine": raw_cosine, "cosine": cosine, "area": area,
                  "baseline": original_pair_scores(self.features[q], self.features, self.histograms[q],
                                                   self.histograms, self.relative_area[q], self.relative_area),
                  "radial_factor": np.exp(-radial), "structure": {}, "learned": {}}
        for key, root in self.roots.items():
            if key not in structure_keys:
                continue
            bc = np.clip(root[q] @ root.T, 0., 1.)
            # Direct sum of squared differences follows the established ordinary
            # Hellinger convention and avoids cancellation for identical rows.
            distance = np.sqrt(np.maximum(.5 * np.square(root[q, None] - root[None]).sum(-1), 0.))
            valid = self.valid[key][q, None] & self.valid[key][None]
            result["structure"][key] = {"bc": bc, "distance": np.clip(distance, 0, 1), "valid": valid}
        result["learned"] = {name: np.clip(z[q] @ z.T, 0, 1) for name, z in self.learned.items()
                             if name in learned_keys}
        return result

    def scores(self, query_indices, spec):
        return score_components(self.components(query_indices, {"requested": spec}), spec)

    @staticmethod
    def score_components(components, spec):
        return score_components(components, spec)


def score_components(components, spec):
    spec = checked_spec(spec)
    family = spec["family"]
    if family == "baseline":
        return components["baseline"]
    if family in ("plain_mlp", "gated_mlp"):
        if family not in components["learned"]:
            raise ValueError(f"Missing frozen learned features for {family}")
        return components["learned"][family]
    key = "independent" if spec["distribution"] == "independent" else spec["descriptor"]
    structure = components["structure"][key]
    if family in ("add_j", "replace_radial"):
        base = (components["baseline"] if family == "add_j"
                else components["cosine"] * components["area"])
        factor = np.where(structure["valid"], np.exp(-spec["strength"] * structure["distance"]), 1.)
        return base * factor
    if family == "concat":
        weight = spec["structure_weight"]
        # Both nonempty branches have unit norm. This is exactly the cosine of
        # [sqrt(1-w)*h, sqrt(w)*sqrt(P)] for nonzero unit CNN vectors.
        score = np.clip((1-weight) * components["raw_cosine"] + weight * structure["bc"], 0, 1)
        score = np.where(structure["valid"], score, components["cosine"])
        return score * components["area"] if spec["area"] else score
    raise ValueError("Unsupported score family")


def pair_scores(query_indices, all_features, geometry, descriptors, spec, learned_features=None):
    """Convenience entry point; reuse RegionPairScorer for a complete query loop."""
    return RegionPairScorer(all_features, geometry, descriptors, learned_features).scores(query_indices, spec)


def independent_pair_scores(query_indices, all_features, geometry, descriptors, spec, learned_features=None):
    """Slow scalar reference, independent of production components/formula helpers.

    Intended for per-class source audits and synthetic checks, not full-grid runs.
    It deliberately builds each region pair separately and does not invoke the
    production matcher or the original scoring implementation.
    """
    spec = checked_spec(spec)
    h = np.asarray(all_features)
    q = _indices(query_indices, len(h))
    hist = np.asarray(geometry["histograms"], dtype=np.float64)
    areas = np.asarray(geometry["relative_area"], dtype=np.float64)
    source = "joint" if spec.get("descriptor") == "retained" else "joint_all"
    p = np.asarray(descriptors[source], dtype=np.float64)
    if spec.get("distribution") == "independent":
        transformed = []
        for row in p:
            cube = row.reshape(8, 4, 4)
            radial = np.array([cube[r].sum() for r in range(8)])
            shape = np.array([[cube[:, s, b].sum() for b in range(4)] for s in range(4)])
            transformed.append(np.array([radial[r]*shape[s, b]
                                         for r in range(8) for s in range(4) for b in range(4)]))
        p = np.asarray(transformed).reshape(-1, 128)
    family = spec["family"]
    z = None
    if family in ("plain_mlp", "gated_mlp"):
        if learned_features is None:
            raise ValueError("Reference requires learned features")
        z = np.asarray(learned_features[family] if isinstance(learned_features, Mapping) else learned_features)
    result = np.empty((len(q), len(h)), dtype=np.float64)
    for row, i in enumerate(q):
        for j in range(len(h)):
            if z is not None:
                result[row, j] = min(1., max(0., float(np.dot(z[i], z[j]))))
                continue
            raw = float(np.dot(h[i], h[j]))
            cos = min(1., max(0., raw))
            scale = .85 + .15*np.exp(-.5*abs(np.log(areas[i])-np.log(areas[j]))/1.5)
            if family == "baseline" or family == "add_j":
                dr = sum(abs(np.cumsum(hist[i])-np.cumsum(hist[j]))) / 16.
                base = cos*np.exp(-dr)*scale
            else:
                base = cos*scale
            if family == "baseline":
                result[row, j] = base
                continue
            available = p[i].sum() > 0 and p[j].sum() > 0
            if family in ("add_j", "replace_radial"):
                distance = np.sqrt(.5*sum((np.sqrt(p[i])-np.sqrt(p[j]))**2)) if available else 0.
                result[row, j] = base*np.exp(-spec["strength"]*distance)
            else:
                if available:
                    bc = min(1., max(0., sum(np.sqrt(p[i]*p[j]))))
                    w = spec["structure_weight"]
                    value = min(1., max(0., (1-w)*raw+w*bc))
                else:
                    value = cos
                result[row, j] = value*scale if spec["area"] else value
    return result
