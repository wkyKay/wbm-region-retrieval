"""Original-mask descriptors and exact weighted one-to-one assignment."""
from itertools import permutations

import numpy as np

from experiments.proposed.f_global_fusion.pipeline import BranchScores
from experiments.proposed.f_segmentation_geometry.scoring import pair_scores
from wbm_retrieval.diagnostics import batched_map_scores

SETTINGS = {
    "alpha_global": 0.5,
    "density_gammas": [0., 2., 5., 10., 20.],
    "angular_gammas": [0., .25, .5, 1., 2.],
    "outer_radius_min": .65,
    "harmonics": [1, 2, 3, 4],
    "angular_distance": "mean absolute difference of normalized circular moment magnitudes",
    "empty_outer_policy": "zero angular distance if either wafer has no outer defect",
    "selection": "valid/all/exact/Precision@10; ties choose smaller strength",
}


def variants():
    result = {"baseline": ("baseline", 0.), "optimal": ("optimal", 0.)}
    for family in ("density", "angular"):
        result.update({f"{family}_{gamma}": (family, gamma)
                       for gamma in SETTINGS[f"{family}_gammas"] if gamma > 0})
    return result


def moment_magnitudes(points, center, harmonics=(1, 2, 3, 4)):
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.zeros(len(harmonics))
    delta = points - np.asarray(center)
    angles = np.arctan2(delta[:, 0], delta[:, 1])
    return np.abs(np.exp(1j * angles[:, None] * np.asarray(harmonics)).mean(axis=0))


def describe(raw):
    """Measure all original defects, never a crop, resized image or Top-4 union."""
    raw = np.asarray(raw)
    if raw.ndim != 2:
        raise ValueError("Expected a 2D raw wafer")
    support = np.argwhere((raw == 1) | (raw == 2))
    defects = np.argwhere(raw == 2)
    if not len(support):
        raise ValueError("Wafer has no valid support")
    center = (support.min(0) + support.max(0)) / 2
    radius = np.linalg.norm(support-center, axis=1).max()
    rho = np.linalg.norm(defects-center, axis=1)/radius if radius > 0 else np.zeros(len(defects))
    outer = defects[rho >= SETTINGS["outer_radius_min"]]
    return len(defects)/len(support), moment_magnitudes(outer, center, SETTINGS["harmonics"]), len(outer)


def optimal_map_scores(pairs, candidate_indices, query_area):
    """Exact max sum(sqrt(area_i)*s_ij), with zero-valued dummy assignments.

    Top-4 makes exhaustive 4! assignment practical, vectorized over all candidates.
    Rectangular and partial matchings extend to a padded square permutation.
    """
    count, width = candidate_indices.shape
    nq = len(query_area)
    if nq > 4 or width > 4:
        raise ValueError("This frozen ablation requires Top-4")
    if not nq or not width:
        return np.zeros(count)
    weights = np.sqrt(np.asarray(query_area, dtype=float))
    scores = np.clip(pairs[:, np.maximum(candidate_indices, 0)].transpose(1, 0, 2), 0, 1)
    scores *= (candidate_indices >= 0)[:, None, :]
    n = max(nq, width)
    weighted = np.zeros((count, n, n))
    weighted[:, :nq, :width] = scores * weights[None, :, None]
    best = np.zeros(count)
    for perm in permutations(range(n)):
        best = np.maximum(best, weighted[:, np.arange(n), perm].sum(axis=1))
    return best / weights.sum()


class Scores(BranchScores):
    def __init__(self, feature_dir, global_ids, global_features, gallery, matching, descriptors):
        super().__init__(feature_dir, global_ids, global_features, gallery, matching)
        if any(matching[k] != 0 for k in ("lambda_pos", "lambda_scale", "shape_threshold", "score_threshold")):
            raise ValueError("Expected original zero-threshold matching settings")
        if matching["weight"] != "sqrt_area" or self.padded.shape[1] > 4:
            raise ValueError("Expected Top-4 sqrt(area) aggregation")
        self.descriptor_rows = {int(s): i for i, s in enumerate(descriptors["sample_ids"])}
        if len(self.descriptor_rows) != len(gallery) or set(self.descriptor_rows) != set(map(int, gallery)):
            raise ValueError("Descriptor IDs must equal the complete valid gallery")
        self.descriptors = descriptors

    def components(self, query, candidates):
        index = self.rows[int(query)]
        meta = self.meta
        tokens = np.arange(meta["offsets"][index], meta["offsets"][index+1])
        pairs = pair_scores(self.features[tokens], self.features, meta["histograms"][tokens],
                            meta["histograms"], meta["relative_area"][tokens], meta["relative_area"])
        indices = self.padded[[self.rows[int(c)] for c in candidates]]
        local = batched_map_scores(pairs, indices, meta["area"][tokens], self.matching)
        optimal = optimal_map_scores(pairs, indices, meta["area"][tokens])
        whole = np.clip(self.global_features[[self.global_rows[int(c)] for c in candidates]] @
                        self.global_features[self.global_rows[int(query)]], 0, 1)
        rows = [self.descriptor_rows[int(c)] for c in candidates]
        qi = self.descriptor_rows[int(query)]
        d = self.descriptors
        density = np.abs(d["density"][rows]-d["density"][qi])
        angular = np.abs(d["moments"][rows]-d["moments"][qi]).mean(axis=1)
        angular *= (d["outer_count"][rows] > 0) & (d["outer_count"][qi] > 0)
        return {"baseline": .5*local+.5*whole, "optimal": .5*optimal+.5*whole,
                "density_distance": density, "angular_distance": angular,
                "f_score": local, "optimal_f_score": optimal, "global_score": whole}


def variant_score(values, family, gamma):
    if family == "optimal":
        return values["optimal"]
    if family == "baseline" or gamma == 0:
        return values["baseline"]
    return values["baseline"] * np.exp(-gamma*values[f"{family}_distance"])
