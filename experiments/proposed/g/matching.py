"""G-only greedy one-to-one matching, aggregated as matched pairs / query count."""

from __future__ import annotations

import numpy as np

from wbm_retrieval.matching import greedy_assignment, pair_components


def method_settings(config, method):
    if method not in ("handcrafted", "proposed"):
        raise ValueError("Unknown G method")
    settings = dict(config["matching"])
    if settings.get("aggregation") != "query_region_count":
        raise ValueError("G requires query_region_count aggregation")
    thresholds = settings.pop("method_thresholds")
    if set(thresholds) != {"handcrafted", "proposed"}:
        raise ValueError("G requires explicit handcrafted and proposed thresholds")
    if set(thresholds[method]) != {"shape_threshold", "score_threshold"}:
        raise ValueError("Each method requires shape_threshold and score_threshold")
    settings.update(thresholds[method])
    validate_settings(settings)
    settings["representation"] = "zernike" if method == "handcrafted" else "learned"
    settings["zernike_distance_scale"] = config["handcrafted"]["distance_scale"]
    # Area is deliberately absent from the aggregation, including in audit records.
    settings.pop("weight", None)
    return settings


def validate_settings(settings):
    if settings.get("lambda_pos", 0) or settings.get("lambda_scale", 0):
        raise ValueError("G retains shape-only matching")
    for key in ("shape_threshold", "score_threshold"):
        if not 0 <= settings[key] <= 1:
            raise ValueError("G thresholds must be finite and lie in [0, 1]")


def count_similarity(pair_scores, settings, details=False):
    """Reference scalar rule; pair confidence orders matches but adds no weight.

    This counts the existing score-sorted greedy assignment, not a maximum
    cardinality matching. A zero score is never accepted, even at threshold zero.
    """
    validate_settings(settings)
    scores = np.asarray(pair_scores, dtype=np.float64)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("Pair scores must be a finite matrix")
    nq, nc = scores.shape
    scores = np.clip(scores, 0, 1)
    scores[scores < settings["shape_threshold"]] = 0
    matches = greedy_assignment(scores, settings["score_threshold"]) if nq and nc else []
    result = len(matches) / nq if nq else 0.0
    if not details:
        return result
    records = [{"query_token": q, "candidate_token": c, "pair_score": value,
                "count_contribution": 1, "similarity_contribution": 1 / nq}
               for q, c, value in matches]
    return result, records


def map_similarity(query, candidate, settings, details=False):
    validate_settings(settings)
    if not len(query["area"]) or not len(candidate["area"]):
        return (0.0, []) if details else 0.0
    scores, *_ = pair_components(query, candidate, settings)
    return count_similarity(scores, settings, details)


def batched_map_scores(similarities, candidate_indices, query_count, settings, *, return_counts=False):
    """Vectorized scalar rule; -1 padding is never counted as a candidate region."""
    validate_settings(settings)
    similarities = np.asarray(similarities)
    candidate_indices = np.asarray(candidate_indices)
    if similarities.ndim != 2 or similarities.shape[0] != query_count or not np.isfinite(similarities).all():
        raise ValueError("Expected finite similarities with one row per query region")
    if candidate_indices.ndim != 2 or not np.issubdtype(candidate_indices.dtype, np.integer):
        raise ValueError("Expected a padded integer candidate matrix")
    if np.any(candidate_indices < -1) or np.any(candidate_indices >= similarities.shape[1]):
        raise ValueError("Invalid candidate token index")
    count, width = candidate_indices.shape
    counts = np.zeros(count, dtype=np.int64)
    if not query_count or not width or not count or not similarities.shape[1]:
        result = np.zeros(count, dtype=np.float64)
        return (result, counts) if return_counts else result
    scores = np.clip(similarities[:, np.maximum(candidate_indices, 0)].transpose(1, 0, 2), 0, 1)
    scores *= (candidate_indices >= 0)[:, None, :]
    scores[scores < settings["shape_threshold"]] = 0
    flat = scores.reshape(count, -1)
    order = np.argsort(-flat, axis=1, kind="stable")
    used_q = np.zeros((count, query_count), dtype=bool)
    used_c = np.zeros((count, width), dtype=bool)
    rows = np.arange(count)
    for column in range(order.shape[1]):
        qi, ci = order[:, column] // width, order[:, column] % width
        value = flat[rows, order[:, column]]
        active = (value > 0) & (value >= settings["score_threshold"])
        if not active.any():
            break
        take = active & ~used_q[rows, qi] & ~used_c[rows, ci]
        counts[take] += 1
        used_q[rows[take], qi[take]] = True
        used_c[rows[take], ci[take]] = True
    result = counts / query_count
    return (result, counts) if return_counts else result
