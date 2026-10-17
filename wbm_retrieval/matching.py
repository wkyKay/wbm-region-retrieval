"""Score-sorted greedy one-to-one matching with a query-only denominator."""

from __future__ import annotations

import numpy as np


def pair_components(query, candidate, config):
    q = np.asarray(query["embedding"], dtype=np.float64)
    c = np.asarray(candidate["embedding"], dtype=np.float64)
    if config.get("representation", "learned") == "zernike":
        distance = np.linalg.norm(q[:, None] - c[None], axis=-1)
        shape = np.exp(-distance / config["zernike_distance_scale"])
    else:
        # The encoders return raw GAP features; normalize explicitly for cosine.
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        c = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
        shape = np.clip(q @ c.T, 0.0, 1.0)
    distance2 = ((query["position"][:, None] - candidate["position"][None]) ** 2).sum(-1)
    position = np.exp(-distance2 / config["sigma_pos"] ** 2)
    qscale = np.maximum(query["scale"], 1e-8)
    cscale = np.maximum(candidate["scale"], 1e-8)
    scale = np.exp(-np.abs(np.log(qscale[:, None] / cscale[None])) / config["sigma_scale"])
    lp, ls = config["lambda_pos"], config["lambda_scale"]
    if min(lp, ls) < 0 or lp + ls > 0.25 + 1e-12:
        raise ValueError("Geometry weights must be nonnegative and sum to <= 0.25")
    score = shape * (1 - lp * (1 - position) - ls * (1 - scale))
    score[shape < config["shape_threshold"]] = 0
    return score, shape, position, scale


def greedy_assignment(scores, minimum=0.0):
    """Sort by descending pair score, then query/candidate index to resolve ties."""
    scores = np.asarray(scores)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("Pair scores must be a finite matrix")
    qi, ci = np.indices(scores.shape)
    order = np.lexsort((ci.ravel(), qi.ravel(), -scores.ravel()))
    used_q, used_c, matches = set(), set(), []
    for index in order:
        q, c, score = int(qi.ravel()[index]), int(ci.ravel()[index]), float(scores.ravel()[index])
        if score <= 0 or score < minimum:
            break
        if q not in used_q and c not in used_c:
            matches.append((q, c, score))
            used_q.add(q)
            used_c.add(c)
    return matches


def map_similarity(query, candidate, config, details=False):
    if not len(query["area"]) or not len(candidate["area"]):
        return (0.0, []) if details else 0.0
    score, shape, position, scale = pair_components(query, candidate, config)
    matches = greedy_assignment(score, config["score_threshold"])
    area = np.asarray(query["area"], dtype=float)
    weights = np.ones_like(area) if config["weight"] == "uniform" else np.sqrt(area) if config["weight"] == "sqrt_area" else area
    total = sum(weights[q] * value for q, _, value in matches) / weights.sum()
    if not details:
        return float(total)
    records = [{"query_token": q, "candidate_token": c, "score": value,
                "shape": float(shape[q, c]), "position_affinity": float(position[q, c]),
                "scale_affinity": float(scale[q, c]), "query_weight": float(weights[q])}
               for q, c, value in matches]
    return float(total), records


def group_tokens(metadata, embeddings):
    if len(embeddings) != len(metadata["area"]):
        raise ValueError("Embedding/token count mismatch")
    result = {}
    for i, sample_id in enumerate(metadata["map_ids"]):
        start, end = metadata["offsets"][i:i+2]
        result[int(sample_id)] = {key: metadata[key][start:end] for key in ("area", "position", "scale")}
        result[int(sample_id)]["embedding"] = embeddings[start:end]
    return result
