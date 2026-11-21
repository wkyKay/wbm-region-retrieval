"""Read-only diagnostics for the actual token-instance NCE negative mechanism.

Class labels only identify diagnostic subsets. Nothing here updates a model or
changes its historical sampling. A single-label wafer supplies a region-label
proxy; it is not pixel-level supervision. Counts concern the complete train
token bank, including mixed-label maps, rather than a class-filtered bank.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
import math

import numpy as np


def _integer_vector(value, name, *, minimum=0, allow_empty=False):
    value = np.asarray(value)
    if (value.ndim != 1 or value.dtype.kind not in "iu"
            or (not allow_empty and not len(value)) or np.any(value < minimum)):
        raise ValueError(f"{name} must be a nonempty integer vector >= {minimum}")
    return value.astype(np.int64, copy=False)


def _metadata(map_ids, singleton_labels):
    ids = _integer_vector(map_ids, "map_ids")
    labels = _integer_vector(singleton_labels, "singleton_labels", minimum=-1)
    if labels.shape != ids.shape:
        raise ValueError("Need one singleton label per token")
    owners = {}
    for sid, label in zip(ids.tolist(), labels.tolist()):
        if sid in owners and owners[sid] != label:
            raise ValueError("Tokens from one map must share its diagnostic label")
        owners[sid] = label
    return ids, labels


def token_metadata(token_map_ids, map_labels):
    """Return JSON-safe original IDs and singleton IDs in train-token order.

    ``map_labels`` is either a full source-row-indexed binary multi-hot matrix
    or a mapping from original sample ID to binary multi-hot row. Normal and
    mixed-label maps are retained with singleton label -1, never filtered out.
    """
    ids = _integer_vector(token_map_ids, "token_map_ids")
    width, owner_labels = None, {}
    if not isinstance(map_labels, Mapping):
        matrix = np.asarray(map_labels)
        if matrix.ndim != 2 or not matrix.shape[1] or int(ids.max()) >= len(matrix):
            raise ValueError("Labels must be indexed by original sample ID")
        map_labels = matrix
    for sid in np.unique(ids):
        try:
            row = np.asarray(map_labels[int(sid)])
        except (KeyError, IndexError) as exc:
            raise ValueError(f"Missing label for original sample ID {sid}") from exc
        if (row.ndim != 1 or not len(row) or row.dtype.kind not in "biuf"
                or not np.isfinite(row).all() or np.any((row != 0) & (row != 1))
                or (width is not None and width != len(row))):
            raise ValueError("Every map label must be a consistent binary multi-hot row")
        width = len(row)
        owner_labels[int(sid)] = int(np.flatnonzero(row)[0]) if row.sum() == 1 else -1
    return {"token_indices": list(range(len(ids))), "map_ids": ids.tolist(),
            "singleton_labels": [owner_labels[int(sid)] for sid in ids]}


def _at_least_one(successes, available, draws):
    if successes == 0 or draws == 0:
        return 0.
    if successes == available:
        return 1.
    return float(-math.expm1(draws * math.log1p(-successes / available)))


def _mean(values):
    values = [float(x) for x in values if x is not None]
    return float(np.mean(values)) if values else None


def _summarize(rows, names):
    by_map = defaultdict(list)
    for row in rows:
        by_map[row["map_id"]].append(row)
    by_class = defaultdict(list)
    for items in by_map.values():
        by_class[items[0]["singleton_label"]].append(
            {key: _mean(item[key] for item in items) for key in names})
    per_class = [
        {"singleton_label": int(label), "maps": len(items),
         "metrics": {key: _mean(item[key] for item in items) for key in names}}
        for label, items in sorted(by_class.items())]
    return {
        "queries": len(rows), "maps": len(by_map),
        "token_mean": {key: _mean(row[key] for row in rows) for key in names},
        "map_mean": {key: _mean(_mean(row[key] for row in items)
                                for items in by_map.values()) for key in names},
        "class_balanced": {key: _mean(item["metrics"][key] for item in per_class)
                           for key in names},
        "per_class": per_class,
    }


def audit_negative_sampling(token_map_ids, singleton_labels, batches, *, num_negatives=5000):
    """Exact conditional sampling expectations for supplied diagnostic batches.

    The original MemoryBank.negative_indices samples uniformly WITH replacement
    from all bank tokens except current batch token IDs. Other regions of the
    same wafer remain eligible. A full-bank batch uses the original ef_training
    fallback: each query excludes only itself. Supply diagnostic batch IDs, not
    an assumed historical batch trace. Every supplied token may appear once.
    ``batches`` may cover a subset of train queries; the negative bank is always
    the complete metadata array. Only singleton-map queries produce rows.
    """
    ids, labels = _metadata(token_map_ids, singleton_labels)
    if (isinstance(num_negatives, (bool, np.bool_))
            or not isinstance(num_negatives, (int, np.integer)) or num_negatives < 1):
        raise ValueError("num_negatives must be a positive integer")
    draws, n = int(num_negatives), len(ids)
    class_counts, map_counts = Counter(labels.tolist()), Counter(ids.tolist())
    rows, seen, batch_count, fallback_batches = [], set(), 0, 0
    for batch_number, batch in enumerate(batches):
        indices = _integer_vector(batch, "batch token indices")
        if (np.any(indices >= n) or len(np.unique(indices)) != len(indices)
                or seen.intersection(indices.tolist())):
            raise ValueError("Batch tokens must be in range, unique and not repeated across batches")
        seen.update(indices.tolist())
        batch_count += 1
        fallback = len(indices) == n
        fallback_batches += int(fallback)
        excluded_class = Counter(labels[indices].tolist())
        excluded_map = Counter(ids[indices].tolist())
        for q in indices:
            label, sid = int(labels[q]), int(ids[q])
            if label < 0:
                continue
            available = n - (1 if fallback else len(indices))
            if available <= 0:
                raise ValueError("No negatives left, matching the original training error")
            same_map = map_counts[sid] - (1 if fallback else excluded_map[sid])
            same_label = class_counts[label] - (1 if fallback else excluded_class[label])
            cross_map = same_label - same_map
            non_singleton = class_counts[-1] - (0 if fallback else excluded_class[-1])
            row = {"token_index": int(q), "map_id": sid, "singleton_label": label,
                   "batch_index": batch_number, "batch_size": len(indices),
                   "exclusion_rule": "self_only_full_batch_fallback" if fallback else "batch_token_ids_only",
                   "negative_pool_tokens": available,
                   "same_map_negative_tokens": same_map,
                   "sameclass_crossmap_negative_tokens": cross_map,
                   "other_singleton_negative_tokens": available - same_label - non_singleton,
                   "nonsingleton_negative_tokens": non_singleton}
            for prefix, count in (("same_map", same_map), ("sameclass_crossmap", cross_map),
                                  ("sameclass_including_same_map", same_label)):
                row[f"expected_{prefix}_count"] = draws * count / available
                row[f"probability_{prefix}_at_least_one"] = _at_least_one(count, available, draws)
            rows.append(row)
    names = [f"{kind}_{prefix}_{suffix}" for prefix in
             ("same_map", "sameclass_crossmap", "sameclass_including_same_map")
             for kind, suffix in (("expected", "count"), ("probability", "at_least_one"))]
    summary = _summarize(rows, names)
    summary.update({"bank_tokens": n, "bank_maps": len(map_counts), "num_negatives": draws,
                    "diagnostic_batches": batch_count, "supplied_query_tokens": len(seen),
                    "full_bank_fallback_batches": fallback_batches, "replacement": True,
                    "scope": "exact conditional expectations for supplied diagnostic batches, not historical draws",
                    "label_use": "diagnostic singleton-wafer proxy only; all token classes remain in bank"})
    return {"summary": summary, "rows": rows}


def sameclass_crossmap_mask(token_map_ids, singleton_labels, query_indices, negative_indices):
    """Boolean B-by-K mask; repeated sampled negatives retain multiplicity."""
    ids, labels = _metadata(token_map_ids, singleton_labels)
    q = _integer_vector(query_indices, "query_indices")
    neg = _integer_vector(negative_indices, "negative_indices")
    if np.any(q >= len(ids)) or np.any(neg >= len(ids)):
        raise ValueError("Token indices exceed metadata")
    return ((labels[q, None] >= 0) & (labels[q, None] == labels[None, neg])
            & (ids[q, None] != ids[None, neg]))


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def singleton_prototypes(memory, token_map_ids, singleton_labels, query_indices):
    """Cross-map, map-balanced singleton centroids of normalized frozen memory.

    Each source wafer contributes the mean of its normalized tokens once.
    The query wafer is excluded entirely. Zero vectors mark undefined (absent
    or cancelling) prototypes; the gradient report then emits null projections.
    """
    ids, labels = _metadata(token_map_ids, singleton_labels)
    q = _integer_vector(query_indices, "query_indices")
    bank = _numpy(memory)
    if (bank.ndim != 2 or len(bank) != len(ids) or not bank.shape[1]
            or not np.isfinite(bank).all() or np.any(q >= len(ids))):
        raise ValueError("Memory features must align with full train-token metadata")
    bank = bank / np.maximum(np.linalg.norm(bank, axis=1, keepdims=True), 1e-12)
    sums, counts, owner = {}, Counter(ids.tolist()), {}
    for i, sid in enumerate(ids.tolist()):
        if sid not in sums:
            sums[sid] = np.zeros(bank.shape[1], dtype=np.float64)
        sums[sid] += bank[i]
        owner[sid] = int(labels[i])
    class_sum, class_count, means = {}, Counter(), {}
    for sid, total in sums.items():
        means[sid] = total / counts[sid]
        label = owner[sid]
        if label < 0:
            continue
        if label not in class_sum:
            class_sum[label] = np.zeros(bank.shape[1], dtype=np.float64)
        class_sum[label] += means[sid]
        class_count[label] += 1
    result = np.zeros((len(q), bank.shape[1]), dtype=np.float64)
    for i, index in enumerate(q):
        sid, label = int(ids[index]), int(labels[index])
        if label >= 0 and class_count[label] > 1:
            result[i] = (class_sum[label] - means[sid]) / (class_count[label] - 1)
    return result


def nce_gradient_diagnostic(current_z, anchors, negatives, sameclass_mask, prototypes, *,
                            temperature, verify_torch=False, return_gradients=False):
    """Exact analytic feature-space derivative of original normalized NCE.

    Anchors/negatives/prototypes are constants. Per-query derivatives are of an
    individual CE (the historical batch-mean derivative times batch size).
    Full loss is the original nce_loss batch mean. Removing selected negatives
    changes the softmax denominator, so full-minus-masked is NOT the direct
    same-class logit term. We report both, along with the actual full gradient.
    Positive descent projection means an infinitesimal z step increases cosine
    to its frozen cross-map class centroid; negative means decreases. These are
    local feature-space pressures, not actual network updates or causal proof.
    Optional torch verification imports the actual historical nce_loss.
    """
    z, a, neg, proto = map(_numpy, (current_z, anchors, negatives, prototypes))
    mask = np.asarray(sameclass_mask)
    if (z.ndim != 2 or not all(z.shape) or a.shape != z.shape or proto.shape != z.shape
            or neg.ndim != 2 or not len(neg) or neg.shape[1] != z.shape[1]
            or mask.shape != (len(z), len(neg)) or mask.dtype.kind != "b"
            or any(not np.isfinite(x).all() for x in (z, a, neg, proto))):
        raise ValueError("Finite aligned BxD features, KxD negatives and boolean BxK mask are required")
    if isinstance(temperature, (bool, np.bool_)) or not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    t = float(temperature)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Current z norms must exceed normalize epsilon for differentiable cosine diagnostics")
    unit = z / norms
    anchor_unit = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    neg_unit = neg / np.maximum(np.linalg.norm(neg, axis=1, keepdims=True), 1e-12)
    positive = (unit * anchor_unit).sum(axis=1, keepdims=True)
    logits = np.concatenate((positive, unit @ neg_unit.T), axis=1) / t

    def probabilities(values):
        shift = values.max(axis=1, keepdims=True)
        weights = np.exp(values - shift)
        totals = weights.sum(axis=1, keepdims=True)
        return weights / totals, (np.log(totals) + shift - values[:, :1]).ravel()

    prob, loss = probabilities(logits)
    masked_logits = logits.copy()
    masked_logits[:, 1:][mask] = -np.inf
    masked_prob, masked_loss = probabilities(masked_logits)

    def pullback(direction):
        return (direction - unit * (direction * unit).sum(axis=1, keepdims=True)) / norms

    full = pullback(((prob[:, :1] - 1.) * anchor_unit + prob[:, 1:] @ neg_unit) / t)
    masked = pullback(((masked_prob[:, :1] - 1.) * anchor_unit
                       + masked_prob[:, 1:] @ neg_unit) / t)
    same = pullback(((prob[:, 1:] * mask) @ neg_unit) / t)
    attractive = pullback((prob[:, :1] - 1.) * anchor_unit / t)
    other = pullback(((prob[:, 1:] * ~mask) @ neg_unit) / t)
    proto_norm = np.linalg.norm(proto, axis=1, keepdims=True)
    proto_unit = proto / np.maximum(proto_norm, 1e-12)
    toward = pullback(proto_unit)
    toward_norm = np.linalg.norm(toward, axis=1)
    valid_direction = (proto_norm[:, 0] > 1e-12) & (toward_norm > 1e-12)
    toward_unit = toward / np.maximum(toward_norm[:, None], 1e-12)
    gradients = {"full": full, "masked": masked, "sameclass_negative_term": same,
                 "full_minus_masked": full - masked, "positive_term": attractive,
                 "other_negative_term": other}
    rows = []
    for i in range(len(z)):
        row = {"query_index": i, "loss_full": float(loss[i]), "loss_masked": float(masked_loss[i]),
               "sampled_sameclass_crossmap_count": int(mask[i].sum()),
               "positive_probability": float(prob[i, 0]),
               "sameclass_negative_probability_mass": float(prob[i, 1:][mask[i]].sum()),
               "prototype_direction_defined": bool(valid_direction[i]),
               "prototype_cosine": float(unit[i] @ proto_unit[i]) if proto_norm[i, 0] > 1e-12 else None}
        for name, gradient in gradients.items():
            row[f"{name}_gradient_norm"] = float(np.linalg.norm(gradient[i]))
            row[f"{name}_descent_toward_class"] = (
                float(-gradient[i] @ toward_unit[i]) if valid_direction[i] else None)
        f_norm, m_norm = np.linalg.norm(full[i]), np.linalg.norm(masked[i])
        row["full_masked_gradient_cosine"] = (float(np.clip(full[i] @ masked[i] / (f_norm * m_norm), -1., 1.))
                                               if f_norm > 1e-12 and m_norm > 1e-12 else None)
        rows.append(row)
    names = [key for key in rows[0] if key not in ("query_index", "prototype_direction_defined")]
    summary = {"queries": len(z), "negative_draws": len(neg), "temperature": t,
               "loss_full": float(loss.mean()), "loss_masked": float(masked_loss.mean()),
               "valid_prototype_directions": int(valid_direction.sum()),
               "mean": {key: _mean(row[key] for row in rows) for key in names},
               "gradient_decomposition_max_error": float(np.abs(full - same - attractive - other).max()),
               "gradient_unit": "individual-example CE derivative with respect to current unnormalized z",
               "scope": "instantaneous frozen-memory feature-space diagnostic; not retraining or causal attribution",
               "torch_verification": None}
    for name in ("full", "masked", "sameclass_negative_term", "full_minus_masked"):
        values = [row[f"{name}_descent_toward_class"] for row in rows if row["prototype_direction_defined"]]
        summary[f"{name}_away_from_class_fraction"] = (_mean(value < -1e-12 for value in values))
    if verify_torch:
        import torch
        from torch.nn import functional as functional
        from experiments.proposed.main.method import nce_loss

        tz = torch.tensor(z, dtype=torch.float64, requires_grad=True)
        ta, tn = torch.tensor(a, dtype=torch.float64), torch.tensor(neg, dtype=torch.float64)
        historical = nce_loss(ta, tz, tn, t)
        historical_gradient = torch.autograd.grad(historical * len(z), tz)[0].detach().numpy()
        tz2 = torch.tensor(z, dtype=torch.float64, requires_grad=True)
        uz, ua, un = (functional.normalize(x, dim=1) for x in (tz2, ta, tn))
        tl = torch.cat(((uz * ua).sum(1, keepdim=True), uz @ un.T), 1) / t
        tm = torch.cat((torch.zeros((len(z), 1), dtype=torch.bool), torch.tensor(mask)), 1)
        masked_torch_loss = functional.cross_entropy(tl.masked_fill(tm, -torch.inf), torch.zeros(len(z), dtype=torch.long))
        masked_torch_gradient = torch.autograd.grad(masked_torch_loss * len(z), tz2)[0].detach().numpy()
        check = {"historical_loss_absolute_error": abs(float(historical.detach()) - float(loss.mean())),
                 "historical_gradient_max_error": float(np.abs(historical_gradient - full).max()),
                 "masked_loss_absolute_error": abs(float(masked_torch_loss.detach()) - float(masked_loss.mean())),
                 "masked_gradient_max_error": float(np.abs(masked_torch_gradient - masked).max())}
        if any(value > 1e-9 for value in check.values()):
            raise AssertionError(f"Analytic diagnostic disagrees with original torch NCE: {check}")
        summary["torch_verification"] = check
    result = {"summary": summary, "rows": rows}
    if return_gradients:
        result["gradients"] = {name: value.tolist() for name, value in gradients.items()}
    return result
