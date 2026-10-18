"""Read-only valid-split diagnostics for saved local models.

This module never calls prepare(), train(), an optimizer or backward(). All
features, manifests, figures and evaluations belong to a separate output tree.
"""

from __future__ import annotations

import csv
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import gc
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .common import csv_rows, digest, file_hash, iter_csv, read_json, write_csv, write_json
from .evaluation import bootstrap_interval, query_metrics
from .models import Encoder, Projector
from .training import load_checkpoint, parameter_hash, training_signature


VARIANTS = tuple(f"{model}_{space}" for model in ("best", "last", "random") for space in ("h512", "z128"))


def source_snapshot(root):
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*")) if p.is_file()}


def guard_output(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Diagnostic output must be outside the experiment, and must not contain the experiment")


def input_hashes(config_path, config):
    source = Path(config["output_dir"])
    paths = [Path(config_path), Path(config["split_manifest"]), source / "cache/prepared.json",
             source / "cache/valid_regions.npz", source / "cache/valid_patches.npy",
             source / "models/local/best.pt", source / "models/local/last.pt"]
    paths += [source / "models/local" / name for name in ("history.json", "training_summary.json")
              if (source / "models/local" / name).is_file()]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required existing diagnostic input is missing: {path}")
    return {str(path.resolve()): file_hash(path) for path in paths}


def valid_source(config):
    source = Path(config["output_dir"])
    with np.load(source / "cache/valid_regions.npz", allow_pickle=False) as archive:
        metadata = {key: archive[key] for key in archive.files}
    rows = csv_rows(config["split_manifest"])
    columns = sorted((key for key in rows[0] if key.startswith("label_") and key[6:].isdigit()),
                     key=lambda key: int(key[6:]))
    labels = {int(row["sample_id"]): np.array([int(row[key]) for key in columns], dtype=np.uint8)
              for row in rows if row["split"] == "valid"}
    ids = metadata["map_ids"].astype(np.int64)
    if len(ids) != len(set(ids.tolist())) or set(ids.tolist()) != set(labels) or len(ids) < 2:
        raise ValueError("The valid cache must match all valid manifest IDs, with at least two maps")
    if any(not label.any() or not np.isin(label, [0, 1]).all() for label in labels.values()):
        raise ValueError("Invalid valid-split labels")
    offsets = metadata["offsets"]
    patches = np.load(source / "cache/valid_patches.npy", mmap_mode="r")
    if (len(offsets) != len(ids) + 1 or offsets[0] != 0 or np.any(np.diff(offsets) < 0)
            or offsets[-1] != len(patches) or len(metadata["area"]) != len(patches)
            or patches.shape[1:] != (1, config["input_size"], config["input_size"])):
        raise ValueError("Inconsistent valid token offsets, metadata or patch shape")
    if np.count_nonzero(np.diff(offsets)) < 2:
        raise ValueError("Region diagnostics require tokens from at least two valid wafers")
    owners = np.repeat(ids, np.diff(offsets))
    local_ids = np.concatenate([np.arange(end - start) for start, end in zip(offsets[:-1], offsets[1:])])
    return metadata, labels, owners, local_ids


def label_text(label):
    return "+".join(str(i) for i in np.flatnonzero(label))


def freeze_protocol(output, metadata, labels, owners, settings):
    """Sample without labels; freeze the exact same pools/pairs for all variants."""
    folder = output / "protocol"
    if (folder / "manifest.json").is_file():
        manifest = read_json(folder / "manifest.json")
        if manifest["settings"] != settings:
            raise ValueError("Frozen diagnostic protocol settings changed; use a new output directory")
        for name, expected in manifest["sha256"].items():
            if file_hash(folder / name) != expected:
                raise ValueError(f"Frozen diagnostic protocol was modified: {name}")
    else:
        rng = np.random.default_rng(settings["seed"])
        ids = np.sort(metadata["map_ids"])
        query_count = min(settings["queries"] or len(ids), len(ids))
        queries = np.sort(rng.choice(ids, query_count, replace=False))
        candidate_count = min(settings["candidates"] or len(ids) - 1, len(ids) - 1)
        write_csv(folder / "queries.csv", ({"sample_id": int(q)} for q in queries), ["sample_id"])
        write_csv(folder / "candidates.csv",
                  ({"query_id": int(q), "candidate_id": int(c)} for q in queries
                   for c in np.sort(rng.choice(ids[ids != q], candidate_count, replace=False))),
                  ["query_id", "candidate_id"])
        write_csv(folder / "valid_maps.csv",
                  ({"sample_id": int(i), "labels": label_text(labels[int(i)]),
                    "num_labels": int(labels[int(i)].sum())} for i in ids),
                  ["sample_id", "labels", "num_labels"])
        count = len(owners)
        positive_ids = np.sort(rng.choice(count, min(settings["pair_samples"], count), replace=False))
        rotations = rng.integers(1, 4, len(positive_ids))
        write_csv(folder / "positive_pairs.csv", ({"token_id": int(i), "rotation_k": int(k)}
                  for i, k in zip(positive_ids, rotations)), ["token_id", "rotation_k"])
        a = rng.integers(count, size=settings["pair_samples"])
        b = rng.integers(count, size=len(a))
        bad = owners[a] == owners[b]
        while bad.any():
            b[bad] = rng.integers(count, size=int(bad.sum()))
            bad = owners[a] == owners[b]
        write_csv(folder / "random_pairs.csv", ({"query_token": int(i), "candidate_token": int(j)}
                  for i, j in zip(a, b)), ["query_token", "candidate_token"])
        # Balance visual-review anchors between single/mixed wafers when available.
        anchors = []
        for single in (True, False):
            eligible = np.array([i for i, owner in enumerate(owners) if (labels[int(owner)].sum() == 1) == single])
            quota = (settings["neighbor_anchors"] + int(single)) // 2
            anchors.extend(rng.choice(eligible, min(quota, len(eligible)), replace=False).tolist())
        remaining = np.setdiff1d(np.arange(count), anchors)
        missing = min(settings["neighbor_anchors"], count) - len(anchors)
        anchors.extend(rng.choice(remaining, missing, replace=False).tolist())
        write_csv(folder / "neighbor_anchors.csv", ({"token_id": int(i)} for i in sorted(anchors)), ["token_id"])
        names = ["queries.csv", "candidates.csv", "valid_maps.csv", "positive_pairs.csv",
                 "random_pairs.csv", "neighbor_anchors.csv"]
        manifest = {"settings": settings, "split": "valid", "sampling": "label-blind uniform without replacement per query",
                    "queries": query_count, "candidates_per_query": candidate_count,
                    "positive_pairs": len(positive_ids), "random_pairs": len(a),
                    "random_pairs_note": "with replacement, excluding the same wafer; not region semantic ground truth",
                    "sha256": {name: file_hash(folder / name) for name in names}}
        write_json(folder / "manifest.json", manifest)
    queries = [int(row["sample_id"]) for row in csv_rows(folder / "queries.csv")]
    candidates = {q: [] for q in queries}
    for row in iter_csv(folder / "candidates.csv"):
        candidates[int(row["query_id"])].append(int(row["candidate_id"]))
    if not set(queries) <= set(labels):
        raise ValueError("Diagnostic query outside valid")
    for q, values in candidates.items():
        if (len(values) != manifest["candidates_per_query"] or len(set(values)) != len(values)
                or q in values or not set(values) <= set(labels)):
            raise ValueError("Invalid frozen valid candidate pool")
    positive = csv_rows(folder / "positive_pairs.csv")
    pairs = csv_rows(folder / "random_pairs.csv")
    anchors = [int(row["token_id"]) for row in csv_rows(folder / "neighbor_anchors.csv")]
    return SimpleNamespace(queries=queries, candidates=candidates, labels=labels), manifest, positive, pairs, anchors


class DiagnosticImages(Dataset):
    def __init__(self, path, positive=None):
        self.path, self.array, self.positive = str(path), None, positive
        self.count = len(positive) if positive is not None else len(np.load(path, mmap_mode="r"))

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if self.array is None:
            self.array = np.load(self.path, mmap_mode="r")
        token = int(self.positive[index]["token_id"]) if self.positive is not None else index
        patch = np.array(self.array[token], dtype=np.float32)
        if self.positive is not None:
            patch = np.rot90(patch, int(self.positive[index]["rotation_k"]), axes=(-2, -1)).copy()
        return torch.from_numpy(patch)

    def __getstate__(self):
        return {**self.__dict__, "array": None}


def normalized(values):
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


@torch.inference_mode()
def extract_features(model, source, output, config, device, positive, batch_size, workers, fingerprint):
    """Write only diagnostic-owned arrays; cached main-experiment features are untouched."""
    marker = output / "complete.json"
    if marker.is_file():
        cached = read_json(marker)
        if cached["fingerprint"] != fingerprint:
            raise ValueError("Diagnostic feature cache mismatch")
        if all((output / name).is_file() and file_hash(output / name) == expected
               for name, expected in cached["sha256"].items()):
            print(f"Reusing diagnostic features: {output.name}", flush=True)
            return
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    checksums = {}
    for view, pairs in (("original", None), ("rotated", positive)):
        dataset = DiagnosticImages(source / "cache/valid_patches.npy", pairs)
        arrays = {space: np.lib.format.open_memmap(output / f"{view}_{space}.npy.tmp", mode="w+", dtype=np.float32,
                                                shape=(len(dataset), dim)) for space, dim in (("h512", 512), ("z128", 128))}
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=False,
                                 pin_memory=device.type == "cuda")
        offset = 0
        for batch in data_loader:
            h = model["encoder"](batch.to(device))
            z = model["projector"](h)
            for space, value in (("h512", h), ("z128", z)):
                if not torch.isfinite(value).all():
                    raise FloatingPointError("Non-finite inference features")
                arrays[space][offset:offset + len(batch)] = value.cpu().numpy()
            offset += len(batch)
        for space, array in arrays.items():
            array.flush()
        arrays.clear()
        del array
        for space in ("h512", "z128"):
            target = output / f"{view}_{space}.npy"
            target.with_suffix(".npy.tmp").replace(target)
            checksums[target.name] = file_hash(target)
    write_json(marker, {"fingerprint": fingerprint, "sha256": checksums})


def load_diagnostic_model(kind, config, source, expected_signature, device):
    torch.manual_seed(config["seed"])
    model = torch.nn.ModuleDict({"encoder": Encoder(512), "projector": Projector(512, 128)})
    initialization = {"encoder_sha256": parameter_hash(model["encoder"]), "model_sha256": parameter_hash(model)}
    if kind == "random":
        info = {"kind": kind, "epoch": 0, "initialization": initialization}
    else:
        checkpoint = load_checkpoint(source / "models/local" / f"{kind}.pt")
        if checkpoint.get("signature") != expected_signature:
            raise ValueError(f"{kind}.pt does not match this configuration/cache/training code")
        model.load_state_dict(checkpoint["model"], strict=True)
        info = {"kind": kind, "epoch": checkpoint.get("epoch"), "best_valid_loss": checkpoint.get("best_loss"),
                "recorded_initial_encoder_sha256": checkpoint.get("initial_encoder_sha256"),
                "recorded_initial_model_sha256": checkpoint.get("initial_parameter_sha256")}
        del checkpoint
    model.requires_grad_(False)
    return model.to(device).eval(), info


def batched_map_scores(cosines, candidate_indices, query_area, settings):
    """Same greedy rule as map_similarity, vectorized across candidate wafers.

    candidate_indices is padded with -1. Stable sorting of the flattened Q x C
    matrix preserves the official query-index/candidate-index tie break.
    """
    if settings["lambda_pos"] or settings["lambda_scale"]:
        raise ValueError("This diagnostic supports the shape-only primary matcher")
    count, width = candidate_indices.shape
    nq = len(query_area)
    if not nq or not width:
        return np.zeros(count)
    scores = np.clip(cosines[:, np.maximum(candidate_indices, 0)].transpose(1, 0, 2), 0, 1)
    scores *= (candidate_indices >= 0)[:, None, :]
    scores[scores < settings["shape_threshold"]] = 0
    flat = scores.reshape(count, -1)
    order = np.argsort(-flat, axis=1, kind="stable")
    area = np.asarray(query_area, dtype=float)
    weights = np.ones_like(area) if settings["weight"] == "uniform" else (
        np.sqrt(area) if settings["weight"] == "sqrt_area" else area)
    used_q, used_c = np.zeros((count, nq), bool), np.zeros((count, width), bool)
    total = np.zeros(count)
    rows = np.arange(count)
    for column in range(order.shape[1]):
        index = order[:, column]
        qi, ci = index // width, index % width
        value = flat[rows, index]
        active = (value > 0) & (value >= settings["score_threshold"])
        if not active.any():
            break
        take = active & ~used_q[rows, qi] & ~used_c[rows, ci]
        total[take] += weights[qi[take]] * value[take]
        used_q[rows[take], qi[take]] = True
        used_c[rows[take], ci[take]] = True
    return total / weights.sum()


def summarize_groups(rows, metrics, repetitions, seed):
    results = []
    for group in ("all", "single", "mixed"):
        selected = [row for row in rows if group == "all" or row["group"] == group]
        rng = np.random.default_rng(seed)
        values = {key: [row[key] for row in selected if row[key] is not None] for key in metrics}
        results.append({"group": group, "num_queries": len(selected),
                        "queries_with_full_coverage_candidates": sum(row["mAP_full"] is not None for row in selected),
                        "metrics": {key: float(np.mean(items)) if items else None for key, items in values.items()},
                        "metric_query_counts": {key: len(items) for key, items in values.items()},
                        "query_bootstrap_95ci": {key: bootstrap_interval(items, repetitions, rng)
                                                  for key, items in values.items()}})
    return results


def retrieve_variant(features, metadata, protocol, output, config, settings):
    offsets = metadata["offsets"]
    indices = {int(i): np.arange(start, end) for i, start, end in
               zip(metadata["map_ids"], offsets[:-1], offsets[1:])}
    width = int(np.diff(offsets).max())
    padded = np.full((len(metadata["map_ids"]), width), -1, dtype=np.int64)
    positions = {}
    for row, sample_id in enumerate(metadata["map_ids"]):
        positions[int(sample_id)] = row
        tokens = indices[int(sample_id)]
        padded[row, :len(tokens)] = tokens
    output.mkdir(parents=True, exist_ok=True)
    rows, distributions = [], []
    map_scores = {name: [] for name in ("all", "full_coverage", "partial_overlap", "disjoint")}
    start_time = time.monotonic()
    with (output / "rankings.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["query_id", "rank", "candidate_id", "similarity_score"])
        writer.writeheader()
        for number, q in enumerate(protocol.queries, 1):
            candidates = np.array(protocol.candidates[q], dtype=np.int64)
            tokens = indices[q]
            cosines = features[tokens] @ features.T
            candidate_indices = padded[[positions[int(c)] for c in candidates]]
            scores = batched_map_scores(cosines, candidate_indices, metadata["area"][tokens], config["matching"])
            if not np.isfinite(scores).all():
                raise FloatingPointError("Non-finite map scores")
            order = np.lexsort((candidates, -scores))
            for rank, position in enumerate(order, 1):
                writer.writerow({"query_id": q, "rank": rank, "candidate_id": int(candidates[position]),
                                 "similarity_score": float(scores[position])})
            label = protocol.labels[q]
            relevance = np.array([np.minimum(label, protocol.labels[int(c)]).sum() / label.sum() for c in candidates])
            metrics = query_metrics(relevance[order], relevance, config["evaluation"]["ks"])
            rows.append({"query_id": q, "group": "single" if label.sum() == 1 else "mixed",
                         "labels": label_text(label), "num_labels": int(label.sum()), "num_tokens": len(tokens),
                         "full_coverage_candidates": int(np.isclose(relevance, 1).sum()), **metrics})
            for name, mask in (("all", np.ones(len(scores), bool)), ("full_coverage", np.isclose(relevance, 1)),
                               ("partial_overlap", (relevance > 0) & (relevance < 1)), ("disjoint", relevance == 0)):
                values = scores[mask]
                map_scores[name].append(values)
                distributions.append({"query_id": q, "relation": name, "count": len(values),
                                      "mean": float(values.mean()) if len(values) else None,
                                      "std": float(values.std()) if len(values) else None,
                                      "min": float(values.min()) if len(values) else None,
                                      "max": float(values.max()) if len(values) else None,
                                      "zero_fraction": float((values == 0).mean()) if len(values) else None})
            if number % 100 == 0 or number == len(protocol.queries):
                print(f"  retrieval {number}/{len(protocol.queries)} queries", flush=True)
    write_csv(output / "per_query_metrics.csv", rows, list(rows[0]))
    write_csv(output / "map_score_by_query.csv", distributions, list(distributions[0]))
    write_json(output / "map_score_distribution.json",
               {name: distribution(np.concatenate(values)) for name, values in map_scores.items()})
    groups = summarize_groups(rows, list(metrics), settings["bootstrap"], settings["seed"])
    result = {"groups": groups, "seconds": time.monotonic() - start_time,
              "pool": "frozen valid-only; self wafer excluded", "relevance": "query label containment",
              "ties": "descending score then ascending candidate_id", "matching": config["matching"]}
    write_json(output / "metrics.json", result)
    return rows, groups


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0}
    return {"count": len(values), "mean": float(values.mean()), "std": float(values.std()),
            "quantiles": {str(q): float(np.quantile(values, q)) for q in (0, .01, .05, .5, .95, .99, 1)},
            "fraction_ge_0.9": float((values >= .9).mean()),
            "fraction_ge_0.95": float((values >= .95).mean()),
            "fraction_ge_0.99": float((values >= .99).mean()),
            "fraction_le_0": float((values <= 0).mean())}


def pair_auc(positive, other):
    """Tie-correct probability that a rotated view pair outranks a random pair."""
    from scipy.stats import rankdata
    if not len(positive) or not len(other):
        return None
    ranks = rankdata(np.concatenate([positive, other]), method="average")
    n = len(positive)
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * len(other)))


def analyze_scores(raw, features, rotated, positive, pairs, owners, labels, output):
    pids = np.array([int(row["token_id"]) for row in positive])
    a, b = (np.array([int(row[key]) for row in pairs]) for key in ("query_token", "candidate_token"))
    positive_cos = np.sum(features[pids] * normalized(rotated), axis=1)
    random_cos = np.sum(features[a] * features[b], axis=1)
    overlap = np.array([np.minimum(labels[int(owners[i])], labels[int(owners[j])]).any() for i, j in zip(a, b)])
    values = {"same_token_rotated": positive_cos, "random_cross_wafer": random_cos,
              "random_wafer_labels_overlap": random_cos[overlap], "random_wafer_labels_disjoint": random_cos[~overlap]}
    stats = {name: {"raw_cosine": distribution(value), "clipped_match_score": distribution(np.clip(value, 0, 1))}
             for name, value in values.items()}
    norms = np.linalg.norm(np.asarray(raw, dtype=np.float64), axis=1)
    mean_std = np.std(features, axis=0)
    geometry = {"num_tokens": len(raw), "dimension": raw.shape[1], "raw_norm": distribution(norms),
                "near_zero_norm_fraction": float((norms < 1e-12).mean()),
                "normalized_mean_vector_norm": float(np.linalg.norm(features.mean(0))),
                "mean_per_dimension_std": float(mean_std.mean()), "max_per_dimension_std": float(mean_std.max()),
                "note": "Mean-vector norm near 1 indicates a common feature direction; this alone does not prove collapse."}
    result = {"distributions": stats, "geometry": geometry,
              "rotation_vs_random_auc_raw": pair_auc(positive_cos, random_cos),
              "rotation_vs_random_auc_clipped": pair_auc(np.clip(positive_cos, 0, 1), np.clip(random_cos, 0, 1)),
              "mean_rotation_minus_random_cosine": float(positive_cos.mean() - random_cos.mean()),
              "note": "Random cross-wafer pairs can share shapes; wafer labels are not region ground truth. AUC is view discrimination, not semantic retrieval accuracy."}
    write_json(output / "score_distribution.json", result)
    pair_rows = [{"pair_type": "same_token_rotated", "query_token": int(i), "candidate_token": int(i),
                  "rotation_k": int(row["rotation_k"]), "wafer_labels_overlap": True,
                  "raw_cosine": float(value), "clipped_score": float(np.clip(value, 0, 1))}
                 for i, row, value in zip(pids, positive, positive_cos)]
    pair_rows += [{"pair_type": "random_cross_wafer", "query_token": int(i), "candidate_token": int(j),
                   "rotation_k": 0, "wafer_labels_overlap": bool(shared),
                   "raw_cosine": float(value), "clipped_score": float(np.clip(value, 0, 1))}
                  for i, j, shared, value in zip(a, b, overlap, random_cos)]
    write_csv(output / "score_pairs.csv", pair_rows, list(pair_rows[0]))
    return result, values


def region_neighbors(features, anchors, owners, local_ids, metadata, labels, output, k):
    records, gallery = [], {}
    for anchor in anchors:
        eligible = np.flatnonzero(owners != owners[anchor])
        cosine = features[eligible] @ features[anchor]
        # Raw cosine keeps useful ordering even when the official clip gives zero.
        order = np.lexsort((eligible, -cosine))[:k]
        gallery[anchor] = [(int(eligible[i]), float(cosine[i])) for i in order]
        for rank, (token, score) in enumerate(gallery[anchor], 1):
            records.append({"anchor_token": anchor, "anchor_map": int(owners[anchor]),
                            "anchor_local_token": int(local_ids[anchor]),
                            "anchor_labels": label_text(labels[int(owners[anchor])]),
                            "anchor_group": "single" if labels[int(owners[anchor])].sum() == 1 else "mixed",
                            "rank": rank, "neighbor_token": token, "neighbor_map": int(owners[token]),
                            "neighbor_local_token": int(local_ids[token]),
                            "neighbor_labels": label_text(labels[int(owners[token])]),
                            "wafer_labels_overlap": bool(np.minimum(labels[int(owners[anchor])], labels[int(owners[token])]).any()),
                            "anchor_area": float(metadata["area"][anchor]), "neighbor_area": float(metadata["area"][token]),
                            "raw_cosine": score, "clipped_score": float(np.clip(score, 0, 1))})
    write_csv(output / "region_neighbors.csv", records, list(records[0]))
    return gallery


def paired_comparisons(all_rows, output, settings):
    comparisons = [(f"{kind}_z128", f"{kind}_h512") for kind in ("best", "last", "random")]
    comparisons += [(f"best_{space}", f"last_{space}") for space in ("h512", "z128")]
    comparisons += [(f"{kind}_{space}", f"random_{space}") for kind in ("best", "last") for space in ("h512", "z128")]
    records = []
    for left, right in comparisons:
        right_rows = {row["query_id"]: row for row in all_rows[right]}
        for group in ("all", "single", "mixed"):
            for metric in ("ContainmentNDCG@10", "mAP_full"):
                diffs = [row[metric] - right_rows[row["query_id"]][metric] for row in all_rows[left]
                         if (group == "all" or row["group"] == group) and row.get(metric) is not None
                         and right_rows[row["query_id"]].get(metric) is not None]
                records.append({"left": left, "right": right, "group": group, "metric": metric,
                                "num_queries": len(diffs), "mean_difference": float(np.mean(diffs)) if diffs else None,
                                "paired_query_bootstrap_95ci": bootstrap_interval(diffs, settings["bootstrap"],
                                                                                  np.random.default_rng(settings["seed"]))})
    write_json(output / "paired_comparisons.json", records)
    return records


def draw_figures(output, source, all_values, galleries, anchors, owners, labels, neighbors, comparison):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, variant in zip(axes.flat, VARIANTS):
        pooled = np.concatenate([all_values[variant]["same_token_rotated"], all_values[variant]["random_cross_wafer"]])
        span = max(float(np.ptp(pooled)), 1e-8)
        bounds = (max(-1., float(pooled.min()) - span * .05), min(1., float(pooled.max()) + span * .05))
        bins = np.linspace(*bounds, 61)
        for key, color in (("same_token_rotated", "C0"), ("random_wafer_labels_disjoint", "C1"),
                           ("random_wafer_labels_overlap", "C2")):
            values = all_values[variant][key]
            if len(values):
                ax.hist(values, bins=bins, weights=np.ones(len(values)) / len(values),
                        histtype="step", label=f"{key} (n={len(values)})", color=color)
        ax.set(title=variant, xlabel="raw cosine (panel-specific range)", ylabel="fraction per bin", xlim=bounds)
        ax.ticklabel_format(axis="x", useOffset=True)
        ax.legend(fontsize=6)
    fig.savefig(figures / "region_cosine_distributions.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    for ax, metric in zip(axes, ("ContainmentNDCG@10", "mAP_full")):
        x = np.arange(len(VARIANTS))
        for offset, group in enumerate(("all", "single", "mixed")):
            vals = [next(row for row in comparison if row["variant"] == variant and row["group"] == group)[metric]
                    for variant in VARIANTS]
            ax.bar(x + (offset - 1) * .25, [np.nan if v is None else v for v in vals], width=.25, label=group)
        ax.set_xticks(x, VARIANTS, rotation=25, ha="right")
        ax.set(title=metric, ylim=(0, 1))
        ax.legend()
    fig.savefig(figures / "retrieval_by_group.png", dpi=150)
    plt.close(fig)
    patches = np.load(source / "cache/valid_patches.npy", mmap_mode="r")
    for anchor in anchors:
        fig, axes = plt.subplots(len(VARIANTS), neighbors + 1, figsize=(2 * (neighbors + 1), 2 * len(VARIANTS)),
                                 squeeze=False, constrained_layout=True)
        for row, variant in enumerate(VARIANTS):
            items = [(anchor, None)] + galleries[variant][anchor]
            for column, ax in enumerate(axes[row]):
                ax.axis("off")
                if column >= len(items):
                    continue
                token, score = items[column]
                ax.imshow(patches[token, 0], cmap="gray", vmin=0, vmax=1)
                title = f"map {owners[token]} / token {token}\nwafer labels {label_text(labels[int(owners[token])])}"
                ax.set_title(title + (f"\ncos={score:.8f}" if score is not None else f"\n{variant} anchor"), fontsize=7)
        fig.savefig(figures / f"neighbors_token_{anchor}.png", dpi=120)
        plt.close(fig)


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


@contextmanager
def diagnostic_log(output):
    with (output / "run.log").open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            yield


def write_report(output, comparison, score_stats, paired, model_info, audit, protocol):
    def fmt(value):
        return "NA" if value is None else f"{value:.4f}"
    lines = ["# Valid 检索诊断", "", "仅推理现有 checkpoint，无训练。所有变体复用相同 valid 查询/候选池，排除查询自身。",
             f"查询数 {protocol['queries']}，每查询候选数 {protocol['candidates_per_query']}。这些结果不可与正式 test 指标直接比较。", "",
             "| 表示 | 分组 | 查询数 | 有全覆盖候选的查询数 | NDCG@10 | mAP_full |", "|---|---|---:|---:|---:|---:|"]
    for row in comparison:
        lines.append(f"| {row['variant']} | {row['group']} | {row['num_queries']} | "
                     f"{row['queries_with_full_coverage_candidates']} | {fmt(row.get('ContainmentNDCG@10'))} | {fmt(row['mAP_full'])} |")
    lines += ["", "## 配对比较", "", "差值为 left − right；95% 区间来自相同查询的配对 bootstrap。区间包含 0 时，当前池不能明确区分两者。", ""]
    for row in paired:
        if row["group"] == "all":
            ci = row["paired_query_bootstrap_95ci"]
            interval = "NA" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            lines.append(f"- {row['left']} − {row['right']}，{row['metric']}：{fmt(row['mean_difference'])}，95% CI {interval}。")
    lines += ["", "## 区域分数与表示分布", "", "| 表示 | 旋转同区域均值 | 随机跨晶圆均值 | 差值 | 视图区分 AUC | 均值向量范数 |", "|---|---:|---:|---:|---:|---:|"]
    for variant, stats in score_stats.items():
        values = stats["distributions"]
        lines.append(f"| {variant} | {values['same_token_rotated']['raw_cosine']['mean']:.8f} | "
                     f"{values['random_cross_wafer']['raw_cosine']['mean']:.8f} | "
                     f"{stats['mean_rotation_minus_random_cosine']:.8f} | {fmt(stats['rotation_vs_random_auc_raw'])} | "
                     f"{stats['geometry']['normalized_mean_vector_norm']:.8f} |")
    lines += ["", "## 如何判断", "",
              "- 先比较 trained 与相同维度 random 的检索结果。随机初始化 ResNet 因 ReLU/BN 也可能有很高余弦；高余弦本身不代表学到了形状。",
              "- 再比较每个 checkpoint 的 z128 与 h512。如果投影空间明显更好，说明当前检索 readout 值得调整；单次 valid 诊断不构成最终结论。",
              "- 比较 best 与 last：best 按 NCE valid loss 选择，未必是检索最优轮次。这里只比较两个已保存轮次，不能寻找其他 epoch 的最优检索点。",
              "- 对照 single/mixed，检查查询数、全覆盖候选数和 map_score_by_query.csv。混合查询全覆盖候选可能更少，组间差距同时受标签难度和候选构成影响。",
              "- 若旋转同区域得分高，但随机跨晶圆也接近且检索改善小，提示视图区分、特征共同方向或实例目标与语义检索的关系需检查，不能据此直接断定表示塌缩或假负样本。",
              "- figures/neighbors_token_*.png 的六行使用同一 anchor，近邻排除同晶圆。按原始 cosine 排序；正式匹配将负 cosine 截为 0。标签来自晶圆，不是区域真值，近邻图用于人工核查。",
              "- 随机区域对可能包含相似形状；AUC 仅衡量旋转同区域与随机对的区分能力，不是区域语义准确率。只有 90/180/270 度旋转，本诊断不检验任意角度旋转。",
              "- bootstrap 只反映当前固定池的查询抽样波动，不涵盖训练 seed 和候选池变化。", "",
              f"原实验文件校验：source_unchanged={audit['source_unchanged']}；详情见 source_audit.json。",
              "随机初始化与训练记录初始权重的对应关系见 model_info.json；不同 PyTorch 版本可能产生不同初始参数。",
              "所有完整指标与分组有效查询数见 comparison.csv、variants/*/metrics.json；环境、输入哈希与设置见 run_manifest.json。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_diagnostics(config_path, config, output, device, settings, resume=False):
    """Execute an isolated inference run; source experiment is opened read-only."""
    source, output = Path(config["output_dir"]).resolve(), Path(output).resolve()
    guard_output(source, output)
    if config["embedding_dim"] != 512 or config["local"]["projector_dim"] != 128:
        raise ValueError("Expected the current 512D backbone / 128D linear-projector experiment")
    if config["matching"]["lambda_pos"] or config["matching"]["lambda_scale"]:
        raise ValueError("Only the current shape-only experiment is supported")
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("Diagnostic output is not empty; use --resume with identical settings or choose a new directory")
    if resume and output.exists() and any(output.iterdir()) and not (output / "run_manifest.json").is_file():
        raise ValueError("Nonempty directory has no diagnostic manifest; choose a new directory")
    before = source_snapshot(source)
    hashes = input_hashes(config_path, config)
    signature = training_signature(config, "local")
    identity = {"input_sha256": hashes, "training_signature": signature, "settings": settings,
                "diagnostic_code_sha256": file_hash(Path(__file__)), "device": str(device),
                "torch_version": str(torch.__version__), "numpy_version": np.__version__,
                "matching": config["matching"], "ks": config["evaluation"]["ks"]}
    fingerprint = digest(identity)
    previous = read_json(output / "run_manifest.json") if (output / "run_manifest.json").is_file() else None
    if previous and previous["fingerprint"] != fingerprint:
        raise ValueError("Inputs/code/settings/environment changed; use a new diagnostic directory")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run_manifest.json", {**identity, "fingerprint": fingerprint, "source": str(source),
                                             "output": str(output), "variants": VARIANTS, "status": "running"})
    with diagnostic_log(output):
        print(f"Read-only valid diagnostics on {device}; source={source}; output={output}", flush=True)
        started = time.monotonic()
        try:
            metadata, labels, owners, local_ids = valid_source(config)
            protocol, manifest, positive, pairs, anchors = freeze_protocol(output, metadata, labels, owners, settings)
            print(f"Frozen valid pool: {manifest['queries']} queries x {manifest['candidates_per_query']} candidates; "
                  f"{len(owners)} regions", flush=True)
            comparison, all_rows, score_stats, all_values, galleries, model_info = [], {}, {}, {}, {}, {}
            for kind in ("best", "last", "random"):
                model, info = load_diagnostic_model(kind, config, source, signature, device)
                model_info[kind] = info
                print(f"Extracting {kind}, epoch={info['epoch']} (eval / inference_mode)", flush=True)
                feature_dir = output / "features" / kind
                extract_features(model, source, feature_dir, config, device, positive, settings["batch_size"],
                                 settings["num_workers"], digest({"run": fingerprint, "kind": kind}))
                del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                elif device.type == "mps":
                    torch.mps.empty_cache()
                for space in ("h512", "z128"):
                    variant = f"{kind}_{space}"
                    print(f"Analyzing {variant}", flush=True)
                    target = output / "variants" / variant
                    raw = np.load(feature_dir / f"original_{space}.npy", mmap_mode="r")
                    features = normalized(raw)
                    rotated = np.load(feature_dir / f"rotated_{space}.npy", mmap_mode="r")
                    all_rows[variant], groups = retrieve_variant(features, metadata, protocol, target, config, settings)
                    comparison.extend({"variant": variant, "group": group["group"], "num_queries": group["num_queries"],
                                       "queries_with_full_coverage_candidates": group["queries_with_full_coverage_candidates"],
                                       **group["metrics"]} for group in groups)
                    score_stats[variant], all_values[variant] = analyze_scores(raw, features, rotated, positive, pairs,
                                                                              owners, labels, target)
                    galleries[variant] = region_neighbors(features, anchors, owners, local_ids, metadata, labels,
                                                          target, settings["neighbors"])
                    del raw, rotated, features
            for kind in ("best", "last"):
                info = model_info[kind]
                info["random_matches_recorded_initial_encoder"] = (
                    model_info["random"]["initialization"]["encoder_sha256"] == info["recorded_initial_encoder_sha256"]
                    if info["recorded_initial_encoder_sha256"] else None)
                info["random_matches_recorded_initial_model"] = (
                    model_info["random"]["initialization"]["model_sha256"] == info["recorded_initial_model_sha256"]
                    if info["recorded_initial_model_sha256"] else None)
            write_json(output / "model_info.json", model_info)
            write_json(output / "comparison.json", comparison)
            write_csv(output / "comparison.csv", comparison, list(comparison[0]))
            paired = paired_comparisons(all_rows, output, settings)
            draw_figures(output, source, all_values, galleries, anchors, owners, labels, settings["neighbors"], comparison)
        finally:
            after = source_snapshot(source)
            changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
            changed_hashes = [path for path, expected in hashes.items() if not Path(path).is_file() or file_hash(path) != expected]
            audit = {"source_unchanged": not changed and not changed_hashes, "changed_source_files": changed,
                     "changed_hashed_inputs": changed_hashes, "source_files_checked": len(before),
                     "check": "entire experiment file paths, size and mtime; SHA256 of required inputs"}
            write_json(output / "source_audit.json", audit)
            status = read_json(output / "run_manifest.json")
            status.update(status="incomplete", seconds=time.monotonic() - started, source_unchanged=audit["source_unchanged"])
            write_json(output / "run_manifest.json", status)
        if not audit["source_unchanged"]:
            raise RuntimeError("Source experiment changed during diagnostics; inspect source_audit.json (including concurrent jobs)")
        write_report(output, comparison, score_stats, paired, model_info, audit, manifest)
        status.update(status="complete", seconds=time.monotonic() - started)
        write_json(output / "run_manifest.json", status)
        print(f"Complete in {status['seconds']:.1f}s; source_unchanged=True; report={output / 'report.md'}", flush=True)
    return output
