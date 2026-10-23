#!/usr/bin/env python3
"""Read-only E/F instance retrieval and feature distribution evaluation.

Kept outside wbm_retrieval so adding this entry point does not alter historical
ablation source fingerprints. No labels, optimization or cache preparation.
"""

import argparse
import copy
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import logsumexp
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wbm_retrieval.ablation_ef import build_model
from wbm_retrieval.common import digest, file_hash, get_device, read_json, seed_everything, write_csv, write_json
from wbm_retrieval.training import load_checkpoint, parameter_hash


def tie_metrics(better, tied, candidates):
    """Expected metrics under uniformly random ordering of the positive tie block."""
    better, tied, candidates = (np.asarray(v, dtype=np.int64) for v in (better, tied, candidates))
    if np.any(tied < 1) or np.any(better < 0) or np.any(better + tied > candidates):
        raise ValueError("Invalid rank counts")
    harmonic = np.r_[0., np.cumsum(1. / np.arange(1, int(candidates.max()) + 1))]
    result = {"mean_rank": better + (tied + 1) / 2.,
              "mrr": (harmonic[better + tied] - harmonic[better]) / tied,
              "chance_top1": 1. / candidates,
              "chance_mrr": harmonic[candidates] / candidates}
    for k, name in ((1, "top1"), (5, "recall_at_5"), (10, "recall_at_10")):
        result[name] = np.clip(k - better, 0, tied) / tied
    return result


def cross_wafer_pairs(map_ids, count, seed):
    if len(np.unique(map_ids)) < 2:
        raise ValueError("Need at least two test wafers")
    rng = np.random.default_rng(seed)
    left = rng.integers(len(map_ids), size=count)
    right = rng.integers(len(map_ids), size=count)
    bad = map_ids[left] == map_ids[right]
    while bad.any():
        right[bad] = rng.integers(len(map_ids), size=int(bad.sum()))
        bad = map_ids[left] == map_ids[right]
    return left, right


def pair_cosines(left_features, right_features, pairs):
    left, right = pairs
    values = []
    for start in range(0, len(left), 4096):
        a, b = left[start:start+4096], right[start:start+4096]
        values.append(np.einsum("ij,ij->i", left_features[a], right_features[b]))
    return np.clip(np.concatenate(values), -1., 1.)


def spectrum_metrics(features, pairs):
    x = np.asarray(features, dtype=np.float64)
    centered = x - x.mean(0)
    eigen = np.maximum(np.linalg.eigvalsh(centered.T @ centered / len(x)), 0.)
    variance = float(eigen.sum())
    if variance > 1e-15:
        probabilities = eigen[eigen > 0] / variance
        effective = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
        top_share = float(eigen[-1] / variance)
    else:
        effective, top_share = 0., None
    cosine = pair_cosines(x, x, pairs)
    return {"effective_rank_covariance": effective, "effective_rank_upper_bound": min(len(x)-1, x.shape[1]),
            "total_centered_variance": variance, "top_eigenvalue_share": top_share,
            "cross_wafer_uniformity_t2": float(logsumexp(-4. * (1. - cosine)) - np.log(len(cosine))),
            "original_cross_wafer_cosine_mean": float(cosine.mean())}


@torch.inference_mode()
def encode(model, patches, indices, turns, batch_size, device):
    values = {"h512": [], "z128": []}
    for start in range(0, len(indices), batch_size):
        image = torch.from_numpy(np.array(patches[indices[start:start+batch_size]], dtype=np.float32)).to(device)
        image = torch.rot90(image, turns, (-2, -1))
        h = model["encoder"](image)
        for name, feature in (("h512", h), ("z128", model["projector"](h))):
            if not torch.isfinite(feature).all() or (feature.norm(dim=1) < 1e-12).any():
                raise ValueError("Nonfinite or zero feature vectors")
            values[name].append(F.normalize(feature, dim=1).cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in values.items()}


@torch.inference_mode()
def score_features(original, rotated, map_ids, token_ids, angle, device, chunk, temperature, tolerance, margin):
    """All cross-wafer augmented regions plus own positive; same-wafer others excluded."""
    gallery = torch.from_numpy(rotated).to(device)
    wafer = torch.from_numpy(map_ids).to(device)
    rows = []
    for start in range(0, len(original), chunk):
        stop = min(start + chunk, len(original))
        q = torch.arange(start, stop, device=device)
        scores = (torch.from_numpy(original[start:stop]).to(device) @ gallery.T).clamp(-1., 1.)
        positive = scores[torch.arange(stop-start, device=device), q].clone()
        negative = wafer[None, :] != wafer[q, None]
        num_negative = negative.sum(1)
        if (num_negative == 0).any():
            raise ValueError("No cross-wafer negatives")
        allowed = negative.clone()
        allowed[torch.arange(stop-start, device=device), q] = True
        difference = scores - positive[:, None]
        better = ((difference > tolerance) & allowed).sum(1).cpu().numpy()
        tied = ((difference.abs() <= tolerance) & allowed).sum(1).cpu().numpy()
        counts = num_negative.cpu().numpy() + 1
        metrics = tie_metrics(better, tied, counts)
        hard = scores.masked_fill(~negative, -torch.inf).max(1).values
        negative_mean = scores.masked_fill(~negative, 0.).sum(1) / num_negative
        crowded = ((scores >= positive[:, None] - margin) & negative).sum(1)
        values = {"positive_cosine": positive, "hard_negative_cosine": hard,
                  "negative_cosine_mean": negative_mean, "hard_negative_margin": positive-hard,
                  "positive_negative_mean_gap": positive-negative_mean,
                  "crowding_fraction": crowded / num_negative, "crowding_count": crowded,
                  "alignment_squared": 2. - 2. * positive,
                  "gallery_nce": torch.logsumexp((scores / temperature).masked_fill(~allowed, -torch.inf), 1)
                                 - positive / temperature}
        values = {key: value.cpu().numpy() for key, value in values.items()}
        for j, index in enumerate(range(start, stop)):
            rows.append({"token_id": int(token_ids[index]), "map_id": int(map_ids[index]), "angle": angle,
                         "candidate_count": int(counts[j]), "tie_size": int(tied[j]),
                         **{key: float(value[j]) for key, value in {**metrics, **values}.items()}})
    return rows


def summarize(rows, repetitions, seed):
    excluded = {"token_id", "map_id", "angle"}
    keys = [key for key in rows[0] if key not in excluded]
    matrix = np.array([[row[key] for key in keys] for row in rows])
    maps = np.array([row["map_id"] for row in rows])
    unique, inverse = np.unique(maps, return_inverse=True)
    sums = np.zeros((len(unique), len(keys)))
    np.add.at(sums, inverse, matrix)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repetitions):
        sampled = rng.integers(len(unique), size=len(unique))
        estimates.append(sums[sampled].sum(0) / counts[sampled].sum())
    bounds = np.quantile(estimates, [.025, .975], axis=0) if estimates else None
    return {"region_view_weighted_means": dict(zip(keys, matrix.mean(0).tolist())),
            "wafer_bootstrap_95ci": {key: bounds[:, i].tolist() for i, key in enumerate(keys)} if estimates else {},
            "median_expected_rank": float(np.median(matrix[:, keys.index("mean_rank")])),
            "hard_margin_quantiles": dict(zip(("p05", "p50", "p95"), np.quantile(matrix[:, keys.index("hard_negative_margin")], [.05, .5, .95]).tolist())),
            "num_region_views": len(rows), "num_wafers": len(unique)}


def distribution_plot(folder, original, rotated, rows, pairs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    negatives = pair_cosines(original, rotated, pairs)
    positives = np.array([r["positive_cosine"] for r in rows])
    hard = np.array([r["hard_negative_cosine"] for r in rows])
    gaps = positives - hard
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    bins = np.linspace(-1, 1, 101)
    combined = np.r_[positives, negatives, hard]
    zoom_low, zoom_high = np.quantile(combined, [.005, .995])
    padding = max(float(zoom_high-zoom_low) * .1, 1e-6)
    zoom_bins = np.linspace(max(-1., zoom_low-padding), min(1., zoom_high+padding), 81)
    for values, label in ((positives, "Positive"), (negatives, "Random cross-wafer negative"), (hard, "Hardest negative")):
        axes[0].hist(values, bins=bins, density=True, histtype="step", linewidth=1.6, label=label)
        weights = np.ones(len(values)) / (len(values) * np.diff(zoom_bins)[0])
        axes[1].hist(values, bins=zoom_bins, weights=weights, histtype="step", linewidth=1.6, label=label)
    axes[0].set(xlabel="Cosine similarity", ylabel="Density", title="Positive / negative distributions")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Cosine similarity (adaptive zoom)", ylabel="Density", title="Zoom: pooled 0.5%-99.5% range")
    axes[1].ticklabel_format(axis="x", useOffset=False)
    axes[2].hist(gaps, bins=60, color="#3176b4")
    axes[2].axvline(0, color="#c14444", linestyle="--")
    axes[2].set(xlabel="Positive cosine - hardest negative cosine", ylabel="Region count", title="Hard-negative margin")
    fig.savefig(folder / "distribution.png", dpi=160)
    plt.close(fig)
    np.savez_compressed(folder / "distribution_samples.npz", positive=positives, random_negative=negatives, hardest_negative=hard)
    ranks = rankdata(np.r_[positives, negatives])
    auc = (ranks[:len(positives)].sum() - len(positives)*(len(positives)+1)/2) / (len(positives)*len(negatives))
    return {"sampled_pair_roc_auc": float(auc), "random_negative_sample_count": len(negatives),
            "random_negative_cosine_quantiles": dict(zip(("p05", "p50", "p95", "p99"), np.quantile(negatives, [.05, .5, .95, .99]).tolist()))}


def plot_loss_history(source, output, seed):
    """Plot actual saved epochs, without smoothing or inventing a downward trend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    series, hashes = {}, {}
    for name in "EF":
        run = source / f"seed_{seed}" / name
        for filename in ("history.json", "training_summary.json"):
            hashes[str(run / filename)] = file_hash(run / filename)
        history = read_json(run / "history.json")
        summary = read_json(run / "training_summary.json")
        epochs = [row["epoch"] for row in history]
        if not history or epochs != list(range(1, summary["epochs_completed"] + 1)):
            raise ValueError(f"{name}: history must contain every completed epoch in order")
        values = np.array([[row["train_loss"], row["valid_nce"]] for row in history], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: nonfinite historical loss")
        best = summary["best_nce_epoch"]
        if best not in epochs or not np.isclose(values[best-1, 1], values[:, 1].min()):
            raise ValueError(f"{name}: best_nce_epoch does not match historical minimum")
        series[name] = (epochs, values, best)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, color in (("E", "#2463b0"), ("F", "#cf681c")):
        epochs, values, best = series[name]
        for column, ax in enumerate(axes):
            ax.plot(epochs, values[:, column], color=color, linewidth=1.7,
                    marker="o" if len(epochs) <= 20 else None, markersize=3, label=name)
        axes[1].scatter(best, values[best-1, 1], marker="*", s=150, color=color,
                        edgecolors="black", linewidths=.5, zorder=5,
                        label=f"{name} best_nce: epoch {best}")
    for ax, title, ylabel in zip(axes, ("Training loss", "Validation NCE loss"), ("Train loss", "Valid NCE")):
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.set_xlim(.5, max(len(values[0]) for values in series.values()) + .5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle("E/F self-supervised training history", fontsize=14)
    note = "Saved per-epoch values; no smoothing. Train and validation losses use different positive-pair definitions."
    if any(len(values[0]) == 1 for values in series.values()):
        note += "\nOnly one epoch is available for at least one run; its loss trend cannot be inferred."
    fig.text(.5, .025, note, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .94))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "loss_history.png"
    fig.savefig(path, dpi=180, metadata={"Description": json.dumps({"source_sha256": hashes, "seed": seed})})
    plt.close(fig)
    for filename, expected in hashes.items():
        if file_hash(filename) != expected:
            raise RuntimeError(f"Training history changed while plotting: {filename}")
    print(f"Saved training loss curves: {path}", flush=True)
    return {"source_sha256": hashes, "file": path.name,
            "epochs": {name: len(values[0]) for name, values in series.items()}}


def verify_source(source, seed):
    suite = read_json(source / "suite_manifest.json")
    prepared = read_json(source / "cache/prepared.json")
    if suite["suite"]["suite_id"] != "ef_v1" or suite["signature"] != prepared["signature"]:
        raise ValueError("Not a matching prepared E/F suite")
    paths = [source / "suite_manifest.json", source / "cache/prepared.json"]
    split_ids = {}
    for split in ("train", "valid", "test"):
        rel = f"cache/{split}_regions.npz"
        if file_hash(source / rel) != prepared["files"][rel]:
            raise ValueError(f"Modified cache: {rel}")
        paths.append(source / rel)
        with np.load(source / rel, allow_pickle=False) as meta:
            ids = meta["map_ids"]
            if len(np.unique(ids)) != len(ids):
                raise ValueError("Duplicate wafer IDs")
            split_ids[split] = set(ids.tolist())
            if split == "test":
                offsets = meta["offsets"]
                if len(offsets) != len(ids)+1 or offsets[0] != 0 or (np.diff(offsets) < 0).any():
                    raise ValueError("Invalid test region offsets")
                map_ids = np.repeat(ids, np.diff(offsets))
    if any(split_ids[a] & split_ids[b] for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))):
        raise ValueError("Wafer split leakage")
    manifests = {}
    for name, family in (("E", "B"), ("F", "F")):
        rel = f"cache/test_{family}.npy"
        if file_hash(source / rel) != prepared["files"][rel]:
            raise ValueError(f"Modified cache: {rel}")
        paths.append(source / rel)
        run = source / f"seed_{seed}" / name
        manifest = read_json(run / "run_manifest.json")
        summary = read_json(run / "training_summary.json")
        signature = digest({"suite": suite["signature"], "seed": seed, "variant": name})
        spec = suite["suite"]["variants"][name]
        if (not summary["completed"] or manifest["signature"] != signature or manifest["input"] != spec
                or manifest["config"]["seed"] != seed or manifest["variant"] != name):
            raise ValueError(f"Incomplete or mismatched {name} run")
        for key, value in suite["settings"].items():
            if manifest["config"].get(key) != value:
                raise ValueError(f"Saved settings differ: {name}/{key}")
        patch = np.load(source / rel, mmap_mode="r")
        if patch.shape != (len(map_ids), spec["input_channels"], 96, 96):
            raise ValueError("Patch/region mismatch")
        paths.extend([run / "run_manifest.json", run / "training_summary.json", run / "history.json"])
        manifests[name] = manifest
    return suite, manifests, map_ids, paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True, help="New directory outside the source experiment")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--evaluation-seed", type=int, default=20261023)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--score-batch-size", type=int, default=128)
    parser.add_argument("--max-regions", type=int, default=0, help="0 = all test regions; positive values are diagnostic subsets")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--distribution-pairs", type=int, default=100000)
    parser.add_argument("--tie-tolerance", type=float, default=1e-6)
    parser.add_argument("--crowding-margin", type=float, default=.05)
    parser.add_argument("--checkpoint", choices=("best_nce", "best_retrieval"), default="best_nce")
    parser.add_argument("--skip-random-baseline", action="store_true")
    parser.add_argument("--loss-only", action="store_true", help="Plot saved E/F history only; no inference or GPU required")
    args = parser.parse_args(argv)
    if (min(args.batch_size, args.score_batch_size, args.distribution_pairs) < 1 or args.max_regions < 0
            or args.bootstrap < 0 or not 0 <= args.tie_tolerance < 1 or not 0 < args.crowding_margin <= 2):
        parser.error("Invalid numerical settings")
    source, output = Path(args.source_root).expanduser().resolve(), Path(args.output_root).expanduser().resolve()
    if output == source or source in output.parents or output in source.parents:
        parser.error("Output must be separate from the source experiment")
    if output.exists():
        parser.error("Output already exists; choose a new directory")
    if args.loss_only:
        plot_loss_history(source, output, args.seed)
        return
    device = get_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    suite, manifests, all_maps, paths = verify_source(source, args.seed)
    for name in "EF":
        paths.append(source / f"seed_{args.seed}" / name / f"{args.checkpoint}.pt")
    hashes = {str(path): file_hash(path) for path in paths}
    indices = np.arange(len(all_maps))
    if args.max_regions:
        if args.max_regions > len(indices):
            parser.error("Requested subset exceeds available test regions")
        indices = np.sort(np.random.default_rng(args.evaluation_seed).choice(indices, args.max_regions, replace=False))
    maps = all_maps[indices]
    pairs = cross_wafer_pairs(maps, args.distribution_pairs, args.evaluation_seed)
    output.mkdir(parents=True)
    np.savez(output / "test_pool.npz", token_ids=indices, map_ids=maps, pair_left=pairs[0], pair_right=pairs[1])
    code = {str(path.relative_to(ROOT)): file_hash(path) for path in sorted((ROOT / "wbm_retrieval").rglob("*.py"))}
    code[str(Path(__file__).relative_to(ROOT))] = file_hash(__file__)
    protocol = {"arguments": vars(args), "source_sha256": hashes, "evaluation_code_sha256": code,
                "suite_signature": suite["signature"], "pool_sha256": file_hash(output / "test_pool.npz"),
                "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
                "num_test_regions": len(indices), "num_test_wafers": len(np.unique(maps)), "angles": [90, 180, 270],
                "augmentation": "Rotate cached square patches, all channels together; no 0-degree identity views",
                "candidates": "One matching augmented instance plus all selected cross-wafer augmented regions",
                "ties": "Expected rank/TopK/MRR over uniform tie ordering within absolute tolerance",
                "labels_read": False, "training_performed": False,
                "selection_uses_class_labels": args.checkpoint == "best_retrieval",
                "nce_note": "Fresh test-gallery NCE; not the historical memory-bank training/validation NCE",
                "distribution_note": "Cross-wafer instance negatives may share shapes/classes; no semantic labels assumed",
                "status": "running"}
    write_json(output / "protocol.json", protocol)
    protocol["loss_history"] = plot_loss_history(source, output, args.seed)
    write_json(output / "protocol.json", protocol)
    summary_rows, paired_rows = [], {}
    for name in "EF":
        manifest = manifests[name]
        config, spec = copy.deepcopy(manifest["config"]), manifest["input"]
        checkpoint = load_checkpoint(source / f"seed_{args.seed}" / name / f"{args.checkpoint}.pt")
        if checkpoint["signature"] != manifest["signature"] or checkpoint["input_spec"] != spec or checkpoint["variant"] != name:
            raise ValueError("Checkpoint identity mismatch")
        patches = np.load(source / "cache" / ("test_B.npy" if name == "E" else "test_F.npy"), mmap_mode="r")
        for kind in ([args.checkpoint, "random"] if not args.skip_random_baseline else [args.checkpoint]):
            seed_everything(args.seed)
            model = build_model(config, spec)
            initial_hash = parameter_hash(model)
            if kind != "random":
                model.load_state_dict(checkpoint["model"], strict=True)
            model.requires_grad_(False).to(device).eval()
            before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            target = output / name / kind
            target.mkdir(parents=True)
            write_json(target / "model.json", {"epoch": checkpoint["epoch"] if kind != "random" else 0,
                       "initialization_matches_training": initial_hash == manifest["initial_parameter_sha256"],
                       "note": "If false, random is a fresh same-architecture baseline, not an exact historical initialization"})
            print(f"Encoding {name}/{kind}: {len(indices)} regions", flush=True)
            originals = encode(model, patches, indices, 0, args.batch_size, device)
            all_rows = {feature: [] for feature in originals}
            for turns in (1, 2, 3):
                rotated = encode(model, patches, indices, turns, args.batch_size, device)
                for feature in originals:
                    folder = target / feature / f"rotation_{turns*90}"
                    folder.mkdir(parents=True)
                    rows = score_features(originals[feature], rotated[feature], maps, indices, turns*90, device,
                                          args.score_batch_size, config["local"]["temperature"], args.tie_tolerance, args.crowding_margin)
                    all_rows[feature].extend(rows)
                    result = summarize(rows, 0, args.evaluation_seed)
                    result.update(distribution_plot(folder, originals[feature], rotated[feature], rows, pairs))
                    write_json(folder / "metrics.json", result)
                print(f"Scored {name}/{kind}: rotation={turns*90}", flush=True)
            for feature, rows in all_rows.items():
                folder = target / feature
                result = summarize(rows, args.bootstrap, args.evaluation_seed)
                result["original_feature_distribution"] = spectrum_metrics(originals[feature], pairs)
                write_json(folder / "metrics.json", result)
                write_csv(folder / "per_query.csv", rows, list(rows[0]))
                summary_rows.append({"variant": name, "checkpoint": kind, "feature": feature,
                                     **result["region_view_weighted_means"], **result["original_feature_distribution"]})
                paired_rows[(name, kind, feature)] = rows
            if any(not torch.equal(value, model.state_dict()[key].detach().cpu()) for key, value in before.items()):
                raise RuntimeError("Inference changed model state")
            del model, before
        del checkpoint
    comparisons = {}
    for feature in ("h512", "z128"):
        comparisons_to_make = [("E_minus_F", ("E", args.checkpoint, feature), ("F", args.checkpoint, feature))]
        if not args.skip_random_baseline:
            comparisons_to_make += [(f"{name}_trained_minus_random", (name, args.checkpoint, feature), (name, "random", feature)) for name in "EF"]
        for label, a, b in comparisons_to_make:
            differences = []
            for left, right in zip(paired_rows[a], paired_rows[b]):
                identity = {key: left[key] for key in ("map_id", "token_id", "angle")}
                if any(right[key] != value for key, value in identity.items()):
                    raise ValueError("Unpaired query records")
                differences.append({**identity, **{key: left[key] - right[key] for key in left if key not in identity}})
            comparisons[f"{label}/{feature}"] = summarize(differences, args.bootstrap, args.evaluation_seed)
    write_json(output / "paired_comparisons.json", comparisons)
    write_csv(output / "summary.csv", summary_rows, list(summary_rows[0]))
    for path, expected in hashes.items():
        if file_hash(path) != expected:
            raise RuntimeError(f"Source changed during test: {path}")
    protocol.update(status="complete", source_unchanged=True, model_state_unchanged=True)
    write_json(output / "protocol.json", protocol)
    print(f"Complete: {output / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
