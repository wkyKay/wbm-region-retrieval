"""Isolated, inference-only top1-region transfer test for single-label wafers.

Historical training signatures are recorded, not recomputed using new training
code. Compatibility is checked through strict state loading, the recorded
encoder specification, and exact equivalence to the source experiment's first
region patch. This is not a training/resume entry point.
"""

from __future__ import annotations

import csv
from importlib.metadata import version
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import csv_rows, digest, file_hash, load_config, read_json, write_csv, write_json
from .diagnostics import DiagnosticImages, diagnostic_log, guard_output, normalized, source_snapshot
from .evaluation import bootstrap_interval, query_metrics
from .methods.handcrafted import handcrafted_descriptor
from .models import Encoder, Projector
from .regions import extract_regions, token_patch
from .training import encoder_spec, load_checkpoint, parameter_hash


CLASS_NAMES = ("center", "donut", "edge-loc", "edge-ring", "loc", "random", "scratch", "near-full")
METHODS = ("handcrafted", "proposed")


def source_inputs(source, config, splits):
    paths = [source / "resolved_config.json", source / "cache/prepared.json", source / "models/local/best.pt",
             source / "cache/maps.npy", source / "cache/labels.npy", Path(config["split_manifest"])]
    paths += [source / "cache" / f"{split}_{suffix}" for split in splits for suffix in ("regions.npz", "patches.npy")]
    summary = source / "models/local/training_summary.json"
    if summary.is_file():
        paths.append(summary)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required existing experiment input is missing: {path}")
    hashes = {str(path.resolve()): file_hash(path) for path in paths}
    prepared = read_json(source / "cache/prepared.json")
    if prepared["source_sha256"]["split_manifest"] != hashes[str(Path(config["split_manifest"]).resolve())]:
        raise ValueError("The frozen split manifest no longer matches the saved preprocessing cache")
    return hashes


def load_saved_model(source, config, device):
    checkpoint = load_checkpoint(source / "models/local/best.pt")
    if checkpoint.get("encoder_spec") != encoder_spec(config) or checkpoint.get("embedding_dim") != 512:
        raise ValueError("best.pt must have the recorded single-channel 512D WaPIRL encoder specification")
    model = torch.nn.ModuleDict({"encoder": Encoder(512), "projector": Projector(512, config["local"]["projector_dim"])})
    model.load_state_dict(checkpoint["model"], strict=True)
    info = {"checkpoint": str(source / "models/local/best.pt"), "epoch": checkpoint.get("epoch"),
            "best_valid_loss": checkpoint.get("best_loss"), "historical_training_signature": checkpoint.get("signature"),
            "encoder_spec": checkpoint["encoder_spec"], "encoder_sha256": parameter_hash(model["encoder"]),
            "training_proposal": config["proposal"], "training_opening": config["local"].get("opening"),
            "inference_proposal": {**config["proposal"], "top_k": 1}, "representation": "raw GAP h512, cosine at scoring",
            "current_training_signature_recomputed": False,
            "compatibility_check": "strict full model state_dict and encoder_spec; exact source first-region patch check",
            "training_performed": False}
    if not all(torch.isfinite(value).all() for value in model.state_dict().values() if value.is_floating_point()):
        raise ValueError("Checkpoint contains non-finite model tensors")
    model.requires_grad_(False)
    return model["encoder"].to(device).eval(), info


def prepare_top1(source, output, config, split):
    """Own top1 cache; preserve the exact source region and its existing descriptor."""
    folder = output / split
    marker = folder / "cache_manifest.json"
    if marker.is_file():
        saved = read_json(marker)
        for name, expected in saved["sha256"].items():
            if not (folder / name).is_file() or file_hash(folder / name) != expected:
                raise ValueError(f"Single-label cache/protocol was modified: {name}")
        return saved
    with np.load(source / "cache" / f"{split}_regions.npz", allow_pickle=False) as archive:
        metadata = {key: archive[key] for key in archive.files}
    source_patches = np.load(source / "cache" / f"{split}_patches.npy", mmap_mode="r")
    maps = np.load(source / "cache/maps.npy", mmap_mode="r")
    labels = np.load(source / "cache/labels.npy", mmap_mode="r")
    rows = csv_rows(config["split_manifest"])
    columns = [f"label_{i}" for i in range(8)]
    if labels.ndim != 2 or labels.shape[1] != 8 or not all(key in rows[0] for key in columns):
        raise ValueError("Expected the dataset's eight base labels")
    split_rows = {int(row["sample_id"]): row for row in rows if row["split"] == split}
    cached_ids = metadata["map_ids"].astype(int)
    offsets = metadata["offsets"]
    if (len(set(cached_ids)) != len(cached_ids) or set(cached_ids) != set(split_rows)
            or len(offsets) != len(cached_ids) + 1 or offsets[0] != 0 or np.any(np.diff(offsets) < 0)
            or offsets[-1] != len(source_patches) or len(metadata["area"]) != len(source_patches)
            or metadata["descriptor"].shape != (len(source_patches), 47)
            or source_patches.shape[1:] != (1, config["input_size"], config["input_size"])):
        raise ValueError("Source split IDs, patch shapes, offsets or Zernike dimensions are inconsistent")
    selected = []
    for index, map_id in enumerate(cached_ids):
        expected = np.array([int(split_rows[map_id][key]) for key in columns])
        if not np.isin(expected, [0, 1]).all() or not np.array_equal(expected, labels[map_id]):
            raise ValueError(f"Manifest/cache label mismatch: {map_id}")
        if expected.sum() == 1:
            selected.append((int(map_id), index, int(expected.argmax())))
    selected.sort()
    if len(selected) < 2:
        raise ValueError(f"Need at least two single-label maps in {split}")
    folder.mkdir(parents=True, exist_ok=True)
    patch_array = np.zeros((len(selected), 1, config["input_size"], config["input_size"]), dtype=np.uint8)
    descriptors = np.zeros((len(selected), 47), dtype=np.float32)
    masks = np.zeros((len(selected), *maps.shape[1:]), dtype=np.uint8)
    coverage = []
    for i, (map_id, index, label_id) in enumerate(selected):
        regions = extract_regions(maps[map_id], **{**config["proposal"], "top_k": 1})
        start, end = offsets[index:index+2]
        if bool(regions) != bool(end > start):
            raise ValueError(f"Source proposal availability changed for {map_id}")
        area = 0
        if regions:
            region = regions[0]
            patch = token_patch(region["mask"], config["input_size"], config["proposal"]["padding"])
            if (region["area"] != metadata["area"][start]
                    or np.any(metadata["area"][start:end] > metadata["area"][start])
                    or not np.array_equal(patch, source_patches[start, 0])
                    or not np.allclose(handcrafted_descriptor(region["mask"], 12), metadata["descriptor"][start], rtol=1e-6, atol=1e-6)):
                raise ValueError(f"Top1 is not equivalent to the source's first region for {map_id}")
            patch_array[i] = source_patches[start]
            descriptors[i] = metadata["descriptor"][start]
            masks[i] = region["mask"]
            area = region["area"]
        defect_pixels = int((maps[map_id] == 2).sum())
        coverage.append({"sample_id": map_id, "label_id": label_id, "label_name": CLASS_NAMES[label_id],
                         "has_region": int(bool(regions)), "source_token_index": int(start) if regions else -1,
                         "defect_pixels": defect_pixels, "selected_area": area,
                         "selected_fraction": area / defect_pixels if defect_pixels else 0.})
    np.save(folder / "patches.npy", patch_array)
    np.save(folder / "zernike47.npy", descriptors)
    np.save(folder / "region_masks.npy", masks)
    write_csv(folder / "region_coverage.csv", coverage, list(coverage[0]))
    ids = [row["sample_id"] for row in coverage]
    write_csv(folder / "protocol/queries.csv", ({"sample_id": q} for q in ids), ["sample_id"])
    write_csv(folder / "protocol/candidates.csv", ({"query_id": q, "candidate_id": c} for q in ids for c in ids if c != q),
              ["query_id", "candidate_id"])
    class_counts = {name: sum(row["label_id"] == i for row in coverage) for i, name in enumerate(CLASS_NAMES)}
    manifest = {"split": split, "queries": len(ids), "candidates_per_query": len(ids) - 1,
                "class_counts": class_counts, "candidate_pool": "all other single-label maps in this same frozen split",
                "selection": "labels.sum()==1; no coverage or descriptor filtering", "self_excluded": True,
                "maps_without_region": sum(not row["has_region"] for row in coverage),
                "first_region_equivalence_checked": True, "inference_proposal": {**config["proposal"], "top_k": 1}}
    write_json(folder / "protocol/manifest.json", manifest)
    summary = {name: {"maps": class_counts[name], "median_selected_fraction":
                     float(np.median([row["selected_fraction"] for row in coverage if row["label_id"] == i])) if class_counts[name] else None}
               for i, name in enumerate(CLASS_NAMES)}
    write_json(folder / "coverage_summary.json", summary)
    names = ("patches.npy", "zernike47.npy", "region_masks.npy", "region_coverage.csv", "coverage_summary.json",
             "protocol/queries.csv", "protocol/candidates.csv", "protocol/manifest.json")
    saved = {**manifest, "sha256": {name: file_hash(folder / name) for name in names}}
    write_json(marker, saved)
    return saved


@torch.inference_mode()
def infer_features(model, folder, device, settings, fingerprint):
    marker = folder / "features_manifest.json"
    target = folder / "h512.npy"
    if marker.is_file():
        saved = read_json(marker)
        if saved["fingerprint"] != fingerprint or not target.is_file() or file_hash(target) != saved["sha256"]:
            raise ValueError("Single-label inference feature cache mismatch")
        return
    data = DiagnosticImages(folder / "patches.npy")
    values = np.empty((len(data), 512), dtype=np.float32)
    batches = DataLoader(data, batch_size=settings["batch_size"], num_workers=settings["num_workers"],
                         shuffle=False, pin_memory=device.type == "cuda")
    offset = 0
    for images in batches:
        value = model(images.to(device))
        if not torch.isfinite(value).all():
            raise FloatingPointError("Non-finite backbone inference output")
        values[offset:offset + len(images)] = value.cpu().numpy()
        offset += len(images)
    np.save(target, values)
    write_json(marker, {"fingerprint": fingerprint, "sha256": file_hash(target), "device": str(device),
                        "eval": not model.training, "training_performed": False})


def metric_means(rows, keys):
    return {key: float(np.mean([row[key] for row in rows if row[key] is not None]))
            if any(row[key] is not None for row in rows) else None for key in keys}


def summarize(rows, method, split, bootstrap, seed):
    keys = [key for key in rows[0] if key not in ("query_id", "label_id", "num_positives", "has_region", "selected_fraction")]
    by_class = {i: [row for row in rows if row["label_id"] == i] for i in range(8)}
    class_results, table = [], []
    for label_id, name in enumerate(CLASS_NAMES):
        group = by_class[label_id]
        metrics = metric_means(group, keys)
        entry = {"method": method, "split": split, "averaging": "class", "label_id": label_id, "label_name": name,
                 "num_queries": len(group), "queries_with_positives": sum(row["num_positives"] > 0 for row in group), **metrics}
        table.append(entry)
        if group:
            class_results.append(metrics)
    micro = metric_means(rows, keys)
    macro = metric_means(class_results, keys)
    for name, metrics in (("micro", micro), ("macro", macro)):
        table.insert(0 if name == "micro" else 1,
                     {"method": method, "split": split, "averaging": name, "label_id": "", "label_name": "all",
                      "num_queries": len(rows), "queries_with_positives": sum(row["num_positives"] > 0 for row in rows), **metrics})
    rng = np.random.default_rng(seed)
    intervals = {key: bootstrap_interval([row[key] for row in rows], bootstrap, rng) for key in ("mAP", "Top1Accuracy")}
    return table, {"micro": micro, "macro": macro, "micro_query_bootstrap_95ci": intervals,
                   "valid_queries_by_metric": {key: sum(row[key] is not None for row in rows) for key in keys},
                   "macro_classes_by_metric": {key: sum(group[key] is not None for group in class_results) for key in keys},
                   "per_class": table[2:], "macro_definition": "equal mean of present class means; undefined AP classes excluded"}


def score_and_evaluate(folder, method, coverage, config, settings):
    ids = np.array([int(row["sample_id"]) for row in coverage])
    classes = np.array([int(row["label_id"]) for row in coverage])
    present = np.array([bool(int(row["has_region"])) for row in coverage])
    features = np.load(folder / ("h512.npy" if method == "proposed" else "zernike47.npy"))
    learned = normalized(features) if method == "proposed" else None
    matrix = np.clip(learned @ learned.T, 0, 1) if learned is not None else None
    output = folder / "methods" / method
    output.mkdir(parents=True, exist_ok=True)
    rows, top, confusion = [], {}, np.zeros((8, 8), dtype=int)
    fields = ["query_id", "rank", "candidate_id", "similarity_score", "same_label"]
    with (output / "rankings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for qi, query in enumerate(ids):
            if matrix is None:
                scores = np.exp(-np.linalg.norm(features.astype(np.float64) - features[qi], axis=1) / config["handcrafted"]["distance_scale"])
            else:
                scores = matrix[qi].copy()
            scores[~present] = 0
            if not present[qi]:
                scores[:] = 0
            candidates = np.flatnonzero(ids != query)
            order = candidates[np.lexsort((ids[candidates], -scores[candidates]))]
            relevant = (classes[order] == classes[qi]).astype(float)
            metrics = query_metrics(relevant, relevant, config["evaluation"]["ks"])
            metrics.update(mAP=metrics["mAP_full"], Top1Accuracy=float(relevant[0]))
            for k in config["evaluation"]["ks"]:
                for target, original in (("NDCG", "ContainmentNDCG"), ("Precision", "FullCoveragePrecision"),
                                         ("Recall", "FullCoverageRecall"), ("Hit", "FullCoverageHit")):
                    metrics[f"{target}@{k}"] = metrics[f"{original}@{k}"]
            rows.append({"query_id": int(query), "label_id": int(classes[qi]), "num_positives": int(relevant.sum()),
                         "has_region": int(present[qi]), "selected_fraction": float(coverage[qi]["selected_fraction"]), **metrics})
            writer.writerows({"query_id": int(query), "rank": rank, "candidate_id": int(ids[ci]),
                              "similarity_score": float(scores[ci]), "same_label": int(classes[qi] == classes[ci])}
                             for rank, ci in enumerate(order, 1))
            top[int(query)] = [(int(ci), float(scores[ci])) for ci in order[:3]]
            confusion[classes[qi], classes[order[0]]] += 1
    write_csv(output / "per_query_metrics.csv", rows, list(rows[0]))
    table, summary = summarize(rows, method, folder.name, settings["bootstrap"], settings["seed"])
    write_json(output / "metrics.json", {**summary, "relevance": "same single label = 1; different label = 0",
               "num_queries": len(ids), "candidates_per_query": len(ids) - 1,
               "ties": "descending score, ascending candidate ID", "no_region_score": 0,
               "representation": "h512 cosine clipped [0,1]" if learned is not None else "unchanged Zernike47, exp(-Euclidean/distance_scale)"})
    write_csv(output / "top1_confusion.csv", ({"true_label": CLASS_NAMES[i],
              **{name: int(confusion[i, j]) for j, name in enumerate(CLASS_NAMES)}} for i in range(8)),
              ["true_label", *CLASS_NAMES])
    return table, rows, top, confusion


def paired_comparison(left, right, settings):
    if [row["query_id"] for row in left] != [row["query_id"] for row in right]:
        raise ValueError("Methods must use identical query order")
    output = []
    for metric in ("mAP", "Top1Accuracy", "NDCG@10"):
        if metric not in left[0]:
            continue
        groups = {i: [a[metric] - b[metric] for a, b in zip(left, right)
                      if a["label_id"] == i and a[metric] is not None and b[metric] is not None] for i in range(8)}
        values = [value for group in groups.values() for value in group]
        rng = np.random.default_rng(settings["seed"])
        class_values = [np.asarray(group) for group in groups.values() if group]
        macro_draws = [float(np.mean([rng.choice(group, len(group), replace=True).mean() for group in class_values]))
                       for _ in range(settings["bootstrap"])] if class_values else []
        output.append({"left": "proposed", "right": "handcrafted", "metric": metric,
                       "micro_mean_difference": float(np.mean(values)) if values else None,
                       "micro_paired_query_bootstrap_95ci": bootstrap_interval(values, settings["bootstrap"], rng),
                       "macro_mean_difference": float(np.mean([group.mean() for group in class_values])) if class_values else None,
                       "macro_stratified_paired_bootstrap_95ci": np.quantile(macro_draws, [.025, .975]).tolist() if macro_draws else None})
    return output


def draw_review(source, folder, coverage, tops, confusions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    maps = np.load(source / "cache/maps.npy", mmap_mode="r")
    masks = np.load(folder / "region_masks.npy", mmap_mode="r")
    patches = np.load(folder / "patches.npy", mmap_mode="r")
    figure_dir = folder / "figures"
    figure_dir.mkdir(exist_ok=True)
    anchors = [next((i for i, row in enumerate(coverage) if int(row["label_id"]) == label), None) for label in range(8)]
    anchors = [i for i in anchors if i is not None]
    figure, axes = plt.subplots(len(anchors), 9, figsize=(23, 2.5 * len(anchors)), squeeze=False)
    for row_id, qi in enumerate(anchors):
        item = coverage[qi]
        query = int(item["sample_id"])
        tiles = [(maps[query] == 2, f"map {query}\n{item['label_name']}"),
                 (masks[qi], f"selected / all defects\n{float(item['selected_fraction']):.1%}"), (patches[qi, 0], "96x96 patch")]
        for method in METHODS:
            for ci, score in tops[method][query]:
                candidate = coverage[ci]
                tiles.append((patches[ci, 0], f"{method}: {candidate['sample_id']}\n{candidate['label_name']}, {score:.5f}"))
            while len(tiles) < (6 if method == "handcrafted" else 9):
                tiles.append((np.zeros_like(patches[qi, 0]), "unavailable"))
        for ax, (pixels, title) in zip(axes[row_id], tiles):
            ax.imshow(pixels, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    figure.tight_layout()
    figure.savefig(figure_dir / "map_region_neighbors.png", dpi=110)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, method in zip(axes, METHODS):
        matrix = confusions[method]
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(8), CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticks(range(8), CLASS_NAMES)
        ax.set_xlabel("top1 candidate label")
        ax.set_ylabel("query label")
        ax.set_title(method)
        for i in range(8):
            for j in range(8):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(figure_dir / "top1_confusion.png", dpi=110)
    plt.close(figure)


def write_report(output, table, model_info):
    def fmt(value):
        return "NA" if value is None else f"{value:.4f}"
    lines = ["# 单标签最大区域检索", "", "仅推理现有 best.pt，无训练。保留原裁剪流程，推理 top_k=1，Zernike47 保持不变。",
             f"checkpoint epoch={model_info['epoch']}，训练 top_k={model_info['training_proposal']['top_k']}。",
             "每个 split 内所有单标签 map 作 query，其余同 split 单标签 map 作 candidate，排除自身。",
             "", "| split | 方法 | 平均方式/类别 | 查询数 | mAP | Top1 | NDCG@10 |", "|---|---|---|---:|---:|---:|---:|"]
    for row in table:
        group = row["label_name"] if row["averaging"] == "class" else row["averaging"]
        lines.append(f"| {row['split']} | {row['method']} | {group} | {row['num_queries']} | {fmt(row['mAP'])} | "
                     f"{fmt(row['Top1Accuracy'])} | {fmt(row.get('NDCG@10'))} |")
    lines += ["", "- macro 对实际出现的类别等权平均；micro 对所有查询等权平均。",
              "- 无同类候选时 AP/Recall/Hit/NDCG 为 NA；Top1 仍为 0。各指标有效查询和类别数见 metrics.json。",
              "- 单标签不等于单区域。top1 消除了多区域分配和面积加权，仍保留最大连通域选择的影响。",
              "- region_coverage.csv 统计最大区域 / 全部缺陷像素；其中包含噪声，比例不是语义覆盖真值。",
              "- shape-only：pos/scale 不融合。裁剪及归一化可能让依赖晶圆位置的类别难分。",
              "- 同标签候选的 ground truth 同为 1，互换不扣分；预测分数并列时按 candidate ID 排序。",
              "- 先用 valid 检查设计，固定后运行 test；不能与主实验混合标签、固定采样池的指标直接比较。",
              "- figures/map_region_neighbors.png 展示完整缺陷图、最大区域、patch 和两方法近邻，标签是晶圆级标签。",
              "- paired_comparison.json 的区间只反映当前池的查询抽样波动，不含训练 seed 波动。",
              "- source_audit.json 核验源实验文件未改变；run_manifest.json 记录输入、代码哈希及环境。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_single_label(source, output, device, settings, resume=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    guard_output(source, output)
    if output.exists() and any(output.iterdir()) and (not resume or not (output / "run_manifest.json").is_file()):
        raise ValueError("Output is not empty; choose a new output or --resume an identical single-label run")
    config = load_config(source / "resolved_config.json")
    if config["handcrafted"]["max_order"] != 12 or config["input_size"] != 96:
        raise ValueError("This test preserves the current Zernike47 and 96x96 region design")
    if any(config["matching"][key] != 0 for key in ("lambda_pos", "lambda_scale", "shape_threshold", "score_threshold")):
        raise ValueError("Expected the existing shape-only zero-threshold matching settings")
    before = source_snapshot(source)
    hashes = source_inputs(source, config, settings["splits"])
    if any(Path(path).is_relative_to(output) for path in hashes):
        raise ValueError("Output must not contain source inputs")
    code_files = [Path(__file__).with_name(name) for name in
                  ("single_label.py", "models.py", "regions.py", "evaluation.py", "diagnostics.py", "training.py", "common.py")]
    code_files += [Path(__file__).parent / "methods/handcrafted.py",
                  Path(__file__).parent.parent / "scripts/evaluate_single_label.py"]
    identity = {"input_sha256": hashes, "code_sha256": {str(path.name): file_hash(path) for path in code_files},
                "settings": settings, "device": str(device), "python": sys.version, "platform": platform.platform(),
                "packages": {name: version(name) for name in ("torch", "numpy", "scipy", "matplotlib")}}
    fingerprint = digest(identity)
    manifest_path = output / "run_manifest.json"
    if manifest_path.is_file() and read_json(manifest_path)["fingerprint"] != fingerprint:
        raise ValueError("Source/code/settings/environment changed; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    status = {**identity, "fingerprint": fingerprint, "source": str(source), "output": str(output),
              "status": "running", "training_performed": False, "mps_fallback": False if device.type == "mps" else None}
    write_json(manifest_path, status)
    started = time.monotonic()
    with diagnostic_log(output):
        print(f"Single-label top1 inference on {device}; source={source}; output={output}", flush=True)
        try:
            model, info = load_saved_model(source, config, device)
            write_json(output / "model_info.json", info)
            table = []
            for split in settings["splits"]:
                protocol = prepare_top1(source, output, config, split)
                print(f"{split}: {protocol['queries']} queries x {protocol['candidates_per_query']} candidates; "
                      f"no-region maps={protocol['maps_without_region']}", flush=True)
                folder = output / split
                infer_features(model, folder, device, settings, fingerprint)
                coverage = csv_rows(folder / "region_coverage.csv")
                rows, tops, confusions = {}, {}, {}
                for method in METHODS:
                    entries, rows[method], tops[method], confusions[method] = score_and_evaluate(folder, method, coverage, config, settings)
                    table.extend(entries)
                    overall = next(row for row in entries if row["averaging"] == "macro")
                    print(f"{split}/{method}: macro mAP={overall['mAP']}, Top1={overall['Top1Accuracy']}", flush=True)
                write_json(folder / "paired_comparison.json", paired_comparison(rows["proposed"], rows["handcrafted"], settings))
                draw_review(source, folder, coverage, tops, confusions)
            write_csv(output / "comparison.csv", table, list(table[0]))
            write_json(output / "comparison.json", table)
            write_report(output, table, info)
            status["status"] = "complete"
        finally:
            after = source_snapshot(source)
            changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
            changed_hashes = [path for path, expected in hashes.items() if not Path(path).is_file() or file_hash(path) != expected]
            audit = {"source_unchanged": not changed and not changed_hashes, "changed_source_files": changed,
                     "changed_hashed_inputs": changed_hashes, "source_files_checked": len(before),
                     "check": "all source paths/size/mtime; SHA256 of required source inputs"}
            write_json(output / "source_audit.json", audit)
            if status["status"] != "complete" or not audit["source_unchanged"]:
                status["status"] = "incomplete"
            status.update(seconds=time.monotonic() - started, source_unchanged=audit["source_unchanged"])
            write_json(manifest_path, status)
        if not audit["source_unchanged"]:
            raise RuntimeError("Source experiment changed during evaluation; inspect source_audit.json, including concurrent jobs")
        print(f"Complete in {status['seconds']:.1f}s; source_unchanged=True; report={output / 'report.md'}", flush=True)
    return output
