#!/usr/bin/env python3
"""Compare geometric region extractors on the previous audit's frozen examples.

Only new audit artifacts are written. Method code does not receive class labels.
Extraction is cached per image with a source signature; mismatched caches fail.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
PREVIOUS = PROJECT / "artifacts/region_proposal_audit_20261111_v2"
sys.path.insert(0, str(PROJECT))
from wbm_retrieval.regions import extract_regions
from render import panel, write_sheets, write_viewer


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def csv_write(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def canonical(labels):
    ids = np.unique(labels[labels > 0])
    masks = [labels == k for k in ids]
    masks.sort(key=lambda m: (-int(m.sum()), int(np.flatnonzero(m)[0])))
    out = np.zeros(labels.shape, np.int32)
    for k, mask in enumerate(masks, 1):
        out[mask] = k
    return out


def budget(labels):
    masks = [labels == k for k in range(1, int(labels.max()) + 1)]
    masks = [m for m in masks if int(m.sum()) >= 5][:4]
    out = np.zeros(labels.shape, np.int32)
    for k, mask in enumerate(masks, 1):
        out[mask] = k
    return out


def cc8_regions(raw, seed=2026):
    labels, _ = ndimage.label(raw == 2, np.ones((3, 3), np.uint8))
    return canonical(labels), {"implementation": "original-defect 8-connected components"}


def verify(raw, labels):
    assert labels.shape == raw.shape
    assert labels.dtype.kind in "iu"
    assert labels.min() >= 0
    assert not np.any((labels > 0) & (raw != 2))
    assert np.array_equal(np.unique(labels[labels > 0]), np.arange(1, int(labels.max()) + 1))


def load_samples():
    with np.load(PREVIOUS / "selected_samples.npz", allow_pickle=False) as archive:
        maps = archive["raw_maps"]
        target_labels = archive["labels"]
        ids = archive["sample_ids"]
    with (PREVIOUS / "sample_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        samples = list(csv.DictReader(stream))
    assert len(samples) == len(maps) == len(ids) == len(target_labels) == 38
    for sample, raw, sid, targets in zip(samples, maps, ids, target_labels):
        for key in ("sample_id", "eligible_count", "defect_pixels"):
            sample[key] = int(sample[key])
        assert int(sid) == sample["sample_id"]
        assert sample["raw_sha256_uint8"] == hashlib.sha256(raw.astype(np.uint8).tobytes()).hexdigest()
        assert sample["split"] in ("train", "normal_train")
        assert np.array_equal(targets, [int(sample[f"label_{i}"]) for i in range(8)])
    assert maps.shape == (38, 52, 52)
    assert len(np.unique(target_labels, axis=0)) == 38
    assert len(np.unique(ids)) == 38
    return samples, maps, ids


def metadata():
    # Actual algorithm details/parameters are also saved in each diagnostics file.
    return [
        dict(id="cc8", title="8-connected / F base", name="8 连通 / F 基础", status="基础方法",
             detail="直接对原始缺陷像素做8连通；F预算视图与当前F逐像素核对。"),
        dict(id="tlinkage", title="T-Linkage (lines)", name="T-Linkage：多直线", status="作者算法 Python 移植；直线模型适配",
             detail="5N直线假设，tau=0.5px（截断2.5px），连续偏好＋Tanimoto聚合；使用作者demo的簇大小跳变离群规则，含至少两个模型的倾向。不是论文完整二项概率过滤。所有类别参数相同。"),
        dict(id="multix", title="Multi-X (lines + circles)", name="Multi-X：直线＋圆", status="核心算法移植；新增线／圆估计器",
             detail="Python外循环＋作者原C++图割；新增线／圆估计器，使用公开代码Mean-Shift（非论文Median-Shift），补论文100次模型验证。gamma=2px、固定带宽2px、hmax=10均统一，不依据类别。详见MULTIX_NOTES.md。"),
    ]


def method_function(name):
    if name == "cc8":
        return cc8_regions
    if name == "tlinkage":
        from tlinkage import tlinkage_regions
        return tlinkage_regions
    from multix import multix_regions
    return multix_regions


def method_signature(name):
    # The whole method source and native solver source participate in the cache key.
    paths = [HERE / (name + ".py")] if name != "cc8" else [PROJECT / "wbm_retrieval/regions.py"]
    if name == "multix":
        paths += sorted(HERE.glob("*.cpp")) + sorted(HERE.glob("*.h"))
        paths += sorted(p for p in (HERE / "author_multi_x").rglob("*")
                        if p.is_file() and p.suffix in (".cpp", ".h", ".hpp", ".dylib", ".so"))
    return {str(path.relative_to(PROJECT)): digest(path) for path in paths}


def synthetic_cases():
    yy, xx = np.indices((52, 52))
    valid = (yy - 25.5) ** 2 + (xx - 25.5) ** 2 <= 25 ** 2
    raw = np.where(valid, 1, 0).astype(np.uint8)
    line1 = (yy == 26) & (xx >= 7) & (xx <= 44)
    line2 = (xx == 26) & (yy >= 7) & (yy <= 44)
    crossing = raw.copy()
    crossing[line1 | line2] = 2
    circle = np.abs(np.hypot(yy - 26, xx - 26) - 14) < .55
    mixed = raw.copy()
    mixed[line1 | circle] = 2
    return [("Crossing lines (synthetic)", crossing), ("Line + circle (synthetic)", mixed)]


def cached_extract(output, name, token, raw, seed=2026):
    folder = output / "cache" / name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{token}.npz"
    signature = dict(source=method_signature(name), raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(), seed=seed)
    if path.exists():
        details = json.loads(path.with_suffix(".json").read_text())
        if details["signature"] != signature:
            raise ValueError(f"Cache signature changed: {path}; use a new output directory")
        with np.load(path, allow_pickle=False) as archive:
            labels = archive["labels"]
        verify(raw, labels)
        return labels, details["diagnostics"]
    start = time.perf_counter()
    original_labels, diag = method_function(name)(raw.copy(), seed=seed)
    original_labels = np.asarray(original_labels)
    labels = canonical(original_labels)
    id_map = {str(int(old)): int(labels[original_labels == old][0])
              for old in np.unique(original_labels[original_labels > 0])}
    verify(raw, labels)
    diag = dict(diag, elapsed_seconds=time.perf_counter() - start,
                original_to_display_region_id=id_map)
    np.savez_compressed(path, labels=labels)
    json_write(path.with_suffix(".json"), dict(signature=signature, diagnostics=diag))
    return labels, diag


def guard(output):
    paths = [PROJECT.parent / "data/wm38k/Wafer_Map_Datasets.npz",
             PROJECT / "wbm_retrieval/regions.py", PROJECT / "experiments/proposed/f/config.json"]
    paths += sorted(p for p in (PROJECT / "protocol").rglob("*") if p.is_file())
    paths += sorted(p for p in PREVIOUS.rglob("*") if p.is_file())
    current = {str(p): digest(p) for p in paths}
    marker = output / "protected_hashes.json"
    if marker.exists():
        assert current == json.loads(marker.read_text()), "A protected input changed"
    else:
        json_write(marker, current)
    return current


def extraction(output, names):
    samples, maps, _ = load_samples()
    for name in names:
        for i, (sample, raw) in enumerate(zip(samples, maps), 1):
            labels, diag = cached_extract(output, name, f"id_{sample['sample_id']}", raw)
            print(f"{name} [{i:02d}/38] ID {sample['sample_id']}: {labels.max()} regions, "
                  f"{np.count_nonzero(labels)}/{np.count_nonzero(raw == 2)} pixels, "
                  f"{diag.get('elapsed_seconds', 0):.2f}s", flush=True)
        for i, (_, raw) in enumerate(synthetic_cases()):
            cached_extract(output, name, f"synthetic_{i}", raw)


def assemble(output):
    import matplotlib.pyplot as plt
    samples, maps, ids = load_samples()
    methods = metadata()
    results = {mode: {m["id"]: [] for m in methods} for mode in ("native", "f_budget")}
    stats, diagnostics = [], []
    f_equal = 0
    for sample, raw in zip(samples, maps):
        sample_diag = dict(sample_id=sample["sample_id"])
        for method in methods:
            key = method["id"]
            path = output / "cache" / key / f"id_{sample['sample_id']}.npz"
            if not path.exists():
                raise FileNotFoundError(f"Run extraction first: {path}")
            native, diag = cached_extract(output, key, f"id_{sample['sample_id']}", raw)
            sample_diag[key] = diag
            filtered = budget(native)
            if key == "cc8":
                expected = extract_regions(raw, min_area=5, top_k=4)
                assert int(filtered.max()) == len(expected)
                for rid, region in enumerate(expected, 1):
                    assert np.array_equal(filtered == rid, region["mask"])
                f_equal += 1
            for mode, labels in (("native", native), ("f_budget", filtered)):
                verify(raw, labels)
                areas = [int((labels == k).sum()) for k in range(1, int(labels.max()) + 1)]
                assert areas == sorted(areas, reverse=True)
                if mode == "f_budget":
                    assert len(areas) <= 4 and all(a >= 5 for a in areas)
                results[mode][key].append(labels)
                total, kept = int((raw == 2).sum()), int((labels > 0).sum())
                stats.append(dict(sample_id=sample["sample_id"], class_name=sample["class_name"],
                    method=key, mode=mode, regions=len(areas), defect_pixels=total,
                    retained_pixels=kept, dropped_pixels=total-kept,
                    retained_fraction=kept/total if total else 0., region_areas="|".join(map(str, areas))))
        diagnostics.append(sample_diag)
    results = {mode: {key: np.stack(value) for key, value in methods_.items()} for mode, methods_ in results.items()}
    for mode, values in results.items():
        np.savez_compressed(output / f"regions_{mode}.npz", sample_ids=ids, **values)
    shutil.copyfile(PREVIOUS / "selected_samples.npz", output / "selected_samples.npz")
    shutil.copyfile(PREVIOUS / "sample_manifest.csv", output / "sample_manifest.csv")
    csv_write(output / "statistics.csv", stats)
    json_write(output / "diagnostics.json", diagnostics)
    print("Rendering all 38 classes and synthetic controls...", flush=True)
    sheets = write_sheets(output, samples, maps, results, methods)
    write_viewer(output, samples, maps, results, methods, stats, sheets)
    shutil.copyfile(HERE / "README.md", output / "README.md")
    # These are fixed indices, selected before reading algorithm outputs, not a success-only gallery.
    selected = [7, 12, 16, 37]  # fixed Scratch-containing rows; full set is always delivered
    preview_ids = []
    for mode in results:
        fig, axes = plt.subplots(len(selected), 4, figsize=(13.6, 14.6), squeeze=False)
        for row, index in enumerate(selected):
            sample, raw = samples[index], maps[index]
            preview_ids.append(sample["sample_id"]) if mode == "native" else None
            label = textwrap.fill(f"{index+1:02d}. {sample['class_name']}", width=27, break_long_words=False)
            panel(axes[row, 0], raw, None, f"{label}\nID {sample['sample_id']}")
            for col, method in enumerate(methods, 1):
                panel(axes[row, col], raw, results[mode][method["id"]][index], method["title"], True)
        fig.suptitle(f"MixedWM38 | geometric region extraction | {mode}", x=.035, ha="left", fontsize=16)
        fig.text(.035, .015, "Colors = region IDs, not semantic classes. Dark = unassigned original defects.\nT-Linkage line-model port; Multi-X line/circle adaptation. All 38 fixed examples are included in the viewer.", fontsize=9)
        fig.subplots_adjust(left=.035, right=.99, top=.93, bottom=.075, wspace=.15, hspace=.66)
        fig.savefig(output / f"preview_{mode}.png", dpi=150, facecolor="white")
        plt.close(fig)
    cases = synthetic_cases()
    fig, axes = plt.subplots(len(cases), 4, figsize=(13.6, 7.8), squeeze=False)
    synthetic = {}
    for i, (title, raw) in enumerate(cases):
        panel(axes[i, 0], raw, None, title)
        synthetic[f"raw_{i}"] = raw
        for col, method in enumerate(methods, 1):
            labels, _ = cached_extract(output, method["id"], f"synthetic_{i}", raw)
            panel(axes[i, col], raw, labels, method["title"], True)
            synthetic[f"{method['id']}_{i}"] = labels
    fig.suptitle("Synthetic controls | same parameters as all MixedWM38 samples", x=.035, ha="left", fontsize=15)
    fig.text(.035, .018, "Diagnostic geometry only; these examples are not MixedWM38 images and do not establish real-data accuracy.\nEach crossing pixel has a single hard region assignment. No artificial curve pixels are added.", fontsize=9)
    fig.subplots_adjust(left=.035, right=.99, top=.89, bottom=.12, wspace=.15, hspace=.65)
    fig.savefig(output / "synthetic_controls.png", dpi=150, facecolor="white")
    plt.close(fig)
    np.savez_compressed(output / "synthetic_controls.npz", **synthetic)
    source_dir = output / "source"
    source_dir.mkdir(exist_ok=True)
    source_hashes = {}
    for path in HERE.iterdir():
        if path.is_file() and path.suffix in (".py", ".md", ".cpp", ".h", ".json", ".sh"):
            shutil.copyfile(path, source_dir / path.name)
            source_hashes[str(path)] = digest(path)
        if path.name.endswith("_NOTES.md"):
            shutil.copyfile(path, output / path.name)
    for name in ("author", "author_multi_x"):
        directory = HERE / name
        if directory.exists():
            shutil.copytree(directory, source_dir / name, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns(".git", "__pycache__", ".DS_Store", "data", ".vs"))
            for path in (source_dir / name).rglob("*"):
                if path.is_file():
                    original = directory / path.relative_to(source_dir / name)
                    source_hashes[str(original)] = digest(original)
    validation = dict(all_checks_passed=True, samples=38, identical_previous_sample_ids=True,
        identical_previous_raw_maps=True, original_defect_pixel_subsets=True,
        hard_mutually_exclusive_region_masks=True, canonical_area_order=True,
        f_budget_min_area=5, f_budget_max_regions=4, f_equivalent_samples=f_equal,
        class_labels_passed_to_extractors=False, frozen_test_used=False,
        protected_files_unchanged=True, preview_sample_ids=preview_ids)
    guard(output)
    json_write(output / "validation.json", validation)
    import scipy, matplotlib
    json_write(output / "manifest.json", dict(seed=2026, samples=38, methods=methods,
        source_sample_file=str(PREVIOUS / "selected_samples.npz"),
        source_sample_sha256=digest(PREVIOUS / "selected_samples.npz"),
        sources=source_hashes, numpy=np.__version__, scipy=scipy.__version__,
        matplotlib=matplotlib.__version__, python=sys.version, argv=sys.argv))
    print(f"COMPLETE: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("extract", "render", "all"), default="all")
    parser.add_argument("--method", choices=("all", "cc8", "tlinkage", "multix"), default="all")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and not (output / "protected_hashes.json").exists():
        raise FileExistsError(f"Refusing to reuse a directory without this audit's marker: {output}")
    output.mkdir(parents=True, exist_ok=True)
    guard(output)
    if args.stage in ("extract", "all"):
        extraction(output, ["cc8", "tlinkage", "multix"] if args.method == "all" else [args.method])
    if args.stage in ("render", "all"):
        assemble(output)


if __name__ == "__main__":
    main()
