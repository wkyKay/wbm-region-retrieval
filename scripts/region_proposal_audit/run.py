#!/usr/bin/env python3
"""Audit label-free region proposals on one frozen-train sample per class.

Run with compared_methods/.venv/bin/python. Existing output directories are
rejected. No training, protocol generation, test evaluation or production edits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from methods import METHODS, WBBS_SOURCE, extract_all, f_budget
from render import palette, write_sheets, write_viewer

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from wbm_retrieval.regions import extract_regions

CLASS_NAMES = ("Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Scratch", "Random")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_samples(data, split_file, seed):
    with np.load(data, allow_pickle=False) as z:
        labels = z["arr_1"]
    if labels.shape != (38015, 8) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Unexpected MixedWM38 labels")
    groups = defaultdict(list)
    with split_file.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != "train":
                continue
            sample_id = int(row["sample_id"])
            if not np.array_equal(labels[sample_id], [int(row[f"label_{i}"]) for i in range(8)]):
                raise ValueError("Frozen split and original labels disagree")
            groups[tuple(np.flatnonzero(labels[sample_id]).tolist())].append(sample_id)
    if len(groups) != 37 or sum(map(len, groups.values())) != 25193:
        raise ValueError("Expected frozen train with 37 classes and 25193 examples")
    # This mirrors the existing Normal policy. No central split is overwritten.
    normal_ids = np.flatnonzero(labels.sum(1) == 0)
    if len(normal_ids) != 1000:
        raise ValueError("Unexpected Normal count")
    groups[()] = sorted(np.random.default_rng(seed).permutation(normal_ids)[:700].tolist())
    rng = np.random.default_rng(seed)
    rows = []
    for signature in sorted(groups, key=lambda s: (len(s), s)):
        ids = sorted(groups[signature])
        sample_id = int(rng.choice(ids))
        rows.append(dict(
            sample_id=sample_id,
            class_name=" + ".join(CLASS_NAMES[i] for i in signature) or "Normal",
            label_signature="|".join(map(str, signature)),
            split="normal_train" if not signature else "train",
            eligible_count=len(ids),
            **{f"label_{i}": int(labels[sample_id, i]) for i in range(8)},
        ))
    return rows, labels[np.array([row["sample_id"] for row in rows])].astype(np.uint8)


def read_selected_maps(data, ids):
    """Stream the compressed NPY in original row order; avoid a full 0.8GB array."""
    selected = {}
    with zipfile.ZipFile(data) as archive, archive.open("arr_0.npy") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError(f"Unsupported NPY version {version}")
        if shape != (38015, 52, 52) or fortran or dtype.hasobject:
            raise ValueError("Unexpected map storage layout")
        row_bytes = 52 * 52 * dtype.itemsize
        current = 0
        for sample_id in sorted(ids):
            remaining = (sample_id - current) * row_bytes
            while remaining:
                piece = f.read(min(remaining, 8 * 1024 * 1024))
                if not piece:
                    raise ValueError("Truncated NPZ")
                remaining -= len(piece)
            values = f.read(row_bytes)
            raw = np.frombuffer(values, dtype=dtype).reshape(52, 52)
            if not np.isin(raw, [0, 1, 2, 3]).all():
                raise ValueError("Unexpected pixel codes")
            selected[sample_id] = raw.astype(np.uint8)
            current = sample_id + 1
    return np.stack([selected[i] for i in ids])


def documentation(output, samples, stats, elapsed):
    methods = "\n".join(f"| {m['name']} | {m['status']} | {m['detail']} |" for m in METHODS)
    text = f'''# MixedWM38 区域像素划分验证

日期：2026-11-11。seed=2026。38种完整组合（37缺陷类＋Normal）各1张，共38张。
这是肉眼检查材料，不报告分割准确率或检索性能；一类一张不代表总体结果。

打开 `index.html` 可切换类别、查看原生候选与F预算过滤后的结果，悬停查询像素区域编号，
点击突出单个区域。`images/` 包含两套各8张五类一页的对照图及每类独立高清对照图。

## 方法与范围

| 方法 | 实现状态 | 具体设置 |
| --- | --- | --- |
{methods}

NPFC-pre 仅是已核实子流程，详见 `NPFC_NOTES.md`。没有完整NPFC的SD选簇/核心簇分配。
Tensor Voting＋Marching (2022, DOI 10.1109/TSM.2022.3183008) 缺少完整算法和参数，未实现。
SePArate、LightWMNet等分割模型需要监督训练，未作为F的无标签替代运行；DECOR是整图聚类。
表中中值、开运算、TV后接8连通都是明确组合方案，不能声称是对应原论文的完整区域方法。

来源：
- WBBS去噪：[Wang & Wang 2023](https://doi.org/10.1016/j.cie.2023.109679)，复用 compared_methods 的已审计源码。sigma_d=9是既有实现约定，非已确认的原文默认值。
- NPFC子流程：[Hou et al. 2024](https://doi.org/10.1007/s10836-024-06132-8)。原文使用WM-811K，本次为迁移检查。
- 中值滤波操作参考：[Lee et al. 2021](https://doi.org/10.1109/TSM.2021.3102679)。这里只借用3×3中值，不复现后续DPGMM/HCM。
- 连通域与开运算参考：[LightWMNet §3.1](https://doi.org/10.1049/cit2.70075)。这里的十字核1次是统一实验约定，不按类别选规则，不复现其监督模型。

## 抽样与协议

缺陷类只从项目 `protocol/deduplicated/split.csv` 的 train 代表ID中抽取。
Normal按已有规则用seed=2026随机排列1000个Normal ID，取前700个作为normal_train候选。
按“模式数量，再按标签索引元组”排序38类，以独立 `default_rng(2026)` 逐类从排序ID中抽1个。
本次没有看图后换样本。sample_id是原始NPZ行号。标签顺序为
Center、Donut、Edge-Loc、Edge-Ring、Loc、Near-full、Scratch、Random。
标签只用于抽样、标题和核查，任何区域提取函数都不读取类别标签。
原始NPZ哈希和split哈希与冻结metadata核对；运行前后再次核对保护文件哈希。
没有修改原始数据、冻结划分、检索清单、F实现或既有产物，也没有运行test或训练。

## 两套视图

- `native`：各算法原生候选，没有统一的min_area5/top4；方法自身去噪仍生效。
- `f_budget`：在每种算法结果上按真实缺陷面积≥5过滤，再按面积降序保留4个，
  同面积按首个像素行优先位置排序。8连通列与当前F的extract_regions逐像素核对。

这一区分能观察碎片化与去噪，也能比较同一F预算下的区域遗漏。
NPFC-pre合并区域可能不连通；不补绘两个片段之间的正常像素。
中值滤波会添加一些支撑像素，先在滤波支撑图上分组再映射回原始缺陷集合；
新增点只可能影响组别，不成为输出缺陷像素。statistics/diagnostics记录相应数量。
所有输出均为raw==2的子集，0/3不参与提取，1代表有效正常die。

## 颜色与阅读

彩色表示区域编号（按面积排序），不表示Center/Scratch等语义类别；跨方法同色不保证对应。
深灰是原始缺陷图中未被当前方法/预算保留的像素；浅灰是正常die，白色是0/3。
原图全部缺陷显示为深灰。PNG为≥20像素区域加编号；小区域用网页悬停读取。
同一个面板的区域使用唯一RGB值，碎片很多时相近颜色仍可能难区分，可在网页中点击突出区域。
两套视图显示原始52×52空间，不旋转、不裁剪、不平滑插值。

## 产物与像素读取

- `sample_manifest.csv`：38个原始ID、类别、抽样来源及缺陷像素数。
- `selected_samples.npz`：raw_maps、labels、sample_ids。
- `regions_native.npz` / `regions_f_budget.npz`：每个方法一个 `[38,52,52]` int32数组，附sample_ids。
- `statistics.csv`：每个样本×方法×视图的区域数、面积、保留和过滤像素数。
- `diagnostics.json`：中值新增支撑及NPFC子流程诊断。
- `validation.json`：F等价性、像素子集、ID/标签完整性、保护文件哈希检查。
- `manifest.json` / `source/`：参数、依赖版本、源码副本与哈希，便于重现。

```python
import numpy as np
raw = np.load('selected_samples.npz')['raw_maps'][0]
label_map = np.load('regions_native.npz')['npfc_pre'][0]
region_1_mask = label_map == 1
region_1_pixels = np.argwhere(region_1_mask)  # 原图(row,col)，从0计数
unassigned_defects = (raw == 2) & (label_map == 0)
```

共有 {len(stats)} 条统计记录。提取38图及校验阶段耗时约 {elapsed:.2f} 秒，不含绘图。

## 重现

从项目根目录运行，输出目录必须不存在：

```bash
../compared_methods/.venv/bin/python scripts/region_proposal_audit/run.py \\
  --output artifacts/region_proposal_audit_NEW --seed 2026
```
'''
    (output / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/region_proposal_audit_20261111_v1")
    parser.add_argument("--seed", type=int, choices=[2026], default=2026,
                        help="This version freezes both sampling and Normal allocation at 2026")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    data = PROJECT.parent / "data/wm38k/Wafer_Map_Datasets.npz"
    split_file = PROJECT / "protocol/deduplicated/split.csv"
    metadata_path = split_file.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    protected = [data, split_file, metadata_path, split_file.parent / "duplicates.csv",
                 PROJECT / "wbm_retrieval/regions.py", PROJECT / "experiments/proposed/f/config.json", WBBS_SOURCE]
    protected += sorted((PROJECT / "protocol/retrieval_v1").glob("*"))
    protected = [p for p in protected if p.is_file()]
    before = {str(p): sha256(p) for p in protected}
    if before[str(data)] != metadata["source_sha256"] or before[str(split_file)] != metadata["files"]["split.csv"]:
        raise ValueError("Source/protocol hash mismatch")
    samples, selected_labels = select_samples(data, split_file, args.seed)
    maps = read_selected_maps(data, [s["sample_id"] for s in samples])
    assert len(samples) == 38 and len({s["sample_id"] for s in samples}) == 38
    assert len(np.unique(selected_labels, axis=0)) == 38
    output.mkdir(parents=True)
    start = time.perf_counter()
    results = {mode: {m["id"]: [] for m in METHODS} for mode in ("native", "f_budget")}
    stats, diagnostics = [], []
    validation = dict(samples=38, classes=38, methods=len(METHODS), test_samples=0,
                      sampling_labels_used=True, extraction_labels_used=False,
                      original_pixel_subsets=True, mutually_exclusive_masks=True,
                      cc4_cc8_cover_original=True, f_equivalent_samples=0,
                      empty_outputs_allowed=True, before_hashes=before)
    for idx, (sample, raw) in enumerate(zip(samples, maps)):
        total = int((raw == 2).sum())
        sample["defect_pixels"] = total
        sample["raw_sha256_uint8"] = hashlib.sha256(raw.tobytes()).hexdigest()
        native, diag = extract_all(raw)
        budget = {key: f_budget(value) for key, value in native.items()}
        expected = extract_regions(raw, min_area=5, top_k=4, padding=4)
        assert len(expected) == int(budget["cc8"].max())
        for rid, region in enumerate(expected, 1):
            assert np.array_equal(region["mask"], budget["cc8"] == rid)
        validation["f_equivalent_samples"] += 1
        for key in ("cc4", "cc8"):
            assert np.array_equal(native[key] > 0, raw == 2)
        assert native["cc4"].max() >= native["cc8"].max()
        for mode, values in (("native", native), ("f_budget", budget)):
            for key, labels in values.items():
                n = int(labels.max())
                assert np.array_equal(np.unique(labels[labels > 0]), np.arange(1, n + 1))
                assert not np.any((labels > 0) & (raw != 2))
                if mode == "f_budget":
                    assert n <= 4 and all((labels == i).sum() >= 5 for i in range(1, n + 1))
                colors = palette(n)
                assert len({tuple(c) for c in colors}) == len(colors)
                kept = int((labels > 0).sum())
                results[mode][key].append(labels)
                stats.append(dict(sample_id=sample["sample_id"], class_name=sample["class_name"],
                                  method=key, mode=mode, regions=n, defect_pixels=total,
                                  retained_pixels=kept, dropped_pixels=total-kept,
                                  retained_fraction=kept/total if total else "",
                                  region_areas="|".join(str(int((labels == i).sum())) for i in range(1, n+1))))
        diagnostics.append(dict(sample_id=sample["sample_id"], **diag))
        print(f"[{idx+1:02d}/38] ID {sample['sample_id']} {sample['class_name']} | " +
              " ".join(f"{k}:{int(v.max())}" for k, v in native.items()), flush=True)
    elapsed = time.perf_counter() - start
    results = {mode: {k: np.stack(v) for k, v in values.items()} for mode, values in results.items()}
    ids = np.array([s["sample_id"] for s in samples], np.int64)
    np.savez_compressed(output / "selected_samples.npz", raw_maps=maps, labels=selected_labels, sample_ids=ids)
    for mode, values in results.items():
        np.savez_compressed(output / f"regions_{mode}.npz", sample_ids=ids, **values)
    write_csv(output / "sample_manifest.csv", samples)
    write_csv(output / "statistics.csv", stats)
    write_json(output / "diagnostics.json", diagnostics)
    print("Rendering sheets and standalone viewer…", flush=True)
    sheets = write_sheets(output, samples, maps, results, METHODS)
    write_viewer(output, samples, maps, results, METHODS, stats, sheets)
    documentation(output, samples, stats, elapsed)
    shutil.copyfile(Path(__file__).with_name("NPFC_NOTES.md"), output / "NPFC_NOTES.md")
    (output / "source").mkdir()
    source_paths = list(Path(__file__).parent.glob("*.py")) + [WBBS_SOURCE]
    source_hashes = {}
    for path in source_paths:
        shutil.copyfile(path, output / "source" / path.name)
        source_hashes[str(path)] = sha256(path)
    after = {str(p): sha256(p) for p in protected}
    assert before == after
    validation.update(after_hashes=after, protected_files_unchanged=True,
                      all_checks_passed=True, figure_pages=len(sheets), per_class_images=76)
    write_json(output / "validation.json", validation)
    import scipy, matplotlib
    write_json(output / "manifest.json", dict(date="2026-11-11", seed=args.seed, samples=38,
               protocol="frozen_deduplicated_train_only", methods=METHODS, source_hashes=source_hashes,
               numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__,
               python=sys.version, command=sys.argv, extraction_seconds=elapsed))
    print(f"COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
