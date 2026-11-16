"""Label-free proposal adapters for a visual audit, not production experiments."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

PROJECT = Path(__file__).resolve().parents[2]
WBBS_SOURCE = PROJECT.parent / "compared_methods/methods/wbbs/wbbs.py"
spec = importlib.util.spec_from_file_location("audit_wbbs", WBBS_SOURCE)
wbbs = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = wbbs
spec.loader.exec_module(wbbs)

METHODS = [
    dict(id="cc4", title="4-connected", name="4 连通域", status="基础方法",
         detail="原始缺陷像素的4邻域连通分量。"),
    dict(id="cc8", title="8-connected", name="8 连通域 / F 基础", status="基础方法",
         detail="原始缺陷像素的8邻域连通分量；加面积≥5、top4后与当前F完全一致。"),
    dict(id="median8", title="Median + 8CC", name="中值滤波 + 8 连通", status="操作组合，非整篇论文复现",
         detail="二值3×3中值滤波（边界补0），8连通分组后仅保留原始缺陷点。滤波新增的支撑像素可能影响分组，但不会作为缺陷输出。"),
    dict(id="opening8", title="Opening + 8CC", name="开运算 + 8 连通", status="操作组合，非LightWMNet复现",
         detail="原网格3×3十字核，开运算1次，随后8连通。不使用真实类别选择处理方式，不加前景保留率回退。"),
    dict(id="tv8", title="WBBS-TV + 8CC", name="WBBS 张量投票去噪 + 8 连通", status="本项目组合，非2022 marching",
         detail="复用已审计WBBS闭式投票；sigma_d=9像素²、阈值0.32；stick和ball均低于各自最大值×0.32才删除。随后8连通；sigma_d是已有实现约定。"),
    dict(id="npfc_pre", title="NPFC-pre*", name="NPFC-pre：路径过滤 + 近邻合并", status="论文子流程，非完整NPFC",
         detail="仅执行原文可核实的初始路径过滤与近邻簇合并；没有未能核定的SD选簇及后续核心簇分配，详见方法说明。"),
]


def canonicalize(masks, shape):
    masks = [np.asarray(m, dtype=bool) for m in masks if np.any(m)]
    # Same tie rule as F: raster order for equal areas.
    masks.sort(key=lambda m: (-int(m.sum()), int(np.flatnonzero(m)[0])))
    labels = np.zeros(shape, dtype=np.int32)
    for i, m in enumerate(masks, 1):
        if m.shape != shape or np.any(labels[m]):
            raise ValueError("Overlapping or invalid proposal masks")
        labels[m] = i
    return labels


def connected(support, defect, connectivity):
    labels, count = ndimage.label(support, ndimage.generate_binary_structure(2, connectivity))
    return canonicalize([(labels == i) & defect for i in range(1, count + 1)], defect.shape)


def f_budget(labels, min_area=5, top_k=4):
    masks = [labels == i for i in range(1, int(labels.max()) + 1)]
    masks = [m for m in masks if int(m.sum()) >= min_area][:top_k]
    return canonicalize(masks, labels.shape)


def extract_all(raw):
    from npfc import extract_npfc
    defect = raw == 2
    support = (raw == 1) | defect
    median = ndimage.median_filter(defect.astype(np.uint8), size=3, mode="constant", cval=0).astype(bool) & support
    opened = ndimage.binary_opening(defect, structure=ndimage.generate_binary_structure(2, 1), iterations=1)
    features = wbbs.encode_wafer(raw, {"sigma_d": 9.0, "noise_threshold": 0.32})
    tv_mask = np.zeros(defect.shape, bool)
    if len(features.points):
        tv_mask[tuple(features.points.astype(int).T)] = True
    npfc = extract_npfc(defect)
    outputs = {
        "cc4": connected(defect, defect, 1),
        "cc8": connected(defect, defect, 2),
        "median8": connected(median, defect, 2),
        "opening8": connected(opened, defect, 2),
        "tv8": connected(tv_mask, defect, 2),
        "npfc_pre": canonicalize(npfc["masks"], defect.shape),
    }
    for key, labels in outputs.items():
        if np.any((labels > 0) & ~defect):
            raise AssertionError(f"{key}: output contains synthetic/nondefective pixels")
    assert np.array_equal(outputs["tv8"] > 0, tv_mask)
    return outputs, {
        "median_added_support_pixels": int((median & ~defect).sum()),
        "npfc": npfc["diagnostics"],
    }
