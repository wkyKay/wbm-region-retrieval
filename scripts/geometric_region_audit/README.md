# T-Linkage / Multi-X 晶圆区域像素验证

本次沿用上一轮 `region_proposal_audit_20261111_v2` 的全部38类固定图像及原始sample_id。
没有重新挑图；37个缺陷组合来自冻结train，另1张来自既有Normal训练分配。
类别仅用于标题和核验，提取函数只接收raw图像和seed=2026，不输入类别或模式数量。

## 查看结果

- `index.html`：离线逐类浏览器，悬停读取原像素坐标和区域编号，点击单独突出一个区域。
- `images/native/`：算法自身筛选后的全部区域，没有额外限制区域数量。
- `images/f_budget/`：统一应用真实缺陷面积≥5、面积最大的4区域，与F预算一致。
- 两套各有8张五类一页汇总图，以及38张单类高清图。
- `preview_native.png`、`preview_f_budget.png`：预先固定的Scratch、Center+Scratch、Donut+Scratch、最后一组合四例。
- `synthetic_controls.png`：合成十字交叉线、直线穿圆，仅供算法检查，不是MixedWM38样例。

每行依次为原图、8连通、T-Linkage多直线、Multi-X直线+圆适配。
彩色表示区域编号，不是缺陷语义类别。区域按面积降序编号，跨方法同色不表示同一模式。
深灰是未分配的原始缺陷；浅灰为正常die；白色为原值0/3。所有图保持原52×52坐标。
同一像素最多属于一个区域，交点没有双重归属；没有补画整条直线、整个圆或被环包围的正常像素。

## 方法及必须保留的边界

### T-Linkage 多直线移植

参考Magri与Fusiello的CVPR2014论文和作者MATLAB demo，Python实现原文连续偏好、Tanimoto距离、
逐元素最小值更新和全局贪心聚合。统一5N个初始直线假设，tau=0.5像素、硬截断2.5像素。
离群筛选采用作者实际发布的簇大小跳变demo规则；该规则带有至少两个模型的倾向，
不是论文中完整的二项概率检验。详见 `TLINKAGE_NOTES.md`。

### Multi-X 直线+圆适配

参考Barath与Matas的ECCV2018论文及作者公开框架。公开仓库未提供可直接运行的线圆示例，
本次Python移植外循环，补充线、圆估计器，调用作者原C++ GCoptimization执行图割及标签代价。
采用公开代码的Mean-Shift，而非论文Median-Shift；使用统一固定带宽和均匀采样。
补入论文Algorithm2的100次最小样本重采模型验证。gamma=2像素，空间权重0.3，
总初始假设2N（两类均分）、hmax=10，邻域半径1.5像素；不以类别设置参数。
详见 `MULTIX_NOTES.md` 的逐项证据与适配说明。

这两列均应称为方法移植／晶圆适配结果，不能称为原作者在MixedWM38上的正式结果复现。
直线/圆是几何先验，无法直接表达所有不规则块状缺陷。宽环、Loc、Near-full等可能被拆成片段，
随机噪声也可能形成偶然的几何簇。保留这些失败是本次目视验证的一部分；没有按类别回退其他算法。
仅38张图不能证明总体分割性能或检索性能，像素保留率也不是准确率。

## 数据与复现

`selected_samples.npz`、`sample_manifest.csv`与上一轮逐字节一致。
`regions_native.npz`、`regions_f_budget.npz`各包含：

- `sample_ids`：原始NPZ行号，顺序与样例文件一致。
- `cc8`、`tlinkage`、`multix`：`[38,52,52]`整型区域编号，0表示未分配，1…K表示区域。

```python
import numpy as np
data = np.load('selected_samples.npz')
regions = np.load('regions_native.npz')
raw = data['raw_maps'][16]
region_mask = regions['multix'][16] == 1
pixel_coordinates = np.argwhere(region_mask)  # row, column
unassigned = (raw == 2) & (regions['multix'][16] == 0)
```

`statistics.csv`保存逐图区域数、面积和保留像素数。
`diagnostics.json`保存模型、方法参数、离群筛选和模型验证记录。
其中方法内部模型编号与展示编号的对应关系见 `original_to_display_region_id`。
`validation.json`记录像素支持、样本一致性、38例F等价性等检查。
`manifest.json`保存版本和源码签名，`source/`保存算法与可视化源码、必要作者依赖及出处。
原始NPZ、冻结协议、F源码和上一轮产物通过运行前后哈希保护。未训练模型或使用test。

从项目根目录运行（输出目录需为新的路径，或带有本脚本marker且签名完全一致的中间目录）：

```bash
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=/private/tmp/region-audit-mpl \
  ../compared_methods/.venv/bin/python scripts/geometric_region_audit/run.py \
  --output artifacts/geometric_region_audit_NEW
```

也可以分别以`--stage extract --method cc8|tlinkage|multix`计算，最后`--stage render`汇总。
每图结果有源码签名缓存；算法改变时必须使用新的输出目录，避免混合结果。

论文与作者项目：

- [T-Linkage论文](https://fusiello.github.io/papers/cvpr14.pdf)、[作者代码](https://fusiello.github.io/demo/jlk/index.html)
- [Multi-X论文](https://cmp.felk.cvut.cz/~matas/papers/barath-2018-multi_X-eccv.pdf)、[作者代码](https://github.com/danini/multi-x)
