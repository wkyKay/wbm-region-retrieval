# F 的六种区域划分对照

2026-11-13：已有模型的径向位置＋相对面积评分，以及 valid/test 的 exact/coverage
两套指标，使用[新的独立评价入口](../f_segmentation_geometry/README.md)。
本页原入口继续保留原 F 评分和训练签名；新入口读取旧模型，将增强结果写到新目录。

本入口独立改变 F 的区域划分，其他训练、输入和匹配规则沿用当前 F。
`all` **明确包含 train → valid → test**；每种方法从头训练自己的模型，
默认自动续训同签名运行。它不会复用原 F 的旧 checkpoint，也不会改写旧输出。

## 远程一条命令

在本仓库根目录、已安装 `requirements.txt` 的 PyTorch CUDA 环境中运行：

```bash
python -u experiments/proposed/f_segmentation/run.py --device cuda --prepare-workers 4
```

默认依次执行 `cc8 cc4 median8 opening8 tv8 npfc_pre`，每组 **100 epoch**，
训练 seed=2026、batch size=256、DataLoader workers=4。每组完成训练、valid
和完整 test 后继续下一组。`--prepare-workers 4` 是 CPU 分割进程数，GPU 训练不并行。
省略它时串行分割；WBBS 全点投票和 NPFC-pre 可能比连通域耗时更多。

NPZ 默认位于仓库外 `../data/wm38k/Wafer_Map_Datasets.npz`，不随 Git 上传。
远程位置不同时直接指定原始文件，哈希仍须与冻结协议一致：

```bash
python -u experiments/proposed/f_segmentation/run.py --device cuda --prepare-workers 4 --data-file /absolute/path/Wafer_Map_Datasets.npz
```

无需同级 `compared_methods` 仓库。`tensor_voting.py` 是已有 WBBS 投票/阈值函数的
自包含副本，文件头记录来源 SHA-256；NPFC-pre 也已包含在当前目录。
不下载预训练权重；项目现有 NumPy/SciPy/PyTorch 依赖即可运行。

其他常用方式：

```bash
# 无写入预览配置
python experiments/proposed/f_segmentation/run.py --dry-run

# 仅运行五种新增方案，跳过重新训练 cc8 对照
python -u experiments/proposed/f_segmentation/run.py --device cuda --prepare-workers 4 --methods cc4 median8 opening8 tv8 npfc_pre

# 单方法；独立的新输出根和批大小须在开始训练前确定
python -u experiments/proposed/f_segmentation/run.py --device cuda --methods opening8 --batch-size 64 --output-root artifacts/f_segmentation_bs64_v1

# 仅准备 train/valid，或分开执行；后续命令须沿用相同配置
python -u experiments/proposed/f_segmentation/run.py --stage prepare --prepare-workers 4
python -u experiments/proposed/f_segmentation/run.py --stage train --device cuda
python -u experiments/proposed/f_segmentation/run.py --stage valid --device cuda
python -u experiments/proposed/f_segmentation/run.py --stage test --device cuda --prepare-workers 4
```

`--methods` 可以分批补跑。`--stage train` 仍包含原 F 在训练期间的 valid NCE 与
检索监测，但不执行最终 valid/test 报告。`--stage prepare` 不启动模型训练，
也不构建 test 缓存。test 的分割和特征计算仅在训练完成后执行。

## 固定的科学设置

| ID | 区域划分 |
|---|---|
| `cc8` | 原始缺陷图 8 连通，当前 F 的对照 |
| `cc4` | 原始缺陷图 4 连通 |
| `median8` | 3×3 二值中值滤波、边界补0、限制在有效晶圆内；8 连通分组后与原始缺陷点取交集 |
| `opening8` | 原图网格上的 3×3 十字核、一次开运算，再 8 连通；无前景保留率回退 |
| `tv8` | WBBS 闭式张量投票去噪，再 8 连通；sigma_d=9、threshold=0.32、block_size=256 |
| `npfc_pre` | 初始 8 连通，删除1/2像素区域，合并最小原始像素距离≤sqrt(8)的剩余簇；取传递闭包 |

六组均在分组完成后按**原始缺陷像素面积≥5、面积最大的4个区域**筛选，等面积按
首个行优先像素排序。输出每个区域的明确像素，不添加缺陷、不按真实类别选择算法。
中值滤波新增的支撑像素可参与连通分组，但不会进入最终 mask。
NPFC 合并后的一个 mask 可以包含互不接触的片段，间隙不会被填充。

NPFC-pre **不是完整 NPFC**，不含 SD 选簇或后续核心簇分配；证据与限制见
[NPFC_NOTES.md](NPFC_NOTES.md)。WBBS 的 sigma_d=9 是现有实现约定，
不是已核实的作者默认值；`tv8` 也不是 2022 年 tensor voting + marching 方法。
本实验没有加入 T-Linkage、Multi-X、Top-6/8、径向项或其他推理改进。

所有方案只改变目标区域及其派生窗口/面积：F 仍以目标质心截取52×52、越界补0、
resize96。三通道仍为**原始全部缺陷、目标区域、原始有效晶圆**，因此去噪不会
删除上下文通道中的原始缺陷。保持 ResNet stride1、512维特征、128维投影、
仅直角旋转、5,000负样本、SGD/原余弦计划，以及原始单向 sqrt(area) 加权贪心匹配。

过滤后无区域的图保留为零 token，仍在固定 query/gallery 内；与空区域图匹配得分为0，
沿用 exact_v1 的确定性同分排序。它们不贡献区域 NCE，不会静默回退为 CC8。
如果整个 split 都没有区域则报错，要求检查方法/数据，不能生成虚构 token。

## 数据与选模

使用本仓库已冻结、去重后的协议：train/valid/test=25,193/3,599/7,199。
valid 是500个固定 query × 全部3,598个候选；test 是7,199个 query × 全部7,198个候选。
每次均排除自身。划分、原始数据和 query/gallery 都校验哈希，不重采样。
Normal 不进入缺陷训练和检索。sample_id 始终是原始 NPZ 行号。

默认预先固定 `--checkpoint best_nce`：最低 valid NCE 的模型用于最终 valid/test。
训练仍按原 F 在第1轮及每5轮监测 valid 检索，并保存辅助 `best_retrieval.pt`。
如需使用 valid 标签选模对照，可在新输出根开始时指定 `--checkpoint best_retrieval`。
test 不选择 checkpoint，不参与训练、过滤阈值拟合或超参数搜索。

## 产物和续跑

默认根目录 `artifacts/f_segmentation_v1/`：

```text
segmentation_comparison.json            # 所有方法共有的配置/数据/源码签名
run.log                                 # 追加式进度日志
comparison_seed_2026_best_nce.csv         # 各方法 valid/test 汇总（完成一组即更新）
cc8/                                    # 其他方法各有同级独立目录
  suite_manifest.json                   # 加入方法/参数后的独立签名
  valid_pool.json                       # 固定 query/gallery，非展开的配对表
  cache/
    train_region_labels.npy             # [晶圆序号,H,W]；0未分配、1..4区域编号
    train_regions.npz                   # map_ids、token offsets、area
    train_F.npy                         # [token,3,96,96]，磁盘映射缓存
    train_segmentation_audit.csv         # 每图区域数、原始/保留像素、空图标记
    train_F_window_audit.csv             # 窗口截断情况
    train_audit.json                     # 汇总诊断；valid/test有相同命名
    train_prepared.json                  # 完成标记及缓存哈希
  seed_2026/F/
    last.pt / best_nce.pt / best_retrieval.pt
    history.json / training_summary.json / run_manifest.json
    valid/best_nce/metrics.json
    test/best_nce/metrics.json
    test/best_nce/per_query.csv
    test/best_nce/per_class.csv
    test/best_nce/rankings.csv            # 每个query仅保存Top-500；评分使用完整候选库
    test/best_nce/completed.json          # checkpoint及全部报告哈希
```

读取第 `i` 张图的明确区域像素：`region_labels[i] == region_id`，对应原图 ID 从
同 split 的 `regions.npz["map_ids"][i]` 获取；不能把 `i` 当作原始样本 ID。

网络/进程中断后，**重复完全相同的命令**即可：训练从 `last.pt` 恢复，已完成的
最终报告经哈希校验后跳过。评估先写独立临时目录，全部成功后原子改名；中断的
`.best_nce_attempt_*` 等目录保留，重跑另开目录，不将部分结果误当成完成。
源码、方法、训练 seed、批大小、选模规则或科学配置变化会拒绝复用，需新输出根。
设备、加载进程数和预处理进程数可改变。不同机器/库版本的训练数值复现仍受环境影响；
原 F 的初始化哈希检查会拒绝明显不兼容的续训环境。

缓存仅保存 F 所需的三通道 patch 与分割标签图，不生成 D/E patch 或 Zernike。
预处理分批写入磁盘，最多同时排队2×分割进程数张图。各方法各有缓存和三个模型文件，
请为正式六组运行预留数十GB的磁盘空间。中断后重跑会读取文件校验哈希，开头可能有较长 I/O。

## 本地验证（2026-11-11）

- 六种方法与此前 train 中38类（含Normal）各一张的审计结果逐像素一致：228/228。
- CC8 随机构造图与原始 `extract_regions` 的区域像素和顺序一致。
- 真实 F 三通道 ResNet 在六种方法的实际样例上完成 CPU 前向、反向与 SGD 更新。
- 新增测试覆盖六组端到端训练/valid/test、重复运行、数据/缓存签名、空区域、
  中值新增支撑、NPFC传递合并、WBBS逐点公式、多进程一致性与中断报告保留。
  端到端测试使用合成冻结协议及轻量编码器；它检验流程，不代表正式模型精度。
- 当前工作区全套143项测试完成验证：沙箱内141项通过，另2项因本机沙箱禁止
  PyTorch共享内存套接字而失败；允许该本机测试权限后，2项独立复跑均通过。
- 未启动正式100 epoch训练或真实7,199查询的完整test。

```bash
python -m unittest discover -s tests -p 'test_f_segmentation.py' -v
python -m unittest discover -s tests -v
```
