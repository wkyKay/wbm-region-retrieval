# 六种检索方法与主、副评价指标总览

整理日期：2026-11-10。范围：`compared_methods/` 的五种对比方法，以及 `wbm-region-retrieval/` 的 F 主方法。

本文将“两组方法”展开为六种具体算法，将“两组指标”统一表述为：**主评价 `coverage_v1`（方向性类别覆盖）**、**副评价 `exact_v1`（完整标签组合匹配）**。算法名称中的“主方法 F”与指标中的“主评价”是两个不同概念；所有六种方法均按同一套主、副指标评价。

依据当前代码、配置、已保存报告及引用聊天整理；本文只汇总，不修改算法、冻结协议、模型或历史结果，也不启动训练/test。早期文档中的“尚未迁移/尚未 test”描述属于历史阶段，当前状态以迁移记录和正式产物为准。

## 1. 方法总表

| 分组 | 方法 | 学习方式 | 表示与检索流程 | 训练预算与选模 |
| --- | --- | --- | --- | --- |
| 本文主方法 | Proposed F | 自监督区域 NCE；训练不用缺陷类别标签 | 三通道固定窗口 → ResNet-18 区域特征 → 一对一区域匹配 → query 区域面积平方根加权 | 100 epoch，seed=2026；正式 test 固定 `best_nce`（最低 valid NCE），本次为 epoch 100 |
| 对比方法 | Wu 2015 / WMSR | 无训练手工方法 | 116 维 Radon/几何/统计/Hough 特征，欧氏距离粗筛 500，再以整图及显著区域 NCC 重排 | 不训练论文的 SVM 分类分支 |
| 对比方法 | Yu 2019 | 监督 CNN 分类 | FC2 1024 维 → train 拟合 PCA256 → train min-max → 256 维十值编码 → categorical Hamming | 检测器 10 epoch、分类器 80 epoch；各自最低 valid CE；主比较为 `global`，不使用检测/预测类别门控 |
| 对比方法 | WBBS 2023 | 无训练点集方法 | 张量投票去噪 → stick/ball/径向描述 → 互为最近邻点对 → 点数差惩罚 | 无 checkpoint；参数由方法配置固定 |
| 对比方法 | Nakazawa CNN 2018 | 监督 CNN 分类 | 37 类分类 CNN → 256 维 sigmoid → 阈值 ≥0.5 二值化 → 256 位 Hamming | 10 epoch；固定最后一轮，不按 valid loss/检索指标择优 |
| 对比方法 | WMHD 2020 | 无训练点集方法 | ESRN 去噪 → mountain 权重 → 方向性加权 modified Hausdorff 距离，默认 W2-O1 | 无 checkpoint；query → candidate，不对称化 |

### 1.1 F 的当前实现

- 区域提议固定 `min_area=5`、最多保留 `top_k=4` 个区域。
- 以目标区域质心为中心截取原网格 `52×52` 窗口，越界补零，再缩放至 `96×96`。窗口可能截断目标区域。
- 三通道依次为窗口内全部缺陷、目标区域 mask、有效晶圆 mask。
- ResNet-18 首层 stride=1；检索表示 512 维；NCE 投影为 128 维。检索表示与训练投影不能混称。
- 同一区域原始/旋转增强视图形成自监督正对。正式配置无尺度/平移增强、无 opening 增强；5,000 个负样本，temperature=0.07。
- SGD，初始学习率 0.01，momentum=0.9，weight decay=0.001，`wapirl_cosine` 调度；训练 100 epoch，不启用早停。
- 区域余弦相似度按分数贪心一对一匹配，query 区域权重为面积平方根；分母为 query 所有区域权重之和，未匹配 query 区域贡献零。当前位置/尺度附加权重均为 0。
- `best_nce` 用于正式 test；每 5 epoch 监测的 valid `mAP@10` 及 `best_retrieval` 不替代正式 checkpoint。

### 1.2 对比方法的复现边界

- **Wu**：当前仅复现检索分支。NCC 分数为 `0.5×整图 NCC + 0.5×显著区域 NCC`；未进入粗筛的候选不补排，不能称为全库 NCC 精排。
- **Yu**：37 种完整缺陷组合是 37 个单标签分类类别，不是八标签 BCE。PCA 和缩放只拟合 train。`global` 是主比较；`predicted_class` 是独立辅助变体，不与主结果混列。9×9 中值滤波、输入颜色等适配以方法 README 为准。
- **WBBS**：分数为 `B/min(N,M) × exp(-alpha×|N-M|/l)`，B 为互为最近邻点对数，N/M 为去噪后点数。保留对称原公式，不加入方向性变体。配置 `alpha=5`、`mu=1`、去噪阈值 0.32；`sigma_d=9`、`l=2061` 等包含公开记录的实现/数据适配约定。
- **Nakazawa**：原论文报告 10 epoch 的训练曲线，并未规定只能训练十轮或最佳 checkpoint 规则。固定训练十轮并使用最后一轮是当前复现约定。原始 22 类适配为 37 类；MixedWM38 缺少每 die 缺陷事件计数，使用缺陷占用图，不能称为恢复原始密度输入。优化器等未披露细节是实现补足。
- **WMHD**：保留方向性、九项 W2 权重之和及 O1 离群惩罚。固定 Cartesian 方向，没有恢复未披露的旋转对齐；ESRN、mountain 和容差参数包含实现约定。边界 `-inf` 分数作为合法最差分数保留。

六种方法的监督信号、训练轮次及每轮计算量不同。可比较固定数据和协议下的实际检索性能，但不能将其描述为等训练预算实验；也不能仅凭 epoch 数判断训练是否充分。对比方法为论文方法的 MixedWM38 重实现，不等于作者官方代码或原论文数值复现。

## 2. 统一数据与检索协议

原始数据为 `data/wm38k/Wafer_Map_Datasets.npz`。先对完整 0/1/2/3 原图精确去重，SHA-256 筛选并逐像素确认，每组保留最小原始行 ID；不按旋转/翻转/近似外观合并。38,015 条原始记录去掉 1,024 条冗余记录，剩余 36,991 张：35,991 张缺陷图与 1,000 张 Normal。

缺陷按 37 种完整 multi-hot 组合分层，划分 seed=2026。`sample_id` 始终是原 NPZ 行号。

| 集合 | 缺陷图数 | query 数 | gallery 数 | 每 query 候选数 |
| --- | ---: | ---: | ---: | ---: |
| train | 25,193 | — | — | — |
| valid | 3,599 | 固定分层 500 | 3,599 | 3,598 |
| test | 7,199 | 全部 7,199 | 7,199 | 7,198 |

- 中央协议：`compared_methods/protocol/deduplicated/` 和 `protocol/retrieval_v1/`；F 使用项目内逐字节一致的冻结镜像，不能使用旧未去重 checkpoint。
- 每个 query 排除自身，其他 query 可以作为候选。Normal 不进入缺陷检索库；Yu 检测训练另使用 seed=2026 的 Normal 700/100/200 划分。
- 八维标签顺序：center、donut、edge-loc、edge-ring、loc、near-full、scratch、random。
- 共同 K：**1、5、10、20、50、100、500**。默认仅流式保存 Top-500 排名；完整候选库分母不因保存截断、粗筛或门控缩小。
- 参数与 checkpoint 用 train/valid 确定，不按 test 结果选择方法变体、checkpoint 或有利 K。
- 排序不读取真实标签。同分使用 SHA-256：ASCII `exact_v1:tie:{seed}:{query_id}:{candidate_id}`，前 8 字节按大端 uint64 升序，碰撞时原始 ID 升序；tie seed=2026。Wu 在剩余同分前保留欧氏距离次排序键。
- coverage 与 exact 使用相同 query/gallery、相同排名、相同 K，仅评价相关性不同。历史 2,000×1,000 test、旧 containment 指标和 smoke 不混入正式结果。

## 3. 主指标组：coverage_v1

主任务是判断候选是否覆盖查询中的缺陷类别，允许候选有额外类别。令非空标签集合为 Lq、Lc：

```text
g(q,c) = |Lq ∩ Lc| / |Lq|
full(q,c) = 1[Lq ⊆ Lc] = 1[g(q,c)=1]
```

例如 query `{Loc,Scratch}`：candidate `{Loc,Scratch,Donut}` 得 1；`{Loc}` 得 0.5；`{Donut}` 得 0。反向覆盖通常不同。这里不采用会惩罚额外类别的 Jaccard，也不采用指数增益 `2^g−1`。

设第 r 名覆盖率为 g_r、完整覆盖标志为 b_r，`H_full(K)=Σ(r≤K)b_r`，`R_full` 为完整候选池内完整覆盖候选数。

| 指标 | 逐 query 定义 | 作用 / 汇总字段 |
| --- | --- | --- |
| Coverage-nDCG@K | `DCG/IDCG`，`DCG=Σ(r≤K) g_r/log2(r+1)`；IDCG 对完整候选池的覆盖率降序后按同式计算 | 主要分级排序指标，兼顾覆盖程度及位置；`Coverage-nDCG@K` |
| MeanCoverage@K | `Σ(r≤K)g_r/K` | 前 K 项平均覆盖程度；`MeanCoverage@K` |
| FullCoverage Hit@K | `1[H_full(K)>0]` | 是否至少找回一个完整覆盖候选；`FullCoverageHit@K` |
| FullCoverage Precision@K | `H_full(K)/K` | 完整覆盖结果比例；`FullCoveragePrecision@K` |
| FullCoverage Recall@K | `H_full(K)/R_full` | 找回全部完整覆盖候选的比例；`FullCoverageRecall@K` |
| FullCoverage AP@K / mAP@K | `Σ(r≤K)(Σ(j≤r)b_j/r)×b_r/R_full`；逐 query AP 求均值得 mAP | 兼顾完整覆盖命中数量与排名；逐 query `FullCoverageAP@K`，汇总 `FullCoverage_mAP@K` |

返回不足 K 时，缺位覆盖率和命中均按零计算，Precision/MeanCoverage 分母仍为 K。IDCG=0 时 nDCG 未定义并排除均值；R_full=0 时 FullCoverage Recall/AP 排除均值，Hit/Precision 仍计入；报告有效样本数。

当前论文汇总重点展示 Coverage-nDCG@10/@20、MeanCoverage@10/@20、FullCoverage mAP@10/@20、FullCoverage Recall@10/@20/@500。完整实现包含上表全部六项及共同七个 K。主表最终选点应遵守预先确定的实验方案，不能依据 test 优势重新选 K。

`coverage≥τ`（例如 τ=0.5）和“任意重合 coverage>0”尚未作为额外正式二值评价启用；`0<coverage<1` 不单独定义成正样本，以免反而排除完整覆盖。

## 4. 副指标组：exact_v1

完整八维 multi-hot 标签相同才相关：`rel(q,c)=1[yq=yc]`。例如 query `{Loc,Scratch}` 与 `{Loc,Scratch,Donut}` 在 exact 下不相关，尽管其 full coverage 为 1。

令 r_i 为第 i 名的 exact 相关标志，`H(K)=Σ(i≤K)r_i`，R 为完整固定候选池中 exact 相关样本总数：

| 指标 | 逐 query 定义 | 含义 |
| --- | --- | --- |
| Hit@K | `1[H(K)>0]` | 至少一个完全匹配结果 |
| Precision@K | `H(K)/K` | 前 K 项完全匹配比例 |
| Recall@K | `H(K)/R` | 找回全部完全匹配候选的比例 |
| AP@K / mAP@K | `AP=Σ(i≤K)(Σ(j≤i)r_j/i)×r_i/R`；mAP 为 AP 的 query 平均 | 完全匹配的截断平均精度 |

**AP 的分母是 R，不是 min(R,K)**；FullCoverage AP 同理使用 R_full。若 R 很大，即使前十全对，AP@10 也可以远小于 1。mAP@500 仍是截断 mAP，不是全库完整 mAP。Hit 不能称为 Recall。

返回不足 K 时补零，Precision 分母仍为 K。R=0 时 Recall/AP 未定义并排除均值，Hit/Precision 仍参与；保留零相关 query 及有效计数。

## 5. 聚合方式与分组

两组指标都应区分以下统计，不能混为一列：

| 统计口径 | 含义 | 报告字段 |
| --- | --- | --- |
| query 平均 | 每个有效 query 等权；主结果表采用此口径 | `metrics` |
| 完整组合类别宏平均 | 每个组合内先求 query 均值，再对实际出现且指标有效的组合类别等权平均 | `class_macro_metrics` |
| 单标签 query | query 标签数为 1 的子集均值 | `groups.single_label` |
| 混合标签 query | query 标签数大于 1 的子集均值 | `groups.mixed_label` |
| 按标签数分组 | 分别汇总各 query 标签数量 | `groups.by_label_count` |

类别宏平均针对 37 种完整组合，不是八个基础标签。单标签 query 的 coverage 只有 0/1，混合标签 query 才能检验部分覆盖等级。训练 NCE、分类 CE/accuracy、运行耗时等是训练或效率诊断，不是另一套检索相关性指标。

## 6. 已有正式 test 结果

下列结果共用 7,199 个 query、每个 7,198 个候选。数值为 [0,1] 小数，保留六位；不表示多训练 seed 的平均或方差。

### 6.1 主评价：coverage_v1

本表从项目已保存的 `doc/EXACT_METRICS_MIGRATION.md` 正式 coverage test 表逐字提取。该记录说明两组 coverage 均从已完成的 Top-500 排名后处理得到，不重新训练、不改变 exact 产物。本次未连接远程重新审计原始 coverage JSON；本地未找到六方法完整 coverage JSON，因此不补造其余 K、FullCoverage Hit/Precision 或类别宏平均数值。

| 方法 | Coverage-nDCG@10 / @20 | MeanCoverage@10 / @20 | FullCoverage mAP@10 / @20 | FullCoverage Recall@10 / @20 | FullCoverage Recall@500 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wu | 0.708569 / 0.694368 | 0.701086 / 0.684555 | 0.004470 / 0.007210 | 0.006329 / 0.011491 | 0.147566 |
| Yu | 0.736098 / 0.686443 | 0.720576 / 0.657216 | 0.007420 / 0.011173 | 0.008972 / 0.014336 | 0.049693 |
| WBBS | 0.878320 / 0.867206 | 0.872929 / 0.859790 | 0.010533 / 0.019018 | 0.012464 / 0.023673 | 0.286166 |
| Nakazawa CNN | 0.959434 / 0.957507 | 0.958760 / 0.956364 | 0.016765 / 0.033020 | 0.017355 / 0.034456 | 0.434949 |
| WMHD | 0.099097 / 0.123000 | 0.106394 / 0.136892 | 0.001922 / 0.003717 | 0.002423 / 0.005311 | 0.195214 |
| Proposed F (`best_nce`) | 0.747610 / 0.733125 | 0.739647 / 0.722990 | 0.006533 / 0.011215 | 0.008611 / 0.016033 | 0.192870 |

### 6.2 副评价：exact_v1，query 平均（全部共同 K）

以下直接读取六份本地正式 test JSON 生成；包含所有七个 K，不仅挑选有利切点。

#### K=1

| 方法 | Hit@1 | Precision@1 | Recall@1 | mAP@1 |
| --- | ---: | ---: | ---: | ---: |
| F | 0.395194 | 0.395194 | 0.002121 | 0.002121 |
| Wu | 0.324212 | 0.324212 | 0.001760 | 0.001760 |
| Yu | 0.408529 | 0.408529 | 0.002182 | 0.002182 |
| WBBS | 0.671482 | 0.671482 | 0.003518 | 0.003518 |
| Nakazawa CNN | 0.865259 | 0.865259 | 0.004488 | 0.004488 |
| WMHD | 0.011113 | 0.011113 | 0.000180 | 0.000180 |

#### K=5

| 方法 | Hit@5 | Precision@5 | Recall@5 | mAP@5 |
| --- | ---: | ---: | ---: | ---: |
| F | 0.789554 | 0.352327 | 0.009511 | 0.007148 |
| Wu | 0.688985 | 0.283039 | 0.007742 | 0.005723 |
| Yu | 0.775941 | 0.382887 | 0.010262 | 0.007945 |
| WBBS | 0.935269 | 0.615919 | 0.016182 | 0.013994 |
| Nakazawa CNN | 0.948604 | 0.859869 | 0.022287 | 0.021544 |
| WMHD | 0.040283 | 0.013280 | 0.000953 | 0.000829 |

#### K=10

| 方法 | Hit@10 | Precision@10 | Recall@10 | mAP@10 |
| --- | ---: | ---: | ---: | ---: |
| F | 0.901931 | 0.326031 | 0.017622 | 0.011679 |
| Wu | 0.829976 | 0.263550 | 0.014357 | 0.009145 |
| Yu | 0.859981 | 0.351729 | 0.018884 | 0.013240 |
| WBBS | 0.969440 | 0.581289 | 0.030586 | 0.024799 |
| Nakazawa CNN | 0.963189 | 0.855091 | 0.044338 | 0.042395 |
| WMHD | 0.074038 | 0.014530 | 0.001970 | 0.001609 |

#### K=20

| 方法 | Hit@20 | Precision@20 | Recall@20 | mAP@20 |
| --- | ---: | ---: | ---: | ---: |
| F | 0.966523 | 0.296340 | 0.031970 | 0.018708 |
| Wu | 0.918600 | 0.242777 | 0.026040 | 0.014300 |
| Yu | 0.910543 | 0.303528 | 0.032163 | 0.020388 |
| WBBS | 0.986526 | 0.540624 | 0.056921 | 0.043141 |
| Nakazawa CNN | 0.975413 | 0.846666 | 0.087821 | 0.083146 |
| WMHD | 0.147104 | 0.018704 | 0.004234 | 0.003001 |

#### K=50

| 方法 | Hit@50 | Precision@50 | Recall@50 | mAP@50 |
| --- | ---: | ---: | ---: | ---: |
| F | 0.994583 | 0.251991 | 0.066779 | 0.032479 |
| Wu | 0.972079 | 0.211140 | 0.054913 | 0.024652 |
| Yu | 0.947493 | 0.210618 | 0.054800 | 0.030520 |
| WBBS | 0.996249 | 0.474446 | 0.123128 | 0.084240 |
| Nakazawa CNN | 0.990276 | 0.828490 | 0.212581 | 0.198445 |
| WMHD | 0.297680 | 0.033219 | 0.012010 | 0.005218 |

#### K=100

| 方法 | Hit@100 | Precision@100 | Recall@100 | mAP@100 |
| --- | ---: | ---: | ---: | ---: |
| F | 0.999583 | 0.214606 | 0.111798 | 0.046048 |
| Wu | 0.986943 | 0.187269 | 0.096196 | 0.037027 |
| Yu | 0.960689 | 0.144999 | 0.074879 | 0.038101 |
| WBBS | 0.998889 | 0.412814 | 0.212099 | 0.132996 |
| Nakazawa CNN | 0.997222 | 0.797733 | 0.406367 | 0.373380 |
| WMHD | 0.455758 | 0.046951 | 0.028554 | 0.010472 |

#### K=500

| 方法 | Hit@500 | Precision@500 | Recall@500 | mAP@500 |
| --- | ---: | ---: | ---: | ---: |
| F | 1.000000 | 0.128715 | 0.328472 | 0.086840 |
| Wu | 0.998889 | 0.115329 | 0.292925 | 0.081793 |
| Yu | 0.987498 | 0.059592 | 0.152133 | 0.054524 |
| WBBS | 0.999861 | 0.230123 | 0.583941 | 0.271098 |
| Nakazawa CNN | 1.000000 | 0.360553 | 0.912929 | 0.726898 |
| WMHD | 0.883595 | 0.055752 | 0.146859 | 0.029758 |

### 6.3 副评价：exact_v1，完整组合类别宏平均

为避免与 query 平均混淆，单独列出全部 K 的四项宏平均。

| 方法 | K | Hit | Precision | Recall | mAP |
| --- | ---: | ---: | ---: | ---: | ---: |
| F | 1 | 0.409861 | 0.409861 | 0.002855 | 0.002855 |
| F | 5 | 0.795270 | 0.367468 | 0.013094 | 0.010689 |
| F | 10 | 0.904595 | 0.340440 | 0.024337 | 0.018240 |
| F | 20 | 0.967432 | 0.308838 | 0.043648 | 0.029915 |
| F | 50 | 0.994730 | 0.258187 | 0.083759 | 0.048362 |
| F | 100 | 0.999595 | 0.216235 | 0.128631 | 0.062042 |
| F | 500 | 1.000000 | 0.127136 | 0.342127 | 0.102268 |
| Wu | 1 | 0.339897 | 0.339897 | 0.002546 | 0.002546 |
| Wu | 5 | 0.695978 | 0.298971 | 0.011579 | 0.009606 |
| Wu | 10 | 0.833641 | 0.277239 | 0.021042 | 0.015934 |
| Wu | 20 | 0.920191 | 0.251547 | 0.035508 | 0.023864 |
| Wu | 50 | 0.972409 | 0.212406 | 0.063993 | 0.034158 |
| Wu | 100 | 0.986963 | 0.186122 | 0.104414 | 0.046319 |
| Wu | 500 | 0.998871 | 0.113393 | 0.297010 | 0.090210 |
| Yu | 1 | 0.421656 | 0.421656 | 0.002883 | 0.002883 |
| Yu | 5 | 0.781693 | 0.396586 | 0.013783 | 0.011398 |
| Yu | 10 | 0.863569 | 0.364867 | 0.025531 | 0.019832 |
| Yu | 20 | 0.912806 | 0.310855 | 0.040778 | 0.028923 |
| Yu | 50 | 0.948776 | 0.211974 | 0.063549 | 0.039139 |
| Yu | 100 | 0.961638 | 0.144854 | 0.083638 | 0.046639 |
| Yu | 500 | 0.987790 | 0.058880 | 0.160440 | 0.062722 |
| WBBS | 1 | 0.680238 | 0.680238 | 0.004271 | 0.004271 |
| WBBS | 5 | 0.936884 | 0.625723 | 0.019923 | 0.017787 |
| WBBS | 10 | 0.970175 | 0.591320 | 0.037884 | 0.032196 |
| WBBS | 20 | 0.986820 | 0.550220 | 0.070619 | 0.056899 |
| WBBS | 50 | 0.996304 | 0.476308 | 0.141313 | 0.102100 |
| WBBS | 100 | 0.998871 | 0.410379 | 0.229624 | 0.150501 |
| WBBS | 500 | 0.999841 | 0.226046 | 0.593149 | 0.285537 |
| Nakazawa CNN | 1 | 0.868003 | 0.868003 | 0.005186 | 0.005186 |
| Nakazawa CNN | 5 | 0.950000 | 0.862136 | 0.025669 | 0.024852 |
| Nakazawa CNN | 10 | 0.964189 | 0.857560 | 0.051136 | 0.049033 |
| Nakazawa CNN | 20 | 0.976081 | 0.849285 | 0.101383 | 0.096276 |
| Nakazawa CNN | 50 | 0.990541 | 0.822615 | 0.230602 | 0.215426 |
| Nakazawa CNN | 100 | 0.997297 | 0.786453 | 0.420589 | 0.386874 |
| Nakazawa CNN | 500 | 1.000000 | 0.353422 | 0.915290 | 0.732449 |
| WMHD | 1 | 0.034110 | 0.034110 | 0.001002 | 0.001002 |
| WMHD | 5 | 0.062512 | 0.036075 | 0.005055 | 0.004935 |
| WMHD | 10 | 0.095636 | 0.037315 | 0.010174 | 0.009820 |
| WMHD | 20 | 0.169722 | 0.040211 | 0.019617 | 0.018338 |
| WMHD | 50 | 0.316757 | 0.046068 | 0.033274 | 0.024845 |
| WMHD | 100 | 0.470541 | 0.055031 | 0.052546 | 0.031455 |
| WMHD | 500 | 0.886757 | 0.056809 | 0.169755 | 0.051714 |

### 6.4 结果阅读边界

- 在上面的 coverage 主表展示项中，Nakazawa CNN 均最高。
- exact query 平均下，Nakazawa 在全部共同 K 的 Precision、Recall、mAP 上最高；Hit 的最优方法随 K 改变。
- F 在 K=50、100、500 的 exact 四项上超过 Yu，在全部共同 K 的四项上超过 Wu、WMHD；这不等于 F 总体优于所有对比方法。
- exact 与 FullCoverage 的正样本数量、Recall/AP 分母不同，两套 mAP 不能横向解释为同一任务的提升或下降。
- 本文记录当前实现与本次固定运行结果，不对尚缺少完整本地报告的 coverage 分组或其他 K 作结论。

## 7. 运行入口、产物与追溯来源

### 7.1 入口

五种对比方法统一入口为 `compared_methods/scripts/evaluate.py`，显式传 `--split valid` 或 `--split test`；Yu/Nakazawa 先完成训练，再用 `--source-run` 指向只读训练产物。统一评价不隐式训练，输出目录必须新建。当前统一评价会写出 exact 与 coverage 两套报告。

F 从 `wbm-region-retrieval/` 执行以下流程，`<RUN>` 为相同运行根目录；这里只记录命令，本次未执行：

```bash
python experiments/proposed/f/run.py --stage train --output-root <RUN>
python experiments/proposed/f/run.py --stage evaluate --output-root <RUN> --skip-handcrafted
python experiments/proposed/f/run.py --stage test --output-root <RUN> --skip-handcrafted --test-checkpoint best_nce
```

F 当前正式训练/test 路径原生输出 exact；本文所列 F coverage 是已保存排名的独立后处理产物，不把上述命令描述成已自动生成 coverage。`--skip-handcrafted` 只跳过附加手工基线；`all` 不自动进入 test。

### 7.2 产物对应

| 内容 | exact | coverage |
| --- | --- | --- |
| 汇总 | `metrics.json` | `coverage_metrics.json` |
| 逐 query | `per_query.csv`（AP 字段） | `coverage_per_query.csv`（FullCoverageAP 字段） |
| 逐完整组合类别 | `per_class.csv` | `coverage_per_class.csv` |
| 排名 | 同一来源的 `rankings.csv`，最多 Top max(K) | 复用同一排名，不重新按标签排序 |

对比方法完成状态检查 `completed.json` 及签名；失败目录或局部 metrics 不等于完成。F 正式运行另有 `selection_before_test.json`、`verification.json` 等审计证据。旧 checkpoint、缓存、签名不匹配的结果不得直接复用。

### 7.3 本地可核对来源（路径均相对于外层 match-test）

- `compared_methods/docs/EXACT_EVALUATION.md`：两类评价、排序、聚合和边界规则。
- `compared_methods/shared/exact_metrics.py`、`shared/coverage_metrics.py`：实际公式和字段。
- `compared_methods/methods/{wu,yu,wbbs,nakazawa,wmhd}/README.md`：方法细节、论文依据、复现限制与命令。
- `compared_methods/protocol/README.md`：冻结数据与检索协议。
- `wbm-region-retrieval/experiments/proposed/f/config.json`、`experiments/_shared/base.json`：F 合并配置。
- `wbm-region-retrieval/experiments/proposed/_shared/ef_suite.py`：F 输入、特征和评价实现。
- `wbm-region-retrieval/doc/DATA_PROTOCOL_MIGRATION.md`、`doc/EXACT_METRICS_MIGRATION.md`：迁移、正式训练/test 及 coverage 表记录。
- `wbm-region-retrieval/artifacts/metric_migration_validation_20261109/formal_f/test_best_nce_metrics.json`：F 完整 exact 报告。
- `wbm-region-retrieval/artifacts/metric_migration_validation_20261109/compared_methods/{wu,yu,wbbs,nakazawa,wmhd}_test_metrics.json`：五方法完整 exact 报告。
- 同审计目录 `compared_test_summary.md`、`formal_f/verification.json`：既有比较与 F 复核证据。

### 7.4 历史正式运行位置（供定位，不表示本次重新检查远程）

- F exact：远程 `/data/wangkaiyan2024/wbm-region-retrieval/artifacts/formal_f_exact_20261109_132219/`，指标在 `run/seed_2026/F/test/best_nce/metrics.json`。
- 五方法 exact：远程 `/data/wangkaiyan2024/wbm-matching/artifacts/formal_exact_20261109_120738/evaluation/`；Wu/Yu/Nakazawa 为对应 `*_test/test/`，WBBS/WMHD 使用 `*_test_r2/test/`。
- F coverage：项目迁移记录中的 `artifacts/formal_f_coverage_v1_test_20261109_194139/F_best_nce/`。
- 五方法 coverage：项目迁移记录中的 `artifacts/coverage_v1_test_20261109_191909/`；具体排名来源和 SHA-256 在各 `evaluation_manifest.json` 中。

引用聊天：`01a1be73-ed9a-7352-8076-8aa16fbdc7e6`（设计五种方法统一 test 指标）。聊天用于恢复上下文；本文的方法定义、公式与数值优先依据当前文件和保存产物。

## 8. 新增独立分析：跨组合 coverage（2026-11-10）

用户已确认排除与 query 完整标签组合相同的候选，保留其他组合的完整、部分和零覆盖；过滤在评分/粗筛之前执行。该模式使用 `coverage_cross_combination_v1` 派生协议，沿用六项 coverage 公式，但 IDCG 和完整覆盖相关数按剩余完整候选库计算。本文第 6 节仍是原正式实验结果，不替换为新模式结果。

六种方法均已提供独立入口，复用既有模型且保留历史结果。冻结 test 身份审计显示每 query 剩余 6,999–7,170 个候选，999 个 query 无跨组合完整覆盖候选；后者保留，按各指标零分母规则处理。本次仅实现和验证，尚未产生新模式正式 test 分数。完整协议、命令及限制见 `compared_methods/docs/CROSS_COMBINATION_COVERAGE.md`（相对于外层工作区）。
