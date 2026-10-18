# WBM 局部区域检索实验

本工程在 Mixed-WM38 上研究有方向的区域检索：query 中的缺陷区域应尽量在 candidate 中找到相似区域，candidate 可以含有额外缺陷。当前三类方法为 Zernike 矩局部基线、自监督局部主方法 proposed、整图监督参考；监督参考使用 5%、10%、50% 三种训练标签预算，最终对比表共五行。

2026-10-17 版本将网络和训练配置向 WaPIRL 对齐，保留已经确认的单通道输入和区域学习任务，采用完整形状的直角旋转增强。运行代码独立于 `wbm-wdm-matching`，不下载外部模型权重。

## 当前方法与已确认的设计

| 项目 | proposed | 整图监督参考 |
| --- | --- | --- |
| 输入 | 单个 CC 区域的 1×96×96 二值 patch | 整张晶圆的 1×96×96 缺陷图 |
| 编码器 | 同一个 `Encoder` 类：ResNet-18，7×7 stride 2 + max-pool，残差块 [2,2,2,2] | 完全相同的类、结构和初始化规则 |
| 表示 | GAP 后的原始 512 维特征，无 512→256 瓶颈 | 同样的原始 512 维特征 |
| 训练头 | 512→128 线性 projector | 有方向的 containment pair scorer |
| 优化 | SGD，lr=0.01，momentum=0.9，weight decay=0.001 | 相同 |
| 正式训练轮数 | 100，关闭早停，保留最低 valid loss checkpoint | 每个标签预算 100，采用相同规则 |
| 增强 | 0°/90°/180°/270°，无随机裁剪、dropout 或噪声 | 相同的增强配置，分别用于两张图 |
| 正式检索 | 512 维特征的 cosine + 区域一对一匹配 | 512 维特征 + 有方向的评分头 |

两类学习模型从相同 seed 随机初始化，各自独立训练，不加载另一方法的权重。每个训练报告和 checkpoint 都记录 `encoder_spec` 与 `initial_encoder_sha256`，可以核对结构和初始编码器参数一致。监督参考继续使用评分头，因为标签包含率具有方向性，单个对称 cosine 无法直接表达它。相同架构与优化配置不意味着两类方法的训练样本数、优化步数或监督信息相同；前者训练样本是区域，后者是有序整图 pair。

### proposed 的训练流程

先以标准 8 连通域提取缺陷区域，默认过滤面积小于 5 的区域，每张图按面积保留最多 6 个。没有学习分割器、ring-aware 规则或腐蚀拆分。仅裁剪每个区域自身的 bbox，居中补成正方形，加 4 像素边距，然后用最近邻 resize 到 96×96；这里的 bbox 裁剪保留全部区域像素，不是随机截断形状。

对全部 train 区域建立 memory bank，正式训练前用编码器和线性 projector 前向初始化。每一步分别计算原始视图及旋转视图相对其 memory 表示的温度 NCE 损失，等权平均；温度为 0.07，负样本数为 5,000，从 train bank 中有放回采样，正确排除当前训练 batch。可用 token 少于 5,000 时仍采样 5,000 次，允许重复；只有小规模诊断的一个 batch 覆盖整个 bank 时才逐 anchor 排除自身。

memory 更新为 `m ← 0.5*m + 0.5*z`，`z` 是原始视图的原始线性投影输出，不对投影或 EMA 结果强制 L2 归一化。损失在计算 cosine 时归一化，使用矩阵乘法避免原版对负样本的三维重复分配。此处继承 WaPIRL 的更新设计，但修正负样本排除问题，并为所有 token 使用唯一全局索引。

验证使用固定旋转和固定随机种子的 train-bank 负样本；valid ID 不用于排除 train-bank ID，两者是不同的索引空间。valid/test 不写入 memory，不参与优化。正式检索仅使用 GAP 的 512 维输出，projector 不参与检索。该方法仍是实例区分，不包含伪正例或聚类第二阶段，相似区域可能被当作负样本。

网络卷积、BatchNorm 和线性 projector 的初始化参照本地原版 WaPIRL，包括其 BatchNorm bias=1；学习率按原版零 warmup 的 cosine multiplier 调度，乘数为 `1e-4 + max(0, 0.5*(1+cos(pi*epoch/epochs)))`。固定验证随机数和完整 checkpoint 恢复属于本工程的可复现实现。

### Zernike 矩传统局部基线

`methods/handcrafted.py` 已整体替换旧的 Hu、径向直方图和统计量拼接。描述子依据：

> A. Khotanzad and Y. H. Hong. **Invariant Image Recognition by Zernike Moments**. IEEE Transactions on Pattern Analysis and Machine Intelligence, 12(5):489–497, 1990. [DOI: 10.1109/34.55109](https://doi.org/10.1109/34.55109).

对完整区域 mask 的前景坐标减去质心，以最远前景点到质心的距离归一化到单位圆。保留孔洞，不填孔、不截掉角落、不做非等比例拉伸。计算 Zernike 矩幅值，并用零阶矩消除总质量因子；离散实现为 `|(n+1)*mean(R_nm(r)*exp(-i*m*theta))|`。

默认 `max_order=12`，按 n 递增、m 递增，保留 `2≤n≤12, 0≤m≤n, n-m为偶数` 的 47 个幅值。负 m 的幅值重复，因此不保留；零阶常量和质心归一化后为零的一阶项也不保留。不再拼接额外描述子，也不对整个向量做 L2 归一化。

区域间先计算欧氏距离 `d=||fQ-fC||2`，再映射成匹配分数 `shape=exp(-d/distance_scale)`，默认 `distance_scale=1.0`。Zernike 矩本身有论文依据；上述离散中心/尺度归一化、阶数、距离到分数的映射和外层 CC 检索属于本工程公开记录的适配，不声称完整复现原论文识别实验。参数是尚未调优的固定起点，应只在 valid 上决定后冻结。`handcrafted` 方法名称和结果路径保留，`retrieval_info.json` 明确标识实际表示为 `zernike`。

矩幅值在连续定义下旋转不变，离散像素旋转可能有误差；有限阶矩也不保证所有环和弧都被完全区分。单元测试验证了已知解析点集、旋转/平移一致性及示例环、弧、实心圆的差异。

### 整图监督参考与标签预算

整图输入使用 `raw==2`，补方并 resize 到 96×96；一个共享编码器分别产生 zQ、zC，评分头先归一化两者，再接收 `[zQ,zC,zQ*zC,zQ-zC]`。MLP 为 `2048→256→128→1`，sigmoid 输出 `[0,1]` 的标签包含率预测值，默认 MSE，也支持 smooth L1。这里不使用自监督 projector，也不把整图标签当作区域标签。

5%、10%、50% 指 train split 中允许使用标签的晶圆比例，不是 pair 比例。按完整 multi-hot 组合分层、每组至少一张，并使用固定顺序形成嵌套子集，实际比例与 ID 保存到 `budget.json` 和 `labeled_train_ids.csv`。每轮从该预算子集采样 50,000 个有序 pair，尽量平衡完整覆盖、部分覆盖、零重合，排除同图配对；三个预算使用相同轮数和 pair 数量。

所有预算共同使用 valid split 中固定的 10,000 对有标签样本选 checkpoint，预算只限制 train 标签，额外 valid 标签使用量如实记录。test 标签不参与训练、pair scorer、超参数或 checkpoint 选择。监督方法是带标签的整图参考，不是保证优于主方法的性能上界。

## 区域匹配、相似性与统一评价

两个局部方法使用相同的 CC 区域、贪心一对一分配和 query 侧聚合。proposed 的区域分数为 `max(0,cosine(hQ,hC))`，Zernike 使用上面的欧氏距离映射；阈值默认都是 0，零分区域对不参与匹配。得分分布不同，因此相同数字阈值不代表相同筛选强度。

区域对按分数降序依次选择，每个 query/candidate 区域最多使用一次，平分时按两侧 token 索引排序；这不是 Hungarian 最优分配。晶圆分数为：

`S(Q,C) = sum((i,j)∈M, sqrt(area(q_i))*shape(q_i,c_j)) / sum(i∈Q, sqrt(area(q_i)))`。

未匹配 query 区域在分母中贡献为零，未匹配 candidate 区域没有直接惩罚。无区域图保留参与排序，得分为零。正式实验固定 shape-only，位置和 PCA 尺度只作为元数据及独立补充示例，不混入主结果。

三类方法共用标签相关性 `R(Q,C)=|TQ∩TC|/|TQ|`。例如 `{A,B}` 查询 `{A,B,C}` 的相关性为 1，反向为 2/3。标签是 wafer 级，不提供区域对应真值，正式结果不能单独证明所有区域配对或工艺根因正确。

| 指标 | 含义 |
| --- | --- |
| `ContainmentNDCG@K` | 包含率作为 identity gain，对前 K 个结果计算归一化折损排序质量 |
| `MeanContainment@K` | 前 K 个结果的平均标签包含率 |
| `FullCoveragePrecision@K` | 前 K 个结果中，完整覆盖 query 标签的比例 |
| `FullCoverageRecall@K` | 前 K 个结果找回候选池全部完整覆盖正例的比例 |
| `FullCoverageHit@K` | 是否至少命中一个完整覆盖结果，再对 query 求平均 |
| `mAP_full` | 基于整个候选池完整排名的完整覆盖 AP 均值 |
| `MRR_full` | 第一个完整覆盖结果排名倒数的均值 |

K 默认为 1、5、10、20，共输出 22 个指标。理想排序与正例总数均根据完整候选池计算。没有完整覆盖正例的 query，其 Recall/Hit/AP/MRR 为空并从相应均值中排除，同时报告数量。按 query bootstrap 输出 95% 区间，正式配置 1,000 次；它反映 query 抽样不确定性，不替代不同训练 seed 的方差。还输出 proposed 相对 Zernike 的配对 NDCG@10 差值及区间。

## 数据与冻结协议

默认数据为同级 `data/wm38k/Wafer_Map_Datasets.npz`，`arr_0` 为原图、`arr_1` 为 multi-hot 标签，路径相对于本工程根目录解析。现有原图为 38,015 张 52×52 图；冻结清单覆盖 37,015 张非空标签图，train/valid/test 为 25,910/3,700/7,405。像素 2 为缺陷，1 和 2 用于有效晶圆几何计算，少量值 3 不自行改写为缺陷。

`protocol/split.csv`、`queries.csv`、`candidates.csv` 保留原清单，正式测试为 2,000 个固定 query，每张有 1,000 个不重复 test candidate，排除 query 自身。候选池来自旧工程的标签分层采样，结果代表该固定池，不能直接等同于全历史库检索。candidate 文件旧有的 jaccard/bucket 列不用于训练或评价，相关性从标签重新计算。

缓存记录输入 SHA-256、预处理配置和代码指纹；checkpoint 记录训练配置与代码指纹。版本、区域定义、描述子或模型变更时拒绝混用旧产物。

## 远程 CUDA 主实验运行方法

将本工程和原始 NPZ 放到远程机器上，保持现有相对位置，或修改 JSON 的 `data_file`。选择适合远程驱动的 CUDA PyTorch 安装；工程依赖见 `requirements.txt`，无需 torchvision 或原 matching 工程。

```bash
cd /path/to/wbm-region-retrieval
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
python run_experiment.py --config configs/experiment_a.json --dry-run
python run_experiment.py --config configs/experiment_a.json --stage audit
python -u run_experiment.py --config configs/experiment_a.json --device cuda
```

最后一条执行 `prepare → train-local → train-global → retrieve → evaluate`：一次局部模型、三次监督模型的真实训练，再输出五组 ranking 和评价。正式输出目录为 `artifacts/experiment_a_wapirl_zernike_seed2026`，旧的 `artifacts/experiment_a_seed2026` 和旧 MVP 都不会复用。

主实验默认 batch=256、推理 batch=256、workers=4，CPU 负责连通域和局部匹配，CUDA 负责两类模型训练、特征提取和监督评分头。当前为单 GPU FP32；如需减小 batch，请先复制配置并设置新的输出目录，所有学习方法共用该 batch。磁盘需保留数 GB 的缓存、checkpoint 和完整排名空间。

中断后从最近完成的 epoch 恢复：

```bash
python -u run_experiment.py --config configs/experiment_a.json --device cuda --resume
```

模型、SGD momentum、scheduler、memory bank、随机状态与最佳 epoch 一并恢复。已完成训练会跳过优化。新模型无法读取旧版 256 维模型或旧描述子缓存，`--resume` 不用于跨方法版本迁移。多 seed 应复制配置、更改 seed 和 output_dir，保留同一 protocol；正式参数只在 valid 上确定。

分阶段或只重跑某些方法：

```bash
python run_experiment.py --config configs/experiment_a.json --stage prepare
python run_experiment.py --config configs/experiment_a.json --stage train-local
python run_experiment.py --config configs/experiment_a.json --stage train-global
python run_experiment.py --config configs/experiment_a.json --stage retrieve
python run_experiment.py --config configs/experiment_a.json --stage evaluate
python run_experiment.py --config configs/experiment_a.json --stage retrieve --methods handcrafted
python run_experiment.py --config configs/experiment_a.json --stage evaluate --methods handcrafted
python run_experiment.py --config configs/experiment_a.json --stage geometry
```

`geometry` 独立输出位置、PCA 尺度变化的分数响应图，不属于默认 all 流程，也不是区域分割或图像鲁棒性 benchmark。位置以晶圆中心/半径归一化，PCA-scale 为 `sqrt(lambda1+lambda2)`；面积仅用于 proposal 和 query 权重。

## 预期输出及含义

```text
artifacts/experiment_a_wapirl_zernike_seed2026/
├── resolved_config.json / environment.json
├── cache/
│   ├── prepared.json
│   ├── maps.npy / labels.npy
│   ├── train_regions.npz / train_patches.npy / train_whole.npy
│   ├── valid_* / test_*
│   └── test_*_embeddings.npy
├── models/
│   ├── local/{best.pt,last.pt,history.json,training_summary.json}
│   ├── global_5pct/
│   ├── global_10pct/
│   └── global_50pct/
├── retrieval/
│   ├── handcrafted/
│   ├── proposed/
│   ├── global_5pct/
│   ├── global_10pct/
│   └── global_50pct/
├── comparison.csv / comparison.json
└── paired_proposed_vs_handcrafted.json
```

- `resolved_config.json`：实际参数；`environment.json`：Python、系统与包版本。
- `cache/prepared.json`：数据/代码指纹、划分与 token 数量、空区域图数量、Zernike 阶数和维数；`*_regions.npz` 保存区域 offsets、面积、位置、PCA 尺度和 Zernike 特征。
- `best.pt`：该模型 valid loss 最低的 checkpoint，供正式检索；`last.pt`：最后一轮状态，供续跑；`history.json`：每轮 train/valid loss、学习率、步数、耗时；`training_summary.json`：实际训练轮数、参数变动校验与编码器可比性信息。
- 每个监督目录另含 `labeled_train_ids.csv`、`budget.json`、`validation_pairs.npz`，用于审计实际标签量与固定验证 pair。
- 每个方法的 `rankings.csv`：`query_id,rank,candidate_id,similarity_score`；每个正式方法为 2,000×1,000=2,000,000 行，五组共 10,000,000 行。按分数降序、candidate ID 升序确定并列。
- 每个方法的 `metrics.json`：22 个均值、95% 区间、有效 query 数；`per_query_metrics.csv`：逐 query 数值；`retrieval_info.json`：表示、打分设置和 ranking 耗时。
- 两个局部方法额外输出 `matches.csv`，仅保存每个 query 前 10 个 candidate 的区域对应、形状分数、位置/尺度 affinity 和 query 权重。监督方法不输出虚构的区域对应。
- `comparison.csv/json`：五行总对比表，`handcrafted` 代表 Zernike；配对 JSON 保存 proposed−Zernike 的 NDCG@10 均值差及 bootstrap 区间。

`retrieval_info.json` 的 ranking 耗时不含提前完成的神经特征提取，不应直接当成端到端在线延迟。主实验的具体指标数值需远程实际训练后得到，代码不会预填性能。

## 本地 300-map MPS 验证与隔离

```bash
python -m unittest discover -s tests -v
python scripts/run_mvp.py
```

MVP 使用独立 `configs/mvp_mps_300.json`，输出到 `artifacts/mvp_mps_300_wapirl_zernike_20261017`。从原 train/valid/test 内分别取 210/30/60 张，共 300 张，保留原 split 并写入 `source_mapping.csv`。60 张 test 全作 query，每张有 59 个 candidate；子集 NPZ/CSV 只写到 MVP 目录。

MVP 与主实验共用新版编码器、512/128 维、Zernike、旋转增强、SGD、5,000 个有放回负样本和评价代码，仅为本地验证将 batch 改为 16、workers=0、各模型训练 3 轮、每监督轮 256 个 pair、验证 128 个 pair、bootstrap 100 次。关闭 MPS 自动 CPU fallback，确实在 MPS 上进行参数更新。数据准备、Zernike 和区域匹配仍按设计在 CPU 上执行。

脚本先后执行 audit、prepare、两类训练、五组检索、评价与几何示例，再执行完成态 `--resume` 检查训练文件哈希不变。它还核验四个模型的 loss/参数有限、参数确有更新、best 对应最低 valid loss、四个初始编码器哈希和结构相同、Zernike 维数正确、五组排名各有 3,540 行，以及正式配置/原始 NPZ/冻结 CSV 的运行前后哈希不变。

结果写入 MVP 的 `mvp_report.json`、`run.log`、`comparison.csv`。`status=passed` 表示流程检查通过；`formal_inputs_unchanged=true` 和 `completed_resume_keeps_training_unchanged=true` 是隔离与续跑校验。MVP 指标不用于判断正式性能，旧 MVP 目录保留作为历史产物。本次 33 项自动化测试全部通过，真实 MPS 全流程及续跑共耗时 122.14 秒。测试环境与实测结果见 `VALIDATION.md`。

MVP 被中断可执行 `python scripts/run_mvp.py --resume`；`--prepare-only` 只生成隔离子集，不运行训练。若进程无法访问 Metal，会明确报错，不降级为 CPU 训练。

## 不重训的 valid 诊断

已有 `models/local/best.pt`、`last.pt` 及 `cache/valid_regions.npz`、`valid_patches.npy` 时，运行独立脚本：

```bash
python -u scripts/diagnose_local.py \
  --config configs/experiment_a.json \
  --device cuda \
  --output-dir artifacts/diagnostics_valid_seed2026 \
  --queries 500 --candidates 1000 --seed 2026
```

这不是训练入口，不调用 `prepare`、优化器或 `backward`。六组分别为 `best_h512`、`best_z128`、`last_h512`、`last_z128`、`random_h512`、`random_z128`。`h512` 是 GAP 后的 backbone 表示，`z128` 是同一模型线性 projector 的输出；六组都做 L2 归一化，使用同一套 cosine 截断、面积权重和一对一区域匹配。随机模型按训练配置的 seed 初始化，未经训练和 BN 数据校准；脚本记录其初始参数哈希是否与 checkpoint 的训练初始哈希一致。它不是均匀随机排名，随机网络也可能保留图像结构偏好。

默认从 valid 均匀选 500 个查询，每查询独立均匀抽 1,000 个 valid 候选，均不使用标签采样，排除自身，冻结在新目录的 `protocol/`，所有变体完全复用。样本不足自动取可用数量；`--queries 0 --candidates 0` 表示全部 valid 查询与全部其余 valid 候选，开销明显增大。这里不用正式 test 的 2,000×1,000 检索池，诊断值不能直接和主实验 test 表比较。

`--pair-samples 20000` 指随机跨晶圆区域对数量（有放回、排除同晶圆）。另外选择至多 20,000 个不同区域，分别与其固定 90/180/270 度旋转比较，诊断旋转一致性。`--neighbor-anchors 24 --neighbors 8` 为人工核查选取最多 24 个 anchor，尽量均分单缺陷/混合缺陷晶圆，每个找 8 个其他 valid 晶圆的区域近邻。近邻按原始 cosine 排序，CSV 同时报告截断分数。晶圆标签仅作为上下文，不当作区域真值；随机区域对也不等于真实语义负样本。

输出目录必须与原实验目录互不包含，默认使用原实验目录名加 `_valid_diagnostics`。脚本只读原实验，写入新目录，并在前后核对原实验全部文件路径/大小/修改时间、配置与主要输入的 SHA256。如果主实验有其他进程同时写文件，校验会失败，应等其结束后诊断。请勿通过本诊断修改训练配置。

```text
artifacts/diagnostics_valid_seed2026/
├── run.log / run_manifest.json / source_audit.json / model_info.json
├── protocol/{manifest.json,queries.csv,candidates.csv,valid_maps.csv,...}
├── features/{best,last,random}/        # 仅本次诊断的原始与旋转 h512/z128
├── variants/<六个变体>/
│   ├── rankings.csv / per_query_metrics.csv / metrics.json
│   ├── map_score_by_query.csv / map_score_distribution.json
│   ├── score_pairs.csv / score_distribution.json
│   └── region_neighbors.csv
├── comparison.csv / comparison.json / paired_comparisons.json
├── figures/{retrieval_by_group.png,region_cosine_distributions.png,neighbors_token_*.png}
└── report.md
```

- `comparison.csv`：六个表示 × `all/single/mixed` 共 18 行，包含正式评价的 22 个指标。single 为查询晶圆只有一种基础缺陷，mixed 为两种及以上；**候选始终是原来的同一混合池**。`metrics.json` 另存分组有效查询数和查询 bootstrap 95% 区间。
- `paired_comparisons.json`：相同查询上的 `z128−h512`、`best−last`、训练模型减相同维度随机模型，分别报告 NDCG@10、mAP 的均值差和配对区间。
- `map_score_*`：晶圆级匹配分数，按 query 标签全覆盖、部分重叠和不相交划分。`score_distribution.json`：区域原始 cosine/截断分数分位数、旋转正对与随机对的差距及视图区分 AUC，以及特征范数、均值方向集中程度。高 cosine 或均值方向集中本身不能证明表示塌缩。
- `region_neighbors.csv` 与近邻图：同一批 anchor 在六个空间的结果，用于查看近邻是否形状相似、是否偏好通用碎片，以及混合晶圆中的区域是否被错误对应。
- `report.md`：本次实际结果表、配对差异及解读边界。mAP/Recall/Hit 在没有全覆盖候选的查询上记为缺失，不能把缺失误判为 0；各指标有效样本数见 `metrics.json`。
- `run_manifest.json`：设备、输入/代码哈希、诊断参数和完成状态；`source_audit.json` 的 `source_unchanged=true` 表示隔离校验通过；`model_info.json` 记录实际 epoch 和随机初始参数对应关系。

中断后可在**完全相同命令**末尾加 `--resume`，复用校验过的诊断特征和 protocol，重新生成排名/分析。修改参数、输入、设备或代码应换新输出目录。`--batch-size` 仅控制诊断推理 batch，`--num-workers` 默认 0，不改变训练参数。CUDA/MPS 只负责网络推理；区域匹配、统计和绘图在 CPU。运行日志自动保存在独立诊断目录的 `run.log`。

远程部署只需同步新增的 `scripts/diagnose_local.py` 和 `wbm_retrieval/diagnostics.py`（以及本 README 和诊断测试）。**不要替换已经训练所用的 `models.py`、`training.py` 或 `methods/proposed.py`**；checkpoint 会验证训练代码指纹。

本地可用已有 300-map MVP checkpoint 验证，不重训：

```bash
python scripts/diagnose_local.py \
  --config configs/mvp_mps_300.json --device mps \
  --output-dir artifacts/diagnostics_mvp_valid_20261018 \
  --queries 12 --candidates 20 --pair-samples 128 \
  --neighbor-anchors 4 --neighbors 3 --bootstrap 100
```

本次已使用已有 300-map 模型完成真实 MPS 验证，无重新训练：12 个 valid 查询、每个 20 个候选、175 个区域，六组共 1,440 条排名，18 行分组对比及 6 张图。最终验证目录为 `artifacts/diagnostics_mvp_valid_20261018_verified`，首次运行耗时 26.66 秒，缓存复用耗时 5.67 秒；特征、protocol、排名的 SHA256 在复用前后相同，原实验 85 个文件的隔离检查通过。该目录的 `validation.json` 记录验证结果；38 项自动化测试通过，包括与正式匹配器及正式评价函数的一致性检查。这些 MVP 数值仅用于验证诊断流程，正式模型仍需执行上面的远程命令。

## 代码位置与参考

`run_experiment.py` 是统一入口。`wbm_retrieval/methods/proposed.py`、`handcrafted.py`、`supervised_global.py` 分别实现三类方法；`models.py` 定义共享网络，`training.py` 保存共同优化/恢复组件，`regions.py` 提取区域和增强，`matching.py` 匹配，`evaluation.py` 统一评价，`scripts/run_mvp.py` 运行隔离测试。

局部训练参考 Kahng 与 Kim 的 *Self-Supervised Representation Learning for Wafer Bin Map Defect Pattern Classification* 及 [WaPIRL 官方代码](https://github.com/hgkahng/WaPIRL)。Zernike 矩参考上述 Khotanzad 与 Hong 的论文；其幅值旋转不变表示亦见 [Rotation Invariant Image Recognition Using Features Selected via a Systematic Method](https://doi.org/10.1016/0031-3203(90)90005-6)。本工程不使用其有标签特征选择流程。

当前范围不包括 Kwon 方法复现、学习分割器、伪正例训练或额外区域标注 benchmark。checkpoint 使用 PyTorch 序列化，仅加载可信来源的文件。
