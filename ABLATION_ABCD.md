# A～D：输入尺度、增强与首层下采样消融

入口：`scripts/run_ablation.py`；配置：`configs/ablation_abcd.json`。
这是独立实验，不修改 `configs/experiment_a.json`、原训练入口、历史缓存或 checkpoint。
只训练 proposed，不重训三组 global 模型；手工 Zernike47 仅在同一评价池上计算一次。

## 实验定义

四组都从相同 seed 的随机权重独立训练，不接着旧 checkpoint 微调。

| 组 | 原始分支与检索输入 | 增强分支 | 首层卷积 stride | 要回答的问题 |
| --- | --- | --- | --- | --- |
| A | bbox 裁剪 → 补方＋原网格边距 4 → 最近邻 resize 到 96 | 直角旋转 | 2 | 关闭开运算后的当前输入基线 |
| B | bbox 裁剪 → 仅补方 → 最近邻 resize 到 88 → 最终网格每边 padding 4，得到 96 | 直角旋转 | 2 | B−A：固定输出边距、统一形状占比是否有效 |
| C | 与 B 完全相同 | 从原始区域补方后，最近邻 resize 到随机 80～88 → 放入 96 画布，中心偏移各轴随机 −2～2 → 直角旋转 | 2 | C−B：轻度尺度＋平移增强是否有效 |
| D | 与 B 完全相同 | 直角旋转 | 1 | D−B：减弱首层下采样是否有效 |

“补方”只是在短边添加零以保持纵横比，不把长方形直接拉伸成正方形。B 的额外边距位于 resize 之后。
C 的尺寸和偏移均为离散均匀采样；缩放直接从原始 mask 生成，不对标准 96 patch 二次 resize。画布足够大，不因平移截掉前景。
所有旋转都是均匀采样 0/90/180/270°，不包含任意角旋转。相同 seed/epoch/token 的旋转一致。

D **只**修改 7×7 首层卷积的 stride，保留 3×3 stride=2 max-pool 和所有残差块。
96 输入时，GAP 前特征图从 A/B/C 的 3×3 变为 D 的 6×6；参数数量与初始化参数不变，但显存和计算量增大。

四组都关闭开运算、dropout、随机加噪，保留原开运算代码供后续单独对照。
本轮不叠加 C 与 D，也不能凭四组都关闭开运算就断言开运算有害。C 测试的是尺度＋平移这一组合，不能单独归因到其中一个因素。

## 固定条件与选择规则

- 原有 split 不变：train 25,910、valid 3,700、test 7,405 张有标签 map；标签不用于 NCE 梯度训练。
- 区域提取不变：原始缺陷 mask、8 连通域、min_area=5、top_k=4；匹配面积仍取原网格，不用 resize 后像素数。
- ResNet-18、h512 检索、线性 projector128、SGD lr=0.01、momentum=0.9、weight_decay=0.001、100 epochs、无早停。
- 保留原 WaPIRL 对齐的 cosine 调度、temperature=0.07、5,000 个有放回负样本、排除当前训练 batch、memory EMA=0.5。
- 默认 batch_size=256（区域数），两视图拼接意味着一次前向 512 张 patch。若改 batch，四组必须统一，并使用新的 output root。
- 匹配仍为 shape-only、cosine 截断到 [0,1]、分数降序贪心一对一分配、query 侧 sqrt_area 加权；并列按 candidate ID 升序。
- 固定 valid 池：500 个均匀查询，各 1,000 个不重复候选，排除自身；抽样不看标签；池 seed 固定 2026，即使训练 seed 改变也复用同一池。
- 第 1、5、10、…、100 epoch 计算 valid 检索，按 `ContainmentNDCG@10` 最大值保存 `best_retrieval.pt`，并列保留较早 epoch。
- 每 epoch 计算固定视图/固定负样本随机数的 valid NCE，另存最低损失的 `best_nce.pt`。它与检索最佳模型可能不是同一个 epoch。
- 初始随机模型也计算 valid 检索作诊断，但不参与最佳训练 checkpoint 选择。
- `all` 只训练并评价 valid；准备阶段可缓存 test 输入，但不会评价 test。test 必须另行显式执行。

注意：`best_retrieval` **使用 valid 标签进行模型选择**，不能描述成“全流程完全不使用标签”。如果论文要求 checkpoint 选择也不使用标签，必须预先指定以 `best_nce` 为主；两者都已保留，不能在看过 test 后选择更有利者。
当前 valid 表用于选择设计，不与之前的 test 指标直接比较。配对 bootstrap 只描述固定池上的查询波动，不含训练 seed 波动，也不能消除 valid 选择偏差。

## 远程运行

先将这些代码同步到服务器，进入服务器上的 `match-test/wbm-region-retrieval`，激活已有 CUDA PyTorch 环境。
目录结构应保留 `../data/wm38k/Wafer_Map_Datasets.npz` 和本工程的三份 `protocol/*.csv`。
依赖见 `requirements.txt`；不要为了本实验覆盖服务器已有的可用 CUDA 环境。

### 1. 检查环境与展开后的配置

```bash
python -c "import torch, numpy, scipy; print('torch:', torch.__version__); assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
python scripts/run_ablation.py --config configs/ablation_abcd.json --dry-run
```

`--dry-run` 不写文件、不加载数据、不训练；检查输出的 opening.enabled=false、top_k=4、四组 spec。

### 2. 一次顺序运行四组

```bash
mkdir -p artifacts/ablation_logs
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
nohup python -u scripts/run_ablation.py \
  --config configs/ablation_abcd.json \
  --stage all --variants A B C D \
  --device cuda --seed 2026 --num-workers 4 \
  > artifacts/ablation_logs/seed2026.log 2>&1 &
```

```bash
tail -f artifacts/ablation_logs/seed2026.log
```

四组不是同时竞争同一张 GPU，而是依次完整运行；每组从相同初始 seed 开始。可改为 `--variants D A B C`，先跑显存开销最大的 D。
同一 output root 有进程锁，不支持多个进程并发写入；不同 GPU 并行运行并非本入口的默认工作方式。

若已知显存有限，第一次运行就统一使用更小 batch，例如将上面的参数加上：

```bash
--batch-size 128 --output-root artifacts/ablation_abcd_bs128
```

这两项是追加参数示例，不是独立 shell 命令。之后所有训练、续跑、评价命令都要保留相同两项。
不要只降低 D 的 batch，也不要在原目录内改变 batch 后续跑；代码会拒绝混用。

### 3. 中断后续跑

确认之前的进程已经退出，再执行：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
nohup python -u scripts/run_ablation.py \
  --config configs/ablation_abcd.json \
  --stage all --variants A B C D \
  --device cuda --seed 2026 --num-workers 4 --resume \
  >> artifacts/ablation_logs/seed2026.log 2>&1 &
```

续跑恢复模型、优化器、学习率调度、memory bank、随机状态与历史记录。已完成组不会再次更新参数；可能重新计算其最终检索报告。
中断发生在 epoch 中间时，从上一个已完成 epoch 重新开始，不是 batch 级恢复。
配置、数据、协议、实现代码与缓存有哈希检查；修改后应换新 output root，不要在长任务运行期间更新代码。

### 4. 分阶段或分组运行

```bash
python scripts/run_ablation.py --stage prepare
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_ablation.py --stage train --variants B --device cuda --seed 2026 --resume
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_ablation.py --stage evaluate --variants B --device cuda --seed 2026
python scripts/run_ablation.py --stage summarize --seed 2026
```

`prepare` 准备共享缓存和冻结 valid 池；`train` 不输出最终对比表所需的完整检索结果，需随后执行 `evaluate`。
`summarize` 汇总已存在的结果，缺少的组不会伪造。所有阶段仍会验证 suite 和缓存一致性。

### 5. 增加训练 seed（可选）

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
python -u scripts/run_ablation.py --stage all --variants A B C D --device cuda --seed 2027 --resume
```

共享相同缓存和 valid 池，训练输出进入 `seed_2027`；可再运行 seed 2028。每个 seed 独立输出对比表，当前入口不会自动把多个 seed 混成一张统计表。
不要从不同 seed 中挑最优一轮作为总体结论。

### 6. 固定设计后才运行 test

例如 **仅当 valid 最后选定 B，且预先决定使用 retrieval checkpoint** 时：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
python -u scripts/run_ablation.py \
  --config configs/ablation_abcd.json --stage test --variants B \
  --device cuda --seed 2026 --test-checkpoint best_retrieval
```

该步骤复用既有 test 查询/候选清单，不训练。若事先决定使用 NCE checkpoint，将参数改为 `--test-checkpoint best_nce`。
不指定 `--variants` 的 test 会被拒绝，以免默认自动测试所有设计。

## 输出与判读

默认根目录：`artifacts/ablation_abcd_20261019/`。

| 路径 | 用途 |
| --- | --- |
| `suite_manifest.json`、`cache/prepared.json` | 输入/配置/代码/缓存 SHA256 与数据规模 |
| `valid_pool.json` | 四组、所有训练 seed 共享的 valid 查询/候选池 |
| `handcrafted/valid/` | 原始 mask 的 Zernike47 对照，无需重复训练 |
| `seed_2026/comparison_valid.csv` | 首先查看：四组 random、best_nce、best_retrieval 的 all/single/mixed 指标，另附手工基线 |
| `seed_2026/paired_valid.json` | B−A、C−B、D−B 的配对差值与查询 bootstrap 区间，基于 best_retrieval |
| `seed_2026/A/`（B/C/D 同理） | `last.pt`、`best_nce.pt`、`best_retrieval.pt`、训练配置和环境、history、training_summary |
| `seed_2026/A/valid/epoch_*/` | 检索检查点的指标与逐查询结果，不存每轮完整排名以节省空间 |
| `seed_2026/A/valid/best_retrieval/` | 最终所选 checkpoint 的完整 rankings、指标、逐 query 结果和置信区间 |
| `seed_2026/A/valid/best_nce/` | NCE 最优 checkpoint 的同池对照 |
| `seed_2026/comparison_test.csv` | 仅在显式执行 test 后生成 |

重点看同一 seed、同一 checkpoint 选择规则下的三组差值，以及训练模型是否超过自己的 random 初始表示。
若 B 改善，支持输入规范化值得保留；若 C 改善，支持当前轻度几何增强组合；若 D 改善，支持首层下采样过强这一解释。
这些是受控实验的证据方向，不是仅凭单次指标就证明形状信息丢失或泛化机制。第一轮筛选后再用多 seed 核对稳定性。

## 本地验证范围

`tests/test_ablation.py` 包含实际 CPU 参数更新：四组各跑一轮，检查输入、二值性、共享初始参数、3×3/6×6 特征图、评分与旧评价器一致、完成后续跑不改 checkpoint，以及两轮训练中断/恢复与连续训练的模型和 memory 逐元素一致。
测试数据是 24 张合成 map（每 split 8 张），只验证实现；它的数值不是 Mixed-WM38 性能。
完整数据集 100 epoch 和 CUDA 显存/速度/收敛仍需远程运行确认。
