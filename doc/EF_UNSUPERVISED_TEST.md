# E/F 冻结模型无监督测试

本入口只评价增强一致性、实例区分和特征分布，不训练分类器、不读取类别标签、不更新模型或 memory bank。它不会调用旧入口的 `prepare_suite` 或改写源实验。脚本放在 `scripts/`，不修改训练模块，也不进入旧入口的包源码哈希集合。

## 服务器运行

将 `scripts/test_ef_unsupervised.py` 同步到服务器项目的同名位置，然后使用已有 PyTorch 环境运行：

```bash
cd /data/wangkaiyan2024/wbm-region-retrieval
python -u scripts/test_ef_unsupervised.py \
  --source-root /data/wangkaiyan2024/wbm-region-retrieval/artifacts/ablation_ef_20261020 \
  --output-root /data/wangkaiyan2024/wbm-region-retrieval/artifacts/ef_unsupervised_test_20261023 \
  --seed 2026 --device cuda \
  --batch-size 128 --score-batch-size 128 \
  --bootstrap 1000 --distribution-pairs 100000
```

输出目录必须不存在；重复运行请使用新目录名。默认同时评价 E/F 的 `best_nce.pt` 及各自架构的随机初始化基线，不会重新训练。默认使用全部 test 区域。显存不足可将两个 batch size 改为 32，不改变候选池。

需要源实验的 `suite_manifest.json`、`cache/prepared.json`、`cache/{train,valid,test}_regions.npz`、`cache/test_B.npy`、`cache/test_F.npy`，以及 `seed_2026/{E,F}/` 下的 `run_manifest.json`、`training_summary.json` 和所选 checkpoint。仅有两个 `.pt` 文件不足以核验数据和输入对应关系。

可用 `--max-regions 256 --bootstrap 100 --distribution-pairs 2000` 检查服务器环境；该模式会改变候选池，结果不得作为全量测试结果。`--skip-random-baseline` 可跳过随机模型。

`--checkpoint best_retrieval` 可另行评价，但该权重使用类别标签选模，报告会标记 `selection_uses_class_labels=true`。不能查看 test 分数后从两个 checkpoint 中选择更优者作为无监督主结果。已有数据划分本身曾按标签分层，也不能将整个历史数据处理过程描述为从未接触标签。

## 固定测试任务

- 原始输入使用训练实验保存的 E/F test patch。分别将方形 patch 旋转 90°、180°、270°，F 的三个通道同时旋转；不包含 0° 恒等变换，不额外加入 E 专有的尺度/平移增强。
- 每个原始区域作为查询，自己的旋转视图为唯一指定正样本。候选是该正样本与所有其他晶圆的旋转区域；同一晶圆的其他区域排除。不同查询的候选数量可能略有差异，会逐条记录。
- E/F 使用完全相同的 token、旋转、随机配对和测试池，分别报告 L2 归一化的 `h512` 和 `z128`。
- 余弦分数相差不超过 `1e-6` 视为与正样本并列，使用均匀随机打破并列的期望 Top-K、MRR、排名。避免全坍塌特征因 token 排序偶然获得高分。`median_expected_rank` 是各查询期望排名的中位数。
- 结果以区域-视图等权平均；置信区间按整张晶圆 bootstrap，同一晶圆的全部区域和旋转视图一并抽样。配对差值使用相同 query。该区间不代表不同训练种子的波动。

## 指标

主指标：`top1`、`recall_at_5`、`recall_at_10`、`mrr` 越高越好；`mean_rank` 越低越好。报告候选规模对应的 `chance_top1` 和 `chance_mrr`。

对于查询 i，记正样本余弦分数为 p，跨晶圆负样本分数为 n_j：

| 字段 | 定义及解释 |
|---|---|
| `hard_negative_margin` | p - max(n_j)。越大表示与最近干扰项拉得越开；负值表示存在负样本比正样本更相似。 |
| `crowding_fraction` | 满足 n_j >= p - 0.05 的负样本比例。越高表示越拥挤，包含超过正样本的负样本。阈值由 `--crowding-margin` 预先固定，不是合格线。 |
| `crowding_count` | 同一条件下的负样本个数，必须结合候选数量阅读。 |
| `negative_cosine_mean` | 全部有效负样本余弦相似度均值。 |
| `positive_negative_mean_gap` | p - mean(n_j)。看整体间隔，需结合最难负样本间隔。 |
| `alignment_squared` | 归一化原图与增强图的平方距离，等于 2-2p；越低一致性越强，但坍塌也能使其很低。 |
| `gallery_nce` | 当前完整测试候选池上的 softmax 对比损失，温度取保存配置。不是历史训练 memory bank 的 NCE，不能直接与训练 loss 数值比较。 |
| `sampled_pair_roc_auc` | 所有正样本分数相对固定抽样跨晶圆负样本分数的 ROC-AUC，精确并列计半；这是分布区分 AUC，不是缺陷分类 AUC。 |
| `random_negative_cosine_quantiles` | 固定抽样负样本分数的 P05、P50、P95、P99，观察高相似度尾部。 |
| `effective_rank_covariance` | 原始归一化特征中心化后，协方差特征值归一化为概率，计算 exp(谱熵)。有效方向数越少，表示越集中于低维子空间。全常量特征记为 0。上界是 min(N-1,D)，不能直接跨不同 D 比绝对大小。 |
| `total_centered_variance` | 原始归一化特征的总中心化方差。接近 0 提示所有向量挤在很小范围；有效秩高也可能伴随极小方差。 |
| `top_eigenvalue_share` | 第一主方向占总方差的比例；常量特征记为 null。 |
| `cross_wafer_uniformity_t2` | log mean exp(-2 * squared_distance)，在固定抽样的跨晶圆原始特征对上计算。更负通常更分散，全相同为 0；它是排除同晶圆后的 uniformity 诊断。 |

高有效秩、更分散都不自动等于更好的形状表示；随机噪声也可能很分散。应同时看增强一致性、排名、margin 和分布。跨晶圆“负样本”只是不同实例，可能有相同缺陷形状，不能把所有高相似度都判断为错误。

## 输出

- `loss_history.png`：一张双面板图。左侧对比 E/F 逐轮 `train_loss`，右侧对比 `valid_nce` 并用星号标出各自 `best_nce` 的轮次；直接读取保存的 `history.json` 和 `training_summary.json`，不平滑，不保证曲线单调下降。仅一轮时显示单点并提示不能推断趋势。训练 loss 与验证 loss 的正样本定义不同，应各自在其面板内观察趋势。
- `summary.csv`：E/F × 已训练/随机 × h512/z128 的总览。
- `paired_comparisons.json`：E-F、各组已训练-随机的配对差值及晶圆 bootstrap 区间。
- `E/best_nce/h512/metrics.json` 等：三种旋转汇总、置信区间及原始特征谱指标。
- `E/best_nce/h512/per_query.csv` 等：每个区域、旋转的完整数值。
- `E/best_nce/h512/rotation_90/distribution.png` 等：正样本、随机负样本、最难负样本分布；中间图放大合并分数的 0.5%-99.5% 范围（各图横轴范围可能不同），右图显示最难负样本间隔。
- 同目录 `metrics.json` 和 `distribution_samples.npz`：每角度摘要与可重新作图的分数样本。
- `protocol.json`、`test_pool.npz`：协议、样本身份、代码/数据/权重哈希、环境及完成状态。

只有 `protocol.json` 的 `status=complete` 才表示全部完成。结束时重新校验源文件哈希，并验证模型整个 state_dict（包含 BatchNorm buffers）保持不变。

随机基线使用原 seed 和当前运行环境重新初始化。`model.json` 记录是否与训练清单的初始参数哈希匹配；若不匹配，只能称为同架构的新随机基线，不能称为历史 epoch 0 的精确复现。

默认完整运行共生成 **97 个文件**（原 96 个加 `loss_history.png`）。绘图额外需要两组的完整 `history.json`；PNG 内嵌历史文件哈希，完整运行的 `protocol.json` 也记录这些来源。

若测试已经运行，只想补画 loss 图，可在同步新版脚本后执行以下命令。它只生成一张图，不加载权重或缓存、不进行推理、不要求 GPU；输出放在一个新的独立目录：

```bash
python -u scripts/test_ef_unsupervised.py \
  --source-root /data/wangkaiyan2024/wbm-region-retrieval/artifacts/ablation_ef_20261020 \
  --output-root /data/wangkaiyan2024/wbm-region-retrieval/artifacts/ef_loss_history_20261023 \
  --seed 2026 --loss-only
```
