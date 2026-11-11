# 2026-11-09：统一完整标签组合检索评价

正式配置使用 `evaluation.metric_version=exact_v1`，K 为
1、5、10、20、50、100、500；实现 `wbm_retrieval/exact_metrics.py` 与
`compared_methods/shared/exact_metrics.py` 逐字节相同。历史 MVP 配置保留旧
containment 语义；历史结果不可与本协议混用。论文主结果同时报告独立的
`coverage_v1`：它使用同一份固定 test 排名，按查询缺陷类别被候选覆盖的比例评价部分覆盖，
并以 `Lq ⊆ Lc` 定义完整覆盖。

## 正样本和公式

query 与 candidate 的完整 8 维 multi-hot 标签完全相同才是正样本。
例如 `[center]` 与 `[center,scratch]` 不相关；后者只与同一完整组合相关。
Normal 不进入检索库。自身必须排除。

令 `rel(r)` 为第 r 名是否相关，`H(K)=sum(rel(r), r<=K)`，`R` 为完整固定
候选库中相关样本总数（包括粗筛后未返回的样本）：

- `Hit@K = 1[H(K)>0]`。
- `Precision@K = H(K)/K`，不足 K 个返回时缺位按未命中处理。
- `Recall@K = H(K)/R`。
- `AP@K = sum(Precision@r * rel(r), r<=K)/R`；`mAP@K` 为 query 的 AP 平均。

AP 的分母是 R，**不是 min(R,K)**。`R=0` 时 Recall/AP 记为未定义并从均值排除，
Hit/Precision 仍参与平均。另报告各完整组合先求均值、再对类别等权平均的
`class_macro_metrics`，避免把 query 平均误称为类别宏平均。

## 固定数据和排序

继续读取 `protocol/deduplicated/` 与 `protocol/retrieval_v1/`；缺陷数据
train/valid/test 为 25193/3599/7199。valid 固定 500 个 query，每个有 3598 个
候选；test 全部 7199 个 query，每个有 7198 个候选。sample_id 是原 NPZ 行号。

先按方法相似度降序排序；同分按 compared_methods 相同的 SHA-256 规则排序，
`tie_seed=2026`，不读取标签，不随训练 seed 改变。SHA 输入为 ASCII
`exact_v1:tie:{seed}:{query_id}:{candidate_id}`，取前 8 字节大端无符号整数，
最后才以原始 ID 兜底。每个 query 在完整候选库计算分数，流式保存 Top max(K)，
默认 Top 500；这不是完整精排输出。

`metrics.json`、`per_query.csv`、`per_class.csv` 与 compared_methods 使用相同
核心字段，另保留套件所需的 `per_query.json`、`per_query_metrics.csv`。
每 query 列名为 `AP@K`，汇总列名为 `mAP@K`。输出不可覆盖；中断的训练评价
会保留原产物，改写到新的 attempt 目录。正式输出使用
`artifacts/deduplicated_exact_v1/`，不读取历史模型或历史排名作为新协议结果。

## F 主方法训练与 test

本轮预先固定 F 为主方法，seed=2026，100 epoch，原 SGD、NCE、数据增强、
三通道固定 52×52 窗口和区域匹配算法不变。`best_nce`（valid NCE 最低）用于
正式 test；不根据 test 结果选择 checkpoint。每 5 epoch 的 valid 检索监测改为
`mAP@10`，产生的 `best_retrieval` 仅作为辅助 valid 监测结果。

以下命令使用同一个全新 `<RUN>` 目录，在项目根目录、`wbm` 环境执行：

```bash
python experiments/proposed/f/run.py --stage train --output-root <RUN>
python experiments/proposed/f/run.py --stage evaluate --output-root <RUN> --skip-handcrafted
python experiments/proposed/f/run.py --stage test --output-root <RUN> --skip-handcrafted --test-checkpoint best_nce
```

`run.py` 固定 variant F。最后一条会对全部 test 执行正式评价；`all` 不自动运行
test。`--skip-handcrafted` 仅跳过附加手工基线，不改变 F 的候选或评分。
评估前检查训练已完成、checkpoint 与代码/配置/数据签名一致。最终结果位于
`<RUN>/seed_2026/F/test/best_nce/metrics.json`。

## 验证与本次正式运行

- 最终本地完整回归 **114 项通过，193.167 秒**；覆盖跨仓库逐字节与数值一致性、
  SHA 排序、Top-K/完整分母、协议摘要、真实小规模优化及两类中断恢复。
- 最终远程完整回归 **114 项执行，113 项通过、1 项跳过，366.199 秒**。
  跳过项依赖本地相邻 `compared_methods` 路径，已另行对实际远程 `wbm-matching`
  完成逐字节核对。初次目录检查发现的旧字节码目录已保留归档并通过复验。
- 远程原 NPZ、8 份冻结协议和指标引擎逐字节一致。原 NPZ SHA-256 为
  `a19e791c85ccd5051c080169a3d2b17902a42545cf749e9869f6c96312bcdc69`。
- 本地审计证据：`artifacts/metric_migration_validation_20261109/`。
- 远程正式目录：
  `/data/wangkaiyan2024/wbm-region-retrieval/artifacts/formal_f_exact_20261109_132219/`。
  `run/` 保存缓存/训练/评价；`status.json`、各 stage 日志记录执行状态；
  `selection_before_test.json` 在 test 前绑定 best_nce checkpoint 的哈希；
  `verification.json` 仅在完整 test 和独立指标复核成功后生成。

## coverage_v1 正式 test 结果（论文对比表）

coverage_v1 与 exact_v1 使用同一份去重数据和固定检索协议：7,199 个 test query，
每个 query 的完整候选库为排除自身后的 7,198 张图，排名保存到 Top-500；共有 37 个完整
multi-hot 组合类别。表中每个单元格按“@10 / @20”排列；Recall 另报告 @500 以反映保存的
完整 Top-500 范围。所有指标均为 query 平均，完整覆盖 Recall/AP 的分母来自完整固定 gallery。

| 方法 | Coverage-nDCG@10 / @20 | MeanCoverage@10 / @20 | FullCoverage mAP@10 / @20 | FullCoverage Recall@10 / @20 | FullCoverage Recall@500 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wu | 0.708569 / 0.694368 | 0.701086 / 0.684555 | 0.004470 / 0.007210 | 0.006329 / 0.011491 | 0.147566 |
| Yu | 0.736098 / 0.686443 | 0.720576 / 0.657216 | 0.007420 / 0.011173 | 0.008972 / 0.014336 | 0.049693 |
| WBBS | 0.878320 / 0.867206 | 0.872929 / 0.859790 | 0.010533 / 0.019018 | 0.012464 / 0.023673 | 0.286166 |
| Nakazawa CNN | 0.959434 / 0.957507 | 0.958760 / 0.956364 | 0.016765 / 0.033020 | 0.017355 / 0.034456 | 0.434949 |
| WMHD | 0.099097 / 0.123000 | 0.106394 / 0.136892 | 0.001922 / 0.003717 | 0.002423 / 0.005311 | 0.195214 |
| Proposed F (`best_nce`) | 0.747610 / 0.733125 | 0.739647 / 0.722990 | 0.006533 / 0.011215 | 0.008611 / 0.016033 | 0.192870 |
| WaPIRL（F 对齐整图，`best_nce`） | 0.728634 / 0.711760 | 0.719899 / 0.700227 | 0.006848 / 0.012078 | 0.008887 / 0.016786 | 0.192725 |

F 使用预先固定的 `best_nce` checkpoint；原五个 compared_methods 行使用各自正式 test
排名。F 与原五方法的 coverage 结果均对已完成的 Top-500 排名做指标后处理，没有修改原有
exact_v1 产物。2026-11-11 新增 WaPIRL：独立完成100轮整图训练，按最低 valid NCE
选出第100轮 `best_nce`，统一评价入口对同一份排名同时输出 exact_v1 和 coverage_v1。
所有方法均不根据 test 结果选择 checkpoint。结果来源如下：

- F：`artifacts/formal_f_coverage_v1_test_20261109_194139/F_best_nce/`，排名来源为
  `artifacts/formal_f_exact_20261109_132219/run/seed_2026/F/test/best_nce/`。
- compared_methods：`artifacts/coverage_v1_test_20261109_191909/`，五个方法的排名来源和
  SHA-256 均记录在各自的 `evaluation_manifest.json` 中。

- WaPIRL：远程 `/data/wangkaiyan2024/compared_methods/artifacts/wapirl_test_v1/test/`；
  [本地 coverage 原始报告](../../compared_methods/artifacts/wapirl_results_20261111/wapirl_test_v1/test/coverage_metrics.json)、
  [exact 原始报告](../../compared_methods/artifacts/wapirl_results_20261111/wapirl_test_v1/test/metrics.json)、
  [训练与验证记录](../../compared_methods/docs/WAPIRL_RESULTS_20261111.md)。
  7,199 个 test query、每query 7,198个候选及两类逐query指标已独立复核，smoke 不进入本表。

该表是当前固定协议下的 test 结果汇总，不代表不同训练 seed 的方差；`coverage_v1` 的
部分覆盖阈值 `τ` 尚未作为额外二值任务启用。exact_v1 仍作为完全匹配参考，不能与本表的
FullCoverage 指标混为同一相关性定义。

F、WaPIRL 和原五个 compared_methods 均已完成同一固定协议下的正式 test。
表中 coverage 结果不包含 smoke 或“排除同组合候选”的跨组合派生实验。


## WaPIRL 与 F、其他五方法的差距（2026-11-11）

以下基于原始JSON未舍入数值计算差值；“百分点”指比例值之差乘100，不是相对提升率。
这是seed=2026下的已观察结果，尚未进行跨seed重复或配对显著性检验。

### 与 F：分级覆盖略低，完整覆盖 mAP 略高

| 指标 | F | WaPIRL | WaPIRL − F（百分点） |
| --- | ---: | ---: | ---: |
| Coverage-nDCG@10 | 0.747610 | 0.728634 | −1.8976 |
| Coverage-nDCG@20 | 0.733125 | 0.711760 | −2.1365 |
| MeanCoverage@10 | 0.739647 | 0.719899 | −1.9748 |
| MeanCoverage@20 | 0.722990 | 0.700227 | −2.2764 |
| FullCoverage mAP@10 | 0.006533 | 0.006848 | +0.0316 |
| FullCoverage mAP@20 | 0.011215 | 0.012078 | +0.0863 |
| FullCoverage Recall@500 | 0.192870 | 0.192725 | −0.0145 |

F 在前10/20名的整体覆盖程度及排序上更好，但优势约2个百分点；WaPIRL 的完整覆盖
mAP@10/@20及Recall@10/@20反而略高，Recall@500几乎相同。因此，当前结果不支持
“区域方法在所有指标上优于整图方法”。分级coverage会给部分覆盖计分，FullCoverage只把
覆盖全部查询类别的候选计为相关，二者还受到排名位置及逐query相关数归一化的影响，
出现相反方向的差异并不矛盾。

作为另一种相关性定义的补充，exact_v1 的对比为：

| exact 指标 | F | WaPIRL | WaPIRL − F（百分点） |
| --- | ---: | ---: | ---: |
| Precision@10 | 0.326031 | 0.345617 | +1.9586 |
| 截断mAP@10 | 0.011679 | 0.013049 | +0.1370 |
| Hit@10 | 0.901931 | 0.892485 | −0.9446 |

这说明 WaPIRL 的前10名平均包含更多“完整组合相同”的图，但至少命中一张的query比例
略低于F；不能将exact mAP与上表FullCoverage mAP当作同一个指标。
F exact数值来自上述 `formal_f_exact_20261109_132219/.../test/best_nce/metrics.json`。

### F 的覆盖优势集中在哪里

下面按query真实标签数量分组，仅用于事后评价，不用于方法打分或模型选择。
数值来自F和WaPIRL各自 `coverage_metrics.json` 的 `groups.by_label_count`。

| query缺陷类别数 | query数 | F Coverage-nDCG@10 | WaPIRL Coverage-nDCG@10 | WaPIRL − F（百分点） |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1,399 | 0.817525 | 0.791158 | −2.6366 |
| 2 | 2,600 | 0.747948 | 0.704708 | −4.3241 |
| 3 | 2,400 | 0.722249 | 0.723638 | +0.1389 |
| 4 | 800 | 0.700334 | 0.712047 | +1.1714 |

混合缺陷整体（2–4类，共5,800个query）的Coverage-nDCG@10为F 0.730746、WaPIRL
0.713553。F的总体优势主要来自单缺陷及双缺陷query；三、四缺陷query上没有观察到
F优于WaPIRL。因而目前不能声称区域划分对越复杂的混合缺陷越有效。
这值得后续在train/valid上审计区域遗漏、粘连、碎裂和窗口截断，但此表本身不能确定原因，
也不能据这次test结果直接选择区域处理参数。

### 与其他五个方法

| 对比对象 | WaPIRL − 对方 Coverage-nDCG@10 / @20（百分点） | 其他差距与结论 |
| --- | ---: | --- |
| Wu | +2.0065 / +1.7392 | 当前表内全部指标均高于Wu；FullCoverage Recall@500高4.5159个百分点。 |
| Yu | −0.7463 / +2.5316 | @10各项略低于Yu；@20各项反超。MeanCoverage@20高4.3011个百分点，Recall@500高14.3032个百分点，差异随K变化。 |
| WBBS | −14.9686 / −15.5446 | 当前表内全部指标均低于WBBS；MeanCoverage@10低15.3031个百分点，Recall@500低9.3441个百分点。 |
| Nakazawa CNN | −23.0800 / −24.5747 | 当前表内全部指标均低于Nakazawa；MeanCoverage@10低23.8862个百分点，Recall@500低24.2223个百分点。 |
| WMHD | +62.9538 / +58.8760 | @10/@20各项均高于WMHD，但Recall@500反而低0.2489个百分点，不能说所有K和指标全面领先。 |

按Coverage-nDCG@10排序，七方法为：Nakazawa > WBBS > F > Yu > WaPIRL > Wu > WMHD；
@20则为：Nakazawa > WBBS > F > WaPIRL > Wu > Yu > WMHD。
WaPIRL位于中间偏后，F与它的差距远小于二者与WBBS、Nakazawa之间的覆盖排序差距。

### 如何解释这些差距

- **监督信息不同。** Yu和Nakazawa使用37类组合标签训练分类器；WaPIRL使用无标签NCE。
  Nakazawa领先是当前基准的事实，但不能仅据此认定其网络结构优于WaPIRL，或将差距全部归因于监督。
  WBBS不需要标签训练且同样领先，说明在当前数据与评价下，仍存在比这套整图自监督表示更强的无训练对比。
- **F与WaPIRL不是只改变区域划分的严格单因素消融。** 二者模型、旋转增强和主要训练超参数对齐，
  但F使用区域质心窗口及目标区域通道、区域匹配和面积加权；WaPIRL使用完整整图、重复的缺陷通道和
  整图余弦。F实际39,000次更新，WaPIRL 9,900次；相同100轮不等于相同计算预算。
- **机制解释仍是假设。** F的query方向区域覆盖得分与coverage目标更接近，整图余弦为对称分数，
  这可能与F在分级覆盖上的小幅优势有关；但现有单次结果不能把收益单独归因于匹配方向或区域划分。
  更稳妥的当前结论是：F改善了部分覆盖排序，但未在完整覆盖检索上形成一致优势，
  也未解决三、四类混合query相对整图方法的优势问题。
