> 历史记录｜适用版本：Zernike47 Handcrafted A 的固定 test；表中 F 是旧 shape-only F。
> 来源：artifacts/formal_handcrafted_a_test_20261111/；run_manifest.json、completed.json。
> 后续状态：手工方法成绩保留；与旧 F 的关系不能直接外推到当前主方法。
> 阅读当前内容：[返回实验结果](../../实验结果.md)。旧命令、未完成状态和建议均按原记录日期理解，不作为新运行指令。
> 原文证据：[整理前版本](https://github.com/wkyKay/wbm-region-retrieval/blob/35b7399cd0cd3a05f6860589a23688435804694b/doc/HANDCRAFTED_TEST_20261111.md)；下文只修正引用位置，原指标与结论按历史版本保留。

# Handcrafted A 正式 test（2026-11-11）

已完成完整 test 及独立逐 query 复核，无模型训练。使用现有 handcrafted A 定义，未修改算法参数。

## 方法与协议

- 原始缺陷图 `raw==2` 的8连通区域，面积≥5，按面积取 Top4。
- 完整区域质心/尺度归一化的 Zernike 幅值，最高阶12、47维；不对描述子额外做单位归一化。
- 区域对分数为 `exp(-EuclideanDistance / 1.0)`，贪心一对一匹配，按 query 区域面积平方根加权。
- 这是 handcrafted A，不是 G 中的计数聚合 B；不使用 CNN、F 三通道窗口、增强或 checkpoint。
- 只读复用 F 新协议运行中的区域/Zernike缓存，调用现有 `ef_suite.evaluate(model=None)` 批量评价实现。缓存来源、源码哈希及参数已校验，与 handcrafted A 的区域选择、描述子和匹配定义一致。
- 冻结去重 test：7,199 query / 7,199 gallery，每 query 排除自身后7,198候选；全部候选均评分，保存 Top500。
- K=1/5/10/20/50/100/500；exact_v1 和 coverage_v1 使用同一排名，统一 SHA 规则处理同分。
- CPU 正式脚本耗时约11.84分钟，包含来源检查、抽样重建、排名、指标及结束时哈希检查，不含独立复核。

## exact_v1

完整八维标签组合相同才相关，以下数值均为 query 平均。

| K | Hit | Precision | Recall | mAP |
|---:|---:|---:|---:|---:|
| 1 | 0.302125 | 0.302125 | 0.001648 | 0.001648 |
| 5 | 0.679400 | 0.265398 | 0.007274 | 0.005252 |
| 10 | 0.829282 | 0.246270 | 0.013458 | 0.008278 |
| 20 | 0.929712 | 0.223219 | 0.024198 | 0.012609 |
| 50 | 0.982498 | 0.188648 | 0.050094 | 0.020460 |
| 100 | 0.994166 | 0.161253 | 0.084368 | 0.028471 |
| 500 | 0.999861 | 0.098543 | 0.252306 | 0.053469 |

## coverage_v1

候选覆盖 query 类别的比例作为分级增益；完整覆盖要求 query 标签集合包含于 candidate。

| K | Coverage-nDCG | MeanCoverage | FullCoverage Hit | Precision | Recall | mAP |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.789543 | 0.789543 | 0.576747 | 0.576747 | 0.000976 | 0.000976 |
| 5 | 0.772086 | 0.767060 | 0.867204 | 0.538825 | 0.004432 | 0.003665 |
| 10 | 0.760610 | 0.753024 | 0.940686 | 0.517391 | 0.008303 | 0.006262 |
| 20 | 0.746450 | 0.736627 | 0.978053 | 0.491381 | 0.015062 | 0.010359 |
| 50 | 0.722077 | 0.708708 | 0.994721 | 0.450679 | 0.031723 | 0.019023 |
| 100 | 0.698646 | 0.683102 | 0.998333 | 0.416370 | 0.054578 | 0.029557 |
| 500 | 0.639131 | 0.598455 | 0.999861 | 0.321499 | 0.180688 | 0.077494 |

AP/Recall 分母始终是完整候选库中的相关候选数。表中的 mAP 为截断 mAP@K，不是 full mAP。

## 与 F、WaPIRL 对照

以下使用相同的7,199个 query 和完整候选库，均为 query 平均。F 固定 best_nce，WaPIRL 使用第100轮 best_nce。

| Metric | Handcrafted A | F | WaPIRL |
|---|---:|---:|---:|
| exact Precision@10 | 0.246270 | 0.326031 | 0.345617 |
| exact mAP@10 | 0.008278 | 0.011679 | 0.013049 |
| Coverage-nDCG@10 | 0.760610 | 0.747610 | 0.728634 |
| Coverage-nDCG@20 | 0.746450 | 0.733125 | 0.711760 |
| MeanCoverage@10 | 0.753024 | 0.739647 | 0.719899 |
| FullCoverage_mAP@10 | 0.006262 | 0.006533 | 0.006848 |

Handcrafted A 的 Coverage-nDCG@10/@20 比 F 高约1.30/1.33个百分点，MeanCoverage@10 也更高；但 exact Precision@10 比 F 低约7.98个百分点，exact 和 FullCoverage mAP@10 均低于 F。

因此，当前手工描述子在分级覆盖排序上有竞争力，但不能称其全面优于 F。与 WaPIRL 也呈现覆盖排序更高、exact Precision 和完整覆盖 mAP 更低的差异。两类相关性必须分别解释。

这是固定配置下的结果比较，不是跨 seed 显著性结论，也不据 test 修改参数。F 与 WaPIRL 来源见[统一比较记录](EXACT_METRICS_MIGRATION.md)。

## 验证与产物

- 相关评分/诊断回归5项通过。
- 从原始 NPZ 重建128张 test 图，区域和描述子与缓存逐元素一致；2,048个图对的原逐对实现与批量实现完全一致，最大误差0。
- 独立复算7,199个 query、3,599,500条 Top500 排名：两类各K逐 query 指标及总体均值一致，自身排除、候选归属、同分排序及报告哈希通过。
- 缓存、F 权重和依赖代码的 SHA256 与 mtime 前后不变；输出到独立目录，未覆盖旧运行。
- 下载归档 SHA256 为 `f36ca6be015bb18273f6d256ff09c034c9d17d2219dba256cbdf12fa790f3f83`，与服务器一致；本地解包后再次核验全部报告哈希。

远程输出：`172.18.36.192:/data/wangkaiyan2024/wbm-region-retrieval/artifacts/formal_handcrafted_a_test_20261111/`。

来源缓存：`artifacts/formal_f_exact_20261109_132219/run/cache/`，只读取区域、标签和 mask，不使用 F 学习特征。本地完整结果已同步到同名 artifacts 目录。

- exact 指标（本地和服务器产物：`wbm-region-retrieval/artifacts/formal_handcrafted_a_test_20261111/test/metrics.json`；不随 Git 发布）
- coverage 指标（本地和服务器产物：`wbm-region-retrieval/artifacts/formal_handcrafted_a_test_20261111/test/coverage_metrics.json`；不随 Git 发布）
- 完成标记（本地和服务器产物：`wbm-region-retrieval/artifacts/formal_handcrafted_a_test_20261111/completed.json`；不随 Git 发布）
- 独立复核（本地和服务器产物：`wbm-region-retrieval/artifacts/formal_handcrafted_a_test_20261111/independent_verification.json`；不随 Git 发布）
- 运行来源及参数（本地和服务器产物：`wbm-region-retrieval/artifacts/formal_handcrafted_a_test_20261111/run_manifest.json`；不随 Git 发布）
- 执行脚本（本地和服务器产物：`wbm-region-retrieval/artifacts/formal_handcrafted_a_test_20261111/run_test.py`；不随 Git 发布）
- 独立复核脚本（本地和服务器产物：`wbm-region-retrieval/artifacts/formal_handcrafted_a_test_20261111/verify_test.py`；不随 Git 发布）
