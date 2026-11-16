> 历史记录｜适用版本：旧 F 加径向 β=1 的 test，不包含相对面积项。
> 来源：artifacts/f_radial_beta1_test_20261111_v1/；登记 key old_f_radial。
> 后续状态：后续有面积方案和新的 cc8 模型；部分 report 派生文件只在本地，原 test 产物在服务器。
> 阅读当前内容：[返回实验结果](../../实验结果.md)。旧命令、未完成状态和建议均按原记录日期理解，不作为新运行指令。
> 原文证据：[整理前版本](https://github.com/wkyKay/wbm-region-retrieval/blob/35b7399cd0cd3a05f6860589a23688435804694b/doc/F_RADIAL_BETA1_TEST_20261111.md)；下文只修正引用位置，原指标与结论按历史版本保留。

# F＋径向 β=1：正式 test 与七方法对比（2026-11-11）

已完成全部7,199个query × 每query 7,198个候选，共51,818,402次有向图对评分。
固定使用valid mAP@10已选定的β=1、16-bin径向直方图及原F `best_nce`（epoch100、seed2026）。
没有重新训练、没有在test上选参、没有四旋转汇聚或双向评分。原模型、缓存和历史结果保持不变。

对照依据为 [精确评价迁移.md](精确评价迁移.md)。使用七方法完整原始JSON的未舍入值，
没有从文档舍入数值重建指标；原七方法保留，新径向F作为第八个对比条目。

## 完整报告与数据

- 完整报告：70项指标、所有分组、八方法对比与差值（仅本地产物：`wbm-region-retrieval/artifacts/f_radial_beta1_test_20261111_v1/report/report.md`；不随 Git 发布）
- 八方法全部总体指标CSV（560行）（仅本地产物：`wbm-region-retrieval/artifacts/f_radial_beta1_test_20261111_v1/report/comparison_long.csv`；不随 Git 发布）
- 八方法全部分组指标CSV（4,480行）（仅本地产物：`wbm-region-retrieval/artifacts/f_radial_beta1_test_20261111_v1/report/grouped_metrics.csv`；不随 Git 发布）
- 全部原始报告与差值JSON（仅本地产物：`wbm-region-retrieval/artifacts/f_radial_beta1_test_20261111_v1/report/report.json`；不随 Git 发布）
- 独立复核记录（本地和服务器产物：`wbm-region-retrieval/artifacts/f_radial_beta1_test_20261111_v1/independent_verification.json`；不随 Git 发布）
- 七方法参考资料来源、协议及哈希（仅本地产物：`wbm-region-retrieval/artifacts/f_radial_test_references_20261111/source_manifest.json`；不随 Git 发布）

exact_v1：完整标签组合相同，4类指标×7个K，共28项。
coverage_v1：分级类别覆盖及完整覆盖，6类指标×7个K，共42项。
K均为1、5、10、20、50、100、500；报告总体query平均、37组合类别宏平均、单/混合及1–4类缺陷分组。
两组指标使用同一份新排名；保存Top500，Recall/AP和graded IDCG仍使用完整候选库。

## 原F与径向F的主要差异

| 指标 | 原F | 径向F | 差值（百分点） |
|---|---:|---:|---:|
| exact Hit@1 | 0.395194 | 0.496319 | +10.1125 |
| exact Precision@10 | 0.326031 | 0.406154 | +8.0122 |
| exact mAP@10 | 0.011679 | 0.015339 | +0.3660 |
| exact mAP@100 | 0.046048 | 0.061550 | +1.5502 |
| Coverage-nDCG@10 | 0.747610 | 0.802887 | +5.5276 |
| MeanCoverage@10 | 0.739647 | 0.795216 | +5.5569 |
| FullCoverage mAP@10 | 0.006533 | 0.007949 | +0.1416 |
| FullCoverage Recall@500 | 0.192870 | 0.217208 | +2.4338 |

总体exact mAP@10相对提高31.3%；混合缺陷从0.008462到0.012635，相对提高49.3%。
混合缺陷Coverage-nDCG@10从0.730746到0.792367，提高6.1620个百分点。
三类和四类缺陷query的Coverage-nDCG@10分别从0.722249/0.700334提高到0.783006/0.766309。
单缺陷exact mAP@10从0.025018到0.026548，本次test也有小幅提升；不能用valid分组结果替代test结果。

不是所有指标都提高：exact Hit@50下降0.0556个百分点，Hit@100下降0.1667个百分点；
FullCoverageHit@100下降0.0556个百分点。完整报告保留了这些变化。

## 与文档中方法的关系

按总体exact mAP@10和Coverage-nDCG@10排序，径向F均列八个条目中的第三，位于Nakazawa和WBBS之后。
所列核心指标高于原F、WaPIRL、Wu、Yu和WMHD，但仍未追平WBBS：

| 方法 | exact mAP@10 | Coverage-nDCG@10 |
|---|---:|---:|
| Nakazawa CNN | 0.042395 | 0.959434 |
| WBBS | 0.024799 | 0.878320 |
| 径向F | 0.015339 | 0.802887 |
| 原F | 0.011679 | 0.747610 |
| WaPIRL | 0.013049 | 0.728634 |

这张短表按展示顺序排列，不代表两列使用同一个完整排序；七方法及全部K的比较见完整报告。
β使用valid标签选择，模型仍为原无标签训练的F，但不能称该流程全过程无监督。
这里只是一轮固定checkpoint的test，尚未估计跨训练seed方差或进行配对显著性检验。

## 校验与执行

- 18项相关数学/指标测试通过；独立复算器自测通过。
- 正式test前重放32个valid query的Top500，与β=1已选版本完全一致。
- 28,517个test区域与原图重新提取结果逐像素一致。
- 1,024个图对的独立标量与批量匹配结果在1e-12容差内一致。
- 独立流式核查3,599,500条Top500排名，候选身份、自身排除、唯一性、降序有效。
- 独立复算两组全部70项及8类聚合，共560个汇总值；最大误差4.44e-16。
- 服务器记录正式评分及两组指标计算耗时630.3秒，独立复算16.3秒。
- 12个同步的运行产物通过完成清单SHA核验；test h512和径向直方图.npy仅留远程。

本地与远程项目根下的结果目录为 `artifacts/f_radial_beta1_test_20261111_v1/`。
完整逐query、逐类别CSV、Top500排名和两组原始JSON保存在其 `test/` 下。
入口为 `scripts/evaluate_f_radial_test.py`，独立复算为 `scripts/verify_f_radial_test.py`，
报告生成器为 `scripts/report_f_radial_test.py`。远程评分与复算脚本以 `_20261111.py` 后缀保存为新文件。
`selection_before_test.json` 绑定checkpoint、valid选择、原实现哈希和保护文件快照；
`completed.json` 确认模型与保护文件未改变。复现必须使用新的输出目录，不能覆盖本轮结果。
