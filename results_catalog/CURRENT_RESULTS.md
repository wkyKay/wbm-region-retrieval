# 当前结果总览

结果核验时间：2026-11-16T18:30:30+08:00（Asia/Shanghai）。本页由 `results_catalog/build.py` 自动生成。

当前主方法为 cc8＋径向位置＋相对面积，冻结 best_nce；融合和角向／最优匹配保留为扩展实验。当前主方法身份在查看历史 test 后确认，不追述为 test 前选定。

## 协议与文件入口

冻结去重 train/valid/test 为 25,193/3,599/7,199；valid 固定 500 query × 3,598 候选，test 为 7,199 query × 7,198 候选。排除自身，Normal 不进检索库。

exact 要求完整标签组合相同；coverage 按查询标签覆盖比例，FullCoverage 允许候选额外标签。所有 AP/Recall 分母和 coverage IDCG 取对应完整候选库。保存的排名是 Top-500，mAP@K 是截断指标。

[使用说明](README.md) · [结果登记表](registry.csv) · [全部指标长表](metrics.csv) · [来源配置](sources.json) · [核验快照及来源签名](snapshot.json)

登记 57 个评价结果、28,672 条指标；模型训练运行作为来源保存在 manifest 中。未进入当前表的服务器目录也列于 snapshot.json 的 inventory，不表示目录无效或可以删除。

数值表均为 test 总体 query 平均，原始比例乘 100 展示。完整 K=1/5/10/20/50/100/500、类别宏平均、单／混合及标签数量分组保存在 CSV。监督方法与自监督／无训练方法的监督信息不同。

## 普通 test 方法比较

| 方法／变体 | exact Hit@1 | exact Precision@10 | exact mAP@500 | coverage Coverage-nDCG@10 | 登记 key |
|---|---:|---:|---:|---:|---|
| wu | 32.4212 | 26.3550 | 8.1793 | 70.8569 | `wu_test` |
| yu | 40.8529 | 35.1729 | 5.4524 | 73.6098 | `yu_test` |
| wbbs | 67.1482 | 58.1289 | 27.1098 | 87.8320 | `wbbs_test` |
| nakazawa | 86.5259 | 85.5091 | 72.6898 | 95.9434 | `nakazawa_test` |
| wmhd | 1.1113 | 1.4530 | 2.9758 | 9.9097 | `wmhd_test` |
| WaPIRL 整图适配 | 41.7141 | 34.5617 | 10.1188 | 72.8634 | `wapirl_test` |
| F cc8 径向＋面积 | 55.4938 | 46.1370 | 13.3147 | 82.4208 | `f_geometry_cc8_test` |

Nakazawa、Yu 使用监督分类训练；WaPIRL 是 F 对齐的整图适配。WMHD 数值代表当前实现和适配设置，不等于原论文结论。当前主方法在上述指标上仍低于 WBBS 和 Nakazawa。

## 六种分割对照

| 方法／变体 | exact Hit@1 | exact Precision@10 | exact mAP@500 | coverage Coverage-nDCG@10 | 登记 key |
|---|---:|---:|---:|---:|---|
| F cc8 径向＋面积 | 55.4938 | 46.1370 | 13.3147 | 82.4208 | `f_geometry_cc8_test` |
| F cc4 径向＋面积 | 52.4101 | 44.3103 | 16.3912 | 80.7408 | `f_geometry_cc4_test` |
| F median8 径向＋面积 | 49.4096 | 42.3962 | 14.8499 | 82.9053 | `f_geometry_median8_test` |
| F opening8 径向＋面积 | 41.8114 | 35.7397 | 12.7532 | 78.7463 | `f_geometry_opening8_test` |
| F tv8 径向＋面积 | 52.3545 | 43.2393 | 11.9844 | 81.6395 | `f_geometry_tv8_test` |
| F npfc_pre 径向＋面积 | 46.2842 | 38.7179 | 11.5918 | 78.9840 | `f_geometry_npfc_pre_test` |

## 融合及消融

| 方法／变体 | exact Hit@1 | exact Precision@10 | exact mAP@500 | coverage Coverage-nDCG@10 | 登记 key |
|---|---:|---:|---:|---:|---|
| F cc8 径向＋面积 | 55.4938 | 46.1370 | 13.3147 | 82.4208 | `f_geometry_cc8_test` |
| 融合 α=0.5 | 57.9525 | 48.7123 | 15.4846 | 83.2032 | `f_fusion_0.5_test` |
| 融合消融 angular_0.5 | 63.3977 | 54.2575 | 19.4941 | 85.8467 | `f_ablation_angular_0.5_test` |
| 融合消融 optimal | 58.0775 | 48.7971 | 15.6342 | 83.2057 | `f_ablation_optimal_test` |
| 融合＋角向＋最优匹配 | 63.4671 | 54.2770 | 19.6678 | 85.8737 | `f_combination_test` |

融合 α=0.5、角向 γ=0.5 由 valid exact/all/Precision@10 选择；占比约束选中零强度，等于基线。最优匹配单独报告。最后一行组合是在查看前一轮 test 后提出的补充实验，posthoc=true，没有 test 参数搜索，也不能描述为 test 前确定的方案。

组合与角向单项的差值不代表统计显著或跨种子稳定收益。valid 参数网格、所有 test 对照和分组见 [融合长表](tables/fusion.csv)。

## 跨组合 test

| 方法／变体 | coverage Coverage-nDCG@10 | coverage FullCoveragePrecision@10 | 登记 key |
|---|---:|---:|---|
| wu | 65.5681 | 24.6798 | `cross_wu` |
| yu | 60.5384 | 24.2784 | `cross_yu` |
| wbbs | 74.8829 | 35.0660 | `cross_wbbs` |
| nakazawa | 80.9360 | 43.1227 | `cross_nakazawa` |
| wmhd | 9.1945 | 6.7787 | `cross_wmhd` |
| wapirl | 60.5532 | 21.1807 | `cross_wapirl` |
| f_radial_area | 71.2707 | 31.8461 | `cross_f_radial_area` |

本表额外排除同标签组合候选，每 query 6,999–7,170 个候选。999 个 query 无 FullCoverage 正样本；仅对应 Recall/AP 平均排除，保留其余指标，Recall/AP 有效 query=6,200。七方法均已完成；F 是 2026-11-11 的旧径向面积来源，不能替代当前 cc8。与普通 test 独立展示。

## 保存位置与追溯

服务器工作区：`/data/wangkaiyan2024`。registry.csv 的 remote_path 是每项报告所在目录；snapshot.json 保存报告哈希、完成标记哈希、来源 manifest、选参记录及协议清单哈希。

当前主实验：`wbm-region-retrieval/artifacts/f_segmentation_radial_area_v1/cc8/{valid,test}/best_nce/`；模型：`wbm-region-retrieval/artifacts/f_segmentation_v1/cc8/seed_2026/F/best_nce.pt`。

原始 artifacts、archives、数据、模型、冻结协议保持原位。登记表的时间 ID 不是重命名后的实验目录。Git 管理本目录的轻量快照、汇总及生成代码，不包含模型或 Top-500 排名。

历史 run_id 时间取完成标记文件 mtime（含 +0800），明确不是实验开始时间；同步可能改变 mtime，故首次登记后保留 ID。原始时间来源在 id_time 中，登记时间另见 collected_at。

本次核验范围：指标报告与 completed.json 中 SHA-256 一致；冻结清单两仓库一致；独立 coverage 后处理与 exact 的排名哈希引用一致。没有重新训练、重新计算指标或逐一重算模型／排名哈希。
