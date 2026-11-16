# WBM 局部区域检索实验

研究 Mixed-WM38 上有方向的缺陷区域检索。当前主方法是 **F：cc8 八连通＋径向位置＋相对面积**；融合和组合保留为扩展实验。

| 入口 | 内容 |
|---|---|
| [文档目录](doc/文档目录.md) | 四篇简明主题文档与历史索引 |
| [实验设计](doc/实验设计.md) | 方法、对照与选择边界 |
| [实验结果](doc/实验结果.md) | 完成状态、结论及保存位置 |
| [数据划分](doc/数据划分.md) | 冻结去重协议、exact 与 coverage |
| [论文初稿](doc/论文初稿.md) | 与当前方法一致的简版正文 |
| [自动结果总览](results_catalog/CURRENT_RESULTS.md) | 从原报告生成的成绩和来源 |

## 代码与运行

- 当前方法：[径向位置＋相对面积入口](experiments/proposed/f_segmentation_geometry/README.md)，主实验显式指定 `--methods cc8`，复用冻结的 `best_nce`，不重训。
- 分割对照：[六种分割入口](experiments/proposed/f_segmentation/README.md)。
- 其他实验：[代码目录与入口说明](experiments/README.md)，A～G 等早期系列不等于当前主方法。
- 公共实现位于 `wbm_retrieval/`；回归测试位于 `tests/`。使用已有 PyTorch 环境；依赖见 `requirements.txt`，监督变体另见 `requirements-supervised.txt`。

默认原始数据为 `../data/wm38k/Wafer_Map_Datasets.npz`，不随 Git 发布。冻结 train/valid/test 为 25,193/3,599/7,199；valid 固定 500 query，test 全量 query，各自检索同集合全部其他图像。详细清单见 [protocol](protocol/README.md)。

## 结果与维护

当前 F、六组分割、融合及相关扩展已有完成结果；普通比较与跨组合评价分表。exact 和 coverage 评价同一排名，Top-500 不能称为完整排名。cc8 主方法身份在查看历史 test 后确认，组合实验为事后补充；保留这些事实。

新实验必须使用真实启动时间型 run_id、新输出目录并登记；现有登记器接入新时间身份的前置要求见 [AGENTS.md](AGENTS.md)。已有目录和历史命令不直接作为新实验输出模板。

本地、服务器和 GitHub 同步代码、文档及轻量汇总，数据、模型、原始排名保持原位。服务器项目为 `/data/wangkaiyan2024/wbm-region-retrieval/`。旧协议、旧 checkpoint 和历史报告只能在各自适用版本下解释，不能与当前结果混表。
