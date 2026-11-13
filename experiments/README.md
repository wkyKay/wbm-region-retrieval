# 实验总览

更新日期：2026-11-09。本文件统一说明现有实验的设计、改进目的、运行位置和比较边界。按 `experiments/<方法>/<实验>/` 查找代码与配置；此处的字母是实验编号，不是训练标签预算。正式配置现已迁移到与 `compared_methods` 相同的去重划分与固定检索池。历史结果继续留在原 `artifacts/` 目录，新协议默认写入 `artifacts/deduplicated_exact_v1/`。

## 1. 目录与使用约定

```text
experiments/
├── README.md                       # 本文件：所有实验的设计与目的
├── catalog.json                    # 运行入口注册表
├── _shared/                        # 入口调度、基础配置、主实验CLI及源码指纹
├── proposed/
│   ├── main/                       # 旧主实验；不是消融 A
│   │   ├── run.py / config.json / config_mvp.json
│   │   └── method.py
│   ├── a/ ... f/                   # 每组都有 run.py + config.json
│   │   └── d/run_ef.py + config_ef.json  # D 在 E/F 套件中的对照入口
│   ├── g/                          # run.py、config.json、matching/suite/training.py
│   └── _shared/                    # A～D 与 E/F 各自共用的 suite/training
├── handcrafted/
│   ├── a/                          # run.py、config.json、descriptor.py
│   └── b/                          # run.py、config.json；复用 G 的计数匹配器
├── supervised/
│   ├── a/                          # run.py、config.json、method.py
│   └── b/                          # run.py、config.json、config_mvp.json
│       └── suite/training/evaluation/figures.py
└── _archive/                       # 原详细说明、运行命令和本地验证记录
```

每个 `run.py` 绑定本目录实验，例如 `proposed/a/run.py` 只运行 A。A～D、E/F 的配置仍保留各自套件的完整对照定义，供原有校验器核验；并不意味着单组入口会训练配置中的全部变体。各组配置可独立保存，但改变科学设置后必须使用新的输出目录，且原有实验定义校验仍然有效。

共用算法集中于 `_shared/`，不复制到每个字母目录。所有内部导入已使用 `experiments.proposed.*`、`experiments.handcrafted.*`、`experiments.supervised.*` 的真实模块路径，**旧兼容链接、旧实验入口及顶层 `configs/` 已删除**。`wbm_retrieval/` 仅保留共用模型、数据、匹配、指标和诊断工具。

共用基础配置位于 `experiments/_shared/base.json`；MVP 配置位于 `proposed/main/config_mvp.json`。正式配置的 `data_protocol` 为 `deduplicated_v1`，历史 MVP 显式为 `legacy`。修改某个实验请编辑它自己的配置；需要改变继承的数据路径时，复制基础配置并修改该实验的 `base_config`。可以直接导入新位置的实现模块，不再依赖旧模块名或符号链接。

移除兼容层会改变导入路径和源码指纹。历史 `artifacts/`、模型、协议及人工记录保留，但新代码**不能直接续跑旧签名目录**，也不会篡改签名来绕过检查。新运行使用新的输出根；消融/监督B通过 `--output-root` 指定，原主实验方法在各自配置副本中修改 `output_dir`。已有结果需要原版本续跑时，应使用产生该结果的代码版本。

## 2. 共同数据、训练和评价

- 数据：Mixed-WM38 原始38,015条记录按完整0/1/2/3原图精确去重，保留最小原始ID，过滤1,024条冗余记录；同图标签冲突报错。35,991张唯一缺陷图按37种完整 multi-hot 组合、seed=2026冻结为 train/valid/test = **25,193/3,599/7,199**。本仓库 `protocol/deduplicated/` 和 `protocol/retrieval_v1/` 是 `compared_methods` 冻结文件的逐字节镜像，不改原始NPZ，不重新编号，也不在运行时重新划分。详细规则见[协议说明](../protocol/README.md)。
- 局部方法：原始缺陷像素 `raw==2` 的 8 连通域，`min_area=5`、每图最多 top4，不使用学习分割器。模型训练单位是区域，监督参考的训练单位是整图或整图 pair。
- 编码器：完整 ResNet-18 `[2,2,2,2]`，原始 512 维 GAP；自监督训练使用 128 维线性 projector，检索使用 h512。除 F 为三通道外均为单通道。
- 默认优化：100 轮、batch256、SGD lr=0.01、momentum=0.9、weight_decay=0.001、WaPIRL cosine。自监督温度 0.07、memory EMA=0.5；A～F 为 5,000 个有放回负样本，G 为 1,000 个。
- 区域对分数：proposed 为截断到 `[0,1]` 的 h512 cosine；handcrafted 为 Zernike 距离的 `exp(-d/scale)`。分数降序贪心一对一匹配，每个区域最多参与一次，不是最大权匹配。
- 除 G/handcrafted B 外，局部分数为 `sum(sqrt(query_area) * pair_score) / sum(sqrt(query_area))`。空图得0，候选额外区域不直接扣分；正式整图同分使用下述 SHA 排序，历史 MVP 仍按 candidate ID 升序。
- 正式评价：完整标签组合相同定义正样本；报告 K=1/5/10/20/50/100/500 的 Hit、Precision、Recall、mAP，区分 query 均值、single/mixed 与类别宏平均。AP/Recall 使用完整候选库的相关数作分母。同分按 compared_methods 的 SHA 规则排序。详见 [统一评价说明](../doc/EXACT_METRICS_MIGRATION.md)。历史 MVP 仍使用 containment。
- 固定检索池：valid 为分层冻结的500个 query，每个检索完整3,599张 valid 除自身的3,598张；test 为全部7,199张轮流查询，每个检索其余7,198张。query/gallery 不随训练seed、训练负样本数或实验组别改变，候选逐query惰性生成。

正式配置已统一数据身份、完整候选库、exact_v1 评价公式和哈希破并列规则；旧 containment 产物保留用于历史复现。历史模型使用了不同训练划分，且旧train与新valid/test有交叉，必须在新train重新训练，再进行新协议评价。

## 3. Proposed：原主实验与 A～G

| 目录 | 实验设计 | 改进目的 / 对照解释 |
|---|---|---|
| [main](proposed/main/) | 区域 bbox 补方＋原网格每边 padding4，再 resize96；stride2；增强为带前景保护的轻度开运算＋直角旋转 | 原始局部自监督主方法，检验区域表示能否替代手工描述子；开运算用于减弱边界毛刺，可能改变细线/环形，不保证一定有效 |
| [a](proposed/a/) | 同 main 的标准输入和 stride2，但关闭开运算，仅旋转；5,000负样本 | 尺度/增强/stride 消融的基线；A 与 main 不等同 |
| [b](proposed/b/) | bbox 补方后 resize88，再在最终画布每边 padding4；stride2，仅旋转 | B−A：统一形状长边占比与最终边距，检验原始区域大小造成的输入差异 |
| [c](proposed/c/) | 原始输入同 B；增强从原始区域 mask 生成80～88尺寸、每轴平移−2～2，再旋转；stride2 | C−B：检验轻度尺度与平移增强的联合作用，不截断前景 |
| [d](proposed/d/) | 同 B 输入，仅旋转；首层 stride 从2改为1，保留 maxpool 和完整残差块 | D−B：减弱早期下采样，保留局部形状细节；可分别作为 A～D、E/F 套件中的 D 对照 |
| [e](proposed/e/) | D 的 stride1 编码器＋C 的尺度/平移/旋转增强 | E−D：stride1 下的增强收益；E−C：同增强下的 stride 收益，补全2×2对照 |
| [f](proposed/f/) | 以区域质心为中心截取52×52固定窗口，三通道为全部缺陷/目标区域/有效晶圆，越界补0后 resize96；stride1，仅旋转 | F−D：联合检验原网格尺度、周围缺陷及晶圆边界信息；不能单独归因于三通道；另输出窗口截断审计 |
| [g](proposed/g/) | D 输入与完整编码器；负样本降至1,000；最终得分改为匹配对数/query保留区域数；两种方法默认零阈值 | G−D：负样本数与聚合规则的联合影响，不能单独归因；用于检验按区域数量衡量覆盖的方案 |

F 不使用 bbox 补方或88＋4输入。它可能截断偏心区域，报告 `F_window_audit`。F0（相同固定窗口但只用目标区域通道）仅是曾提出的后续对照，**尚未实现**。旧“轻量化 G”已被当前 G 替换，不应混作同一个实验。

2026-11-11 新增 [F 分割对照](proposed/f_segmentation/README.md)：独立运行 `cc8`、`cc4`、
`median8`、`opening8`、`tv8`、`npfc_pre`，固定原 F 的 Top-4、面积阈值、输入和匹配，
每组独立训练及保存产物。`python -u experiments/proposed/f_segmentation/run.py --device cuda --prepare-workers 4`
会顺序完成六组训练、valid、完整test；默认 `best_nce`，与原 F 入口的 `all` 范围不同。

2026-11-13：[六组径向＋相对面积评价](proposed/f_segmentation_geometry/README.md)
读取上述六组的已完成模型，固定16-bin径向β=1、面积σ=1.5及权重0.15，
重新评分valid/test并同时输出exact/coverage指标；独立目录保存，不修改旧结果或训练签名。

**G 已知局限：** 当所有区域对分数都大于0时，零阈值计数退化为 `min(|Q|,|C|)/|Q|`，丢失相似度强弱。2026-10-27 用户提供的远程汇总中，random、best_nce、best_retrieval 与 handcrafted 的三分组全部22项指标一致，与此退化机制吻合；不能据此证明编码器没有学习，也不能证明1,000负样本导致退化。此处记录观察，不替代原始远程日志审计；本次目录整理没有改动该评分算法。

## 4. Handcrafted：两种聚合基线

| 目录 | 实验设计 | 目的 / 比较对象 |
|---|---|---|
| [a](handcrafted/a/) | 完整区域 mask 的质心/尺度归一化 Zernike 幅值，最高阶12、47维；距离转分数后按 query 的 sqrt(area) 加权 | 无训练的形状基线，与 proposed main、A～F 的区域匹配比较；不再使用早期 Hu/直方图拼接 |
| [b](handcrafted/b/) | 同一 Zernike 描述子，改用匹配对数/query区域数；阈值来自 G 配置 | 与 proposed G 对齐聚合规则，区分手工特征和学习特征；同样存在零阈值计数退化 |

两个入口都不会启动神经网络训练。A 默认与当前主实验共用新协议缓存/输出根；B 默认与 G 套件共用新协议缓存/输出根，以保证 query、candidate 和区域完全一致。A～F 的 valid 手工对照由各自套件入口自动生成，不应将旧协议主实验的 test 指标直接作为这些 valid 指标的对照。

## 5. Supervised：两种整图监督参考

| 目录 | 实验设计 | 目的 / 与主方法的区别 |
|---|---|---|
| [a](supervised/a/) | 整图 ResNet-18 stride2；有序特征 `[zQ,zC,zQ*zC,zQ-zC]` 输入 MLP，MSE 回归真实标签包含率 | 检验直接学习有方向整图匹配函数的效果；依赖有标签 pair，不是区域自监督，也不是保证优于主方法的上界 |
| [b](supervised/b/) | 同 main 的整图 stride2 编码器＋8维线性分类头；BCE 训练多标签分类；sigmoid≥0.5转硬标签，按预测标签包含率排序 | 检验先预测缺陷标签、再按包含关系检索；直接使用预测标签，不用真实 test 标签参与打分 |

两种监督实验均有 **5%、10%、50% train 标签预算**，预算不是新增字母实验。按完整标签组合分层形成嵌套子集，每组合至少一张，实际使用数量写入输出；valid 标签另外使用，不算在 train 预算内。

- A：每轮采样50,000个训练有序 pair，固定10,000个 valid pair，按最低 valid 回归损失保存 `best.pt`。
- B：每轮遍历预算内整图一次；每轮完整 valid 分类，按 **Macro-F1最大 → BCE最小 → 更早轮次** 保存 `best_classifier.pt`。保留 `best_loss.pt` 和 `last.pt`；固定 valid 检索池只监控、不选模。预测空标签时检索得0，并记录空预测与同分审计。
- B 同时输出全 valid/test 分类指标、固定池检索指标、学习曲线和分类图表。分类 Macro-AP 与检索 mAP_full 是不同指标。
- 两者编码器与 proposed main 结构对齐，**不是 G 的 stride1 编码器**；预算不同也意味着优化步数不同。不能仅凭同架构声称训练工作量或监督信息完全相同。

## 6. 选模规则：区分历史实现与无监督要求

| 流程 | 实际选模规则 | 是否使用 valid 类别标签 |
|---|---|---|
| proposed main | 全部 valid 区域最低 NCE，`best.pt` | 否 |
| proposed A～G 的 `best_nce` | 全部 valid 保留区域最低 NCE；负样本数量与各自训练一致 | 否 |
| proposed A～G 的 `best_retrieval` | 固定500个 valid query × 各3,598个候选的 mAP@10 最大，并列保留早轮次 | **是** |
| supervised A | 固定 valid pair 的最低回归损失 | 是 |
| supervised B | valid Macro-F1、BCE、轮次的既定规则 | 是 |

区域 NCE 验证每轮遍历全部 valid 保留区域，不是只使用500张图；检索验证在第1、5、10、…、100轮额外运行。验证不反向传播、不更新 memory。A～G 保存两个最佳 checkpoint；正式 test 默认 `best_nce`，辅助 `best_retrieval` 监测指标改为 mAP@10。数据划分与检索池采用新协议；旧1,000候选验证成绩仅供历史追溯。

**按当前要求，完整无监督主结果应预先指定 `--test-checkpoint best_nce`。** `best_retrieval` 只能另列为使用验证标签选模的对照；即使不参与梯度更新，也不是完整无监督选模。若要求整个开发/选模阶段都不接触类别标签，还需后续修改协议，关闭标签检索监控及其对超参数决策的影响；只换 test 参数不代表已完成这一改造。test 标签只在固定方案后交给评价器，不能据 test 再选 checkpoint。

## 7. 运行示例

以下命令从项目根目录执行；单组入口也支持从其他目录用绝对脚本路径启动。先确认 `--dry-run` 展开的数据路径、输出根、变体和设置。`--config` 中的相对路径按项目根目录解释。

```bash
# proposed A～G：将 g 换成 a/b/c/d/e/f 即切换实验
python experiments/proposed/g/run.py --dry-run
python -u experiments/proposed/g/run.py --stage all --device cuda --output-root artifacts/deduplicated_exact_v1/proposed_g
python -u experiments/proposed/g/run.py --stage all --device cuda --resume --output-root artifacts/deduplicated_exact_v1/proposed_g
python -u experiments/proposed/g/run.py --stage test --test-checkpoint best_nce --device cuda --output-root artifacts/deduplicated_exact_v1/proposed_g

# 在 E/F 套件中运行 D 对照
python experiments/proposed/d/run_ef.py --stage all --device cuda

# 手工基线：A 的 all 计算原主实验 test；B 的 all 计算 G 的 valid
python experiments/handcrafted/a/run.py --dry-run
python experiments/handcrafted/a/run.py --stage all
python experiments/handcrafted/b/run.py --stage all
python experiments/handcrafted/b/run.py --stage test

# 旧 proposed 主实验，或旧整图回归；仅训练本目录方法
python experiments/proposed/main/run.py --stage all --device cuda
python experiments/supervised/a/run.py --stage all --device cuda

# 新多标签监督实验：all 为 train+valid，test 显式执行
python experiments/supervised/b/run.py --dry-run
python -u experiments/supervised/b/run.py --stage all --device cuda
python -u experiments/supervised/b/run.py --stage test --device cuda
python experiments/supervised/b/run.py --stage plot --device cpu
```

注意各入口的既有语义：**A～G、handcrafted B、supervised B 的 `all` 不评价 test；proposed main、handcrafted A、supervised A 的 `all` 沿用原主实验流程，会评价 test。** 后三者支持 `--stage prepare/train/retrieve/evaluate/test`（handcrafted 没有 train）；需要先做训练再决定最终评价时，用 prepare、train 分阶段运行。它们的 `test` 仅执行检索和评价，不重新训练。

正式默认输出如下；旧目录原样保留，新目录需要从头训练，不能用旧checkpoint恢复。

| 实验 | 新协议默认输出 |
| --- | --- |
| main / handcrafted A / supervised A | `artifacts/deduplicated_exact_v1/experiment_a_top4_opening_seed2026` |
| A～D | `artifacts/deduplicated_exact_v1/ablation_abcd_20261019` |
| D(EF) / E / F | `artifacts/deduplicated_exact_v1/ablation_ef_20261020` |
| G / handcrafted B | `artifacts/deduplicated_exact_v1/ablation_g_20261025` |
| supervised B | `artifacts/deduplicated_exact_v1/supervised_multilabel_seed2026` |

同一输出根不能并行写入；各组科学设置发生变化时，消融/多标签入口使用新的 `--output-root`，原主实验方法修改配置中的 `output_dir`。E/F 的 `--reference-abcd-root` 只能引用同一去重协议下重新完成的 D，例如 `artifacts/deduplicated_exact_v1/ablation_abcd_20261019`，不能引用旧 D。监督 B 默认 `budget_source=null`，按新train建立预算；如显式引用预算来源，同样必须来自新协议。

单组入口一次只运行一个字母。批量运行可使用 `python experiments/proposed/_shared/abcd_cli.py --variants A B C D`、`python experiments/proposed/_shared/ef_cli.py --variants E F`。共享实现不重复训练其他实验；各自套件仍会计算需要的手工对照。

## 8. 已有补充实验与诊断

| 实验/工具 | 设计与目的 | 入口或记录 |
|---|---|---|
| 单标签 top1 区域检索 | 只保留单标签晶圆及其最大区域；复用历史 local/best.pt，不训练；用同类相关性分析局部形状区分能力 | `scripts/evaluate_single_label.py`；[历史主实验说明](_archive/MAIN_EXPERIMENT.md) |
| 局部表示诊断 | 对比 random/best/last 的 h512、z128；使用共同 valid 池，区分架构先验与训练收益 | `scripts/diagnose_local.py`；[历史主实验说明](_archive/MAIN_EXPERIMENT.md) |
| E/F 无标签冻结模型测试 | 全部 test 区域的旋转一致性、实例排名、margin、特征谱等；默认 best_nce＋随机基线，无类别标签；gallery NCE 不等于训练 memory NCE | `scripts/test_ef_unsupervised.py`；[详细协议](_archive/EF_UNSUPERVISED_TEST.md) |
| 几何响应示例 | 单独观察位置/尺度打分变化，不混入正式 shape-only 结果 | `python experiments/_shared/main_cli.py --stage geometry` |
| 300-map MVP | 小样本验证真实训练、推理、续跑和多进程；不作为正式性能结论 | `scripts/run_mvp.py`、`proposed/main/config_mvp.json`；[验证记录](_archive/VALIDATION.md) |
| 监督 B 小样本 | 复用300-map协议，三组标签预算各训练2轮，检查分类/检索/图表 | `supervised/b/config_mvp.json`；[验证记录](_archive/SUPERVISED_MULTILABEL_LOCAL_VALIDATION.md) |
| 监督学习曲线重绘 | 只读已保存的 history，补充曲线，不训练 | `scripts/plot_global_learning_curves.py` |

辅助工具保持原脚本入口，不另造 A～G 字母。`proposed/main/config_mvp.json` 用于历史 MVP 子集调试，源配置固定为 `_shared/legacy_mvp_source.json`（复制迁移前的基础配置，并显式标记 legacy），不从新的去重基础配置抽取或重新编号数据。首次构建子集仍运行 `scripts/run_mvp.py`。两个 MVP 配置显式使用 legacy 模式，supervised B 的 MVP 配置依赖已存在的300-map数据与预算ID。它们只用于历史实现调试，不能作为新协议的 smoke 或正式结果；新协议 smoke 如需新增，必须从冻结train/valid派生、使用正式valid query交集并禁止test。

旧 MVP 产物原样保留。源配置路径或代码变化后，历史签名可能不再匹配，不得修改签名强制续跑或覆盖旧产物；需要重新检查时复制 MVP 配置，将其 `output_dir` 及派生数据/协议路径一并改到新的独立目录。

## 9. 迁移与历史记录

2026-10-27 的目录整理后源码是唯一运行实现；旧 `run_experiment.py`、`scripts/run_ablation*.py`、`scripts/run_supervised_multilabel.py`、`wbm_retrieval/ablation*.py`、`wbm_retrieval/methods/`、`experiments/g/`、supervised 的旧平铺模块及顶层 `configs/` 已移除。辅助诊断脚本仍在 `scripts/`，并已切换到新模块。该次目录整理未改变训练/评分算法和数据划分。

2026-11-09 的本次迁移改变正式数据划分和检索池，保留模型与评价算法，源码和协议签名随之更新。旧协议CSV、原始NPZ、历史运行及人工记录保留。本次代码迁移未启动正式训练或完整test，下面的83项测试记录仅描述2026-10-27的目录整理，不是本次协议迁移的验证结论。

原分散文档已移到 `_archive/`，旧位置只留下导航。实验设计以本文件为统一入口，详细历史命令/记录可追溯：

- [原主实验](_archive/MAIN_EXPERIMENT.md)、[A～D](_archive/ABLATION_ABCD.md)、[A～G历史设计](_archive/ABLATION_A_G.md)。
- [G运行和验证](_archive/G_EXPERIMENT.md)、[E/F本地验证](_archive/EF_LOCAL_VALIDATION.md)。
- [监督B详细运行](_archive/SUPERVISED_MULTILABEL.md)、[监督B本地验证](_archive/SUPERVISED_MULTILABEL_LOCAL_VALIDATION.md)。

本地测试只能证明实现/迁移正确，不能证明检索性能提升。完整性能判断需要固定协议的实际训练、与自身随机初始化比较，并区分验证标签选模与无监督选模。

第一阶段（保留兼容链接）的历史验证：83项回归测试中81项在沙箱内通过，2项因 PyTorch 共享内存权限受限，在沙箱外定向重跑通过；最终入口专项6项也全部通过。迁移前后核对63个原有代码/配置/记录文件内容及A～G源码指纹，均保持一致。第一阶段报告位于 `artifacts/organization_validation_20261027/`；移除兼容层后的检查另存于 `artifacts/cleanup_validation_20261027/`。这些本地产物不纳入Git，均不代表完整数据性能测试。

第二阶段（删除旧路径）的验证：完整83项测试全部通过（含6项目录/入口检查、训练/评估与中断续跑、多进程入口），耗时223.296秒。因 PyTorch 多进程需要共享内存，本次完整测试在沙箱外运行。路径审计确认26个旧兼容/迁移路径已删除、新位置文件齐全、实验目录无符号链接，71个Python文件语法检查通过，`record/10.25.txt` 与清理前的SHA-256一致。测试日志为 `artifacts/cleanup_validation_20261027/full_tests.log`，路径审计为同目录 `path_audit.json`。
