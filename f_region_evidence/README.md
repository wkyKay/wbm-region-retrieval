# F 的区域补全与背景校准

本包在已冻结的区域 J 融合 F 上，分别验证区域遗漏和弱区域对应两个问题。当前尚未启动本轮正式 valid/test，没有新增检索成绩；实现检查不代表方法有效。实验属于查看既往结果后提出的研究，不改写历史选择过程，也不自动替换项目主方法。

公共运行与结果管理见[项目说明](../doc/项目说明.md)，正式 test 比较统一维护在[消融实验](../doc/消融实验.md)。来源配方见[区域 J 与面积融合](../f_region_area_fusion/README.md)；这里只记录本轮专属设计与入口。

## 固定区域与评分

基线为此前 valid 选出的 `add_a0p4_b0p25`：cc8、最小面积 5、Top-4、冻结 h512 `best_nce`、seed=2026、单向贪心一对一匹配，以及查询区域 `sqrt(area)` 加权聚合。区域对分数保留原运算顺序：

```text
C = clip(cos(h_query, h_candidate), 0, 1)
T = exp(-0.5 * Hellinger(J_query, J_candidate))
A = exp(-0.5 * abs(log(r_query) - log(r_candidate)) / 1.5)
S = 0.75 * (0.4*C + 0.6*T) + 0.25*A
```

`J` 是原始区域全部缺陷点的 128 维径向位置与形态联合分布，`r` 为区域原始缺陷面积除以有效晶圆面积。这里的区域结构强度为 0.5，与此前整图 `exp(-2D_J)` 不同；没有新增整图融合或最优匹配。公式在文稿中可写为 `0.30C+0.45T+0.25A`，实现不改变嵌套浮点算术。

三种区域方案都先逐像素核对原 cc8 Top-4；前面的旧区域、旧 h512 特征、区域 J、面积与径向描述符保持原顺序、原 dtype 和逐字节一致。

| 方案 | 固定规则 | 主要比较问题 |
| --- | --- | --- |
| `base4` | 完整复用原 Top-4，不增加区域 | 历史基线重放 |
| `top8` | 原 Top-4 后最多补 4 个面积至少 5 的剩余完整 cc8 区域 | Top-4 截断遗漏 |
| `residual8` | 从原 Top-4 之外的全部原始缺陷点出发，包括被最小面积规则丢弃的小片；按像素间距离平方不超过 8 的邻接图求传递闭包，保留面积至少 5 的前 4 组 | 能否用可追溯的碎片归组补充遗漏 |

新增区域均按原始面积降序、首个行优先像素索引升序排列。实际旧区域不足 4 个时，从其后接续编号；新增最多 4 个，总数不超过 8。归组只改变哪些缺陷像素属于同一区域，不填正常像素，也不将原 Top-4 像素纳入补充区域。距离不超过 `sqrt(8)` 的邻近归组思想与项目现有 NPFC-pre 有关联，不宣称原创分割算法；本轮区别在于冻结原 Top-4 后，只对其余全部缺陷点归组，并保留小片供合并。

新增区域使用原三通道固定 52×52 窗口、resize 至 96×96，由冻结 CNN 编码并采用原 NumPy L2 归一化；无重训或微调。每张原图的全部原始缺陷点只做一次 tensor voting，再按各方案的完整原始 mask 池化 J。原 Top-4 的描述符重算核对通过后仍复制历史值。新增大跨度区域可能被固定窗口截断，缓存另存原图区域面积和窗口保留率，不将完整 mask 的几何量冒充 CNN 已观察到的全部像素。

## 固定训练参考与净证据

从原冻结 train Top-4 缓存中，用 seed=2026 在非空训练晶圆间无放回抽取 4,096 张，每张再均匀抽取一个区域。该参考身份只生成一次，并在三个区域方案、valid 和 test 之间共享；不使用标签，不从 valid/test 构建参考库，也不按每个方案重新抽样。

对每个待检索区域 `i`，计算它与这 4,096 个参考区域的原始分数 `S`，用 NumPy `quantile(method='linear')` 得到预设背景分位数 `tau_i`。每个方案固定比较四项：`none`、`q75`、`q90`、`q95`。

```text
none: E_ij = S_ij
q75/q90/q95: E_ij = max((S_ij - tau_i) / (1 - tau_i), 0)
tau_i == 1: E_ij = 0
```

净证据同时完成区域间重标定和低分拒配，因此不是“只改变拒配”的纯消融。分位数表示对固定训练参考的相对位置，不是匹配正确概率或已验证的误报率。候选标签与候选图库分数分布均不参与阈值计算。

贪心匹配按 `E_ij` 降序进行，零证据不配对；并列规则保持原查询区域／候选区域索引顺序。最终仍除以查询的**全部已选区域** `sqrt(area)` 权重之和，未匹配区域贡献零，不改成只除以成功匹配区域。候选多余区域不直接扣分。补全方案会同时改变可参与匹配的区域及查询聚合分母，不能把成绩差异全部归因于像素保留率。

## valid 选择与 test 门禁

固定网格为 `3 个区域方案 × 4 个净证据设置 = 12 项 valid`，必须全部完成并通过核验，再按固定 500 个 query 的整数 Top-10 命中总数选择。命中同分依次优先原基线、较少的最大区域数、较弱的背景分位数，最后按方案名排序；不按浮点显示后的百分比打破并列。

除基线 `base4_none` 外，预设三个互斥实验族：

| 实验族 | valid 成员 | 冻结后正式 test |
| --- | --- | --- |
| 背景校准与拒配 | `base4_q75/q90/q95` | 族内 valid 胜者 1 项 |
| 区域补全 | `top8_none`、`residual8_none` | 族内 valid 胜者 1 项 |
| 区域补全＋背景校准 | `top8/residual8 × q75/q90/q95` | 族内 valid 胜者 1 项 |

唯一主方案是 **12 项中的全局 valid 胜者**，可能仍为原基线；不能改成在三项 test 中挑最高者。三个族胜者都在普通 test 和跨组合 coverage test 上评价，共六项新 test 评价，回答各方向的预设比较。原基线另生成完整分数用于历史重放，不将这份重放再次登记为新 test 方法。

普通协议保持 valid 500 query × 3,598 候选、test 7,199 query × 7,198 候选，均仅排除自身。跨组合协议只额外排除相同完整 multi-hot 组合，使用完整过滤候选库重新排名，不过滤旧 Top-500。保留原 exact/coverage 定义、K=1/5/10/20/50/100/500、总体／类别宏平均／单混合／标签数量分组；跨组合仍为原六项 coverage 指标，不增加新的覆盖等级指标。两种协议分表，test 不改权重、参考、分位数、区域规则或主方案。

## 运行入口

在仓库根目录使用原 NumPy/PyTorch 环境。正式新增区域特征默认在 CUDA 提取；CPU 选项供实现检查。为保持原分数重放，真实运行要求三个 BLAS/OpenMP 线程变量均为 1：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
python -m unittest discover -s f_region_evidence/tests -v
python -m f_region_evidence.run --dry-run
python -m f_region_evidence.run --stage valid --check-inputs
python -u -m f_region_evidence.run --stage valid --device cuda --prepare-workers 4 --workers 8
```

默认创建 `artificial/f_region_evidence_<时间>/`；实际开始时间与 run_id 以持久化 manifest／各评价身份为准。可通过 `--output-root` 指定一个全新独立目录。valid 结束后，使用该目录及已经封存的选择继续：

```bash
python -u -m f_region_evidence.run --stage test --resume-root artificial/<本轮目录> \
  --device cuda --prepare-workers 4 --workers 8
```

`--stage all` 是默认值，依次执行完整 valid、冻结选择、普通 test 与跨组合 test。`--stage test` 必须显式指定 `--resume-root`，且十二项 valid 的完成文件、选择、索引及签名一致。完整缓存可按签名复用；未完成的直接目标目录拒绝覆盖，未完成 attempt 保留用于排查，不承诺任意阶段或逐 batch 断点续跑。模型、来源、科学源码、配置或受绑定的数值环境变化需新目录。

## 产物、验证与解释边界

主要产物如下，所有路径均在本轮独立根目录内：

- `run_manifest.json`、`reference.json`、`inputs_valid.json`、`inputs_test.json`：设计、固定训练参考、两阶段来源 SHA 绑定及真实运行身份。test 特征与评价报告只在 valid 选择冻结后加载；共同原始 NPZ 的读取用于冻结数据协议核对及按 split 提取原图。
- `cache/{valid,test}/{base4,top8,residual8}/`：特征、原图 mask、完整区域描述符、`region_trace.jsonl` 与覆盖／窗口审计。trace 保存 sample_id、区域像素索引、原 cc8 成员及保留原因。
- `calibration/{valid,test}/{proposal}/`：所有真实区域的三组背景阈值、参考身份及封存证据。
- `score_cache/{valid,test}/{variant}/`：float64 全候选分数；valid 为 500×3,599，test 为 7,199×7,199。自身位置保存零但永不进入候选排名；跨组合复用完整分数。
- `valid/`、`test/`、`cross_test/`：各评价身份、Top-500 排名、逐查询／类别及聚合报告、独立核验与完成标记；`selection.json` 和 `valid_completed.json` 冻结选择，`comparison_*.csv` 汇总相应协议成绩。
- `preserved_regions_*.json`、`formula_audit_*.json`、`calibration_audit_*.json`、`baseline_replay_*.json`：旧区域全部字节核对、区域公式／贪心抽查、训练参考背景校准抽查，以及原基线全部查询 Top-500 顺序与分数重放。
- `explanations/{valid,test}/{variant}/examples.json`：预设每种完整标签组合的首个固定查询，展示其第 1、10、100 名候选的匹配、阈值与贡献，不按 test 成功案例挑图。标签只用于分组展示。

完整候选排名与指标由独立实现从已封存的完整分数缓存核验；区域公式与背景校准另做预设代表查询抽查。该核验范围不是第二遍全量独立区域公式计算。正式完成需子项与根封存、来源和科学源码复核、历史文件保护全部通过，再按项目规则登记、生成汇总并打包校验归档。

已有模型、NPZ、冻结清单、训练参考、旧区域缓存及所有历史排名／报告均只读。`base4_none` 必须重现原区域 J 配方，不因本轮实验修改原来源或阈值来通过重放。

本轮可解释的是每个区域为何按固定规则被保留或归组、哪些区域对应被背景阈值削弱／拒绝，以及它们对最终分数的贡献；不把连通区域等同于缺陷语义，不把贡献分解等同于区域必要性的因果证明。碎片可能是噪声，邻近归组可能混合不同缺陷，新增区域的固定窗口也可能丢失信息；以上风险均需根据正式结果与预设审计判断，当前不预告性能提升。
