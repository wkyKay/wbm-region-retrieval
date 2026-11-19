# F、G、WBBS 网格搜索

独立代码包。31项筛选和默认166项完整valid网格均已完成；当前完整网格选中原分数加法`0.1F+0.4G+0.5W`，未运行test，完成证据见下文。科学设计与公式见[消融实验](../doc/消融实验.md#fgwbbs网格搜索)，公共协议和结果管理见[项目说明](../doc/项目说明.md)。旧 `fgw_fusion/`、模型、冻结协议与历史结果保持不变。

## 配置与选择

默认配置为 [config.json](config.json)。F 使用此前 FGW 的冻结 cc8 **最优区域匹配**分数，G 为冻结 WaPIRL 分数，W 为原 WBBS 相似度；不含角向 A 或 J，不训练或重新选 checkpoint。

| 配置 | 默认值与约束 |
| --- | --- |
| `denominator` | 10；整数 1–20，权重以 `1/denominator` 为步长 |
| `min_f_ticks` | 1；整数 1–denominator，F 权重／指数严格大于 0 |
| `families` | `sum_raw`、`sum_minmax`、`weighted_product`，可选非空子集 |
| `include_reference` | true；额外评价固定旧公式 `(0.5F+0.5G)*W`，参与全局选择 |
| `weight_ticks` | null；默认完整网格；非空时只取明确列出的唯一整数三元组，仍须满足总和及F下限 |

三项 ticks 之和等于 denominator；G、W 允许为 0。默认每族 55 组，共 165 组，加固定参考合计 **166 项 valid 评价**。参考的 `[0.5,0.5,1]` 表示其内层 F/G 系数和外层 W 指数，不属于权重和为 1 的网格。

`sum_minmax` 在每个 query 的**完整候选库**分别归一化三项，常数分支置 0。纯 F 端点 `[1,0,0]` 在所有族均逐值返回原始 F，保留可核对基线；这与 min-max F 排名等价。乘法省略指数为 0 的分支，活跃分支为 0 则结果为 0，不加 epsilon。

唯一选参指标为 valid 的 `exact/all/Precision@10`，比较所有 query 的 Top-10 **整数总命中数**，避免浮点均值造成假同分。并列依次取活跃分支较少、族优先级 `sum_raw → weighted_product → sum_minmax → legacy_fgw`、F 较大、W 较小、变体 ID 字典序较小。mAP@100、Coverage-nDCG@10 只作描述，不参与选择。`selection.json` 保存全局胜者、每族胜者、完整排名及规则；test 只接受全局胜者。

除上表五项外，K、选参 split、指标、同分规则等科学设置冻结，不允许用配置覆盖。重复／等价端点不是独立重复实验。“最优”仅指该有限网格及所含参考中的最优，不表示全局连续权重最优。方案在看过历史 test 后提出，标记为事后研究。

## 两小时内的31项候选

[config_valid_2h.json](config_valid_2h.json)对三族各指定同一10组`(F,G,W)`权重，另含原固定FGW参考，共31项：

```text
(0.5,0.5,0.0)  (0.5,0.0,0.5)  (0.1,0.0,0.9)
(0.3,0.3,0.4)  (0.2,0.2,0.6)  (0.1,0.1,0.8)
(0.2,0.1,0.7)  (0.1,0.2,0.7)  (0.4,0.2,0.4)  (0.2,0.4,0.4)
```

仅依据既有valid：WBBS优于单独F/G融合，而加入F/G又能提高WBBS，因此重点覆盖W主导、F/G对称与不对称、G为0以及不含W的融合端点。候选清单在运行前固定，不根据新结果临时增删；F始终大于0。三族共享权重点，以便比较算子与归一化，沿用唯一的valid P@10选择规则。

这31项是预算约束下的候选搜索，不能保证覆盖被排除权重中的更优点。使用原500个valid query和每query全部3598个候选，不缩小数据协议。首次执行的命令为：

```bash
timeout --signal=TERM --kill-after=30s 6570s \
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u -m fgw_grid_search.run --stage search --config fgw_grid_search/config_valid_2h.json --workers 8
```

外层timeout控制本次父进程及worker进程组，最迟110分钟终止，预留10分钟核对与整理。只有31项全部封存、根验证通过才能冻结胜者并登记；超时部分结果保留，不能冒充完整搜索、不能删减原清单后补造完成标记。本轮31项已全部完成，实际墙钟266.86秒，退出码0；test未运行。

## 入口与恢复

在仓库根目录的原 PyTorch/CUDA 环境执行。只查看计划或只读检查来源：

```bash
python -m fgw_grid_search.run --dry-run
python -m fgw_grid_search.run --check-inputs
```

`--dry-run` 不加载模型，不创建输出；`--check-inputs` 校验旧完成标记、来源文件、完整 WBBS 矩阵及固定池，不构建新缓存或评价。底层旧来源装载器只读检查两 split 的来源，不进行 test 搜索。

以下为已完成的**166项完整valid网格**命令；31项历史筛选使用上节配置：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u -m fgw_grid_search.run --stage search --config fgw_grid_search/config.json --workers 8
```

默认仅搜索 valid，500 query × 3,598 候选；自动创建真实 Asia/Shanghai 时间目录 `artificial/fgw_grid_YYYYMMDDTHHMMSS+0800/`。每个变体在 worker 实际启动时保存自己的时间身份。可用 `--config path/to/grid.json` 提供网格配置，用 `--output-root artificial/fgw_grid_<本次时间>` 显式指定新目录。`--workers` 为 1–8，只控制变体并行，默认 4；矩阵只读共享，不改变分数。

中断后只允许显式恢复原目录，代码、配置、环境、来源和协议必须一致：

```bash
python -u -m fgw_grid_search.run --resume-root artificial/fgw_grid_原启动时间 --workers 4
```

完整分项缓存按 query 封存；续跑校验已封存行并仅补缺失行。已完成变体按哈希复用；未完成变体重新评价，其旧 attempt 保留。全部完成的运行只读核验，不改身份、已有文件或 mtime；若封存后中断、尚未生成登记提案，仅补齐缺失的提案。

搜索完成并冻结后，显式单独评价一次 test：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u -m fgw_grid_search.run --stage test \
  --search-root artificial/fgw_grid_搜索启动时间 --workers 1
```

test 创建另一个新时间目录，固定 7,199 query × 7,198 候选，仅运行冻结全局胜者。拒绝 `--config`、未完成搜索、变更来源、代码或数值环境；不能在该入口上跑 test 网格。

## 来源、缓存与验证

默认只读来源为 `artificial/fgw_fusion_20261119T174724+0800/`，沿旧完成链校验 F/G 特征、模型、协议和原 WBBS。复用已核验的完整 W 矩阵，不重新计算 WBBS；在新目录从冻结特征计算完整 F/G 分数。旧 Top-500 无法代替新融合所需的全候选分数。

`sources.py` 负责来源与保护清单，`cache.py` 保存 float64 F/G/W 矩阵及原始 query/gallery ID。`scoring.py` 只处理对齐的单 query 全候选向量，`selection.py` 实现有限网格及固定选择规则，`pipeline.py` 流式评价，`verify.py` 独立重算公式、全候选 Top-500 及所有指标，`state.py` 核验封存链。

源审计每类取最小 query ID，独立复核 F 最优匹配、G 与原 FGW 完整候选排序；W 绑定旧逐对审计与完整矩阵哈希。新评价逐 query 独立检查全候选 Top-500、分项、exact/coverage 的所有 K 与类别／分组，分母始终来自完整候选库。来源保护前后比较哈希、大小和 mtime。禁止覆盖旧目录或嵌套新输出到来源目录。

根目录主要产物为 `run_manifest.json`、`settings.json`、`inputs.json`、`source_audit.json`、`component_cache/`、`evaluation_index_valid.json`、`grid_results.csv`、`selection.json`、`source_protection_{before,after}.json` 和 `completed.json`。test 用 `evaluation_index_test.json`、`frozen_selection.json`。每个 `valid/test/<variant>/` 含真实时间身份、共享 Top-500 排名及原始分项、exact/coverage 报告、独立 verification 与完成标记。只有全部评价、选择和保护核验成功才能封存根目录。

完成后生成 `catalog_sources.json` 登记提案；不会在搜索过程中修改现有成绩。后续实际完成后执行：

```bash
python -m fgw_grid_search.register --run-root artificial/fgw_grid_原启动时间 --apply
python results_catalog/build.py check
```

注册器拒绝不完整或哈希不一致的产物，先校验并生成汇总，再备份、原子更新；失败回滚。原 FGW/WBBS 不重复登记。两轮分别新增31与166项valid登记，总计299项；旧133项在第二轮采集前后深度一致，未新增test。

实现检查使用临时目录和合成数据：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m unittest discover -s fgw_grid_search/tests -v
```

初次代码交付验证（2026-11-19）：本地与远程各74项合成测试通过；旧FGW的51项回归与登记器19项测试通过，当时102项结果检查通过。本地旧FGW测试须把`TMPDIR`设为其`resolve()`后的真实路径，避免macOS的`/var`与`/private/var`临时目录别名影响旧测试fixture，未修改旧源码。远程另做valid只读来源加载及37类代表query审计，F独立重算最大误差4.44×10⁻¹⁶、G误差0，旧FGW Top-500及保存分项精确一致。以上是实现与来源检查，不是正式网格搜索结果。


## 31项valid完成记录

运行目录为`artificial/fgw_grid_20261119T195143+0800/`，启动代码`ab621d9`，8个worker。外层预算记录为同名前缀`.launch.json`：2026-11-19 19:51:43启动，19:56:10退出0，墙钟266.86秒，含来源检查、缓存、全部评价、独立验证与封存；后续结果登记不计入该执行计时。新增筛选功能在本地、远程各通过78项合成测试。

`selection.json`冻结胜者为`sum_minmax_d10_f02_g02_w06`：

```text
S(q,c) = 0.2*Nq(F(q,c)) + 0.2*Nq(G(q,c)) + 0.6*Nq(W(q,c))
```

选参指标valid总体P@10为55.02%，共2751/5000次命中；旧固定融合参考为54.38%，增加32次命中、0.64个百分点。第二名是原分数加法`0.3F+0.3G+0.4W`，P@10为54.88%，差距仅0.14个百分点。胜者只代表冻结31项中按指定valid指标选出的方案，尚无新test泛化结论；不能宣称连续权重全局最优或全部指标最佳。

31项都使用500个query及各自3598个候选，全部Top-500与原始分项独立核对通过，评分误差为0，指标最大绝对误差2.34×10⁻¹⁵以内。固定参考与旧FGW的500个query逐项P@10、25万行Top-500及分数完全一致，排名文件SHA-256相同。8707个受保护文件的哈希、大小和mtime前后未变。31项valid已登记，旧102项保留，总计133项；没有运行test。完整分项矩阵和排名留在服务器，本地保存轻量报告与原始签名副本。


## 166项完整valid完成记录

独立新目录为`artificial/fgw_grid_20261119T200653+0800/`，启动代码`01b98c1`，8个worker，显式使用`config.json`，`weight_ticks=null`，未改变评分或选择规则。2026-11-19 20:06:53启动，20:20:15退出0，实际墙钟801.92秒（约13分22秒），含来源检查、缓存、166项评价、独立验证及封存；登记与同步另计。31项历史目录、模型及原结果均原样保留。

全局唯一P@10最高项为`sum_raw_d10_f01_g04_w05`，已写入该目录的`selection.json`：

```text
S(q,c) = 0.1*F(q,c) + 0.4*G(q,c) + 0.5*W(q,c)
```

这里直接使用原始分数，不做min-max归一化；F仍保留，没有A或J。各族最优valid结果由封存选择与报告摘录，以下数值均为百分数：

| 公式族 | F/G/W权重或参考公式 | P@10 | mAP@100 | Coverage-nDCG@10 |
| --- | --- | ---: | ---: | ---: |
| 原分数加法 | 0.1/0.4/0.5 | 55.10 | 20.5478 | 85.8053 |
| 按query min-max加法 | 0.2/0.2/0.6 | 55.02 | 20.5144 | 85.9768 |
| 加权乘法 | 0.1/0.5/0.4 | 54.74 | 20.5132 | 85.7135 |
| 旧固定公式参考 | `(0.5F+0.5G)W` | 54.38 | 20.3444 | 85.8974 |

新胜者共2755/5000次Top-10命中，比原31项胜者多4次、提升0.08个百分点；比旧固定参考多36次、提升0.72个百分点。相对31项胜者，Coverage-nDCG@10下降0.17个百分点，因此只称为指定P@10目标下的网格胜者。F=0.1位于允许下界；未搜索更细权重或F=0，不宣称连续空间最优、统计显著优势或test泛化优势。

166项均在500个valid query×3598候选上完成，同一排名输出exact/coverage所有K及分组。根完成标记绑定1839个文件，全部保存评分和分项独立复算误差为0，指标最大绝对误差≤2.45×10⁻¹⁵；8707个来源文件前后哈希、大小及mtime不变。另独立重算83,000条逐query P@10整数目标及同分规则，确认全局／每族第一均无并列。31项交集的31对排名及124对逐query／逐类别CSV逐字节一致，62对exact/coverage JSON去除运行metadata后全部内容一致；这些交集是同一来源的一致性复核，不是独立重复实验。

新增166项valid后总登记299项，旧133项记录全部深度一致。为了同步汇总，登记器只将`collect`生成的snapshot改为紧凑JSON；紧凑前后299项内容完全相同，数据schema与指标不变，相关23项测试在本地和远程通过。完整排名与矩阵保存在服务器，本地保存轻量报告、冻结选择和签名副本；代码、文档及结果汇总同步GitHub。test未执行。
