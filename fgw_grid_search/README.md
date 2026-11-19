# F、G、WBBS 网格搜索

独立代码包。当前仅完成代码设计与实现验证，**未运行正式搜索或 test，尚无最优配置**。科学设计与公式见[消融实验](../doc/消融实验.md#fgwbbs网格搜索)，公共协议和结果管理见[项目说明](../doc/项目说明.md)。旧 `fgw_fusion/`、模型、冻结协议与历史结果保持不变。

## 配置与选择

默认配置为 [config.json](config.json)。F 使用此前 FGW 的冻结 cc8 **最优区域匹配**分数，G 为冻结 WaPIRL 分数，W 为原 WBBS 相似度；不含角向 A 或 J，不训练或重新选 checkpoint。

| 配置 | 默认值与约束 |
| --- | --- |
| `denominator` | 10；整数 1–20，权重以 `1/denominator` 为步长 |
| `min_f_ticks` | 1；整数 1–denominator，F 权重／指数严格大于 0 |
| `families` | `sum_raw`、`sum_minmax`、`weighted_product`，可选非空子集 |
| `include_reference` | true；额外评价固定旧公式 `(0.5F+0.5G)*W`，参与全局选择 |

三项 ticks 之和等于 denominator；G、W 允许为 0。默认每族 55 组，共 165 组，加固定参考合计 **166 项 valid 评价**。参考的 `[0.5,0.5,1]` 表示其内层 F/G 系数和外层 W 指数，不属于权重和为 1 的网格。

`sum_minmax` 在每个 query 的**完整候选库**分别归一化三项，常数分支置 0。纯 F 端点 `[1,0,0]` 在所有族均逐值返回原始 F，保留可核对基线；这与 min-max F 排名等价。乘法省略指数为 0 的分支，活跃分支为 0 则结果为 0，不加 epsilon。

唯一选参指标为 valid 的 `exact/all/Precision@10`，比较所有 query 的 Top-10 **整数总命中数**，避免浮点均值造成假同分。并列依次取活跃分支较少、族优先级 `sum_raw → weighted_product → sum_minmax → legacy_fgw`、F 较大、W 较小、变体 ID 字典序较小。mAP@100、Coverage-nDCG@10 只作描述，不参与选择。`selection.json` 保存全局胜者、每族胜者、完整排名及规则；test 只接受全局胜者。

除上表四项外，K、选参 split、指标、同分规则等科学设置冻结，不允许用配置覆盖。重复／等价端点不是独立重复实验。“最优”仅指该有限网格及所含参考中的最优，不表示全局连续权重最优。方案在看过历史 test 后提出，标记为事后研究。

## 入口与恢复

在仓库根目录的原 PyTorch/CUDA 环境执行。只查看计划或只读检查来源：

```bash
python -m fgw_grid_search.run --dry-run
python -m fgw_grid_search.run --check-inputs
```

`--dry-run` 不加载模型，不创建输出；`--check-inputs` 校验旧完成标记、来源文件、完整 WBBS 矩阵及固定池，不构建新缓存或评价。底层旧来源装载器只读检查两 split 的来源，不进行 test 搜索。

以下为**后续正式运行命令，本次没有执行**：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u -m fgw_grid_search.run --stage search --workers 4
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

注册器拒绝不完整或哈希不一致的产物，先校验并生成汇总，再备份、原子更新；失败回滚。原 FGW/WBBS 不重复登记。当前未产生正式新结果，已有 102 项登记保持不变。

实现检查使用临时目录和合成数据：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m unittest discover -s fgw_grid_search/tests -v
```

2026-11-19验证记录：本地与远程各74项合成测试通过；旧FGW的51项回归与登记器19项测试通过，现有102项结果检查通过。本地旧FGW测试须把`TMPDIR`设为其`resolve()`后的真实路径，避免macOS的`/var`与`/private/var`临时目录别名影响旧测试fixture，未修改旧源码。远程另做valid只读来源加载及37类代表query审计，F独立重算最大误差4.44×10⁻¹⁶、G误差0，旧FGW Top-500及保存分项精确一致。以上是实现与来源检查，不是正式网格搜索结果。
