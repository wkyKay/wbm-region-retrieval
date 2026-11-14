# F 主方法 + 整图 WaPIRL 分数融合

使用两个已训练模型进行推理分数融合，不重训、不修改原训练目录、checkpoint、协议或旧结果。
借鉴 Yin 等（2025）全局与局部互补的思路；该论文联合的是训练损失，本入口是独立的分数级融合实验，
不称为其原方法复现，也不预设融合一定优于单分支。

```text
score(q,c) = (1-alpha) * F(q,c) + alpha * WaPIRL(q,c)
```

- `F` 固定为 cc8 主方法：min_area=5、Top-4、52×52 三通道窗口、h512、径向位置与相对面积、
  单向贪心一对一匹配、查询区域 sqrt(area) 聚合；原来的几何公式全部保留。
- `WaPIRL` 使用整图 h512 的 L2 归一化余弦并截断到 [0,1]，读取已完成正式评价的特征缓存。
- `alpha` 是整图权重。默认 valid 比较 `0,0.1,0.2,0.3,0.5,0.75,1`；0 和 1 必须保留作对照。
  不做每 query min-max、分数乘积、特征拼接或标签门控。
- 默认按 valid **总体 exact Precision@10** 最大值选权重；精确并列选更小 alpha。
  `selection.json` 冻结结果。test 仅评价已选权重以及 0/1 对照，相同权重只算一次。
  若最优为 0，则保留原 F；若为 1，则说明 valid 更支持整图分支，不将其宣称为融合收益。

## 运行

在服务器的 `wbm-region-retrieval` 仓库根目录，激活原 PyTorch 环境。路径参数相对本仓库根目录解析。
原有 F 模型以及 WaPIRL 的 valid/test 正式评价需要已经完成；本入口不会启动任何训练。

### tmux 后台完成 valid 和 test

在服务器仓库根目录、已激活原 PyTorch 环境时执行以下一条命令（反斜杠仅用于换行）：

```bash
git pull --ff-only origin main && tmux new-session -d -s f_fusion -c "$PWD" \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  bash -c 'exec "$@" > f_global_fusion.launch.log 2>&1' bash \
  "$(command -v python)" -u experiments/proposed/f_global_fusion/run.py \
  --stage all --device cuda --prepare-workers 4 \
  --wapirl-valid-root /data/wangkaiyan2024/compared_methods/artifacts/wapirl_valid_v1 \
  --wapirl-test-root /data/wangkaiyan2024/compared_methods/artifacts/wapirl_test_v1
```

使用当前环境的 Python 绝对路径，避免已有 tmux server 的 PATH 不同。
`f_global_fusion.launch.log` 捕获从启动开始的全部输出，包括正式 `run.log` 创建前的检查错误。
同名 tmux 会话存在时命令会拒绝再启动；程序结束后会话自动退出，日志仍保留。
断开 SSH 不影响后台任务，服务器重启不在此保障范围内。

在同一仓库根目录查看：

```bash
tail -n 50 -f f_global_fusion.launch.log
```

每100个 query 输出评分进度，例如 `Fusion valid: 100/500 queries, 7 weights`、
`Fusion test: 100/7199 queries, 3 weights`。权重命中0或1时 test 只有2组。
启动时先核验文件哈希和数据协议，可能暂时没有 query 计数；评分结束后还要生成各组指标。
按 `Ctrl+C` 只退出日志查看，不停止任务。另可运行 `tmux ls` 查看会话是否存在，
`cat artifacts/f_global_fusion_v1/status.json` 查看状态（来源预检查通过后才创建）。
成功结束应为 `"status": "completed", "stage": "all"`；会话消失本身不代表成功，异常原因查看启动日志。

先在 valid 上比较并保存选择（默认 stage 就是 valid）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u experiments/proposed/f_global_fusion/run.py --stage valid --device cuda --prepare-workers 4
```

固定上述选择后执行 test：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u experiments/proposed/f_global_fusion/run.py --stage test --device cuda --prepare-workers 4
```

也可以一条命令顺序完成 valid 选择和 test：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u experiments/proposed/f_global_fusion/run.py --stage all --device cuda --prepare-workers 4
```

默认来源目录：

| 参数 | 默认路径 | 用途 |
|---|---|---|
| `--source-root` | `artifacts/f_segmentation_v1` | 只读核验 cc8 旧训练、冻结选模规则、checkpoint 及协议 |
| `--f-feature-root` | `artifacts/f_segmentation_radial_area_v1` | 优先复用 `cc8/features/{valid,test}` |
| `--wapirl-valid-root` | `../compared_methods/artifacts/wapirl_valid_v1` | 读取 `valid/prepared/features.npz` 及完整评价签名 |
| `--wapirl-test-root` | `../compared_methods/artifacts/wapirl_test_v1` | 读取 `test/prepared/features.npz` 及完整评价签名 |
| `--output-root` | `artifacts/f_global_fusion_v1` | 本次独立输出 |

若两仓库不相邻，显式指定 WaPIRL 路径。例如已知服务器 compared_methods 目录：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u experiments/proposed/f_global_fusion/run.py --stage all --device cuda --prepare-workers 4 --wapirl-valid-root /data/wangkaiyan2024/compared_methods/artifacts/wapirl_valid_v1 --wapirl-test-root /data/wangkaiyan2024/compared_methods/artifacts/wapirl_test_v1
```

F 缓存不存在时，使用冻结的 F 模型在新输出的 `f_cache/features/` 下提取，缺失分割缓存也只在新输出重建。
存在但未完成、损坏或签名不符的缓存会报错；不会悄悄替换。只做分数融合时使用 NumPy CPU 计算；
`--device cuda` 只影响缺失 F 特征的提取。WaPIRL 特征缺失时须先运行其原正式评价入口，不能用 Top-500 排名补齐。
原始 NPZ 路径迁移可追加 `--data-file /absolute/path/Wafer_Map_Datasets.npz`，仍严格核验内容哈希。

可用 `--dry-run` 只显示计划，不加载数据或创建目录。
重跑相同命令会核验并复用已完成结果；中断的临时 attempt 保留，重试另建 attempt。
改变 alpha 网格、选权指标、科学设置或代码后，必须另选 `--output-root`；不能用旧选择直接测试新设置。

如需另一套预先指定的选择标准，例如类别宏平均 coverage-nDCG@10：

```bash
python -u experiments/proposed/f_global_fusion/run.py --stage valid --device cuda --prepare-workers 4 --selection-family coverage --selection-group class_macro --selection-metric Coverage-nDCG@10 --output-root artifacts/f_global_fusion_coverage_macro_v1
```

随后 test 必须携带相同的选权参数、alpha 网格和输出路径。不要根据 test 再换权重或选权指标。
主方法 cc8 本身是在查看历史 test 结果后由用户确定的；本实验只保证新增融合权重在 valid 上选择，
不追述为整个方法设计从未查看过 test。

## 评价与来源保护

- 固定去重 train/valid/test=25,193/3,599/7,199；valid 固定500 query×3,598候选，
  test 全7,199 query×7,198候选。按原始 sample ID 对齐，排除自身。
- 每个 query 对完整候选库计算两路分数，一次计算供所有待评权重复用；内存只保留当前 query 的分数，
  每个权重保存 Top-500，不从两份截断排名的并集重排。
- exact 和 coverage 使用同一排名；K=1,5,10,20,50,100,500。AP/Recall 分母及 coverage IDCG
  来自完整候选库。输出总体、类别宏平均、单/混合缺陷及标签数量分组。
- 复用原 F 的确定性 SHA 同分排序。F 空区域图仍参与检索，F 分数为0，整图分支正常参与融合。
- WaPIRL 缓存校验完成标记、文件哈希、原始数据与冻结协议、gallery ID、h512 归一化、训练完成及
  `best_nce` 来源；valid/test 必须来自同一 checkpoint、推理设置与代码版本。拒绝 smoke 和跨组合筛除协议。
- F 原训练锁防止同时训练；继续使用原来源签名检查，不绕过源码兼容性验证。
  新目录位于原训练和几何评分模块之外，不修改其源码或历史签名。

## 结果

```text
artifacts/f_global_fusion_v1/
├── fusion_manifest.json          # 输入、参数及源码签名
├── valid_manifest.json
├── test_manifest.json            # 绑定 valid 选择及 test 特征来源
├── selection.json                # 冻结整图权重、选权标准、各权重 valid 得分
├── comparison.csv                # 全部已完成权重、split、指标及分组的长表
├── comparison.json
├── status.json / run.log
├── f_cache/features/             # 仅在需要补提取 F 特征时生成
├── valid/alpha_0.0/              # 其余 alpha_* 同构
└── test/alpha_0.0/
    ├── rankings.csv              # 融合分数、f_score、global_score
    ├── metrics.json / per_query.csv / per_class.csv
    ├── coverage_metrics.json / coverage_per_query.csv / coverage_per_class.csv
    └── completed.json
```

先看 `selection.json` 和 `comparison.csv`。正式对比用 selected alpha 与两个纯分支的同一 split/group。
CSV 中的 Precision、Recall 等均为0～1，不是百分数。

## 验证

新增测试覆盖：原 F 端点评分和排名一致、整图端点数值一致、乱序原始 ID 对齐、空区域、多区域匹配、
全候选融合可提升两路 Top-1 之外的候选、exact/coverage 区别、valid 选择及并列规则、
test 绑定选择、来源/缓存/协议/模型校验、旧文件哈希和 mtime 不变、完成结果复用、
外部缓存复用、缺缓存只在新目录生成，以及 exact 后中断的恢复。

```bash
python -m unittest discover -s tests -p 'test_f_global_fusion.py' -v
```

端到端测试使用合成冻结协议、轻量编码器及模拟 WaPIRL 特征，验证实现与来源保护，
不代表正式融合精度。本地已有真实 WaPIRL valid/test 特征可用于只读来源校验；正式 cc8 融合由用户在服务器执行。

2026-11-14 本地验证：新增6项融合测试通过；连同原 F 分割与几何评分入口共21项相关测试通过。
真实 WaPIRL valid/test 缓存分别为3599×512、7199×512，全部完成文件哈希、冻结数据与清单、
两边 checkpoint 和推理设置/代码一致性检查通过。原有 tracked 源码没有改动。
