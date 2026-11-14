# 固定参数的角向约束＋最优匹配组合 test

用户在查看四组 test 结果后，于2026-11-15明确要求追加这一组组合实验。
α=0.5、角向 γ=0.5、占比 γ=0，其余设置全部固定。只运行 test，不进行参数搜索或训练。
这是观察 test 后提出的补充组合实验，产物以 `posthoc_test_combination=true` 记录，
不能将组合方案表述为查看 test 之前确定。

## 组合公式

```text
S_combined = (0.5 * S_local_optimal + 0.5 * S_global) * exp(-0.5 * D_angular)
```

- `S_local_optimal`：原区域对的特征、径向位置、相对面积评分不变，改用最大化
  `sqrt(area)` 加权总分的最优一对一匹配；Top-4、min_area=5、h512与原模型不变。
- `S_global`：同一个冻结 WaPIRL 模型的整图 h512 余弦相似度，截断到[0,1]。
- `D_angular`：原始完整缺陷图外圈（归一化半径≥0.65）的1～4阶圆周矩幅值差的均值，
  保持整体旋转不变；任一图没有外圈缺陷时距离为0。
- 角向项作用在融合后的整项分数上，与角向单项保持同一施加位置。
  不额外添加占比约束，也不更换checkpoint、区域、数据或评价规则。

组合提高某些候选的分数不等于提高检索指标，必须用实际排名比较。
对每个 query 的全部候选重新计算组合分数，不从此前任何Top500列表筛候选。

## 云端后台命令

在服务器 `wbm-region-retrieval` 仓库根目录、已激活原 Python 环境后执行：

```bash
git pull --ff-only origin main && tmux new-session -d -s f_combo_test -c "$PWD" \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  bash -c 'exec "$@" > f_combo_test.launch.log 2>&1' bash \
  "$(command -v python)" -u experiments/proposed/f_fusion_combination_test/run.py \
  --stage test \
  --wapirl-test-root /data/wangkaiyan2024/compared_methods/artifacts/wapirl_test_v1
```

全部复用已有特征，评分使用 CPU NumPy，无需 GPU，也不需要 `--device` 或 `--prepare-workers`。
保留上一轮完整 test 目录及其引用的 F/WaPIRL 特征；只有 CSV 文件不够。

查看进度：

```bash
tail -n 50 -f f_combo_test.launch.log
cat artifacts/f_fusion_combination_test_v1/status.json
tmux ls
```

前期核验已有排名和特征的哈希，尚未创建 status 时以启动日志为准。
评分每100个query打印 `Combination test: 100/7199 queries`，之后分别输出 exact、coverage 指标阶段。
成功以 `status=completed` 为准；tmux 退出本身不证明成功。Ctrl+C只退出日志查看。

## 来源与兼容性

旧 valid/test 源码完全不改，避免破坏已有结果签名。只新增本独立目录与测试文件。
新输出默认 `artifacts/f_fusion_combination_test_v1/`，不能与任何输入目录重叠。

| 参数 | 默认路径 | 用途 |
| --- | --- | --- |
| `--previous-test-root` | `artifacts/f_fusion_ablation_test_v1` | 原融合、角向单项、最优匹配的正式test结果；完整签名及描述子 |
| `--source-root` | `artifacts/f_segmentation_v1` | 核验原cc8/best_nce模型 |
| `--f-feature-root` | `artifacts/f_segmentation_radial_area_v1` | 优先复用 `cc8/features/test/` |
| `--fusion-root` | `artifacts/f_global_fusion_v1` | 后备 `f_cache/features/test/` |
| `--wapirl-test-root` | `../compared_methods/artifacts/wapirl_test_v1` | 原WaPIRL test全图特征 |
| `--output-root` | `artifacts/f_fusion_combination_test_v1` | 本次独立输出 |

若前两处 F 缓存不存在，读取 `--previous-test-root` 下的 `f_cache/features/test/`。
使用的缓存完成标记必须与上一轮test实际使用的哈希一致；不存在、损坏或不一致均明确报错，
不偷偷重提取。`--data-file`允许迁移原始NPZ路径，但内容哈希必须不变。
所有相对路径以仓库根目录解析。`--dry-run`只显示固定计划，不加载/核验数据或写文件。

上一轮test要求固定 α=0.5、angular=0.5、density=0；组合入口不接受强度覆盖参数。
它核验旧test运行、评价、冻结选择、源码、每组结果及描述子，核对当前模型、数据、协议、
特征、K和tie seed。旧来源加只读锁且不改写，新输出使用独占锁。
同设置重跑会验证并复用已完成组合结果；中断时留下未完成attempt，下一次从头计算未完成排名。

## 输出与评价

固定 test 全部7199个 query，每个7198个候选，排除自身。
逐 query 计算全库后保存Top500，exact与coverage共用排名。
K=1、5、10、20、50、100、500，分母及coverage IDCG使用完整候选库。
保留总体、类别宏平均、单/混合缺陷、标签数量、逐query及逐类别报告。

```text
artifacts/f_fusion_combination_test_v1/
├── combination_manifest.json / plan_before_evaluation.json
├── comparison.csv       # 原融合、角向单项、最优匹配单项、本次组合，共4种不同排名
├── combination.csv      # 仅组合，含 delta_vs_baseline / delta_vs_angular / delta_vs_optimal
├── status.json / run.log
└── test/optimal_angular_0.5/
    ├── rankings.csv     # 组合分数、三种参考分数、局部/整图分数及角向距离
    ├── metrics.json / per_query.csv / per_class.csv
    ├── coverage_metrics.json / coverage_per_query.csv / coverage_per_class.csv
    └── completed.json
```

只计算新组合的排名和指标。对照表中的前三组读取旧签名报告，不重新跑；零占比组与基线相同，
不重复列出。所有数值为比例，差值乘100才是百分点。不生成新的 `selection.json`。

## 本地验证

```bash
PYTHONPATH=tests python -m unittest test_f_fusion_combination_test test_f_fusion_ablation_test test_f_fusion_ablation test_f_global_fusion test_f_segmentation_geometry test_f_segmentation -v
```

新增测试覆盖固定公式和施加位置、角向距离为0的回退、相同匹配下回退角向单项、
完整候选库中提升两个单项Top1之外的候选、双指标、中断重试与完成结果复用、
真实旧评价器生成的合成test产物交接、来源哈希和mtime保持、对照差值、
更换模型/强度/缓存以及报告损坏拒绝。使用合成数据与轻量编码器，不代表正式test收益。

2026-11-15验证：原31项相关回归测试通过；新增4项测试修正空分组无值的断言后全部通过。
本地未执行真实数据训练或正式组合test，正式运行由用户在服务器启动。
