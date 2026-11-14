# 四组融合：冻结 valid 选择后的 test 评价

本入口承接 `../f_fusion_ablation/` 已完成的 valid 实验，固定 α=0.5，读取其
`selection.json`，只评价基线、缺陷占比、外圈角向结构、最优一对一匹配四组。
不重训、不重新运行 valid、不在 test 上搜索强度；density 与 angular 仍是各自独立的单项实验。

独立目录是为了保持原 valid 源码及签名完全不变，已有 valid 产物可直接使用。
必须保留服务器上的完整 valid 输出目录，只有下载的 CSV 或 selection.json 不够。
入口核验 valid 的运行/评价/选择签名、全部十个配置的完成哈希、源码和选参一致性，
随后核验 F 模型、协议、WaPIRL valid/test 的 checkpoint 与推理配置一致性。
重新核对 valid 指标只用于验证已有选择，不使用 test 指标修改选择。

2026-11-15 提供的 valid 选择为 density=0、angular=0.5。因此四组对应：

| 汇总 experiment | 实际排名 variant |
| --- | --- |
| baseline | baseline |
| density | baseline（零强度复用） |
| angular | angular_0.5 |
| optimal | optimal |

四组只需生成三套不同排名；程序从签名文件读取选择，不把上述值硬编码为测试参数。

## 云端运行

在服务器仓库根目录、已激活原 PyTorch 环境后执行：

```bash
git pull --ff-only origin main && tmux new-session -d -s f_test4 -c "$PWD" \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  bash -c 'exec "$@" > f_test4.launch.log 2>&1' bash \
  "$(command -v python)" -u experiments/proposed/f_fusion_ablation_test/run.py \
  --stage test --device cuda --prepare-workers 4 \
  --wapirl-test-root /data/wangkaiyan2024/compared_methods/artifacts/wapirl_test_v1
```

保留原 valid 目录 `artifacts/f_fusion_ablation_valid_v1/`。
如目录不同，用 `--valid-root` 指定完整目录。WaPIRL test 需已完成正式评价；
本入口复用其冻结整图特征，不启动 WaPIRL 的训练或评价。
`--dry-run` 只显示路径和策略，不读取/验证产物，也不创建目录。

查看进度：

```bash
tail -n 50 -f f_test4.launch.log
cat artifacts/f_fusion_ablation_test_v1/status.json
tmux ls
```

每100个 query 输出 `Ablation test: 100/7199 queries, 3 variants`；
评分后逐个输出 `Metrics test: 1/3 baseline (exact + coverage)`。
来源哈希核验期间以启动日志为准，status 在核验通过后创建。
最终必须见 `status=completed`；tmux 会话消失本身不代表成功。
Ctrl+C 只退出 tail。失败后原命令可重跑，已完成配置通过签名核验后复用；
未完成的排名需从头计算，保留旧 attempt，不覆盖原结果。

## 协议、来源与输出

固定 test 全部7199个 query，每个完整候选库7198张（排除自身）。
逐 query 计算全候选分数，保存 Top500；exact 与 coverage 使用同一排名。
K=1、5、10、20、50、100、500；AP/Recall 分母和 coverage IDCG 使用完整候选库。
总体、类别宏平均、单/混合缺陷和标签数量分组与 valid 保持一致。

默认只读来源：

- `--source-root artifacts/f_segmentation_v1`：原 cc8/best_nce 模型。
- `--f-feature-root artifacts/f_segmentation_radial_area_v1`：优先复用 `cc8/features/test/`。
- `--fusion-root artifacts/f_global_fusion_v1`：后备 `f_cache/features/test/`。
- `--valid-root artifacts/f_fusion_ablation_valid_v1`：完整 valid 实验与冻结选择。
- `--wapirl-test-root ../compared_methods/artifacts/wapirl_test_v1`：已完成的 WaPIRL test。

F 缓存均不存在时，只向本次新输出提取特征；`--device` 仅控制该提取，评分使用 CPU NumPy。
存在但损坏或签名不符的缓存会报错。`--data-file` 可指定迁移后的同内容 NPZ。
所有相对路径以仓库根目录为基准；`--output-root` 可更换新输出目录。

```text
artifacts/f_fusion_ablation_test_v1/
├── test_manifest.json / evaluation_manifest.json / plan_before_evaluation.json
├── frozen_selection.json      # 原 valid selection 的完整副本；不生成新的 test 选参
├── descriptors/test.npz
├── four_groups.csv            # 四组，density=0 时显式复用 baseline
├── comparison.csv             # 实际计算的不同配置，含 delta_vs_baseline
├── status.json / run.log
└── test/{baseline,angular_0.5,optimal}/
    ├── rankings.csv
    ├── metrics.json / per_query.csv / per_class.csv
    ├── coverage_metrics.json / coverage_per_query.csv / coverage_per_class.csv
    └── completed.json
```

CSV 数值为比例，`delta_vs_baseline * 100` 为百分点差。这里的 baseline 始终是
α=0.5 的原融合 F。WBBS 不在本次计算范围中，后续用相同协议的 WBBS test 报告比较。

## 验证

```bash
PYTHONPATH=tests python -m unittest test_f_fusion_ablation_test test_f_fusion_ablation test_f_global_fusion test_f_segmentation_geometry test_f_segmentation -v
```

新增测试覆盖：完整 valid→test 交接、选择与模型不一致拒绝、所有 test query 与排除自身、
原融合基线排名和分数逐项复现、双指标、只计算所选强度、零强度组复用、来源只读、
中断重试和完成结果复用、缓存损坏拒绝、参数搜索接口拒绝。
合成协议及轻量编码器用于本地验证；不代表已完成真实服务器 test 或证明检索收益。

2026-11-15 本地验证：新增3项测试通过，连同原四组 valid、融合、几何和分割入口共31项测试通过。
未启动真实数据训练或正式 test。
