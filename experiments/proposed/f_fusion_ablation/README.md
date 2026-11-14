# 固定 α=0.5 的四组 valid 消融

仅使用冻结的 valid 500个query、3599个gallery，每query对全部3598个候选打分，排除自身。
复用 cc8 F 的 best_nce 和已完成的 WaPIRL valid 特征，不重训、不改变原有模型/协议/源码/结果。
入口只有 `--stage valid`，拒绝 test/all，也不提供调整 α 的参数。

## 四组定义与预定参数

基线 `B(q,c)=0.5*F(q,c)+0.5*G(q,c)`，F完整保留径向位置、相对面积、Top-4、
单向贪心一对一与query sqrt(area)权重；G保持整图h512截断余弦。
此次每组只改变一个因素，不把占比项、角向项和最优匹配叠加。

| 组 | 定义 | 预定参数 |
|---|---|---|
| baseline | B | α固定0.5 |
| density | B × exp(-γ_d × abs(d_q-d_c)) | γ_d=0,2,5,10,20 |
| angular | B × exp(-γ_a × D_angle) | γ_a=0,0.25,0.5,1,2 |
| optimal | 0.5×F_optimal + 0.5×G | 无新增待选参数 |

两项强度独立按 valid **总体 exact Precision@10** 最大值选择，精确并列取较小强度。
γ=0直接引用baseline，不重复评价，防止浮点差异；最优为0时该方向未胜过基线。
总共10个不同评分配置（baseline、optimal、4个非零density、4个非零angular），最终汇总为四组。
参数及选择标准在评分前写入 `plan_before_evaluation.json`，不在运行后扩展网格。

### 整图缺陷占比

`d = count(raw==2) / count(raw==1 or raw==2)`。
统计全部原始缺陷，不用Top-4区域之和、不按裁切/resize图计算，也不额外去噪。
当前gamma单位对应0～1占比；这与WBBS去噪后点数惩罚并不相同。
该项乘在完整融合分数上，α仍保持0.5。它可能降低额外缺陷候选的排名，因此也可能损伤coverage。

### 角向结构

晶圆中心和半径与原径向项一致：raw==1/2支撑包围盒中心、支撑像素到中心的最大距离。
从所有原始缺陷中取归一化半径 `rho>=0.65` 的点，计算：

```text
m_k = abs(mean(exp(i*k*theta))), k=1,2,3,4
D_angle = mean_k(abs(m_k(q)-m_k(c)))
```

这是按外圈缺陷点数归一化的圆周矩幅值，反映外圈角向分布，不引入绝对方位。
连续点集整体旋转只改变复数相位，因此幅值不变；将图像任意角度旋转再栅格化可能引入像素变化，
不宣称对这种操作严格不变。不同形状也可能具有相同的四阶描述，它不是完整结构编码。
若任一晶圆外圈无缺陷，则 `D_angle=0`，不从缺失的角向证据施加惩罚。
半径阈值0.65与阶数1～4为本轮预先固定的实现选择，不使用test调节。

### 最优加权匹配

保留原区域对分数s_ij，替换原贪心选择为：

```text
F_optimal = max_M sum_(i,j in M)(sqrt(area_i)*s_ij) / sum_i sqrt(area_i)
```

M为一对一匹配；未匹配query区域贡献0，不额外惩罚候选未匹配区域。
补零到最多4×4，枚举至多4!=24种分配，并对当前query的全部候选向量化计算。
优化的是最终加权和，区别于原先先按未加权区域对分数贪心选边、再面积加权。
原始空区域query仍保留，F/F_optimal均为0，整图分支照常评分。
优化分数不保证检索指标提高；错误候选也可能得到更高分。

## 服务器运行（tmux）

在服务器 `wbm-region-retrieval` 仓库根目录，激活原PyTorch环境，执行一条命令：

```bash
git pull --ff-only origin main && tmux new-session -d -s f_valid4 -c "$PWD" \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  bash -c 'exec "$@" > f_valid4.launch.log 2>&1' bash \
  "$(command -v python)" -u experiments/proposed/f_fusion_ablation/run.py \
  --stage valid --device cuda --prepare-workers 4 \
  --wapirl-valid-root /data/wangkaiyan2024/compared_methods/artifacts/wapirl_valid_v1
```

当前环境Python绝对路径传入tmux，不依赖旧tmux server的PATH。
同名会话存在时会拒绝另开任务；SSH断开不影响任务，服务器重启则需重新运行。
日志捕获包括来源预检查在内的全部标准输出与错误；同命令重启会重写启动日志，旧run.log仍保留。

查看进度：

```bash
tail -n 50 -f f_valid4.launch.log
```

每100个query显示 `Ablation valid: 100/500 queries, 10 variants`。
评分结束后逐配置生成exact与coverage报告，显示 `Metrics valid: 1/10 baseline ...`。
按Ctrl+C只停止日志查看，不结束后台任务。另可查看：

```bash
cat artifacts/f_fusion_ablation_valid_v1/status.json
tmux ls
```

`status.json`在来源核验通过后创建；含phase、queries_done、queries_total或当前指标配置。
最终应显示 `status=completed`、`split=valid`、`test_evaluated=false`；会话退出本身不证明成功。

## 路径、来源与重跑

默认读取：

- `--source-root artifacts/f_segmentation_v1`：只读核验cc8旧训练与checkpoint。
- `--f-feature-root artifacts/f_segmentation_radial_area_v1`：优先复用 `cc8/features/valid/`。
- `--fusion-root artifacts/f_global_fusion_v1`：若上述不存在，复用此前融合补提取的 `f_cache/features/valid/`。
- `--wapirl-valid-root ../compared_methods/artifacts/wapirl_valid_v1`：已完成正式valid评价的整图特征。
- `--output-root artifacts/f_fusion_ablation_valid_v1`：本轮独立输出。

路径相对仓库根目录解析。F特征均不存在时，只向新输出 `f_cache/features/valid/` 提取，
不重训、不写源目录；存在但未完成/损坏/签名不符则拒绝复用。WaPIRL特征要求原正式valid已完成。
`--device`只用于缺失F特征的提取，评分使用CPU NumPy；`--data-file`允许迁移NPZ路径，仍核验内容哈希。
原始NPZ与冻结协议包含全部split，原Protocol会读取它们做完整性核验；本入口不提取test特征、
不计算test描述子/排名/指标、不读取WaPIRL test评价目录。

源码、参数、模型、数据、特征等绑定签名。相同设置重跑会校验并复用已完成配置，
失败的attempt保留、重试另建，不覆盖旧结果；改变科学设置/源码需新输出目录。
`--dry-run`只显示预定方案，不加载数据、不创建目录。

## 结果与判断

```text
artifacts/f_fusion_ablation_valid_v1/
├── plan_before_evaluation.json / ablation_manifest.json / evaluation_manifest.json
├── descriptors/valid.npz          # 原始ID、density、moments、outer_count与完成哈希
├── selection.json                # density/angular各自的强度选择，含零强度基线
├── four_groups.csv               # 四组最终对照，含delta_vs_baseline（比例差）
├── comparison.csv                # 全部10个配置，不隐藏未选中结果
├── status.json / run.log
└── valid/{baseline,optimal,density_2.0,...,angular_0.25,...}/
    ├── rankings.csv              # Top500融合分数、两路分数及两项距离
    ├── metrics.json / per_query.csv / per_class.csv
    ├── coverage_metrics.json / coverage_per_query.csv / coverage_per_class.csv
    └── completed.json
```

K=1,5,10,20,50,100,500，exact与coverage共用每配置的同一排名。
AP/Recall分母及coverage IDCG仍取完整候选库；所有查询逐个计算完整分数后才截取Top500。
CSV保留总体、类别宏平均、单/混合与标签数量分组，数值为0～1，差值乘100才是百分点。
先看four_groups.csv的总体exact P@10，再检查宏平均、混合缺陷与coverage的取舍。
本轮只按预定标准选两个单项的强度，不自动组合，不进入test。
这仍是用于选参的valid结果，不能作为独立泛化证据或未校正的显著性结论。

## 本地验证

```bash
python -m unittest discover -s tests -p 'test_f_fusion_ablation.py' -v
```

数学测试包含连续旋转不变性、原图占比、空外圈、精确最优部分匹配对独立穷举、贪心反例，
以及baseline/零强度对旧融合的精确回退。端到端采用合成冻结协议与轻量编码器，
检查valid-only、双指标、四组独立选择、来源只读、外部缓存复用、损坏拒绝、签名变化拒绝和中断恢复。
这些测试不代表正式检索收益；真实valid实验由用户在服务器执行。

2026-11-15验证：新增7项测试通过；连同原融合、几何评分和分割入口共28项相关测试通过。
新baseline的候选顺序与分数逐项精确复现原α=0.5入口；端到端测试确认旧来源文件哈希和mtime未变。
本地未执行真实数据的四组valid评价，也未启动正式训练或test。
