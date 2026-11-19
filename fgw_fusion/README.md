# F、G、WBBS 固定融合

独立实验包，复用冻结cc8最优局部匹配F、WaPIRL整图G及原WBBS描述子；旧代码与历史产物只读。不含角向A或J，不重新训练、选模或搜索权重。

| 变体 | 固定评分 | 含义 |
| --- | --- | --- |
| FGW | `(0.5*F + 0.5*G)*W` | 完整三项融合 |
| GW | `G*W` | 删除局部F，剩余深度权重归一为1 |
| FW | `F*W` | 删除整图G，剩余深度权重归一为1 |

W使用原始WBBS分数 `B/min(Nq,Nc)*exp(-5*abs(Nq-Nc)/2061)`，N为去噪后点数，B为互为最近邻点对数，任一空集得0。不将W代入负指数，不按query做min-max归一化。W∈[0,1]与原J∈[exp(-2),1]的尺度不同；比较反映评分项整体替换。直接对照为首轮B1的 `no_angle`，即 `(0.5F+0.5G)*J`。

## 来源与计算

F/G来源与协议通过完成标记和签名绑定到 `artificial/b1_structure_all_20261117T193244+0800/`；FGJ对照来自 `artificial/b1_ablation_20261118T150805+0800/`。WBBS描述子来自相邻对比仓库 `artifacts/formal_exact_20261109_120738/evaluation/wbbs_valid/valid/prepared/` 与 `wbbs_test_r2/test/prepared/`，保持原参数、原始sample_id及完整候选库。

最近邻加速使用CPU float64、精确cKDTree（eps=0）；同分及近同分回退原cdist平方距离与最小行号argmin。完全重复描述只保留首次点用于匹配，归一化与惩罚仍使用原点数；先从小点集查forward，reverse只查被选中的目标，保持原BBP计数。所有查询的新W Top-500须与旧WBBS排名及分数完全一致；每split的37类各取最小query ID，对全部候选逐对与原CPU实现位级核对，并从原图复算该query描述子。

test的W对称性仅用于计算上三角，随后重建磁盘矩阵；F和最终融合仍逐query计算。三组共享一次完整候选W和F/G计算，绝不只融合旧Top-500。独立动态规划复核37类query的F最优配对、G及三组全候选Top-500；所有新排名另行复算全部query、各K、类别与分组的exact/coverage指标。

## 运行与恢复

在仓库根目录的原PyTorch环境中执行：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u -m fgw_fusion.run --workers 8
```

默认运行valid/test三组，共6项评价。valid固定500query×3598候选，test为7199query×7198候选；K=1/5/10/20/50/100/500。全候选W缓存较耗时，`--workers`限定1–32，默认8；机器资源允许时可提高，不改变分数。

启动时分配真实Asia/Shanghai时间身份和新目录 `artificial/fgw_fusion_YYYYMMDDTHHMMSS+0800/`；可显式传`--output-root`。`--dry-run`只显示计划，`--check-inputs`只读校验来源。中断后使用原目录：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u -m fgw_fusion.run --resume-root artificial/fgw_fusion_原启动时间 --workers 8
```

续跑要求代码、环境、来源、公式与协议签名不变。W逐行缓存和融合逐query分片校验后复用，已完成评价与运行保持原身份；未封存矩阵保留后由已验证行恢复，不覆盖历史运行。

## 产物、验证与登记

每个`valid/test/{FGW,GW,FW}`含时间身份、评价manifest、Top-500排名与F/G/W分项、两套指标、逐query/类别报告及完成标记。根目录保存独立来源审计、所有query指标复核、前后保护清单、`analysis/{split}`分组与10,000次分层配对bootstrap。valid分析比较GW/FW/FGJ；test另加已有WBBS双评价。旧WBBS valid没有coverage，不补造旧报告。

根`completed.json`与引用文件的哈希共同确定完成，不能仅以日志、进程退出或部分CSV判定。全部完成后登记6个新评价：

```bash
python -m fgw_fusion.register --run-root artificial/fgw_fusion_原启动时间 --apply
python results_catalog/build.py check
```

原FGJ/WBBS只作分析对照，不重复登记。三组均使用冻结自监督分支，标为`self_supervised`与`posthoc=true`。统计条件于当前模型、图库和类别构成；不作跨seed或速度优势宣称。共享计时不等于各变体独立部署耗时。

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s fgw_fusion/tests -v
```

公共协议、结果和登记说明见[项目说明](../doc/项目说明.md)，详细消融对比见[消融实验](../doc/消融实验.md)。
