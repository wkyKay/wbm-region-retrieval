# B1 固定组件消融与独立验证

复用已完成 B1 `artificial/b1_structure_all_20261117T193244+0800/` 的冻结模型、特征和描述子；旧 B1 目录只读。本目录独立保存代码、配置、测试和启动器。公共协议和状态见[项目说明](../doc/项目说明.md)。

## 固定评分

令 F_o/F_g 为最优/贪心匹配的完整局部分数，G 为 WaPIRL 整图分数，A=exp(-0.5 D_angle)，J=exp(-2 D_J)。基线 B1=(0.5 F_o+0.5 G) A J。

| 配置 | 公式 | 解释 |
|---|---|---|
| no_local | G A J | 删除完整局部分支，包括其中的 CNN、区域径向与面积项 |
| no_global | F_o A J | 删除整图分支，剩余权重归一为1 |
| no_angle | (0.5 F_o+0.5 G) J | 删除外圈角向约束 |
| greedy | (0.5 F_g+0.5 G) A J | 仅替换配对算法 |
| structure_only | A J | 两项固定结构描述，不使用学习分数 |
| I_2 | (0.5 F_o+0.5 G) A exp(-2 D_I) | 同128维、同强度的去关联对照 |

valid 新增前五项；test 新增六项；valid I_2 和原 B0/R/T/I/J/J_only 只读复用。共11项新的双评价结果，不重训、不做新的参数选择。固定删除反映当前 B1 中的条件贡献，不代表该简化结构重新选参后的最优表现。各差值不能相加为独立贡献。该矩阵在查看 B1 test 后制定，明确记录 posthoc。

全体正式候选参与评分，按原 SHA 同分规则保存 Top500。每个 query 的 F/G/A 和结构距离只计算一次，供该split全部变体共享。exact 与 coverage 使用相同排名及完整候选分母；valid缺少WBBS coverage时显示NA。

2026-11-18新增独立配方 `no_global_angle`：同时删除整图G与角向A，固定 `S=F_o*exp(-2D_J)`，局部分支权重归一为1，保留最优匹配、F内的径向面积项及联合结构J。只运行valid/test各1项，不改变旧5/6项默认矩阵、不重训或调参。对照只读引用上一轮完成目录的no_global、no_angle、no_local及原B1；统计差值以新变体为anchor。J仍来自完整原图，不能称为纯局部方法。2026-11-18正式valid/test各1项及验证、统计已完成，输出 `artificial/b1_ablation_20261118T183806+0800/`；旧B1及首轮消融共513个保护文件保持不变。该轮37项消融测试、旧B1 26项及登记器19项通过，结论见项目说明。

新增独立配方 `no_local_angle`：同时删除完整局部F与角向A，固定 `S=G*exp(-2D_J)`，整图权重归一为1。仅新增valid/test各1项，不重新调参或训练；统计以新变体为anchor，与完整B1、首轮no_local/no_angle/no_global、J_only及可用WBBS对照比较。评分不使用局部区域分数或区域匹配；验证流程仍计算并保存原分项用于核对，不能据运行耗时宣称独立部署速度。2026-11-18正式valid/test各1项及独立验证、统计已全部完成，输出 `artificial/b1_ablation_20261118T190927+0800/`。47项消融测试、旧B1 26项及登记器19项通过，591个保护文件保持不变；结论见项目说明。

## 使用

激活原 PyTorch CUDA 环境后，在仓库根目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s b1_ablation/tests -v
PYTHONDONTWRITEBYTECODE=1 python b1_ablation/run.py --check-inputs
bash b1_ablation/launch_tmux.sh
```

仅执行新增双删除消融：

```bash
bash b1_ablation/launch_tmux.sh --recipe no_global_angle
```

执行仅保留G与J的消融，并保护上一轮F与J的运行产物：

```bash
bash b1_ablation/launch_tmux.sh --recipe no_local_angle --protected-root artificial/b1_ablation_20261118T183806+0800
```

`--protected-root`可重复指定已完成运行，仅纳入前后文件保护，不自动加入统计对照；路径及manifest哈希进入新运行身份。恢复时需保留原recipe和这些路径。

默认只读对照为 `artificial/b1_ablation_20261118T150805+0800/`，可用 `--reference-ablation-root` 指定同一B1来源的已完成旧组件消融。新运行前后核对原B1和该对照目录全部文件。恢复独立配方时也必须传对应的 `--recipe`；源码变化后旧运行需检出其原Git版本续跑，不能修改历史签名。

启动器默认运行完整 valid→test→验证→统计流程，创建 `artificial/b1_ablation_真实启动时间/`，打印 tmux 会话和输出位置。实际评价各自保存带 Asia/Shanghai 时区的 run_id/started_at。日志为 launch.log/run.log，状态为 status.json；以完成标记及验证文件判定完成。

`--dry-run` 仅打印计划；`--check-inputs` 只验证来源；`--audit-only` 在新的分析目录执行真实数据抽样审计，不产生新排名。恢复使用：

```text
python b1_ablation/run.py --resume-root artificial/b1_ablation_实际时间 --device cuda
```

恢复要求代码、配置、来源和后端相同。已完成结果只在哈希一致时复用；未完成 attempt 原样保留。改变科学设置必须使用新目录。不能用旧Top500做新消融的候选名单。

## 验证与统计

- 原始 B1 完成标记、文件哈希、选择与身份链，F/WaPIRL缓存、冻结协议和代码签名先验证；旧 B1 代码与产物在新运行前后核对哈希、大小、mtime。
- valid/test各37类固定选择最小原始query ID。独立矩阵公式重建query的TV、归一化、软直方图；用原图重建gallery角向，并从冻结特征以独立动态规划复算区域最优匹配，对完整gallery重算原B1 Top500。
- 源抽查要求排名身份和顺序一致。数值容差不能静默豁免排名变化。gallery的TV和CNN特征仍是冻结输入；抽样审计不等于全query重新提取或评分。
- 新输出独立复算分项公式、SHA排序、全部query各K的exact/coverage及总体、37类、宏平均、单混合与标签数分组，拒绝重复query与错误候选。
- 跨方法用原始query_id合并、冻结multi-hot分组，不直接合并class_id；exact索引列表和coverage位串统一转换。
- 主指标exact P@10，次指标mAP@100、Coverage-nDCG@10。37类分层配对bootstrap 10000次、seed2026，同一抽样索引用于各比较。区间为固定模型/图库/类别构成下的事后描述，未校正多重比较，不包含训练seed或图库变化，也不自动消除共享图库依赖。
- Top1/10错误分为exact、仅多、仅少、同时缺失和额外、无共同标签。案例同时保留改善与退化，按固定规则选择并记录分项。
- 保存共享分量、结构距离和各公式排序计时，明确排除编码、TV与IO；不将共享批次耗时当某一变体独立耗时，不据此宣称相对WBBS的速度优势。

## 输出与登记

每项结果含 run_identity.json、evaluation_manifest.json、rankings.csv、metrics.json、coverage_metrics.json、逐query/类别报告及completed.json。根目录保存原始计划、来源清单、抽样审计、分组验证、旧文件保护报告及analysis/{valid,test}/统计。完整排名、模型和缓存不加入Git。

完整完成后生成catalog_sources.json提案；以下命令验证后登记结果（旧配方含原B1，新双删除配方仅登记两项新评价，保留已有来源元数据），重复登记保留原身份，更新失败会恢复备份：

```text
python b1_ablation/register.py --run-root artificial/b1_ablation_实际时间 --apply
python results_catalog/build.py check
```

共享登记器和生成汇总按项目规则同步。原J_only对照按实际评分标记为untrained；历史登记提案原样保留。若共享来源元数据已经校正，重复应用旧提案会拒绝冲突，避免回退人工核实的监督信息。2026-11-18首轮组件消融valid 5项/test 6项、独立验证及统计已全部完成，输出为 `artificial/b1_ablation_20261118T150805+0800/`。首轮新增29项、旧B1 26项（含CUDA）、登记器19项测试通过；74个真实query全候选抽查一致，旧B1 327个文件保持不变。结论统一见[项目说明的实验结果](../doc/项目说明.md#实验结果)。
