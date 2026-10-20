# E/F 本地验证记录

日期：2026-10-20。对应设计与远程命令见 [ABLATION_A_G.md](ABLATION_A_G.md)。

## 环境

- macOS 27.0.1，Apple ARM64，CPU。
- 独立临时环境 `/private/tmp/wbm-ef-env`，未修改服务器或用户现有 Python 环境。
- Python 3.12.14；PyTorch 2.14.1；NumPy 2.5.4；SciPy 1.18.1。
- 真实数据检查使用 CPU、batch16、workers0；多进程 CLI 另用2个spawn worker检查。
- 未在本机执行 CUDA 或完整100轮训练。

## 自动化测试

最终版本执行完整测试集，共59项：原有52项、新增7项。**59项全部通过，耗时164.542秒。**

最终复验命令：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
MPLCONFIGDIR=/private/tmp/wbm-ef-mpl \
/private/tmp/wbm-ef-env/bin/python -m unittest discover -s tests -v
```

前期沙箱内执行时，多进程CLI曾因禁止 `torch_shm_manager` 报 `Operation not permitted`。最终完整复验在沙箱外执行，保留两个spawn worker，没有降低workers或跳过断言。日志为 `artifacts/ef_local_validation_setup_20261020/unittest_final.log`。

核心验证内容：

- E与原C的原始/增强输入逐元素一致；D/E与原D初始化一致。
- F固定窗口的每个输出像素与独立坐标采样参考一致；通道含义、越界补零、截断审计正确。
- F除首层卷积以外与D共享相同初始参数；D/E/F的GAP前特征为6×6，参数量符合设计。独立首层初始化不调用CUDA重新设种，且不会改变后续CPU随机状态。
- E/F都进行了真实CPU反向传播和优化，模型参数确实改变。
- E/F各训练2轮：中断1轮后恢复与连续训练的最终模型、memory逐元素相同，scheduler与优化步数相同。
- 完成态续跑不修改checkpoint；缺失的完成态JSON报告可恢复；非resume重复启动明确拒绝。
- 学习方法的完整rankings经原有评价器重算后，指标逐项一致。
- 默认all不评价test；显式test不修改训练权重；best_nce测试也进入汇总。
- 两个spawn worker完整执行CLI的准备、训练、评价、汇总。
- 原入口生成的历史D结果可被新入口引用，历史目录内所有文件的哈希和文件集合保持不变。
- 科学设置不一致、输出覆盖风险、缓存篡改会被拒绝。

原始测试日志保存在本地 `artifacts/ef_local_validation_setup_20261020/`。

## 真实 Mixed-WM38 子集检查

只读复用已有 `mvp_mps_300_top4_opening_20261018` 的300张真实晶圆子集及划分。最终版本输出独立放在 `artifacts/ef_local_validation_300_final_20261020`；设置与源实验不同，因此不复用源模型、特征或ranking。

本地专用配置为 `artifacts/ef_local_validation_setup_20261020/suite.json`。E/F各训练1轮，固定12个valid query、每个20个candidate；负样本仍为5000个有放回采样。实际命令：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
MPLCONFIGDIR=/private/tmp/wbm-ef-mpl \
/private/tmp/wbm-ef-env/bin/python -u scripts/run_ablation_ef.py \
  --config artifacts/ef_local_validation_setup_20261020/suite.json \
  --stage all --variants E F --device cpu --num-workers 0
```

数据和F窗口审计：

| split | 晶圆数 | 区域数 | F被截断区域数 | 总前景保留比例 |
|---|---:|---:|---:|---:|
| train | 210 | 776 | 50 | 99.0709% |
| valid | 30 | 117 | 6 | 99.2001% |
| test | 60 | 229 | 17 | 98.8561% |

test仅构建输入和窗口审计，不评价检索。截断结果按约定如实保留，没有通过自动扩窗改变F定义。全量远程数据须重新查看审计，不能把这些比例外推。

最终版本完整执行成功，进程退出码为0：

| 检查 | E | F |
|---|---|---|
| 完成训练轮数 | 1 | 1 |
| 优化步数 | 49 | 49 |
| 初始与最终参数哈希不同 | 是 | 是 |
| train loss、valid NCE、NDCG均有限 | 是 | 是 |
| 两种最佳checkpoint各自产出ranking | 各240行，12个query | 各240行，12个query |
| 完成态resume保留所有checkpoint | 3个文件哈希不变 | 3个文件哈希不变 |

再次执行`--stage train --resume`成功跳过已完成的优化；全部6个checkpoint逐文件SHA-256不变。原有代码、配置、协议及复用数据共51个文件的SHA-256与运行前一致。原有主入口和A～D入口的`--dry-run`也均成功。

完整执行状态、checkpoint与源文件校验记录在本地 `artifacts/ef_local_validation_setup_20261020/validation.json`。本检查的检索指标只用于确认输出是有限且有效的数值，不作为正式方法性能结论。

## 使用范围

这些检查覆盖实现正确性、实际训练更新、恢复、评价协议和CLI多进程，不覆盖Linux CUDA驱动兼容性、完整训练收敛或远程GPU显存容量。正式远程运行需使用 `configs/ablation_ef.json` 的100轮配置，不要误用本地一轮子集配置。
