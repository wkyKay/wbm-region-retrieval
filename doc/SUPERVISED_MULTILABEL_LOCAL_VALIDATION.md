# Supervised 多标签分类：本地验证记录

日期：2026-10-25。正式设计、部署清单及远程命令见 [SUPERVISED_MULTILABEL.md](SUPERVISED_MULTILABEL.md)。

## 环境与范围

- macOS、CPU，Python 3.11.16，PyTorch 2.14.1，NumPy 2.4.6，scikit-learn 1.9.1。
- 复用临时环境 `/private/tmp/wbm-g-env`；仅向该临时环境安装scikit-learn及其依赖。
- 使用已有300张真实Mixed-WM38子集：train210、valid30、test60；不重新切分或修改源文件。
- 输入为真实整图mask，保持完整ResNet-18和96×96输入；三组预算各训练2轮，batch16、CPU线程2。
- 训练workers=0；最终test另用2个spawn worker成功验证。未执行Linux CUDA或完整100轮收敛测试。

## 小规模训练结果

| 预算 | 选中train map数 | 轮数 | 优化步数 | 选中分类器epoch | 参数发生更新 |
|---|---:|---:|---:|---:|---|
| 5% | 10 | 2 | 2 | 2 | 是 |
| 10% | 20 | 2 | 4 | 2 | 是 |
| 50% | 104 | 2 | 14 | 1 | 是 |

预算ID与旧MVP的 `models/global_*pct/labeled_train_ids.csv` 逐项一致。50%选中第1轮是因为其valid Macro-F1更高，证明选模遵循分类规则，而不是无条件选择最后一轮或最低BCE。

三组均产出：

- valid全部30张map的分类预测；固定12个query×20候选，240行ranking。
- test全部60张map的分类预测；冻结60个query×59候选，3,540行ranking。
- 分类指标、每类指标、single/mixed分组、8维概率和硬标签、检索指标及置信区间、逐query指标、同分审计。
- 共38张PNG和38份PDF；已人工查看学习曲线、预测案例、多标签混淆矩阵、PR曲线的代表性输出。
- `--stage plot` 可从保存的历史和预测重建报告。

两轮只验证真实训练与完整输出流程。5%/10%模型在这个极小测试中预测为空；50%的选中模型主要预测同一个类别。三组test NDCG@10均约0.1724，反映本次预测排序大量同分，不代表正式方法性能。保留这些真实输出，不调整阈值或修改排序来改善小规模指标。

## 自动化与保护检查

新增7项专项测试通过，覆盖：

1. 编码器架构、初始化哈希与主入口proposed一致，输出为8维logits。
2. 标签包含率方向性、空预测、同分candidate ID排序。
3. 排序函数不访问真实标签；真实标签仅用于评价。
4. 已知分类指标、缺失类别AP/null策略、空预测指标。
5. Macro-F1/BCE/epoch的checkpoint选择优先级。
6. 不认识的非空输出目录拒绝写入，配置变化与缓存篡改拒绝复用。
7. 真实反向传播、断点恢复与连续训练的全部模型张量位级一致；完成态resume不改checkpoint；ranking由原评价器重算完全一致；未显式test时不生成test评价。

全套回归初次执行77项：75项通过，原E/F、G的两个多进程CLI测试因沙箱阻止 `torch_shm_manager` 创建共享内存而失败；使用相同测试和worker设置在沙箱外复验，**两项均通过**，不修改实验代码或跳过断言。因此77项测试均获得通过结果；初次完整运行161.237秒，两个多进程测试复验149.163秒。具体结果记录于下述日志。

- 训练前后核对 **133个受保护文件**：既有公共代码、实验代码、配置、协议及源MVP文件SHA-256全部未变。
- 三组共 **9个checkpoint** 在最终test与完成态resume前后SHA-256全部未变。
- 完成态resume只恢复缺失的报告/重画曲线，不执行优化。
- 新增实现与旧实验使用独立源码目录、配置、缓存、模型和结果目录。

## 可复现命令与日志

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/private/tmp/wbm-supervised-mpl \
/private/tmp/wbm-g-env/bin/python -m unittest discover -s tests -p 'test_supervised_multilabel.py' -v

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/private/tmp/wbm-supervised-mpl \
/private/tmp/wbm-g-env/bin/python -u scripts/run_supervised_multilabel.py \
  --config configs/supervised_multilabel_mvp.json --stage all

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/private/tmp/wbm-supervised-mpl \
/private/tmp/wbm-g-env/bin/python -u scripts/run_supervised_multilabel.py \
  --config configs/supervised_multilabel_mvp.json --stage test --num-workers 2

/private/tmp/wbm-g-env/bin/python scripts/run_supervised_multilabel.py \
  --config configs/supervised_multilabel_mvp.json --stage train --resume
```

真实输出：`artifacts/supervised_multilabel_mvp_20261025/`。

日志与机器可读检查：`artifacts/supervised_multilabel_validation_20261025/`：

- `unittest.log`：新增专项测试。
- `regression.log`：全套77项初次运行，保留共享内存限制的原始错误。
- `regression_workers_retry.log`：两个旧多进程测试的沙箱外复验。
- `mvp_all.log`、`mvp_test_workers2.log`、`mvp_resume.log`、`mvp_plot.log`：各阶段实际运行。
- `protected_before.json`、`checkpoints_before.json`：文件及checkpoint初始哈希。
- `validation.json`：训练步数、图表数量、分类与ranking覆盖范围、环境及保护检查。

复用此MVP目录时必须保持源码/科学设置不变；要修改配置或代码后重跑，请使用全新的 `--output-root`。
