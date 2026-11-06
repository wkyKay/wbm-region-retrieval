# WBM 局部区域检索实验

本项目研究 Mixed-WM38 上有方向的缺陷区域检索：query 的缺陷区域应在 candidate 中得到覆盖，candidate 可以含有额外缺陷。代码独立于同级 `wbm-wdm-matching/`，不下载预训练模型权重。

**实验设计、改进目的、代码位置和运行方式统一见 [experiments/README.md](experiments/README.md)。** 原来的分散说明已归档，旧链接保留跳转。

## 实验入口

| 方法 | 实验与入口 | 配置 |
|---|---|---|
| Proposed | [main](experiments/proposed/main/run.py)：原主实验；[a](experiments/proposed/a/run.py)、[b](experiments/proposed/b/run.py)、[c](experiments/proposed/c/run.py)、[d](experiments/proposed/d/run.py)、[e](experiments/proposed/e/run.py)、[f](experiments/proposed/f/run.py)、[g](experiments/proposed/g/run.py)：消融 | 每个实验目录的 `config.json` |
| Handcrafted | [a](experiments/handcrafted/a/run.py)：Zernike 面积加权；[b](experiments/handcrafted/b/run.py)：区域计数 | 每个实验目录的 `config.json` |
| Supervised | [a](experiments/supervised/a/run.py)：整图 pair 包含率回归；[b](experiments/supervised/b/run.py)：多标签分类与预测标签包含率检索 | 每个实验目录的 `config.json`，均含5%/10%/50%预算 |

每个入口只运行对应实验。A～D、E/F 的训练循环集中于 `experiments/proposed/_shared/`，其他通用模型、数据、指标工具仍在 `wbm_retrieval/`。旧实验代码路径、兼容链接和顶层 `configs/` 已删除，统一使用新入口；共享基础配置位于 `experiments/_shared/base.json`。

## 开始运行

使用已有的 PyTorch 环境；新环境安装 `requirements.txt`，监督多标签实验另需 `requirements-supervised.txt`。先确认数据和配置，再启动训练：

```bash
python experiments/proposed/g/run.py --dry-run
python experiments/supervised/b/run.py --dry-run
python experiments/handcrafted/a/run.py --dry-run

# G 训练及 valid 评价
python -u experiments/proposed/g/run.py --stage all --device cuda --output-root artifacts/experiments/proposed/g
# 固定无监督选模规则后，显式评价 test
python -u experiments/proposed/g/run.py --stage test --test-checkpoint best_nce --device cuda --output-root artifacts/experiments/proposed/g
```

默认数据为 `../data/wm38k/Wafer_Map_Datasets.npz`；`protocol/` 的冻结划分为 train/valid/test = 25,910/3,700/7,405。正式 test 使用2,000个固定 query，每个1,000个固定 candidate。详见 [协议说明](protocol/README.md)。

A～G 历史实现同时保存 `best_nce` 与使用 valid 标签选择的 `best_retrieval`。完整无监督主结果应预先指定 `best_nce`；本次目录整理没有改变训练或选模算法，严格禁止接触 valid 标签的流程还需要独立协议修改。各入口 `all` 是否包含 test 不完全相同，运行前请看 [统一运行说明](experiments/README.md#7-运行示例)。

## 验证与历史资料

```bash
python -m unittest discover -s tests -p 'test_experiment_layout.py' -v
python -m unittest discover -s tests -v
```

- [统一实验设计与比较边界](experiments/README.md)
- [原主实验详细方法和命令](experiments/_archive/MAIN_EXPERIMENT.md)
- [E/F 冻结模型无标签测试](experiments/_archive/EF_UNSUPERVISED_TEST.md)
- [原本地验证记录](experiments/_archive/VALIDATION.md)

运行产物继续保存于 `artifacts/`，每日人工记录位于 `record/`。项目已不依赖兼容符号链接。删除旧路径后源码指纹已变化，历史模型和结果保留，但新代码不能直接续跑旧签名目录；新训练请使用新的输出根。
