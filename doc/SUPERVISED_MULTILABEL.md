# Supervised：整图多标签分类与预测标签包含率检索

## 实验定义

本实验独立于旧 `supervised_global.py` 的 pair-MSE 回归。代码位于 `experiments/supervised/`，入口为 `scripts/run_supervised_multilabel.py`；不修改 `wbm_retrieval/`、A～G、旧入口或旧配置，历史训练签名与结果保持原样。

- 数据：现有 Mixed-WM38 NPZ 的 `arr_0` map 与 `arr_1` 八维 multi-hot 标签。仍使用 `protocol/` 的冻结划分与测试 query/candidate 清单；正常/空真实标签图不在此协议内。
- 输入：完整 map 的 `raw==2` 二值缺陷图，补方后最近邻 resize 为 `1×96×96`，不提取区域。保留现有 `whole_map_patch` 语义，原始值3不视为缺陷。
- 模型：复用主入口 proposed 的 `Encoder(512)`：单通道完整 ResNet-18、首层7×7 stride=2、现有 maxpool、GAP；增加 `Linear(512,8)`。同 seed 的编码器初始化与 proposed 逐参数相同；保存 `initial_encoder_sha256`。这是与 **proposed-main** 对齐，**不是 G 的 stride=1 编码器**。配置明确锁定 `encoder_reference=proposed-main`。
- 训练：8维 logits 输入无类别加权的 `BCEWithLogitsLoss`，编码器与分类头共同更新。每张训练图每轮随机选择0/90/180/270度旋转，标签不变；验证/测试不增强。
- 预算：5%、10%、50%，复用已有 signature-stratified 子集算法和 seed；每个标签组合至少一张，实际预算记录于 `budget.json`。可用 `budget_source` 逐项验证旧实验的训练ID一致。
- 优化：SGD，lr=0.01，momentum=0.9，weight_decay=0.001，现有 cosine 调度，batch256，100轮。每轮遍历该预算的 map 一次，无人工配对/重采样。不同预算优化步数不同，和区域 proposed 的训练工作量也不等价。
- 推理：`sigmoid(logits)>=0.5` 得硬标签。概率仅用于分类AP/PR曲线和审计，不参与检索排序。

## 排序与评价

设 query、candidate 的预测标签为 `q_hat`、`c_hat`，排序分数为：

```text
score(Q,C) = sum(q_hat * c_hat) / sum(q_hat)
```

query 预测全零时所有候选得0，保留该 query；candidate 预测全零时得0。严格按分数降序、candidate ID升序排序，禁止概率或真实标签作为二级排序。记录空预测率、全零得分query比例、每query不同分数数量和最高分并列数量。

真实测试标签仅在评价阶段用于构造相关性，**不作为 query 或 candidate 的预测标签**。复用原 `evaluate_ranking`，输出 K=1/5/10/20 的 ContainmentNDCG、MeanContainment、FullCoveragePrecision/Recall/Hit，以及 mAP_full、MRR_full、逐query指标和bootstrap 95%区间。single/mixed组额外输出均值，组报告不声称包含bootstrap区间。

分类指标覆盖完整 valid/test 划分：Macro/Micro-F1、Macro/Micro Precision/Recall、Samples-F1、Subset Accuracy（8位全部正确）、Hamming Loss、每类P/R/F1/support/AP、Macro-AP、空标签预测率、标签数量MAE、BCE。分类 `macro_ap` 与检索 `mAP_full` 分开命名。P/R/F1分母为零时计0；某类别无正样本时AP为null，从Macro-AP排除并报告有效类别数；不存在的single/mixed组记null。类名先使用 `label_0`～`label_7`，不猜测未经核验的名称映射。

## 模型选择与训练诊断

每轮在全部valid图上计算分类指标；按 **valid Macro-F1最大 → BCE最小 → 更早epoch** 选择 `best_classifier.pt`。另存 `best_loss.pt`、`last.pt`。最终分类和检索统一使用 `best_classifier.pt`，不根据test挑模型。

固定训练诊断子集最多512张，在每轮结束后用eval模式、无增强计算BCE/F1/Exact Match；与valid计算条件一致。训练期间累计的在线BCE单独记录，不能把它当成相同模型状态下的轮末损失。

第1、5、10、…、100轮，在固定500个valid query、每个1000个candidate上记录检索指标（训练监测阶段不做bootstrap）。候选池只根据ID抽样，固定seed，不按标签挑候选；池大小不足时明确报错，不静默缩小。检索曲线只用于诊断，不参与本版checkpoint选择。

默认 `all` 仅做训练和valid评价；固定方案后显式执行 `test`。测试阶段不执行优化，不更新checkpoint。训练未达到配置轮数时拒绝最终评价。

## 远程运行（Linux/CUDA）

将以下新增文件同步到远程项目同名路径：

```text
experiments/supervised/                 # 包内全部 .py
scripts/run_supervised_multilabel.py
configs/supervised_multilabel.json
requirements-supervised.txt
doc/SUPERVISED_MULTILABEL.md
```

项目原有的 `wbm_retrieval/`、`requirements.txt`、`configs/experiment_a.json` 和 `protocol/` 也须存在。不要复制本地的训练产物或本地绝对路径配置作为远程正式配置。

在已有CUDA环境下：

```bash
cd /data/wangkaiyan2024/wbm-region-retrieval

python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()'
python -c 'import numpy, scipy, matplotlib, sklearn; print(sklearn.__version__)'
```

若仅缺scikit-learn：

```bash
python -m pip install 'scikit-learn>=1.3,<2'
```

新环境可使用 `python -m pip install -r requirements-supervised.txt`；PyTorch应选与服务器CUDA兼容的版本，已有可用环境无需重新安装torch。

**先检查实际路径与配置：**

```bash
python scripts/run_supervised_multilabel.py \
  --config configs/supervised_multilabel.json --dry-run
```

默认数据路径从 `configs/experiment_a.json` 读取，解析为项目根目录旁的 `../data/wm38k/Wafer_Map_Datasets.npz`。若远程不同，复制原配置为新base配置，只修改该副本的数据路径，再将新supervised配置的 `base_config` 指向副本；不修改旧实验正在使用的配置。配置文件路径和内部相对路径均以项目根目录为准。

**训练三组预算并生成valid完整报告：**

```bash
mkdir -p artifacts/supervised_logs
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
nohup python -u scripts/run_supervised_multilabel.py \
  --config configs/supervised_multilabel.json --stage all \
  --device cuda --num-workers 4 \
  > artifacts/supervised_logs/train_seed2026.log 2>&1 &
```

日志放在独立目录，不要提前在新实验输出根目录写入 `run.log`；新入口会拒绝不认识的非空目录，防止覆盖已有结果。

```bash
tail -f artifacts/supervised_logs/train_seed2026.log
```

**中断恢复：**使用相同配置和output root，增加 `--resume`。

```bash
python -u scripts/run_supervised_multilabel.py \
  --config configs/supervised_multilabel.json --stage all \
  --device cuda --num-workers 4 --resume
```

模型、优化器、scheduler、历史、CPU/CUDA/MPS RNG一并恢复；完成态跳过优化。科学设置、batch、源码、输入文件变更会拒绝跨版本恢复；device/workers/CPU线程数可切换，但不承诺跨设备位级一致。不同torch版本也拒绝继续训练。

**固定实验设计后，生成test分类与检索报告：**

```bash
python -u scripts/run_supervised_multilabel.py \
  --config configs/supervised_multilabel.json --stage test \
  --device cuda --num-workers 4
```

**重新绘图（使用已保存的历史与预测，不重新训练/推理）：**

```bash
python scripts/run_supervised_multilabel.py \
  --config configs/supervised_multilabel.json --stage plot --device cpu --num-workers 0
```

可选 `--stage prepare` 只准备缓存，`--stage train` 只训练及记录监测指标，`--stage evaluate` 只评价选定分类器的valid表现。`--output-root` 可指定全新输出目录，路径保护及签名校验仍有效。

如需核验旧supervised的标签子集，在新配置中设置：

```json
"budget_source": "artifacts/experiment_a_top4_opening_seed2026"
```

它只读取 `models/global_*pct/labeled_train_ids.csv` 并比较ID；不会读取或修改旧权重。预算来源属于实验身份，确定后不要在同一目录中更改。CLI也支持 `--budget-source`，使用时后续所有阶段需保持一致。

## 输出目录与figures

```text
artifacts/supervised_multilabel_seed2026/
├── experiment_manifest.json / resolved_config.json
├── cache/                              # 独立整图缓存、ID和标签
├── protocol/                           # 固定valid检索池
├── supervised_5pct/                    # 10pct、50pct结构相同
│   ├── models/{best_classifier,best_loss,last}.pt
│   ├── budget.json / labeled_train_ids.csv / train_diagnostic_ids.json
│   ├── history.json / training_summary.json / run_manifest.json
│   ├── classification/{valid,test}/
│   │   ├── metrics.json / per_class_metrics.csv
│   │   ├── predictions.npz / predictions.csv
│   │   └── provenance.json / case_selection.json
│   ├── retrieval/{valid,test}/
│   │   ├── rankings.csv / metrics.json / per_query_metrics.csv
│   │   ├── group_metrics.json / retrieval_info.json / tie_audit.csv
│   │   └── provenance.json
│   ├── retrieval/valid_monitor/epoch_*/
│   └── figures/
│       ├── learning_curves.{png,pdf}
│       ├── valid_retrieval_curves.{png,pdf}
│       └── {valid,test}/
│           ├── per_class_metrics.{png,pdf}
│           ├── multilabel_confusion.{png,pdf}
│           ├── precision_recall.{png,pdf}
│           ├── label_count_confusion.{png,pdf}
│           └── prediction_examples.{png,pdf}
├── {valid,test}_classification_comparison.{csv,json}
├── {valid,test}_retrieval_comparison.{csv,json}
└── figures/{valid,test}_budget_comparison.{png,pdf}
```

PR和AP使用保存的概率；其余硬标签指标与检索使用相同0.5阈值。预测案例取每组按ID排序的前四张，组允许重叠，ID与规则写入JSON。全图案例显示实际送入分类器的完整map mask。不存在某组案例时明确留空。

## 本地小规模验证

`configs/supervised_multilabel_mvp.json` 只读使用已有300张真实晶圆子集（train210/valid30/test60），明确核验旧模型使用的预算ID。三组各训练2轮，CPU、batch16、workers0、固定12个valid query×20候选，输出独立于源MVP。

```bash
python -m unittest discover -s tests -p 'test_supervised_multilabel.py' -v
python -u scripts/run_supervised_multilabel.py --config configs/supervised_multilabel_mvp.json --stage all
python -u scripts/run_supervised_multilabel.py --config configs/supervised_multilabel_mvp.json --stage test
```

该配置依赖已有 `artifacts/mvp_mps_300_top4_opening_20261018` 数据与协议，主要用于本地验证；远程正式训练应使用100轮的 `configs/supervised_multilabel.json`。本地测试结果及保护检查另见 `SUPERVISED_MULTILABEL_LOCAL_VALIDATION.md`；小规模结果仅验证实现可运行，不用于推断正式性能。
