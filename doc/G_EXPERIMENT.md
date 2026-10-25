# 实验 G：运行与验证

更新日期：2026-10-25。

## 定义

- 保留 D 的 resize 88 + padding 4、96×96单通道输入、直角旋转、完整 ResNet-18 [2,2,2,2]、首层 stride=1、512维GAP与128维线性projector。
- 训练和验证NCE均使用 **1000** 个负样本。有放回采样；常规训练batch排除本batch并共享1000次抽样。极小数据恰好填满一个batch时，沿用逐anchor排除自身的分支。验证从训练memory中采样，不写memory。
- 手工Zernike与proposed都按 **匹配对数 / query保留区域数** 评分。分母经过原有min_area=5、top4筛选，不是原始图全部连通域数，也不是类别标签数。
- 沿用分数降序、query/candidate区域索引破同分的贪心一对一分配。每个区域最多匹配一次；不是最大基数匹配。面积和区域对置信度不进入最终求和；后者仍参与阈值筛选和匹配顺序。
- 空query/空candidate得0；candidate多余区域不扣分。map得分并列时按candidate ID升序；不偷偷使用原面积加权分数破同分。
- 标签相关性以及NDCG、mAP、FullCoveragePrecision等评价指标不变。每轮valid NCE选择`best_nce.pt`；周期valid检索使用G评分选择`best_retrieval.pt`（并列保留较早轮次）。random也使用G评分，但不参与最佳checkpoint选择。
- 默认100轮、batch=256、workers=4，valid池为500个query、每个1000个candidate。训练目标不使用标签，但`best_retrieval`使用valid标签选模型。`all`仅进行valid评价，test必须显式执行。

### 匹配阈值

`configs/ablation_g.json`中的`overrides.matching.method_thresholds`分别配置`handcrafted`和`proposed`的`shape_threshold`、`score_threshold`，默认均为0。区域对必须`score > 0`，且满足两个阈值；恰好等于正阈值时被接受。

按本次要求保留零阈值，不自动调参。Zernike的`exp(-distance/scale)`通常为正，所以在零阈值下，手工方法可能退化为只看区域数：`min(|Q|, |C|)/|Q|`。proposed在全部区域对为正时也有同样现象。若后续探索非零阈值，应只在valid上确定，然后固定后运行test，并使用新的输出目录。两种特征的分数分布不同，不假设同一个阈值一定合适。

基础配置继承的`matching.weight`在G中不参与计算；实际生效的聚合为`query_region_count`，方法级阈值覆盖基础阈值。`metrics.json`记录实际使用的匹配设置。

## 文件与隔离

| 文件 | 用途 |
|---|---|
| `configs/ablation_g.json` | 独立G配置、1000负样本、方法级阈值、输出目录 |
| `scripts/run_ablation_g.py` | prepare/train/evaluate/summarize/test/all及dry-run、resume入口 |
| `experiments/g/__init__.py` | 独立模块入口 |
| `experiments/g/matching.py` | 标量与批量计数匹配、匹配明细 |
| `experiments/g/suite.py` | 配置校验、缓存和签名、两种方法的评价及汇总 |
| `experiments/g/training.py` | G训练、1000负采样、按G评分选模型和恢复训练 |
| `tests/test_ablation_g.py` | 公式、接口、实际CPU训练、断点恢复与隔离测试 |
| `doc/ABLATION_A_G.md` | 替换旧轻量化G定义 |
| `doc/G_EXPERIMENT.md` | 本运行说明与验证记录 |

公共`wbm_retrieval/`源码、A～F配置、入口和现有结果均不修改。E/F会递归扫描`wbm_retrieval/**/*.py`计算源码哈希，因此G刻意放在目录外。共享的模型、数据视图、优化器、memory bank、指标等组件直接导入使用，不使用monkey patch。G独立保留必要的训练/评价流程，避免修改历史模块后使旧签名失效。

G的套件签名包含其配置、数据/协议指纹、G源码与公共依赖源码。不同负采样、阈值、batch或源码不允许混用同一输出目录；续跑校验suite、seed和组别签名。已完成训练再次`--resume`会跳过参数更新，允许重建评价结果。

## 远程运行

将上述文件连同原项目同步到服务器，进入服务器上的`wbm-region-retrieval`目录，并激活已有PyTorch CUDA环境。以下命令均在该项目根目录执行。无需同步本地临时测试环境或本地测试产物。

如环境尚未安装依赖：

```bash
python -m pip install -r requirements.txt
```

默认数据路径为项目旁的`../data/wm38k/Wafer_Map_Datasets.npz`，协议仍为项目的`protocol/split.csv`、`protocol/queries.csv`、`protocol/candidates.csv`。不要重新划分数据。如服务器路径不同，创建G专用的基础配置副本，使用绝对数据路径，并将G配置副本的`base_config`指向它；不要改动A～F正在使用的基础配置。后续所有命令使用同一个G配置副本。

先检查生效参数（不读取数据、不训练、不写结果）：

```bash
python scripts/run_ablation_g.py --config configs/ablation_g.json --dry-run
```

完整训练及valid评价（同时计算G的手工基线与proposed）：

```bash
python -u scripts/run_ablation_g.py \
  --config configs/ablation_g.json \
  --stage all --variants G --device cuda \
  --output-root artifacts/ablation_g_20261025
```

默认100轮，负样本始终为1000。显存不足时可在首次运行增加`--batch-size 64`等参数；这会改变训练设置，后续续跑和评价必须提供相同batch参数。`--num-workers`可以调整，默认4。

中断后继续（配置、seed、batch、输出目录保持一致）：

```bash
python -u scripts/run_ablation_g.py \
  --config configs/ablation_g.json \
  --stage all --variants G --device cuda --resume \
  --output-root artifacts/ablation_g_20261025
```

训练完成后，只重算valid评价：

```bash
python -u scripts/run_ablation_g.py \
  --config configs/ablation_g.json \
  --stage evaluate --variants G --device cuda \
  --output-root artifacts/ablation_g_20261025
```

固定配置和checkpoint选择规则后，再运行正式test（同时评价手工方法与所选proposed）：

```bash
python -u scripts/run_ablation_g.py \
  --config configs/ablation_g.json \
  --stage test --variants G --device cuda \
  --test-checkpoint best_retrieval \
  --output-root artifacts/ablation_g_20261025
```

若预先选择按NCE损失选模型，则使用`--test-checkpoint best_nce`。不要看到test结果后再决定哪条选择规则更好。test不进行优化更新。

还支持`--stage prepare`（只准备缓存）、`--stage train`（训练及周期valid检查）、`--stage summarize`（valid汇总）。`--seed`默认2026。改变seed时，后续运行始终显式传入同一seed。

## 查看结果

默认输出目录：`artifacts/ablation_g_20261025/`。

```text
suite_manifest.json
valid_pool.json
cache/
handcrafted/{valid,test}/
  metrics.json
  rankings.csv
  matches_top_k.json
  per_query.json
  per_query_metrics.csv
seed_2026/
  G/
    run_manifest.json
    history.json
    training_summary.json
    last.pt
    best_nce.pt
    best_retrieval.pt
    valid/{random,epoch_001,...,best_nce,best_retrieval}/
    test/{best_retrieval或best_nce}/
  comparison_valid.csv
  comparison_valid.json
  paired_valid.json
  comparison_test.csv
  comparison_test.json
  paired_test.json
```

test相关文件仅在显式test后生成。最终评价目录含与手工方法同格式的结果文件；random和周期epoch目录仅输出指标。

`rankings.csv`增加`matched_pairs`与`query_regions`，每行满足`similarity_score = matched_pairs / query_regions`（空query为0）。`matches_top_k.json`记录每对区域的索引、区域对分数、计数贡献1及归一化贡献`1/query_regions`，默认保存前10名候选。`training_summary.json`记录负样本1000和聚合方式。比较表包括all/single/mixed；配对统计为G的best_retrieval与G手工基线，不能混用A～F的面积加权手工结果。

## 本地小规模验证

可复现命令：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m unittest discover -s tests -p 'test_ablation_g.py' -v

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m unittest discover -s tests -v
```

G测试使用24张合成图（train/valid/test各8张），96×96输入、完整D编码器，batch=2，实际1000负样本和优化器更新。验证两轮训练的中断恢复与连续训练逐张量一致；CLI额外使用worker=1跑通all、resume及显式test。训练与验证NCE的实际负样本张量均检查为1000行。完整数据的CUDA训练及正式性能测试由远程运行完成。

2026-10-25实际验证结果：**全套70项测试通过，耗时288.960秒**，包含G的6项测试和原有A～F等64项回归测试。环境为Python 3.11.16、PyTorch 2.14.1、NumPy 2.4.6、SciPy 1.17.1，设备为CPU。为允许PyTorch多进程共享内存，完整测试在本机沙箱外运行；项目源码无需为此修改。

修改前后校验了28个原有代码/配置文件，SHA-256全部一致；`wbm_retrieval/`中的Python文件集合也未变，A～D及E/F的`code_hashes()`结果与修改前完全一致。G的dry-run、9个新增/修改文件的空白检查及`git diff --check`通过。

本地验证产物（被`artifacts/`规则忽略，不需要同步服务器）：

- `artifacts/g_local_validation_20261025/tests.log`：完整70项测试日志。
- `artifacts/g_local_validation_20261025/verification.json`：环境、结果、修改前后哈希与A～F源码指纹。
- `artifacts/g_local_validation_20261025/dry_run.json`：G正式默认配置的生效参数。

本地验证证明流程和计算规则可运行，不代表真实数据上的检索性能，也未执行完整CUDA训练。
