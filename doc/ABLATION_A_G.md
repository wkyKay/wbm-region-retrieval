# A～G：局部区域检索实验设计、作用与运行方法

更新日期：2026-10-20。

本文区分**已实现的 A～F**与**仅设计、尚未实现的 G**。A～D 继续使用原入口；本次增加独立 E/F 入口，并支持可选 D 对照。F0 仅是后续解释性对照，当前不提供运行选项。本地可运行性验证不代表远程完整数据的性能结果。

## 1. 共同问题与比较原则

目标是在 Mixed-WM38 上学习跨晶圆可比较的区域表示，再通过 query 方向的区域覆盖进行检索。三个需要分别检验的问题是：

1. **输入**：形状在画布中的占比、原始尺度和周围上下文如何影响表示？
2. **学习**：增强与网络空间采样是否帮助训练获得超过随机初始化的检索收益？
3. **容量**：较浅、较小的模型是否更适合当前区域数据？

随机初始化 CNN 不等于随机排序；连通域筛选、输入规范化、网络结构和匹配器已经包含先验。每个学习方案必须报告自身的 random、best_nce、best_retrieval，不能只比较训练后的绝对分数。

训练和推理都固定原始 8 连通域、min_area=5、最多 top4；不恢复旧 matching 的 ring-aware proposal。面积权重使用完整原始区域的像素面积。E/F 不修改主实验的开运算配置，只在独立实验配置中关闭开运算。

## 2. A～G 实验总表

| 组别 | 原始分支和正式检索输入 | 增强分支 | 编码器 | 主要对照及作用 | 状态 |
|---|---|---|---|---|---|
| A | bbox 裁剪→补方及原网格每边 padding 4→最近邻 resize 96 | 直角旋转 | ResNet-18，首层 stride=2 | 原输入基线 | 原入口已实现 |
| B | bbox 裁剪→仅补方→最近邻 resize 88→最终每边 padding 4 | 直角旋转 | 同 A | B−A：固定输出边距、统一长边占比 | 原入口已实现 |
| C | 同 B | 从原始 mask 生成随机80～88尺寸、各轴平移−2～2，再旋转 | 同 B | C−B：轻度尺度＋平移增强的联合作用 | 原入口已实现 |
| D | 同 B | 直角旋转 | ResNet-18，首层 stride=1 | D−B：减弱首层下采样 | 两个入口均可运行 |
| E | 同 D | 同 C | 同 D | E−D：增强在 stride=1 下的作用；E−C：同增强下的 stride 作用 | 本次实现 |
| F | 以区域质心为中心，三通道固定52×52窗口，越界补0，再最近邻 resize 96 | 三通道同步直角旋转 | D 的结构，仅首层输入通道1→3 | F−D：旧版输入方案整体的作用 | 本次实现 |
| G | 同 D | 同 D | 每阶段残差块从[2,2,2,2]改为[1,1,1,1]，首层 stride=1 | G−D：较浅、较小的残差编码器是否适合本任务 | **仅设计，未实现** |

所有旋转为 0/90/180/270°，不是任意角旋转。A～F 均关闭开运算、丢点与加噪。不同实验改变输入与结构，不代表整个方法仅有一个因素不同；只有表中相应的成对对照可以支持对应解释。

### E：完整的 stride × 增强组合对照

| | 仅旋转 | 旋转＋尺度/平移 |
|---|---|---|
| stride=2 | B | C |
| stride=1 | D | E |

E 的原始分支始终使用标准 88＋4 输入。增强分支从原始区域 mask 裁剪补方，尺寸在整数80～88均匀采样，两个轴的平移在整数−2～2独立采样，最后旋转。画布为96×96，不截断前景，不对标准 patch 二次 resize。给定 seed、epoch、token 后增强可复现；验证视图固定，训练视图随 epoch 变化。

E 和 D 的全部初始参数一致；E 和 C 的增强实现等价。E−D 和 C−B 可比较增强是否与 stride 交互，不能预设二者收益可直接相加。模型选择仍按既定 valid 检索规则，不因为 E 而改动训练轮数或负样本策略。

### F：准确区分两种 padding

F 的三个二值通道按以下固定顺序排列：

1. 当前窗口内的全部缺陷：`raw == 2`；
2. 当前目标连通域的 mask；
3. 有效晶圆范围：`(raw == 1) | (raw == 2)`。

原始图中少量值3不改写成缺陷或有效晶圆。所有通道使用同一个中心、窗口、resize 索引和旋转。

F 使用原始区域前景坐标的质心，各轴按 Python `round` 取整，再减去26得到52×52窗口左上角。窗口超出原始晶圆数组的部分填0，随后整体最近邻 resize 到96×96。它**不按 bbox 裁剪，不补“额外4像素”，不使用88＋4输入**；配置中的 `content_size=88` 仅为共享结构字段，在 F 固定窗口路径中不使用。

这一输入保留原网格尺度、周围其他缺陷和晶圆边界，因此 F 表示的是**带上下文的区域特征**。F−D 是输入方案的联合比较，不能把收益单独归因到通道数，也不能据此证明纯局部形状表示更好。F 仍使用当前 top4 连通域和 shape-only 匹配，不混入旧版几何类型、位置、尺度打分。

质心居中的52×52窗口可能截断不对称区域。实现不偷偷扩大窗口，而是保存：

- `cache/{train,valid,test}_F_window_audit.csv`：map_id、token_id、原始面积、窗口内保留像素、保留比例、是否截断、窗口左上角。
- `cache/F_window_audit.json`：各 split 的区域数、截断数、平均/最低保留比例和总前景保留比例。

保留比例在 **resize 前的原始网格**上计算；晶圆匹配仍以完整原始区域面积加权，不使用窗口内面积或放大后的面积。审计结果用于解释输入变化，不自动修改实验设置。

F 的初始化先按 D 的顺序构建完整单通道编码器和 projector，再用独立、固定种子的 Kaiming 初始化替换首层三通道卷积。该过程只临时设置 CPU 生成器，不重置 CUDA/MPS 随机状态。因此 D/E/F 中形状相同的其余参数逐元素一致，F 首层单独初始化；不复制、扩展训练后的 D 权重。

若 F 有改善，后续可增加 **F0：相同固定窗口，仅输入目标区域通道**。F0−D 检验窗口/尺度处理，F−F0 检验上下文。F0 尚未实现，不是当前可运行实验。

### G：轻量化假设与边界

G 保持 D 的输入、增强、通道数[64,128,256,512]、阶段下采样位置、512维 GAP 输出和128维线性 projector，仅把每阶段两个残差块减为一个。这样 GAP 前仍为6×6，输出维度不变。减少深度也改变感受野和优化过程，不能把结果只归因于参数个数。

| 模型 | 编码器参数量 | 含 projector 的训练参数量 |
|---|---:|---:|
| D/E：单通道 ResNet-18 | 11,170,240 | 11,235,904 |
| F：三通道 ResNet-18 | 11,176,512 | 11,242,176 |
| G：每阶段一个残差块，设计值 | 4,899,520 | 4,965,184 |
| 旧 matching 默认小 CNN，供参考 | 615,904 | 747,488 |

统计包含可训练权重和偏置，不包含 BN 运行统计。B 和 D 的参数量相同，已有 D 的改善不能由参数减少解释。G 只作为独立容量/结构对照，当前 CLI 不接受 `G`。

## 3. E/F 的固定训练与评价协议

- 数据、split 和正式 test query/candidate 清单与 A～D 相同。默认 train/valid/test 为25,910/3,700/7,405张有标签图，标签不进入 NCE 梯度训练。
- 单GPU，FP32；默认区域 batch=256，两视图拼接后一次前向512张 patch。workers=4，正式训练100轮，不早停。
- SGD lr=0.01、momentum=0.9、weight_decay=0.001；沿用零 warmup 的 WaPIRL cosine 调度。
- 原始分支与增强分支的 memory NCE 等权平均，temperature=0.07；5000个有放回负样本，训练排除当前 batch；memory EMA=0.5。投影和 EMA 不强制归一化，只在 cosine 损失中归一化。
- 验证使用固定视图与固定负采样随机数，valid/test 不写 memory、不参与优化。
- h512 cosine 截断到[0,1]，阈值0；贪心一对一分配；query侧 sqrt(area) 加权；主实验不混合位置/尺度。
- 固定 valid 池：500 query、每个1000 candidate，排除自身，池seed=2026，抽样不使用标签。与 A～D 的池生成过程一致。
- 第1、5、10、…、100轮评价 valid 检索，按 `ContainmentNDCG@10` 最大值保存 `best_retrieval.pt`，并列保留较早轮次。
- 每轮计算 valid NCE，保存最低损失 `best_nce.pt`。random 只做诊断，不参与最佳训练 checkpoint 选择。
- `best_retrieval` 使用 valid 标签选模型，不能描述为“全流程完全不使用标签”。`all` 不评价 test；固定方案后显式执行 test。
- 汇总包含 all/single/mixed、NDCG、mAP、首位全覆盖率等；配对查询 bootstrap 不包含模型选择偏差和训练 seed 方差。

## 4. 与现有代码和结果的隔离

新文件为：

```text
configs/ablation_ef.json
scripts/run_ablation_ef.py
wbm_retrieval/ablation_ef.py
wbm_retrieval/ablation_ef_training.py
tests/test_ablation_ef.py
doc/ABLATION_A_G.md
doc/EF_LOCAL_VALIDATION.md
```

原 `run_experiment.py`、`scripts/run_ablation.py`、A～D配置和训练模块保持不变。独立训练模块保留原协议，避免修改旧模块后使历史 A～D 的代码指纹失效。E/F 不依赖同级 `wbm-wdm-matching`；只需部署本工程及数据。

新默认目录为 `artifacts/ablation_ef_20261020`，与主实验、旧 A～D 分离。代码、输入文件、科学设置、缓存均记录指纹；变更后要求新 output root，不跨版本强行续跑。不要在运行中的新实验目录部署更新代码。输出目录有进程锁；同一 root 下依次训练，不同时启动多个写进程。

可选 `--reference-abcd-root` **只读**历史 A～D 结果，用于同 seed 的 valid 对比，不加载其权重训练、不改写历史文件。它核验数据/协议哈希、batch及训练设置、D定义、选模规则、实际valid池、seed与完成状态；不兼容时明确报错，不混合指标。历史 PyTorch/CUDA 环境仍应人工核对，跨环境比较不保证训练随机过程一致。

没有历史 D 目录时仍可独立跑 E/F；也可在新目录运行 D。若同时存在新 D 和历史 D，须去掉历史引用，避免歧义。

## 5. 远程运行

先将新代码同步到服务器，进入服务器 `match-test/wbm-region-retrieval` 并激活已有 CUDA 环境。默认数据路径仍是 `../data/wm38k/Wafer_Map_Datasets.npz`；若路径不同，在配置中指定适当的基础配置副本。保留 protocol 三份清单。新缓存含单通道和三通道 patch，建议为新目录预留至少10～15 GB磁盘空间。

### 检查依赖和展开配置

```bash
python -c "import torch, numpy, scipy, matplotlib; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python scripts/run_ablation_ef.py --config configs/ablation_ef.json --dry-run
```

缺依赖时可在独立虚拟环境安装 `requirements.txt`；远程 torch 应选择兼容当前驱动的 CUDA 版本，不为了本地测试覆盖服务器现有环境。`--dry-run` 不读数据、不写缓存、不训练，也不验证历史引用目录。

### 只训练 E/F，并引用已经完成的 A～D

下例假定历史目录确为 `artifacts/ablation_abcd_20261019`，且 `seed_2026/D` 已完成最终 valid 评价。若实际目录不同，请替换；若尚未取回历史完整目录，去掉 `--reference-abcd-root`。

```bash
mkdir -p artifacts/ablation_logs
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
nohup python -u scripts/run_ablation_ef.py \
  --config configs/ablation_ef.json \
  --stage all --variants E F \
  --device cuda --seed 2026 --num-workers 4 \
  --reference-abcd-root artifacts/ablation_abcd_20261019 \
  > artifacts/ablation_logs/ef_seed2026.log 2>&1 &
```

E、F顺序执行。日志会先显示连通域与patch/Zernike预处理，此时GPU利用率低是正常的。

```bash
tail -f artifacts/ablation_logs/ef_seed2026.log
```

退出 `tail -f` 不会停止后台训练。确认旧训练进程已结束后，从最后完成的 epoch 续跑：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
nohup python -u scripts/run_ablation_ef.py \
  --config configs/ablation_ef.json \
  --stage all --variants E F \
  --device cuda --seed 2026 --num-workers 4 --resume \
  --reference-abcd-root artifacts/ablation_abcd_20261019 \
  >> artifacts/ablation_logs/ef_seed2026.log 2>&1 &
```

恢复模型、SGD momentum、scheduler、memory、随机状态、最佳轮次；不承诺恢复到 epoch 中间的某个 batch。已完成组跳过优化，最终评价可以重新执行。

### 无历史 D，或者需要统一 batch 的对照

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
python -u scripts/run_ablation_ef.py \
  --stage all --variants D E F --device cuda --seed 2026 --num-workers 4
```

D/E/F默认都为batch256。如果显存不足，在首次运行时统一追加：

```text
--batch-size 128 --output-root artifacts/ablation_ef_bs128
```

之后所有阶段和续跑保留同样参数；若旧 D 是batch256，不能作为batch128实验的严格对照，历史引用检查会拒绝。此时在新目录重跑batch128的D。F增加的参数很少，但D/E/F的stride=1仍有较大中间特征显存开销；不要只单独改F的batch。

### 分阶段运行与补充历史对照

```bash
python scripts/run_ablation_ef.py --stage prepare
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_ablation_ef.py --stage train --variants E --device cuda
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_ablation_ef.py --stage evaluate --variants E --device cuda
python scripts/run_ablation_ef.py --stage summarize --reference-abcd-root artifacts/ablation_abcd_20261019
```

`train`会保存随机及逐检查点检索，但完整最终对比需要随后执行`evaluate`。历史引用可以仅在`summarize`时添加；如果引用中其他A/B/C结果存在，会一并展示。没有C结果时不伪造E−C，没有D结果且未传历史引用时不伪造E−D/F−D。

### 增加训练 seed

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
python -u scripts/run_ablation_ef.py \
  --stage all --variants D E F --device cuda --seed 2027 --num-workers 4
```

复用相同科学设置和valid池，训练结果分到`seed_2027`。只有历史目录也有同seed、兼容设置的D时才引用它。不要从多个seed中只挑最好的一个作为结论。

### 最终 test

只在 valid 选定方案和checkpoint规则后执行。例如选定E：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_ablation_ef.py \
  --stage test --variants E --device cuda --seed 2026 \
  --test-checkpoint best_retrieval
```

如果预先选择NCE规则，改为`--test-checkpoint best_nce`。test不接受历史valid引用；缺少显式`--variants`会拒绝执行。测试使用原冻结test query/candidate清单，不优化权重。

## 6. 输出与判读

```text
artifacts/ablation_ef_20261020/
├── suite_manifest.json
├── valid_pool.json
├── cache/
│   ├── {train,valid,test}_masks.npy
│   ├── {train,valid,test}_B.npy       # D/E原始输入，1×96×96
│   ├── {train,valid,test}_F.npy       # F原始输入，3×96×96
│   ├── {train,valid,test}_regions.npz
│   ├── {train,valid,test}_F_window_audit.csv
│   ├── F_window_audit.json
│   └── prepared.json
├── handcrafted/valid/
└── seed_2026/
    ├── E/                           # F同结构；D仅显式运行才生成
    │   ├── run_manifest.json
    │   ├── best_nce.pt / best_retrieval.pt / last.pt
    │   ├── training_summary.json / history.json
    │   └── valid/{random,epoch_001,…,best_nce,best_retrieval}/
    ├── comparison_valid.csv / comparison_valid.json
    ├── paired_valid.json
    └── comparison_valid_sources.json
```

先看`comparison_valid.csv`：各组训练后是否超过自身random；E与D、F与D及Zernike的差距；single/mixed分组；所选epoch。`training_summary.json`记录完成状态、参数变化、优化步数、参数量和本次训练调用的CUDA峰值已分配显存，`history.json`记录每轮损失、检索检查点和耗时。显存统计不是进程总占用；已完成组再次续跑时只反映该次加载调用，比较训练显存应保留首次训练完成时的报告。

`paired_valid.json`分别按best_retrieval和best_nce计算可用的E−D、F−D、E−C、E−F，以及E/F−Zernike。它不混用不同checkpoint规则。逐查询结果只反映当前固定池的不确定性，不能替代多seed与最终test。`comparison_valid_sources.json`记录引用指标和逐query文件的来源与哈希。

判断顺序：

1. E优于D，支持当前尺度＋平移增强在stride=1下有益；否则保留D，不强行叠加更多增强。
2. F优于D，支持上下文/固定尺度输入方案；结合窗口截断审计再解释，不直接归因到通道数。
3. 训练结果不超过自身random时，先检查训练与检索关系，不以模型绝对分数较高作为学习有效的证据。
4. 两组都保留NCE与检索最优模型，不能在看到test后选择更有利的规则。
5. G只在需要检验轻量化时实现；不能从D相对B的改善推断参数更少必然更好。

## 7. 本地可运行性验证

运行：

```bash
python -m unittest discover -s tests -p 'test_ablation_ef.py' -v
python -m unittest discover -s tests -v
```

新增测试覆盖F固定窗口/三通道/越界补零与截断统计、E与原C增强一致性、D/E与旧D初始化一致性、F共享层初始化、参数量及6×6特征图、缓存篡改拒绝、真实CPU参数更新、E/F中断续跑与连续训练逐元素一致、完成态续跑不改变checkpoint、固定池评价与原评价器一致、显式test不更新权重、两个spawn worker的CLI完整运行，以及历史D只读引用和配置不一致拒绝。

本次实际执行记录见同目录 `EF_LOCAL_VALIDATION.md`。本地小样本测试用于验证实现；完整100轮的CUDA显存、吞吐与检索性能需按上述远程命令验证。
