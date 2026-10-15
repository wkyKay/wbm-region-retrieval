# WBM 局部区域检索实验

这个工程实现的是在 Mixed-WM38 上使用一张 WBM 检索历史 WBM 的实验，检索时关注 query 中的缺陷区域能否在 candidate 中找到相似区域，因此 candidate 含有额外缺陷区域时不会因为这些未匹配区域而被直接扣分。工程包含主方法、手工局部描述子方法和整图监督方法这三类方法，其中整图监督方法分别使用 5%、10% 和 50% 的训练标签，所以最终对比表包含五行结果。代码独立实现了数据校验、区域划分、输入构建、ResNet-18、自监督训练、监督训练、匹配、排序和评价，运行时不需要导入或保留 `wbm-wdm-matching` 的代码，也不会下载 ImageNet 权重。

## 当前采用方案 A，以及为什么整图监督方法需要评分头

实验一所有方法统一使用 `R(Q,C)=|T_Q∩T_C|/|T_Q|` 作为相关性真值，其中 `T_Q` 和 `T_C` 是整张 wafer 的缺陷类型标签集合。对于 `Q={A,B,C}`，`C={A,B,C}` 和 `C={A,B,C,D}` 的相关性都是 1，`C={B,C,D}` 的相关性是 2/3，`C={A,D}` 的相关性是 1/3，这保留了 query 被覆盖而 candidate 可以包含额外模式的原始任务定义。

这个相关性是有方向的，例如 `{A,B,C}` 查询 `{A,B,C,D}` 的真值为 1，反向查询的真值为 3/4，因此整图监督方法使用一个共享 ResNet-18 encoder 和一个有方向的 pair scorer，不能只用单个 embedding 的 cosine 直接拟合该目标，因为 cosine 交换 query 和 candidate 后不会变化。这里采用的就是此前确认的方案 A，方法一、主方法和整图监督方法在同一组 query、同一组 candidate 和同一个 containment 评价脚本下比较，Jaccard 不参与训练目标或正式排序指标的计算。

这里的真值仍然是 wafer-level 标签包含关系，标签本身没有提供每个区域的像素标注，因此实验一能够验证按标签定义的局部模式检索能力，不能单独证明所有区域对应都正确，也不能直接证明已经完成工艺根因追溯。位置与 PCA 尺度的附加示例独立输出，实验一的两个局部方法固定使用 shape-only 打分。

## 文件夹和文件的作用

```text
wbm-region-retrieval/
├── run_experiment.py          # 数据准备、两类训练、三类方法检索与统一评价的唯一入口
├── configs/
│   └── experiment_a.json     # 数据路径、模型、训练、matching 和评价的完整参数
├── protocol/
│   ├── split.csv             # 从原工程逐字节复制的 70/10/20 冻结划分
│   ├── queries.csv           # 从原工程复制的 2,000 个固定 test query
│   ├── candidates.csv        # 每个 query 的 1,000 个固定 candidate
│   └── README.md             # 三份清单的来源、校验值和沿用范围
├── wbm_retrieval/
│   ├── methods/
│   │   ├── proposed.py       # 主方法的 token dataset、memory bank、NCE、自监督训练和 learned 检索
│   │   ├── handcrafted.py    # 方法一的手工描述子定义和局部检索流程，无需训练
│   │   └── supervised_global.py # 方法二的 pair dataset、监督训练、各标签预算和整图检索
│   ├── common.py             # 配置读取、文件哈希、随机种子和通用 CSV/JSON 操作
│   ├── data.py               # NPZ 和 manifest 校验、缓存准备、标签预算和有序 pair 采样
│   ├── regions.py            # 所有方法可复用的标准 8-CC、token patch、PCA 尺度和增强
│   ├── models.py             # 共享架构的 ResNet-18 encoder、projector 和 pair scorer
│   ├── training.py           # 共享 DataLoader、checkpoint、随机状态恢复和训练代码指纹
│   ├── matching.py           # 区域分数、贪心一对一匹配和 query 侧加权聚合
│   ├── retrieval.py          # 共享特征缓存、局部排序输出以及三个方法的检索分发
│   ├── evaluation.py         # containment 指标、query bootstrap 和方法对比表
│   └── geometry.py           # 固定形状表征的单变量位置与 PCA 尺度扰动示例
├── tests/
│   ├── test_core.py          # 数据边界、8-CC、匹配和评价公式的测试
│   ├── test_models.py        # 模型前向、梯度、memory bank 和无训练集成检查
│   └── test_methods.py       # 统一入口和按方法选择的调用测试，不执行训练
├── requirements.txt          # Python 运行依赖，不需要 torchvision
└── artifacts/                # 运行命令后生成的缓存、checkpoint、ranking 和实验结果
```

`configs` 保存每次实验使用的参数，`protocol` 固定可比较的数据身份，`wbm_retrieval` 保存独立算法实现，`tests` 用可控的小样本检验核心约束，`artifacts` 则保存一次具体实验生成的全部文件。不同 seed、proposal、输入方式或训练配置应使用不同的 `output_dir`，这样不会把旧 token、旧 embedding 和新 checkpoint 混在同一次实验中。

三个方法的具体实现现在分别放在 `methods/proposed.py`、`methods/handcrafted.py` 和 `methods/supervised_global.py`，查看主方法时可以在一个文件中顺着 token dataset、memory bank、NCE loss、`train()`、`load_encoder()` 和 `retrieve()` 阅读，查看方法一时只需关注手工描述子和 `retrieve()`，查看方法二时则顺着 pair dataset、`train()`、`rank_global()` 和 `retrieve()` 阅读。模型层、CC、matching、DataLoader、checkpoint 和 evaluator 继续在公共模块中复用，`run_experiment.py` 仍然是统一调用入口，原来的命令、stage 名称、方法名称和结果文件格式保持不变。

`run_experiment.py` 的 `train-local` 阶段调用 `methods.proposed.train()`，`train-global` 阶段调用 `methods.supervised_global.train()`，`retrieve` 阶段通过公共分发函数调用相应方法文件的 `retrieve()`，所以手工方法不会进入神经网络训练循环，单独指定某个方法检索时也不会加载其它方法的 checkpoint。共享的 `training.py` 只保存训练基础组件，`retrieval.py` 只保存公共推理与输出组件以及调用分发，三个方法的算法流程不再混放在这两个文件中。

## 数据读取和冻结测试协议

默认数据路径是工程同级 `data/wm38k/Wafer_Map_Datasets.npz`，相对路径始终相对于本工程根目录解析，因此在远程机器上只需要修改 JSON 中的 `data_file`，也可以保持 `wbm-region-retrieval` 与 `data` 的现有相对位置。NPZ 的 `arr_0` 是 `(N,H,W)` 的原始 wafer map，`arr_1` 是 `(N,K)` 的 multi-hot 标签，本地现有文件实际上是 38,015 张 52×52 图和 8 维标签，其中 37,015 张具有至少一个缺陷标签，原项目的冻结划分使用的就是这 37,015 张有标签样本，程序沿用这个范围而不会把无缺陷样本作为空标签 query 加进来。

原始图中只有像素值 2 被认为是 defect，像素值 1 和 2 构成计算晶圆中心与半径时的有效区域，这与参考工程的前处理口径一致；源文件中少量值为 3 的像素保留在原始缓存中，但不作为 defect 或有效区域参与上述计算，当前实现不会自行把它们改成缺陷。这个编码决定写在 `regions.py` 中，如果远程数据使用不同编码，应先统一数据口径再开始新实验。

三份 manifest 已经复制到本工程中，固定 train/valid/test 数量分别是 25,910、3,700 和 7,405，其中固定 test query 有 2,000 张，每张 query 检索 1,000 个 candidate，query 自身被排除。这里的 20% test 是测试图集合的范围，并不表示把全部 7,405 张图都作为 query，也不表示每张 query 检索全部 test 图。候选清单在原工程中按标签关系采样过，本实验沿用它以保证协议一致，结果代表这个固定候选池上的性能，不能直接当作全历史库检索性能。

程序检查 NPZ 标签是否与 split manifest 一致，检查 `sample_id`、`original_index` 和 `valid_index` 是否指向正确的原始样本，检查 split 是否重叠、query 是否属于 test、candidate 是否越界、是否包含自身、是否重复，以及每张 query 是否确实有 1,000 个候选。candidate 文件保留的 `jaccard` 和 `bucket` 列不会被模型或评价代码读取，正式相关性始终从原始标签重新计算 containment。缓存元数据保存输入文件 SHA-256、区域代码签名和配置签名，输入变化时会拒绝复用旧缓存。

## 主方法怎样训练和检索

首先在 defect mask 上执行标准 8 连通域分析，删除小于 `min_area` 的区域，再按照像素面积从大到小保留 `top_k` 个区域，同面积时按连通域扫描顺序决定先后。这里没有 ring-aware 规则、类别优先级、腐蚀拆分或学习分割器，所以 CC 粘连、细线破碎和 Top-K 漏掉小区域都仍然是这个 proposal 的限制，后端不会声称恢复了 CC 无法分开的真实语义区域。

每个保留区域只取自身的二值 mask，先裁剪到 bbox，再对较短边居中补齐成正方形，再在四周加固定像素边距，最后用最近邻插值缩放为统一画布，因此输入不会包含其它缺陷块或整张晶圆的绝对位置。默认输入是 1×96×96，原始面积、质心和 PCA 尺度只保存在 token 元数据中，patch 的 resize 不会改写这些元数据。

训练使用 WaPIRL-style memory-bank instance discrimination，把全部 train wafer 提取出的保留 token 作为训练样本，并在正式优化前用 encoder 与 projector 对它们做一次前向来初始化 memory bank。每次读取一个原始 token patch 和它的增强视图，把该 token 的 memory 表征作为正 anchor，把 train bank 中抽到的其它 token 作为负样本，分别计算原图和增强图的温度缩放 NCE loss 后等权求平均，再使用归一化的指数移动平均更新对应 memory。默认抽取 1,024 个互不重复的负样本，并排除当前 batch 中的 token；如果剩余数量不足就使用全部可用负样本，因此不会为了凑数量而重复抽样。

旋转增强保留 0°、90°、180° 和 270° 四种情况，同时使用轻度平移、像素 dropout 和稀疏噪声，增强只改变训练视图，不改变原始 proposal 的位置与尺度。训练视图的随机种子包含 epoch 和 token 索引，所以同一 token 在不同 epoch 会看到不同增强，在相同 seed、epoch 和索引下又可以复现。这里采用的是“旋转不改变当前检索语义”的任务假设，绝对朝向不会另设一个 orientation token。

encoder 使用 `[2,2,2,2]` 残差块的 ResNet-18 结构，考虑到 WBM 输入较小，把 stem 改成单通道 3×3、stride 1 卷积并去掉初始 max-pool，其余四个 stage 的通道为 64、128、256、512，最后通过全局平均池化和线性层输出 256 维 L2-normalized embedding。projector 是 `256→256→128` 的 MLP，只用于对比学习和 memory bank，正式检索使用 encoder 的 256 维输出，不使用 projector 输出。

checkpoint 按 valid token 的对比损失选择，valid token 使用固定增强和固定随机种子的 train-bank 负样本进行比较，训练优化与 memory 更新都不会写入 valid/test token。这个初版没有 pseudo-positive 第二阶段，同类但不同 wafer 的 token 仍可能被作为负样本，这是 instance discrimination 的已知限制，不能把本实现描述成已经解决了 false negatives。

## 方法一怎样与主方法对比

方法一与主方法共享同一份 CC 缓存、同一个区域集合、相同 matching 阈值、相同 query 权重和相同贪心一对一规则，唯一替换的部分是 shape representation。手工特征由 7 维 Hu 矩、12 维归一化质心径向质量直方图和 6 维无绝对尺度的统计量组成，统计量包括 bbox 填充率、PCA 特征值比、周长紧凑度、孔洞比例、归一化径向距离均值与标准差，各组分别归一化后拼接为 25 维向量，再做整体 L2 normalization 并用 cosine 打分。

这个方法无需训练，也不使用标签拟合描述子参数，所以全流程会直接生成它的 test ranking；它是用来检验 learned embedding 的贡献的透明手工局部基线，并不是 Kwon 2025 或某篇论文的逐项复现。两个局部方法虽然都使用相同数值阈值，但各自 cosine 的分布不一定一致，当前默认阈值为 0 只是可复现的初始设置，若以后在 valid 上调阈值，应在论文中明确调参协议并冻结后再运行 test。

## 方法二怎样使用相同 encoder 做整图监督训练

整图监督方法输入整张 WBM 的单通道 defect mask，将整图补成正方形并 resize 到 96×96，不裁剪单独缺陷区域，因此它可以看到绝对位置与整个缺陷布局。它的 encoder 架构和主方法完全相同，但训练权重独立，从相同随机初始化开始训练，不加载局部模型的 checkpoint，也不使用外部预训练数据。

对于有序 pair `(Q,C)`，同一个 encoder 分别产生 `zQ` 和 `zC`，评分头接收 `[zQ,zC,zQ*zC,zQ-zC]`，经过 `1024→256→128→1` 的 MLP 和 sigmoid 输出 `[0,1]` 内的 containment 预测值。输入顺序使 `S(Q,C)` 与 `S(C,Q)` 可以不同，代码中只有一个共享参数的 encoder，不是两套独立训练的 backbone，正式检索时会先缓存每张 test 图的 embedding，再分块运行 pair scorer。

5%、10% 和 50% 标签预算指的是 train split 中可以使用标签的 wafer 数量，而不是 pair 数量，也不是整个数据集的比例。程序按完整 multi-hot 组合分层抽样，在每个组合内使用相同种子的固定随机顺序，所以 5% 子集包含在 10% 子集中、10% 包含在 50% 子集中，每个组合至少保留一张，因此实际比例可能与名义比例略有差别，实际数量、实际比例和全部被选 wafer ID 都会保存。

训练 pair 的 query 和 candidate 必须都在对应预算的 train 子集中，程序按 full coverage、partial coverage 和 zero overlap 三个相关性档位尽量均衡采样，并排除同一张图与自身配对，遇到某个 query 缺少某档候选时在其余可用档位采样。默认每个 epoch 生成 50,000 个有序 pair，使用 `MSE(predicted_containment, true_containment)` 直接监督，也可以把 `global.loss` 改成 `smooth_l1`。三个标签预算的 epoch 数与每 epoch pair 数相同，因此比较的是标签量变化，而不是让大预算自动获得更多优化步数。

三个预算共同使用 valid split 中的 10,000 个固定有序 pair 选择 checkpoint，valid pair 只用于评估而不反向传播，这意味着名义标签预算只限制 train 标签，额外的 valid 标签使用量会在 `budget.json` 中如实记录。test 标签不会进入 encoder、pair scorer、训练 pair 采样或 checkpoint 选择，只有统一 evaluator 在生成 ranking 后读取 test 标签计算指标。方法二是有额外标签监督的整图参考，并不是数学上保证高于主方法的严格上界，也不保证监督训练一定优于无标签局部方法。

## 区域打分和贪心一对一匹配怎样计算

实验一使用 `shape(q,c)=max(0,cosine(zq,zc))`，丢弃低于 `shape_threshold` 的区域对，再按区域对分数从高到低扫描，每个 query token 和每个 candidate token 最多使用一次，分数并列时依次按 query token 索引与 candidate token 索引决定先后，低于 `score_threshold` 或得分为 0 的区域对不会被保留。这里采用的是按高置信区域对优先的贪心规则，并不求全局最高总分的 Hungarian assignment。

最终分数是 `S(Q,C)=sum_(i,j in M) w_i*s(q_i,c_j)/sum_(i in Q) w_i`，默认 `w_i=sqrt(area(q_i))`，也可以在配置中选择 `uniform` 或 `area`。分母始终包含全部 query token，所以 query 中未被匹配的区域贡献为 0，而 candidate 中未被匹配的额外区域没有直接扣分项；如果 query 或 candidate 没有任何保留区域，分数定义为 0 并保留这张图参与排序，程序不会通过丢弃困难 query 提高指标。

这里使用的 directional coverage 指 query→candidate 的聚合方向，同时保留一对一约束，不是每个 query token 独立取 `max_j` 的多对一覆盖。标签相关性按类型数量计数，而局部分数默认按区域面积平方根加权，因此两者不是逐点相等的标定关系，主方法是否能得到好的标签相关性排序需要由实验检验，不能仅凭公式保证。

## 位置和 PCA 尺度怎样保留给补充实验

质心先减去有效晶圆像素的中心，再除以该晶圆的最大有效半径，所以不同图尺寸的坐标处在同一相对尺度；位置相似度使用 `exp(-||posQ-posC||²/sigma_pos²)`。尺度由区域像素坐标的总体协方差计算，在晶圆半径单位下取 `l=sqrt(lambda1+lambda2)`，其中 `lambda1` 和 `lambda2` 是 PCA 特征值，再计算 `scale_affinity=exp(-abs(log(lQ/lC))/sigma_scale)`，极小尺度使用 epsilon 防止除零。

这个字段叫 PCA-scale 而不是面积，因为协方差的空间跨度不能完整表示缺陷像素数量或密度；细长 line 的第二特征值可能接近 0，所以没有使用两个特征值的乘积作为尺度。像素面积只用于 CC 过滤、Top-K 和 query 聚合权重，不再与 PCA 尺度共同组成一个重复加权的面积相似度。

补充评分公式是 `s=shape*[1-lambda_pos*(1-position_affinity)-lambda_scale*(1-scale_affinity)]`，并要求两个 lambda 非负且和不大于 0.25，形状低于门槛时仍然得 0。这个限制控制几何项最多扣除多少分，但并不保证所有形状分数的排序永远不被改变，是否需要严格形状门槛仍然由 `shape_threshold` 决定。实验一配置会拒绝非零 geometry lambda，补充示例通过独立的 `geometry` 配置运行。

`--stage geometry` 使用训练后模型的 test token embedding 固定形状表征，只改变归一化质心距离或 PCA-scale，并输出原 token 图片、分数曲线和 CSV，以检查评分函数是否按预期随几何差异变化。这个阶段是特征级单变量控制实验，并没有重新移动或缩放像素后重新分割图像，所以它不能替代图像级扰动鲁棒性实验，也不作为实验一的排名结果。

## 远程 GPU 环境和一条命令执行全流程

建议在 Linux GPU 机器使用 Python 3.10 或以上版本，并先安装与远程 CUDA 驱动匹配的 PyTorch，然后安装本工程其余依赖；本工程没有 torchvision 依赖，`requirements.txt` 中的 PyTorch 版本范围允许保留已经安装且满足要求的版本。下面命令都在本工程根目录运行，修改 `configs/experiment_a.json` 中的数据绝对路径后即可执行，`--dry-run` 和 `--stage audit` 都不会训练模型。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_experiment.py --config configs/experiment_a.json --dry-run
python run_experiment.py --config configs/experiment_a.json --stage audit
python run_experiment.py --config configs/experiment_a.json
```

最后一条命令默认执行 `prepare → train-local → train-global → retrieve → evaluate`，其中会训练一次局部自监督模型和三次整图监督模型，再生成手工方法、主方法和三个标签预算的监督方法的完整 test ranking、每方法指标和总对比表。手工方法无需训练，所以没有为它增加一个无意义的训练步骤，位置与尺度示例也不会自动混入这个全流程。

默认 batch size 是 128，ResNet-18 使用较大的中间特征图，具体显存消耗需要根据远程 GPU 调整，如果出现显存不足可以先把 `batch_size` 和 `inference_batch_size` 改成 32 或 64 并使用新的 `output_dir`，代码当前不使用 AMP 或多 GPU。预处理将 token patch 与整图输入缓存为 uint8 的 `.npy` 文件并通过 memory map 读取，因此磁盘需要保留数 GB 的缓存和 ranking 空间，`num_workers=4` 适合远程 Linux，在本地排查时可改成 0。

如果训练进程被中断，可使用下方命令从最近完成的 epoch 继续，checkpoint 同时保存模型、optimizer、scheduler、memory bank 和随机状态，恢复粒度是 epoch 而不是 batch。配置不兼容时会拒绝续训，已完成的兼容训练会被跳过；如果目的是开新实验，请改 `output_dir`，不要把 `--resume` 当作允许更换方法定义的开关。

```bash
python run_experiment.py --config configs/experiment_a.json --resume
```

## 分阶段执行和复用训练结果

可以用同一个入口逐阶段执行，以便先检查缓存或单独重跑检索和评价。单独训练时仍然要求先完成 `prepare`，单独检索时需要已经存在相应的 `best.pt`，如果只想跑手工方法可在检索与评价时用 `--methods handcrafted`，如果只想跑主方法则使用 `--methods proposed`。

```bash
python run_experiment.py --stage prepare
python run_experiment.py --stage train-local
python run_experiment.py --stage train-global
python run_experiment.py --stage retrieve
python run_experiment.py --stage evaluate
python run_experiment.py --stage geometry
```

```bash
python run_experiment.py --stage retrieve --methods handcrafted
python run_experiment.py --stage evaluate --methods handcrafted
python run_experiment.py --stage retrieve --methods proposed global_5pct
python run_experiment.py --stage evaluate --methods proposed global_5pct
```

如果需要改 matching 阈值或权重，可以在保持输入和训练参数不变的前提下重新运行 retrieve 和 evaluate，程序会复用同一 checkpoint 和 embedding；如果修改 CC、padding 或 input size，必须重新 prepare 并使用新的输出目录，因为区域定义或模型输入已经改变。正式参数只能在 valid 数据上决定，当前工程提供的是固定配置的完整实验流程，没有在 test 上自动寻找最优超参数。

## 默认超参数和它们控制的行为

| 参数 | 默认值 | 具体作用 |
| --- | --- | --- |
| `seed` | 2026 | 控制初始化、增强、标签子集、pair 和 bootstrap，改变它时需要使用新的输出目录 |
| `proposal.min_area` | 5 | 删除像素数少于 5 的 8 连通区域 |
| `proposal.top_k` | 6 | 每张图最多保留 6 个区域，按原始像素面积选择 |
| `proposal.padding` | 4 | bbox 补成正方形后在原始像素坐标中增加的四周边距 |
| `input_size` | 96 | 两种 learned 方法共同使用的输入画布边长 |
| `embedding_dim` | 256 | ResNet-18 输出的归一化向量维数 |
| `batch_size` / `num_workers` | 128 / 4 | 训练读取与并行预取设置，batch 中局部方法同时前向原图和增强图 |
| `local.epochs` / `global.epochs` | 100 / 50 | 局部模型和每个监督预算模型各自的训练轮数 |
| `local.learning_rate` / `global.learning_rate` | 0.001 / 0.001 | AdamW 的初始学习率，均使用 cosine learning-rate schedule |
| `local.weight_decay` / `global.weight_decay` | 0.0001 / 0.0001 | 两类训练的权重衰减系数 |
| `local.projector_dim` | 128 | 对比学习输出与 memory bank 的维数，正式检索不用这一层 |
| `local.temperature` | 0.07 | NCE logits 的温度参数 |
| `local.num_negatives` | 1024 | memory bank 中每步最多抽取的负样本数 |
| `local.memory_momentum` | 0.5 | memory 更新中旧表征所占的权重，更新后重新归一化 |
| `augmentation.rotation` | true | 从四个直角旋转中均匀抽样，保留已确认的旋转增强设计 |
| `augmentation.max_shift` | 3 | 在输入画布上最多平移 3 个像素并用零填充 |
| `augmentation.dropout` / `noise` | 0.02 / 0.005 | 像素删除概率与额外 defect 噪声概率 |
| `global.budgets` | 0.05、0.1、0.5 | 监督训练允许看到标签的 train wafer 比例 |
| `global.pairs_per_epoch` | 50000 | 每个监督预算每个 epoch 的有序训练 pair 数量 |
| `global.validation_pairs` | 10000 | 所有预算共同使用的固定 valid pair 数量 |
| `global.loss` | mse | 对 containment 预测值做直接回归，也支持 smooth_l1 |
| `matching.shape_threshold` / `score_threshold` | 0 / 0 | 两种局部方法共享的形状门槛和最终匹配门槛，零分区域对始终被排除 |
| `matching.weight` | sqrt_area | query token 的聚合权重，像素面积用于重要性而不是尺度相似度 |
| `matching.lambda_pos` / `lambda_scale` | 0 / 0 | 实验一强制关闭位置与尺度修正 |
| `matching.sigma_pos` / `sigma_scale` | 0.35 / 0.5 | 补充评分使用的位置距离与对数尺度差异带宽 |
| `geometry.lambda_pos` / `lambda_scale` | 0.125 / 0.125 | 独立补充示例的两个修正权重，总和限制为 0.25 |
| `evaluation.ks` | 1、5、10、20 | 所有方法共同报告的 Top-K 位置 |
| `evaluation.bootstrap_samples` | 1000 | 对固定 query 集合重采样得到均值的 95% 区间 |

以上数值是可运行的初始设置，不是已经通过训练证实最优的参数，也没有在本地启动正式训练来产生性能结论。如果论文要报告多 seed 结果，保留相同的 protocol 清单，仅改变模型随机种子与输出目录，并把每次的 `comparison.csv` 汇总后再报告均值和标准差。

## 实验结果和指标怎么读取

每个方法都会输出完整的 `query_id,rank,candidate_id,similarity_score`，排序按分数降序、candidate ID 升序处理并列，所以相同输入可以得到确定结果。evaluator 会检查每张 query 的返回候选集合与冻结清单完全相同，再对这 1,000 个候选计算理想排序和正样本总数，IDCG、Recall 与 AP 不会只根据已返回的 Top-K 临时计算。

主要排序指标是 `ContainmentNDCG@K`，使用 `[0,1]` 的连续 containment 作为 identity gain；`MeanContainment@K` 表示靠前候选平均覆盖了多少 query 标签；`FullCoveragePrecision@K`、`FullCoverageRecall@K` 和 `FullCoverageHit@K` 只把 containment 为 1 的 candidate 当作完整覆盖正样本；`mAP_full` 使用全部 1,000 个候选的完整排名计算 AP；`MRR_full` 使用首个完整覆盖候选的位置。没有任何完整覆盖候选的 query，其 full-coverage Recall、Hit、AP 和 MRR 标记为空并从这些均值中排除，同时保存这类 query 的数量，避免用未定义的分母生成误导分数。

`metrics.json` 保存各指标和按 query bootstrap 的 95% 区间，`per_query_metrics.csv` 保存每张 query 的结果，`comparison.csv` 和 `comparison.json` 汇总五行主结果；主方法和手工方法都存在时还会生成 `paired_proposed_vs_handcrafted.json`，在同一 query 上计算 NDCG 差值的配对 bootstrap 区间。这个区间反映固定 query 集合的抽样不确定性，并不代替多次独立训练的 seed 方差。

```text
artifacts/experiment_a_seed2026/
├── resolved_config.json
├── environment.json                       # 准备缓存时的 Python、系统与依赖版本
├── cache/
│   ├── prepared.json                     # 数据指纹、协议数量和空区域统计
│   ├── maps.npy / labels.npy             # 原始数据的轻量缓存
│   ├── train_regions.npz                 # map IDs、token offsets、area、position、PCA-scale 和手工特征
│   ├── train_patches.npy                 # 当前 split 的单通道 token patch
│   ├── train_whole.npy                   # 当前 split 的整图输入，valid/test 有对应文件
│   └── test_*_embeddings.npy             # 与 checkpoint SHA-256 绑定的推理特征缓存
├── models/
│   ├── local/{best.pt,last.pt,history.json}
│   └── global_5pct/                      # global_10pct 和 global_50pct 结构相同
│       ├── best.pt / last.pt / history.json
│       ├── labeled_train_ids.csv
│       ├── budget.json
│       └── validation_pairs.npz
├── retrieval/
│   ├── handcrafted/{rankings.csv,matches.csv,metrics.json,per_query_metrics.csv,retrieval_info.json}
│   ├── proposed/                         # 与 handcrafted 相同，matches 只保存每 query 的前 10 个候选
│   └── global_5pct/                      # 含 ranking、指标和耗时，不伪造区域对应，另有两个标签预算
├── geometry/                             # 仅在手动执行 geometry 阶段后生成图片与扰动 CSV
├── comparison.csv / comparison.json
└── paired_proposed_vs_handcrafted.json
```

`retrieval_info.json` 的耗时是 ranking 与结果写入的墙钟时间，不包含此前的 encoder 特征提取，因此比较运行效率时应分清预计算成本和检索成本，不能把它直接当作端到端在线延迟。

## 本地验证方式和当前范围

下面的测试使用临时小数据检查 8 连通、Top-K、平移后 patch 一致性、手工旋转不变特征、PCA 尺度、一对一匹配、额外候选区域、标签预算嵌套、pair 集合边界、理想排序和空正样本处理。模型测试检查 ResNet 前向与梯度、评分头方向性、memory bank 排除规则和 epoch 增强变化，集成测试会生成随机权重的临时 checkpoint 来验证预处理、五行 ranking、评价和几何示例之间的接口，过程中不调用 optimizer step、不进行模型训练，临时 ranking 也不会当作正式实验结果。

```bash
python -m unittest discover -s tests -v
```

当前完成范围是固定 Mixed-WM38 协议下的实验一和补充几何评分接口，没有加入新的合成区域真值 benchmark、oracle proposal、腐蚀拆分、多阶段伪正例学习或 Kwon 对比方法，这与当前确认的工程范围一致。训练 checkpoint 使用 PyTorch 序列化并包含随机状态，只应加载本工程在可信机器上生成的文件。

## 技术参考

局部自监督训练参考 Kahng 与 Kim 的 *Self-Supervised Representation Learning for Wafer Bin Map Defect Pattern Classification* 及其 [WaPIRL 官方实现](https://github.com/hgkahng/WaPIRL)，本工程把训练单位改为 CC token，并独立实现了 ResNet、NCE、memory bank 和运行流程，因此称作 WaPIRL-style adaptation，不声称完整复现其原始分类实验。整图监督评分头是为了当前有方向的 containment 任务设置的可复现对照，不将它归属于 WaPIRL 或 Kwon 原文的方法。
