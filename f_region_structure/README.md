# F 区域结构实验

独立实现“区域 J 参与配对”和可学习区域联合表示。旧代码、模型、固定协议及历史结果只读，新产物进入 `artificial/f_region_structure_<实际启动时间>/`。本次研究方向在观察历史 test 后提出，属于事后扩展；所有新方案仅用 valid 选择，test 只评价一个冻结胜者，不自动更换项目主方法。

**当前状态：25项正式valid与唯一冻结胜者的test均已完成、核验并登记。** 启动器于2026-11-20 12:03:31（Asia/Shanghai）启动；根目录manifest记录的实际开始时间为 `2026-11-20T12:03:48+08:00`，目录为 `artificial/f_region_structure_20261120T120331+0800/`，启动代码 `fcda93f`。选择冻结于 `2026-11-20T12:35:27+08:00`。本地和远程各69项新实验测试通过，本地另有30项旧几何与登记回归通过。两种融合层训练及25项valid核验完成；独立valid审计覆盖12,500条逐query记录。test全部7,199个query、51,818,402个候选配对及3,599,500行Top-500通过核验，保存分数误差为0，指标最大误差为1.8874e-15。根完成标记绑定350个文件；54,572个既有产物通过保护检查。新增26项登记，原322项结果保持深度一致，总计348项。

公共数据、评价和结果管理约定见 [项目说明](../doc/项目说明.md) 与 [结果登记说明](../results_catalog/README.md)。

## 固定设计

保留 cc8、min_area=5、Top-4、原 h512、冻结 best_nce/seed2026、原贪心一对一匹配及查询 sqrt(area) 聚合权重。所有区域对均先计算新相似性，再进入同一匹配器；没有整图 G、角向 A 或整图 J 分数后融合。

完整原图的缺陷点只做一次张量投票，沿用原 J 的 sigma_d=9、去噪前整图通道最大值归一和晶圆支撑中心/半径。按原始区域 mask 池化 rho×stick×ball 的 8×4×4 软直方图，不从裁剪图测几何。

- `retained`：沿用原 J 的0.32硬阈值，只汇总保留点。
- `all_points`：不做硬去噪，汇总区域所有原始缺陷点；投票与归一规则相同。
- 空的结构描述表示缺失；结构作用退回剩余 CNN/面积项，不将双空作为高置信匹配。
- 128维独立分布对照使用 rho边缘×完整stick/ball边缘，维数与联合分布相同。
- 区域覆盖率、空结构比例、去噪保留点比例保存在各split缓存审计。保留比例是质量代理，当前不另行学习置信度，也不加入剩余区域或改变Top-4。

令 C=clip(cos(hq,hc),0,1)，R=exp(-D_radial)，B=0.85+0.15exp(-D_area/1.5)，H为区域分布的普通Hellinger距离，K=点积(sqrt(Pq),sqrt(Pc))。以下25项清单在运行前固定，均参与唯一valid选择：

| 方案 | 计算方式 | 数量 |
| --- | --- | ---: |
| 原F | C×R×B；逐位复用旧函数 | 1 |
| 加区域J | C×R×B×exp(-λH)；两种点集，λ=0.5/1/2 | 6 |
| 区域J替换径向 | C×B×exp(-λH)；两种点集，λ=0.5/1/2 | 6 |
| 区域向量拼接 | clip((1-w)cos(hq,hc)+wK,0,1)×B；两种点集，w=0.25/0.5/0.75 | 6 |
| 拼接删除面积 | 上一公式删除B，仅all_points，三个w | 3 |
| 独立分布对照 | 替换径向，all_points，λ=1 | 1 |
| MLP联合表示 | CNN、sqrt(J)、log面积、缺失标记输入小MLP，128维单位向量余弦 | 1 |
| 门控联合表示 | 结构调制视觉特征后投影为128维单位向量，取余弦 | 1 |

拼接的点积等价于分支核加权，它是无训练机制对照，不声称学习了交互。两种MLP才进行新的融合训练。区域J包含8个径向软节点，原F为16-bin CDF距离；替换同时改变表示和度量。所有方案仅是有限候选集合，不代表连续参数空间最优。

## 融合层训练与选优

原CNN始终冻结。仅对冻结train 25,193张图的区域，用原三通道固定窗口及90°旋转窗口编码两种h512视图；结构和面积从完整原始区域计算，保持对应的旋转不变量。无标签参与训练，同一区域跨视图为正对，同晶圆其他区域排除负对，其他晶圆的批内区域作负对。

两种融合层固定20epoch、batch512、seed2026、AdamW lr=0.001、对比温度0.1，结构/面积重建辅助权重分别0.1/0.05；其他优化细节见 `learning.DEFAULTS`。仅train拟合面积标准化，固定保存最终轮，不用valid/test选epoch。其参数属于本实验预先设定的实现选择，不是原F或WBBS论文参数。

valid仍为500个固定query，每query全部3598个候选；test为7199×7198。exact与coverage共享Top-500排名，分母与coverage IDCG使用完整候选库。保存K=1/5/10/20/50/100/500、总体、宏平均、单/混合、标签数量和逐类报告。

唯一选优指标为valid总体exact P@10，比较全部query的Top-10整数命中总数。并列依次选原F、无需新融合训练、较少输入分支、较小结构强度/权重、字典序；规则由 `selection.RULE` 固定。原F也参与，避免新方法不增益时仍被强制选中。只有25项全部完成核验后才写冻结 `selection.json`，随后test仅接受该胜者。既有整图J/B1成绩仅作外部参考，不参与本次区域方案的选优。

## 本轮valid选择与观察

唯一胜者为 `replace_radial_all_points_l0p5`：用区域全部原始缺陷点构造J，区域对评分为 `C×B×exp(-0.5H)`，由J取代原径向项，保留面积项与原贪心匹配。500个query的Top-10共命中2,494次，P@10为49.88%；原F为1,986次、39.72%，相差10.16个百分点。以下为各方案族的valid最高P@10，用于说明本轮选择，不能当作独立test泛化成绩。

| 方案族 | 本轮族内最高配置 | valid P@10（%） |
| --- | --- | ---: |
| 原F | 冻结原评分 | 39.72 |
| 区域J替换径向 | all_points，λ=0.5，保留面积 | **49.88** |
| 独立分布对照 | all_points，λ=1，保留面积 | 48.60 |
| 加区域J | all_points，λ=1 | 48.34 |
| 区域向量拼接 | all_points，w=0.25，保留面积 | 47.90 |
| 拼接删除面积 | all_points，w=0.5 | 45.34 |
| MLP联合表示 | 固定20epoch最终轮 | 35.60 |
| 门控联合表示 | 固定20epoch最终轮 | 34.18 |

九组其余参数相同的配对中，all_points均优于retained。三个相同w的拼接对照中，删除面积项分别下降2.78、2.46、2.46个百分点（w依次为0.25、0.5、0.75），支持本轮保留面积信息。同λ=1时，联合分布为48.80%，独立分布为48.60%，仅相差0.20个百分点；不能把λ=0.5胜者与λ=1独立分布之间的1.28个百分点归因于联合关系。两种可学习融合在当前训练配方下均低于原F，这不说明其他训练目标或神经融合设计均无效。

valid完整图库的Top-4累计保留原始缺陷像素66.2307%；硬去噪使151/14,240个区域结构描述为空，约1.06%。按类别逐图保留率再取平均，单Scratch为26.42%，Random为89.45%；当前审计提示优先关注区域筛选造成的Scratch像素丢失，不能沿用“Random最容易丢失”的猜测。该类别均值与前述全图库累计像素比例采用不同聚合方式。

原F valid基线Top-500共250,000行，排名文件SHA-256与历史文件一致，确认逐字节复现。完整25项成绩、冻结选择及缓存审计分别保存在本轮目录的 `comparison_valid.csv`、`selection.json`、`cache/valid/audit.json`；全候选评分与Top-500排序、双指标核验见各 `valid/<方案>/verification.json`。这些valid差值是本次有限搜索的观察；仅其冻结胜者进入本轮test，未据此更改项目主方法。

## 本轮test结果与解释

仅评价valid冻结的 `replace_radial_all_points_l0p5`，其余24项没有新增test成绩。下表均为相同去重协议、7,199个query及每query全部7,198个候选的总体指标，单位为%；F、WBBS和整图J引用既有结果。本轮test及其全候选排序与指标核验已完成；不依据它们重新选方案或调整参数。

| 方法 | exact P@10 | exact mAP@100 | Coverage-nDCG@10 |
| --- | ---: | ---: | ---: |
| 区域J替换径向（本轮valid胜者） | 56.4453 | 11.2966 | 87.2084 |
| 原F（cc8＋径向＋面积） | 46.1370 | 7.4567 | 82.4208 |
| WBBS（项目复现） | 58.1289 | 13.2996 | 87.8320 |
| 仅整图J（不含F、G、A） | 63.2546 | 15.0794 | 89.8759 |

本轮方案相对原F的P@10点估计提高10.3084个百分点，三个总体指标均提高；1—4标签组的P@10也均高于原F。它仍低于WBBS和仅整图J的三个总体指标，因此目前支持“区域结构增强了原F”，尚不支持“区域化优于整图J”或“已超过WBBS”。按37种完整标签组合分层、10,000次配对bootstrap（seed=2026），新方案相对F的P@10差值95%区间为[9.9042, 10.6945]个百分点；相对WBBS为−1.6836 [−2.2517, −1.1127]，相对整图J为−6.8093 [−7.3246, −6.2925]。区间条件于固定模型与图库，不包含训练种子变化，未作多重比较校正。逐query配对、三项指标分组和可复现脚本保存在 `artificial/f_region_structure_analysis_20261120T130922+0800/`；CSV指标使用0—1比例，明确标注的`*_pp`使用百分点。

两种MLP的35.60%和34.18%是valid P@10，均低于原F；它们未进入本轮test。结论限于冻结CNN、两种旋转视图、固定20epoch及当前损失配方。区域实例对比目标与类别检索可能不完全一致，其他晶圆中语义相似区域可能被当成负样本，有限增强也可能使模型过度依赖不变几何；这些是待检验假设，不能由当前成绩认定原因，也不能推断所有神经融合都无效。

区域覆盖率仍是方法的限制。前述66.2307%的累计像素保留率、Scratch 26.42%与Random 89.45%的类别逐图均值均来自valid缓存，不能当成test审计结果。`min_area=5`与Top-4筛选共同丢弃部分缺陷点；全图投票保留其间接影响，但不能完整恢复它们在区域池化中的分布。后续若继续研究，宜先检查Scratch碎片与遗漏区域，再单独比较残余区域描述或区域合并；是否提高检索仍需新实验验证。可学习路线则先诊断表征、负样本及损失目标，不直接以增加网络深度作为结论。当前保留全部区域点和面积信息是valid机制对照支持的选择。

本轮原始test报告位于运行目录的 `test/replace_radial_all_points_l0p5/metrics.json`、`coverage_metrics.json` 与 `rankings.csv`；独立核验见同目录 `verification.json`。来源身份见 `results_catalog/sources.json` 和分析目录的 `analysis_manifest.json`；新test登记key为 `f_region_structure_f_region_structure_20261120T120331_0800_test_replace_radial_all_points_l0p5`。统计分析最初读取报告时主流程尚未封存；最终封存后、登记前，128份分析输入哈希再次核验一致，补充凭据见分析目录 `formal_source_check.json`。根完成标记SHA-256为 `bafc12f8c4535927802f32f5a2172baee5a9a4eb55ca639312c50979597979f8`；`source_protection_after.json`确认历史产物未变。正式登记器与官方collect/check均通过。

## 执行与恢复

在服务器仓库根目录、已激活原PyTorch CUDA环境后：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s f_region_structure/tests -v
python -m f_region_structure.run --dry-run
python -m f_region_structure.run --check-inputs --device cuda
bash f_region_structure/launch_tmux.sh
```

启动器创建独立tmux会话，日志在本目录被Git忽略的 `launch_logs/`，运行产物在新的 `artificial/` 子目录。原F的train缓存、valid/test冻结特征必须存在且签名通过，缺失不会静默重建历史缓存。

前台执行全部流程，或先只运行valid：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u -m f_region_structure.run --stage all --device cuda --prepare-workers 4
# 仅valid把 --stage all 改为 --stage valid。
```

恢复同一代码、来源与环境的运行：

```bash
python -u -m f_region_structure.run --stage all --device cuda --prepare-workers 4 \
  --resume-root artificial/f_region_structure_原启动时间
```

单独启动冻结test，使用 `--stage test --resume-root <已完成valid目录>`。完成的缓存、模型和评价按哈希复用；未完成的训练保留旧目录后重新训练该融合层，未封存的排名重新生成。科学代码、来源、数值环境或配置变化须用新输出；不存在修改选择记录后继续test的入口。

## 产物与核验

根目录保存科学签名、来源、训练缓存/模型绑定、全部valid结果、冻结选择、单一test结果和旧产物保护记录。各评价有真实Asia/Shanghai时间身份、双指标、逐query/逐类报告、Top-500、核验报告和完成标记。训练身份独立于评价身份。

核验包括原F valid排序重放；每类代表query的20个候选独立区域公式与标量贪心聚合；全部query的全候选生产评分重放与独立SHA排序；所有exact/coverage指标与各类汇总独立重算。前者的独立公式抽查不冒称全候选独立公式验证。来源绑定完整文件哈希；旧artifacts/artificial/protocol目录全文件大小与mtime受保护，小于1MiB的文件另做哈希保护。

完成后使用新目录的登记器生成提案并合并到唯一结果目录，旧登记保留：

```bash
python -m f_region_structure.register --run-root artificial/f_region_structure_实际启动时间 --apply
python results_catalog/build.py check
```

尚未完成的运行不登记正式成绩。完整缓存、模型和排名留在服务器，Git仅管理实现、说明与轻量结果汇总。
