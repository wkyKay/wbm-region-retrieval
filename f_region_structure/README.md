# F 区域结构实验

独立实现“区域 J 参与配对”和可学习区域联合表示。旧代码、模型、固定协议及历史结果只读，新产物进入 `artificial/f_region_structure_<实际启动时间>/`。本次研究方向在观察历史 test 后提出，属于事后扩展；所有新方案仅用 valid 选择，test 只评价一个冻结胜者，不自动更换项目主方法。

**当前状态：正式流程已于2026-11-20 12:03:31（Asia/Shanghai）启动**，运行目录 `artificial/f_region_structure_20261120T120331+0800/`，启动代码 `fcda93f`。本地和远程各69项新实验测试通过，源模型/train缓存/valid特征/协议检查通过；已完成全部train区域结构提取，正在提取冻结CNN双视图并训练小融合层。尚无新valid/test成绩；完成状态以运行目录校验通过的 `completed.json` 为准。

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
