# 区域J、CNN与面积融合

本轮固定已有区域表示，比较29组预设评分。先完成全部valid，以整数exact P@10选出10组并冻结，再对这10组做完整test；不根据test结果继续调参或重选主方法。**当前已实现独立入口和登记核验，正式完成状态以新运行根的完成凭据为准；本页尚未填入本轮实验成绩。** 本轮属于观察历史结果后的事后扩展，不自动替换项目主方法。

公共协议、评价与结果登记见[项目说明](../doc/项目说明.md)和[登记说明](../results_catalog/README.md)。旧[区域结构实验](../f_region_structure/README.md)、[15项评分搜索](../f_region_score_search/README.md)和[等权加法test](../f_region_score_test/README.md)的源码、模型及结果保持只读。

## 固定表示与29项设计

沿用cc8、最小区域面积5、Top-4、冻结h512 `best_nce`（seed2026）、原始区域全部缺陷点的128维J联合分布、单向贪心一对一匹配及查询区域sqrt(area)聚合权重。所有几何量来自完整原始mask，无新训练。记：

- C＝clip(cos(hq,hc),0,1)。
- H为区域J的普通Hellinger距离；T＝exp(−0.5H)。J已包含晶圆径向与stick/ball联合分布，不另加位置项。
- r为区域缺陷像素数／有效晶圆像素数；D_area＝0.5|log(rq)−log(rc)|，A＝exp(−D_area/1.5)。
- Mα＝αC＋(1−α)T。α是CNN在C/T加法中的权重；β是本轮面积融合权重，与J的固定指数系数0.5不同。

分数保持原尺度，不做min-max、分位数、温度搜索或训练集标定。四个固定β为0.05、0.1、0.15、0.25；β＝0只保留一个无面积评分项端点，避免加乘产生重复配置。无面积评分项仍保留原匹配的查询sqrt(area)聚合，不能称为完全移除了面积信息。

| 序号 | 名称 | 区域对评分 |
| ---: | --- | --- |
| 1 | `no_area_a0p4` | M₀.₄ |
| 2 | `no_area_a0p5` | M₀.₅ |
| 3 | `no_area_a0p6` | M₀.₆ |
| 4 | `add_a0p4_b0p05` | 0.95M₀.₄＋0.05A |
| 5 | `add_a0p4_b0p1` | 0.9M₀.₄＋0.1A |
| 6 | `add_a0p4_b0p15` | 0.85M₀.₄＋0.15A |
| 7 | `add_a0p4_b0p25` | 0.75M₀.₄＋0.25A |
| 8 | `add_a0p5_b0p05` | 0.95M₀.₅＋0.05A |
| 9 | `add_a0p5_b0p1` | 0.9M₀.₅＋0.1A |
| 10 | `add_a0p5_b0p15` | 0.85M₀.₅＋0.15A |
| 11 | `add_a0p5_b0p25` | 0.75M₀.₅＋0.25A |
| 12 | `add_a0p6_b0p05` | 0.95M₀.₆＋0.05A |
| 13 | `add_a0p6_b0p1` | 0.9M₀.₆＋0.1A |
| 14 | `add_a0p6_b0p15` | 0.85M₀.₆＋0.15A |
| 15 | `add_a0p6_b0p25` | 0.75M₀.₆＋0.25A |
| 16 | `mul_a0p4_b0p05` | (0.95＋0.05A)M₀.₄ |
| 17 | `mul_a0p4_b0p1` | (0.9＋0.1A)M₀.₄ |
| 18 | `mul_a0p4_b0p15` | (0.85＋0.15A)M₀.₄ |
| 19 | `mul_a0p4_b0p25` | (0.75＋0.25A)M₀.₄ |
| 20 | `mul_a0p5_b0p05` | (0.95＋0.05A)M₀.₅ |
| 21 | `mul_a0p5_b0p1` | (0.9＋0.1A)M₀.₅ |
| 22 | `mul_a0p5_b0p15` | **(0.85＋0.15A)M₀.₅，旧等权加法基线** |
| 23 | `mul_a0p5_b0p25` | (0.75＋0.25A)M₀.₅ |
| 24 | `mul_a0p6_b0p05` | (0.95＋0.05A)M₀.₆ |
| 25 | `mul_a0p6_b0p1` | (0.9＋0.1A)M₀.₆ |
| 26 | `mul_a0p6_b0p15` | (0.85＋0.15A)M₀.₆ |
| 27 | `mul_a0p6_b0p25` | (0.75＋0.25A)M₀.₆ |
| 28 | `equal3` | (C＋T＋A)/3 |
| 29 | `entropy` | wC·C＋wT·T＋wA·A |

其中add通式为(1−β)Mα＋βA；mul为[(1−β)＋βA]Mα。基线沿用旧计算顺序，必须逐字节重现旧等权加法Top-500；本轮基线不是更早的CTB乘法方案。所有29项均使用冻结自监督CNN，熵权无标签、无训练不代表整套方法属于无训练方法。

## 文献依据与熵权适配

[Wang与Wang（2023），Similarity searching for fault diagnosis of defect patterns in wafer bin maps](https://doi.org/10.1016/j.cie.2023.109679)在结构／位置距离内使用加法，并将BBP相似度乘缺陷数量差异惩罚；它没有CNN融合，也没有给出本项目α／β网格。原文α＝5控制缺陷数量惩罚，μ＝1控制径向距离，不能移作当前同名字母的权重。

[Kang、Shin与Lee（2024），Similarity searching for wafer bin maps by measuring shape, location, and size similarities of defect patterns](https://doi.org/10.1016/j.cie.2024.110486)的§3.5将整图shape/location/size加权相加，并从同一query对候选的分数分布计算熵权；内部shape与size又分别使用乘法。本文借鉴其融合框架，**不声称复现其整图特征或采用其给定的最优数值**。α、β网格是本项目预设实验参数。该论文数量相似度的印刷公式与数值例子不一致，本轮没有采用该数量公式。

本轮entropy对每个查询区域分别计算。其样本集合是完整固定gallery中的全部真实区域，排除同一query晶圆的所有区域；每个区域计一次，不补齐到Top-4、不按晶圆平均、不仅使用Top-500或抽查候选。每个通道m∈{C,T,A}使用：

`p_mj = s_mj / sum_j s_mj`

`H_m = -sum_j p_mj * ln(p_mj)`

`w_m = exp(-H_m) / sum_k exp(-H_k)`

同一查询区域的权重用于其全部候选区域，然后执行原贪心匹配和原查询面积聚合。自然对数、softmax温度1；不除以log(候选区域数)。不使用候选标签来算熵或权重，test时只执行已经冻结的计算规则，不用test标签选参数。各通道全零时该通道权重为0；正数常数通道保留H＝log(N)；三个通道都为零时使用等权。这些边界处理是项目实现约定。正式all_points J要求每个真实区域有正质量，缺失或非法描述直接拒绝；空晶圆仍是固定候选库成员，无区域匹配时得分为0。

## 运行、冻结选择与登记

代码先读取并校验旧区域缓存、模型和冻结协议。valid使用500个query及3599张gallery，每query排除自身后3598个候选；test使用全部7199个query及7199张gallery，每query7198个候选。exact与coverage复用同一排名，保留K＝1/5/10/20/50/100/500和全部分组；AP／Recall分母、coverage IDCG来自完整候选库。

在仓库根目录、原PyTorch环境执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s f_region_area_fusion/tests -v
python -m f_region_area_fusion.run --stage valid --dry-run
python -m f_region_area_fusion.run --stage valid --check-inputs
python -u -m f_region_area_fusion.run --stage valid --workers 8
python -u -m f_region_area_fusion.run --stage test --workers 8 \
  --source-run artificial/f_region_area_fusion_valid_实际启动时间
```

也可用`bash f_region_area_fusion/launch_tmux.sh`，它在valid成功完成后自动启动test。valid/test分别写入`artificial/f_region_area_fusion_valid_<实际时间>/`和`artificial/f_region_area_fusion_test_<实际时间>/`；每个变体×split单独持久化真实Asia/Shanghai启动时间与run_id。恢复使用同阶段`--resume-root`；参数、来源、代码或数值环境改变必须新建运行，拒绝覆盖既有产物。

仅以valid总体exact P@10的整数命中总数排序全部29项，取前10项。并列时先考虑基线`mul_a0p5_b0p15`，再按名称字典序；**基线只在并列时优先，没有保送名额**。完整选择及有序10项写入带签名的`selection.json`。test只读取该选择，绑定原valid选择、完整索引和根完成标记SHA；不得按test成绩重新指定最优参数。多项test结果属于已授权的预设补充比较，论文主结论不能把test最高者追述为valid唯一选中方案。

核验覆盖独立公式抽查（37类各1个query×20候选×阶段全部配置）、熵权使用完整候选区域的独立计算、贪心聚合、所有query的全候选评分重放、SHA同分排序及逐query／汇总双指标重算。valid基线必须重现旧2585/5000命中；若基线进入前10，test也重放原等权加法结果，否则不额外评价基线。全量评分重放使用生产公式，不能与独立公式抽查混称。

阶段正式完成以根`completed.json`及其绑定文件校验通过为准。登记器核验完整29项valid或恰好10项冻结test，拒绝部分结果、失效来源或缺失双指标。每根可单独登记：

```bash
python -m f_region_area_fusion.register --run-root artificial/实际运行目录
python -m f_region_area_fusion.register --run-root artificial/实际运行目录 --apply
```

默认只生成该运行目录内的`catalog_sources.json`；`--apply`在目录锁内collect/render/check，先备份共享目录文件，核对全部既有登记记录深度不变，写入失败则回滚。exact与coverage合并为一个变体×split登记项。登记、汇总和后续项目说明由主流程统一完成，本轮专属结果在正式核验后补充，不以合成测试作为实验成绩。
