# 区域J、CNN与面积融合

本轮固定已有区域表示，比较29组预设评分。先完成全部valid，以整数exact P@10选出10组并冻结，再对这10组做完整test；不根据test结果继续调参或重选主方法。**29项valid与冻结前10项的完整test均已完成并通过正式核验，共39项新结果已登记。** 本轮属于观察历史结果后的事后扩展，不自动替换项目主方法。

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

默认只生成该运行目录内的`catalog_sources.json`；`--apply`在目录锁内collect/render/check，先备份共享目录文件，核对全部既有登记记录深度不变，写入失败则回滚。exact与coverage合并为一个变体×split登记项。登记、汇总和后续项目说明由主流程统一完成；下节记录已核验的valid与test结果，不以合成测试或运行中产物作为实验成绩。

## 本轮valid结果

正式valid目录为`artificial/f_region_area_fusion_valid_20261120T155825+0800/`，父目录时间对应启动器，根manifest的实际开始时间为2026-11-20 15:58:38（+08:00）。29项全部完成后，于**16:04:14（+08:00）**冻结前10项。

valid主选为`add_a0p4_b0p25`，展开后为**S＝0.30C＋0.45T＋0.25A**。Top-10命中2613/5000，exact P@10＝**52.26%**；相对原等权面积乘法基线的2585/5000、51.70%，增加28次命中，即**+0.56个百分点**。同协议WBBS为2615/5000、52.30%，本轮主选仍低2次命中、0.04个百分点，不能表述为超过WBBS。

以下按冻结的valid顺序排列，均为总体指标；三个指标列的单位为%。完整29项见[原始valid汇总CSV](../artificial/f_region_area_fusion_valid_20261120T155825+0800/comparison_valid.csv)，选择身份见[selection.json](../artificial/f_region_area_fusion_valid_20261120T155825+0800/selection.json)。

| valid排名 | 配置 | Top-10命中／5000 | exact P@10 | exact mAP@100 | Coverage-nDCG@10 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `add_a0p4_b0p25` | 2613 | 52.26 | 16.2784 | 84.5690 |
| 2 | `add_a0p5_b0p25` | 2606 | 52.12 | 16.0816 | 84.5490 |
| 3 | `add_a0p5_b0p15` | 2592 | 51.84 | 16.2581 | 84.8576 |
| 4 | `mul_a0p4_b0p25` | 2589 | 51.78 | 16.2014 | 84.7545 |
| 5 | `add_a0p4_b0p15` | 2587 | 51.74 | 16.3526 | 84.7084 |
| 6 | `mul_a0p5_b0p15`（旧基线） | 2585 | 51.70 | 15.9755 | 84.7891 |
| 7 | `mul_a0p5_b0p25` | 2571 | 51.42 | 16.0983 | 84.6790 |
| 8 | `add_a0p5_b0p1` | 2568 | 51.36 | 15.9849 | 84.7939 |
| 9 | `add_a0p6_b0p25` | 2562 | 51.24 | 15.6133 | 84.0459 |
| 10 | `add_a0p4_b0p1` | 2560 | 51.20 | 15.9825 | 84.6643 |

主选的mAP@100比旧基线提高0.3029个百分点，但Coverage-nDCG@10降低0.2201个百分点；WBBS的mAP@100为19.2257%，也仍高于主选的16.2784%。因此结论限定为本轮valid网格的P@10最优，不是所有指标最优。WBBS未登记该valid的coverage，不补数。

`equal3`为2536/5000、50.72%，排名第13；`entropy`为2528/5000、50.56%，排名第16，两者均未入选本轮test。文献启发的区域熵权在当前固定分数尺度下未改善基线；这个结果不能推广为熵权在其他表示或尺度下无效。

[独立valid审计](../artificial/f_region_area_analysis_20261120T160516+0800/valid_audit.json)逐项核对29组保存排名、原始标签Top-10命中、AP@100、Coverage-nDCG@10及各分组，合计48,669个指标值，最大绝对误差4.44×10⁻¹⁶。正式流水线另完成全候选评分重放与全部K／双指标核验；独立区域公式／贪心抽查共21,460次，最大误差7.77×10⁻¹⁶。基线250,000行排名与旧结果逐字节相同，55,202个历史文件保护检查为unchanged。

根完成标记绑定359个文件，SHA-256为`37742b9d426b22863217d9106be60823125d8b830d40e07e4d7cbf92016cba05`。29项valid已登记，原364项记录深度保持不变，服务器官方collect/check通过，共393项登记。后续test按以上冻结10项完成，没有依据test分数改变主选；结果见下节。

## 完整test结果

正式test目录为`artificial/f_region_area_fusion_test_20261120T160426+0800/`，根manifest实际开始于2026-11-20 **16:04:47（+08:00）**，进程于**17:17:33.682488（+08:00）**成功退出，合计**72分46.68秒**，包含10组完整评价与正式核验。该耗时不是单对检索延迟。

主方案仍为valid事先选中的`add_a0p4_b0p25`，区域对评分**S＝0.30C＋0.45T＋0.25A**；没有按test成绩另选参数。下表严格沿用冻结的valid顺序，所有行采用同一7199-query、每query7198候选协议；指标列单位为%。来源：[完整test汇总CSV](../artificial/f_region_area_fusion_test_20261120T160426+0800/comparison_test.csv)及[冻结valid选择](../artificial/f_region_area_fusion_test_20261120T160426+0800/frozen_selection.json)。

| valid排名 | 配置 | test Top-10命中／71990 | exact P@10 | exact mAP@100 | Coverage-nDCG@10 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `add_a0p4_b0p25` | 43037 | 59.7819 | 12.4575 | 87.8135 |
| 2 | `add_a0p5_b0p25` | 42708 | 59.3249 | 12.2685 | 87.5906 |
| 3 | `add_a0p5_b0p15` | 42442 | 58.9554 | 12.1834 | 87.7352 |
| 4 | `mul_a0p4_b0p25` | 42705 | 59.3207 | 12.2531 | 87.8942 |
| 5 | `add_a0p4_b0p15` | 42650 | 59.2443 | 12.2706 | 87.8414 |
| 6 | `mul_a0p5_b0p15`（旧基线） | 41858 | 58.1442 | 11.8643 | 87.6135 |
| 7 | `mul_a0p5_b0p25` | 42502 | 59.0388 | 12.1579 | 87.7634 |
| 8 | `add_a0p5_b0p1` | 41751 | 57.9956 | 11.8247 | 87.5263 |
| 9 | `add_a0p6_b0p25` | 41887 | 58.1845 | 11.8746 | 87.0724 |
| 10 | `add_a0p4_b0p1` | 41887 | 58.1845 | 11.8819 | 87.5694 |

| 同协议test对照 | exact P@10（%） | exact mAP@100（%） | Coverage-nDCG@10（%） |
| --- | ---: | ---: | ---: |
| 本轮valid主选 | 59.7819 | 12.4575 | 87.8135 |
| 旧等权面积乘法基线 | 58.1442 | 11.8643 | 87.6135 |
| WBBS | 58.1289 | 13.2996 | 87.8320 |

主方案命中43037/71990，比旧基线多1179次，P@10提高**1.6377个百分点**；mAP@100提高0.5931个百分点，Coverage-nDCG@10提高0.2001个百分点。相对WBBS，P@10提高**1.6530个百分点**，但mAP@100低0.8421个百分点，Coverage-nDCG@10低0.0185个百分点。因此支持前排exact检索改善，不能声称全面超过WBBS。WBBS取[登记指标表](../results_catalog/metrics.csv)中`wbbs_test`、`retrieval_v1`、`all`，不混入跨组合coverage协议。

正式核验覆盖10×7199＝71,990个query实例、518,184,020个完整候选分数、35,995,000行Top-500；全部候选排序与保存分数一致，保存分数最大误差0，双指标独立重算最大绝对误差2.22×10⁻¹⁵。37类各1个query×20候选×10配置共7400次独立区域公式／贪心抽查，最大误差5.55×10⁻¹⁶。独立公式抽查与使用生产评分器的全量重放分别记录，不混称为独立重算全部分数。

旧基线的3,599,500行排名逐字节重现；55,562个历史文件保护检查为unchanged。test根完成标记绑定131个文件，SHA-256为`512c06b2406cc39ff3e1e9fe12756f9de45555025fbe91fab1c771f54531a8db`。10项test已追加登记，共403项，原393项记录保持不变；官方collect/check通过。

主方案的β＝0.25位于本次预设搜索的上界。结果只支持这29个配方中的valid选择，不能推断连续参数空间的全局最优，也不能推断面积权重越大越好。

[最终独立审计](../artificial/f_region_area_analysis_20261120T160516+0800/test_audit.json)已通过，结合全部valid与10组test共核对280,657个指标值，最大绝对误差4.44×10⁻¹⁶。它从本轮保存的完整Top-500与原始标签重新核对P@10、AP@100、Coverage-nDCG@10及总体／分组／37类汇总，检查候选身份、自身排除、分数顺序、SHA同分排序和签名来源。旧基线与WBBS的逐query报告有哈希绑定并重聚合；该审计没有重新运行旧方法或CNN/J评分。

主方案在单标签与混合标签两组的P@10均高于旧基线及WBBS，但各类收益不一致。37个完整组合类别中，相对旧基线为24类提高、1类持平、12类降低；相对WBBS为21类提高、16类降低。三标签组相对旧基线低0.0958个百分点，四标签组相对WBBS低5.6875个百分点；不能由总体提升推断所有混合复杂度均改善。

[配对bootstrap产物](../artificial/f_region_area_analysis_20261120T160516+0800/bootstrap.json)采用2000次重抽、seed2026，在37个完整标签组合内有放回抽取query并保留各类原有数量；三种方法、三个指标共用同一组抽样。下表为“主方案减对照”的差值，单位均为百分点，区间为逐项95%百分位区间。

| 对照 | 指标 | 差值 | 逐项95%区间 |
| --- | --- | ---: | --- |
| 旧等权面积乘法基线 | exact P@10 | +1.6377 | [+1.3752, +1.8781] |
| 旧等权面积乘法基线 | exact mAP@100 | +0.5931 | [+0.5473, +0.6386] |
| 旧等权面积乘法基线 | Coverage-nDCG@10 | +0.2001 | [+0.0870, +0.3092] |
| WBBS | exact P@10 | +1.6530 | [+1.1527, +2.2004] |
| WBBS | exact mAP@100 | -0.8421 | [-0.9856, -0.6859] |
| WBBS | Coverage-nDCG@10 | -0.0185 | [-0.2761, +0.2472] |

在此重抽样口径下，主方案相对旧基线的三个指标差值区间均高于0；相对WBBS的P@10区间高于0、mAP@100区间低于0，Coverage-nDCG@10区间跨0。这些区间以固定模型、配方、候选库及观察到的类别比例为条件，只重抽query，不包含跨数据集、训练种子或共享候选库变化。六条区间是逐项区间，未对先前test观察、29项valid选参、10项test报告或六项同时比较作校正，不是同时覆盖保证。
