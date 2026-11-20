# 区域内J融合F的跨组合coverage

本轮只评价原29项valid事先选出的`add_a0p4_b0p25`，不搜索其他配方。区域对评分为`0.75*(0.4*C+0.6*T)+0.25*A`，文稿等价式为`0.30C+0.45T+0.25A`；实现保留原嵌套运算顺序。C为冻结CNN相似度，T为区域128维联合分布的`exp(-0.5*Hellinger)`，A为原相对面积相似度。沿用cc8、Top-4、原单向贪心一对一及查询区域`sqrt(area)`聚合，不另加径向位置项，无新增训练。

这项是区域内J改进F，与上一轮最优匹配F分数和整图J分数的`0.5F+0.5J`不同。来源valid为`artificial/f_region_area_fusion_valid_20261120T155825+0800/`，已有普通test为`artificial/f_region_area_fusion_test_20261120T160426+0800/`；比较引用六项`artificial/fgj_cross_test_20261120T193642+0800/`，不重复登记旧六项。

## 协议与运行

状态：实现与来源预检通过，正式评价待启动。

沿用`coverage_cross_combination_v1`：全部7,199个test查询，排除自身和相同完整标签组合后，每query有6,999–7,170个候选，合计50,395,860个有效配对。保留原六项coverage指标、K=1/5/10/20/50/100/500及全部分组；IDCG与Recall/AP分母使用完整过滤候选池，不从旧Top-500过滤。nDCG有效query为7,000，FullCoverage Recall/mAP为6,200，其余指标为7,199。主展示按Coverage-nDCG@10排序，同时保存每项指标独立名次，仅作描述性比较。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest discover -s f_region_j_cross_test/tests -v
python -m f_region_j_cross_test.run --dry-run
python -u -m f_region_j_cross_test.run --workers 8
```

默认自动分配实际Asia/Shanghai开始时间和独立目录；支持`--output-root`及签名相同的`--resume-root`。科学设置、模型、源码或数值环境改变需新目录。使用原NumPy/PyTorch环境，8个CPU进程按query分片，每进程BLAS单线程；没有改写GPU算子或浮点精度。

## 完整分数与验证范围

先复用原区域缓存与评分器，为每query生成全部7,198个普通候选分数，在新目录保存只读float64完整矩阵（约414.6MB）。然后逐query核对原普通test的全部Top-500身份、顺序和分数，确认没有改变原方法，再按跨组合候选资格从完整矩阵排名。旧模型、区域缓存、数据协议及历史结果保持只读。

37类代表query各取20个跨组合合格候选，进行独立区域公式与标量贪心审计。全部新排名由独立SHA并列排序重验完整候选池，独立复算六项coverage及所有query／类别／宏平均／标签数量分组。此核验读取新完整分数缓存，不是再次计算全部区域分数；独立区域公式核对范围是上述740对，不将其扩大称为全量独立公式审计。

正式完成要求根及子项完成标记、来源链、缓存、历史重放与独立报告全部校验通过。登记及官方汇总命令：

```bash
python -m f_region_j_cross_test.register --run-root <本轮正式目录> --apply
python results_catalog/build.py collect --workspace /data/wangkaiyan2024
python results_catalog/build.py check
python results_catalog/snapshot_archive.py pack
python results_catalog/snapshot_archive.py verify
python results_catalog/generated_archive.py pack
python results_catalog/generated_archive.py verify
```

公共概览见[项目说明](../doc/项目说明.md)，完整test比较维护在[消融实验](../doc/消融实验.md)，来源方法见[区域J与面积融合](../f_region_area_fusion/README.md)。新完整分数和排名留在服务器，本地同步轻量报告；不依据本次test重选参数或替换项目主方法。
