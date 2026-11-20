# F与F×WBBS的跨组合评价

用户在查看前十项跨组合结果后指定的两项补充实验：单独`F`与原始分数直接相乘`F*WBBS`。不搜索权重，不自动更换原valid主选方案。

F沿用前十项来源中的cc8最优区域匹配分项；它不是历史七方法表中的旧F，也不是改回贪心匹配。WBBS沿用同一来源的原始分数。两项均不使用G，不做min-max归一化，不重训或重新提取特征。

读取`artificial/fgw_topk_test_20261119T205138+0800/`经签名和哈希校验的完整分项缓存，在排序前排除自身及完整标签组合相同的候选。7,199个query、每query 6,999–7,170个候选，沿用`coverage_cross_combination_v1`和原`coverage_v1`六项指标、K=1/5/10/20/50/100/500及全部原分组。候选身份、分母与前十项跨组合评价一致。

## 运行与核验

在持有完整来源产物的服务器仓库根目录、原PyTorch环境执行：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u -m fw_cross_test.run \
  --source-root artificial/fgw_topk_test_20261119T205138+0800 --workers 2
```

默认创建带真实启动时间的`artificial/fw_cross_test_<时间>/`；`--output-root`指定新目录，`--resume-root`仅恢复签名一致的原运行，`--dry-run`只读检查计划。每项保存实际启动身份、Top-500、原coverage报告及独立核验。全部query的排序、同分顺序、评分和原指标用完整合格候选库复核，来源保护检查通过后封存。

完成后执行：

```bash
python -m fw_cross_test.register --run-root artificial/fw_cross_test_原启动时间 --apply
python results_catalog/build.py collect --workspace /data/wangkaiyan2024
python results_catalog/build.py check
```

公共规则见[项目说明](../doc/项目说明.md)，实验设计、完成状态和结果见[消融实验](../doc/消融实验.md)。本实验检验固定组件的表现，不能单独证明F权重与性能存在单调关系。

## 本次完成记录

`artificial/fw_cross_test_20261120T104447+0800/`已于2026-11-20完成，启动代码`5de869b`，2个worker。实际启动2026-11-20T10:44:47.534490+08:00，结束2026-11-20T10:53:35.494480+08:00，耗时527.96秒，退出0；登记与同步另计。

两项各完成7,199个query和50,395,860个合格查询—候选对，共保存7,199,000行Top-500。独立评分及保存分项误差均为0，全部原指标复核最大绝对误差为5.551e-16；8,981个受保护来源文件的哈希、大小与mtime不变。完成标记绑定的27个轻量文件已同步本地并逐项核对哈希，完整排名和分项缓存保留在服务器。

单独F的Coverage-nDCG@10／MeanCoverage@10／FullCoverage P@10／FullCoverage mAP@100为71.1717%／66.3145%／31.5433%／1.8366%；F×WBBS为75.9327%／70.8553%／35.8050%／2.6445%。F×WBBS前三项高于同协议WBBS，FullCoverage mAP@100仍低于WBBS；单独F四项均低于WBBS。结果支持当前固定F与WBBS搭配改善前排覆盖检索，不能推出F权重越高越好。完整各K与分组见[消融实验](../doc/消融实验.md#7-f与wbbs的跨组合固定对照)。

本地与远程原数值环境均通过25项新入口检查；原评分、模型与指标实现保持不变。这是观察前十项test后提出的固定组件诊断，保留原主选方案，不作显著性或跨种子结论。

两项已正式登记，原319项记录深度一致，总计321项；服务器官方`collect`与`check`通过。完整六项指标的取舍仍须分别看：单独F的FullCoverage Hit@10为68.6206%，高于F×WBBS的66.0369%及WBBS的65.7036%，所以不能概括为单独F在所有coverage指标上更差。
