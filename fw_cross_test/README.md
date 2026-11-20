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
