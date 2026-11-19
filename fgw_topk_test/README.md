# F/G/WBBS valid前十名的test评价

独立扩展入口。按用户2026-11-19的明确要求，已对166项valid搜索的冻结前十名完成完整test评价。候选顺序和全部权重由原`selection.json`冻结，不根据test分数增删候选、调权重或改选主方案；原valid第一名仍为`0.1F+0.4G+0.5W`。科学定义与结果见[消融实验](../doc/消融实验.md)，公共协议及登记见[项目说明](../doc/项目说明.md)。

## 范围与接口

原`fgw_grid_search/`源码、配置、valid排名和单胜者test入口保持原样。本包复用其冻结评分、完整候选缓存和独立验证器；每个配方通过原公开评价函数单独执行，父进程最多并行8项。每个子目录包含父批次名，确保跨批次身份唯一。

仅接受已完成valid搜索的前十名，不接受新权重、任意变体列表或test选参。每个组合均使用全部7199个test查询及每query排除自身后的7198个候选；exact和coverage共享Top-500排名，保留全部K、类别及标签分组。归一化使用该query的完整候选库分数范围，不使用候选标签。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u -m fgw_topk_test.run \
  --search-root artificial/fgw_grid_20261119T200653+0800 --workers 8
```

输出为新的`artificial/fgw_topk_test_<实际启动时间>/`。支持`--dry-run`只读检查计划、`--output-root`指定新目录和`--resume-root`按原签名恢复。新目录只构建一份完整test F/G/W分项缓存，WBBS复用已校验矩阵；旧模型、协议、来源和valid产物只读。启动与恢复均校验原评分代码、数值环境及来源签名。

## 完成与登记

只有十项均完成全候选评分、排名与指标独立复核，且来源保护检查通过后才封存父目录。`frozen_selection.json`记录十项valid选择依据，`evaluation_index_test.json`保存按valid顺序的test汇总；每项有独立真实时间身份、原始报告、排名和完成标记。

```bash
python -m fgw_topk_test.register --run-root artificial/fgw_topk_test_原启动时间 --apply
python results_catalog/build.py collect --workspace /data/wangkaiyan2024
python results_catalog/build.py check
```

登记保留所有旧记录，快照采用既有紧凑JSON格式。代码、文档和结果汇总同步GitHub；完整排名与缓存保存在服务器，轻量报告同步本地。该方案是在观察历史test后提出的补充研究，十项test结果是预先固定候选的比较，不作为新的调参依据。

## 本次完成记录

独立目录为`artificial/fgw_topk_test_20261119T205138+0800/`，来源搜索为`artificial/fgw_grid_20261119T200653+0800/`。启动代码`da63a2f`，8个worker，2026-11-19 20:51:38启动，21:30:40退出0；[实际启动记录](../artificial/fgw_topk_test_20261119T205138+0800.launch.json)给出2341.72秒（约39分02秒），包含来源检查、共享缓存、十项评价、独立验证和封存，登记与同步另计。

十项均完成7199个query×7198候选，按valid原顺序保存，根完成标记绑定152个文件。独立验证逐query重算完整候选公式与Top-500顺序，并核对全部K、逐类和标签分组指标；保存评分／分项误差为0，指标最大绝对误差2.3315×10⁻¹⁵，8712个受保护来源文件的哈希、大小及mtime保持不变。新包33项合成测试在本地和远程通过，旧网格78项本地回归通过，原评分依赖57个文件哈希不变。

[额外只读缓存审计](../artificial/fgw_topk_test_20261119T205138+0800.cache_audit.json)确认三个分项矩阵为7199×7199、float64、原始query/gallery身份一致；全部51,818,402个非自身W分数与旧完整W矩阵逐位一致，历史FGW全部3,599,500行Top-500中的F/G/W及固定融合分数也逐位一致。该来源对照没有覆盖历史Top-500之外的F/G，也没有重新运行WBBS原始匹配。

原valid主选方案仍为`0.1F+0.4G+0.5W`，test P@10／mAP@100／Coverage-nDCG@10为61.5974%／14.3711%／88.8642%。test P@10最高观察值来自valid第3名（61.8878%），其他两项最高值分别来自第4／10名；这些结果不改写valid选择，也不构成本轮差值显著性的证明。按valid排名排列的十项主表、完整K和分组结果见[消融实验](../doc/消融实验.md#5-valid前十项的独立test评价)。

十项test已单独登记，旧299项记录完整深度一致，总登记309项；官方`collect`／`check`通过。紧凑`snapshot.json`为90,108,594字节（约85.93 MiB），未删除数据或修改旧记录。完整排名及缓存保留服务器，轻量报告、冻结选择和签名副本同步本地，代码、文档与登记汇总纳入GitHub同步。
