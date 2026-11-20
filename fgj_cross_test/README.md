# F、G、J与WBBS的跨组合coverage评价

用户固定指定六项：0.5G+0.5J、0.5F+0.5J、F、G、J、WBBS。F经用户确认沿用前次跨组合评价的cc8最优区域匹配分项，包含原CNN、径向位置与相对面积评分；不是项目主实验的贪心版本，也不是区域内CNN/J加法。G沿用冻结WaPIRL整图特征分数，J固定为原128维整图联合分布的exp(-2D_H)，WBBS沿用原分数。

本轮无新增训练、分数标定或权重搜索。G/F仍使用已有自监督模型。六项均为当前阶段的固定补充评价，指标名次只作描述，不替换valid主选或项目主方法。

## 候选与指标

只改变候选资格：排除自身及与query完整multi-hot标签组合相同的候选，沿用coverage_cross_combination_v1。全部7,199个test query，每query 6,999–7,170个候选；每方法完整评分50,395,860个合格配对后保存Top-500，不能从旧Top-500过滤后冒充新检索结果。标签只决定benchmark候选资格及评价相关性，不进入模型评分。

原coverage_v1六项指标、线性覆盖增益、K=1/5/10/20/50/100/500及全部分组不变。Recall/AP分母与nDCG理想排名取过滤后的完整候选池。199个query无部分覆盖候选，nDCG有效query为7,000；999个query无FullCoverage正例，Recall/AP有效query为6,200；MeanCoverage、Hit、Precision仍使用全部7,199个query。只报告coverage，不添加exact指标或新的相关性指标。

主展示按Coverage-nDCG@10降序；metric_orderings.csv同时给出六指标各K的独立排名，相等数值共享名次。AP@K使用完整候选相关样本数作分母，不是full AP。四标签组的nDCG高不等于找到完整覆盖候选。

## 运行与核验

状态：六项正式test已于2026-11-20完成，实际开始19:36:42、结束19:56:20（Asia/Shanghai）。运行目录为`artificial/fgj_cross_test_20261120T193642+0800/`；[原始比较表](../artificial/fgj_cross_test_20261120T193642+0800/comparison.csv)与[逐指标排序](../artificial/fgj_cross_test_20261120T193642+0800/metric_orderings.csv)保留原始比例。

Coverage-nDCG@10的描述性次序为J（75.9569%）＞0.5F+0.5J（75.7287%）＞0.5G+0.5J（75.6399%）＞WBBS（74.8829%）＞F（71.1717%）＞G（60.5532%）。两个等权融合均未超过单独J；但FullCoverageHit@10以F最高（68.6206%，J为62.8004%），所以该次序不是全部指标的统一排序。完整六指标、七个K及分组解释见[消融实验](../doc/消融实验.md)。

本地与服务器49项专项测试通过。正式独立核验逐query重排全部50,395,860个合格配对，检查六公式的全部Top-500及全部coverage报告；保存分数／分项误差为0，指标最大绝对差5.55e-16。F与WBBS的7,199个query历史Top-500顺序完全一致，分数差0。769个使用来源绑定SHA；55,929个历史文件的大小／mtime保护检查通过，其中小于1MiB的文件额外核对SHA。完整封存核验在服务器完成，本地仅同步轻量报告。

来源为artificial/fgw_topk_test_20261119T205138+0800/的完整F/G/W矩阵，以及artificial/b1_structure_all_20261117T193244+0800/的G/J来源；核对最近整图G/J评价身份。所有旧模型、缓存、排名、协议及报告只读，新输出采用真实Asia/Shanghai时间目录。

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s fgj_cross_test/tests -v
python -m fgj_cross_test.run --check-inputs --device cuda
python -u -m fgj_cross_test.run --device cuda
```

支持显式新目录--output-root和签名一致的--resume-root。源码、模型或科学设置变更时必须使用新目录。全部六项共享每query的完整分项读取和平局排序键，独立验证另以六公式和独立SHA键重排全部候选，复核全部query/K/类别/宏平均/标签数量分组及零分母计数。37类代表各20候选独立核对G点积与标量Hellinger，另外核对37类代表的全部7198个原始候选G缓存。完成后逐query重放历史同协议F和WBBS Top-500，检查历史来源保护记录，再封存新目录。

完成判据为根completed.json、六子项完成标记和引用哈希通过，不能仅按退出状态或部分CSV判断成功。登记命令为：

```bash
python -m fgj_cross_test.register --run-root <本轮正式目录> --apply
python results_catalog/build.py collect --workspace /data/wangkaiyan2024
python results_catalog/build.py check
python results_catalog/snapshot_archive.py pack
python results_catalog/snapshot_archive.py verify
python results_catalog/generated_archive.py pack
python results_catalog/generated_archive.py verify
```

公共协议与结果概览见[项目说明](../doc/项目说明.md)，详细test结果维护在[消融实验](../doc/消融实验.md)，归档恢复见[登记说明](../results_catalog/README.md)。完整排名保留服务器，轻量报告同步本地；GitHub同步不包含原始数据和模型。
