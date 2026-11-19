# 冻结前十项的跨组合 coverage 评价

复用原 valid 搜索冻结的十组 F/G/WBBS 配方及已完成普通 test 的完整分项分数。唯一实验变化为：每个 query 在粗筛、归一化和排序之前，排除自身及完整标签组合相同的候选。使用现有 `coverage_cross_combination_v1` 候选协议和原 `coverage_v1` 评价器，不增加指标、重训或重新选参。

## 来源与范围

来源为 `artificial/fgw_topk_test_20261119T205138+0800/`；其原 valid 来源为 `artificial/fgw_grid_20261119T200653+0800/`。十项按冻结 valid 顺序保存，原第一项 `0.1F+0.4G+0.5W` 仍是原选方案。此次跨组合结果不产生新的胜者选择。

完整 F/G/WBBS 矩阵经来源链和内容哈希校验后只读使用。两项 min-max 配方在过滤后的完整候选向量上重新归一化；其他配方使用同样过滤后的分项向量。标签只用于候选资格和评价，不进入方法评分。

保留全部7,199个 test query，每个 query 有6,999–7,170个候选。沿用来源的 K=1/5/10/20/50/100/500、总体、类别宏平均、单／混合及标签数量分组，只生成原 coverage 系列报告。FullCoverage Recall/AP 的有效 query 为6,200，Coverage-nDCG为7,000，其余指标为7,199；零分母处理完全沿用原实现。普通 exact/coverage 结果保留原位，跨组合结果独立登记。

## 执行与验证

在持有完整来源产物的服务器、仓库根目录及原 PyTorch 环境执行：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u -m fgw_cross_test.run \
  --source-root artificial/fgw_topk_test_20261119T205138+0800 --workers 8
```

默认新建带真实启动时间的 `artificial/fgw_cross_test_<时间>/`。支持 `--output-root` 指定新目录、`--resume-root` 按原签名恢复及 `--dry-run` 只读查看计划。每项评价保存独立实际启动身份、Top-500排名、原coverage报告、独立核验及完成标记。

独立核验按每个 query 重新计算过滤后的完整候选分数、同分顺序、Top-500和全部原指标，并检查逐query、逐类及分组汇总。来源数据、模型、协议、缓存和旧结果在运行前后比对哈希、大小及mtime。只有全部十项完成、独立核验和来源保护检查通过，父批次才封存。

完成后登记：

```bash
python -m fgw_cross_test.register --run-root artificial/fgw_cross_test_原启动时间 --apply
python results_catalog/build.py collect --workspace /data/wangkaiyan2024
python results_catalog/build.py check
```

公共协议与登记规则见[项目说明](../doc/项目说明.md)，实验设计和结果见[消融实验](../doc/消融实验.md)。本实验是已观察普通test后的追加候选协议诊断，不作为训练未见组合的泛化证据。

## 本次完成记录

独立运行目录为`artificial/fgw_cross_test_20261119T223644+0800/`，启动代码`363b710`，8个worker。来源仍为`artificial/fgw_topk_test_20261119T205138+0800/`的完整矩阵和原valid冻结前十项；没有新训练、权重搜索或test选参。实际启动记录显示2026-11-19T22:36:44.924610+08:00启动，2026-11-19T22:52:16.076995+08:00退出0，耗时931.15秒（约15分31.15秒），包含来源核验、十项评价、独立复核和封存；登记与同步另计。

十项均完成7,199个query及其全部合格候选的评分和独立Top-500核验；每项50,395,860个查询—候选对，合计503,958,600个配对、35,995,000行Top-500。保存评分最大绝对误差为0，保存分项最大绝对误差为0，指标最大绝对误差为5.551e-16。8,881个受保护来源文件的哈希、大小及mtime保持不变。本次复用已经校验的完整F/G/W矩阵，独立重算融合公式、排名及指标；没有重新提取特征或重新运行WBBS原始匹配。

两项min-max配方在过滤后的完整候选池上重新计算极值；原六项coverage指标及全部七个K保持不变。Coverage-nDCG、FC Recall/mAP和其余指标的有效query分别为7,000、6,200和7,199，逐query／逐类／分组报告均已核验。

原valid主选$0.1F+0.4G+0.5W$的Coverage-nDCG@10、MeanCoverage@10、FullCoverage P@10、FullCoverage mAP@100为74.9292%、69.9498%、34.4201%、2.4777%；仍保持原选方案身份。完整十项、六指标×七K及同协议WBBS历史参考见[消融实验的跨组合结果](../doc/消融实验.md#6-valid前十项的跨组合coverage评价)。

原始排名和完整分项缓存保留服务器；本地保存轻量报告及来源证据。本轮未重新计算WBBS独立基线，也未新增指标或显著性检验。

十项已独立登记，原309项登记内容保持深度一致，总计319项；官方`collect`和`check`通过。新入口27项测试在本地和远程原环境均通过，原61个评分及评价依赖文件哈希未变。轻量报告、文档和登记汇总同步本地；完整排名及分项矩阵保留远程来源位置。
