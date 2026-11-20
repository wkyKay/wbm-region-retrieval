# F与WBBS等权加法的跨组合评价

用户指定固定评分`0.5*F + 0.5*WBBS`，在已完成的F、F×WBBS跨组合实验之后补充一项评价。两项系数均为0.5，直接使用原始分数；不引入G或min-max归一化，不搜索权重。

F仍为前十项融合实验中的cc8最优区域匹配分项，W仍为同一来源的原始WBBS分数。只读复用`artificial/fgw_topk_test_20261119T205138+0800/`经签名和哈希校验的完整分项缓存，不重训、不重新提取特征。与F、F×WBBS和WBBS的比较使用同一个`coverage_cross_combination_v1`候选集：7,199个query，每query排除自身及完整标签组合相同的候选后有6,999–7,170个候选。

沿用原`coverage_v1`六项指标、K=1/5/10/20/50/100/500、全部原分组及零分母规则。等权系数不意味着两分项的数值分布相同，本轮不额外调整其尺度。

## 运行与核验

在持有完整来源产物的服务器仓库根目录、原PyTorch环境执行：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u -m fw_add_cross_test.run \
  --source-root artificial/fgw_topk_test_20261119T205138+0800 --workers 1
```

默认使用真实启动时间创建`artificial/fw_add_cross_test_<时间>/`；`--output-root`指定新目录，`--resume-root`仅恢复签名一致的原运行，`--dry-run`只读检查计划。每项保存时间身份、Top-500、原coverage报告、独立全候选复核及来源保护证据，完成后封存。

```bash
python -m fw_add_cross_test.register --run-root artificial/fw_add_cross_test_原启动时间 --apply
python results_catalog/build.py collect --workspace /data/wangkaiyan2024
python results_catalog/build.py check
```

公共规则见[项目说明](../doc/项目说明.md)，设计与结果见[消融实验](../doc/消融实验.md)。这是用户在查看前述test后指定的固定公式补充，不作为新的valid选参，也不替换原主选方案。

## 本次完成记录

`artificial/fw_add_cross_test_20261120T110737+0800/`于2026-11-20完成，启动代码`5d7cd13`，1个worker。实际启动2026-11-20T11:07:37.802505+08:00，结束2026-11-20T11:16:21.216527+08:00，耗时523.41秒，退出0；登记与同步另计。

全部7,199个query、50,395,860个合格配对及3,599,500行Top-500通过独立评分、同分顺序与原指标复核。保存评分及分项误差均为0，指标最大绝对误差为4.441e-16；9,009个受保护旧来源文件的哈希、大小和mtime不变。完成标记绑定的18个轻量文件已同步本地并核对哈希，完整排名与分项缓存保留服务器。

等权加法的Coverage-nDCG@10／MeanCoverage@10／FullCoverage P@10／FullCoverage mAP@100为76.1288%／71.0041%／36.1022%／2.5221%。相对乘法，前三项提高，mAP@100下降；相对WBBS也存在相同方向的取舍。FullCoverage Hit@10为66.3287%，高于乘法及WBBS、低于单独F。完整指标与分组见[消融实验](../doc/消融实验.md#7-f与wbbs的跨组合固定对照)。

新增入口25项检查在本地及远程原数值环境均通过，旧评分与指标实现保持不变。单项公式由用户预先指定，结果不构成新的valid选参或主方法替换，不作统计显著性宣称。

该项已正式登记，原321项记录深度一致，总计322项；服务器官方`collect`与`check`通过。已报告K=1/5/10/20的nDCG、MeanCoverage及FullCoverage P/Recall高于乘法，K=50/100/500则低于乘法；FullCoverage Hit的取舍另按原指标报告。
