# 区域 CNN 的两组受控改进

实验隔离于本目录，来源模型和历史科学代码只读。公共协议、结果登记与论文证据分别见[项目说明](../doc/项目说明.md)、[登记规则](../results_catalog/README.md)和[论文实验证据总表](../doc/论文实验证据总表.md)。

## 固定设置

只训练两个新模型。原基线为八连通区域编码器的 seed=2026、`best_nce` 第100轮，直接复用。两个新模型均从原 seed=2026 初始化开始，保持 ResNet18、三通道52→96窗口、GAP、h512／z128、SGD、100轮／39,000步、每样本5,000个有放回负例以及原memory bank更新。不是继续微调已有100轮模型。全部参数及预设判据在实际运行前写入 `frozen_plan.json`；源码和输入逐阶段绑定SHA-256。

| 新模型 | 唯一干预 | 不使用的信息 |
|---|---|---|
| `context_dropout` | 第二视图在原52×52网格以0.5概率删除非目标缺陷像素，再resize并使用原旋转；第一视图和检索保持完整窗口 | 类别标签、测试表现 |
| `filtered_negatives` | 冻结旧基线在train的原图／90°特征；两视图均为跨晶圆双向Top500的区域作为伪邻居；将原负例抽样中的命中槽位补抽至恰好5,000个 | 类别标签、valid/test特征、额外正例 |

两组均固定使用第100轮评价；第15／25／50／75／100轮快照只用于诊断。valid的原旋转NCE只是共同诊断，不选最低损失checkpoint。新增随机数流独立，不改变原batch顺序、旋转和共享负例抽样流。目标保留指原固定窗口中已有的目标像素不再被增强删除；不声称恢复原窗口已裁去的区域。过滤图为固定FP32穷举余弦排序，关闭TF32，边界并列按原token索引处理；同晶圆负例不额外排除。

## 运行入口与接口

在仓库根目录、原PyTorch CUDA环境执行。`init`打印带真实启动时间的结果目录；后续命令的`RUN`填该完整路径。各训练／评价另存自身真实开始时间；断点恢复保留原身份。

```bash
python -u -m cnn_refinement.run --stage init
python -u -m cnn_refinement.run --stage train --variant context_dropout --output-root "$RUN" --device cuda:0
python -u -m cnn_refinement.run --stage teacher --output-root "$RUN" --device cuda:2
python -u -m cnn_refinement.run --stage train --variant filtered_negatives --output-root "$RUN" --device cuda:1
python -u -m cnn_refinement.run --stage evaluate --output-root "$RUN" --device cuda:3 --workers 8
python -u -m cnn_refinement.run --stage negative-diagnostics --output-root "$RUN"
python -u -m cnn_refinement.run --stage diagnose --output-root "$RUN" --device cuda:3
python -u -m cnn_refinement.run --stage report --output-root "$RUN"
python -u -m cnn_refinement.run --stage finalize --output-root "$RUN"
python -m cnn_refinement.register --run-root "$RUN" --apply
```

两组训练可并行；负例过滤训练必须等待教师图完成。CPU评分使用`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`。恢复训练增加`--resume`；改变科学设置、模型或已绑定源码必须使用新的运行身份与目录。完成登记后仍须执行公共登记说明中的`collect/check`和两类归档核验。

本轮已有进程的续行使用`python -u -m cnn_refinement.followthrough --output-root "$RUN"`，依赖已持久化的`launch_*.json`。它只等待这两次训练并启动后续评价、诊断、报告和封存，不能启动训练；进程或校验失败即停止。再次接入可加`--resume`，只复用已封存阶段或跟随原活进程，不自动重试失败阶段。

核心接口为 `training.train_variant`、`neighbors.build_neighbor_graph`、`evaluation.evaluate_model`、`diagnostics.run_diagnostics`。`features.extract`对完整原窗口推理，保留原token身份、顺序和旧float64归一化口径。基线检索直接使用旧h512文件；不重新提取而冒充逐字节重放。

## 评价与解释边界

三个模型分别评价CNN单独评分，以及固定融合`0.75*(0.4*C+0.6*T)+0.25*A`；只替换深度特征，原联合分布、面积、单向贪心一对一和查询`sqrt(area)`聚合不变。使用去重后的固定valid500／test7199 query及完整gallery。普通与跨组合均从全候选评分重新排名；每项均核对全候选排序、指标分母、分组汇总和Top500。四项旧融合基线按原key引用，不重复登记。

主比较列出ordinary exact P@10及mAP@100、cross Coverage-nDCG@10及FullCoverage P@10，保留七个K与全部分组。2000次配对区间按完整标签组合分层；其含义是固定模型／库条件下的query差异，不是多seed稳定性。上下文诊断使用相同原区域与合法四视图，先在晶圆内聚合，单类标签仅作区域语义代理。过滤图另报告真实过滤曝光、零干预比例、重复采样与教师相似度变化。

预设主要标准：上下文drop50与dropall的h512距离均相对减少至少30%，且无塌缩或近邻退化；完整融合exact P@10至少提高1个百分点且配对95%区间下界大于0，cross nDCG@10区间下界高于−0.5个百分点；表示改善还须得到CNN单独检索的支持。两个新模型均评价并报告，不按valid表现筛除。不以单次训练证明稳定优越，也不把负例变容易与语义假负例减少作因果区分。

## 本轮状态

2026-11-23完成，独立结果根为`results/20261123T105220+0800__cnn_refinement/`。两模型均完成100轮／39,000步，原基线未重训；全部valid、普通／跨组合test、诊断和根封存通过核验。24个评价角色中20项新增正式登记，4个历史融合基线精确重放并保留原key；总表由491增至511项，原491项结果对象不变。服务器官方collect/check及两套归档核验通过。

两组均未达到预设检索提升标准。原基线／上下文增强／无标签负例过滤的test融合exact P@10为59.7819%／58.7957%／59.5374%，CNN单独评分为32.5156%／23.8130%／30.3681%。上下文增强将drop50／dropall距离降低87.01%／33.90%，但单类近邻质量和检索同时下降；降低敏感性不等于更好的形状表示。负例实际替换0.27312%的槽位，不能将其结果推广为所有假负例处理均无效。详细结果、条件配对区间及论文展示建议见[消融实验](../doc/消融实验.md#cnn-refinement)。

实现验证为服务器93项通过、本地92项通过及1项CUDA跳过；完成后独立核对40个query均值差、6个关键配对区间与6个上下文均值，数值一致。四组PNG经过视觉检查，PDF/SVG一并保存。写作所需414个文件、174,460,104字节已下载到本地同名结果目录并逐SHA-256核验；[下载回执](../results/20261123T105220+0800__cnn_refinement__local_download.json)列明范围。权重、特征／评分数组及完整排名留在服务器，原完成标记不代表本地子集拥有全部大产物。

G6证据已完成、尚未写入中英文稿；建议用作负结果与局限讨论，保留原论文主模型。单seed、固定模型／图库条件区间、事后研究及教师负例难度改变的边界不变。
