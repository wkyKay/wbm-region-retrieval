# 旋转编码器的区域评分解释与遮蔽验证

本入口完成当前主方法的D1–D4，复用任意角旋转CNN固定第100轮的原视图特征，评分为`0.85*(0.4*C+0.6*T)+0.15*A`。来源必须是已封存的本轮`paper_c_rotation`运行；历史`paper_d_explanation`及旧产物保持不变。本入口不训练、不选参、不新增正式方法×split成绩。论文状态统一见[证据总表](../doc/论文实验证据总表.md)。

## 固定设计

D1/D2按37种完整标签组合各取固定查询顺序中的首query，在普通和跨组合排名取rank1/10/100，共222对；逐例核对原始mask、实际一对一匹配、深度／结构／面积分项及候选总分。D3按预设规则选择精确成功、跨组合完整覆盖、部分覆盖、无覆盖四类案例。无覆盖案例要求完整合格图库中存在正覆盖候选，不能把图库缺失当作模型失败。

D4对全部7,199个查询精确重放未遮蔽分数；至少两个区域的查询比较高贡献、低贡献和随机单区域遮蔽。将同一查询区域对全部候选的评分行置零，保留原查询sqrt(area)分母，重新匹配与排序。随机策略枚举全部单区域遮蔽并平均结果指标。面积控制先按sqrt(area)权重比不超过1.2选择最接近的一对，再依据贡献区分高低；阈值不随结果变化。普通报告P@10和Coverage-nDCG@10，跨组合报告Coverage-nDCG@10，保留无定义值和有效query数。

## 执行与核验

在服务器原PyTorch环境和仓库根目录执行，`--source-root`替换为本轮C的实际目录：

```bash
python -u -m paper_d_rotation.run --stage compute --source-root result1124/<C实际运行目录> --workers 8
python -u -m paper_d_rotation.run --stage render --output-root result1124/<D实际运行目录>
python -u -m paper_d_rotation.run --stage seal --output-root result1124/<D实际运行目录>
python -u -m paper_d_rotation.run --stage verify --output-root result1124/<D实际运行目录>
```

compute默认以真实启动时间创建`result1124/YYYYMMDDTHHMMSS+0800__paper_d_rotation/`。使用`--check-inputs`只核查输入，不计算。输入冻结包含源C主方法分数与排名、旋转模型及独立特征路径、原始区域顺序和mask；多进程显式打开旋转特征文件，不读取旧h512。完整未遮蔽分数要求逐query精确重放；实际匹配和0.34／0.51／0.15分项使用独立标量与离线核验。

render后必须实际检查生成图像，并在根`figure_review.json`记录`status=verified`和`reviewed_images`相对路径到SHA-256的映射，再执行seal。科学代码先冻结，绘图代码另记哈希；已完成输出拒绝覆盖。运行错误保留失败产物，新科学设置另建运行。

## 输出与解释边界

`explanations/`保存222对真实匹配、得分分量和原图／mask；`figures/`保存组合图、四个单例及完整样例图集；`d4/`保存逐查询基线、逐区域遮蔽、策略和汇总。根目录保存冻结计划、来源、源码身份、视觉检查、验收及完成标记。用户指定的`result1124/`作为本轮独立输出根。

数值复原表明解释与实际评分一致；遮蔽分析的是评分证据敏感性，不表示区域语义真值、物理因果、工艺根因或用户研究结果。普通和跨组合各自使用原rank1参考，不混为同一候选。
