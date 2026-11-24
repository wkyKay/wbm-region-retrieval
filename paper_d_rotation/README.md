# 旋转编码器的区域评分解释与遮蔽验证

本入口完成当前主方法的D1–D4，复用任意角旋转CNN固定第100轮的原视图特征，评分为`0.85*(0.4*C+0.6*T)+0.15*A`。来源必须是已封存的本轮`paper_c_rotation`运行；历史`paper_d_explanation`及旧产物保持不变。本入口不训练、不选参、不新增正式方法×split成绩。论文状态统一见[证据总表](../doc/论文实验证据总表.md)。

## 已完成运行

正式运行保存在[`result1124/20261124T195710+0800__paper_d_rotation/`](../result1124/20261124T195710+0800__paper_d_rotation/)，实际启动于`2026-11-24T19:57:10+08:00`，已封存为`completed`，并于`2026-11-24T20:21:11+08:00`通过独立`verified`核验。来源为本轮[`20261124T192819+0800__paper_c_rotation`](../result1124/20261124T192819+0800__paper_c_rotation/)的主方法结果。当前证据已完成；本轮论文`e83bfb8`已将当前四案例图、贡献复原与遮蔽表写入中英文§V-D，替换工作稿中的旧方法图表。完整222例图集及逐查询明细仍保留项目资料，历史产物不改。

已核验222对解释（两协议各111对）、870对区域匹配及244张原图／mask，最大代数复原误差为`3.33e-16`。遮蔽验证精确重放全部7,199个查询的完整候选分数；两协议各有7,161个多区域查询参加主分析、5,920个查询参加面积控制，另38个单区域查询保留基线。共核验56,958条单区域遮蔽记录、104,648条策略记录及112条汇总记录。完整结果和解释边界见[汇总](../result1124/20261124T195710+0800__paper_d_rotation/summary.json)、[独立核验](../result1124/20261124T195710+0800__paper_d_rotation/verification.json)及[完成标记](../result1124/20261124T195710+0800__paper_d_rotation/completed.json)，论文结论维护于[消融实验](../doc/消融实验.md)。

最终图像包含[四类案例主图](../result1124/20261124T195710+0800__paper_d_rotation/figures/main_cases.pdf)、[四张独立案例](../result1124/20261124T195710+0800__paper_d_rotation/figures/individual/)及[37页完整图集](../result1124/20261124T195710+0800__paper_d_rotation/figures/gallery/all_222_anchor_examples.pdf)，共42个PNG、6个PDF；48个文件的实际视觉审阅范围和SHA记录于[figure_review.json](../result1124/20261124T195710+0800__paper_d_rotation/figure_review.json)。图集全部页已通过PNG／PDF联系表检查，12个标签位置变化页另作整页细查；不将联系表检查表述为对全部小字逐字审查。

两轮排版修复仅影响显示：先增加主图和单例的列间距，避免长分项说明压入候选图；再检测区域编号碰撞，保留小区域编号，将冲突的大区域编号移至自身原始mask内最近的可用像素。原始mask、科学数值、真实区域中心及对应连线端点均未改变；[244张资产的布局审计](../result1124/20261124T195710+0800__paper_d_rotation/label_layout_audit.json)保存了验证范围。前两轮图像、旧绘图源码和未通过说明保留于[`render_history/attempt_1/`](../result1124/20261124T195710+0800__paper_d_rotation/render_history/attempt_1/)和[`attempt_2/`](../result1124/20261124T195710+0800__paper_d_rotation/render_history/attempt_2/)。

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
