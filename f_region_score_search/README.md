# 区域结构评分的valid加法对照

本实验固定已有区域J与CNN特征，只比较区域对评分的乘法、加法及其混合。**当前处于实现与合成验证阶段，尚无本轮正式valid成绩。** 不重训、不新增test评价，也不依据既有test成绩重新选择本轮权重。本轮在观察此前实验后提出，属于事后扩展，不自动替换项目主方法。

公共数据、评价和登记约定见[项目说明](../doc/项目说明.md)与[结果登记说明](../results_catalog/README.md)。本轮是[区域结构实验](../f_region_structure/README.md)的独立评分对照，旧实现及产物保持只读。

## 固定输入与15项评分

默认源为 `artificial/f_region_structure_20261120T120331+0800/`，仅复用已冻结的valid胜者 `replace_radial_all_points_l0p5` 与 `cache/valid/`。沿用cc8、最小面积5、Top-4、h512、原始区域全部缺陷点的128维联合描述、原贪心一对一匹配及查询区域平方根面积权重。这里的乘法基线是上一轮胜者，**不是未加入区域J的原F**。

记C为截断到[0,1]的CNN余弦，H为区域J的普通Hellinger距离，T＝exp(−0.5H)，B＝0.85＋0.15exp(−D_area/1.5)。J仍包含晶圆径向与stick/ball联合分布，不新增独立径向项；所有配置保留相同面积因子B。

| 对照 | 区域对评分 | 固定参数 | 数量 |
| --- | --- | --- | ---: |
| 乘法基线 | C×T×B | 复用旧评分运算顺序 | 1 |
| 加法／乘法混合 | B×[(1−η)CT＋η(αC＋(1−α)T)] | α＝0.25/0.5/0.75，η＝0.25/0.5/1 | 9 |
| 纯加法零余弦门控 | C>0时取B×[αC＋(1−α)T]，否则0 | η＝1，α＝0.25/0.5/0.75 | 3 |
| CNN端点 | C×B | 不含结构T | 1 |
| 区域结构端点 | T×B | 不含CNN余弦C | 1 |

α控制加法内部CNN与结构的权重，η控制从乘法向加法过渡；η＝1即纯加法。门控只检验C恰为零时是否仍允许结构分数形成匹配，不引入新的可调阈值。结构缺失时退回C×B；结构端点缺失时为0。实际all_points区域应有非空分布，空晶圆仍保留在完整候选库中并得分为0。

本轮T固定使用exp(−0.5H)，与上一轮向量拼接采用的sqrt(J)点积不同。由此比较相同结构核、相同面积项下的评分方式；不混用上一轮拼接成绩代替本轮加法结果。15项是有限预设网格，不代表连续参数空间的最优值。

## 来源与数据边界

加载时核验旧run_manifest、selection、完整valid选择索引、唯一旧valid胜者的完成／核验凭据及valid缓存，确认模型、原始ID、区域顺序、面积、描述子与来源哈希一致。不得把旧运行根目录当作新缓存来源，或从其test报告／特征中构建本轮分数。

成熟来源加载器会读取完整原始NPZ的arr_0/arr_1，复核冻结去重与划分；原始数据、四份query/gallery清单及协议代码也有哈希绑定。这些是协议完整性检查，**不提取新的test特征，不计算test排名，不将test结果用于评价或选参**。本轮评价显式仅取valid标签和已完成的valid特征。历史保护允许对旧test产物做文件大小、mtime及必要字节哈希校验，以确认未覆盖；不解析其分数或特征参与本轮研究。

固定valid为500个query、完整3599张图库，每query排除自身后3598个候选。exact与coverage共享同一Top-500，沿用K＝1/5/10/20/50/100/500及全部总体、宏平均、单／混合、标签数量和逐类报告；分母与coverage IDCG取完整候选库。

## 运行、选优与核验

在仓库根目录、已有PyTorch环境执行；评分使用CPU，`--workers`控制并行报告与核验，范围1—8。

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s f_region_score_search/tests -v
python -m f_region_score_search.run --dry-run
python -m f_region_score_search.run --check-inputs
python -u -m f_region_score_search.run --workers 4
```

后台入口为 `bash f_region_score_search/launch_tmux.sh`。新运行自动建立 `artificial/f_region_score_search_<实际时间>/`，日志位于本包 `launch_logs/`；源实验不写入。用 `--source-run`显式指定同身份的来源目录。恢复时使用：

```bash
python -u -m f_region_score_search.run --workers 4 \
  --resume-root artificial/f_region_score_search_原启动时间
```

入口没有训练或test阶段。已完成结果按哈希复用；来源、科学代码或数值环境变化须新建运行目录，不能覆盖旧结果。

选优只使用valid总体exact P@10，以500个query的Top-10整数命中总数比较全部15项。并列优先乘法基线，再依次选择较小η、α更接近0.5、无门控、字典序；两个端点在同命中时排在混合配置之后。全部配置完成核验后冻结 `selection.json`，结果只表示本次valid搜索的胜者，不自动追加test或替换上轮冻结方案。

核验包括独立区域评分公式抽查、原匹配聚合、全候选生产评分重放、独立SHA同分排序，以及逐query和全部汇总的双指标重算。`product`的250,000行valid Top-500必须与旧胜者排名文件逐字节一致。核验报告明确区分独立公式抽查与全量生产分数重放。

运行目录保留 `comparison_valid.csv`、冻结选择、各配置的报告／排名／完成标记、真实Asia/Shanghai时间身份、输入绑定与历史保护记录。正式完成以通过校验的根 `completed.json` 为准，进程退出或部分CSV存在不代表成功。完成后用本包登记器登记15项valid，并运行官方结果collect/check；不登记合成测试或未完成结果。
