# 区域结构等权加法的test评价

本入口对[valid评分搜索](../f_region_score_search/README.md)已经冻结的唯一胜者执行完整test，不再搜索权重、不训练模型。**当前为实现与输入核验阶段，尚无本轮正式test结果。** 本轮延续在观察此前实验后提出的事后扩展，不自动替换项目主方法。

公共数据、评价和登记规则见[项目说明](../doc/项目说明.md)与[结果登记说明](../results_catalog/README.md)。本包只维护本次追加test的来源、参数、运行方式与完成证据。

## 唯一固定配置

valid来源为 `artificial/f_region_score_search_20261120T140544+0800/`。该轮完整15项搜索于 `2026-11-20T14:16:26+08:00` 冻结胜者 `mixed_a0p5_e1p0`：α＝0.5、η＝1、无零余弦门控、`all_points`联合J、结构距离系数0.5，保留相对面积项。valid总体exact P@10为2585/5000＝51.70%；本次只评价该配置。

固定区域对评分为 **S＝B×(0.5C＋0.5T)**，其中C为截断到[0,1]的冻结CNN余弦，T＝exp(−0.5H)，H为128维区域联合J的普通Hellinger距离，B＝0.85＋0.15exp(−D_area/1.5)。J由整张原图的缺陷点投票一次后，按原始cc8区域池化全部点，不作硬去噪。缺少结构时沿用C×B回退；本次使用有完整分布的all_points缓存。

沿用cc8、最小面积5、Top-4、冻结best_nce的h512、单向贪心一对一匹配及查询区域sqrt(area)聚合。面积、径向及结构来自原始区域；不改变区域顺序、裁切、分割、编码器、匹配或聚合。本轮没有训练步骤，但评分使用已训练的CNN，不能称为无训练方法。

## 来源与冻结链

新test同时绑定两条来源：上述valid运行的完成标记、签名manifest、选择、完整选择索引及核验报告；以及 `artificial/f_region_structure_20261120T120331+0800/cache/test/` 的已完成区域特征缓存。旧运行及缓存保持只读，不向其中增加或覆盖结果。

valid选择必须通过全15项整数Top-10命中数与预设并列规则复核，并且胜者与全部固定参数严格一致；不能从命令行换成其他候选。旧test缓存签名仍使用原区域结构运行的context与split=test，不能用新valid运行的context代替。缓存、上游完整区域标签、h512、模型、原始数据与冻结协议均需通过哈希和原始ID对应关系核验。

复用已完成的test缓存，不提取新的CNN特征、不重新生成区域J。原始NPZ和冻结清单可用于协议完整性检查及读取test标签。旧test评分或报告不参与权重、checkpoint或方法选择；完成后可在同一协议下作结果比较。本次授权不会改写上一轮“仅valid搜索、未追加test”的历史记录，而是通过独立运行关联该选择。

## 运行与恢复

在仓库根目录、已有PyTorch环境执行。评分使用CPU，`--workers`控制排名分片并行，允许1—8，默认4；报告与全量复核在主进程执行。没有训练、参数网格或其他split入口。

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s f_region_score_test/tests -v
python -m f_region_score_test.run --dry-run
python -m f_region_score_test.run --check-inputs
python -u -m f_region_score_test.run --workers 4
```

`--dry-run`只显示固定设置，不读取正式输入或创建输出；`--check-inputs`读取并核验完整来源链，不执行test评分。默认 `--source-run` 为上述已完成的valid搜索目录，不能用旧区域结构目录代替。

后台入口为 `bash f_region_score_test/launch_tmux.sh`，自动创建 `artificial/f_region_score_test_<实际时间>/`，日志在本包 `launch_logs/`。需要恢复时只指定同一test运行根：

```bash
python -u -m f_region_score_test.run --workers 4 \
  --resume-root artificial/f_region_score_test_原启动时间
```

新运行拒绝已有输出目录；恢复需保持同一有效选择、模型、协议、缓存、科学代码及数值环境，沿用原运行身份。已完成的同签名结果经哈希验证后直接复用。运行中不修改科学代码；方法改变须新建独立实验。

## 评价与核验

完整test为7199张图逐一查询，每次使用排除自身后的7198张候选，共51,818,402个query-candidate评分。输出每个query的Top-500，共3,599,500行；不把该文件称为保存了全部候选排名。

exact与coverage使用同一排名，K＝1/5/10/20/50/100/500；报告总体、类别宏平均、单／混合、标签数量与逐类指标。AP/Recall分母和coverage IDCG使用完整固定候选库，同分排序沿用seed=2026的SHA规则。

执行前需通过来源与协议检查，以及合成数据上的评分、排名、双指标和完成标记测试。正式核验包含独立区域评分公式与匹配抽查、全query完整候选生产评分重放、独立同分排序和所有评价报告重算；独立公式抽查与全量生产评分重放分开记录。正式结果以根完成标记及全部引用文件校验通过为准。

新运行使用独立的时间目录，保存真实Asia/Shanghai启动身份、冻结valid选择绑定、来源哈希、评价报告、核验与历史保护记录。科学设置或来源变化须另建运行，不能覆盖历史。完成后仅登记这一项test，并检查既有登记保持不变；新增参数或其他候选的test不属于本入口。
