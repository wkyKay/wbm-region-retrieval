# B1：形态与径向位置联合分布

本目录包含这一设计的代码、测试和说明。它只读使用原 F/WaPIRL 模型与特征，不训练模型。当前为**已实现、待用户启动正式评价**；本地合成测试不代表正式检索效果。

## 固定设计

基线 `B0=(0.5*F_optimal+0.5*G)*exp(-0.5*D_angle)`，完整复用原 cc8、Top-4、h512、径向面积评分、query sqrt(area) 权重和外圈角向项。B1 为 `B0*exp(-lambda*D_H)`。

完整原图的 `raw==2` 点参与张量投票，`raw==1/2` 定义支撑；固定 sigma_d=9 pixel²、单次投票、排除自身、阈值0.32。stick/ball 分别除以**去噪前**通道最大值，全零通道置零；仅两通道同时严格低于阈值才删点。沿用原支撑框中心与最大支撑半径计算rho。

将保留点的 `[rho,stick,ball]` 等权累计为8×4×4联合软直方图。B个节点为k/(B−1)，权重 `max(1-abs((B-1)*x-k),0)`，三轴权重相乘；总质量归一为1。普通Hellinger距离为 `sqrt(0.5*sum((sqrt(P)-sqrt(Q))**2))`。双空距离0，单空距离1，非法支撑报错。

该分支源于WBBS的结构表示，但不使用BBP匹配或点数惩罚；sigma_d等是现有复现约定。它舍弃绝对点数/绝对显著性，不能保证不同图案必有不同描述，也不宣称任意缩放、像素重采样旋转不变。既有角向和深度分支继续提供互补信息。

| 家族 | 新描述 | 目的 |
|---|---|---|
| B0 | 无 | 固定最佳组合 |
| R | rho边缘分布，8维 | 额外径向信息 |
| T | stick/ball边缘分布，16维 | 局部形态统计 |
| I | rho边缘×stick/ball边缘，128维 | 删除形态与位置关联的同维对照 |
| J | 原联合分布，128维 | **预先指定的B1主候选** |
| J_only | exp(-D_J) | 结构分支单独检索 |

R/T/I/J各自比较lambda=0/0.25/0.5/1/2，按valid总体exact P@10选择，并列取较小值。四个零端点都引用B0，因此valid共18套评分配置。不得因为控制组成绩较高就把它重新命名为J。test只评价冻结选择和B0/J_only，无test搜索。

固定valid为500query×3598候选，test为7199×7198。每个query对完整池打分后保存Top500，exact/coverage共用排名，AP/Recall分母与coverage IDCG使用完整池。保留K=1/5/10/20/50/100/500及总体、类别宏平均、单混合、标签数分组。既有test参与过研究设计，记录posthoc，不包装为从未接触test。

## 运行

在服务器拉取代码并激活**原来的PyTorch CUDA环境**，不要为B1替换已有训练环境：

```bash
cd /data/wangkaiyan2024/wbm-region-retrieval
git pull --ff-only origin main
python -m unittest discover -s b1_structure_fusion/tests -v
python b1_structure_fusion/run.py --stage all --device cuda --check-inputs
bash b1_structure_fusion/launch_tmux.sh all
```

`all` 是显式请求先valid、冻结选择、再test。若主候选J选中零强度，则完成valid后停止，不自动进入test。默认不传stage时仅valid。运行入口无需GPU编码，但 `--device cuda` 确实使用GPU计算float64结构距离；张量投票准备和冻结B0仍在CPU上执行，以保留原公式和精确零强度回退。CPU/CUDA仅承诺数值容差一致，后端写入科学签名，valid/test必须相同。

分两次执行可用：

```bash
bash b1_structure_fusion/launch_tmux.sh valid
# 等valid完成，将下行替换为真实输出路径
bash b1_structure_fusion/launch_tmux.sh test --selection-file artificial/b1_structure_valid_实际时间/selection.json
```

launcher自动创建 `artificial/b1_structure_<stage>_<真实启动时间>/`，打印会话名和日志路径。输出同时显示在tmux并保存至 `launch.log`，另有 `run.log`、`status.json`；每套结果完成即打印exact/coverage全部K与WBBS差距。数值为百分数，差值为 **本实验−WBBS的百分点**，负值表示落后。历史WBBS valid仅有exact，因此WBBS的valid coverage参考值及差值显示NA；B1自身的coverage照常评价，绝不用test数据补充。

CLI支持 `--dry-run`（不加载数据、不写文件）、`--check-inputs`（只核验来源）、`--prepare-workers`、`--device cpu|cuda`，以及旧模型和特征的显式路径。test必须提供已完成valid的selection；不提供test强度覆盖参数。缺少旧特征时明确失败，不静默重提取。

中断后在同一环境、同一源码与设置下使用 `--resume-root`，不能重复指定新output-root。例如：

```bash
python -u b1_structure_fusion/run.py --stage all --device cuda --prepare-workers 4 \
  --resume-root artificial/b1_structure_all_实际时间
```

已完成配置按哈希复用；未完成排名从头重算，失败attempt保留。启动身份保留，科学设置变化必须新建输出目录。运行耗时分准备、共享评分排序、指标及验证记录，不能把共享批次耗时误算成每个变体的独立耗时。

## 输出、验证与登记

输出目录包括计划、来源manifest、按split隔离的描述子缓存、各variant排名与双指标、真实时间身份、冻结selection、`comparison_valid.csv`/`comparison_test.csv`及合并的`comparison.csv`。CSV存0～1比例和比例差，控制台乘100展示。WBBS只读使用唯一结果登记器的同split普通协议快照，存在原报告时再次校验报告哈希。

`verification_<split>.json`保存独立复核：候选身份、同分规则、保存分数公式、全部query各K的exact/coverage以及总体均值。数学单测另覆盖直接矩阵投票、解析特征值、手工软分箱、关联互换反例、空集与旋转。正式结果应先看J对B0、I及J_only的差值，再看相对WBBS；valid选参结果不作为独立泛化或跨训练seed证据。

2026-11-17本地验证：B1的26项测试中25项通过，CUDA一致性测试因本机无GPU跳过；原融合、分割、几何及登记器的54项相关回归测试通过。现有57条结果与7个生成文件校验通过；未启动正式valid/test。上述服务器测试命令会在CUDA可用时执行该GPU测试，`--check-inputs`另行核验服务器上的真实模型和特征。

完成后在输出目录生成 `catalog_sources.json`，尚不自动写共享登记器。检查结果后将该次正式结果登记到唯一目录：

```bash
python b1_structure_fusion/register.py --run-root artificial/b1_structure_all_实际时间 --apply
python results_catalog/build.py check
```

登记命令先校验完整产物，在内存完成采集/渲染验证后才更新共享登记；旧登记备份到本次输出中。按项目规则同步新生成的轻量登记文件，原模型、排名和大缓存不加入Git。实验尚未运行时不会登记不存在的成绩。

## 独立删除边界

全部B1专属代码、测试和本说明均位于本目录；旧训练、分割、评分源码不依赖B1。未登记的试验可删除本目录与自己的 `artificial/b1_structure_*_时间/`，不会影响原F或WBBS。共享改动只有通用`artificial/`忽略规则与真实时间登记支持，均不导入B1。

若结果已经正式登记，删除代码不影响读取历史报告；若还要删除已登记产物，应先按项目历史保护约定处理其来源记录，不能直接破坏登记引用。旧模型和旧运行目录始终保留原位。
