# 整图 G 与 J 融合验证

独立比较整图深度分数G与联合结构分数J的加权加法和加权乘法。复用冻结WaPIRL h512 best_nce及整图J描述子，本轮无新增训练，不加入F或角向A、不校准分数。G仍来自已训练的自监督模型。研究在观察历史test后提出，不自动替换主方法。公共协议及结果见[项目说明](../doc/项目说明.md)，详细结论见[消融实验](../doc/消融实验.md)。

## 冻结设计

G为截断到[0,1]的整图特征余弦，J=exp(-2D_J)，D_J为原128维联合软直方图的普通Hellinger距离。统一α∈{0,0.25,0.5,0.75,1}，比较αG+(1-α)J与G^α J^(1-α)。两个端点共享；几何α=0.5以历史G×J原算术表示，平方根只改变分数、不改变数学排序。共8项独特配置，不将重复端点或等价角色当作独立实验。

valid固定500个query、完整3599图库，排除自身；test全部7199个query、完整7199图库，排除自身。所有候选先评分再保存Top-500，exact/coverage使用同一排名；K=1/5/10/20/50/100/500，分母与IDCG来自完整候选库。

按valid总体exact P@10整数命中数冻结全局胜者与两族胜者。同分依次优先历史product、J_only、G_only，再按变体名排序。test只评价这三个胜者与G/J/product共同对照的去重集合，最多5项；不按test重选。两族都包含G/J端点，因此也允许选中无需融合的方案。历史J_only保存exp(-D_J)，本轮保存exp(-2D_J)，重放逐query Top-500核验排名等价，不宣称分数相同。

## 运行与核验

状态：正式valid 8项、冻结test 4项及配对分析已完成；本地与服务器72项测试通过。valid选择等权加法，几何族选择历史product；不按test改选。test等权加法P@10/mAP@100/Coverage-nDCG@10为64.8076%/16.0857%/90.0711%。相对product的差值分别为−0.0667/+0.0845/−0.0541个百分点，逐项95%配对区间分别为[−0.1903,+0.0528]/[+0.0660,+0.1025]/[−0.0919,−0.0175]：前排精度未见提升证据，mAP微升而coverage微降，不能称为整体提升或证明等效。

原始来源及封存目录：

- valid：`artificial/gj_fusion_valid_20261120T181856+0800/`，实际启动18:18:56，选择冻结18:22:22。
- test：`artificial/gj_fusion_test_20261120T182234+0800/`，实际启动18:22:35（目录分配早1秒，以manifest为准）。
- 配对分析：`artificial/gj_fusion_analysis_20261120T190052+0800/`，实际启动19:00:52，完成19:01:04；以上均为2026-11-20、Asia/Shanghai。

完整产物保留在服务器同名相对路径。本地同步轻量报告，不含完整排名；正式封存检查在持有完整产物的服务器完成。G/J/product历史Top-500回放覆盖valid全部500及test全部7199个query，排名一致；G/product分数差为0，J仅验证单调变换后的排名等价。全部新排名按完整候选独立核验，历史来源保护检查通过。结果与分组详见[消融实验](../doc/消融实验.md)。

在仓库根目录、原PyTorch CUDA环境执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s gj_fusion/tests -v
python -m gj_fusion.run --stage valid --check-inputs
python -u -m gj_fusion.run --stage all --device cuda
```

`all`先封存valid及选择，再打开test评分输入，两阶段各写独立的`artificial/gj_fusion_<split>_<真实启动时间>/`。也可分别执行`--stage valid`及`--stage test --valid-run <本轮valid目录>`；断点重试用同阶段`--resume-root`，test仍须指定原valid目录。源码、参数或来源变化必须新开目录。

每次评价保留真实Asia/Shanghai run_id及started_at。来源绑定涵盖原始数据、冻结协议、模型身份、特征、描述子及历史对照；旧来源只读。每类固定1个query×最多20候选独立核对G点积和标量Hellinger距离；每种配方全部query独立重算融合公式、完整候选排序和双指标/分组。重放原G、J、G×J排名，并记录浮点分数差异。全库评分与完整排名不同时驻留内存。

以根`completed.json`及全部绑定文件校验确认完成；进程退出或部分CSV不能代表成功。登记用：

```bash
python -m gj_fusion.register --run-root <本轮valid或test目录> --apply
python results_catalog/build.py check
```

共享登记仅新增本轮结果；G/J/product端点为数值回放对照，不是独立训练重复。后续统计分析写独立时间目录，以query原始ID配对，区间条件于固定模型、图库与标签组合构成。

完成两阶段后运行预设配对分析：

```bash
python -m gj_fusion.analysis \
  --valid-root <本轮valid目录> --test-root <本轮test目录> \
  --wbbs-exact-root ../compared_methods/artifacts/formal_exact_20261109_120738/evaluation/wbbs_test_r2/test \
  --wbbs-coverage-root ../compared_methods/artifacts/coverage_v1_test_20261109_191909/wbbs/test
```

分析核验WBBS的exact与coverage引用同一排名，并将原数据、冻结query/gallery清单及完整候选身份与本轮来源对齐。按37种完整标签组合进行2,000次配对bootstrap（seed=2026），保持每类query数量，各指标/对比共用抽样。输出逐项95%百分位区间，未校正多重比较、既往test探索、训练种子或图库变化；区间跨零不证明等效。加法系数相同不代表两分支影响相同，原始G/J分数没有做尺度校准；这组网格不能排除其他融合设计。
