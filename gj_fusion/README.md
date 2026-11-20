# 整图 G 与 J 融合验证

独立比较整图深度分数G与联合结构分数J的加权加法和加权乘法。复用冻结WaPIRL h512 best_nce及整图J描述子，不训练、不加入F或角向A、不校准分数。研究在观察历史test后提出，不自动替换主方法。公共协议及结果见[项目说明](../doc/项目说明.md)，详细结论更新到[消融实验](../doc/消融实验.md)。

## 冻结设计

G为截断到[0,1]的整图特征余弦，J=exp(-2D_J)，D_J为原128维联合软直方图的普通Hellinger距离。统一α∈{0,0.25,0.5,0.75,1}，比较αG+(1-α)J与G^α J^(1-α)。两个端点共享；几何α=0.5以历史G×J原算术表示，平方根只改变分数、不改变数学排序。共8项独特配置，不将重复端点或等价角色当作独立实验。

valid固定500个query、完整3599图库，排除自身；test全部7199个query、完整7199图库，排除自身。所有候选先评分再保存Top-500，exact/coverage使用同一排名；K=1/5/10/20/50/100/500，分母与IDCG来自完整候选库。

按valid总体exact P@10整数命中数冻结全局胜者与两族胜者。同分依次优先历史product、J_only、G_only，再按变体名排序。test只评价这三个胜者与G/J/product共同对照的去重集合，最多5项；不按test重选。两族都包含G/J端点，因此也允许选中无需融合的方案。历史J_only保存exp(-D_J)，本轮保存exp(-2D_J)，重放逐query Top-500核验排名等价，不宣称分数相同。

## 运行与核验

状态：实现与验证中，正式结果尚未确认完成。

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
