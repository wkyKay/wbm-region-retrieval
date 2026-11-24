# 旋转区域编码器与新权重的受控消融

本入口复用固定第100轮任意角旋转CNN及原始视图特征，执行当前主方法的C1–C5；不重训、不重新选权重。公共状态维护在[论文实验证据总表](../doc/论文实验证据总表.md)，协议及来源规则见[项目说明](../doc/项目说明.md)。历史`paper_c_ablation`源码与产物保持不变。

## 固定设计

| 变体 | 区域对评分或唯一改变 |
|---|---|
| primary | `0.85*(0.4*C+0.6*T)+0.15*A` |
| no_c | `(0.51*T+0.15*A)/0.66` |
| no_t | `(0.34*C+0.15*A)/0.49` |
| no_a | `0.4*C+0.6*T` |
| independent_j | 用128维`p(rho)*p(stick,ball)`代替联合分布，保留形态两轴内部关系 |
| reuse_candidate | 每个查询区域独立取最佳候选区域，允许候选区域复用 |

固定cc8、min_area=5、Top-4、h512、seed=2026和查询sqrt(area)完整分母；除最后一项外均保留单向贪心一对一。几何来自原图区域mask，CNN取`20261123T211208+0800__cnn_augmentation`的rotation original.npy；模型、特征哈希和区域行序均绑定。主配方沿用G8原运算顺序，不换成可能改变浮点近并列的等价展开式。

## 执行与验证

服务器原PyTorch环境、仓库根目录：

```bash
python -u -m paper_c_rotation.run --stage all --workers 8
```

默认创建`result1124/<真实启动时间>__paper_c_rotation/`，也可显式`--output-root result1124/<时间>__paper_c_rotation`。`--dry-run`打印计划；`--check-inputs`只验证来源。科学签名相同可用`--resume-root <原目录>`恢复；未封存的残留缓存拒绝覆盖，改变科学设置须新建运行。

六个变体全部执行valid、普通test、跨组合test，共18份报告。valid仅为实现／协议核验门槛，不能按成绩筛除变体。普通和跨组合使用各自完整合格候选库，同一普通排名分别评价exact/coverage；保留K=1/5/10/20/50/100/500、逐查询、逐类和分组报告。

primary的三份报告与G8新主配方要求Top-500候选、顺序与得分精确一致，并逐块对比valid/test全部float64分数矩阵。三个主方法端点沿用原登记身份，其余15项新增登记。每项的完整候选排序和指标独立复算；区域公式另按37类固定查询×20候选独立抽查，再用生产区域对矩阵独立核验匹配，避免float32余弦近并列导致伪差异。

完成后使用`python -m paper_c_rotation.register --run-root <运行目录> --apply`，再执行项目规定的正式collect/check及归档核验。完整分数矩阵、Top-500排名、轻量报告、冻结计划、来源哈希和独立核验均保存在该独立运行目录。完成状态以`completed.json`及其引用文件验收为准，README不替代实际完成证明。
