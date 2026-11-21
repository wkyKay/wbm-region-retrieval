# 区域J＋F的C组受控消融

论文证据进度统一维护在[论文实验证据总表](../doc/论文实验证据总表.md)，test结果与结论见[消融实验](../doc/消融实验.md)，公共协议和登记流程见[项目说明](../doc/项目说明.md)。本入口仅补充C1–C5；不训练模型，不重新选择主方法，不执行解释遮蔽或效率实验。

## 固定配方与对照

主配方按历史实现的运算顺序计算 `0.75*(0.4*C+0.6*T)+0.25*A`。`C=clip(cos(hq,hc),0,1)`，`T=exp(-0.5*Hellinger(Jq,Jc))`，`A=exp(-0.5*abs(log(rq/rc))/1.5)`；相对面积r由原始区域和有效晶圆面积计算。J为全图张量投票后按原始区域全部缺陷点汇总的128维8×4×4软直方图。

| 变体 | 改动 | 对应证据 |
|---|---|---|
| `primary` | 完整主配方，作为历史重放对照 | B1/B2及C组参照 |
| `no_c` | `(0.45*T+0.25*A)/0.70` | C1 |
| `no_t` | `(0.30*C+0.25*A)/0.55` | C2 |
| `no_a` | `0.4*C+0.6*T` | C3 |
| `independent_j` | 用`p(rho)*p(stick,ball)`替换J，保留128维及stick/ball内部关系 | C4 |
| `reuse_candidate` | 每个查询区域独立取最佳候选区域，允许候选区域重复使用 | C5 |

固定cc8、min_area=5、Top-4、三通道窗口、冻结best_nce h512及seed=2026。除C5外均采用原单向贪心一对一；所有变体使用原查询sqrt(area)权重与完整查询区域分母。C3并未删除面积聚合权重，C5不是最优一对一对照。删除分项后按剩余权重和归一化，其他参数保持不变。

## 执行与核验

在持有完整历史产物的服务器、原PyTorch环境和仓库根目录执行：

```bash
python -u -m paper_c_ablation.run --stage all --workers 8
```

默认创建`results/<实际启动时间>__paper_c_ablation/`；整个`/results/`已加入`.gitignore`。`--dry-run`只列计划，`--check-inputs`只核查来源；`--stage valid`完成valid后停止；同一科学签名下可用`--resume-root results/<原目录名> --stage test`继续。失败留下的未封存阶段不得当作完成结果复用，改变实现必须建立新运行。

运行开始即签名封存六变体×valid/普通test/跨组合test的全部18项计划。valid仅作实现和协议核验门槛，不按表现筛选；全部预设变体均评价两种test。原主配方的选择仍来自29项valid exact P@10；本组是事后受控消融，不将研究历史表述为从未接触test。

每个阶段保存独立真实时间身份、Top-500排名、各K总体／类别／单混合／标签数量指标、逐query与逐类报告、验证记录及完成标记。AP/Recall/IDCG基于完整合格候选库。跨组合先从完整分数矩阵排除相同完整标签组合，再重新排名；只报告coverage。

完整缓存的所有查询排序与指标均独立核验；另对每类最小query ID和最小20个合格候选抽查六种区域公式及匹配。浮点分项公式容差2e-6，使用已核查区域对矩阵的独立匹配容差1e-12；后者避免float32原始余弦近并列差异，并非所有区域对的独立重算。四项历史控制（主方法valid/普通test/跨组合test、C3 valid）要求Top-500候选ID、顺序及得分完全相同，关联原key，不重复登记。因此18份报告新增14项正式登记。

运行用时只用于追溯，不能当作E区部署准备成本或单查询延迟测量。大规模完整缓存与排名保存服务器，Git只管理代码、文档、来源清单及汇总。

## 本轮产物

本轮目录为`results/20261121T175924+0800__paper_c_ablation/`，根manifest实际启动时间为2026-11-21 17:59:44（Asia/Shanghai）；目录时间17:59:24是进程启动时间，不代替manifest中的开始时间。18份报告全部完成核验，四项历史控制精确重放，新增14项正式登记。

服务器绝对路径：`/data/wangkaiyan2024/wbm-region-retrieval/results/20261121T175924+0800__paper_c_ablation/`。本地[同名目录](../results/20261121T175924+0800__paper_c_ablation/)保存轻量报告与元数据；完整分数矩阵和Top-500排名文件保留服务器。主要入口为`comparison_test.csv`、`comparison_cross_test.csv`、`test/<变体>/`、`cross_test/<变体>/`；`score_cache/test/<变体>/scores.npy`为完整评分矩阵，`rankings.csv`为Top-500排名。完成状态见`completed.json`和各评价的`verification.json`，来源、原选择与重放证明见`run_manifest.json`、`frozen_plan.json`及`replay_*.json`。
