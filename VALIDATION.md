# 当前验证记录：独立 A～D 消融（2026-10-19）

新增入口与设计见 [ABLATION_ABCD.md](ABLATION_ABCD.md)。既有主实验实现和配置没有修改。

本地 CPU 上完整 `python -m unittest discover -s tests -v` **52 项通过**（49.012 秒）；随后补充完成态报告恢复逻辑，再次执行 6 项消融测试全部通过（16.809 秒）。入口 `--dry-run`、`--help` 与 `git diff --check` 通过。

- 新测试使用 24 张合成 map，train/valid/test 各 8 张，min_area=5、top4、输入 96、h512/projector128；小测试使用 batch=2、8 个负样本和 1～2 epochs，不代表正式 100 epoch 设置。
- A～D 均执行真实 SGD 参数更新、valid 检索、两种 checkpoint 评价，以及完成态 resume；初始化参数 SHA256 相同，训练后参数与初始化不同。
- 原输入与旧 `token_patch` 相同；B 与 `pad(token_patch(mask, 88, 0), 4)` 相同；C 二值、可重现、平移不截前景；D GAP 前特征图 6×6，其余组 3×3。
- 验证候选池不看标签、不含自身、只含 valid；Zernike 分数与原逐对 matcher 一致，排名指标与原 evaluator 一致。
- B 的两轮训练分别连续运行和在一轮后中断恢复，最终所有模型张量、memory bank 逐元素一致，调度器与更新次数一致。
- 模拟 `last.pt` 已提交、JSON 报告尚未写入时中断：resume 可重建报告，不改 checkpoint。
- 默认流程未产生 test 指标；额外显式测试 B 的 test 入口后，checkpoint 哈希不变，使用原冻结测试池。

完整 Mixed-WM38 的 CUDA 显存、速度、收敛和性能尚未验证，需按文档在远程服务器运行。这里的合成数据指标只证明流程可运行，不作为方法优劣结论。

---

# 历史验证记录：top4、轻度开运算与单标签迁移（2026-10-18）

本节对应 2026-10-18 主实验版本。新增测试与所有既有测试合计 **46 项通过**，耗时 10.118 秒；语法检查、正式配置 dry-run、`git diff --check` 均通过。新增检查覆盖毛边去除及前景保护、纯细线/过度删除回退、原始分支不变、关闭增强时保持原 RNG 行为、增强仅作用于局部方法、历史 checkpoint 严格结构加载、top1 与原 patch/描述子一致、单标签/同 split/排除自身、空 proposal、macro/micro 和缺失正例，以及与正式匹配/评价的一致性。推理集成测试禁止调用 SGD 和 backward。

真实 MPS 验证命令：

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python -u scripts/run_mvp.py \
  --single-label-source artifacts/mvp_mps_300_wapirl_zernike_20261017
```

MPS GPU 在允许访问 Metal 的本地进程中运行，关闭自动 CPU fallback。环境为 macOS 27.0.1 arm64，Python 3.12.14、PyTorch 2.14.1、NumPy 2.5.4、SciPy 1.18.1、Matplotlib 3.11.2。总耗时 **107.42 秒**，完整训练、检索、评价、几何示例、完成态续跑，以及历史 best.pt 的单标签推理和缓存复用均通过。

- 新 MVP 目录：`artifacts/mvp_mps_300_top4_opening_20261018`。共 300 张原始图，按原 split 取 train/valid/test=210/30/60；提取 776/117/229 个区域，最大区域数均不超过 4。
- 局部模型和三个监督预算分别真实训练 3 轮；local 更新 147 次，三个监督模型各更新 48 次。四个模型结构、初始编码器哈希相同，参数确有变化且有限，best 对应各自最低 valid loss。
- 五个方法各输出 60×59=3,540 条完整排名。完成态 `--resume` 后训练文件 SHA256 不变，没有额外优化更新。
- 开运算训练触发率分别为 51.03%、48.20%、49.74%；实际改变视图占比分别为 45.10%、42.40%、44.20%；尝试后的回退率分别为 11.62%、12.03%、11.14%。每轮整体前景像素保留比例为 97.04%、97.58%、97.29%。valid 三轮固定为 117 个视图、62 次尝试、55 次实际改变、7 次回退。
- 已人工查看 `models/local/augmentation_examples.png`，原始/开运算/组合视图顺序正确，细薄区域存在回退，图与逐行审计信息对应。增强不保证类别语义，因此正式训练后仍需检查该图和回退率。

单标签验证读取**修改前已经存在**的 `artifacts/mvp_mps_300_wapirl_zernike_20261017/models/local/best.pt`，原训练 topk=6、epoch=1；仅本次推理 topk=1。未对这个旧模型进行重训，也没有将其转成全图输入。

- 单独目录：`artifacts/mvp_mps_300_top4_opening_20261018_single_label_top1`。
- valid 单标签为 16 张，每查询 15 个候选；test 为 32 张，每查询 31 个候选。300 是整个 MVP 的输入数量，辅助测试只使用其中的单标签 valid/test 图。
- 两个方法分别产生 valid 240 行、test 992 行，候选池完全一致、无自身或跨 split，40 行汇总指标及四张检查图完整。
- 固定池及特征缓存复用后，h512、query/candidate 清单和两个方法的全部排名 SHA256 均保持不变；源旧实验的 `source_unchanged=true`。已查看 map/区域/patch/近邻图，显示顺序与标签、分数对应。
- 自动保护检查再次核对改动前的 **341 个旧 artifact 文件**路径、大小、mtime，以及原始 NPZ、冻结 CSV、旧 best/last checkpoint 共 20 项 SHA256，全部未变。没有启动正式全数据训练，也没有创建 `artifacts/experiment_a_top4_opening_seed2026`。

机器报告见新 MVP 的 `mvp_report.json`、`validation.json` 和 `run.log`；单标签报告见其独立目录的 `report.md`、`run_manifest.json`、`source_audit.json`。这些测试证明本地 MPS 全链路和历史模型迁移可运行，不用于推断正式训练效果。完整数据上的 CUDA 训练与单标签评测仍需按 README 在远程运行。

---

## 历史验证记录：WaPIRL 对齐与 Zernike（2026-10-17）

本节对应 2026-10-17 的代码和配置，保留为历史记录；数值不能作为当前版本的验证结论。

- 环境：macOS 27.0.1 arm64，Python 3.12.14，PyTorch 2.14.1，NumPy 2.5.4，SciPy 1.18.1，Matplotlib 3.11.2。
- `python -m unittest discover -s tests -v`：33 项全部通过，耗时 28.244 秒。覆盖 Zernike 解析点集的已知矩值、旋转/平移不变性、环/弧/实心圆差异、欧氏距离打分、完整形状增强、原版 stem、线性投影、共同初始编码器，以及向量化 NCE 与原版 cosine 公式的数值和梯度一致性。
- 正式配置 dry-run 和 audit 已通过：train/valid/test=25,910/3,700/7,405；2,000 个 query，每个 1,000 个合法 candidate。
- `python scripts/run_mvp.py`：使用真实 MPS GPU，关闭自动 CPU fallback，完整流程和续跑检查通过，总耗时 122.14 秒。沙箱内无法访问 Metal，实际 GPU 运行通过授权后的沙箱外 Python 进程执行，没有改为 CPU 训练。

MVP 独立目录为 `artifacts/mvp_mps_300_wapirl_zernike_20261017`，300 张图按源 split 取 210/30/60；分别产生 1,140/175/337 个 CC token。局部和三个监督预算都实际训练 3 轮，每个 checkpoint 的 loss 和参数均有限，参数哈希均不同于初始化，最佳模型均对应最低 valid loss。

| 模型 | 训练轮数 | 最佳 epoch | optimizer 更新次数 | 最佳 valid loss |
| --- | --- | --- | --- | --- |
| local | 3 | 1 | 216 | 7.892848 |
| global_5pct | 3 | 1 | 48 | 0.211720 |
| global_10pct | 3 | 1 | 48 | 0.209632 |
| global_50pct | 3 | 1 | 48 | 0.209617 |

四个模型的 `encoder_spec` 一致，初始编码器 SHA-256 均为 `433ad653274c4f5819051cbc1e665b43a3363829e76f7ce7942d91e0735532bc`。编码器为单通道 7×7 stride 2 + max-pool 的 ResNet-18，512 维 GAP 输出，无可学习压缩层；proposed 使用线性 512→128 projector 和 5,000 个有放回负样本。Zernike 为 12 阶、47 维，通过欧氏距离映射得分。三个监督预算实际使用 10/20/104 张 train 图。

五组 ranking 各有 60×59=3,540 行，共 17,700 行。候选完整性、去重、排除自身和有限分数检查通过，指标、置信区间、配对比较、区域对应和几何示例均已生成。再次 `--resume` 后，所有 best/last checkpoint、history 和 training summary 的 SHA-256 均保持不变。

下表仅记录 MVP 输出，训练很短，不用于判断正式方法优劣，也不作为远程主实验结果：

| 方法 | ContainmentNDCG@10 | mAP_full | MRR_full |
| --- | --- | --- | --- |
| handcrafted | 0.667933 | 0.573106 | 0.833970 |
| proposed | 0.562825 | 0.470283 | 0.675476 |
| global_5pct | 0.180907 | 0.243108 | 0.229638 |
| global_10pct | 0.206928 | 0.243951 | 0.251934 |
| global_50pct | 0.208606 | 0.243753 | 0.252579 |

阶段耗时：局部训练 46.42 秒，三个监督预算合计 33.90 秒，检索 5.84 秒，评价 0.46 秒，几何示例 22.25 秒，完成态续跑 6.70 秒。

机器报告 `mvp_report.json` 的 `status=passed`、`formal_inputs_unchanged=true`、`completed_resume_keeps_training_unchanged=true`。额外对照修改前快照确认原始 NPZ 和三份冻结 CSV 的哈希不变；正式配置作为本次授权改动已更新，MVP 运行期间其哈希保持不变。新的正式输出目录 `artifacts/experiment_a_wapirl_zernike_seed2026` 尚未创建，旧正式目录与旧 MVP 未用于本次测试。

本次实测验证的是 MPS 全链路与接口；完整数据集的 CUDA 训练、收敛和正式排名仍需按 README 在远程 GPU 上运行。

---

## 历史记录：旧版 256 维网络与组合手工描述子

### 交付前验证记录

本次验证使用临时 Python 3.12.14 环境及 NumPy 2.5.4、SciPy 1.18.1、PyTorch 2.14.1 和 Matplotlib 3.11.2，自动化单元与接口测试使用 CPU，另在 M4 MacBook Air、16 GB 内存的本机上用 MPS 实际完成了 300 图 MVP 的参数更新、检索、评价与续跑检查。正式完整数据集没有被用于训练，MVP 的少量训练结果也不作为正式方法性能结论。

`python -m unittest discover -s tests -v` 已通过 26 项测试，覆盖纯 8 连通域、面积过滤与 Top-K、token patch 的平移一致性、手工特征的直角旋转一致性、PCA-scale、贪心一对一约束、query 侧分母、额外 candidate 区域、非对称 containment、标签预算嵌套、训练 pair 边界、评价指标、模型前向与梯度、memory bank 负样本排除、epoch 增强变化以及 checkpoint 权重与随机状态恢复，新增测试还验证了早停计数、累计改善、最少轮数、停止状态恢复和 300 图子集的来源隔离。自动化接口测试使用临时小样本与随机 checkpoint，不调用 optimizer step，下面的真实 MPS MVP 则实际执行参数更新。

三个方法已经分别移入 `methods/proposed.py`、`methods/handcrafted.py` 和 `methods/supervised_global.py`，新增的两项测试验证统一入口会调用对应方法文件的训练函数，并且单独指定检索方法时只会调用被选中的方法。拆分后的完整测试和入口 dry-run 均已通过，训练损失、区域打分、贪心匹配规则和结果文件格式保持不变，缓存与 checkpoint 指纹已经包含对应方法文件的实现代码。

`python run_experiment.py --stage audit` 已在现有原始 NPZ 和复制后的冻结清单上通过，实际核验到 train 25,910、valid 3,700、test 7,405，固定 query 2,000，每个 query 都恰好有 1,000 个不重复的合法 test candidate，标签维数为 8，完整标签组合数量为 37。三份清单的 SHA-256 与 `protocol/README.md` 一致，数据没有重新划分，原项目文件没有被修改。

入口的 `--dry-run` 和 Python 语法检查已通过，运行时代码中没有导入原项目模块，CUDA 训练速度、完整数据上的收敛和正式排序性能仍需在远程 GPU 上验证。

### 旧版真实 MPS MVP 结果

`python scripts/run_mvp.py` 已成功完成，机器使用 Apple MPS，自动 CPU fallback 被关闭，总耗时 643.52 秒，约 10 分 44 秒。子集为 train 210、valid 30、test 60，分别提取出 1,140、175 和 337 个区域 token，test 的 60 张图各有其它 59 张图作为 candidate。

| 模型 | 实际训练轮数 | 最佳 epoch | optimizer 更新次数 | 停止原因 |
| --- | --- | --- | --- | --- |
| 主方法 local | 3 | 2 | 429 | early_stopping |
| 整图监督 5% | 2 | 1 | 64 | early_stopping |
| 整图监督 10% | 2 | 1 | 64 | early_stopping |
| 整图监督 50% | 2 | 1 | 64 | early_stopping |

主方法阶段耗时 396.56 秒，三个监督预算合计耗时 189.35 秒，五组检索耗时 16.23 秒，统一评价耗时 0.88 秒，几何示例生成耗时 25.56 秒，再次执行统一入口 `--resume` 的检查耗时 7.67 秒，其余时间用于文件和参数校验。监督预算实际分别使用 10、20 和 104 张 train 图，所有模型的 loss 与参数均为有限值，参数 SHA-256 均不同于初始化，best checkpoint 均对应各自最低 valid loss。

五组方法各生成了 3,540 条完整 ranking，共 17,700 条记录，候选集合、去重与排除自身的检查全部通过，对比表、每 query 指标、置信区间、配对比较和两张几何示例图均已生成。再次 `--resume` 后，四个模型的 best/last checkpoint、history 和 training summary 的 SHA-256 全部保持不变，确认没有重复参数更新；另外逐条核验了 300 个子集 ID 与原 map 的标签及原 split 对应关系。

完整机器可读报告保存在 `artifacts/mvp_mps_300/mvp_report.json`，日志保存在同目录的 `run.log`，对比表为 `comparison.csv`。报告的 `status` 为 `passed`，`completed_resume_keeps_training_unchanged` 和 `formal_inputs_unchanged` 均为 true，正式配置文件、原始 NPZ 和三份正式冻结清单的内容哈希与测试前一致，所有测试产物保存在独立 MVP 目录中。
