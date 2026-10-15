# 交付前验证记录

本次验证使用临时 Python 3.12.14 环境及 NumPy 2.5.4、SciPy 1.18.1、PyTorch 2.14.1 和 Matplotlib 3.11.2，模型计算使用 CPU，正式数据只用于读取协议和标签校验，没有执行主方法或整图监督方法的正式训练，也没有生成任何真实数据上的训练性能结果。

`python -m unittest discover -s tests -v` 已通过 17 项测试，覆盖纯 8 连通域、面积过滤与 Top-K、token patch 的平移一致性、手工特征的直角旋转一致性、PCA-scale、贪心一对一约束、query 侧分母、额外 candidate 区域、非对称 containment、标签预算嵌套、训练 pair 边界、评价指标、模型前向与梯度、memory bank 负样本排除、epoch 增强变化以及 checkpoint 权重与随机状态恢复。集成测试使用临时的小样本和随机权重 checkpoint，完成五组检索文件、统一评价、对比表、配对 bootstrap 和几何示例生成，测试中没有调用 optimizer step。

三个方法已经分别移入 `methods/proposed.py`、`methods/handcrafted.py` 和 `methods/supervised_global.py`，新增的两项测试验证统一入口会调用对应方法文件的训练函数，并且单独指定检索方法时只会调用被选中的方法。拆分后的完整测试和入口 dry-run 均已通过，训练损失、区域打分、贪心匹配规则和结果文件格式保持不变，缓存与 checkpoint 指纹已经包含对应方法文件的实现代码。

`python run_experiment.py --stage audit` 已在现有原始 NPZ 和复制后的冻结清单上通过，实际核验到 train 25,910、valid 3,700、test 7,405，固定 query 2,000，每个 query 都恰好有 1,000 个不重复的合法 test candidate，标签维数为 8，完整标签组合数量为 37。三份清单的 SHA-256 与 `protocol/README.md` 一致，数据没有重新划分，原项目文件没有被修改。

入口的 `--dry-run` 和 Python 语法检查已通过，运行时代码中没有导入原项目模块。GPU 显存占用、CUDA 训练速度、完整 epoch 优化与实际排序性能需要按 README 的命令在远程 GPU 上验证，本次没有用随机权重的临时结果代替正式模型结果。
