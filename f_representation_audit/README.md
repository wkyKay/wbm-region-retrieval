# F 表示诊断

独立检查冻结 cc8 F 的单类语义分离、同类负例和窗口上下文依赖。只做 train/valid 推理及表示梯度计算，不训练、不更换主实验 checkpoint、不运行 test。公共协议与结果概览见[项目说明](../doc/项目说明.md)。

## 验证设计

- 固定原训练模型及 memory：`artifacts/f_segmentation_v1/cc8/seed_2026/F/` 的第15轮 `best_retrieval`、第100轮 `best_nce`，另按 seed=2026 重建并校验初始化参数哈希。第15轮曾按 valid 检索选出，因此阶段比较属于探索性诊断。
- 单类图：完整 multi-hot 标签只有一位为1。主分析每图取最大原始 cc8 区域；valid 使用全部可用单类图，train 每类最多128图；另报告 valid 单类图全部区域。所有近邻均排除同晶圆的全部区域。标签只用于事后分析，单类晶圆标签仍不是区域像素真值。
- 分别分析检索用 h512、损失用 z128；使用不作负值裁剪的 cosine。报告类内/类间相似度、两者间隔、1-NN、P@K与每类明细。先图内平均再类别等权；类内/类间另按候选晶圆及类别均衡。单类诊断 gallery 不属于正式完整检索协议。
- 第15→100轮及初始化→100轮使用同一原始ID配对，类内按晶圆整簇 bootstrap 2000次。置信区间条件于固定 gallery，不反映训练种子不确定性。同类相似度下降不能单独证明语义表示变差。
- 负例分析保留训练库全部99,786个区域，包括混合图和同图其他区域；按原规则排除 batch 后有放回抽5000个负例。固定诊断 shuffle 计算单类跨图负例期望，不冒充历史 batch。
- 每类8个训练图、4次负例抽样，使用 checkpoint memory，冻结 eval/BN，计算原图与90°旋转图的 z 梯度。比较完整 NCE、去除同类跨图负例的反事实及同类负例直接项；去除项会改变 softmax 分母，不等价于保持负例数不变的替换实验。使用原 `nce_loss` 的 autograd 核对，不执行优化器更新。
- 上下文分析：valid 每种完整标签组合最多4图，保留所选图全部区域。先在52×52原始窗口移除非目标缺陷，再按原规则 resize，目标 mask 和有效晶圆支持不变。比较旋转、随机移除50%上下文、移除全部上下文。报告无操作数量及共同有效样本集合。上下文可能含同一缺陷的其他部分，敏感不自动等于有害。
- `target_removed_control` 保留 mask 却删除对应缺陷，是故意无效的定位通道探针，不是合法区域输入，不作为主改进证据。

## 运行与产物

在仓库根目录、原 PyTorch 环境中执行：

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -u -m f_representation_audit.run --device cuda:0
python -m unittest discover -s f_representation_audit/tests -v
```

默认使用真实启动时间创建 `artificial/YYYYMMDDTHHMMSS+0800__f_representation_audit/`。可用 `--output-root` 指定新的独立目录；已有目录拒绝覆盖，不支持复用部分失败结果。`--dry-run` 仅显示参数。

入口核对源 manifest、完整依赖代码、原始数据、冻结协议、训练完成状态、缓存及 checkpoint；结束再校验保护文件与模型全部参数/BN缓冲。不会读取 test 输入缓存。`summary.json`、`summary.txt`、逐区域CSV、配对统计及 `figures/` 是主要产物；`selected_regions.csv` 固定原始ID，`source_protection.json` 保存来源哈希，`completed.json` 覆盖全部输出。训练缓存、模型和历史结果保持原位只读。

2026-11-21已在服务器完成首轮：`artificial/20261121T153704+0800__f_representation_audit/`，75份输出哈希核验通过，服务器46项测试通过。z128后期同类结构下降；h512代表区域近邻下降不明确、全部区域分析有小幅下降；另确认上下文敏感。完整数值和边界见[项目说明的实验结果](../doc/项目说明.md#实验结果)。诊断按登记规则进入目录盘点，不添加到正式检索成绩表。
