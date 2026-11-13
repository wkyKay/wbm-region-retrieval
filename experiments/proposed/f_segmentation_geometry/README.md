# 六组 F：径向位置＋相对面积，valid/test 双指标

此入口为六组已训练的 F 分割模型增加推理评分，默认一次完成六组的 valid 和 test，
每组同时输出 **exact_v1 28项＋coverage_v1 42项**（7个K，共70项总体指标）。
不重新训练、不重新选 checkpoint、不用 test 调参。旧训练目录全程只读。

## 远程用 tmux 一条命令

先进入远程仓库根目录并激活原来的 PyTorch CUDA 环境。已有模型在默认
`artifacts/f_segmentation_v1/` 时，执行：

```bash
git pull --ff-only origin main && tmux new-session -d -s fseg_geo -c "$PWD" env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$(command -v python)" -u experiments/proposed/f_segmentation_geometry/run.py --device cuda --prepare-workers 4
```

这会以后台 tmux 会话运行，无需保持 SSH 连接。`$(command -v python)` 将当前激活环境的
解释器路径传给 tmux，避免已有 tmux server 的 PATH 指向另一个环境。
查看会话：`tmux attach -t fseg_geo`；返回后台：按 `Ctrl+b`，松开后按 `d`。
tmux 可以保护程序免受普通连接中断影响，服务器重启或任务被杀死仍需要重新启动。
程序结束后该会话自动退出，结果和日志保留；再次执行同一命令可校验并跳过已完成结果。
会话仍存在时，同名启动会报错，不会另起一份重复任务。
参考：[tmux 官方说明](https://github.com/tmux/tmux/wiki/Getting-Started)、
[new-session 参数](https://man.openbsd.org/tmux#new-session)。

本地没有安装 tmux；命令的 `-d/-s/-c` 与多参数直接执行方式已核对官方手册，
未在远程替用户启动正式评价。

运行前可预览（不写文件）：

```bash
python experiments/proposed/f_segmentation_geometry/run.py --dry-run
```

非默认训练目录和数据位置可追加：

```bash
python -u experiments/proposed/f_segmentation_geometry/run.py --device cuda --prepare-workers 4 --source-root /absolute/path/f_segmentation_v1 --data-file /absolute/path/Wafer_Map_Datasets.npz --output-root artifacts/f_segmentation_radial_area_custom_v1
```

原始 NPZ 默认沿用各组 `run_manifest.json` 中的路径；可用 `--data-file` 更换路径，
数据内容哈希必须相同。协议路径若已迁移，则回退到本仓库内冻结清单并校验原哈希。
训练批大小、seed等科学设置自动读取旧 manifest，不需要凭记忆重填。
可用 `--methods cc8 tv8` 分批运行，或 `--splits valid` 仅做 valid；默认两套 split 全做。

六组应先完成原有训练。未完成时本入口明确报错，不把半途模型作为正式结果，
也不自动更新旧模型。原训练入口仍可用相同配置和输出根续训：

```bash
python -u experiments/proposed/f_segmentation/run.py --stage train --device cuda --prepare-workers 4
```

完成后运行上述新 tmux 命令即可，不必先运行原始 test。若原训练任务仍在运行，
新入口会因源目录锁而停止，防止边读取边更新 checkpoint。

## 固定评分

沿用已有 valid 对照所用的固定值，六种分割不单独重新调参：

- 位置：原始区域 mask 的16-bin径向直方图；中心为有效晶圆像素包围盒中心，
  半径为有效晶圆像素到该中心的最大距离；归一化半径截断到[0,1]。
- 径向距离：`D_radial = sum(abs(CDF_q - CDF_c)) / 16`，β=1。
- 相对面积：`r = 区域原始缺陷像素数 / 有效晶圆像素数`；有效晶圆为 `raw==1或2`。
- 面积尺度距离：`D_area = 0.5 * abs(log(r_q / r_c))`，σ=1.5、λ=0.15。

区域对分数：

```text
S = clip(cos(h_q, h_c), 0, 1)
    * exp(-D_radial)
    * (0.85 + 0.15 * exp(-D_area / 1.5))
```

几何量取自完整原始 mask，不能从裁切/resize后的输入测量。
保留原 F 的 h512、三通道固定窗口、min_area=5、Top-4、单向贪心一对一匹配，
以及查询区域 `sqrt(area)` 聚合权重。没有添加PCA长短轴、绝对二维位置、四旋转汇聚或双向匹配。
`tv8` 仍是WBBS去噪＋F，`npfc_pre` 仍是前置子流程。

六组使用各自模型、各自区域。已完成且签名正确的源分割缓存可只读复用；
缺少 valid/test 完成缓存时用同一分割实现重建，所有重建产物只进入新的输出根。
不会借用8连通的几何量来评价另外五组。
空区域图保留在固定 query/gallery 中，得分为0，使用原exact的确定性同分排序。

## 两套指标及输出

冻结数据身份保持 train/valid/test=25,193/3,599/7,199；valid固定500个query，
各检索3,598个候选；test固定7,199个query，各检索7,198个候选，均排除自身。
按query流式评分完整候选库，只保存Top-500排名；两套指标读取**同一份排名**。

- exact：完整multi-hot组合相同才算相关；Hit、Precision、Recall、mAP。
- coverage：查询标签覆盖比例为线性gain；Coverage-nDCG、MeanCoverage、
  FullCoverage Hit/Precision/Recall/mAP。FC表示候选包含查询的全部缺陷类别，允许额外类别。
- 两套AP/Recall的相关样本分母和coverage的IDCG都来自完整固定候选库。
  截断AP不使用 `min(R,K)`；没有相关候选时的排除规则沿用原评价器。
- K=1、5、10、20、50、100、500；同时报告总体、类别宏平均、单/混合缺陷及标签数量分组。

默认读取旧训练时冻结的 `final_checkpoint`（此前命令为 `best_nce`）。
显式 `--checkpoint` 也必须与该规则一致，不通过test选模。

```text
artifacts/f_segmentation_radial_area_v1/
├── run.log
├── status.json                       # running/completed/failed及当前方法
├── comparison.csv                    # 六组×valid/test×两套指标及分组，长表
├── comparison.json
├── geometry_comparison.json
└── cc8/                              # cc4/median8/opening8/tv8/npfc_pre结构相同
    ├── evaluation_manifest.json      # 原模型/配置签名、checkpoint哈希和新评分源码哈希
    ├── features/
    │   ├── valid/
    │   └── test/
    │       ├── h512.npy
    │       ├── region_labels.npy     # map_ids顺序；0未分配、1..4区域
    │       ├── regions.npz           # map_ids/offsets/area/relative_area/histograms
    │       ├── provenance.json
    │       └── completed.json
    ├── valid/best_nce/
    └── test/best_nce/
        ├── rankings.csv
        ├── metrics.json             # exact总体/宏平均/分组
        ├── per_query.csv
        ├── per_class.csv
        ├── coverage_metrics.json    # coverage总体/宏平均/分组
        ├── coverage_per_query.csv
        ├── coverage_per_class.csv
        └── completed.json
```

每完成一组即更新汇总。每个split在临时attempt目录中完成两套报告，再原子改名；
中途失败的attempt保留，重跑不会覆盖。已完成的新结果会校验哈希后跳过。
配置、数据、checkpoint或新评分源码改变会拒绝复用，需指定新的 `--output-root`。
原结果的路径、内容与完成标记均不改动。

## 旧模型兼容与验证

原 `f_segmentation/*.py`、公共训练代码和 `wbm_retrieval/*.py` 均未修改，
新模块位于独立兄弟目录，避免被原训练源码指纹收录。
核对原签名涉及的34个源码文件，与提交 `814a605` 逐字节一致；没有绕过旧签名检查。

coverage评价器是 `compared_methods/shared/coverage_metrics.py` 的逐字节副本，
SHA-256：`e9ef38a1b0c8b60fb468290438c7f1569a6ea74c14940fe44b30f186b2a436ba`。
只有NumPy/标准库依赖，远程不需要另一个仓库。原exact评价器直接复用。

验证包括：

- 六组旧入口训练出的测试模型，完成valid/test两套指标；原源目录所有文件哈希和mtime不变。
- 重跑复用、缺失test缓存重建、exact后中断重试、损坏缓存拒绝、运行中的源锁、输出目录隔离。
- 标量公式独立核验；空区域图、多区域贪心及完整候选库保留；包含额外标签时exact与coverage确实不同。
- 38类真实样例在六种分割下的825个区域与旧径向/面积实现一致；26,400个区域对分数
  与旧实现最大差异为 `2.78e-17`。
- 全项目150项测试通过；新增7项专门覆盖本入口，使用干净的Git待提交副本再次验证。

端到端测试使用合成冻结协议与轻量编码器，不代表正式模型精度；本地没有运行真实数据
7,199查询的完整test，也没有重训正式模型。

```bash
python -m unittest discover -s tests -p 'test_f_segmentation_geometry.py' -v
```
