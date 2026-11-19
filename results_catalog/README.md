# 结果登记与自动汇总

本目录统一索引 `compared_methods` 与 `wbm-region-retrieval` 的已完成正式评价，保留原始运行目录和产物。阅读最新成绩从 [当前结果总览](CURRENT_RESULTS.md) 开始。

## 文件分工

| 文件 | 用途 |
|---|---|
| `sources.json` | 人工维护的来源清单：方法、变体、split、原始路径、必需的报告类型和实验边界 |
| `snapshot.json` | 从服务器只读采集的报告、完成标记哈希、来源 manifest、选参记录、协议哈希和目录盘点；支持离线生成 |
| `registry.csv` | 每项评价一行的登记表；`remote_path` 指向原始结果目录 |
| `metrics.csv` | 全部指标长表，保留原始比例和精度；缺失值为空，不填零 |
| `tables/methods.csv` | 普通 test 的六对比方法与当前 cc8 主方法 |
| `tables/segmentation.csv` | 六种分割的 valid/test 对照 |
| `tables/fusion.csv` | 全局融合、单项消融和事后组合；保留 valid 网格和 test 对照 |
| `tables/cross_combination.csv` | 排除相同完整标签组合候选的独立 coverage 协议 |
| `CURRENT_RESULTS.md` | 自动生成的当前状态与代表性结果 |
| `build.py` | 仅依赖 Python 标准库的采集与生成程序 |

登记单位是一次“方法／变体 × split”的评价。训练模型作为来源引用，不给同一权重重复创建训练记录；同一排名的 exact 与 coverage 合并到一条登记。五个早期方法的 valid 只有 exact，不补造 coverage。两项旧 F 结果只作历史参照。

本目录不是所有历史实验的完整数据库。未逐项纳入的历史协议、诊断、smoke、失败尝试、训练目录与重复后处理批次，在 `snapshot.json` 的 `inventory` 中保留目录索引；`not_in_current_tables` 不表示无效或可删除。

## 时间型 run_id

格式为 `YYYYMMDDTHHMMSS+0800__稳定语义key`，例如 `20261109T121535+0800__wu_valid`。时间用于排序，语义后缀避免同秒或同批多方法重名；不修改原始目录名。

历史运行缺少统一的开始时间字段，保留完成标记 `completed.json` 的文件 mtime 作为登记时间依据，`id_time.basis=completion_marker_mtime`，`is_execution_start=false`。这是可核对的文件时间，不宣称是训练或评价开始时间。`collected_at` 单独记录本次采集时间。已有 ID 在重采集时保留，即使复制文件改变 mtime，也不重新分配 ID。

新增实验在实际启动时记录真实时间。每个“变体×split”的来源条目声明 `identity_schema_version: 1` 和 `identity`；后者是身份文件相对于所属仓库的路径，例如 `artificial/<本次运行>/valid/<变体>/run_identity.json`。身份文件内容为：

```json
{
  "schema_version": 1,
  "key": "example_valid",
  "split": "valid",
  "started_at": "2026-11-17T09:30:00+08:00",
  "run_id": "20261117T093000+0800__example_valid"
}
```

示例时间不能代替实际启动时间。`started_at` 必须精确到秒并带 `+08:00`；run_id 的时间部分必须与之相符，后缀必须等于登记 key。身份文件必须位于该完成标记覆盖的目录内，并纳入 `completed.json` 的文件哈希。登记为 `id_time.basis=started_at`、`is_execution_start=true`，同时保存身份文件及哈希。断点续跑保持原身份。

已声明新格式但身份缺失、损坏、key/split 不符或时间不合法时拒绝采集，不回退到 mtime。manifest 已包含 `run_id/started_at` 却未声明身份来源也会拒绝。旧运行不补造开始时间，已有 key 不切换时间依据；`key`、身份和原始结果目录一经登记保持绑定，已完成产物内容改变时须另建实验和登记。

## 更新命令

两仓库相邻的服务器工作区为 `/data/wangkaiyan2024/`。采集先核对两仓库冻结协议一致、报告和完成标记哈希一致、split/query/K 合法、跨组合候选身份正确、独立 coverage 的排名哈希引用与 exact 一致；全部通过才生成输出。

```bash
cd /data/wangkaiyan2024/wbm-region-retrieval
python3 results_catalog/build.py collect --workspace /data/wangkaiyan2024
python3 results_catalog/build.py check
```

本地无需下载模型、特征或原始排名，可以从已经核验并纳入 Git 的快照重建：

```bash
python3 results_catalog/build.py render
python3 results_catalog/build.py check
python3 -m unittest discover -s tests -p 'test_results_catalog.py' -v
```

新增结果先编辑 `sources.json`，再在有完整产物的服务器 `collect`。修改来源清单后直接离线 render/check 会拒绝，避免使用旧快照。相同快照生成的文件逐字节稳定；`check` 只检查已提交文件是否与快照一致，不连接服务器，也不代表源模型或原始排名已重新核验。

正式 `collect` 将 `snapshot.json` 保存为紧凑 JSON，以减少重复来源元数据占用的文件体积。这只压缩空白排版，所有字段、浮点数、历史记录及来源内容保持不变；其他 JSON 默认仍用两空格缩进。通过官方采集生成，无需改写历史 manifest，离线 `render/check` 的报告内容不受影响。

`artifacts/` 与 `artificial/` 下的运行目录均进入只读目录盘点；目录存在不等于已完成或已登记。普通协议下新增的 `fusion`、`fusion_ablation`、`posthoc` 类别 test 结果会追加到总览融合表，valid 与跨组合结果仍分开处理。新增方法的注册入口可以生成来源条目，只有报告、完成标记和冻结选择全部就绪后才能写入唯一 `sources.json`；登记器不依赖任何具体实验包。

## 指标读取规则

- `metrics.csv` 每行由 `run_id + metric_version + group + metric + K` 唯一确定。`value` 是原始比例；展示百分数时乘 100，差值乘 100 是百分点。
- `group=all` 是 query 平均，`class_macro` 是完整组合类别宏平均；`single_label`、`mixed_label`、`by_label_count/1` 等是独立分组，不能再次混合求平均。
- exact 与 coverage 来自同一排名，但相关性不同。AP/Recall 分母和 coverage IDCG 来自对应完整候选库，mAP@K 不是 full mAP。
- 普通 `retrieval_v1` 与 `coverage_cross_combination_v1` 分表。后者额外排除同组合候选，不能混入普通 test。
- `posthoc=true` 的角向＋最优匹配组合是在查看前一轮 test 后提出。主方法 cc8 的身份同样是在查看历史 test 后确认，不能以 `posthoc=false` 推导整个方案从未接触 test；完整设计边界见 `design_note` 和当前总览。
- 基线与某些对照端点可能具有相同排名，这些是不同实验里的对照角色，不能当作重复种子独立统计。

## 保存与验证边界

原始数据、冻结协议、模型、排名和运行目录均只读。当前仓库提交登记、生成代码、轻量报告快照与汇总；原始大产物仍由服务器及其独立备份保存。GitHub 同步不等于实验数据备份。

快照保留历史 manifest 原文，包括旧绝对路径、当时源码和 checkpoint 哈希；当前仓库 HEAD 只表示采集环境，不替代实验当时的代码身份。服务器旧路径 `wbm-matching` 是 `compared_methods` 的符号链接，不改写历史路径。

本登记核验报告哈希，不重训、不重跑评价，不逐一重新计算模型、特征及 Top-500 排名哈希。`status=report_hashes_verified` 精确表达此范围。完整独立复算记录仍留在原实验目录。
