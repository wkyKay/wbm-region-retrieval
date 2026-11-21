# 结果登记与自动汇总

本目录统一索引 `compared_methods` 与 `wbm-region-retrieval` 的已完成正式评价，保留原始运行目录和产物。阅读最新成绩从 [当前结果总览](CURRENT_RESULTS.md) 开始。

## 文件分工

| 文件 | 用途 |
|---|---|
| `sources.json` | 人工维护的来源清单：方法、变体、split、原始路径、必需的报告类型和实验边界 |
| `snapshot.json` | 从服务器只读采集的报告、完成标记哈希、来源 manifest、选参记录、协议哈希和目录盘点；支持离线生成 |
| `snapshot.json.gz`、`snapshot.sha256` | Git保存的完整快照无损归档及原始／压缩字节数、SHA-256；工作区仍使用原路径`snapshot.json`，不删字段或改成绩 |
| `snapshot_archive.py` | 归档、恢复和逐字节校验工具；不修改现有采集、登记或评分代码 |
| `generated_archive.py`、`generated.sha256` | 大型派生表的无损归档、恢复与校验；Git保存`metrics.csv.gz`及`tables/fusion.csv.gz`，本地仍使用完整CSV原路径 |
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
python3 results_catalog/snapshot_archive.py pack
python3 results_catalog/snapshot_archive.py verify
python3 results_catalog/generated_archive.py pack
python3 results_catalog/generated_archive.py verify
```

本地无需下载模型、特征或原始排名，可以从已经核验并纳入 Git 的快照重建。新克隆先恢复完整JSON：

```bash
python3 results_catalog/snapshot_archive.py unpack
python3 results_catalog/generated_archive.py unpack
python3 results_catalog/build.py render
python3 results_catalog/build.py check
python3 -m unittest discover -s tests -p 'test_results_catalog.py' -v
```

新增结果先编辑 `sources.json`，再在有完整产物的服务器 `collect`。修改来源清单后直接离线 render/check 会拒绝，避免使用旧快照。相同快照生成的文件逐字节稳定；`check` 只检查已提交文件是否与快照一致，不连接服务器，也不代表源模型或原始排名已重新核验。

正式 `collect` 仍将 `snapshot.json` 保存为紧凑JSON；现有登记器和`build.py`仍读取该路径。随着登记增长，完整JSON超过GitHub单文件大小限制，因此Git只跟踪`snapshot.json.gz`和`snapshot.sha256`，本地与服务器继续保留完整JSON。归档不解析或重新序列化原文件，恢复后逐字节一致，所有字段、浮点数、历史记录及来源内容原样保留；不改写Git历史或旧实验manifest。

每次完成官方`collect/check`后运行`pack`和`verify`，同步归档与校验文件。归档固定gzip时间和文件名，同环境相同输入产生相同归档；不同zlib版本可能产生不同压缩字节，但原始JSON的SHA必须一致。`unpack`只创建缺失的JSON，或接受已有的相同内容；已有不同内容、符号链接或哈希不符时拒绝覆盖。已有工作区拉取更新后，若旧JSON与新归档不同，先将旧JSON另存为保留副本，再执行`unpack`，避免误覆盖尚未登记的新结果。归档校验只证明字节完整性，正式成绩仍以原有`build.py check`和来源核验为准。

415项登记后，`metrics.csv`超过GitHub的100MiB单文件限制，`tables/fusion.csv`也接近该限制。两表因此同样改为Git保存无损gzip与独立`generated.sha256`，完整CSV仍留在本地／服务器原路径，并可由原有`build.py render`从快照逐字节重建；不删行、不截断、不改旧记录。`generated_archive.py`固定只处理这两表，恢复拒绝覆盖不同现有CSV；拉取更新后先另存旧CSV再恢复，或按已核验的新快照执行原有render。每次正式collect/check后对快照和两表分别pack/verify，再同步全部归档及校验清单。GitHub上的原始表归档为[完整指标](metrics.csv.gz)和[融合表](tables/fusion.csv.gz)。

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

## 论文材料副本

用户指定的`result/A`、`B`、`C`、`D`、`F`、`G`保存写作所需资料，原运行保持原位。范围与入稿状态见[证据总表](../doc/论文实验证据总表.md)，E1/E2已免去。`paper_bundle_selection.json`固定所选来源、字节哈希及正文入口；`paper_bundle.py`逐文件校验后复制，并从既有catalog精确筛选B/C/F表及指标有效query／类别数，不新增正式评价或另建成绩登记。

在完整来源所在服务器构建，已有不同副本会拒绝覆盖；`--output`也可指定新的导出目录：

```bash
python3 results_catalog/paper_bundle.py build --selection results_catalog/paper_bundle_selection.json --workspace /data/wangkaiyan2024 --output result
python3 results_catalog/paper_bundle.py verify --output result
```

本地下载同名目录后执行第二条即可离线验收。`result/`由Git忽略，须单独传输；原`source_completed.json`仍描述原运行，根`package_manifest.json`才校验整理子集。论文数据入口为`paper_use.json`，`verification.json`记录本地独立文件及指标筛选核验；不将这份部分导出称为完整训练／排名档案。
