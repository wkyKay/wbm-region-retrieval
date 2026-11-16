# 项目协作与结果管理约定

本文件适用于 `wbm-region-retrieval/` 全部目录。修改前检查 Git 状态、相关目录更具体的 `AGENTS.md` 和当前代码，保留已有改动。用户仅询问或要求方案时，不擅自修改代码、启动训练或正式评价。

## 主方法与查阅入口

- 未另行指定时，当前主方法为 F 的 cc8 八连通＋径向位置＋相对面积；复用冻结的 `best_nce`，seed=2026。其余五种分割是对照；融合和组合属于扩展实验，不自动替换主方法。
- 方法定义和来源保护见 `experiments/proposed/f_segmentation_geometry/README.md`；固定评分、原始 mask 几何量、Top-4、h512、单向贪心和 sqrt(area) 聚合等科学设置不得因整理结果而改变。
- 最新成绩先读 [当前结果总览](results_catalog/CURRENT_RESULTS.md)，结果更新规则读 [登记说明](results_catalog/README.md)，来源身份以 `results_catalog/sources.json`、`snapshot.json` 及原始 manifest 为准。旧日期文档和 `experiments/_archive/` 只作历史追溯。
- 改实验实现前查明源码签名依赖；结果管理代码保持在 `results_catalog/`，不为目录整理改动冻结训练或评分源码。

## 新实验必须使用时间型 run_id

- 每个新实验在实际启动时分配并持久化 `run_id` 和带时区的 `started_at`。统一采用 Asia/Shanghai：`YYYYMMDDTHHMMSS+0800__稳定语义key`；`started_at` 用 ISO 8601，如 `2026-11-17T09:30:00+08:00`。
- 时间必须是真实启动时间，不用计划时间、当前整理时间或文件复制时间代替。后缀区分方法、变体、seed、checkpoint、split及同秒运行，确保唯一；示例：`20261117T093000+0800__f_cc8_radial_area_seed2026_best_nce_test`。
- 结果登记单位是“方法／变体 × split”的评价；训练运行单独记录其身份并由评价引用，同一排名的 exact 与 coverage 合并登记。训练启动时间不能冒充后续独立评价的启动时间。批量任务可有带时间的父目录，各评价保留自己的身份。
- 新运行使用含时间身份的新输出根，通过显式 `--output-root` 等参数传入。实现新入口时支持该规则；不修改旧入口默认目录来冒充新版本，不重命名既有目录。科学设置和签名一致的断点续跑沿用原 ID、started_at 和目录。
- 科学设置、源码签名、数据协议、模型、checkpoint 或候选规则变化时，创建新 key、run_id 和输出目录；用来源字段关联旧实验，禁止改写旧 manifest 或完成标记以强行复用。
- 历史登记仍保留原 ID 和 `id_time`。`completion_marker_mtime` 仅是明确标注的历史回退依据，不是开始时间；不能给旧运行补造 started_at，也不因同步 mtime 改变而重分配 ID。
- 当前登记器对历史数据使用完成标记 mtime。接入第一项带真实 started_at 的新实验时，必须先扩展 `results_catalog/build.py` 读取并验证持久化的时间身份，保留已登记 ID，并验证重采集稳定性；不能让新实验继续静默回退到 mtime。此项是新实验接入要求，不授权现在启动实验或重跑历史结果。

## 完成后必须登记并生成汇总

1. 实验开始前先确认当前协议、来源模型、输出隔离和时间身份；将运行命令、选择规则、监督信息及是否为事后实验写入新运行的配置／manifest 或独立元数据。
2. 以完成标记和其引用文件的校验结果确认完成，不以进程退出、tmux 消失或部分 CSV 存在判断成功。失败、运行中、smoke 和历史不同协议不得加入正式成绩表，应在运行记录或目录盘点中明确状态。
3. 完成后更新唯一的跨项目来源清单 `wbm-region-retrieval/results_catalog/sources.json`，记录唯一 key、方法、变体、split、协议、报告类型、来源路径、完成标记、manifest、选择记录和设计边界。检查原有记录，不给同一结果重复分配身份。
4. 在持有完整产物的服务器从本仓库根目录执行：

   ```bash
   python3 results_catalog/build.py collect --workspace /data/wangkaiyan2024
   python3 results_catalog/build.py check
   ```

5. 将新的快照和生成文件同步到本地；可离线执行 `python3 results_catalog/build.py render` 和 `check`。若改变生成器或新增协议适配，运行 `python3 -m unittest discover -s tests -p 'test_results_catalog.py' -v` 及适用的新增验证。
6. `snapshot.json` 由采集生成，`registry.csv`、`metrics.csv`、`tables/*.csv` 和 `CURRENT_RESULTS.md` 由脚本生成。禁止手工改数、绕过签名或修改快照来掩盖失败；来源清单改变后须重新 collect，不能继续使用旧快照。
7. 更新相关实验 README、运行／验证记录和当前入口，注明完成状态、原始路径、结果及限制；不得留下失效的“已启动、尚未确认完成”。有保存价值的旧分析用固定 Git 版本保留原文并在项目说明的历史记录节提供入口，不把旧结论改写为新实验结论。

## 汇总与论文比较边界

- 复用冻结去重协议：train/valid/test=25,193/3,599/7,199；valid 固定 500 query，每 query 3,598 候选；test 全 7,199 query，每 query 7,198 候选。排除自身，Normal 不入缺陷检索库，sample_id 保留原始 NPZ 行号。
- 普通 `retrieval_v1` 与排除相同完整标签组合的 `coverage_cross_combination_v1` 必须分表。新增协议须独立命名、显式适配并验证，不能仅给旧成绩改标签。
- exact 与 coverage 使用同一排名、分别定义相关性；AP/Recall 分母及 coverage IDCG 取对应完整候选库。Top-500 不称为完整排名，mAP@K 不称为 full mAP。
- 保留 K=1/5/10/20/50/100/500、总体、类别宏平均、单／混合及标签数量分组；不得混合 split、group、checkpoint 或候选池求均值。缺失报告不补造，缺失数值不填零。
- CSV 保存原始比例；展示百分数乘 100，差值乘 100 为百分点。监督／自监督／无训练、valid 选参规则和事后 test 组合必须显式记录。重复端点或相同排名的对照角色不是独立重复实验。
- cc8 主方法身份在查看历史 test 后确认；2026-11-15 角向＋最优匹配组合属于事后补充。保留这些事实，不将 `posthoc=false` 解读成整个方案从未接触 test，也不未经授权用新结果替换主方法。

## 历史保护与同步

- 原始数据、冻结协议、已有模型／特征／排名／报告、`artifacts/`、`archives/` 和历史 manifest 保持原位、原样；未经明确要求，不删除、覆盖、改名、重建或批量去重。新登记不搬迁原实验目录。
- 新评估及后处理写入独立的新目录，来源只读。历史绝对路径和签名不改写；登记中以映射／来源字段说明迁移关系。
- Git 管理代码、配置、冻结协议、文档、登记器及轻量快照／汇总；不加入模型、原始数据、大型缓存或完整排名。GitHub 同步不等于大产物备份。
- 按本会话已确认的工作约定，完成本地修改和验证后，同步相关仓库的本地、服务器和 GitHub。提交前检查三方分支和未提交改动，采用普通推送及服务器快进；不强推、不覆盖他人改动。若远程有新提交，先安全整合并验证。
- 两项目统一使用 `main`；本项目 origin 为 `https://github.com/wkyKay/wbm-region-retrieval.git`。同步后核对提交号、受 Git 管理文件及工作区状态，准确报告未完成的同步或验证，不宣称原始大产物也已同步。

## 文档整合与维护

- 当前说明统一维护在 [doc/项目说明.md](doc/项目说明.md)，用二级标题划分设计、协议、运行、结果和登记等内容；根 README 只保留入口，不再按主题拆出重复文件。
- 详细数值以主项目 results_catalog 自动汇总为准；项目说明的结果节只写结论、状态、来源和限制。方法专属参数仍留在对应实验／方法 README。
- 有价值的旧分析在项目说明的历史记录节概括，原文用固定 Git 提交链接保留；来源清单保存原路径、版本和 SHA-256。已整合的旧正文及跳转文件直接删除，更新引用，不保留重复 Markdown；不能把旧模型成绩改写成新模型成绩。
- 修复引用时优先使用仓库相对链接；跨仓库用 GitHub 链接。未随 Git 发布的产物写明路径及所在端，不留下只在某台机器可用的 Markdown 文件链接。
- 更新论文时同步项目说明中的论文初稿节与 LaTeX，明确监督信息、valid 选择及事后 test 实验边界。历史产物与冻结协议仍保持原位、原样。

- `doc/` 下的 Markdown（含历史归档）统一使用简短中文文件名；保留必要的方法标识，改名时同步更新链接、来源清单和生成报告中的引用，不新增英文跳转文件。
