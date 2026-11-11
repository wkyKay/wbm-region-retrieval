# 去重数据协议迁移验证（2026-11-09）

> 本文记录先前的数据迁移阶段。其后已完成 exact_v1 指标接入，最新状态见 [统一评价说明](EXACT_METRICS_MIGRATION.md)。

本次将正式入口迁移到与 `compared_methods` 相同的去重划分和固定 query/gallery。
本地与服务器 `/data/wangkaiyan2024/wbm-region-retrieval` 已同步本次实现。
原始 NPZ、旧 `protocol/split.csv`、`queries.csv`、`candidates.csv` 和历史运行产物保留。

## 固定数据身份

`protocol/deduplicated/` 的3个文件与 `protocol/retrieval_v1/` 的5个文件是
`compared_methods/protocol/` 的逐字节镜像。独立仓库运行不需要依赖相邻仓库的位置；
源数据、划分、重复映射、四份查询/候选清单及元数据均通过 SHA-256 绑定。
服务器另外与 `/data/wangkaiyan2024/wbm-matching/protocol/` 实际比较，一致。

- 完整原图按全部0/1/2/3像素精确去重，保留最小原始行ID；同图标签冲突报错。
- 原38,015条记录过滤1,024条冗余记录，保留36,991张唯一图。
- 缺陷图35,991张，train/valid/test为 **25,193 / 3,599 / 7,199**；Normal不加入本项目缺陷训练和检索。
- valid固定500个query，gallery为3,599张，每query排除自身后3,598候选。
- test固定全部7,199个query，每query排除自身后7,198候选。
- `sample_id`仍为原始NPZ行号；训练seed改变不重新抽取检索池。
- 固定检索签名：`6a428fb8d8bdf34fb6c60852b6d9a7467bf6880cae8c802a6a9c5a36f96cb8a1`。

## 接入和兼容性

`Protocol` 验证完整去重代表、重复映射、原始标签/valid_index和声明的分层归属。
ABCD、EF、G和监督B共用固定验证池，query/gallery以线性空间保存，候选按query生成。
完整排名评价也按query读取，避免一次载入全部51,818,402条test配对。

13个正式入口及共享基础配置已切换到 `data_protocol="deduplicated_v1"`，
默认输出使用 `artifacts/deduplicated_v1/` 下的独立目录。缓存/训练签名覆盖新协议文件及
协议实现，拒绝旧运行；旧训练集与新valid/test重叠，旧checkpoint不能用于正式新协议成绩。
`single_label` 显示名称统一为索引5 `near-full`、索引7 `random`。

历史300图MVP显式使用legacy模式和独立历史源配置；不视为新协议smoke。
原有模型、区域匹配、containment指标及选模算法未改为exact评价，
因此“数据协议一致”不表示已经接入五方法的exact_v1指标。详见[协议说明](../protocol/README.md)。

## 已完成验证

| 检查 | 结果 |
| --- | --- |
| 本地完整 `unittest discover -s tests -v` | **102项通过，191.956秒**，含多进程和训练恢复 |
| 服务器wbm环境：去重协议、流式评价、四套件接入 | **19项通过**，14+3+2 |
| 真实数据与13个正式入口审计 | train/valid/test原始ID及固定query/gallery一致 |
| ABCD/EF/G微型prepare与手工valid | 相同6 query、每query8候选，排除自身，源文件不变 |
| 监督B微型prepare | 固定候选池、预算仅来自新train，无模型训练 |
| 新旧协议隔离 | 旧缓存拒绝且不覆写；篡改/漏代表/重复副本/错误划分均拒绝 |
| 原始NPZ、旧3份CSV及compared_methods的8份冻结文件 | **12份保护文件哈希未变化** |

本地证据位于 `artifacts/protocol_migration_20261109_azt96h9c/`，包含本地/服务器测试日志、
真实数据审计、保护文件哈希和 `verification.json`。远程同步逐文件核对更新前版本，
更新后再次核对内容哈希；历史产物目录未同步或覆盖。

本次只完成实现和验证，**未启动新协议正式训练或完整test模型评价**。
