# 结果登记验证记录

## 2026-11-20 六项F/G/J跨组合coverage

- `artificial/fgj_cross_test_20261120T193642+0800/`于19:36:42实际启动、19:56:20完成（Asia/Shanghai）。固定评价0.5G+0.5J、0.5F+0.5J、F、G、J及WBBS；F经用户确认采用冻结最优区域匹配版，J为整图`exp(-2D_H)`。不重训、不调权、不以test排序选择新的主方法。
- 本地和服务器49项专项测试通过。正式评价沿用`coverage_cross_combination_v1`，7,199个query、每项50,395,860个完整合格配对，保留原六指标、七个K及全部分组。独立重排核验全部Top-500、逐query和汇总报告；保存分数／分项误差0，指标最大绝对误差5.55e-16。nDCG有效query为7,000，FullCoverage Recall/AP为6,200，其他指标为7,199。
- F、WBBS分别逐query重放历史同协议Top-500，7,199个query的候选顺序完全一致，分数误差0。37类代表的266,326个原完整池G配对与原缓存逐位一致；另740个G/J配对通过独立点积与标量Hellinger核验。769个使用来源绑定SHA；55,929个历史文件大小／mtime不变，小于1MiB的文件额外核对SHA。该范围不表示重新训练CNN或重新计算全部WBBS底层分数。
- 六项均按真实时间身份登记为coverage-only的事后固定方案诊断。服务器官方`collect/check`和两套无损归档`pack/verify`通过，共421项；旧415项逐字段深度不变，新增恰为六项。普通检索与跨组合检索继续分表，完整排名保留服务器。
- 数值主表、指标独立排序与边界见[消融实验](../doc/消融实验.md)，运行和证据入口见[实验README](../fgj_cross_test/README.md)。Coverage-nDCG@10以J领先，FullCoverageHit@10以F领先；这些是描述性比较，未作本轮显著性检验。

## 2026-11-20 区域面积融合与无损快照归档

- 独立`f_region_area_fusion/`的29项valid及冻结前十项test均已完成、封存并登记。两阶段登记后服务器官方`collect/check`通过，现有403项结果、七份自动生成文件；相对于启动提交`81b659d`的全部364条旧记录逐字段深度不变，新增恰为29条valid和10条test。
- 每项同一排名上的exact与coverage仍合并登记，保持完整固定候选协议；主选、冻结次序、来源SHA及双指标报告均由登记器验证。所有39项的P@10、mAP@100、Coverage-nDCG@10与原始comparison CSV逐值一致。验证细节与原始报告链接见[实验说明](../f_region_area_fusion/README.md)。
- 新实验49项测试在本地与服务器通过。正式valid/test核验覆盖全部query、完整候选评分、排序及全部K／分组双指标；55,202／55,562个历史文件保护检查通过。源码、模型、数据和旧运行目录未覆盖。
- 额外独立审计重读29项valid及10项test共43,245,000行Top-500，按原始标签重算三项主指标、各组与37类报告，共核对280,657个值，最大绝对误差4.44e-16。对旧WBBS缺失的报告协议字段，仅在新审计脚本中核验其完成标记绑定的manifest及排名SHA，原WBBS与早期审计文件不变。配对bootstrap另存独立分析目录，不参与选参；证据见[实验说明](../f_region_area_fusion/README.md)。
- 完整`snapshot.json`现为107,166,631字节，超过GitHub单文件大小限制；无损gzip为28,544,167字节。Git保存`snapshot.json.gz`及`snapshot.sha256`，本地／服务器仍保留原路径完整JSON，原采集、登记、评分源码无需改动；未改写Git历史。
- 原始JSON SHA-256为`613f3747123f4ae50a921a52fc5fd62f0358d02c0eee0258375e9c2883b2a848`，gzip SHA-256为`ac2e18b08c3bbe6b5dffca3a1795808a1941612b1880d7b62f529370f2760727`。`pack/verify`通过；独立临时目录仅复制归档及目录工具后`unpack`，恢复字节和SHA完全相同，离线`check`仍为403项／七份文件全部通过。
- 新归档工具12项测试在本地与服务器通过，覆盖损坏／截断归档、哈希和大小不符、已有不同JSON拒绝覆盖、符号链接拒绝、相同内容恢复、归档失败回滚、锁及原始字节保留。归档校验只证明字节完整性，不替代实验完成证据和正式登记检查。

## 首批登记

2026-11-16，首批登记 57 项已完成评价，包含 102 份 exact／coverage 报告，生成 28,672 条指标。只读取原始结果，不启动训练或评价。

## 已完成检查

- 102 份报告的 SHA-256 与各自 `completed.json` 一致；五份独立 coverage 后处理的排名哈希引用与对应 exact 完成标记一致。
- 两仓库的去重 split、检索 metadata 和四份 query/gallery 清单逐字节一致。登记报告的基础协议签名、query 数、split、K及跨组合候选身份通过检查。
- 本地与服务器各 9 项 `test_results_catalog.py` 专项测试通过：报告篡改／缺失拒绝、冻结协议与 split 错配拒绝、相同 query 数的旧协议拒绝、跨组合与普通协议隔离、完成结果改变拒绝复用、时间 ID 稳定、原始文件只读、逐指标来源一致、分表边界和确定性生成。
- 本地 `render` 后 `check` 通过；服务器与本地七份生成文件 SHA-256 一致。总览和新增入口中的本地 Markdown 链接已检查。
- 服务器两个仓库的 `artifacts/`、`archives/`、`protocol/` 共 5,102 个文件／符号链接，操作前后路径、大小、mtime_ns、链接目标一致；protocol 文件同时核对内容哈希。大产物未逐一计算内容哈希。
- 跨组合实验根完成标记为 `completed`、`all_seven_methods_verified=true`，其汇总、REPORT 和独立验证记录哈希一致。
- 新登记器位于 `results_catalog/`，不修改或加入原训练／推理模块的源码签名范围。旧 2026-11-10 总览原文完整保留于 `experiments/_archive/ALL_METHODS_AND_METRICS_20261110.md`。

## 范围

本次验证不重新计算排名或指标，不重新核验所有模型／特征哈希，不重跑算法全套测试。历史实验独立逐 query 复算结论仍以其原验证记录为准。登记状态使用 `report_hashes_verified`，不将报告读取误称为重新完成实验。

```bash
python3 results_catalog/build.py check
python3 -m unittest discover -s tests -p 'test_results_catalog.py' -v
```
