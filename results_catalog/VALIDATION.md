# 结果登记验证记录

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
