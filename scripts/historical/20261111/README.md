# 2026-11-11 历史实验脚本与归档

本目录于 2026-11-16 从服务器 `/data/wangkaiyan2024/` 根目录整理而来。脚本与原文件逐字节一致；本次不修改算法、配置或历史运行结果。

## 文件用途

| 实验 | 文件 | 用途 |
|---|---|---|
| `f_handcrafted_distributions` | [analyze_f_handcrafted_distributions_20261111.py](f_handcrafted_distributions/analyze_f_handcrafted_distributions_20261111.py) | F 与 Handcrafted A 的区域及整图分数分布统计和绘图。 |
| `f_threshold_05` | [audit_f_threshold_05_similarity_20261111.py](f_threshold_05/audit_f_threshold_05_similarity_20261111.py) | 固定冻结 F 模型，阈值 0.5 的事后 test 对照、相似度审计与结果核验。 |
| `f_threshold_05` | [compare_f_threshold_05_test_20261111.py](f_threshold_05/compare_f_threshold_05_test_20261111.py) | 固定冻结 F 模型，阈值 0.5 的事后 test 对照、相似度审计与结果核验。 |
| `f_threshold_valid_select` | [compare_f_threshold_valid_select_20261111.py](f_threshold_valid_select/compare_f_threshold_valid_select_20261111.py) | valid 上按 Coverage-nDCG@10 选择阈值，再冻结并测试；当时选中 0.0。 |
| `f_handcrafted_distributions` | [f_handcrafted_distributions_20261111.log](../../../archives/20261111/f_handcrafted_distributions/logs/f_handcrafted_distributions_20261111.log) | F 与 Handcrafted A 的区域及整图分数分布统计和绘图。 |
| `f_handcrafted_distributions` | [f_handcrafted_score_distributions_20261111.tar.gz](../../../archives/20261111/f_handcrafted_distributions/bundles/f_handcrafted_score_distributions_20261111.tar.gz) | F 与 Handcrafted A 的区域及整图分数分布统计和绘图。 |
| `f_threshold_05` | [f_threshold_05_comparison_20261111.log](../../../archives/20261111/f_threshold_05/logs/f_threshold_05_comparison_20261111.log) | 固定冻结 F 模型，阈值 0.5 的事后 test 对照、相似度审计与结果核验。 |
| `f_threshold_05` | [f_threshold_05_test_20261111.log](../../../archives/20261111/f_threshold_05/logs/f_threshold_05_test_20261111.log) | 固定冻结 F 模型，阈值 0.5 的事后 test 对照、相似度审计与结果核验。 |
| `f_threshold_05` | [f_threshold_05_test_results_20261111.tar.gz](../../../archives/20261111/f_threshold_05/bundles/f_threshold_05_test_results_20261111.tar.gz) | 固定冻结 F 模型，阈值 0.5 的事后 test 对照、相似度审计与结果核验。 |
| `f_threshold_valid_select` | [f_threshold_valid_select_20261111.log](../../../archives/20261111/f_threshold_valid_select/logs/f_threshold_valid_select_20261111.log) | valid 上按 Coverage-nDCG@10 选择阈值，再冻结并测试；当时选中 0.0。 |
| `f_threshold_valid_select` | [f_threshold_valid_select_comparison_20261111.log](../../../archives/20261111/f_threshold_valid_select/logs/f_threshold_valid_select_comparison_20261111.log) | valid 上按 Coverage-nDCG@10 选择阈值，再冻结并测试；当时选中 0.0。 |
| `f_threshold_valid_select` | [f_threshold_valid_select_results_20261111.tar.gz](../../../archives/20261111/f_threshold_valid_select/bundles/f_threshold_valid_select_results_20261111.tar.gz) | valid 上按 Coverage-nDCG@10 选择阈值，再冻结并测试；当时选中 0.0。 |
| `handcrafted_a` | [handcrafted_a_test_20261111.log](../../../archives/20261111/handcrafted_a/logs/handcrafted_a_test_20261111.log) | Handcrafted A（Zernike47）正式 test、独立指标核验与诊断测试记录。 |
| `handcrafted_a` | [handcrafted_a_test_results_20261111.tar.gz](../../../archives/20261111/handcrafted_a/bundles/handcrafted_a_test_results_20261111.tar.gz) | Handcrafted A（Zernike47）正式 test、独立指标核验与诊断测试记录。 |
| `handcrafted_a` | [handcrafted_a_verify_20261111.log](../../../archives/20261111/handcrafted_a/logs/handcrafted_a_verify_20261111.log) | Handcrafted A（Zernike47）正式 test、独立指标核验与诊断测试记录。 |
| `handcrafted_a` | [handcrafted_diagnostics_tests_20261111.log](../../../archives/20261111/handcrafted_a/logs/handcrafted_diagnostics_tests_20261111.log) | Handcrafted A（Zernike47）正式 test、独立指标核验与诊断测试记录。 |
| `f_handcrafted_distributions` | [plot_f_handcrafted_distributions_20261111.py](f_handcrafted_distributions/plot_f_handcrafted_distributions_20261111.py) | F 与 Handcrafted A 的区域及整图分数分布统计和绘图。 |
| `f_threshold_05` | [run_f_threshold_05_test_20261111.py](f_threshold_05/run_f_threshold_05_test_20261111.py) | 固定冻结 F 模型，阈值 0.5 的事后 test 对照、相似度审计与结果核验。 |
| `f_threshold_valid_select` | [run_f_threshold_valid_select_20261111.py](f_threshold_valid_select/run_f_threshold_valid_select_20261111.py) | valid 上按 Coverage-nDCG@10 选择阈值，再冻结并测试；当时选中 0.0。 |
| `handcrafted_a` | [run_handcrafted_a_test_20261111.py](handcrafted_a/run_handcrafted_a_test_20261111.py) | Handcrafted A（Zernike47）正式 test、独立指标核验与诊断测试记录。 |
| `f_threshold_05` | [verify_f_threshold_05_test_20261111.py](f_threshold_05/verify_f_threshold_05_test_20261111.py) | 固定冻结 F 模型，阈值 0.5 的事后 test 对照、相似度审计与结果核验。 |
| `f_threshold_valid_select` | [verify_f_threshold_valid_select_20261111.py](f_threshold_valid_select/verify_f_threshold_valid_select_20261111.py) | valid 上按 Coverage-nDCG@10 选择阈值，再冻结并测试；当时选中 0.0。 |
| `handcrafted_a` | [verify_handcrafted_a_test_20261111.py](handcrafted_a/verify_handcrafted_a_test_20261111.py) | Handcrafted A（Zernike47）正式 test、独立指标核验与诊断测试记录。 |

## 运行和保留约定

- Python 脚本通过原有命令行参数指定项目、来源运行、输出目录；迁移没有改变参数。仍应使用全新输出目录，遵守脚本已有的冻结来源和哈希检查。
- 这些 F 阈值/分布脚本属于早期 `formal_f_exact_20261109_132219` 实验，不是当前 cc8＋径向位置＋相对面积尺度主实验入口。
- 日志与压缩包只在服务器相应仓库的 `archives/20261111/<实验>/logs/` 或 `bundles/` 中归档，保留文件名、内容及 mtime；不要求将大文件下载到本机。归档目录被 Git 忽略。
- `relocation-manifest.json` 记录全部旧路径、新路径、原始 SHA-256、大小和 mtime；其中归档文件的相对链接仅在服务器存有归档的工作副本可用。
- `artifacts/` 内原有运行产物与脚本快照保持原位、原样；旧 manifest 中的绝对路径是当时运行的历史记录，不为本次目录整理改写。核验旧根目录脚本时，应根据路径映射定位保留的原件并核对哈希和原 mtime。
- 原始根目录脚本及小型哈希 JSON 保留在服务器 `.sync-backups/20261116-organize-root/original-scripts/`；仓库内另有完全相同的受 Git 管理副本。
- 忽略规则针对归档目录、压缩包、数组、缓存和权重；Git 忽略规则不按文件大小自动筛选。既有受跟踪的冻结协议和第三方测试资料不取消跟踪。
- 本次没有删除任何原文件，也没有重建协议、重训模型或重跑正式检索。
