# 冻结数据协议

这三个 CSV 从 `wbm-wdm-matching/artifacts/splits` 原样复制而来，`split.csv` 对应 `wm38k_seed2026_sig_70_10_20.csv`，`queries.csv` 对应 `wm38k_seed2026_test_queries_2000.csv`，`candidates.csv` 对应 `wm38k_seed2026_test_candidates_1000.csv`，它们是固定实验输入而不是新工程运行时重新抽样得到的结果。新工程只读取复制后的文件，远程运行无需原工程目录。

原来的 candidate 清单含有 `bucket`、`jaccard`、`query_signature` 和 `candidate_signature`，新工程只取 `query_id` 与 `candidate_id` 来确定候选身份，训练与评价需要的 containment 从原始 multi-hot 标签重新计算。因为候选池本身是在原协议中按标签关系抽样形成的，这个实验衡量固定的 1,000 候选协议，不等同于从全部 test 图或全部生产历史库检索。

| 文件 | SHA-256 |
| --- | --- |
| split.csv | e1785da387a70da63b65052975872da1377d15f690d4c37c5009840d08444cda |
| queries.csv | e2a03177a4aae071c7d7de85a041dab86ed97ea50d978e9a3b99693ff5f22e7d |
| candidates.csv | 675b18035fddf692f77da4a33b028792963e958514b6f1ecd060550b21193185 |

split 中 train、valid 和 test 分别有 25,910、3,700 和 7,405 个样本，query 清单有 2,000 个 test ID，每个 query 的 candidate 清单有 1,000 个不同的 test ID 并排除自身，程序在正式准备缓存前会校验这些约束和 NPZ 行号的一致性。
