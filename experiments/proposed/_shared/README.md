# Proposed 共用实现

`abcd_suite.py` / `abcd_training.py` 由 A～D 共用；`ef_suite.py` / `ef_training.py` 由 D(EF)、E、F 共用。各字母目录的 `run.py` 绑定变体和配置，避免复制训练循环。

共享CLI为 `abcd_cli.py` 和 `ef_cli.py`，支持 `--variants` 批量运行。所有模块直接从新包路径导入，旧 `wbm_retrieval.ablation*` 模块及兼容链接已删除。源码指纹由 `experiments/_shared/provenance.py` 根据当前真实依赖生成；历史旧签名不能直接续跑。

完整设计和运行方式见 [实验总览](../../README.md)。
