# WBM 局部区域检索实验

当前主方法为 F：cc8 八连通＋径向位置＋相对面积；融合与组合为扩展实验。

- [项目说明](doc/项目说明.md)：项目设计、公共数据与评价、运行、结果概览和登记。
- [消融实验](doc/消融实验.md)：B1及WBBS融合的符号、实验设计、对比数据和结论。
- [F/G/WBBS融合](fgw_fusion/README.md)：三组固定公式、来源、运行与恢复。
- [F/G/WBBS网格搜索](fgw_grid_search/README.md)：166项完整valid网格已完成，主选公式为0.1F＋0.4G＋0.5W。
- [前十项test评价](fgw_topk_test/README.md)：冻结valid前十项的完整test已完成；结果按原valid顺序展示，不按test重新选方案。
- [前十项跨组合评价](fgw_cross_test/README.md)：排除同标签组合候选，复用原十组参数和coverage指标。
- [B1改进简述](doc/B1改进简述.md)：计算步骤、J与WBBS的区别、五组最佳实验及结论的简要版。
- [自动结果总览](results_catalog/CURRENT_RESULTS.md)：完整成绩与来源。
- [实验代码目录](experiments/README.md)：各入口和专属参数。
- [协作规范](AGENTS.md)：时间型 run_id、历史保护与三端同步规则。

服务器：`/data/wangkaiyan2024/wbm-region-retrieval/`。数据、模型和原始实验产物保持原位，不随 Git 发布。
