# WBM 局部区域检索实验

当前主方法为**任意角旋转CNN＋新权重的结构增强区域检索**：`S=0.34C+0.51T+0.15A`，使用固定第100轮旋转模型与`add_a0p4_b0p15`。用户于2026-11-24查看test结果后指定；原F及旧CNN／旧权重保留为历史实验。

- [项目说明](doc/项目说明.md)：项目设计、公共数据与评价、运行、结果概览和登记。
- [论文实验证据总表](doc/论文实验证据总表.md)：当前主方法、历史证据与中英文稿件的对齐状态。
- [CNN改进实验记录](doc/CNN改进实验记录.md)：三轮改进、当前模型／权重来源及与WBBS的对照。
- [消融实验](doc/消融实验.md)：B1及WBBS融合的符号、实验设计、对比数据和结论。
- [F/G/WBBS融合](fgw_fusion/README.md)：三组固定公式、来源、运行与恢复。
- [F/G/WBBS网格搜索](fgw_grid_search/README.md)：166项完整valid网格已完成，主选公式为0.1F＋0.4G＋0.5W。
- [前十项test评价](fgw_topk_test/README.md)：冻结valid前十项的完整test已完成；结果按原valid顺序展示，不按test重新选方案。
- [前十项跨组合评价](fgw_cross_test/README.md)：十项已完成；排除同标签组合候选，复用原十组参数和coverage指标，与普通test分表展示。
- [F与F×WBBS跨组合评价](fw_cross_test/README.md)：两项已完成；同一候选集、同一最优匹配F与原coverage指标，比较单独F和原分数相乘。
- [F与WBBS等权加法](fw_add_cross_test/README.md)：已完成固定0.5F＋0.5WBBS；同一跨组合候选和原coverage指标，对比乘法的前排提升与mAP取舍。
- [B1改进简述](doc/B1改进简述.md)：计算步骤、J与WBBS的区别、五组最佳实验及结论的简要版。
- [自动结果总览](results_catalog/CURRENT_RESULTS.md)：完整成绩与来源。
- [实验代码目录](experiments/README.md)：各入口和专属参数。
- [协作规范](AGENTS.md)：时间型 run_id、历史保护与三端同步规则。

服务器：`/data/wangkaiyan2024/wbm-region-retrieval/`。数据、模型和原始实验产物保持原位，不随 Git 发布。
