# 去重划分与固定检索协议

2026-11-09 起，正式配置使用与 `compared_methods/protocol/` 逐字节一致的冻结清单。
本仓库的 `deduplicated/` 和 `retrieval_v1/` 是该协议的独立运行镜像，远程运行无需依赖
另一个仓库的位置。源清单及元数据不重建、不改写；加载时校验数据、划分和清单哈希。
元数据中的绝对路径记录生成时的来源，运行时以配置中的路径及文件内容哈希为准。
`wbm_retrieval/data_protocol.py` 与 `retrieval_protocol.py` 复用 `compared_methods/shared/`
的同名协议实现，便于两个仓库独立运行。

## 数据身份和去重

原始数据为 `../data/wm38k/Wafer_Map_Datasets.npz`，不覆盖、不重排、不重新编号。
`sample_id` 始终等于原始 NPZ 行号；`valid_index` 仍是原始非空标签序列中的索引，
不能拿去重清单的行号或 `valid_index` 代替 `sample_id`。

去重规则与 `compared_methods` 相同：

1. 对完整原图的所有 0/1/2/3 像素做精确比较，先用 SHA-256 定位候选，再逐像素确认。
2. 每组保留最小原始行 ID。同图不同标签报错，不按旋转、翻转、缺陷 mask 或近似外观合并。
3. 先去重再划分，按37种完整 multi-hot 组合分层，seed=2026，比例70%/10%/20%。

38,015 条原始记录中有1,023个重复组，移除1,024条冗余记录，保留36,991张唯一图：
35,991张缺陷图及1,000张 Normal。本项目的训练和检索协议只使用非空标签缺陷图，
Normal 不加入缺陷检索库，也不额外启动 Yu 检测器使用的 Normal 划分。

| 集合 | 去重后缺陷图数 |
| --- | ---: |
| train | 25,193 |
| valid | 3,599 |
| test | 7,199 |

标签列顺序固定为 center、donut、edge-loc、edge-ring、loc、**near-full**、scratch、**random**；
索引5和7分别对应 near-full 和 random。

## 固定 query 和完整 gallery

| 集合 | Query | Gallery | 每个 query 的候选 |
| --- | ---: | ---: | ---: |
| valid | 固定分层500张 | 完整 valid：3,599张 | 排除自身后的全部3,598张 |
| test | 全部7,199张 | 完整 test：7,199张 | 排除自身后的全部7,198张 |

valid query 按完整组合类别比例分配、每类至少5张（不足则全取），使用有约束的余数分配，
类内 seed=2026 无放回抽样。当前 Near-full 5张、Random 11张，其余35类中29类各14张、
6类各13张。清单已经冻结；更换训练 seed 不改变 query/gallery，也不在各套件运行时重抽。

候选规则为 `all_gallery_except_query`，query 之间可以互为候选。四份 ID 清单与该规则
完整确定候选身份。按 query 惰性生成候选，不把51,818,402条 test 配对同时装入内存。
运行目录中的 valid 池保存 query/gallery 和规则，而非逐 query 展开的候选副本。

| 文件 | 用途 |
| --- | --- |
| `deduplicated/split.csv` | 去重代表的冻结 train/valid/test 划分及原始 ID、标签 |
| `deduplicated/duplicates.csv` | 被过滤 ID 与保留代表的对应关系 |
| `deduplicated/metadata.json` | 原始数据哈希、去重规则、计数、划分和重复清单哈希 |
| `retrieval_v1/valid_queries.csv` | 固定500个 valid query |
| `retrieval_v1/valid_gallery.csv` | 完整 valid gallery |
| `retrieval_v1/test_queries.csv` | 全部 test query |
| `retrieval_v1/test_gallery.csv` | 完整 test gallery |
| `retrieval_v1/metadata.json` | 数据/split/四份清单哈希、配额和候选规则 |

主要身份哈希：

| 对象 | SHA-256 |
| --- | --- |
| 原始 NPZ | `a19e791c85ccd5051c080169a3d2b17902a42545cf749e9869f6c96312bcdc69` |
| 去重 `split.csv` | `2c1d259fec3e541adb9eed7d8042936f04c252d1ec8e4572add9f9ca3f2682d0` |
| `duplicates.csv` | `668a6fb6ac34790c0bf7490cdc11c4336d7e74402236e2b94869f28f752bb6f3` |

其余文件哈希直接记录在两份元数据中。协议缺失、内容改动、数据不匹配或样本归属不一致时
应停止运行，不生成替代清单。去重审计读取全源像素只用于验证样本身份，不表示使用 test
图像训练模型或对 test 做性能评价。

## 配置与运行隔离

正式基础配置以及原主实验/handcrafted A/supervised A 的独立配置包含：

```json
{
  "data_protocol": "deduplicated_v1",
  "split_manifest": "protocol/deduplicated/split.csv",
  "deduplication_metadata": "protocol/deduplicated/metadata.json",
  "retrieval_protocol_dir": "protocol/retrieval_v1"
}
```

A～G、handcrafted B 和 supervised B 从 `experiments/_shared/base.json` 继承协议。
相对路径均按本仓库根目录解释。正式默认输出统一放在 `artifacts/deduplicated_v1/` 下，
具体目录见 [实验总览](../experiments/README.md#7-运行示例)。新协议改变训练样本及数据签名，
必须重新训练；旧 checkpoint、缓存、预算 ID 和旧指标不能复用于新运行。E/F 引用 D 对照
时也必须使用同一新协议产生的运行目录。

本次仅迁移数据身份与检索候选池。模型、增强、区域匹配和评价公式沿用现有实现：
相关性仍为标签包含率，指标仍是 `ContainmentNDCG`、`FullCoverage*`、`mAP_full` 等，
剩余同分仍按 candidate ID 排序。它们尚未切换为 `compared_methods` 的 exact_v1 指标及
哈希破并列规则；两种指标不能直接混表。新协议下性能须由新训练及后续评价确定。

历史300图 MVP 配置显式标为 `data_protocol="legacy"`，源配置固定为迁移前基础配置的
独立副本 `experiments/_shared/legacy_mvp_source.json`，保留原派生数据及调试行为，不从新的
去重清单抽取并重新编号。原产物不覆写；源码或配置路径变化后不得修改旧签名强制续用，
重新检查需复制 MVP 配置并使用全新的输出及派生数据目录。
它不是新协议 smoke 或正式成绩；不要用 MVP 的 test 输出作为正式 test。若后续需要新协议
smoke，应从冻结 train/valid 中派生，valid query 取正式 query 的交集，不运行正式 test。

## 历史协议原样保留

根目录下的 `split.csv`、`queries.csv`、`candidates.csv` 保留原文件，仅用于历史追溯和
显式 legacy 调试。它们来自 `wbm-wdm-matching/artifacts/splits`，对应未去重的
25,910/3,700/7,405划分及2,000 query × 每 query 1,000候选。旧候选曾按标签关系分层抽样，
不等同于当前完整 gallery。

| 历史文件 | SHA-256 |
| --- | --- |
| `split.csv` | `e1785da387a70da63b65052975872da1377d15f690d4c37c5009840d08444cda` |
| `queries.csv` | `e2a03177a4aae071c7d7de85a041dab86ed97ea50d978e9a3b99693ff5f22e7d` |
| `candidates.csv` | `675b18035fddf692f77da4a33b028792963e958514b6f1ecd060550b21193185` |

旧训练集与新 valid/test 存在样本交叉。历史 A～G 指标及其离线 exact 重算仅描述旧验证池，
不能作为新划分的结果。本次配置与代码迁移未启动正式训练或完整 test。
