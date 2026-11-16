#!/usr/bin/env python3
"""Read-only full test report for radial F with relative-area scaling."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.report_f_radial_test import (KS, FIELDS, FILES, GROUPS, GROUP_LABELS,
    read_json, file_hash, fields, validate_report, extract_groups, difference,
    display, table, write_csv)

NEW = "f_radial_area"
RADIAL = "f_radial_beta1"
EXTERNAL = ("wu", "yu", "wbbs", "nakazawa", "wmhd", "wapirl")
PRIMARY_METHODS = (NEW, RADIAL, *EXTERNAL)
ALL_METHODS = (*PRIMARY_METHODS, "f")
LABELS = {NEW: "F＋径向＋相对面积", RADIAL: "F＋径向 β=1", "wu": "Wu", "yu": "Yu",
          "wbbs": "WBBS", "nakazawa": "Nakazawa CNN", "wmhd": "WMHD",
          "wapirl": "WaPIRL（F 对齐整图）", "f": "原 F（附加参考）"}
QUERY_FILES = {"exact_v1": "per_query.csv", "coverage_v1": "coverage_per_query.csv"}
BOOTSTRAP_FIELDS = (("exact_v1", "mAP@10"), ("exact_v1", "mAP@100"),
                    ("coverage_v1", "Coverage-nDCG@10"), ("coverage_v1", "FullCoverage_mAP@10"))


def raw_metric(field):
    return "AP@" + field.split("@")[1] if field.startswith("mAP@") else field.replace("FullCoverage_mAP@", "FullCoverageAP@")


def read_queries(directory, version):
    path = directory/QUERY_FILES[version]
    if not path.exists():
        return None
    result = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            qid = int(row["query_id"])
            if qid in result:
                raise ValueError(f"Duplicate query in {path}")
            item = {name: int(row[name]) for name in ("class_id", "label_count", "pool_size")}
            if item["pool_size"] != 7198:
                raise ValueError(f"Candidate count differs from full test: {path}")
            item["metrics"] = {}
            for field in fields(version):
                value = row[raw_metric(field)]
                value = None if value == "" else float(value)
                if value is not None and not math.isfinite(value):
                    raise ValueError(f"Nonfinite query metric in {path}")
                item["metrics"][field] = value
            result[qid] = item
    if len(result) != 7199:
        raise ValueError(f"Expected all 7,199 test queries in {path}")
    return result


def selected_rows(queries, group):
    values = list(queries.values())
    if group in ("all", "class_macro"):
        return values
    if group == "single_label":
        return [r for r in values if r["label_count"] == 1]
    if group == "mixed_label":
        return [r for r in values if r["label_count"] > 1]
    return [r for r in values if r["label_count"] == int(group.rsplit("_", 1)[1])]


def mean_defined(values):
    defined = [v for v in values if v is not None]
    return float(np.mean(defined)) if defined else None


def aggregate_queries(queries, version, group):
    rows = selected_rows(queries, group)
    classes = sorted({r["class_id"] for r in rows})
    metrics = {}
    for field in fields(version):
        if group == "class_macro":
            metrics[field] = mean_defined([mean_defined([r["metrics"][field] for r in rows if r["class_id"] == c])
                                          for c in classes])
        else:
            metrics[field] = mean_defined([r["metrics"][field] for r in rows])
    return {"num_queries": len(rows), "num_classes": len(classes), "metrics": metrics,
            "computed_from_per_query": True}


def paired_comparisons(queries):
    comparisons = []
    for group in ("all", "single_label", "mixed_label"):
        for version, metric in BOOTSTRAP_FIELDS:
            a, b = queries[RADIAL][version], queries[NEW][version]
            ids = [q for q in sorted(a) if group == "all" or
                   (a[q]["label_count"] == 1 if group == "single_label" else a[q]["label_count"] > 1)]
            diffs = np.asarray([b[q]["metrics"][metric]-a[q]["metrics"][metric] for q in ids
                                if a[q]["metrics"][metric] is not None and b[q]["metrics"][metric] is not None])
            if not len(diffs):
                raise ValueError(f"No defined paired metrics: {group}/{metric}")
            rng = np.random.default_rng(2026)
            means = np.empty(2000)
            for start in range(0, 2000, 100):
                sample = rng.integers(0, len(diffs), size=(min(100, 2000-start), len(diffs)))
                means[start:start+len(sample)] = diffs[sample].mean(axis=1)
            low, high = np.quantile(means, [.025, .975])
            comparisons.append({"comparison": "radial_area_minus_radial", "metric_version": version,
                "group": group, "metric": metric, "queries": len(ids), "paired_queries": len(diffs),
                "mean_delta": float(diffs.mean()), "delta_percentage_points": float(diffs.mean()*100),
                "bootstrap_95ci_low": float(low), "bootstrap_95ci_high": float(high),
                "bootstrap_95ci_low_percentage_points": float(low*100), "bootstrap_95ci_high_percentage_points": float(high*100),
                "improved_queries": int((diffs > 1e-12).sum()), "worsened_queries": int((diffs < -1e-12).sum()),
                "tied_queries": int((abs(diffs) <= 1e-12).sum()), "repetitions": 2000, "seed": 2026})
    return comparisons


def ranking(groups, method, version, group, metric):
    available = {m: groups[m][version].get(group, {}).get("metrics", {}).get(metric) for m in PRIMARY_METHODS}
    available = {m: v for m, v in available.items() if v is not None}
    if method not in available:
        return None, len(available)
    # Competition ranks: exact ties share rank; no tie-breaking on rounded values.
    return 1+sum(v > available[method] for v in available.values()), len(available)


def generate(result_dir, radial_dir, references, output):
    result_dir, radial_dir, references, output = [Path(p).resolve() for p in (result_dir, radial_dir, references, output)]
    if output.exists():
        raise FileExistsError(f"Choose a new report directory: {output}")
    if any(output == p or p in output.parents for p in (result_dir, radial_dir, references)):
        raise ValueError("Report output must be separate from input test/reference directories")
    reference_manifest = read_json(references/"source_manifest.json")
    protocol = reference_manifest["retrieval_protocol_signature"]
    if reference_manifest.get("num_queries") != 7199 or reference_manifest.get("candidates_per_query") != 7198:
        raise ValueError("Reference manifest does not describe the full test protocol")
    with (ROOT/"protocol/retrieval_v1/test_queries.csv").open(newline="") as handle:
        frozen_queries = {int(r["sample_id"]) for r in csv.DictReader(handle)}
    with (ROOT/"protocol/retrieval_v1/test_gallery.csv").open(newline="") as handle:
        frozen_gallery = {int(r["sample_id"]) for r in csv.DictReader(handle)}
    if len(frozen_queries) != 7199 or frozen_queries != frozen_gallery:
        raise ValueError("Project test pool differs from full-gallery leave-one-out protocol")
    reports, grouped, sources, queries = {}, {}, {}, {}
    reconstructed, missing = [], []
    identity = None
    aggregation_error = 0.0
    for method in ALL_METHODS:
        directory = result_dir if method == NEW else radial_dir if method == RADIAL else references/method
        reports[method], grouped[method], sources[method], queries[method] = {}, {}, {}, {}
        for version, filename in FILES.items():
            path = directory/filename
            data = read_json(path)
            validate_report(data, version, path)
            digest = file_hash(path)
            declared_protocol = data.get("retrieval_protocol_signature") or data.get("metadata", {}).get("retrieval_protocol_signature")
            if version == "exact_v1" and declared_protocol != protocol:
                raise ValueError(f"Exact protocol signature mismatch: {path}")
            if declared_protocol is not None and declared_protocol != protocol:
                raise ValueError(f"Protocol signature mismatch: {path}")
            if method not in (NEW, RADIAL):
                label = "exact" if version == "exact_v1" else "coverage"
                if digest != reference_manifest["methods"][method][label]["sha256"]:
                    raise ValueError(f"Reference report changed: {path}")
            reports[method][version] = data
            grouped[method][version] = extract_groups(data, directory, version)
            source = {"path": str(path), "sha256": digest}
            query_rows = read_queries(directory, version)
            queries[method][version] = query_rows
            if query_rows is not None:
                if set(query_rows) != frozen_queries:
                    raise ValueError(f"Query IDs differ from frozen test: {directory}")
                query_identity = {q: (r["class_id"], r["label_count"], r["pool_size"]) for q, r in query_rows.items()}
                if method in (NEW, RADIAL):
                    identity = identity or query_identity
                    if identity != query_identity:
                        raise ValueError("Radial/area exact/coverage query identities differ")
                source["per_query_path"] = str(directory/QUERY_FILES[version])
                source["per_query_sha256"] = file_hash(directory/QUERY_FILES[version])
            elif method in (NEW, RADIAL):
                raise FileNotFoundError(f"Paired evaluation requires {directory/QUERY_FILES[version]}")
            for group in GROUPS:
                if group not in grouped[method][version]:
                    if query_rows is not None:
                        grouped[method][version][group] = aggregate_queries(query_rows, version, group)
                        reconstructed.append(f"{method}/{version}/{group}")
                    else:
                        missing.append(f"{method}/{version}/{group}")
                        continue
                group_metrics = grouped[method][version][group]["metrics"]
                if not set(fields(version)) <= set(group_metrics):
                    raise ValueError(f"Incomplete {group} metrics in {path}")
                if method in (NEW, RADIAL):
                    checked = aggregate_queries(query_rows, version, group)["metrics"]
                    for field, value in checked.items():
                        saved = group_metrics[field]
                        if value is None or saved is None:
                            if value != saved:
                                raise ValueError(f"Undefined value mismatch: {method}/{version}/{group}/{field}")
                        else:
                            error = abs(value-saved)
                            aggregation_error = max(aggregation_error, error)
                            if error > 1e-12:
                                raise ValueError(f"Query aggregate mismatch: {method}/{version}/{group}/{field}")
            sources[method][version] = source

    metric_rows, comparisons = [], []
    for method in ALL_METHODS:
        for version in FILES:
            for group, group_values in grouped[method][version].items():
                baseline = grouped[RADIAL][version].get(group, {}).get("metrics", {})
                for metric in fields(version):
                    value = group_values["metrics"].get(metric)
                    rank, count = ranking(grouped, method, version, group, metric)
                    metric_rows.append({"method": method, "method_label": LABELS[method],
                        "primary_comparison": method in PRIMARY_METHODS, "metric_version": version, "group": group,
                        "num_queries": group_values.get("num_queries"), "num_classes": group_values.get("num_classes"),
                        "metric": metric, "k": int(metric.split("@")[1]), "value": value,
                        "radial_value": baseline.get(metric), **difference(value, baseline.get(metric)),
                        "rank_among_primary_methods": rank, "primary_methods_with_metric": count})
    for other in (*PRIMARY_METHODS[1:], "f"):
        for version in FILES:
            for group, values in grouped[NEW][version].items():
                reference = grouped[other][version].get(group, {}).get("metrics", {})
                for metric in fields(version):
                    value, base = values["metrics"].get(metric), reference.get(metric)
                    comparisons.append({"new_method": NEW, "reference_method": other, "metric_version": version,
                                        "group": group, "metric": metric, "new_value": value, "reference_value": base,
                                        **difference(value, base)})
    paired = paired_comparisons(queries)
    audit_sources = {}
    for directory_label, directory in (("new_run", result_dir.parent), ("radial_run", radial_dir.parent)):
        audit_sources[directory_label] = {}
        for name in ("selection_before_test.json", "plan_before_evaluation.json", "completed.json", "independent_verification.json"):
            path = directory/name
            if path.exists():
                audit_sources[directory_label][name] = {"path": str(path), "sha256": file_hash(path), "contents": read_json(path)}
    selection = audit_sources["new_run"].get("selection_before_test.json", {}).get("contents")

    md = ["# F＋径向＋相对面积：完整 test 对照报告", "",
          "主比较集合为八方法：新面积尺度F、当前径向F及Wu、Yu、WBBS、Nakazawa CNN、WMHD、WaPIRL。"
          "最初的原F仅作为附加参考，不参与八方法排名。", "",
          "## 固定方案与评价口径", "",
          "本轮采用此前valid验证后固定的径向＋相对面积方案：β=1、径向16-bin、σ_s=1.5、λ_s=0.15。"
          "相对面积r=A_region/A_valid，尺度距离D_area=0.5×|log(r_q/r_c)|；"
          "区域对分数为S_radial×[0.85+0.15×exp(-D_area/1.5)]。沿用同一F best_nce、h512、区域和单向面积加权匹配，"
          "没有四旋转汇聚、双向评分、长短轴尺度或test参数搜索。具体来源及冻结参数文件见末尾审计。", "",
          "test含7,199个查询和37种完整标签组合，每个query评价其余7,198张完整候选。两组指标使用同一份新test排名，"
          "只保存Top500，不把它称为全部候选的完整排名文件。K为1、5、10、20、50、100、500。",
          "exact_v1将完整标签组合相同定义为相关；coverage_v1采用查询类别覆盖比例的线性gain，"
          "FullCoverage定义为Lq⊆Lc。两组AP/Recall均使用完整候选库中的相应相关数作分母。",
          f"协议签名：`{protocol}`。Markdown显示9位小数，CSV/JSON保留原始浮点数。", "",
          "参数在valid上确定，不能称全过程不使用标签；方法间训练监督不同，单次固定checkpoint排名不等于跨seed结论。"
          "下列主表排名按未舍入数值计算，相同值共享最小名次；原F附加参考不计入排名。", "",
          "## 新面积尺度F：全部指标与分组", ""]
    for group in GROUPS:
        md += [f"### {GROUP_LABELS[group]}", ""]
        for version in FILES:
            if group not in grouped[NEW][version]:
                md += [f"{version}：原报告及逐query文件未提供该组，未补造数值。", ""]
                continue
            values = grouped[NEW][version][group]
            md += [f"**{version}**：底层query数{values.get('num_queries')}，完整组合类别数{values.get('num_classes')}。", ""]
            if group == "class_macro":
                md += ["以下对完整组合类别等权平均，而非对所有query直接平均。", ""]
            md += table(["K", *FIELDS[version]], [[str(k), *[display(values['metrics'].get(f'{f}@{k}')) for f in FIELDS[version]]]
                                                   for k in KS])

    md += ["## 八方法：全部总体指标并列", "", "以下均为query平均，共70项；原F附加参考另列。", ""]
    for version in FILES:
        md += [f"### {version}", ""]
        md += table(["指标", *[LABELS[m] for m in PRIMARY_METHODS]],
                    [[field, *[display(reports[m][version]['metrics'].get(field)) for m in PRIMARY_METHODS]] for field in fields(version)])
    md += ["## 新方法的八方法名次与逐方法差值", "",
           "下表显示主要指标的总体比较；所有K、所有分组和全部70项的名次与差值见CSV/JSON。", ""]
    keys = (("exact_v1", "mAP@10"), ("exact_v1", "mAP@100"), ("coverage_v1", "Coverage-nDCG@10"),
            ("coverage_v1", "FullCoverage_mAP@10"), ("coverage_v1", "FullCoverageRecall@500"))
    md += table(["指标", "新面积F", "八方法名次", "径向F", "差值（百分点）"],
                [[field, display(reports[NEW][version]['metrics'][field]), str(ranking(grouped, NEW, version, 'all', field)[0]),
                  display(reports[RADIAL][version]['metrics'][field]),
                  display(100*(reports[NEW][version]['metrics'][field]-reports[RADIAL][version]['metrics'][field]), True)]
                 for version, field in keys])
    md += table(["对比对象", "exact mAP@10差值（百分点）", "Coverage-nDCG@10差值（百分点）", "70项高/平/低"],
                [[LABELS[m], display(100*(reports[NEW]['exact_v1']['metrics']['mAP@10']-reports[m]['exact_v1']['metrics']['mAP@10']), True),
                  display(100*(reports[NEW]['coverage_v1']['metrics']['Coverage-nDCG@10']-reports[m]['coverage_v1']['metrics']['Coverage-nDCG@10']), True),
                  "/".join(str(sum((reports[NEW][v]['metrics'][f] > reports[m][v]['metrics'][f]) if direction == 'high' else
                                      (reports[NEW][v]['metrics'][f] < reports[m][v]['metrics'][f]) if direction == 'low' else
                                      (reports[NEW][v]['metrics'][f] == reports[m][v]['metrics'][f])
                                      for v in FILES for f in fields(v))) for direction in ('high','equal','low'))]
                 for m in PRIMARY_METHODS[1:]])
    md += ["## 新面积F相对径向F：配对bootstrap", "",
           "按同一query ID对齐两组逐query CSV，对差值有放回重抽样2,000次，seed=2026，报告百分位95%区间。"
           "区间只反映当前固定test查询及对应单/混合子组的配对差异，不含跨训练seed、valid方案选择的不确定性，"
           "且未作多重比较校正。没有对六个外部方法进行配对显著性检验。", ""]
    md += table(["分组", "指标", "配对query数", "面积−径向", "95%区间", "改善/下降/相同query数"],
                [[GROUP_LABELS[r['group']], r['metric'], str(r['paired_queries']), display(r['mean_delta'], True),
                  f"[{display(r['bootstrap_95ci_low'], True)}, {display(r['bootstrap_95ci_high'], True)}]",
                  f"{r['improved_queries']}/{r['worsened_queries']}/{r['tied_queries']}"] for r in paired])
    md += ["## 附加参考：最初的原F", "", "本表不参与主比较排名。所有差值为新面积尺度F−最初原F。", ""]
    for version in FILES:
        md += [f"### {version}", ""]
        md += table(["指标", "原F", "新面积F", "差值（百分点）"],
                    [[f, display(reports['f'][version]['metrics'][f]), display(reports[NEW][version]['metrics'][f]),
                      display(100*(reports[NEW][version]['metrics'][f]-reports['f'][version]['metrics'][f]), True)] for f in fields(version)])
    md += ["## 来源、完整数据与核查", "",
           f"- 新test：`{result_dir}`。",
           f"- 当前径向F：`{radial_dir}`。",
           f"- 原方法参考：`{references}`，已逐文件匹配source_manifest.json中的SHA。",
           "- `overall_metrics.csv`：八主方法＋原F附加参考×70总体指标；含vs径向差值及八方法名次。",
           "- `grouped_metrics.csv`：全部八分组的完整指标长表。",
           "- `new_vs_methods.csv`：新面积F对径向、六外部及原F附加参考的全部指标/分组差值。",
           "- `paired_bootstrap.csv/json`：新面积F与径向F的12项主要配对比较。",
           "- `report.json`：全部原始汇总、分组、排名、差值、冻结/完成/审计内容。",
           "- `source_hashes.json`：输入、生成器、复用helper和审计来源哈希。",
           f"- 新面积F/径向F全部组从逐query CSV重新聚合核对的最大误差：`{aggregation_error}`。",
           f"- 从CSV补出的缺失组：`{json.dumps(reconstructed, ensure_ascii=False)}`。",
           f"- 无数据因而保留缺失的组：`{json.dumps(missing, ensure_ascii=False)}`。", "",
           "所有原始文件只读。参考文档为项目`doc/EXACT_METRICS_MIGRATION.md`，本生成器不改写该文档或替换既有方法。", ""]
    if selection is not None:
        md += ["### 新方案test前冻结记录", "", "```json", json.dumps(selection, ensure_ascii=False, indent=2), "```", ""]

    output.mkdir(parents=True, exist_ok=False)
    write_csv(output/"overall_metrics.csv", [r for r in metric_rows if r['group'] == 'all'])
    write_csv(output/"grouped_metrics.csv", metric_rows)
    write_csv(output/"new_vs_methods.csv", comparisons)
    write_csv(output/"paired_bootstrap.csv", paired)
    payload = {"primary_methods": list(PRIMARY_METHODS), "additional_reference": "f", "ks": KS,
               "retrieval_protocol_signature": protocol, "raw_reports": reports, "groups": grouped,
               "metric_rows": metric_rows, "new_vs_methods": comparisons, "paired_bootstrap": paired,
               "sources": sources, "audit_sources": audit_sources, "aggregation_max_error": aggregation_error,
               "reconstructed_groups": reconstructed, "missing_groups": missing}
    for name, value in (("report.json", payload), ("paired_bootstrap.json", paired)):
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (output/"report.md").write_text("\n".join(md), encoding="utf-8")
    hashes = {"inputs": sources, "reference_manifest": {"path": str(references/'source_manifest.json'),
              "sha256": file_hash(references/'source_manifest.json')},
              "generator": {"path": str(Path(__file__).resolve()), "sha256": file_hash(Path(__file__))},
              "helper": {"path": str(ROOT/'scripts/report_f_radial_test.py'), "sha256": file_hash(ROOT/'scripts/report_f_radial_test.py')},
              "audit_sources": {k: {name: {f: value for f, value in entry.items() if f != 'contents'} for name, entry in v.items()}
                                for k, v in audit_sources.items()}}
    (output/"source_hashes.json").write_text(json.dumps(hashes, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return {"output_dir": str(output), "overall_rows": sum(r['group'] == 'all' for r in metric_rows),
            "grouped_rows": len(metric_rows), "comparison_rows": len(comparisons), "paired_comparisons": len(paired),
            "aggregation_max_error": aggregation_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--radial-dir", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.result_dir, args.radial_dir, args.references, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
