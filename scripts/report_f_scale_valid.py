#!/usr/bin/env python3
"""Report the three frozen valid radial/scale variants without rescoring."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("radial", "radial_area", "radial_axes")
LABELS = {"radial": "径向基线", "radial_area": "径向＋面积尺度", "radial_axes": "径向＋长短轴尺度"}
KS = (1, 5, 10, 20, 50, 100, 500)
VERSIONS = ("exact_v1", "coverage_v1")
REPORT_FILES = {"exact_v1": "metrics.json", "coverage_v1": "coverage_metrics.json"}
QUERY_FILES = {"exact_v1": "per_query.csv", "coverage_v1": "coverage_per_query.csv"}
METRICS = {"exact_v1": ("Hit", "Precision", "Recall", "mAP"),
           "coverage_v1": ("Coverage-nDCG", "MeanCoverage", "FullCoverageHit",
                           "FullCoveragePrecision", "FullCoverageRecall", "FullCoverage_mAP")}
GROUPS = ("all", "class_macro", "single_label", "mixed_label", "label_count_1",
          "label_count_2", "label_count_3", "label_count_4")
GROUP_LABELS = {"all": "全部查询平均", "class_macro": "完整组合类别宏平均",
                "single_label": "单缺陷", "mixed_label": "混合缺陷",
                **{f"label_count_{i}": f"{i} 类缺陷" for i in range(1, 5)}}
PRIMARY = (("exact_v1", "mAP@10"), ("exact_v1", "mAP@100"),
           ("coverage_v1", "Coverage-nDCG@10"), ("coverage_v1", "FullCoverage_mAP@10"))


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def metric_fields(version):
    return [f"{field}@{k}" for field in METRICS[version] for k in KS]


def raw_field(field):
    if field.startswith("mAP@"):
        return "AP@" + field.split("@")[1]
    return field.replace("FullCoverage_mAP@", "FullCoverageAP@")


def finite_or_none(value):
    if value in (None, ""):
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite metric in source report")
    return number


def load_queries(path, version):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        qid = int(row["query_id"])
        if qid in result:
            raise ValueError(f"Duplicate query ID in {path}")
        result[qid] = {name: int(row[name]) for name in ("class_id", "label_count", "pool_size")}
        result[qid]["metrics"] = {field: finite_or_none(row[raw_field(field)]) for field in metric_fields(version)}
        if int(row["pool_size"]) != 3598:
            raise ValueError(f"Unexpected candidate pool size in {path}")
    if len(result) != 500:
        raise ValueError(f"Expected exactly 500 valid queries in {path}")
    return result


def signature(report):
    return (report.get("retrieval_protocol_signature")
            or report.get("metadata", {}).get("retrieval_protocol_signature")
            or report.get("metadata", {}).get("protocol_signature"))


def choose_rows(queries, group):
    if group in ("all", "class_macro"):
        return list(queries.values())
    if group == "single_label":
        return [r for r in queries.values() if r["label_count"] == 1]
    if group == "mixed_label":
        return [r for r in queries.values() if r["label_count"] > 1]
    count = int(group.rsplit("_", 1)[1])
    return [r for r in queries.values() if r["label_count"] == count]


def mean_defined(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def aggregate_queries(queries, version, group):
    selected = choose_rows(queries, group)
    classes = sorted({r["class_id"] for r in selected})
    metrics = {}
    for field in metric_fields(version):
        if group == "class_macro":
            metrics[field] = mean_defined([mean_defined([r["metrics"][field] for r in selected if r["class_id"] == c])
                                          for c in classes])
        else:
            metrics[field] = mean_defined([r["metrics"][field] for r in selected])
    return {"num_queries": len(selected), "num_classes": len(classes), "metrics": metrics}


def saved_group(report, group):
    if group == "all":
        return report.get("metrics")
    if group == "class_macro":
        return report.get("class_macro_metrics")
    groups = report.get("groups", {})
    if group.startswith("label_count_"):
        return groups.get("by_label_count", {}).get(group.rsplit("_", 1)[1], {}).get("metrics")
    return groups.get(group, {}).get("metrics")


def delta(value, base):
    if value is None or base is None:
        return {"delta": None, "delta_percentage_points": None, "relative_change": None}
    change = value - base
    return {"delta": change, "delta_percentage_points": 100 * change,
            "relative_change": change / base if base != 0 else None}


def paired_bootstrap(queries):
    """Paired query resampling; same deterministic draws for equal-sized groups."""
    output = []
    for method in METHODS[1:]:
        for group in ("all", "single_label", "mixed_label"):
            for version, field in PRIMARY:
                baseline = queries["radial"][version]
                candidate = queries[method][version]
                ids = [qid for qid in sorted(baseline)
                       if (group == "all" or (baseline[qid]["label_count"] == 1 if group == "single_label"
                                              else baseline[qid]["label_count"] > 1))]
                pairs = [(baseline[q]["metrics"][field], candidate[q]["metrics"][field]) for q in ids]
                diffs = np.asarray([b-a for a, b in pairs if a is not None and b is not None], dtype=float)
                if not len(diffs):
                    raise ValueError(f"No paired queries for {method}/{group}/{field}")
                rng = np.random.default_rng(2026)
                sample_means = np.empty(2000)
                for repetition in range(2000):
                    sample_means[repetition] = rng.choice(diffs, len(diffs), replace=True).mean()
                lower, upper = np.quantile(sample_means, [.025, .975])
                output.append({"method": method, "baseline": "radial", "metric_version": version, "group": group,
                               "metric": field, "query_count": len(ids), "paired_query_count": len(diffs),
                               "mean_delta": float(diffs.mean()), "delta_percentage_points": float(diffs.mean()*100),
                               "bootstrap_95ci_low": float(lower), "bootstrap_95ci_high": float(upper),
                               "bootstrap_95ci_low_percentage_points": float(lower*100),
                               "bootstrap_95ci_high_percentage_points": float(upper*100),
                               "improved_queries": int((diffs > 1e-12).sum()), "worsened_queries": int((diffs < -1e-12).sum()),
                               "tied_queries": int((np.abs(diffs) <= 1e-12).sum()), "repetitions": 2000, "seed": 2026})
    return output


def number(value, signed=False):
    return "未定义" if value is None else format(value, "+.9f" if signed else ".9f")


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows], ""]


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate(run_dir, output):
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh report directory: {output}")
    if output == run_dir or any((run_dir/m) == output or (run_dir/m) in output.parents for m in METHODS):
        raise ValueError("Output may be a new report subdirectory, not a source method directory")
    plan_paths = [run_dir/n for n in ("plan_before_evaluation.json", "run_plan.json", "plan.json") if (run_dir/n).exists()]
    if not plan_paths:
        raise FileNotFoundError("A frozen run plan is required before reporting")
    plans = {p.name: load_json(p) for p in plan_paths}
    primary_plan = plans[plan_paths[0].name]
    reports, queries, grouped, hashes = {}, {}, {}, {}
    expected_signature = None
    reference_identity = None
    aggregation_max_error = 0.0
    reconstructed_groups = []
    with (ROOT/"protocol/retrieval_v1/valid_queries.csv").open(newline="") as handle:
        frozen_ids = {int(r["sample_id"]) for r in csv.DictReader(handle)}
    for method in METHODS:
        reports[method], queries[method], grouped[method], hashes[method] = {}, {}, {}, {}
        for version in VERSIONS:
            directory = run_dir/method
            path = directory/REPORT_FILES[version]
            report = load_json(path)
            if (report.get("metric_version") != version or report.get("split") != "valid"
                    or report.get("test_evaluated") is not False or report.get("num_queries") != 500
                    or report.get("num_classes") != 37 or report.get("ks") != list(KS)):
                raise ValueError(f"Expected a complete valid-only report: {path}")
            if not set(metric_fields(version)) <= set(report.get("metrics", {})):
                raise ValueError(f"Missing metrics: {path}")
            current_signature = signature(report)
            if not current_signature:
                raise ValueError(f"Missing protocol signature in {path}")
            expected_signature = expected_signature or current_signature
            if expected_signature != current_signature:
                raise ValueError(f"Protocol signatures differ: {path}")
            rows_path = directory/QUERY_FILES[version]
            rows = load_queries(rows_path, version)
            if set(rows) != frozen_ids:
                raise ValueError(f"Query IDs differ from frozen valid list: {rows_path}")
            identity = {q: (v["class_id"], v["label_count"], v["pool_size"]) for q, v in rows.items()}
            reference_identity = reference_identity or identity
            if identity != reference_identity:
                raise ValueError(f"Query identity/label count differs: {rows_path}")
            queries[method][version] = rows
            reports[method][version] = report
            grouped[method][version] = {}
            hashes[method][version] = {"report": {"path": str(path), "sha256": sha256(path)},
                                       "per_query": {"path": str(rows_path), "sha256": sha256(rows_path)}}
            for group in GROUPS:
                computed = aggregate_queries(rows, version, group)
                original = saved_group(report, group)
                if original is None:
                    reconstructed_groups.append(f"{method}/{version}/{group}")
                else:
                    for field in metric_fields(version):
                        value, saved = computed["metrics"][field], finite_or_none(original[field])
                        if value is None or saved is None:
                            if value != saved:
                                raise ValueError(f"Undefined-metric mismatch: {method}/{group}/{field}")
                        else:
                            error = abs(value-saved)
                            aggregation_max_error = max(aggregation_max_error, error)
                            if error > 1e-12:
                                raise ValueError(f"Per-query report does not reproduce summary: {method}/{group}/{field}: {error}")
                grouped[method][version][group] = computed
            for filename in ("per_class.csv", "coverage_per_class.csv", "rankings.csv"):
                p = directory/filename
                if p.exists():
                    hashes[method].setdefault("additional", {})[filename] = {"path": str(p), "sha256": sha256(p)}

    long_rows = []
    for method in METHODS:
        for version in VERSIONS:
            for group in GROUPS:
                selected = grouped[method][version][group]
                base = grouped["radial"][version][group]["metrics"]
                for field in metric_fields(version):
                    value = selected["metrics"][field]
                    long_rows.append({"method": method, "method_label": LABELS[method], "metric_version": version,
                                      "group": group, "num_queries": selected["num_queries"], "num_classes": selected["num_classes"],
                                      "metric": field, "k": int(field.split("@")[1]), "value": value,
                                      "radial_baseline": base[field], **delta(value, base[field])})
    paired = paired_bootstrap(queries)
    audits = {}
    for path in sorted(run_dir.glob("*.json")):
        if path not in plan_paths:
            audits[path.name] = {"path": str(path), "sha256": sha256(path)}

    md = ["# F 径向＋尺度约束：完整 valid 对照报告", "",
          "本报告仅汇总已完成的三组固定方案，不运行训练、不重新评分或选择参数。"
          "三组为径向基线、径向＋面积尺度、径向＋长短轴尺度；具体几何定义和参数以本轮冻结方案为准。", "",
          "## 协议与指标", "",
          "同一份固定valid：500个查询，完整gallery 3,599张，每个query排除自身后3,598个候选。"
          "K固定为1、5、10、20、50、100、500。exact_v1为完整标签组合相同；coverage_v1为查询类别覆盖，"
          "FullCoverage条件为查询标签集合包含于候选标签集合。AP/Recall分母来自完整候选库。",
          "两组共70项指标：exact 28项，coverage 42项。每项均报告全部查询、类别宏平均、单/混合缺陷及1–4类缺陷分组。"
          "类别宏平均对完整组合类别等权，其余表为query平均。单缺陷与1类分组是同一集合，仍按约定分别列出。",
          f"协议签名：`{expected_signature}`。全表显示9位小数，CSV/JSON保留原始浮点精度。", "",
          "## 冻结方案", "",
          "以下直接读取本轮方案文件；轴归一化等实现约定不从旧实验推断。", "",
          f"三组固定β={primary_plan.get('beta')}；两个尺度组共用σ_s={primary_plan.get('sigma_scale')}、"
          f"λ_s={primary_plan.get('scale_weight')}，长短轴距离权重为{primary_plan.get('axis_weights')}。"
          "这些值在评价前固定，本轮不做参数搜索。", "",
          "面积组将相对面积开平方后取对数比差；长短轴组在完整原始区域的总体协方差特征值上加1，"
          "开平方后除以每图有效支撑面积的平方根，再比较长短轴的对数比差。有效支撑面积逐图计算，不固定为常数。",
          "两种描述统一为名义线性尺度，使用相同σ_s和λ_s；但轴上的+1平滑让细小/细长区域并非严格等响应，"
          "不能宣称二者具有完全相同的几何尺度等变性。",
          "本轮在径向分数上乘以尺度软因子，具体公式如下；这不是原PDF的三项加权和。", ""]
    for name, plan in plans.items():
        visible_keys = ("methods", "beta", "bins", "sigma_scale", "scale_weight", "axis_weights", "scale_definitions",
                        "rotation_pooling", "bidirectional", "hyperparameter_search", "aggregation", "selection")
        visible_plan = {key: plan[key] for key in visible_keys if key in plan}
        md += [f"### {name}", "", f"来源：`{run_dir/name}`；完整方案另存于report.json。", "", "```json",
               json.dumps(visible_plan, ensure_ascii=False, indent=2), "```", ""]
    md += ["## 三方法完整并列指标", ""]
    for group in GROUPS:
        counts = grouped["radial"]["exact_v1"][group]
        md += [f"### {GROUP_LABELS[group]}", "",
               f"底层查询数：{counts['num_queries']}；完整组合类别数：{counts['num_classes']}。", ""]
        for version in VERSIONS:
            md += [f"**{version}**", ""]
            rows = []
            for field in metric_fields(version):
                rows.append([field, *[number(grouped[m][version][group]['metrics'][field]) for m in METHODS]])
            md += table(["指标", *[LABELS[m] for m in METHODS]], rows)
    md += ["## 相对径向基线的总体变化", "", "单元格是比例差×100后的百分点变化；完整绝对差与相对变化另见CSV。", ""]
    for version in VERSIONS:
        md += [f"### {version}", ""]
        md += table(["指标", "面积尺度−径向（百分点）", "长短轴尺度−径向（百分点）"],
                    [[field, *[number(100*(grouped[m][version]['all']['metrics'][field]-grouped['radial'][version]['all']['metrics'][field]), True)
                               if grouped[m][version]['all']['metrics'][field] is not None and grouped['radial'][version]['all']['metrics'][field] is not None
                               else '未定义' for m in METHODS[1:]]] for field in metric_fields(version)])
    md += ["## 配对 bootstrap", "",
           "按相同query ID对齐原始CSV，对每个查询的“尺度版本−径向基线”差值进行2,000次有放回重抽样，"
           "seed=2026，报告均值差的百分位95%区间。重抽样单位为query，不是候选配对。",
           "区间仅描述这组固定500查询（或对应单/混合子组）的配对差异；不包含跨训练seed、checkpoint选择或"
           "参数搜索的不确定性，也未进行多重比较校正。不能将此valid内区间当成独立test泛化结论。", ""]
    md += table(["对照", "分组", "指标", "配对query数", "均值差", "95%区间", "改善/下降/相同query数"],
                [[LABELS[r['method']]+'−径向', GROUP_LABELS[r['group']], r['metric'], str(r['paired_query_count']),
                  number(r['mean_delta'], True), f"[{number(r['bootstrap_95ci_low'], True)}, {number(r['bootstrap_95ci_high'], True)}]",
                  f"{r['improved_queries']}/{r['worsened_queries']}/{r['tied_queries']}"] for r in paired])
    md += ["## 核查与文件", "",
           "三方法两组报告均通过valid500、37类别、K七档、70指标完整性检查；所有逐查询ID、标签数、类别ID和候选数一致，"
           "query ID集合与项目冻结valid清单一致。全部可用汇总组均从原逐查询CSV重新聚合核对。",
           f"逐查询重聚合与原汇总最大绝对误差：`{aggregation_max_error}`。",
           f"从CSV补出的原报告缺失组：`{json.dumps(reconstructed_groups, ensure_ascii=False)}`。", "",
           "- `comparison_long.csv`：三方法×70指标×8分组，共1,680行，含相对径向基线的差值、百分点和相对变化。",
           "- `paired_bootstrap.csv` / `paired_bootstrap.json`：24个主要指标/组别配对比较及完整区间。",
           "- `report.json`：原始汇总报告、统一分组指标、冻结方案、差值长表、bootstrap与来源审计。",
           "- `source_hashes.json`：原始汇总、逐查询、逐类别、排名、方案及可用审计文件的SHA-256。",
           "- 逐类别原始完整结果位于各方法的`per_class.csv`和`coverage_per_class.csv`，报告只读引用并记录哈希。", "",
           f"原始运行目录：`{run_dir}`。本报告没有改写其中的排名、指标或方案。", ""]

    output.mkdir(parents=True, exist_ok=False)
    write_csv(output/"comparison_long.csv", long_rows)
    write_csv(output/"paired_bootstrap.csv", paired)
    payload = {"status": "reported", "split": "valid", "test_evaluated": False, "ks": list(KS),
               "methods": list(METHODS), "retrieval_protocol_signature": expected_signature,
               "plans": plans, "raw_reports": reports, "groups": grouped, "comparison_long": long_rows,
               "paired_bootstrap": paired, "aggregation_max_error": aggregation_max_error,
               "reconstructed_groups": reconstructed_groups, "audit_sources": audits}
    for name, value in (("report.json", payload), ("paired_bootstrap.json", paired)):
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (output/"report.md").write_text("\n".join(md), encoding="utf-8")
    (output/"source_hashes.json").write_text(json.dumps({"inputs": hashes, "plans": {p.name: {"path": str(p), "sha256": sha256(p)}
        for p in plan_paths}, "audit_sources": audits, "generator": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)}},
        ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return {"output_dir": str(output), "comparison_rows": len(long_rows), "bootstrap_comparisons": len(paired),
            "aggregation_max_error": aggregation_max_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.run_dir, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
