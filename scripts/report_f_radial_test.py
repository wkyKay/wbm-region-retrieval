#!/usr/bin/env python3
"""Report the frozen beta=1 F test and seven existing method references.

This is postprocessing only: no scoring, training, parameter selection, or writes
to source reports. The requested output directory must not already exist.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


KS = [1, 5, 10, 20, 50, 100, 500]
METHODS = ["wu", "yu", "wbbs", "nakazawa", "wmhd", "f", "wapirl", "f_radial_beta1"]
LABELS = {"wu": "Wu", "yu": "Yu", "wbbs": "WBBS", "nakazawa": "Nakazawa CNN",
          "wmhd": "WMHD", "f": "原 F（best_nce）", "wapirl": "WaPIRL（F 对齐整图）",
          "f_radial_beta1": "F＋径向 β=1"}
FILES = {"exact_v1": "metrics.json", "coverage_v1": "coverage_metrics.json"}
FIELDS = {"exact_v1": ["Hit", "Precision", "Recall", "mAP"],
          "coverage_v1": ["Coverage-nDCG", "MeanCoverage", "FullCoverageHit",
                          "FullCoveragePrecision", "FullCoverageRecall", "FullCoverage_mAP"]}
GROUPS = ["all", "class_macro", "single_label", "mixed_label", "label_count_1",
          "label_count_2", "label_count_3", "label_count_4"]
GROUP_LABELS = {"all": "全部查询平均", "class_macro": "完整组合类别宏平均",
                "single_label": "单缺陷查询", "mixed_label": "混合缺陷查询",
                **{f"label_count_{n}": f"{n} 类缺陷查询" for n in range(1, 5)}}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fields(version: str) -> list[str]:
    return [f"{field}@{k}" for k in KS for field in FIELDS[version]]


def validate_report(report: dict, version: str, source: Path) -> None:
    if (report.get("metric_version") != version or report.get("split") != "test"
            or report.get("test_evaluated") is not True or report.get("ks") != KS
            or report.get("num_queries") != 7199 or report.get("num_classes") != 37):
        raise ValueError(f"Not the complete frozen test protocol: {source}")
    expected = set(fields(version))
    for name in ("metrics", "class_macro_metrics"):
        if not expected <= set(report.get(name, {})):
            raise ValueError(f"Missing {version} metrics in {source}: {name}")
        if any(v is not None and not math.isfinite(v) for v in report[name].values()):
            raise ValueError(f"Nonfinite metrics in {source}: {name}")


def by_label_count_from_rows(directory: Path, version: str) -> dict:
    """Fallback for reports that predate the by_label_count summary."""
    name = "per_query.csv" if version == "exact_v1" else "coverage_per_query.csv"
    path = directory / name
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for count in range(1, 5):
        chosen = [r for r in rows if int(r["label_count"]) == count]
        metrics = {}
        for field in fields(version):
            raw_field = ("AP@" + field.split("@")[1] if field.startswith("mAP@") else
                         field.replace("FullCoverage_mAP@", "FullCoverageAP@"))
            values = [float(r[raw_field]) for r in chosen if r.get(raw_field) not in (None, "")]
            metrics[field] = sum(values) / len(values) if values else None
        result[str(count)] = {"num_queries": len(chosen),
                              "num_classes": len({r["class_id"] for r in chosen}), "metrics": metrics,
                              "source": str(path), "computed_from_per_query": True}
    return result


def extract_groups(report: dict, directory: Path, version: str) -> dict:
    groups = {
        "all": {"num_queries": report["num_queries"], "num_classes": report["num_classes"],
                "metrics": report["metrics"]},
        "class_macro": {"num_queries": report["num_queries"], "num_classes": report["num_classes"],
                        "metrics": report["class_macro_metrics"]},
    }
    supplied = report.get("groups", {})
    for name in ("single_label", "mixed_label"):
        if name in supplied:
            groups[name] = supplied[name]
    by_count = supplied.get("by_label_count") or by_label_count_from_rows(directory, version)
    for count in range(1, 5):
        if str(count) in by_count:
            groups[f"label_count_{count}"] = by_count[str(count)]
    return groups


def difference(value: float | None, base: float | None) -> dict:
    if value is None or base is None:
        return {"delta": None, "delta_percentage_points": None, "relative_change": None}
    delta = value - base
    return {"delta": delta, "delta_percentage_points": 100 * delta,
            "relative_change": delta / base if base != 0 else None}


def display(value: float | None, signed=False, digits=9) -> str:
    if value is None:
        return "未定义"
    return format(value, f"{'+' if signed else ''}.{digits}f")


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows], ""]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate(result_dir: Path, references: Path, output: Path) -> dict:
    result_dir, references, output = result_dir.resolve(), references.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh report directory: {output}")
    if result_dir == output or result_dir in output.parents or references == output or references in output.parents:
        raise ValueError("Do not place reports inside the source test/reference directories")
    reference_manifest = read_json(references / "source_manifest.json")
    expected_signature = reference_manifest["retrieval_protocol_signature"]
    reports, groups, sources = {}, {}, {}
    for method in METHODS:
        directory = result_dir if method == "f_radial_beta1" else references / method
        reports[method], groups[method], sources[method] = {}, {}, {}
        for version, filename in FILES.items():
            path = directory / filename
            data = read_json(path)
            validate_report(data, version, path)
            digest = file_hash(path)
            if method != "f_radial_beta1":
                manifest_key = "exact" if version == "exact_v1" else "coverage"
                if digest != reference_manifest["methods"][method][manifest_key]["sha256"]:
                    raise ValueError(f"Reference changed since source audit: {path}")
            if version == "exact_v1" and data.get("retrieval_protocol_signature") != expected_signature:
                raise ValueError(f"Protocol signature differs: {path}")
            reports[method][version] = data
            groups[method][version] = extract_groups(data, directory, version)
            sources[method][version] = {"path": str(path), "sha256": digest}

    rows, grouped_rows, deltas = [], [], {}
    for method in METHODS:
        deltas[method] = {}
        for version in FILES:
            deltas[method][version] = {}
            for group, values in groups[method][version].items():
                baseline = groups["f"][version].get(group, {}).get("metrics", {})
                delta_group = {}
                for field in fields(version):
                    value, base = values["metrics"].get(field), baseline.get(field)
                    change = difference(value, base)
                    delta_group[field] = {"value": value, "baseline_f": base, **change}
                    row = {"method": method, "method_label": LABELS[method], "metric_version": version,
                           "group": group, "num_queries": values.get("num_queries"),
                           "num_classes": values.get("num_classes"), "metric": field,
                           "k": int(field.split("@")[1]), "value": value, "baseline_f": base, **change}
                    grouped_rows.append(row)
                    if group == "all":
                        rows.append(row)
                deltas[method][version][group] = delta_group

    new_key = "f_radial_beta1"
    versus = {}
    for other in ("f", "wbbs", "wapirl"):
        versus[other] = {}
        for version in FILES:
            versus[other][version] = {}
            for group, values in groups[new_key][version].items():
                baseline = groups[other][version].get(group, {}).get("metrics", {})
                versus[other][version][group] = {
                    field: {"new_value": values["metrics"].get(field), "reference_value": baseline.get(field),
                            **difference(values["metrics"].get(field), baseline.get(field))}
                    for field in fields(version)}

    audit_sources = {}
    for name in ("selection_before_test.json", "independent_verification.json", "completed.json"):
        path = result_dir.parent / name
        if path.exists():
            audit_sources[name] = {"path": str(path), "sha256": file_hash(path), "contents": read_json(path)}
    source_doc = Path(__file__).resolve().parents[1] / "doc/EXACT_METRICS_MIGRATION.md"
    doc_link = f"[EXACT_METRICS_MIGRATION.md]({source_doc})"
    md = ["# F＋径向 β=1：完整 test 报告", "",
          f"对照口径与方法集合依据 {doc_link}。本报告在原七方法之外新增径向F，保留原F基线。", "",
          "## 实验边界与协议", "",
          "径向β=1已在此前固定valid网格中选定，本轮test不重新选参数或checkpoint。复用原F的best_nce；"
          "使用原h512、原区域及单向贪心面积加权评分，没有四旋转汇聚或双向评分。",
          "径向项为16-bin归一化半径直方图的CDF距离，区域对分数乘以exp(-D)。"
          "β使用valid标签选参，不能称为全过程无监督。", "",
          "test包含7,199个查询、37种完整标签组合；每个查询对其余7,198个候选评分，排除自身。"
          "两组指标来自同一份新test排名；保存Top500，不能将其称为完整排序文件。",
          f"K固定为：{', '.join(map(str, KS))}。协议签名：`{expected_signature}`。", "",
          "exact_v1以完整multi-hot标签相同为相关；报告Hit、Precision、Recall、mAP。"
          "coverage_v1使用查询类别覆盖率的线性gain，报告Coverage-nDCG、MeanCoverage和四项FullCoverage指标；"
          "完整覆盖定义为Lq⊆Lc。两组AP/Recall分母分别来自完整候选库的相应相关数，均不使用min(R,K)。", "",
          "Markdown保留9位小数；CSV/JSON保留原始浮点数。百分点为比例差×100，相对变化为差值/原值，二者不同。"
          "这是一轮固定checkpoint结果，不据此推断跨训练seed方差或宣称统计显著性。", "",
          "## 新方法完整指标", ""]
    for version in FILES:
        md += [f"### {version}", ""]
        for group in GROUPS:
            if group not in groups[new_key][version]:
                md += [f"#### {GROUP_LABELS[group]}", "", "原报告及逐查询文件未提供该组，未补造数值。", ""]
                continue
            values = groups[new_key][version][group]
            description = (f"对{values.get('num_classes')}个完整组合类别等权平均；底层查询数{values.get('num_queries')}。"
                           if group == "class_macro" else
                           f"查询数：{values.get('num_queries')}；完整组合类别数：{values.get('num_classes')}。")
            md += [f"#### {GROUP_LABELS[group]}", "", description, ""]
            md += table(["K", *FIELDS[version]], [[str(k), *[display(values['metrics'].get(f'{f}@{k}')) for f in FIELDS[version]]]
                                                  for k in KS])

    md += ["## 八方法对比", "", "以下均为query平均，按文档的方法次序排列；没有重新运行或改写七个参考方法。", "",
           "### coverage_v1：沿用原文档的紧凑表", ""]
    cov_columns = [("Coverage-nDCG", [10, 20]), ("MeanCoverage", [10, 20]),
                   ("FullCoverage_mAP", [10, 20]), ("FullCoverageRecall", [10, 20]), ("FullCoverageRecall", [500])]
    cov_headers = ["方法", "Coverage-nDCG@10 / @20", "MeanCoverage@10 / @20", "FullCoverage mAP@10 / @20",
                   "FullCoverage Recall@10 / @20", "FullCoverage Recall@500"]
    md += table(cov_headers, [[LABELS[m], *[" / ".join(display(reports[m]['coverage_v1']['metrics'][f'{field}@{k}'])
                    for k in cutoffs) for field, cutoffs in cov_columns]] for m in METHODS])
    md += ["### exact_v1：紧凑表", ""]
    exact_compact = ["Hit@1", "Precision@10", "mAP@10", "mAP@100", "Recall@500"]
    md += table(["方法", *exact_compact], [[LABELS[m], *[display(reports[m]['exact_v1']['metrics'][field])
                                                        for field in exact_compact]] for m in METHODS])

    md += ["## 与原F的完整总体差值", "", "所有单元格为“新F−原F”的原始比例差；完整相对变化另见CSV/JSON。", ""]
    for version in FILES:
        md += [f"### {version}", ""]
        md += table(["K", *FIELDS[version]], [[str(k), *[display(versus['f'][version]['all'][f'{f}@{k}']['delta'], True)
                                                        for f in FIELDS[version]]] for k in KS])
    md += ["## 与原F、WBBS、WaPIRL的主要差距", "", "差值均使用未舍入JSON计算；正值表示新径向F更高，负值表示更低。", ""]
    key_fields = {"exact_v1": ["Hit@1", "Precision@10", "mAP@10", "mAP@100", "Recall@500"],
                  "coverage_v1": ["Coverage-nDCG@10", "Coverage-nDCG@20", "MeanCoverage@10",
                                  "FullCoverage_mAP@10", "FullCoverage_mAP@20", "FullCoverageRecall@500"]}
    for version, keys in key_fields.items():
        md += [f"### {version}", ""]
        md += table(["指标", "新径向F", "−原F（百分点）", "−WBBS（百分点）", "−WaPIRL（百分点）"],
                    [[key, display(reports[new_key][version]['metrics'][key]),
                      *[display(versus[m][version]['all'][key]['delta_percentage_points'], True) for m in ('f', 'wbbs', 'wapirl')]]
                     for key in keys])
    md += ["### 分组变化", "", "以下用于描述结果分布，不用于test选参或按真实标签改变方法评分。", ""]
    for version, key in (("exact_v1", "mAP@10"), ("coverage_v1", "Coverage-nDCG@10")):
        md += [f"{version}：{key}", ""]
        md += table(["分组", "原F", "径向F", "差值（百分点）", "相对变化（%）"],
                    [[GROUP_LABELS[g], display(versus['f'][version][g][key]['reference_value']),
                      display(versus['f'][version][g][key]['new_value']),
                      display(versus['f'][version][g][key]['delta_percentage_points'], True),
                      display(None if versus['f'][version][g][key]['relative_change'] is None else
                              100*versus['f'][version][g][key]['relative_change'], True)]
                     for g in GROUPS if g in versus['f'][version]])

    md += ["## 文件与可追溯性", "",
           f"- 新test输入：`{result_dir}`。",
           f"- 七方法参考资料：`{references}`；来源及原始SHA见该目录的source_manifest.json。",
           "- `comparison_long.csv`：8方法×70项总体指标，含原始浮点值、相对原F的绝对差/百分点/相对变化。",
           "- `grouped_metrics.csv`：全部可用分组的同类长表，包含类别宏平均及1–4标签分组。",
           "- `report.json`：保留全部原始汇总报告、分组、差值及原F/WBBS/WaPIRL逐组比较。",
           "- `source_hashes.json`：本报告输入、参考文档、生成脚本以及可用冻结/完成/独立审计文件的哈希。", ""]
    for name, source in audit_sources.items():
        md += [f"- `{name}`：`{source['path']}`。"]
    md += ["", "报告不修改原始test排名、checkpoint、冻结参数或既有EXACT文档。", ""]

    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "comparison_long.csv", rows)
    write_csv(output / "grouped_metrics.csv", grouped_rows)
    payload = {"new_method": new_key, "ks": KS, "retrieval_protocol_signature": expected_signature,
               "sources": sources, "raw_reports": reports, "groups": groups, "deltas_vs_f": deltas,
               "new_vs_references": versus, "audit_sources": audit_sources,
               "table_rows": {"comparison_long": len(rows), "grouped_metrics": len(grouped_rows)}}
    (output / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text("\n".join(md), encoding="utf-8")
    hashes = {"inputs": sources, "reference_manifest": {"path": str(references/'source_manifest.json'),
              "sha256": file_hash(references/'source_manifest.json')},
              "reference_document": {"path": str(source_doc), "sha256": file_hash(source_doc)},
              "generator": {"path": str(Path(__file__).resolve()), "sha256": file_hash(Path(__file__))},
              "audit_sources": {name: {k: v for k, v in value.items() if k != "contents"} for name, value in audit_sources.items()}}
    (output / "source_hashes.json").write_text(json.dumps(hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output_dir": str(output), **payload["table_rows"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path, help="New run/test directory")
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(generate(args.result, args.references, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
