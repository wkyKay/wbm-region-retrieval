"""B1 result tables. References must be verified and split-matched by the caller."""
import csv
from pathlib import Path

from wbm_retrieval.common import read_json, write_csv


def groups(report):
    return {"all": report["metrics"], "class_macro": report["class_macro_metrics"],
            **{name: report["groups"][name]["metrics"] for name in ("single_label", "mixed_label")},
            **{f"by_label_count/{number}": value["metrics"]
               for number, value in report["groups"]["by_label_count"].items()}}


def reference_index(reference, split):
    """Do not substitute test values when historical valid coverage is missing."""
    result = {}
    for row in reference or []:
        if row["split"] != split:
            continue
        name = row["metric"] if "@" in row["metric"] else f"{row['metric']}@{row['K']}"
        key = (row["metric_version"], row["group"], name)
        value = row.get("value")
        value = None if value in (None, "") else float(value)
        if key in result and result[key] != value:
            raise ValueError(f"Conflicting WBBS reference metric: {key}")
        result[key] = value
    return result


def comparison_rows(reports, specs, split, reference=None):
    refs = reference_index(reference, split)
    baseline = reports.get("B0", {})
    rows = []
    for variant, families in reports.items():
        for family, report in families.items():
            base_groups = groups(baseline[family]) if family in baseline else {}
            for group, metrics in groups(report).items():
                for metric, value in metrics.items():
                    wbbs = refs.get((report["metric_version"], group, metric))
                    base = base_groups.get(group, {}).get(metric)
                    rows.append({"split": split, "variant": variant, **specs[variant],
                                 "metric_version": report["metric_version"], "group": group,
                                 "metric": metric, "value": value, "baseline_value": base,
                                 "wbbs_value": wbbs,
                                 "delta_vs_baseline": value - base if value is not None and base is not None else None,
                                 "delta_vs_wbbs": value - wbbs if value is not None and wbbs is not None else None})
    return rows


def print_report(name, split, families, reference=None):
    refs = reference_index(reference, split)
    print(f"\nB1 {split} / {name}: 总体 query 平均；数值为 %，Δ = 本实验 − WBBS（百分点）", flush=True)
    for family, report in families.items():
        print(f"  {family} | 指标 | 本实验 | WBBS | Δ(pp)", flush=True)
        for k in report["ks"]:
            for metric, value in report["metrics"].items():
                if metric.rsplit("@", 1)[-1] != str(k):
                    continue
                reference_value = refs.get((report["metric_version"], "all", metric))
                def fmt(number):
                    return "NA" if number is None else f"{number * 100:.4f}"
                difference = None if value is None or reference_value is None else value-reference_value
                print(f"  {family} | {metric} | {fmt(value)} | {fmt(reference_value)} | {fmt(difference)}", flush=True)
        if not any(key[0] == report["metric_version"] for key in refs):
            print(f"  WBBS {split}/{family} 缺少同协议报告，差值记 NA，不使用其他 split 替代。", flush=True)


def summarize(output, split, specs, reference=None, *, print_all=False):
    output = Path(output)
    reports = {name: {family: read_json(output / split / name / filename)
                      for family, filename in (("exact", "metrics.json"), ("coverage", "coverage_metrics.json"))}
               for name in specs}
    rows = comparison_rows(reports, specs, split, reference)
    if rows:
        write_csv(output / f"comparison_{split}.csv", rows, list(rows[0]))
        combined = []
        for current_split in ("valid", "test"):
            path = output / f"comparison_{current_split}.csv"
            if path.exists():
                with path.open(newline="", encoding="utf-8") as handle:
                    combined.extend(csv.DictReader(handle))
        write_csv(output / "comparison.csv", combined, list(rows[0]))
    if print_all:
        for name, families in reports.items():
            print_report(name, split, families, reference)
    return reports


def print_selection_summary(output, selection, split="valid"):
    """A compact final view after the exhaustive per-configuration metric output."""
    path = Path(output) / f"comparison_{split}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        records = {(row["variant"], row["metric_version"], row["metric"]): row
                   for row in csv.DictReader(handle) if row["group"] == "all"}
    columns = [("exact_v1", "Precision@10"), ("exact_v1", "mAP@100"),
               ("coverage_v1", "Coverage-nDCG@10"), ("coverage_v1", "FullCoveragePrecision@10")]
    roles = [("B0", "B0", 0.)] + [(family, selection["selections"][family]["selected"]["variant"],
                selection["selections"][family]["selected"]["strength"]) for family in ("R", "T", "I", "J")]
    print(f"\n{split} 选中摘要：总体指标（%）；括号为相对 WBBS 的百分点差；J 是 B1 主候选。", flush=True)
    print("角色 / 配置 / lambda | exact P@10 | exact mAP@100 | Coverage-nDCG@10 | FullCoverage P@10", flush=True)
    for role, name, strength in roles:
        cells = []
        for family, metric in columns:
            row = records.get((name, family, metric), {})
            value, difference = row.get("value"), row.get("delta_vs_wbbs")
            value_text = "NA" if value in (None, "") else f"{100*float(value):.4f}"
            delta_text = "NA" if difference in (None, "") else f"{100*float(difference):+.4f}"
            cells.append(f"{value_text} ({delta_text})")
        print(f"{role} / {name} / {strength:g} | " + " | ".join(cells), flush=True)
    wbbs = []
    for family, metric in columns:
        value = records.get(("B0", family, metric), {}).get("wbbs_value")
        wbbs.append("NA" if value in (None, "") else f"{100*float(value):.4f}")
    print("WBBS / 同 split 参考 / — | " + " | ".join(wbbs), flush=True)
    print("零强度角色共用 B0；这些角色不是独立重复实验。", flush=True)
