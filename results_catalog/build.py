#!/usr/bin/env python3
"""Read-only result collection; deterministic offline report generation (stdlib only)."""
import argparse
import csv
import hashlib
import io
import json
import math
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
TZ = ZoneInfo("Asia/Shanghai")
PROTOCOL_FILES = ["deduplicated/split.csv", "retrieval_v1/metadata.json"] + [
    f"retrieval_v1/{s}_{part}.csv" for s in ("valid", "test") for part in ("queries", "gallery")]
BASE_PROTOCOL_SIGNATURE = "6a428fb8d8bdf34fb6c60852b6d9a7467bf6880cae8c802a6a9c5a36f96cb8a1"
CROSS_SPLITS = {
    "valid": {"candidate_identity_sha256": "e9e865ff288db79bf622f5016c17ac22d61d5ca071d39893a6202b9b0dd013b7",
              "candidates_per_query": "3499-3585"},
    "test": {"candidate_identity_sha256": "68c6875330ed70841586c654da776f8d0bc72bcd4cd341b79cea29705543f0b8",
             "candidates_per_query": "6999-7170"},
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value, *, compact=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    formatting = {"separators": (",", ":")} if compact else {"indent": 2}
    text = json.dumps(value, ensure_ascii=False, allow_nan=False, **formatting) + "\n"
    path.write_text(text)


def ref(root, path):
    return {"path": str(path.relative_to(root)), "sha256": sha(path)}


def report(root, folder, marker, name):
    path = folder / name
    if not path.exists():
        return None
    evidence = read(marker)
    relative = str(path.relative_to(marker.parent))
    expected = evidence.get("files", {}).get(relative)
    if expected is None or sha(path) != expected:
        raise ValueError(f"Report missing from completion marker or hash mismatch: {path}")
    data = read(path)
    if data.get("metadata", {}).get("smoke"):
        raise ValueError(f"Smoke cannot enter registered formal reports: {path}")
    return {"source": ref(root, path), "completion": ref(root, marker), "data": data}


def metric_groups(data):
    yield "all", data["metrics"]
    if "class_macro_metrics" in data:
        yield "class_macro", data["class_macro_metrics"]

    def walk(value, prefix):
        if isinstance(value, dict):
            if "metrics" in value:
                yield prefix, value["metrics"]
            else:
                for key, child in sorted(value.items()):
                    yield from walk(child, f"{prefix}/{key}" if prefix else key)
    yield from walk(data.get("groups", {}), "")


def execution_identity(root, spec, marker, manifest):
    """Read explicit new-run identities; legacy sources keep their mtime basis."""
    identity_name = spec.get("identity")
    identity_schema = spec.get("identity_schema_version")
    if identity_name is not None or identity_schema is not None:
        if type(identity_schema) is not int or identity_schema != 1 or not isinstance(identity_name, str) or not identity_name:
            raise ValueError(f"New run requires identity_schema_version=1 and identity: {spec['key']}")
        relative = Path(identity_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Identity must be a repository-relative path: {spec['key']}")
        path = root / relative
        try:
            path.resolve().relative_to(root.resolve())
            marker_relative = str(path.relative_to(marker.parent))
        except ValueError as error:
            raise ValueError(f"Identity must be inside the completed run: {spec['key']}") from error
        if not path.is_file():
            raise ValueError(f"Missing declared run identity: {path}")
        expected = read(marker).get("files", {}).get(marker_relative)
        if expected is None or sha(path) != expected:
            raise ValueError(f"Run identity hash mismatch or missing completion binding: {path}")
        identity = read(path)
        if (not isinstance(identity, dict) or type(identity.get("schema_version")) is not int or
                identity.get("schema_version") != 1 or identity.get("key") != spec["key"] or
                identity.get("split") != spec["split"]):
            raise ValueError(f"Run identity schema/key/split mismatch: {spec['key']}")
        stamp_text = identity.get("started_at")
        if not isinstance(stamp_text, str) or not re.fullmatch(
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00", stamp_text):
            raise ValueError(f"started_at must contain seconds and +08:00: {spec['key']}")
        try:
            stamp = datetime.fromisoformat(stamp_text)
        except ValueError as error:
            raise ValueError(f"Invalid started_at: {spec['key']}") from error
        key = spec["key"]
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", key):
            raise ValueError(f"Invalid run identity key: {key}")
        run_id = stamp.strftime("%Y%m%dT%H%M%S%z") + "__" + key
        if identity.get("run_id") != run_id:
            raise ValueError(f"run_id does not match started_at/key: {key}")
        return run_id, {"value": stamp_text, "basis": "started_at", "source": identity_name,
                        "is_execution_start": True}, {"identity": ref(root, path), "identity_data": identity}
    if "started_at" in manifest or "run_id" in manifest:
        raise ValueError(f"Manifest declares a new run identity; add an explicit identity source: {spec['key']}")
    # Historical sources have no persisted start identity. Do not invent one.
    stamp = datetime.fromtimestamp(marker.stat().st_mtime, TZ)
    return (stamp.strftime("%Y%m%dT%H%M%S%z") + "__" + spec["key"],
            {"value": stamp.isoformat(timespec="seconds"), "basis": "completion_marker_mtime",
             "source": str(marker.relative_to(root)), "is_execution_start": False}, {})


def collect(workspace, specs, previous=None):
    workspace = workspace.resolve()
    roots = {name: workspace / name for name in ("compared_methods", "wbm-region-retrieval")}
    protocol = {name: {p: sha(root / "protocol" / p) for p in PROTOCOL_FILES} for name, root in roots.items()}
    if protocol["compared_methods"] != protocol["wbm-region-retrieval"]:
        raise ValueError("Frozen protocol files differ across repositories")
    old = {r["key"]: r for r in (previous or {}).get("runs", [])}
    runs = []
    for spec in specs["runs"]:
        root = roots[spec["repository"]]
        folder = root / spec["path"]
        marker = root / spec["marker"] if spec.get("marker") else folder / "completed.json"
        if not marker.is_file():
            raise ValueError(f"Missing completion marker: {marker}")
        manifest_path = root / spec["manifest"] if spec.get("manifest") else None
        if manifest_path and not manifest_path.is_file():
            raise ValueError(f"Missing manifest: {manifest_path}")
        manifest = read(manifest_path) if manifest_path else {}
        if manifest.get("smoke"):
            raise ValueError(f"Smoke manifest: {manifest_path}")
        reports = {}
        for family, name in (("exact", "metrics.json"), ("coverage", "coverage_metrics.json")):
            item = report(root, folder, marker, name)
            if item:
                reports[family] = item
        cov = spec.get("coverage")
        extra = None
        if cov:
            cm = root / cov["marker"]
            extra = read(root / cov["manifest"])
            source_marker = read(marker)
            ranking_relative = str((folder / "rankings.csv").relative_to(marker.parent))
            ranking_hash = source_marker.get("files", {}).get(ranking_relative)
            if not ranking_hash or extra.get("ranking_source_sha256") != ranking_hash:
                raise ValueError(f"Coverage and exact do not reference the same ranking: {spec['key']}")
            reports["coverage"] = report(root, root / cov["path"], cm, "coverage_metrics.json")
            if reports["coverage"] is None:
                raise ValueError(f"Missing declared coverage report: {spec['key']}")
        expected_n = 500 if spec["split"] == "valid" else 7199
        protocol_name = spec.get("protocol", "retrieval_v1")
        if set(reports) != set(spec["required_families"]) or protocol_name == "coverage_cross_combination_v1" and "exact" in reports:
            raise ValueError(f"Invalid report families: {spec['key']}")
        for family, item in reports.items():
            d = item["data"]
            if d["split"] != spec["split"] or d["num_queries"] != expected_n:
                raise ValueError(f"Query/split mismatch: {spec['key']}/{family}")
            if d["metric_version"] != family + "_v1":
                raise ValueError(f"Metric version mismatch: {spec['key']}/{family}")
            if d["ks"] != [1, 5, 10, 20, 50, 100, 500]:
                raise ValueError(f"Unexpected K values: {spec['key']}/{family}")
            derived = d.get("metadata", {}).get("derived_protocol")
            if (protocol_name == "coverage_cross_combination_v1") != bool(derived):
                raise ValueError(f"Candidate protocol mismatch: {spec['key']}/{family}")
            if derived and (spec["split"] not in CROSS_SPLITS or derived.get("candidate_identity_sha256") != CROSS_SPLITS[spec["split"]]["candidate_identity_sha256"]):
                raise ValueError(f"Cross-combination candidate identity changed: {spec['key']}")
            signature = derived.get("base_protocol_signature") if derived else d.get("retrieval_protocol_signature", d.get("metadata", {}).get("retrieval_protocol_signature"))
            if (family == "exact" or derived or signature is not None) and signature != BASE_PROTOCOL_SIGNATURE:
                raise ValueError(f"Report frozen protocol signature mismatch: {spec['key']}/{family}")
        run_id, time_info, identity_provenance = execution_identity(root, spec, marker, manifest)
        if spec["key"] in old:
            prior = old[spec["key"]]
            if prior["repository"] != spec["repository"] or prior["path"] != spec["path"]:
                raise ValueError(f"Existing key moved; allocate a new key: {spec['key']}")
            if prior["completion"]["sha256"] != sha(marker):
                raise ValueError(f"Completed run changed; allocate a new key: {spec['key']}")
            if (prior["id_time"]["basis"] != time_info["basis"] or
                    time_info["is_execution_start"] and (prior["run_id"] != run_id or
                    prior["id_time"] != time_info or
                    prior.get("provenance", {}).get("identity") != identity_provenance.get("identity"))):
                raise ValueError(f"Registered run identity changed; allocate a new key: {spec['key']}")
            run_id, time_info = prior["run_id"], prior["id_time"]
        provenance = {"manifest": ref(root, manifest_path), "manifest_data": manifest} if manifest_path else {}
        provenance.update(identity_provenance)
        if cov:
            provenance["coverage_manifest"] = ref(root, root / cov["manifest"])
            provenance["coverage_manifest_data"] = extra
        if spec.get("selection"):
            p = root / spec["selection"]
            provenance["selection"] = ref(root, p)
            provenance["selection_data"] = read(p)
        for name in spec.get("supporting_manifests", []):
            p = root / name
            provenance.setdefault("supporting_manifests", []).append({"source": ref(root, p), "data": read(p)})
        runs.append({**spec, "run_id": run_id, "id_time": time_info,
                     "protocol": protocol_name, "completion": ref(root, marker),
                     "status": "report_hashes_verified", "query_count": expected_n,
                     "gallery_count": 3599 if spec["split"] == "valid" else 7199,
                     "candidates_per_query": CROSS_SPLITS[spec["split"]]["candidates_per_query"] if protocol_name != "retrieval_v1" else expected_n - 1 if spec["split"] == "test" else 3598,
                     "provenance": provenance, "reports": reports})
        print("Verified", spec["key"], flush=True)
    if len({r["run_id"] for r in runs}) != len(runs) or len({r["key"] for r in runs}) != len(runs):
        raise ValueError("Duplicate run IDs or keys")
    inventory = []
    for name, root in roots.items():
        outputs = [p for directory in ("artifacts", "artificial") if (root / directory).is_dir()
                   for p in sorted((root / directory).iterdir())]
        for p in outputs:
            if not p.is_dir():
                continue
            members = [r["key"] for r in runs if r["repository"] == name and r["path"].startswith(str(p.relative_to(root)) + "/")]
            auxiliary = [r["key"] for r in runs if r["repository"] == name and r.get("coverage") and r["coverage"]["path"].startswith(str(p.relative_to(root)) + "/")]
            status_path = p / "status.json"
            status_data = read(status_path) if status_path.exists() else {}
            inventory.append({"repository": name, "path": str(p.relative_to(root)),
                              "registered_results": members, "auxiliary_coverage": auxiliary,
                              "role": "registered" if members else "auxiliary" if auxiliary else "not_in_current_tables",
                              "recorded_status": status_data.get("status") if isinstance(status_data, dict) else "per_method_list"})
    return {"schema_version": 1, "collected_at": datetime.now(TZ).isoformat(timespec="seconds"),
            "source_workspace": str(workspace), "protocol_files": protocol,
            "repository_heads": {n: subprocess.check_output(["git", "-C", str(p), "rev-parse", "HEAD"], text=True).strip() for n, p in roots.items()},
            "verification_scope": "Report SHA-256 checked against completed.json; referenced manifests and selections captured. Rankings/model hashes are provenance references, not recomputed in this audit. No training or metric recomputation.",
            "source_spec_sha256": hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest(),
            "inventory": inventory, "runs": runs}


def csv_text(rows, fields):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def render(snapshot):
    runs = snapshot["runs"]
    rows, registry = [], []
    for run in runs:
        registry.append({**run, "metric_families": ";".join(run["reports"]),
                         "report_sources": ";".join(f + ":" + v["source"]["path"] for f, v in run["reports"].items()),
                         "completion_sha256": run["completion"]["sha256"],
                         "time_basis": run["id_time"]["basis"], "id_time": run["id_time"]["value"],
                         "posthoc": bool(run.get("posthoc")),
                         "remote_path": snapshot["source_workspace"] + "/" + run["repository"] + "/" + run["path"]})
        for family, report_ in run["reports"].items():
            for group, metrics in metric_groups(report_["data"]):
                for name, value in sorted(metrics.items()):
                    metric, k = name.rsplit("@", 1)
                    if value is not None and not math.isfinite(value):
                        raise ValueError(f"Nonfinite metric: {run['key']} {name}")
                    rows.append({**run, "metric_version": family + "_v1", "group": group,
                                 "metric": metric, "K": int(k), "value": "" if value is None else value,
                                 "posthoc": bool(run.get("posthoc")), "source": report_["source"]["path"],
                                 "source_sha256": report_["source"]["sha256"]})
    fields = ["run_id", "key", "repository", "category", "method", "variant", "supervision", "split", "protocol", "posthoc", "metric_version", "group", "metric", "K", "value", "source", "source_sha256"]
    output = {"metrics.csv": csv_text(rows, fields),
              "registry.csv": csv_text(registry, ["run_id", "key", "repository", "category", "method", "variant", "supervision", "split", "protocol", "metric_families", "posthoc", "status", "query_count", "gallery_count", "candidates_per_query", "id_time", "time_basis", "remote_path", "report_sources", "completion_sha256"])}
    table_filters = {
        "methods": lambda r: r["split"] == "test" and (r["category"] == "methods" or r["key"] == "f_geometry_cc8_test"),
        "segmentation": lambda r: r["category"] == "segmentation",
        "fusion": lambda r: r["category"] in ("fusion", "fusion_ablation", "posthoc"),
        "cross_combination": lambda r: r["category"] == "cross_combination",
    }
    for name, predicate in table_filters.items():
        output[f"tables/{name}.csv"] = csv_text([r for r in rows if predicate(r)], fields)
    lines = ["# 当前结果总览", "", f"结果核验时间：{snapshot['collected_at']}（Asia/Shanghai）。本页由 `results_catalog/build.py` 自动生成。", "",
             "当前主方法为 cc8＋径向位置＋相对面积，冻结 best_nce；融合和角向／最优匹配保留为扩展实验。当前主方法身份在查看历史 test 后确认，不追述为 test 前选定。", "",
             "## 协议与文件入口", "",
             "冻结去重 train/valid/test 为 25,193/3,599/7,199；valid 固定 500 query × 3,598 候选，test 为 7,199 query × 7,198 候选。排除自身，Normal 不进检索库。", "",
             "exact 要求完整标签组合相同；coverage 按查询标签覆盖比例，FullCoverage 允许候选额外标签。所有 AP/Recall 分母和 coverage IDCG 取对应完整候选库。保存的排名是 Top-500，mAP@K 是截断指标。", "",
             "[使用说明](README.md) · [结果登记表](registry.csv) · [全部指标长表](metrics.csv) · [来源配置](sources.json) · [核验快照及来源签名](snapshot.json)", "",
             f"登记 {len(runs)} 个评价结果、{len(rows):,} 条指标；模型训练运行作为来源保存在 manifest 中。未进入当前表的服务器目录也列于 snapshot.json 的 inventory，不表示目录无效或可以删除。", "",
             "数值表均为 test 总体 query 平均，原始比例乘 100 展示。完整 K=1/5/10/20/50/100/500、类别宏平均、单／混合及标签数量分组保存在 CSV。监督方法与自监督／无训练方法的监督信息不同。", ""]
    by_key = {r["key"]: r for r in runs}

    def table(title, selected, coverage_only=False):
        lines.extend(["## " + title, ""])
        columns = [("coverage", "Coverage-nDCG@10"), ("coverage", "FullCoveragePrecision@10")] if coverage_only else [("exact", "Hit@1"), ("exact", "Precision@10"), ("exact", "mAP@500"), ("coverage", "Coverage-nDCG@10")]
        lines.append("| 方法／变体 | " + " | ".join(f"{f} {m}" for f, m in columns) + " | 登记 key |")
        lines.append("|---|" + "---:|" * len(columns) + "---|")
        for r in selected:
            values = [r["reports"].get(f, {}).get("data", {}).get("metrics", {}).get(m) for f, m in columns]
            lines.append("| " + r.get("display", r["method"]) + " | " + " | ".join("—" if v is None else f"{v*100:.4f}" for v in values) + " | `" + r["key"] + "` |")
        lines.append("")

    table("普通 test 方法比较", [r for r in runs if table_filters["methods"](r)])
    lines.extend(["Nakazawa、Yu 使用监督分类训练；WaPIRL 是 F 对齐的整图适配。WMHD 数值代表当前实现和适配设置，不等于原论文结论。当前主方法在上述指标上仍低于 WBBS 和 Nakazawa。", ""])
    table("六种分割对照", [r for r in runs if r["category"] == "segmentation" and r["split"] == "test"])
    keys = ["f_geometry_cc8_test", "f_fusion_0.5_test", "f_ablation_angular_0.5_test", "f_ablation_optimal_test", "f_combination_test"]
    # Historical endpoints duplicate the F/WaPIRL or original fusion reference rows.
    reference_aliases = {"f_fusion_0.0_test", "f_fusion_1.0_test", "f_ablation_baseline_test"}
    additional_fusion = [r for r in runs if r["key"] not in keys and r["key"] not in reference_aliases and r["split"] == "test"
                         and r.get("protocol", "retrieval_v1") == "retrieval_v1" and table_filters["fusion"](r)]
    table("融合及消融", [by_key[k] for k in keys] + additional_fusion)
    historical_combination = "历史角向＋最优匹配组合" if additional_fusion else "最后一行组合"
    lines.extend([f"融合 α=0.5、角向 γ=0.5 由 valid exact/all/Precision@10 选择；占比约束选中零强度，等于基线。最优匹配单独报告。{historical_combination}是在查看前一轮 test 后提出的补充实验，posthoc=true，没有 test 参数搜索，也不能描述为 test 前确定的方案。", "",
                  "组合与角向单项的差值不代表统计显著或跨种子稳定收益。valid 参数网格、所有 test 对照和分组见 [融合长表](tables/fusion.csv)。", ""])
    if additional_fusion:
        lines.extend(["新增融合实验按各自已冻结的 valid 选择记录评价；监督信息、事后设计背景和来源见登记表及 manifest，不将新增变体自动替换为当前主方法。", ""])
    table("跨组合 test", [r for r in runs if r["category"] == "cross_combination" and r["split"] == "test"], True)
    lines.extend(["本表额外排除同标签组合候选，每 query 6,999–7,170 个候选，沿用原 coverage_v1 指标。999 个 query 无 FullCoverage 正样本，FullCoverage Recall/AP 有效 query=6,200；其中199个 query 的 IDCG=0，Coverage-nDCG 有效 query=7,000；MeanCoverage、FullCoverage Hit/Precision 保留7,199个 query。历史七方法中的 F 是2026-11-11旧径向面积来源，不能替代当前 cc8；后续融合变体以各自来源、冻结配置和运行身份区分。所有跨组合结果与普通 test 分表展示。", "",
                  "## 保存位置与追溯", "",
                  f"服务器工作区：`{snapshot['source_workspace']}`。registry.csv 的 remote_path 是每项报告所在目录；snapshot.json 保存报告哈希、完成标记哈希、来源 manifest、选参记录及协议清单哈希。", "",
                  "当前主实验：`wbm-region-retrieval/artifacts/f_segmentation_radial_area_v1/cc8/{valid,test}/best_nce/`；模型：`wbm-region-retrieval/artifacts/f_segmentation_v1/cc8/seed_2026/F/best_nce.pt`。", "",
                  "原始 artifacts、archives、数据、模型、冻结协议保持原位。登记表的时间 ID 不是重命名后的实验目录。Git 管理本目录的轻量快照、汇总及生成代码，不包含模型或 Top-500 排名。", "",
                  "历史 run_id 时间取完成标记文件 mtime（含 +0800），明确不是实验开始时间；同步可能改变 mtime，故首次登记后保留 ID。原始时间来源在 id_time 中，登记时间另见 collected_at。", "",
                  "本次核验范围：指标报告与 completed.json 中 SHA-256 一致；冻结清单两仓库一致；独立 coverage 后处理与 exact 的排名哈希引用一致。没有重新训练、重新计算指标或逐一重算模型／排名哈希。", ""])
    output["CURRENT_RESULTS.md"] = "\n".join(lines)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["collect", "render", "check"])
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--catalog", type=Path, default=HERE)
    args = parser.parse_args()
    folder = args.catalog.resolve()
    if any(name in folder.parts for name in ("artifacts", "artificial", "archives", "protocol")):
        parser.error("Catalog output must be outside experiment artifacts, artificial, archives and protocol")
    snapshot_path = folder / "snapshot.json"
    if args.command == "collect":
        if not args.workspace:
            parser.error("collect requires --workspace (both repository directories)")
        snapshot = collect(args.workspace, read(folder / "sources.json"), read(snapshot_path) if snapshot_path.exists() else None)
        generated = render(snapshot)  # Validate everything before any output writes.
        dump(snapshot_path, snapshot, compact=True)
    else:
        snapshot = read(snapshot_path)
        expected = hashlib.sha256(json.dumps(read(folder / "sources.json"), sort_keys=True).encode()).hexdigest()
        if snapshot["source_spec_sha256"] != expected:
            raise ValueError("Source specification changed; recollect before rendering")
        generated = render(snapshot)
    for name, content in generated.items():
        path = folder / name
        if args.command == "check":
            if not path.exists() or path.read_text() != content:
                raise ValueError(f"Generated file is stale: {name}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    print(f"{args.command}: {len(snapshot['runs'])} results, {len(generated)} generated files verified")


if __name__ == "__main__":
    main()
