"""Paired query analysis of sealed, valid-selected whole-map G/J results.

This postprocessing never ranks methods on test or changes the frozen choice.
All comparisons use original query IDs and raw eight-bit multi-hot labels;
class_id numbers from different method implementations are deliberately ignored.
"""
from __future__ import annotations

import argparse
import csv
import math
from numbers import Integral
from pathlib import Path

import numpy as np

from wbm_retrieval.common import digest, file_hash, read_json, write_csv, write_json
from .selection import select_candidates
from .state import now, read_signed, signed


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("Precision@10", "mAP@100", "Coverage-nDCG@10")
COMPARISON_ROLES = ("geometric_winner", "product_reference", "structure_reference",
                    "global_reference", "WBBS")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class SourceBinding:
    """Bind all read evidence and recheck it before completing the analysis."""
    def __init__(self):
        self.files = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        _require(path.is_file(), "Missing source file: " + str(path))
        actual = file_hash(path)
        _require(expected is None or expected == actual, "Source hash differs: " + str(path))
        _require(str(path) not in self.files or self.files[str(path)] == actual,
                 "Source changed while loading: " + str(path))
        self.files[str(path)] = actual
        return path

    def sealed(self, root, *, signature=None, require_status=False):
        root = Path(root).resolve()
        marker = read_json(self.bind(root / "completed.json"))
        _require(not require_status or marker.get("status") == "completed", "Run is not completed")
        if signature is not None:
            _require(marker.get("signature") == signature, "Completion signature differs")
        files = marker.get("files")
        _require(isinstance(files, dict) and bool(files), "Completion marker does not bind outputs")
        for relative, expected in files.items():
            rel = Path(relative)
            target = (root / rel).resolve()
            _require(not rel.is_absolute() and ".." not in rel.parts and root in target.parents,
                     "Completion path escapes its directory")
            self.bind(target, expected)
        return marker

    def recheck(self):
        for path, expected in self.files.items():
            _require(file_hash(path) == expected, "Source changed during analysis: " + path)


def _query_ids(path, expected_count):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    ids = np.asarray(sorted(int(row["sample_id"]) for row in rows), dtype=np.int64)
    _require(len(ids) == expected_count and len(set(ids.tolist())) == len(ids)
             and np.all(ids >= 0), "Frozen query manifest has unexpected identities/count")
    return ids


def _bound_path(source_files, basename, explicit=None):
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        _require(str(path) in source_files, "Explicit input is absent from the frozen source binding")
    else:
        paths = [Path(name) for name in source_files if Path(name).name == basename]
        _require(len(paths) == 1, "Ambiguous or missing bound source: " + basename)
        path = paths[0]
    return path, source_files[str(path)]


def _records(path, ids):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {int(row["query_id"]): row for row in rows}
    _require(len(rows) == len(by_id) == len(ids) and set(by_id) == set(map(int, ids)),
             "Per-query report is incomplete or has duplicate/foreign IDs")
    return [by_id[int(q)] for q in ids]


def load_method(exact_root, coverage_root, ids, labels, *, split, pool_size, binding=None):
    """Load and cross-check paired reports, ignoring implementation class IDs."""
    exact_root, coverage_root = Path(exact_root), Path(coverage_root)
    paths = [exact_root / "per_query.csv", coverage_root / "coverage_per_query.csv",
             exact_root / "metrics.json", coverage_root / "coverage_metrics.json"]
    if binding is not None:
        for path in paths:
            binding.bind(path)
    exact, coverage = _records(paths[0], ids), _records(paths[1], ids)
    labels = np.asarray(labels)
    _require(labels.ndim == 2 and labels.shape[1] == 8 and ids.ndim == 1 and len(ids) > 0
             and np.min(ids) >= 0 and np.max(ids) < len(labels)
             and np.isin(labels[ids], [0, 1]).all() and np.all(labels[ids].sum(1) > 0),
             "Invalid raw multi-hot labels for original query IDs")
    hits = np.empty(len(ids), dtype=np.int64)
    continuous = np.empty((len(ids), 2), dtype=np.float64)
    for i, (q, er, cr) in enumerate(zip(ids, exact, coverage)):
        label = labels[q].astype(np.int64)
        _require(er.get("label_signature") == "|".join(map(str, np.flatnonzero(label)))
                 and cr.get("label_signature") == "".join(map(str, label)),
                 "Per-query labels differ from the original NPZ row")
        _require(int(er["label_count"]) == int(cr["label_count"]) == int(label.sum())
                 and int(er["pool_size"]) == int(cr["pool_size"]) == pool_size
                 and int(er["returned"]) == int(cr["returned"]) == 500,
                 "Per-query labels, full candidate count or Top-500 scope differ")
        p = float(er["Precision@10"])
        values = [p, float(er["AP@100"]), float(cr["Coverage-nDCG@10"])]
        _require(all(math.isfinite(x) and 0 <= x <= 1 for x in values), "Invalid query metric")
        hits[i] = round(p * 10)
        _require(abs(p - hits[i] / 10) <= 1e-12, "P@10 is not an integer hit fraction")
        continuous[i] = values[1:]
    values = np.column_stack((hits / 10.0, continuous))
    means = np.asarray([math.fsum(values[:, i].tolist()) / len(ids) for i in range(3)])
    reports = [read_json(paths[2]), read_json(paths[3])]
    for report, family in zip(reports, ("exact", "coverage")):
        _require(report.get("split") == split and report.get("num_queries") == len(ids)
                 and report.get("metric_version") == family + "_v1",
                 "Aggregate report split, query count or metric family differs")
    expected = [reports[0]["metrics"]["Precision@10"], reports[0]["metrics"]["mAP@100"],
                reports[1]["metrics"]["Coverage-nDCG@10"]]
    _require(np.allclose(means, expected, atol=1e-12, rtol=0),
             "Aggregate metrics differ from paired original-ID query reports")
    return {"hits": hits, "continuous": continuous, "values": values,
            "means": means, "top10_hits": int(hits.sum())}


def label_strata(ids, labels):
    patterns = sorted({tuple(map(int, row)) for row in np.asarray(labels)[ids]})
    indices = [np.flatnonzero(np.all(np.asarray(labels)[ids] == pattern, axis=1))
               for pattern in patterns]
    return patterns, indices


def paired_bootstrap(hit_differences, continuous_differences, strata, *, repetitions=2000,
                     seed=2026, block_size=200):
    """Return repetitions × comparisons × three metric differences, as ratios.

    P@10 is accumulated as integer hit differences. Every metric/comparison
    shares the same query draws, stratified by raw complete label combination.
    """
    h, f = np.asarray(hit_differences), np.asarray(continuous_differences)
    _require(h.ndim == 2 and h.shape[0] > 0 and h.shape[1] > 0 and h.dtype.kind in "iu"
             and f.shape == (*h.shape, 2) and np.isfinite(f).all(), "Invalid paired difference arrays")
    _require(isinstance(repetitions, Integral) and not isinstance(repetitions, bool) and repetitions > 0
             and isinstance(seed, Integral) and not isinstance(seed, bool) and seed >= 0
             and isinstance(block_size, Integral) and not isinstance(block_size, bool) and block_size > 0,
             "Invalid bootstrap repetitions, seed or block size")
    strata = [np.asarray(indices) for indices in strata]
    _require(bool(strata) and all(x.ndim == 1 and len(x) > 0 and x.dtype.kind in "iu" for x in strata)
             and sorted(np.concatenate(strata).tolist()) == list(range(len(h))),
             "Strata must partition each paired query exactly once")
    rng = np.random.default_rng(seed)
    hit_totals = np.zeros((repetitions, h.shape[1]), dtype=np.int64)
    other_totals = np.zeros((repetitions, h.shape[1], 2), dtype=np.float64)
    for indices in strata:
        ch, cf = h[indices], f[indices]
        for start in range(0, repetitions, block_size):
            size = min(block_size, repetitions - start)
            draws = rng.integers(0, len(indices), size=(size, len(indices)))
            hit_totals[start:start + size] += ch[draws].sum(axis=1, dtype=np.int64)
            other_totals[start:start + size] += cf[draws].sum(axis=1, dtype=np.float64)
    result = np.empty((repetitions, h.shape[1], 3), dtype=np.float64)
    result[:, :, 0] = hit_totals / (10.0 * len(h))
    result[:, :, 1:] = other_totals / len(h)
    return result


def comparison_plan(selection):
    """Prespecified additive-winner contrasts, retaining duplicate role aliases."""
    left = selection["family_winners"]["additive"]
    targets = {"geometric_winner": selection["family_winners"]["geometric"],
               "product_reference": "product", "structure_reference": "J_only",
               "global_reference": "G_only", "WBBS": "WBBS"}
    unique, aliases = [], []
    for role in COMPARISON_ROLES:
        pair = (left, targets[role])
        if pair not in unique:
            unique.append(pair)
        aliases.append({"role": role, "left": pair[0], "right": pair[1],
                        "pair_index": unique.index(pair), "identical_variant": pair[0] == pair[1]})
    return unique, aliases


def _load_runs(valid_root, test_root, binding):
    valid_root, test_root = Path(valid_root).resolve(), Path(test_root).resolve()
    manifests = {}
    for root, split in ((valid_root, "valid"), (test_root, "test")):
        manifest = read_signed(binding.bind(root / "run_manifest.json"))
        _require(manifest.get("version") == "gj_fusion_v1" and manifest.get("split") == split,
                 "Expected completed G/J valid and test roots")
        binding.sealed(root, signature=digest(manifest["context"]), require_status=True)
        _require(read_json(root / "status.json").get("status") == "completed", "Run status is incomplete")
        manifests[split] = manifest
    frozen = read_signed(valid_root / "selection.json")
    index = read_json(valid_root / "evaluation_index_valid.json")
    expected = select_candidates(index["results"])
    _require(all(frozen.get(k) == value for k, value in expected.items())
             and frozen.get("context_signature") == digest(manifests["valid"]["context"])
             and frozen.get("valid_index_sha256") == file_hash(valid_root / "evaluation_index_valid.json"),
             "Frozen choice differs from the complete valid-only result grid")
    _require(read_signed(test_root / "frozen_selection.json") == frozen
             and manifests["test"]["context"].get("frozen_selection_signature") == frozen["signature"],
             "Test does not use the exact frozen valid choice")
    test_index = read_json(test_root / "evaluation_index_test.json")
    _require(test_index.get("variants") == frozen["selected_specs"]
             and set(test_index.get("results", {})) == set(frozen["selected_variants"]),
             "Test variants differ from the deduplicated frozen roles")
    for root, split, names in ((valid_root, "valid", expected["ranking"]),
                                (test_root, "test", frozen["selected_variants"])):
        names = [r["variant"] for r in names] if split == "valid" else names
        for name in names:
            folder = root / split / name
            checked = read_json(folder / "verification.json")
            _require(checked.get("status") == "verified" and checked.get("split") == split
                     and checked.get("variant") == name
                     and checked.get("num_queries") == (500 if split == "valid" else 7199)
                     and checked.get("all_queries_full_candidate_topk_exact") is True,
                     "Missing successful full-query independent verification")
    return frozen, manifests


def _group_summaries(method, split, roles, data, ids, labels):
    counts = np.asarray(labels)[ids].sum(1)
    groups = {"all": np.arange(len(ids)), "single_label": np.flatnonzero(counts == 1),
              "mixed_label": np.flatnonzero(counts > 1)}
    groups.update({"label_count_" + str(int(k)): np.flatnonzero(counts == k) for k in sorted(set(counts))})
    rows = []
    for group, indices in groups.items():
        if not len(indices):
            continue
        means = data["values"][indices].mean(0)
        rows.append(dict(method=method, split=split, roles="|".join(roles), group=group,
                         num_queries=len(indices), top10_hits=int(data["hits"][indices].sum()),
                         **dict(zip(METRICS, map(float, means)))))
    _, strata = label_strata(ids, labels)
    macro = np.asarray([data["values"][indices].mean(0) for indices in strata]).mean(0)
    rows.append(dict(method=method, split=split, roles="|".join(roles), group="class_macro",
                     num_queries=len(ids), top10_hits="", **dict(zip(METRICS, map(float, macro)))))
    return rows


def analyze(valid_root, test_root, *, wbbs_exact_root, wbbs_coverage_root,
            labels_npz=None, output_parent=None, repetitions=2000):
    _require(repetitions in (2000, 10000), "Formal analysis allows 2000 or 10000 paired resamples")
    started = now()
    valid_root, test_root = Path(valid_root).resolve(), Path(test_root).resolve()
    binding = SourceBinding()
    frozen, manifests = _load_runs(valid_root, test_root, binding)
    source_files = manifests["test"]["context"]["source_binding"]
    label_path, label_sha = _bound_path(source_files, "Wafer_Map_Datasets.npz", labels_npz)
    binding.bind(label_path, label_sha)
    with np.load(label_path, allow_pickle=False) as archive:
        labels = archive["arr_1"]
    ids = {}
    for split, count in (("valid", 500), ("test", 7199)):
        path, sha = _bound_path(source_files, split + "_queries.csv")
        ids[split] = _query_ids(binding.bind(path, sha), count)
    all_ids = np.concatenate((ids["valid"], ids["test"]))
    _require(labels.ndim == 2 and labels.shape[1] == 8 and np.max(all_ids) < len(labels)
             and np.isin(labels[all_ids], [0, 1]).all() and np.all(labels[all_ids].sum(1) > 0),
             "Bound NPZ does not supply valid eight-bit defect labels for the frozen query IDs")
    patterns, strata = label_strata(ids["test"], labels)
    _require(len(patterns) == 37, "Formal test must contain the 37 frozen complete label combinations")
    checked_markers = {}
    for folder, filenames in ((Path(wbbs_exact_root).resolve(), ("per_query.csv", "metrics.json")),
                              (Path(wbbs_coverage_root).resolve(), ("coverage_per_query.csv", "coverage_metrics.json"))):
        marker_root = folder if (folder / "completed.json").exists() else folder.parent
        if marker_root not in checked_markers:
            checked_markers[marker_root] = binding.sealed(marker_root)
        marker = checked_markers[marker_root]
        for filename in filenames:
            relative = str((folder / filename).relative_to(marker_root))
            _require(relative in marker["files"], "WBBS completion does not bind required report")
    loaded, summary = {}, []
    for split, root, names in (("valid", valid_root, [x["variant"] for x in frozen["ranking"]]),
                               ("test", test_root, frozen["selected_variants"])):
        loaded[split] = {}
        for name in names:
            folder = root / split / name
            data = load_method(folder, folder, ids[split], labels, split=split,
                               pool_size=3598 if split == "valid" else 7198, binding=binding)
            loaded[split][name] = data
            summary.extend(_group_summaries(name, split, frozen["variant_roles"][name], data, ids[split], labels))
    loaded["test"]["WBBS"] = load_method(wbbs_exact_root, wbbs_coverage_root, ids["test"], labels,
                                         split="test", pool_size=7198, binding=binding)
    summary.extend(_group_summaries("WBBS", "test", ["external_reference"], loaded["test"]["WBBS"], ids["test"], labels))
    pairs, aliases = comparison_plan(frozen)
    h = np.column_stack([loaded["test"][a]["hits"] - loaded["test"][b]["hits"] for a, b in pairs])
    f = np.stack([loaded["test"][a]["continuous"] - loaded["test"][b]["continuous"] for a, b in pairs], axis=1)
    binding.bind(__file__)
    # The plan and sources are frozen before generating any bootstrap draws.
    parent = Path(output_parent).resolve() if output_parent else ROOT / "artificial"
    for source in (valid_root, test_root, Path(wbbs_exact_root).resolve(), Path(wbbs_coverage_root).resolve()):
        _require(parent != source and source not in parent.parents, "Analysis must be outside its source runs")
    output = parent / ("gj_fusion_analysis_" + started.strftime("%Y%m%dT%H%M%S%z"))
    output.mkdir(parents=True, exist_ok=False)
    manifest = signed(dict(schema_version=1, analysis_version="gj_fusion_analysis_v1",
        run_id=started.strftime("%Y%m%dT%H%M%S%z") + "__gj_fusion_analysis",
        started_at=started.isoformat(), valid_root=str(valid_root), test_root=str(test_root),
        frozen_selection_signature=frozen["signature"], overall_winner=frozen["winner"],
        family_winners=frozen["family_winners"], comparison_aliases=aliases,
        unique_pairs=[list(pair) for pair in pairs], no_test_parameter_selection=True,
        query_ids_digest=digest(ids["test"].tolist()),
        raw_query_labels_digest=digest(labels[ids["test"]].astype(int).tolist()),
        source_hashes=dict(sorted(binding.files.items())), repetitions=int(repetitions), seed=2026,
        metrics=list(METRICS), units="ratios; multiply by 100 for percentages/percentage-point differences",
        stratum_sizes={"".join(map(str, pattern)): len(indices) for pattern, indices in zip(patterns, strata)},
        estimator="query-mean paired difference; raw 8-bit multi-hot stratification; class IDs ignored",
        sampling="with replacement within each stratum, preserving class counts; same draws for every contrast and metric",
        bootstrap_order="lexicographic label tuple; ascending original query ID; stratum outer and blocks of 200 draws inner",
        interval="pointwise percentile 95%, numpy linear quantile",
        limitations="Conditional on fixed models, scores, gallery and observed class proportions. No gallery/dataset or training-seed resampling; no correction for prior test-informed research or multiple comparisons. Not a simultaneous coverage guarantee."))
    write_json(output / "analysis_manifest.json", manifest)
    samples = paired_bootstrap(h, f, strata, repetitions=repetitions, seed=2026)
    rows, intervals = [], []
    for index, (left, right) in enumerate(pairs):
        points = np.r_[h[:, index].sum() / (10.0 * len(h)), f[:, index].mean(0)]
        pair_rows = []
        for k, metric in enumerate(METRICS):
            lower, upper = np.quantile(samples[:, index, k], [.025, .975], method="linear")
            row = dict(pair_index=index, left=left, right=right,
                roles="|".join(a["role"] for a in aliases if a["pair_index"] == index),
                metric=metric, difference=float(points[k]), ci95_lower=float(lower), ci95_upper=float(upper),
                identical_variant=left == right, num_queries=len(h))
            rows.append(row); pair_rows.append(row)
        intervals.append(dict(left=left, right=right, metrics=pair_rows))
    binding.recheck()
    write_csv(output / "summary.csv", summary,
              ["method", "split", "roles", "group", "num_queries", "top10_hits", *METRICS])
    write_csv(output / "paired_bootstrap.csv", rows,
              ["pair_index", "left", "right", "roles", "metric", "difference", "ci95_lower", "ci95_upper", "identical_variant", "num_queries"])
    write_json(output / "paired_bootstrap.json", dict(status="completed", manifest_signature=manifest["signature"],
        finished_at=now().isoformat(), comparison_aliases=aliases, comparisons=intervals,
        units="ratios", repetitions=repetitions, seed=2026, no_test_parameter_selection=True))
    files = {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output / "completed.json", dict(status="completed", signature=manifest["signature"], files=files))
    print("ANALYSIS_COMPLETED=" + str(output), flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid-root", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--labels-npz", help="Optional; otherwise read the raw NPZ path from the frozen source binding")
    parser.add_argument("--wbbs-exact-root", required=True)
    parser.add_argument("--wbbs-coverage-root", required=True)
    parser.add_argument("--output-parent", default=str(ROOT / "artificial"))
    parser.add_argument("--replicates", type=int, choices=(2000, 10000), default=2000)
    args = parser.parse_args(argv)
    analyze(args.valid_root, args.test_root, labels_npz=args.labels_npz,
            wbbs_exact_root=args.wbbs_exact_root, wbbs_coverage_root=args.wbbs_coverage_root,
            output_parent=args.output_parent, repetitions=args.replicates)


if __name__ == "__main__":
    main()
