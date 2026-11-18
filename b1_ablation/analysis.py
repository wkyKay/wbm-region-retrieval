"""Read-only B1 postprocessing; outputs belong to a new analysis directory.

All inference conditions are explicit: fixed models/gallery, post-hoc analysis,
stratified query bootstrap, no multiplicity correction or confirmatory claims.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import warnings

import numpy as np

ERRORS = ("exact", "extra_only", "missing_only", "missing_and_extra", "disjoint")
LABEL_NAMES = ("center", "donut", "edge-loc", "edge-ring", "loc", "near-full", "scratch", "random")
DEFAULT_KS = (1, 5, 10, 20, 50, 100, 500)
INFERENCE_SCOPE = ("Post-hoc descriptive paired query bootstrap conditional on frozen models, "
                   "gallery and class composition; shared-gallery dependence, training-seed, "
                   "split and gallery uncertainty are not estimated. Percentile intervals "
                   "are not multiplicity-corrected; no confirmatory significance claim.")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_signature(value):
    """Accept legacy index lists and coverage's eight-character bit strings."""
    value = str(value).strip()
    if len(value) == 8 and set(value) <= {"0", "1"}:
        result = tuple(int(x) for x in value)
    else:
        parts = value.split("|") if value else []
        indices = [int(x) for x in parts]
        if len(indices) != len(set(indices)) or any(i < 0 or i >= 8 for i in indices):
            raise ValueError("Invalid label signature")
        result = tuple(int(i in indices) for i in range(8))
    if not any(result):
        raise ValueError("Normal/empty labels are not retrieval queries")
    return result


def canonical(labels, sample_id):
    values = np.asarray(labels[int(sample_id)])
    if values.shape != (8,) or not np.isin(values, (0, 1)).all() or not values.any():
        raise ValueError("Expected nonempty frozen eight-dimensional multi-hot labels")
    return tuple(map(int, values))


def signature(label):
    return "".join(map(str, label))


def label_name(label):
    return "+".join(name for name, present in zip(LABEL_NAMES, label) if present)


def unique_csv(path, key="query_id"):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            identity = int(row[key])
            if identity in rows:
                raise ValueError("Duplicate %s in %s" % (key, path))
            rows[identity] = row
    return rows


def metric_fields(ks, family):
    prefixes = (("Hit", "Precision", "Recall", "AP") if family == "exact" else
                ("Coverage-nDCG", "MeanCoverage", "FullCoverageHit",
                 "FullCoveragePrecision", "FullCoverageRecall", "FullCoverageAP"))
    return [prefix + "@" + str(k) for k in ks for prefix in prefixes]


def report_name(field):
    if field.startswith("AP@"):
        return "mAP" + field[2:]
    if field.startswith("FullCoverageAP@"):
        return "FullCoverage_mAP" + field[len("FullCoverageAP"):]
    return field


def mean(values, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=axis)


def close(actual, expected, context):
    expected = np.nan if expected in (None, "") else float(expected)
    if np.isnan(actual) and np.isnan(expected):
        return
    if not np.isfinite(actual) or not np.isfinite(expected) or abs(actual - expected) > 1e-11:
        raise ValueError("Aggregation mismatch: " + context)


def group_indices(query_ids, frozen):
    classes = {}
    for i, q in enumerate(query_ids):
        classes.setdefault(frozen[q], []).append(i)
    classes = {key: np.asarray(classes[key], dtype=int) for key in sorted(classes)}
    counts = np.asarray([sum(frozen[q]) for q in query_ids])
    groups = {"all": np.arange(len(query_ids)), "single_label": np.flatnonzero(counts == 1),
              "mixed_label": np.flatnonzero(counts > 1)}
    groups.update({"label_count_" + str(k): np.flatnonzero(counts == k) for k in sorted(set(counts))})
    groups.update({"class_" + signature(key): indices for key, indices in classes.items()})
    return classes, {key: indices for key, indices in groups.items() if len(indices)}


def aggregate(values, classes, groups):
    output = {key: mean(values[indices], axis=0) for key, indices in groups.items()}
    output["macro_class"] = mean(np.stack([mean(values[i], axis=0) for i in classes.values()]), axis=0)
    # Exact weighted recomposition of class means into overall means, per defined metric.
    valid_counts = np.stack([np.isfinite(values[i]).sum(axis=0) for i in classes.values()])
    sums = np.stack([np.nansum(values[i], axis=0) for i in classes.values()]).sum(axis=0)
    denominators = valid_counts.sum(axis=0)
    recomposed = np.divide(sums, denominators, out=np.full_like(sums, np.nan), where=denominators > 0)
    if not np.allclose(recomposed, output["all"], rtol=0, atol=1e-12, equal_nan=True):
        raise ValueError("Class weighting did not recover overall aggregation")
    return output


def _load_family(directory, family, query_ids, frozen, gallery_labels, ks, classes, groups):
    directory = Path(directory)
    prefix = "" if family == "exact" else "coverage_"
    rows = unique_csv(directory / (prefix + "per_query.csv"))
    if set(rows) != set(query_ids):
        raise ValueError("Query IDs differ from frozen protocol")
    label_counts = {}
    for item in gallery_labels.values():
        label_counts[item] = label_counts.get(item, 0) + 1
    full_counts = {key: sum(n for label, n in label_counts.items()
                           if all(not value or label[i] for i, value in enumerate(key))) - 1
                   for key in classes}
    fields = metric_fields(ks, family)
    values = np.empty((len(query_ids), len(fields)), dtype=float)
    for i, q in enumerate(query_ids):
        row = rows[q]
        if parse_signature(row["label_signature"]) != frozen[q] or int(row["label_count"]) != sum(frozen[q]):
            raise ValueError("Query labels differ from frozen labels")
        if int(row["pool_size"]) != len(gallery_labels) - 1:
            raise ValueError("Incorrect complete candidate pool size")
        if int(row["returned"]) != max(ks):
            raise ValueError("Incorrect saved ranking length")
        denominator_field = "relevant_in_pool" if family == "exact" else "full_coverage_in_pool"
        denominator = label_counts[frozen[q]] - 1 if family == "exact" else full_counts[frozen[q]]
        if int(row[denominator_field]) != denominator:
            raise ValueError("Incorrect complete-gallery relevant denominator")
        for j, field in enumerate(fields):
            value = row[field]
            values[i, j] = np.nan if value == "" else float(value)
            if value != "" and (not math.isfinite(values[i, j]) or not -1e-12 <= values[i, j] <= 1 + 1e-12):
                raise ValueError("Invalid metric value")
    aggregated = aggregate(values, classes, groups)
    report = json.loads((directory / (prefix + "metrics.json")).read_text(encoding="utf-8"))
    if report.get("num_queries") != len(query_ids) or list(report.get("ks", [])) != list(ks):
        raise ValueError("Summary dimensions differ from frozen evaluation")
    for group, vector in aggregated.items():
        if group == "all":
            expected = report["metrics"]
        elif group == "macro_class":
            expected = report["class_macro_metrics"]
        elif group in ("single_label", "mixed_label"):
            expected = report["groups"][group]["metrics"]
        elif group.startswith("label_count_"):
            expected = report["groups"]["by_label_count"][group[len("label_count_"):]]["metrics"]
        else:
            continue
        for field, actual in zip(fields, vector):
            close(actual, expected[report_name(field)], family + "/" + group + "/" + field)
    with (directory / (prefix + "per_class.csv")).open(newline="", encoding="utf-8") as handle:
        class_rows = {}
        for row in csv.DictReader(handle):
            key = parse_signature(row["label_signature"])
            if key in class_rows:
                raise ValueError("Duplicate canonical class")
            class_rows[key] = row
    if set(class_rows) != set(classes):
        raise ValueError("Per-class coverage differs from frozen labels")
    for key, indices in classes.items():
        row = class_rows[key]
        if int(row["num_queries"]) != len(indices):
            raise ValueError("Wrong class query count")
        for field, actual in zip(fields, aggregated["class_" + signature(key)]):
            close(actual, row[report_name(field)], family + "/class/" + field)
    return {"rows": rows, "fields": fields, "values": values, "aggregated": aggregated}


def paired_bootstrap(deltas, strata, replicates=10000, seed=2026):
    """Columns share every resampled query; strata map canonical labels to indices."""
    deltas = np.asarray(deltas, dtype=float)
    if deltas.ndim != 2 or not np.isfinite(deltas).all() or replicates < 1:
        raise ValueError("Bootstrap expects finite N by comparison matrix")
    flattened = np.concatenate([np.asarray(strata[k], dtype=int) for k in sorted(strata)])
    if sorted(flattened.tolist()) != list(range(len(deltas))):
        raise ValueError("Strata must partition all queries exactly")
    rng = np.random.default_rng(seed)
    bootstrap = np.zeros((replicates, deltas.shape[1]), dtype=float)
    for key in sorted(strata):
        indices = np.asarray(strata[key], dtype=int)
        # The same matrix is used across comparisons; batches bound temporary memory.
        for start in range(0, replicates, 250):
            end = min(start + 250, replicates)
            draws = rng.integers(0, len(indices), size=(end - start, len(indices)))
            bootstrap[start:end] += deltas[indices[draws]].sum(axis=1)
    bootstrap /= len(deltas)
    return {"difference": deltas.mean(axis=0), "lower": np.percentile(bootstrap, 2.5, axis=0),
            "upper": np.percentile(bootstrap, 97.5, axis=0)}


def error_category(query, candidate):
    query, candidate = np.asarray(query, dtype=bool), np.asarray(candidate, dtype=bool)
    if query.shape != candidate.shape or not query.any() or not candidate.any():
        raise ValueError("Error taxonomy requires nonempty equal-sized labels")
    missing = np.any(query & ~candidate)
    extra = np.any(candidate & ~query)
    if not missing and not extra:
        return "exact"
    if not missing:
        return "extra_only"
    if not extra:
        return "missing_only"
    return "missing_and_extra" if np.any(query & candidate) else "disjoint"


def select_cases(query_ids, classes, deltas, comparisons):
    """Per comparison and class: largest positive/negative delta, tie by original ID."""
    cases = []
    for col, name in enumerate(comparisons):
        for key in sorted(classes):
            indices = classes[key]
            for direction, eligible in (
                    ("improved", [i for i in indices if deltas[i, col] > 1e-12]),
                    ("degraded", [i for i in indices if deltas[i, col] < -1e-12])):
                if not eligible:
                    continue
                selected = min(eligible, key=lambda i: ((-deltas[i, col] if direction == "improved"
                                                         else deltas[i, col]), query_ids[i]))
                cases.append({"comparison": name, "direction": direction, "query_id": query_ids[selected],
                              "label_signature": signature(key), "label_name": label_name(key),
                              "delta_Precision_at_10": float(deltas[selected, col])})
    return cases


def stream_ranking_errors(path, method, query_ids, frozen, gallery_labels, exact_rows, coverage_rows,
                          case_ids, output_handle, ks):
    """Stream every ranking row; retain only one Top10 plus preselected case rows."""
    positions = {q: i for i, q in enumerate(query_ids)}
    error_values = np.zeros((len(query_ids), 2, len(ERRORS)), dtype=float)
    case_rows, seen_queries = {}, set()
    digest = hashlib.sha256()
    metadata_before = Path(path).stat()
    with Path(path).open("rb") as binary:
        def lines():
            for line in binary:
                digest.update(line)
                yield line.decode("utf-8")
        reader = csv.DictReader(lines())
        current, count, used, top10, previous_score = None, 0, set(), [], math.inf

        def finish(q, returned, top):
            if q is None:
                return
            if returned != max(ks) or len(top) != 10:
                raise ValueError("Ranking is truncated")
            for ki, k in enumerate((1, 10)):
                counts = {key: 0 for key in ERRORS}
                for row in top[:k]:
                    counts[error_category(frozen[q], gallery_labels[int(row["candidate_id"])])] += 1
                if sum(counts.values()) != k:
                    raise ValueError("Error categories do not partition returns")
                close(counts["exact"] / k, exact_rows[q]["Precision@" + str(k)], "ranking exact prefix")
                close((counts["exact"] + counts["extra_only"]) / k,
                      coverage_rows[q]["FullCoveragePrecision@" + str(k)], "ranking full-coverage prefix")
                error_values[positions[q], ki] = [counts[key] / k for key in ERRORS]
                output_handle.writerow({"method": method, "query_id": q, "k": k,
                    "label_signature": signature(frozen[q]), "label_count": sum(frozen[q]),
                    **{key + "_count": counts[key] for key in ERRORS},
                    **{key + "_fraction": counts[key] / k for key in ERRORS}})
            if q in case_ids:
                case_rows[str(q)] = [{"candidate_labels": signature(gallery_labels[int(row["candidate_id"])]),
                    "candidate_label_name": label_name(gallery_labels[int(row["candidate_id"])]),
                    "error_category": error_category(frozen[q], gallery_labels[int(row["candidate_id"])]),
                    **row} for row in top]

        for row in reader:
            q, rank, candidate = int(row["query_id"]), int(row["rank"]), int(row["candidate_id"])
            if q != current:
                finish(current, count, top10)
                if q not in positions or q in seen_queries:
                    raise ValueError("Invalid/duplicate ranking query block")
                seen_queries.add(q)
                current, count, used, top10, previous_score = q, 0, set(), [], math.inf
            count += 1
            if rank != count or rank > max(ks) or candidate == q or candidate not in gallery_labels or candidate in used:
                raise ValueError("Invalid rank or candidate identity")
            used.add(candidate)
            score = float(row.get("similarity_score", row.get("score", "nan")))
            if not math.isfinite(score) or score > previous_score:
                raise ValueError("Ranking scores are invalid or not descending")
            previous_score = score
            if rank <= 10:
                top10.append(row)
        finish(current, count, top10)
    if seen_queries != set(query_ids):
        raise ValueError("Ranking misses frozen queries")
    metadata_after = Path(path).stat()
    if (metadata_before.st_size, metadata_before.st_mtime_ns) != (metadata_after.st_size, metadata_after.st_mtime_ns):
        raise ValueError("Source ranking changed while reading")
    return error_values, case_rows, digest.hexdigest()


def validate_completed(output, plan, normalized, report_hashes):
    """Reuse only when plan, all result files and every source still match."""
    output = Path(output)
    marker = json.loads((output / "completed.json").read_text(encoding="utf-8"))
    if marker.get("status") != "complete":
        raise ValueError("Analysis completion marker is not complete")
    if json.loads((output / "analysis_plan.json").read_text(encoding="utf-8")) != plan:
        raise ValueError("Completed analysis plan/source reports changed")
    for filename, expected in marker["files"].items():
        if Path(filename).name != filename or file_hash(output / filename) != expected:
            raise ValueError("Completed analysis artifact changed: " + filename)
    current = {name: dict(hashes) for name, hashes in report_hashes.items()}
    for name, (_, _, ranking) in normalized.items():
        current[name][str(ranking.resolve())] = file_hash(ranking)
    if json.loads((output / "sources.json").read_text(encoding="utf-8")) != current:
        raise ValueError("Completed analysis source changed")
    return marker


def postprocess(output, sources, labels, query_ids, gallery_ids, *, split="test", ks=DEFAULT_KS,
                bootstrap_replicates=10000, seed=2026, anchor="J_2", comparisons=None):
    """sources[name]={exact_dir, coverage_dir?, ranking_path?, completion_path?}.

    Source directories must already contain complete per-query/per-class reports,
    summary JSON and rankings. Only output is written. Caller verifies upstream
    signatures/completion; this function verifies reports and ranking identities.
    """
    output = Path(output)
    query_ids, gallery_ids = list(map(int, query_ids)), list(map(int, gallery_ids))
    if len(set(query_ids)) != len(query_ids) or len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError("Duplicate frozen protocol IDs")
    query_ids = sorted(query_ids)
    if not query_ids or not set(query_ids) <= set(gallery_ids) or max(ks) > len(gallery_ids) - 1:
        raise ValueError("Invalid full-gallery protocol")
    if 1 not in ks or 10 not in ks:
        raise ValueError("Postprocessing requires K=1 and K=10")
    frozen = {q: canonical(labels, q) for q in query_ids}
    gallery_labels = {q: canonical(labels, q) for q in gallery_ids}
    classes, groups = group_indices(query_ids, frozen)
    if anchor not in sources:
        raise ValueError("Missing anchor method")
    comparisons = list(comparisons) if comparisons is not None else [name for name in sources if name != anchor]
    if not comparisons or len(set(comparisons)) != len(comparisons) or any(n not in sources or n == anchor for n in comparisons):
        raise ValueError("Invalid comparisons")
    normalized, source_hashes = {}, {}
    for name, source in sources.items():
        source = {"exact_dir": source} if isinstance(source, (str, Path)) else dict(source)
        exact = Path(source["exact_dir"])
        coverage = Path(source.get("coverage_dir", exact))
        ranking = Path(source.get("ranking_path", exact / "rankings.csv"))
        paths = [exact / x for x in ("per_query.csv", "per_class.csv", "metrics.json")]
        paths += [coverage / x for x in ("coverage_per_query.csv", "coverage_per_class.csv", "coverage_metrics.json")]
        if "completion_path" in source:
            paths.append(Path(source["completion_path"]))
        if (exact / "timing.json").is_file():
            paths.append(exact / "timing.json")
        for path in paths + [ranking]:
            if not path.is_file():
                raise FileNotFoundError(path)
        source_hashes[name] = {str(path.resolve()): file_hash(path) for path in paths}
        normalized[name] = (exact, coverage, ranking)
    # No output is created until all required source files exist.
    plan = {"split": split, "query_ids": query_ids, "gallery_ids": sorted(gallery_ids),
            "frozen_query_labels_sha256": hashlib.sha256(json.dumps(
                [[q, frozen[q]] for q in query_ids], separators=(",", ":")).encode()).hexdigest(),
            "ks": list(ks), "anchor": anchor, "comparisons": comparisons,
            "bootstrap_replicates": bootstrap_replicates, "seed": seed,
            "primary_metric": "exact/Precision@10",
            "secondary_metrics": ["exact/AP@100", "coverage/Coverage-nDCG@10"],
            "strata": [{"label_signature": signature(key), "n": len(indices)} for key, indices in classes.items()],
            "bootstrap_order": "lexicographic multi-hot strata, ascending original query_id within stratum; all comparisons share draws",
            "scope": INFERENCE_SCOPE, "source_report_hashes": source_hashes}
    if (output / "completed.json").exists():
        return validate_completed(output, plan, normalized, source_hashes)
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "analysis_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Incomplete analysis has a different plan or source reports")
    write_json(plan_path, plan)
    loaded = {}
    summary_rows = []
    for name, (exact, coverage, _) in normalized.items():
        print("Analysis: verify reports", split, name, flush=True)
        loaded[name] = {family: _load_family(directory, family, query_ids, frozen, gallery_labels, ks, classes, groups)
                        for family, directory in (("exact", exact), ("coverage", coverage))}
        for family, data in loaded[name].items():
            for group, vector in data["aggregated"].items():
                n = len(query_ids) if group == "macro_class" else len(groups[group])
                for field, value in zip(data["fields"], vector):
                    summary_rows.append({"method": name, "split": split, "family": family,
                        "group": group, "num_queries": n, "metric": report_name(field),
                        "value": None if np.isnan(value) else float(value)})
    write_csv(output / "metrics.csv",
              ["method", "split", "family", "group", "num_queries", "metric", "value"], summary_rows)
    metric_specs = [("exact", "Precision@10"), ("exact", "AP@100"), ("coverage", "Coverage-nDCG@10")]
    available_specs = [(f, m) for f, m in metric_specs if m in loaded[anchor][f]["fields"]]
    columns, differences = [], []
    for family, metric in available_specs:
        field_index = loaded[anchor][family]["fields"].index(metric)
        for other in comparisons:
            j = loaded[other][family]["fields"].index(metric)
            delta = loaded[anchor][family]["values"][:, field_index] - loaded[other][family]["values"][:, j]
            if not np.isfinite(delta).all():
                raise ValueError("Primary/secondary paired metric is undefined")
            columns.append((family, metric, other))
            differences.append(delta)
    deltas = np.stack(differences, axis=1)
    bootstrap = paired_bootstrap(deltas, classes, bootstrap_replicates, seed)
    bootstrap_rows = [{"anchor": anchor, "other": other, "family": family, "metric": report_name(metric),
                       "difference_pp": float(100 * bootstrap["difference"][i]),
                       "lower_95_pp": float(100 * bootstrap["lower"][i]),
                       "upper_95_pp": float(100 * bootstrap["upper"][i]),
                       "query_wins": int(np.count_nonzero(deltas[:, i] > 1e-12)),
                       "query_ties": int(np.count_nonzero(np.abs(deltas[:, i]) <= 1e-12)),
                       "query_losses": int(np.count_nonzero(deltas[:, i] < -1e-12))}
                      for i, (family, metric, other) in enumerate(columns)]
    write_csv(output / "paired_bootstrap.csv", list(bootstrap_rows[0]), bootstrap_rows)
    write_json(output / "paired_bootstrap.json", {"scope": INFERENCE_SCOPE, "replicates": bootstrap_replicates,
                "seed": seed, "interval": "unadjusted 95% percentile", "results": bootstrap_rows})
    primary_deltas = deltas[:, :len(comparisons)]
    cases = select_cases(query_ids, classes, primary_deltas, comparisons)
    write_json(output / "case_selection.json", {"rule": "Within each frozen class and comparison choose largest positive and largest negative P@10 delta; tie by smallest original query_id. Include both directions when present.",
                                              "cases": cases})
    case_ids = {item["query_id"] for item in cases}
    error_fields = ["method", "query_id", "k", "label_signature", "label_count"]
    error_fields += [key + "_count" for key in ERRORS] + [key + "_fraction" for key in ERRORS]
    error_summary = []
    case_candidates = {}
    with (output / "errors_per_query.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=error_fields)
        writer.writeheader()
        for name, (exact, coverage, ranking) in normalized.items():
            print("Analysis: stream ranking/errors", split, name, flush=True)
            error_values, selected, ranking_hash = stream_ranking_errors(ranking, name, query_ids, frozen,
                gallery_labels, loaded[name]["exact"]["rows"], loaded[name]["coverage"]["rows"],
                case_ids, writer, ks)
            source_hashes[name][str(ranking.resolve())] = ranking_hash
            case_candidates[name] = selected
            for ki, k in enumerate((1, 10)):
                for group, vector in aggregate(error_values[:, ki], classes, groups).items():
                    for error, value in zip(ERRORS, vector):
                        error_summary.append({"method": name, "group": group, "k": k,
                                              "error_category": error, "fraction": float(value)})
    write_csv(output / "errors_by_group.csv", ["method", "group", "k", "error_category", "fraction"], error_summary)
    write_json(output / "case_candidates.json", {"scores_are_source_fields": True, "top_k": 10,
                                               "query_labels": {str(q): signature(frozen[q]) for q in sorted(case_ids)},
                                               "methods": case_candidates})
    timing = {}
    for name, (exact, _, _) in normalized.items():
        path = exact / "timing.json"
        if path.exists():
            timing[name] = {"source_path": str(path.resolve()), "source_sha256": file_hash(path),
                            "record": json.loads(path.read_text(encoding="utf-8"))}
    write_json(output / "timing_sources.json", {
        "scope": "Source phase records only. Shared ranking-batch time is not per-method latency, is not divided into inferred per-method cost, and is not a controlled benchmark.",
        "sources": timing})
    write_json(output / "sources.json", source_hashes)
    artifacts = [p for p in output.iterdir() if p.is_file() and p.name != "completed.json"]
    completed = {"status": "complete", "split": split, "num_queries": len(query_ids),
                 "num_classes": len(classes), "methods": list(sources),
                 "verified": ["frozen IDs/labels/pool denominators", "all-K per-class and group aggregation",
                              "Top1/Top10 error partitions and source prefix metrics",
                              "full saved-ranking identity/length/order"],
                 "scope": INFERENCE_SCOPE, "files": {p.name: file_hash(p) for p in artifacts}}
    write_json(output / "completed.json", completed)
    return completed
