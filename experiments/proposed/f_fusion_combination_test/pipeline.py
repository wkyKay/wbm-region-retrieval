"""One additional full-gallery ranking, with read-only prior test comparisons."""
import csv
from pathlib import Path
import tempfile

import numpy as np

from experiments.proposed.f_global_fusion import pipeline as fusion
from experiments.proposed.f_segmentation_geometry import pipeline as geo
from wbm_retrieval.common import ROOT, file_hash, read_json, write_csv, write_json
from wbm_retrieval.exact_metrics import evaluate_rankings, order_scores

VARIANT = "optimal_angular_0.5"
SETTINGS = {"alpha_global": .5, "angular_gamma": .5, "density_gamma": 0.,
            "matching": "exact maximum sqrt(area)-weighted one-to-one assignment",
            "formula": "(0.5 * optimal_local + 0.5 * global) * exp(-0.5 * angular_distance)",
            "posthoc_test_combination": True, "test_parameter_search": False,
            "training_required": False}
REFERENCES = ("baseline", "angular_0.5", "optimal")


def combined_score(values):
    return values["optimal"] * np.exp(-SETTINGS["angular_gamma"] * values["angular_distance"])


def load_previous(root):
    root = Path(root)
    manifest = fusion.signed_manifest(root / "test_manifest.json")
    evaluation = fusion.signed_manifest(root / "evaluation_manifest.json")
    selection = fusion.signed_manifest(root / "frozen_selection.json")
    expected = {"baseline": "baseline", "density": "baseline", "angular": "angular_0.5", "optimal": "optimal"}
    if (manifest.get("version") != "f_fusion_four_test_v1" or manifest["alpha_global"] != .5 or
            manifest["selected_variants"] != expected or evaluation["run_signature"] != manifest["signature"] or
            selection["signature"] != manifest["selection_signature"] or selection["alpha_global"] != .5 or
            selection["split"] != "valid" or selection["test_evaluated"] is not False or
            selection["selections"]["angular"]["selected"]["variant"] != "angular_0.5" or
            selection["selections"]["angular"]["selected"]["strength"] != .5 or
            selection["selections"]["density"]["selected"]["strength"] != 0.):
        raise ValueError("Expected previous test with frozen alpha=.5, angular=.5, density=0")
    fusion.checked_files(ROOT, manifest["code"])
    fusion.checked_files(ROOT, manifest["valid_code"])
    reports, hashes = {}, {}
    for name in REFERENCES:
        folder = root / "test" / name
        if not geo.completed(folder, evaluation["signature"]):
            raise ValueError("Complete the previous four-group test first")
        hashes[name] = file_hash(folder / "completed.json")
        reports[name] = [read_json(folder / f) for f in ("metrics.json", "coverage_metrics.json")]
    if (not geo.completed(root / "descriptors", manifest["signature"]) or
            file_hash(root / "descriptors/completed.json") != evaluation["descriptors"]):
        raise ValueError("Previous test descriptors changed")
    with np.load(root / "descriptors/test.npz", allow_pickle=False) as archive:
        descriptors = dict(archive)
    provenance = {"files": {name: file_hash(root / name) for name in
                  ("test_manifest.json", "evaluation_manifest.json", "frozen_selection.json")},
                  "completed_sha256": hashes}
    return manifest, evaluation, reports, descriptors, provenance


def status(output, phase, **details):
    write_json(output / "status.json", {"status": "running", "split": "test", "variant": VARIANT,
               "posthoc_test_combination": True, "phase": phase, **details})


def evaluate(output, signature, scorer, pool, bundle, ks, tie_seed):
    if getattr(pool, "split_name", None) != "test":
        raise ValueError("Combination only permits the frozen test pool")
    parent = output / "test"
    parent.mkdir(exist_ok=True)
    target = parent / VARIANT
    if geo.completed(target, signature):
        print("Verified completed combination test", flush=True)
        return
    attempt = Path(tempfile.mkdtemp(prefix=f".{VARIANT}_attempt_", dir=parent))
    fields = ["query_id", "rank", "candidate_id", "similarity_score", "baseline_score", "angular_score",
              "optimal_score", "f_score", "optimal_f_score", "global_score", "angular_distance"]
    status(output, "scoring", queries_done=0, queries_total=len(pool))
    with (attempt / "rankings.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for number, query in enumerate(pool, 1):
            candidates = pool[query]
            values = scorer.components(query, candidates)
            if any(not np.isfinite(v).all() for v in values.values()):
                raise ValueError("Nonfinite score component")
            scores = combined_score(values)
            angular = values["baseline"] * np.exp(-.5 * values["angular_distance"])
            order = order_scores(candidates, scores, int(query), tie_seed)[:max(ks)]
            writer.writerows((int(query), rank, int(candidates[i]), float(scores[i]),
                              float(values["baseline"][i]), float(angular[i]),
                              *(float(values[k][i]) for k in ("optimal", "f_score", "optimal_f_score",
                                                            "global_score", "angular_distance")))
                             for rank, i in enumerate(order, 1))
            if number % 100 == 0 or number == len(pool):
                print(f"Combination test: {number}/{len(pool)} queries", flush=True)
                status(output, "scoring", queries_done=number, queries_total=len(pool))
    metadata = {"signature": signature, "variant": VARIANT, **SETTINGS,
                "candidates_scored": "full_pool", "ranking_scope": "top_max_k"}
    for family in ("exact", "coverage"):
        print(f"Combination test metrics: {family}", flush=True)
        status(output, "metrics", metric_family=family)
        if family == "exact":
            evaluate_rankings(bundle, pool, fusion.iter_rankings(attempt / "rankings.csv"), ks, attempt,
                              split_name="test", metadata=metadata)
        else:
            geo.evaluate_rankings_file(bundle, pool, attempt / "rankings.csv", ks, attempt,
                                       split_name="test", metadata=metadata)
    geo.commit_marker(attempt, signature)
    attempt.rename(target)


def summarize(output, signature, previous_reports):
    folder = output / "test" / VARIANT
    if not geo.completed(folder, signature):
        raise ValueError("Incomplete combination test")
    reports = {**previous_reports, VARIANT: [read_json(folder / f) for f in ("metrics.json", "coverage_metrics.json")]}
    def groups(report):
        return {"all": report["metrics"], "class_macro": report["class_macro_metrics"],
                **{g: report["groups"][g]["metrics"] for g in ("single_label", "mixed_label")},
                **{f"label_count_{k}": v["metrics"] for k, v in report["groups"]["by_label_count"].items()}}
    rows = []
    for name, values in reports.items():
        for index, report in enumerate(values):
            refs = {n: groups(reports[n][index]) for n in REFERENCES}
            for group, metrics in groups(report).items():
                for metric, value in metrics.items():
                    row = {"split": "test", "variant": name, "alpha_global": .5,
                           "angular_gamma": .5 if name in ("angular_0.5", VARIANT) else 0.,
                           "matching": "optimal" if name in ("optimal", VARIANT) else "greedy",
                           "posthoc_test_combination": name == VARIANT,
                           "group": group, "metric_version": report["metric_version"], "metric": metric, "value": value}
                    for ref, column in zip(REFERENCES, ("baseline", "angular", "optimal")):
                        b = refs[ref][group][metric]
                        row[f"delta_vs_{column}"] = value-b if value is not None and b is not None else None
                    rows.append(row)
    write_csv(output / "comparison.csv", rows, list(rows[0]))
    write_csv(output / "combination.csv", [r for r in rows if r["variant"] == VARIANT], list(rows[0]))
