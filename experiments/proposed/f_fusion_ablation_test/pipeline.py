"""Test-only evaluation; no parameter search or writes to the valid source."""
from contextlib import ExitStack
import csv
from pathlib import Path
import tempfile

import numpy as np

from experiments.proposed.f_global_fusion import pipeline as fusion
from experiments.proposed.f_segmentation_geometry import pipeline as geo
from wbm_retrieval.common import ROOT, file_hash, read_json, write_csv, write_json
from wbm_retrieval.exact_metrics import evaluate_rankings, order_scores
from experiments.proposed.f_fusion_ablation.scoring import describe, variant_score, variants


def load_selection(root):
    manifest = fusion.signed_manifest(root / "ablation_manifest.json")
    evaluation = fusion.signed_manifest(root / "evaluation_manifest.json")
    selection = fusion.signed_manifest(root / "selection.json")
    if (manifest.get("version") != "f_fusion_four_valid_v1" or
            manifest["plan"]["settings"]["alpha_global"] != .5 or
            evaluation["run_signature"] != manifest["signature"] or
            selection["evaluation_signature"] != evaluation["signature"] or
            selection["split"] != "valid" or selection["test_evaluated"] is not False or
            selection["alpha_global"] != .5 or selection["selection_metric"] != "exact/all/Precision@10" or
            selection["tie_rule"] != "smaller strength, including baseline zero"):
        raise ValueError("Expected the completed alpha=.5 valid selection")
    fusion.checked_files(ROOT, manifest["code"])
    specs, reports, hashes = variants(), {}, {}
    for name in specs:
        folder = root / "valid" / name
        if not geo.completed(folder, evaluation["signature"]):
            raise ValueError("Complete the valid sweep first")
        hashes[name] = file_hash(folder / "completed.json")
        reports[name] = read_json(folder / "metrics.json")
    if hashes != selection["completed_sha256"]:
        raise ValueError("Valid completed markers changed")
    chosen = {"baseline": "baseline", "optimal": "optimal"}
    for family in ("density", "angular"):
        candidates = [{"variant": name, "strength": strength,
                       "value": reports[name]["metrics"]["Precision@10"]}
                      for name, (group, strength) in specs.items() if group in ("baseline", family)]
        best = max(candidates, key=lambda c: (c["value"], -c["strength"]))
        if selection["selections"][family] != {"selected": best, "candidates": candidates}:
            raise ValueError("Selection does not match signed valid metrics")
        chosen[family] = best["variant"]
    chosen = {k: chosen[k] for k in ("baseline", "density", "angular", "optimal")}
    return manifest, selection, chosen


def summarize(output, signature, selected):
    reports = {}
    for name in dict.fromkeys(selected.values()):
        folder = output / "test" / name
        if not geo.completed(folder, signature):
            raise ValueError("Incomplete test reports")
        reports[name] = [read_json(folder / f) for f in ("metrics.json", "coverage_metrics.json")]
    def groups(report):
        return {"all": report["metrics"], "class_macro": report["class_macro_metrics"],
                **{g: report["groups"][g]["metrics"] for g in ("single_label", "mixed_label")},
                **{f"label_count_{k}": v["metrics"] for k, v in report["groups"]["by_label_count"].items()}}
    rows = []
    for name, values in reports.items():
        family, strength = variants()[name]
        for index, report in enumerate(values):
            base = groups(reports["baseline"][index])
            for group, metrics in groups(report).items():
                for metric, value in metrics.items():
                    b = base[group][metric]
                    rows.append({"split": "test", "variant": name, "family": family, "strength": strength,
                                 "alpha_global": .5, "group": group, "metric_version": report["metric_version"],
                                 "metric": metric, "value": value,
                                 "delta_vs_baseline": value-b if value is not None and b is not None else None})
    four = [{"experiment": family, **row} for family, name in selected.items() for row in rows if row["variant"] == name]
    write_csv(output / "comparison.csv", rows, list(rows[0]))
    write_csv(output / "four_groups.csv", four, list(four[0]))


def descriptors(output, maps, ids, signature):
    target = output / "descriptors"
    if geo.completed(target, signature):
        with np.load(target / "test.npz", allow_pickle=False) as archive:
            values = dict(archive)
        if not np.array_equal(values["sample_ids"], ids):
            raise ValueError("Cached descriptor ID order differs from test gallery")
        return values
    attempt = Path(tempfile.mkdtemp(prefix=".descriptors_attempt_", dir=output))
    rows = [describe(maps[int(s)]) for s in ids]
    values = {"sample_ids": np.asarray(ids, dtype=np.int64), "density": np.array([r[0] for r in rows]),
              "moments": np.array([r[1] for r in rows]), "outer_count": np.array([r[2] for r in rows])}
    np.savez_compressed(attempt / "test.npz", **values)
    geo.commit_marker(attempt, signature)
    attempt.rename(target)
    return values


def status(output, phase, **details):
    write_json(output / "status.json", {"status": "running", "split": "test", "test_evaluated": True,
                                        "phase": phase, **details})


def evaluate(output, signature, scorer, pool, bundle, ks, tie_seed, selected):
    if getattr(pool, "split_name", None) != "test":
        raise ValueError("This ablation only permits the frozen test pool")
    parent = output / "test"
    parent.mkdir(exist_ok=True)
    pending = []
    for name in dict.fromkeys(selected.values()):
        family, gamma = variants()[name]
        target = parent / name
        if not geo.completed(target, signature):
            attempt = Path(tempfile.mkdtemp(prefix=f".{name}_attempt_", dir=parent))
            pending.append((name, family, gamma, target, attempt))
    if not pending:
        print("Verified all completed test ablations", flush=True)
        return
    fields = ["query_id", "rank", "candidate_id", "similarity_score", "baseline_score", "f_score",
              "global_score", "optimal_f_score", "density_distance", "angular_distance"]
    with ExitStack() as stack:
        writers = []
        for name, family, gamma, _, attempt in pending:
            writer = csv.writer(stack.enter_context((attempt / "rankings.csv").open("w", newline="")))
            writer.writerow(fields)
            writers.append((family, gamma, writer))
        status(output, "scoring", queries_done=0, queries_total=len(pool), pending_variants=len(pending))
        for number, query in enumerate(pool, 1):
            candidates = pool[query]
            values = scorer.components(query, candidates)
            if any(not np.isfinite(value).all() for value in values.values()):
                raise ValueError("Nonfinite score component")
            for family, gamma, writer in writers:
                scores = variant_score(values, family, gamma)
                order = order_scores(candidates, scores, int(query), tie_seed)[:max(ks)]
                writer.writerows((int(query), rank, int(candidates[i]), float(scores[i]),
                                  *(float(values[k][i]) for k in ("baseline", "f_score", "global_score",
                                     "optimal_f_score", "density_distance", "angular_distance")))
                                 for rank, i in enumerate(order, 1))
            if number % 100 == 0 or number == len(pool):
                print(f"Ablation test: {number}/{len(pool)} queries, {len(pending)} variants", flush=True)
                status(output, "scoring", queries_done=number, queries_total=len(pool), pending_variants=len(pending))
    for index, (name, family, gamma, target, attempt) in enumerate(pending, 1):
        print(f"Metrics test: {index}/{len(pending)} {name} (exact + coverage)", flush=True)
        status(output, "metrics", current_variant=name, variants_done=index-1, variants_total=len(pending))
        metadata = {"signature": signature, "alpha_global": .5, "family": family, "strength": gamma,
                    "test_evaluated": True, "candidates_scored": "full_pool", "ranking_scope": "top_max_k"}
        evaluate_rankings(bundle, pool, fusion.iter_rankings(attempt / "rankings.csv"), ks, attempt,
                          split_name="test", metadata=metadata)
        geo.evaluate_rankings_file(bundle, pool, attempt / "rankings.csv", ks, attempt,
                                   split_name="test", metadata=metadata)
        geo.commit_marker(attempt, signature)
        attempt.rename(target)
