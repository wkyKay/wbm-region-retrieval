"""Stream full-gallery valid rankings; select each single-factor group independently."""
from contextlib import ExitStack
import csv
from pathlib import Path
import tempfile

import numpy as np

from experiments.proposed.f_global_fusion import pipeline as fusion
from experiments.proposed.f_segmentation_geometry import pipeline as geo
from wbm_retrieval.common import digest, file_hash, read_json, write_csv, write_json
from wbm_retrieval.exact_metrics import evaluate_rankings, order_scores
from .scoring import SETTINGS, describe, variants, variant_score


def descriptors(output, maps, ids, signature):
    target = output / "descriptors"
    if geo.completed(target, signature):
        with np.load(target / "valid.npz", allow_pickle=False) as archive:
            values = dict(archive)
        if not np.array_equal(values["sample_ids"], ids):
            raise ValueError("Cached descriptor ID order differs from valid gallery")
        return values
    attempt = Path(tempfile.mkdtemp(prefix=".descriptors_attempt_", dir=output))
    rows = [describe(maps[int(s)]) for s in ids]
    values = {"sample_ids": np.asarray(ids, dtype=np.int64), "density": np.array([r[0] for r in rows]),
              "moments": np.array([r[1] for r in rows]), "outer_count": np.array([r[2] for r in rows])}
    np.savez_compressed(attempt / "valid.npz", **values)
    geo.commit_marker(attempt, signature)
    attempt.rename(target)
    return values


def status(output, phase, **details):
    write_json(output / "status.json", {"status": "running", "split": "valid", "test_evaluated": False,
                                        "phase": phase, **details})


def evaluate(output, signature, scorer, pool, bundle, ks, tie_seed):
    if getattr(pool, "split_name", None) != "valid":
        raise ValueError("This ablation only permits the frozen valid pool")
    parent = output / "valid"
    parent.mkdir(exist_ok=True)
    pending = []
    for name, (family, gamma) in variants().items():
        target = parent / name
        if not geo.completed(target, signature):
            attempt = Path(tempfile.mkdtemp(prefix=f".{name}_attempt_", dir=parent))
            pending.append((name, family, gamma, target, attempt))
    if not pending:
        print("Verified all completed valid ablations", flush=True)
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
                print(f"Ablation valid: {number}/{len(pool)} queries, {len(pending)} variants", flush=True)
                status(output, "scoring", queries_done=number, queries_total=len(pool), pending_variants=len(pending))
    for index, (name, family, gamma, target, attempt) in enumerate(pending, 1):
        print(f"Metrics valid: {index}/{len(pending)} {name} (exact + coverage)", flush=True)
        status(output, "metrics", current_variant=name, variants_done=index-1, variants_total=len(pending))
        metadata = {"signature": signature, "alpha_global": .5, "family": family, "strength": gamma,
                    "test_evaluated": False, "candidates_scored": "full_pool", "ranking_scope": "top_max_k"}
        evaluate_rankings(bundle, pool, fusion.iter_rankings(attempt / "rankings.csv"), ks, attempt,
                          split_name="valid", metadata=metadata)
        geo.evaluate_rankings_file(bundle, pool, attempt / "rankings.csv", ks, attempt,
                                   split_name="valid", metadata=metadata)
        geo.commit_marker(attempt, signature)
        attempt.rename(target)


def summarize(output, signature):
    reports, selections, hashes = {}, {}, {}
    specs = variants()
    for name in specs:
        folder = output / "valid" / name
        if not geo.completed(folder, signature):
            raise ValueError("Complete all valid ablations before selection")
        hashes[name] = file_hash(folder / "completed.json")
        reports[name] = [read_json(folder / file) for file in ("metrics.json", "coverage_metrics.json")]
    for family in ("density", "angular"):
        candidates = [{"variant": name, "strength": strength,
                       "value": reports[name][0]["metrics"]["Precision@10"]}
                      for name, (group, strength) in specs.items() if group in ("baseline", family)]
        best = max(candidates, key=lambda x: (x["value"], -x["strength"]))
        selections[family] = {"selected": best, "candidates": candidates}
    payload = {"evaluation_signature": signature, "alpha_global": .5, "split": "valid",
               "test_evaluated": False, "selection_metric": "exact/all/Precision@10",
               "tie_rule": "smaller strength, including baseline zero", "selections": selections,
               "completed_sha256": hashes}
    selected = {"baseline": "baseline", "density": selections["density"]["selected"]["variant"],
                "angular": selections["angular"]["selected"]["variant"], "optimal": "optimal"}
    selection = {**payload, "signature": digest(payload)}
    path = output / "selection.json"
    if path.exists() and read_json(path) != selection:
        raise ValueError("Frozen valid selection changed; use a new output")
    if not path.exists():
        write_json(path, selection)

    def groups(report):
        return {"all": report["metrics"], "class_macro": report["class_macro_metrics"],
                **{g: report["groups"][g]["metrics"] for g in ("single_label", "mixed_label")},
                **{f"label_count_{k}": v["metrics"] for k, v in report["groups"]["by_label_count"].items()}}

    rows = []
    for name, (family, strength) in specs.items():
        for index, report in enumerate(reports[name]):
            baseline = groups(reports["baseline"][index])
            for group, metrics in groups(report).items():
                for metric, value in metrics.items():
                    base = baseline[group][metric]
                    rows.append({"split": "valid", "variant": name, "family": family, "strength": strength,
                                 "alpha_global": .5, "group": group, "metric_version": report["metric_version"],
                                 "metric": metric, "value": value,
                                 "delta_vs_baseline": value-base if value is not None and base is not None else None})
    four = [{"experiment": family, **row} for family, name in selected.items() for row in rows if row["variant"] == name]
    write_csv(output / "comparison.csv", rows, list(rows[0]))
    write_csv(output / "four_groups.csv", four, list(four[0]))
    return selected
