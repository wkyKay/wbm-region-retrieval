#!/usr/bin/env python3
"""Valid-only paired F ablation; read an existing run and write a fresh directory."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def directional_scores(scores, areas, settings):
    """Batched original greedy matching, with independently ordered source rows.

    scores: [batch, source region, target region]; areas: [batch, source].
    Zero-area rows and zero-score columns are padding. Stable flatten-order ties
    match greedy_assignment, including when the matrix is transposed.
    """
    scores = np.clip(np.asarray(scores), 0, 1).copy()
    areas = np.asarray(areas, dtype=float)
    batch, nq, nc = scores.shape
    if areas.shape != (batch, nq) or np.any(areas < 0):
        raise ValueError("Invalid source areas")
    if not np.isfinite(scores).all() or not np.isfinite(areas).all():
        raise ValueError("Nonfinite input")
    if settings["lambda_pos"] or settings["lambda_scale"]:
        raise ValueError("F ablation requires shape-only matching")
    if not nq or not nc:
        return np.zeros(batch)
    scores *= (areas > 0)[:, :, None]
    scores[scores < settings["shape_threshold"]] = 0
    flat = scores.reshape(batch, -1)
    order = np.argsort(-flat, axis=1, kind="stable")
    weight = settings["weight"]
    weights = ((areas > 0).astype(float) if weight == "uniform" else
               np.sqrt(areas) if weight == "sqrt_area" else areas)
    used_q, used_c = np.zeros((batch, nq), bool), np.zeros((batch, nc), bool)
    total = np.zeros(batch)
    rows = np.arange(batch)
    for column in range(order.shape[1]):
        index = order[:, column]
        qi, ci = index // nc, index % nc
        value = flat[rows, index]
        take = ((value > 0) & (value >= settings["score_threshold"])
                & ~used_q[rows, qi] & ~used_c[rows, ci])
        total[take] += weights[rows[take], qi[take]] * value[take]
        used_q[rows[take], qi[take]] = True
        used_c[rows[take], ci[take]] = True
    denominator = weights.sum(axis=1)
    return np.divide(total, denominator, out=np.zeros(batch), where=denominator > 0)


def bidirectional_scores(scores, query_areas, candidate_areas, settings):
    forward = directional_scores(scores, query_areas, settings)
    reverse = directional_scores(scores.transpose(0, 2, 1), candidate_areas, settings)
    denominator = forward + reverse
    # Exact harmonic mean with an explicit zero boundary; no epsilon bias.
    harmonic = np.divide(2 * forward * reverse, denominator,
                         out=np.zeros_like(forward), where=denominator > 0)
    return forward, reverse, harmonic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True,
                        help="Existing suite run containing cache and suite_manifest.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    from experiments.proposed._shared.ef_suite import build_model, embeddings
    from wbm_retrieval.common import digest, file_hash, get_device, read_json, write_json
    from wbm_retrieval.data import decode_pool
    from wbm_retrieval.diagnostics import batched_map_scores
    from wbm_retrieval.exact_evaluation import evaluate_suite_rankings
    from wbm_retrieval.exact_metrics import order_scores
    from wbm_retrieval.training import load_checkpoint

    source, output = args.source_run.resolve(), args.output_dir.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Output must be separate from the source run")
    if output.exists():
        raise FileExistsError("Choose a new output directory")
    manifest = read_json(source / "suite_manifest.json")
    config = dict(manifest["settings"])
    config.update(seed=args.seed, device=args.device, num_workers=0)
    if config.get("data_protocol") != "deduplicated_v1" or config["evaluation"]["metric_version"] != "exact_v1":
        raise ValueError("Only frozen deduplicated exact_v1 runs are supported")
    checkpoint_path = source / f"seed_{args.seed}" / "F" / "best_nce.pt"
    checkpoint = load_checkpoint(checkpoint_path)
    expected = digest({"suite": manifest["signature"], "seed": args.seed, "variant": "F"})
    if checkpoint["signature"] != expected:
        raise ValueError("Checkpoint/suite signature mismatch")
    spec = manifest["suite"]["variants"]["F"]
    if checkpoint["input_spec"] != spec or checkpoint["variant"] != "F":
        raise ValueError("Checkpoint is not the source F variant")
    prepared = read_json(source / "cache" / "prepared.json")
    if prepared["signature"] != manifest["signature"]:
        raise ValueError("Cache/suite signature mismatch")
    required = ("cache/valid_regions.npz", "cache/valid_F.npy", "cache/labels.npy", "valid_pool.json")
    hashes = {}
    for name in required:
        hashes[name] = file_hash(source / name)
        if prepared["files"].get(name) != hashes[name]:
            raise ValueError(f"Unverified or modified source cache: {name}")
    pool_payload = read_json(source / "valid_pool.json")
    queries, candidates = decode_pool(pool_payload)
    def frozen_ids(name):
        with (ROOT / "protocol" / "retrieval_v1" / name).open() as handle:
            return [int(row["sample_id"]) for row in csv.DictReader(handle)]
    if (list(queries) != frozen_ids("valid_queries.csv") or
            list(pool_payload.get("gallery", [])) != frozen_ids("valid_gallery.csv")):
        raise ValueError("Source pool differs from frozen valid protocol")
    with np.load(source / "cache" / "valid_regions.npz") as archive:
        meta = dict(archive)
    if set(meta["map_ids"].tolist()) != set(pool_payload["gallery"]):
        raise ValueError("Region cache does not cover exactly the valid gallery")
    labels = np.load(source / "cache" / "labels.npy", mmap_mode="r")
    device = get_device(args.device)
    model = build_model(config, spec).to(device)
    model.load_state_dict(checkpoint["model"])
    output.mkdir(parents=True, exist_ok=False)
    provenance = {"source_run": str(source), "checkpoint": str(checkpoint_path),
                  "checkpoint_sha256": file_hash(checkpoint_path), "epoch": checkpoint["epoch"],
                  "suite_signature": manifest["signature"], "source_files": hashes,
                  "script_sha256": file_hash(__file__), "split": "valid", "test_accessed": False,
                  "formula": "2*a*b/(a+b); zero when a+b=0; independent directional greedy",
                  "protocol_signature": pool_payload["protocol_signature"]}
    write_json(output / "provenance.json", provenance)
    features = embeddings(model, source / "cache" / "valid_F.npy", config, device)
    np.save(output / "valid_h512.npy", features)
    indices = {int(i): np.arange(a, b) for i, a, b in
               zip(meta["map_ids"], meta["offsets"][:-1], meta["offsets"][1:])}
    width = int(np.diff(meta["offsets"]).max())
    padded = np.full((len(meta["map_ids"]), width), -1, dtype=int)
    areas = np.zeros(padded.shape)
    positions = {int(i): n for n, i in enumerate(meta["map_ids"])}
    for sample_id, tokens in indices.items():
        row = positions[sample_id]
        padded[row, :len(tokens)] = tokens
        areas[row, :len(tokens)] = meta["area"][tokens]
    ranked = {"baseline": [], "bidirectional": []}
    max_error = 0.0
    k = max(config["evaluation"]["ks"])
    for number, q in enumerate(queries, 1):
        c = np.asarray(candidates[q], dtype=int)
        tokens = indices[q]
        rows = [positions[int(i)] for i in c]
        ci = padded[rows]
        cosines = features[tokens] @ features.T
        matrices = cosines[:, np.maximum(ci, 0)].transpose(1, 0, 2).copy()
        matrices *= (ci >= 0)[:, None, :]
        qa = np.broadcast_to(meta["area"][tokens], (len(c), len(tokens)))
        forward, reverse, harmonic = bidirectional_scores(matrices, qa, areas[rows], config["matching"])
        reference = batched_map_scores(cosines, ci, meta["area"][tokens], config["matching"])
        np.testing.assert_allclose(forward, reference, rtol=1e-12, atol=1e-12)
        max_error = max(max_error, float(np.max(np.abs(forward-reference))))
        for name, scores in (("baseline", forward), ("bidirectional", harmonic)):
            order = order_scores(c, scores, q, config["evaluation"].get("tie_seed", 2026))[:k]
            ranked[name].append((int(q), c[order]))
        if number % 50 == 0:
            print(f"Valid scoring {number}/{len(queries)}", flush=True)
    results = {}
    for name in ranked:
        results[name] = evaluate_suite_rankings(config, labels, candidates, iter(ranked[name]),
            output / name, "valid", split_ids=meta["map_ids"],
            metadata={"representation": "h512", "aggregation": name, "checkpoint": str(checkpoint_path),
                      "epoch": checkpoint["epoch"], "candidates_scored": "full_pool",
                      "pool_sha256": hashes["valid_pool.json"], "bootstrap_samples": 1000})
    comparison = {key: {"baseline": value, "bidirectional": results["bidirectional"]["metrics"][key],
                       "delta": results["bidirectional"]["metrics"][key] - value}
                  for key, value in results["baseline"]["metrics"].items()}
    paired = {}
    baseline_rows = read_json(output / "baseline" / "per_query.json")
    changed_rows = read_json(output / "bidirectional" / "per_query.json")
    if [r["query_id"] for r in baseline_rows] != [r["query_id"] for r in changed_rows]:
        raise ValueError("Paired query identities differ")
    rng = np.random.default_rng(2026)
    for group in ("all", "single", "mixed"):
        selected = [i for i, row in enumerate(baseline_rows) if group == "all" or row["group"] == group]
        paired[group] = {}
        for key in comparison:
            field = key[1:] if key.startswith("mAP@") else key
            delta = np.array([changed_rows[i][field]-baseline_rows[i][field] for i in selected])
            means = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)])
            paired[group][key] = {"mean_delta": float(delta.mean()),
                                 "bootstrap_95ci": np.quantile(means, [.025, .975]).tolist(),
                                 "improved": int((delta > 1e-12).sum()),
                                 "worsened": int((delta < -1e-12).sum()), "queries": len(delta)}
    recorded_path = source / f"seed_{args.seed}" / "F" / "valid" / "best_nce" / "metrics.json"
    recorded_check = None
    if recorded_path.exists():
        recorded = read_json(recorded_path)
        recorded_check = {key: value-recorded["metrics"][key]
                          for key, value in results["baseline"]["metrics"].items()}
    write_json(output / "comparison.json", {"metrics": comparison, "forward_max_error": max_error,
                                            "paired": paired, "delta_from_saved_baseline": recorded_check,
                                            "test_accessed": False})
    print(comparison, flush=True)


if __name__ == "__main__":
    main()
