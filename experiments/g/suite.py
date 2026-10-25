"""G orchestration lives outside wbm_retrieval to preserve historical E/F hashes.

The cache/evaluation flow is adapted from the frozen A-D implementation.
Shared preprocessing, model and metric helpers are imported without mutation.
"""

from __future__ import annotations

import csv
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from wbm_retrieval.ablation import (RegionViews, build_model, embeddings, exclusive,
                                   merge, resize_then_pad, valid_pool)
from wbm_retrieval.common import ROOT, digest, file_hash, load_config, read_json, write_csv, write_json
from wbm_retrieval.data import Protocol, containment, source_fingerprint
from wbm_retrieval.diagnostics import summarize_groups
from wbm_retrieval.evaluation import bootstrap_interval, query_metrics
from wbm_retrieval.methods.handcrafted import handcrafted_descriptor
from wbm_retrieval.regions import extract_regions
from .matching import batched_map_scores, count_similarity, method_settings


def guard_output(root):
    root = Path(root).resolve()
    protected = [Path(load_config(ROOT / "configs/experiment_a.json")["output_dir"])]
    for name in ("ablation_abcd.json", "ablation_ef.json"):
        protected.append((ROOT / read_json(ROOT / "configs" / name)["output_root"]).resolve())
    if root == ROOT or root in ROOT.parents or any(
            root == p or root in p.parents or p in root.parents for p in protected):
        raise ValueError("G requires a separate output directory outside historical experiments")
    for parent in root.parents:
        if (parent / "suite_manifest.json").exists() or (parent / "resolved_config.json").exists():
            raise ValueError("G output must not be nested inside an existing experiment")


def code_hashes():
    paths = sorted((ROOT / "wbm_retrieval").rglob("*.py"))
    paths += sorted(Path(__file__).parent.glob("*.py")) + [ROOT / "scripts/run_ablation_g.py"]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def load_suite(path, *, output_root=None, seed=None, device=None, batch_size=None, workers=None):
    suite = read_json(path)
    config = merge(load_config(ROOT / suite["base_config"]), suite["overrides"])
    root = Path(output_root or suite["output_root"]).expanduser()
    root = (ROOT / root).resolve()
    guard_output(root)
    config["output_dir"] = str(root)
    for key, value in (("seed", seed), ("device", device), ("batch_size", batch_size), ("num_workers", workers)):
        if value is not None:
            config[key] = value
    if config["batch_size"] < 2 or config["num_workers"] < 0 or config["inference_batch_size"] < 1:
        raise ValueError("Invalid batch size / worker count")
    if config["local"].get("opening", {}).get("enabled") or config["local"]["early_stopping"]["enabled"]:
        raise ValueError("G disables opening and early stopping")
    if config["augmentation"] != dict(rotation=True, max_shift=0, dropout=0., noise=0.):
        raise ValueError("Base augmentation must be right-angle rotations only")
    if config["input_size"] != 96 or config["proposal"] != dict(min_area=5, top_k=4, padding=4):
        raise ValueError("This suite fixes input=96, min_area=5, top_k=4, old-grid padding=4")
    if suite["selection_metric"] != "ContainmentNDCG@10" or 10 not in config["evaluation"]["ks"]:
        raise ValueError("Checkpoint selection requires ContainmentNDCG@10")
    if suite["evaluate_every"] < 1 or config["local"]["epochs"] < 1:
        raise ValueError("Epoch count and evaluation interval must be positive")
    expected = dict(preprocess="resize_pad", content_size=88, stem_stride=1, scale_range=None, max_shift=0)
    if suite.get("suite_id") != "g_count_v1" or suite["variants"] != {"G": expected}:
        raise ValueError("Expected isolated G suite with D input, augmentation and full encoder")
    if config["local"]["num_negatives"] != 1000:
        raise ValueError("G fixes local.num_negatives=1000")
    for method in ("handcrafted", "proposed"):
        method_settings(config, method)
    return suite, config


def prepare_suite(suite, config):
    root = Path(config["output_dir"])
    guard_output(root)
    root.mkdir(parents=True, exist_ok=True)
    sources = source_fingerprint(config)
    # Device/workers/training seed may change; all scientific settings and batch size may not.
    settings = {k: v for k, v in config.items() if k not in (
        "device", "num_workers", "seed", "output_dir", "data_file", "split_manifest", "query_manifest", "candidate_manifest")}
    identity = {"sources": sources, "settings": settings,
                "suite": {k: suite[k] for k in ("suite_id", "variants", "valid_pool", "selection_metric", "evaluate_every")},
                "code": code_hashes()}
    signature = digest(identity)
    manifest = root / "suite_manifest.json"
    if manifest.exists():
        if read_json(manifest)["signature"] != signature:
            raise ValueError("Suite inputs/config/code changed; choose a new --output-root")
    else:
        if any(p.name != ".ablation.lock" for p in root.iterdir()):
            raise ValueError("Nonempty output has no ablation manifest; choose a new --output-root")
        write_json(manifest, {"signature": signature, **identity})
    cache = root / "cache"
    done = cache / "prepared.json"
    if done.exists():
        status = read_json(done)
        if status["signature"] != signature:
            raise ValueError("Cache signature mismatch")
        for name, sha in status["files"].items():
            if not (root / name).is_file() or file_hash(root / name) != sha:
                raise ValueError(f"Missing or modified ablation cache: {name}")
        print("Reusing verified ablation cache and fixed valid pool", flush=True)
        return signature
    cache.mkdir(exist_ok=True)
    with np.load(config["data_file"], allow_pickle=False) as archive:
        maps, labels = archive["arr_0"], archive["arr_1"]
    if maps.ndim != 3 or labels.ndim != 2 or len(maps) != len(labels) or not np.isin(maps, [0, 1, 2, 3]).all():
        raise ValueError("Expected Mixed-WM38 NPZ encoding")
    maps = maps.astype(np.uint8)
    protocol = Protocol(config, labels)
    np.save(cache / "labels.npy", labels)
    for split, ids in protocol.ids.items():
        regions, offsets = [], [0]
        for n, sample_id in enumerate(ids, 1):
            regions.extend(extract_regions(maps[sample_id], **config["proposal"]))
            offsets.append(len(regions))
            if n % 2000 == 0:
                print(f"CC {split}: {n}/{len(ids)}", flush=True)
        count = len(regions)
        if not count:
            raise ValueError(f"No retained regions: {split}")
        patches = np.lib.format.open_memmap(cache / f"{split}_B.npy", mode="w+", dtype=np.uint8,
                    shape=(count, 1, config["input_size"], config["input_size"]))
        descriptors = []
        for index, region in enumerate(regions):
            patches[index, 0] = resize_then_pad(region["mask"], config["input_size"], suite["variants"]["G"]["content_size"])
            descriptors.append(handcrafted_descriptor(region["mask"], config["handcrafted"]["max_order"]))
            if (index+1) % 2000 == 0:
                print(f"Patches/Zernike {split}: {index+1}/{count}", flush=True)
        patches.flush()
        del patches
        np.savez_compressed(cache / f"{split}_regions.npz", map_ids=ids, offsets=offsets,
                            area=np.array([r["area"] for r in regions]), descriptor=np.asarray(descriptors, dtype=np.float32))
        print(f"Prepared {split}: maps={len(ids)}, tokens={count}", flush=True)
    write_json(root / "valid_pool.json", valid_pool(protocol.ids["valid"], suite["valid_pool"]))
    files = list(cache.glob("*.npy")) + list(cache.glob("*.npz")) + [root / "valid_pool.json"]
    write_json(done, {"signature": signature, "files": {str(p.relative_to(root)): file_hash(p) for p in files},
                     "protocol": protocol.summary(), "test_note": "Preprocessed only; never used for training or model selection"})
    return signature


def evaluate(config, suite, split, output, *, model=None, spec=None, device=None, rankings=False, bootstrap=0):
    root, output = Path(config["output_dir"]), Path(output)
    cache = root / "cache"
    with np.load(cache / f"{split}_regions.npz") as archive:
        meta = dict(archive)
    labels = np.load(cache / "labels.npy", mmap_mode="r")
    if split == "valid":
        pool = read_json(root / "valid_pool.json")
        queries = pool["queries"]
        candidates = {int(q): c for q, c in pool["candidates"].items()}
    elif split == "test":
        protocol = Protocol(config)
        queries, candidates = protocol.queries, protocol.candidates
    else:
        raise ValueError("Evaluation supports only valid and explicitly requested test")
    if model is None:
        features = meta["descriptor"].astype(np.float64)
    else:
        family = "A" if spec["preprocess"] == "pad_resize" else "B"
        features = embeddings(model, cache / f"{split}_{family}.npy", config, device)
    indices = {int(i): np.arange(a, b) for i, a, b in zip(meta["map_ids"], meta["offsets"][:-1], meta["offsets"][1:])}
    width = int(np.diff(meta["offsets"]).max())
    padded = np.full((len(meta["map_ids"]), width), -1, dtype=np.int64)
    positions = {int(i): n for n, i in enumerate(meta["map_ids"])}
    for sample_id, tokens in indices.items():
        padded[positions[sample_id], :len(tokens)] = tokens
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    matching = method_settings(config, "handcrafted" if model is None else "proposed")
    match_audit = []
    with ((output / "rankings.csv").open("w", newline="") if rankings else nullcontext(None)) as stream:
        writer = csv.writer(stream) if stream else None
        if writer:
            writer.writerow(["query_id", "rank", "candidate_id", "similarity_score", "matched_pairs", "query_regions"])
        for number, q in enumerate(queries, 1):
            c = np.asarray(candidates[q], dtype=np.int64)
            tokens = indices[q]
            if model is None:
                distances = np.linalg.norm(features[tokens, None] - features[None], axis=-1)
                similarities = np.exp(-distances / config["handcrafted"]["distance_scale"])
            else:
                similarities = features[tokens] @ features.T
            scores, counts = batched_map_scores(similarities, padded[[positions[int(i)] for i in c]], len(tokens), matching, return_counts=True)
            if not np.isfinite(scores).all():
                raise FloatingPointError("Nonfinite retrieval score")
            order = np.lexsort((c, -scores))
            if writer:
                writer.writerows((q, rank, int(c[i]), float(scores[i]), int(counts[i]), len(tokens))
                                 for rank, i in enumerate(order, 1))
                for i in order[:config["evaluation"]["save_matches_top_k"]]:
                    candidate_tokens = indices[int(c[i])]
                    score, pairs = count_similarity(similarities[:, candidate_tokens], matching, details=True)
                    match_audit.append({"query_id": int(q), "candidate_id": int(c[i]),
                                        "similarity_score": score, "query_regions": len(tokens),
                                        "matched_pairs": len(pairs), "matches": pairs})
            relevance = containment(labels[q], labels[c])
            metrics = query_metrics(relevance[order], relevance, config["evaluation"]["ks"])
            rows.append({"query_id": q, "group": "single" if labels[q].sum() == 1 else "mixed", **metrics})
            if number % 100 == 0:
                print(f"Retrieval {split}: {number}/{len(queries)}", flush=True)
    groups = summarize_groups(rows, list(metrics), bootstrap, suite["valid_pool"]["seed"])
    result = {"split": split, "groups": groups, "metrics": groups[0]["metrics"], "num_queries": len(rows),
              "matching": matching, "aggregation": "query_region_count",
              "representation": "h512" if model is not None else "Zernike47",
              "pool_sha256": file_hash(root / "valid_pool.json") if split == "valid" else file_hash(config["candidate_manifest"])}
    if rankings:
        write_json(output / "matches_top_k.json", match_audit)
    write_json(output / "metrics.json", result)
    write_json(output / "per_query.json", rows)
    write_csv(output / "per_query_metrics.csv", rows, list(rows[0]))
    return result


def summarize(config, suite, split="valid"):
    root = Path(config["output_dir"]) / f"seed_{config['seed']}"
    rows, per_variant, paired = [], {}, []
    for name in ("G",):
        for kind in ("random", "best_nce", "best_retrieval"):
            folder = root / name / split / kind
            if not (folder / "metrics.json").exists():
                continue
            result = read_json(folder / "metrics.json")
            rows.extend({"variant": name, "checkpoint": kind, "group": g["group"],
                         "epoch": result.get("epoch", 0), "num_queries": g["num_queries"], **g["metrics"]} for g in result["groups"])
            if kind == "best_retrieval":
                per_variant[name] = {r["query_id"]: r for r in read_json(folder / "per_query.json")}
    hc = Path(config["output_dir"]) / "handcrafted" / split / "metrics.json"
    if hc.exists():
        result = read_json(hc)
        rows.extend({"variant": "handcrafted", "checkpoint": "none", "group": g["group"], "epoch": 0,
                     "num_queries": g["num_queries"], **g["metrics"]} for g in result["groups"])
        per_variant["handcrafted"] = {r["query_id"]: r for r in read_json(hc.with_name("per_query.json"))}
    for left, right in (("G", "handcrafted"),):
        if left not in per_variant or right not in per_variant:
            continue
        if set(per_variant[left]) != set(per_variant[right]):
            raise ValueError("Query sets differ")
        for metric in (suite["selection_metric"], "mAP_full", "FullCoveragePrecision@1"):
            delta = [row[metric] - per_variant[right][q][metric] for q, row in per_variant[left].items()
                     if row[metric] is not None and per_variant[right][q][metric] is not None]
            paired.append({"left": left, "right": right, "metric": metric, "num_queries": len(delta),
                           "difference": float(np.mean(delta)) if delta else None,
                           "query_bootstrap_95ci": bootstrap_interval(delta, config["evaluation"]["bootstrap_samples"],
                                                                      np.random.default_rng(suite["valid_pool"]["seed"]))})
    root.mkdir(parents=True, exist_ok=True)
    if rows:
        write_csv(root / f"comparison_{split}.csv", rows, list(rows[0]))
        write_json(root / f"comparison_{split}.json", rows)
    write_json(root / f"paired_{split}.json", {"comparisons": paired,
        "caution": "valid is used for selection; exploratory query intervals are not final test evidence and exclude training-seed variance"})
    print(f"Summary: {root / f'comparison_{split}.csv'}", flush=True)
