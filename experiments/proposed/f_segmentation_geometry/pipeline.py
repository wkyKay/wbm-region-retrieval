"""Versioned evaluation without modifying training runs or their signatures."""
from contextlib import contextmanager
import csv
import fcntl
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from experiments.proposed._shared import ef_suite, ef_training
from experiments.proposed.f_segmentation import suite as original
from wbm_retrieval.common import ROOT, digest, file_hash, get_device, read_json, write_csv, write_json
from wbm_retrieval.data import Protocol, validation_pool
from wbm_retrieval.diagnostics import batched_map_scores, normalized
from wbm_retrieval.exact_metrics import evaluate_rankings, order_scores
from wbm_retrieval.training import load_checkpoint
from .coverage_metrics import evaluate_rankings_file
from .scoring import SETTINGS, geometry, pair_scores


@contextmanager
def source_lock(source):
    """Use the original runner's lock in read mode; never touch its mtime."""
    with (source / ".ablation.lock").open("r") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Source training is running; finish it before geometry evaluation") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def load_source(source, method, data_file=None, device="cuda", checkpoint_kind=None):
    folder = source / method
    manifest_path = folder / "suite_manifest.json"
    manifest = read_json(manifest_path)
    signature = manifest["signature"]
    if digest({k: v for k, v in manifest.items() if k != "signature"}) != signature:
        raise ValueError("Source suite manifest signature is invalid")
    config = dict(manifest["settings"])
    kind = checkpoint_kind or config["final_checkpoint"]
    if kind != config["final_checkpoint"]:
        raise ValueError("Use the checkpoint rule frozen before the source run")
    run = folder / f"seed_{config['seed']}" / "F"
    summary = read_json(run / "training_summary.json")
    if not summary["completed"] or summary["epochs_completed"] != config["local"]["epochs"]:
        raise ValueError(f"{method}: training incomplete; resume the original training command first")
    recorded = read_json(run / "run_manifest.json")
    runtime = recorded["config"]
    for key in ("data_file", "split_manifest", "deduplication_metadata", "retrieval_protocol_dir"):
        path = Path(runtime[key])
        if key == "data_file" and data_file is not None:
            path = Path(data_file)
        elif not path.exists() and key != "data_file":
            path = ROOT / {"split_manifest": "protocol/deduplicated/split.csv",
                           "deduplication_metadata": "protocol/deduplicated/metadata.json",
                           "retrieval_protocol_dir": "protocol/retrieval_v1"}[key]
        config[key] = str(path.resolve())
    config.update(output_dir=str(folder), device=device, num_workers=0)
    suite = manifest["suite"]
    if config["segmentation"]["method"] != method or digest(original.identity(suite, config)) != signature:
        raise ValueError("Source method/config/protocol/code mismatch; do not bypass training provenance")
    common_path = source / "segmentation_comparison.json"
    common = read_json(common_path)
    common_config = {k: v for k, v in config.items() if k != "segmentation"}
    if (digest({k: v for k, v in common.items() if k != "signature"}) != common["signature"]
            or digest(original.identity(suite, common_config)) != common["signature"]):
        raise ValueError("Method does not match the source comparison's common protocol/settings")
    run_signature = ef_training.run_signature(signature, config["seed"], "F")
    if recorded["signature"] != run_signature:
        raise ValueError("Training manifest signature mismatch")
    checkpoint_path = run / f"{kind}.pt"
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint["signature"] != run_signature or checkpoint["input_spec"] != suite["variants"]["F"]:
        raise ValueError("Checkpoint signature/input specification mismatch")
    if checkpoint["epoch"] != summary[f"{kind}_epoch"]:
        raise ValueError("Checkpoint epoch does not match the frozen training selection")
    files = [manifest_path, run / "run_manifest.json", run / "training_summary.json", checkpoint_path]
    inputs = {str(p.relative_to(folder)): file_hash(p) for p in files}
    inputs["../segmentation_comparison.json"] = file_hash(common_path)
    return config, suite, checkpoint, kind, inputs


def new_identity(config, inputs):
    return {"version": "f_segmentation_radial_area_v1", "scoring": SETTINGS,
            "source_files": inputs, "method": config["segmentation"]["method"],
            "code": {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}


def commit_marker(folder, signature):
    write_json(folder / "completed.json", {"signature": signature, "files": {
        str(p.relative_to(folder)): file_hash(p) for p in sorted(folder.rglob("*"))
        if p.is_file() and p.name != "completed.json"}})


def completed(folder, signature):
    if not folder.exists():
        return False
    marker = read_json(folder / "completed.json")
    if marker["signature"] != signature:
        raise ValueError(f"Result signature changed; choose a new --output-root: {folder}")
    original.verify_files(folder, marker["files"])
    return True


def build_features(folder, config, suite, checkpoint, protocol, maps, split, signature, workers):
    target = folder / "features" / split
    if completed(target, signature):
        original.verify_files(Path(config["output_dir"]), read_json(target / "provenance.json")["source_cache_files"])
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=f".{split}_attempt_", dir=target.parent))
    source = Path(config["output_dir"])
    ids = np.asarray(protocol.ids[split], dtype=np.int64)
    cache = source / "cache"
    source_inputs = {}
    # Reuse only a fully signed cache. Partial/missing source caches are ignored;
    # reconstruction below writes exclusively to the new output.
    done = cache / f"{split}_prepared.json"
    labels_map = source_meta = None
    if done.exists():
        marker = read_json(done)
        if marker["signature"] != read_json(source / "suite_manifest.json")["signature"]:
            raise ValueError("Source cache signature mismatch")
        original.verify_files(source, marker["files"])
        source_inputs = {**marker["files"], str(done.relative_to(source)): file_hash(done)}
        labels_map = np.load(cache / f"{split}_region_labels.npy", mmap_mode="r")
        with np.load(cache / f"{split}_regions.npz") as z:
            source_meta = dict(z)
        if not np.array_equal(source_meta["map_ids"], ids) or labels_map.shape != (len(ids), *maps.shape[1:]):
            raise ValueError("Source region IDs/shape differ from frozen split")
        iterator = ((row, {}) for row in labels_map)
    else:
        iterator = original.segmented_maps(maps, ids, config, workers)
    device = get_device(config["device"])
    model = ef_suite.build_model(config, suite["variants"]["F"]).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    output_labels = np.lib.format.open_memmap(attempt / "region_labels.npy", mode="w+", dtype=np.uint8,
                                             shape=(len(ids), *maps.shape[1:]))
    patches, features, histograms, areas, relative, offsets = [], [], [], [], [], [0]

    def flush():
        if patches:
            with torch.inference_mode():
                batch = torch.from_numpy(np.asarray(patches, dtype=np.float32)).to(device)
                features.append(model["encoder"](batch).cpu().numpy())
            patches.clear()

    for index, (region_ids, _) in enumerate(iterator):
        raw = maps[ids[index]]
        if np.any((region_ids > 0) & (raw != 2)):
            raise ValueError("Regions contain non-original defective pixels")
        output_labels[index] = region_ids
        count = int(region_ids.max())
        if count > config["proposal"]["top_k"]:
            raise ValueError("Region budget exceeded")
        for token in range(1, count+1):
            mask = region_ids == token
            if mask.sum() < config["proposal"]["min_area"]:
                raise ValueError("Invalid empty/small region in cache")
            histogram, fraction = geometry(mask, (raw == 1) | (raw == 2))
            histograms.append(histogram)
            areas.append(int(mask.sum()))
            relative.append(fraction)
            patch, _ = ef_suite.fixed_window_patch(raw, mask, config["input_size"], suite["variants"]["F"]["window_size"])
            patches.append(patch)
            if len(patches) == config["inference_batch_size"]:
                flush()
        offsets.append(len(areas))
        if (index+1) % 500 == 0:
            print(f"Geometry/features {config['segmentation']['method']}/{split}: {index+1}/{len(ids)}", flush=True)
    flush()
    output_labels.flush()
    del output_labels
    if source_meta is not None:
        if not np.array_equal(source_meta["offsets"], offsets) or not np.array_equal(source_meta["area"], areas):
            raise ValueError("Cached token order/area does not match original masks")
    values = normalized(np.concatenate(features)) if features else np.empty((0, config["embedding_dim"]), dtype=np.float32)
    np.save(attempt / "h512.npy", values)
    np.savez_compressed(attempt / "regions.npz", map_ids=ids, offsets=offsets, area=np.asarray(areas),
                        relative_area=np.asarray(relative), histograms=np.asarray(histograms).reshape(-1, SETTINGS["radial_bins"]))
    original.verify_files(source, source_inputs)
    write_json(attempt / "provenance.json", {"source_cache_files": source_inputs,
               "region_source": "verified_source_cache" if source_meta is not None else "recomputed_same_segmentation",
               "empty_maps": int(np.sum(np.diff(offsets) == 0)), "num_tokens": len(areas),
               "environment": {"numpy": np.__version__, "torch": torch.__version__, "device": str(device)}})
    commit_marker(attempt, signature)
    attempt.rename(target)
    return target


def evaluate_split(folder, config, protocol, labels, feature_dir, split, kind, epoch, signature):
    target = folder / split / kind
    if completed(target, signature):
        print(f"Verified completed {config['segmentation']['method']}/{split}: exact + coverage", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=f".{kind}_attempt_", dir=target.parent))
    with np.load(feature_dir / "regions.npz") as z:
        meta = dict(z)
    features = np.load(feature_dir / "h512.npy", mmap_mode="r")
    pool = getattr(protocol.retrieval, split)
    ids, offsets, area = meta["map_ids"], meta["offsets"], meta["area"]
    positions = {int(sid): i for i, sid in enumerate(ids)}
    width = int(np.diff(offsets).max())
    padded = np.full((len(ids), width), -1, dtype=np.int64)
    for i, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
        padded[i, :b-a] = np.arange(a, b)
    _, classes = np.unique(labels, axis=0, return_inverse=True)
    bundle = SimpleNamespace(labels=labels, class_ids=classes, valid_ids=protocol.ids["valid"], reserved_ids=protocol.ids["test"])
    ks = config["evaluation"]["ks"]
    metadata = {"scoring": SETTINGS, "checkpoint_kind": kind, "epoch": epoch,
                "signature": signature, "method": config["segmentation"]["method"],
                "ranking_scope": "top_max_k", "candidates_scored": "full_pool", "training_required": False}
    with (attempt / "rankings.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "rank", "candidate_id", "similarity_score"])

        def rankings():
            for number, q in enumerate(pool, 1):
                index = positions[int(q)]
                tokens = np.arange(offsets[index], offsets[index+1])
                candidates = pool[q]
                similarity = pair_scores(features[tokens], features, meta["histograms"][tokens],
                                         meta["histograms"], meta["relative_area"][tokens], meta["relative_area"])
                scores = batched_map_scores(similarity, padded[[positions[int(c)] for c in candidates]], area[tokens], config["matching"])
                if not np.isfinite(scores).all():
                    raise FloatingPointError("Nonfinite geometric score")
                order = order_scores(candidates, scores, int(q), config["evaluation"].get("tie_seed", 2026))[:max(ks)]
                writer.writerows((q, rank, int(candidates[i]), float(scores[i])) for rank, i in enumerate(order, 1))
                yield int(q), candidates[order]
                if number % 100 == 0:
                    print(f"Retrieval {config['segmentation']['method']}/{split}: {number}/{len(pool)}", flush=True)

        evaluate_rankings(bundle, pool, rankings(), ks, attempt, split_name=split, metadata=metadata)
    evaluate_rankings_file(bundle, pool, attempt / "rankings.csv", ks, attempt, split_name=split, metadata=metadata)
    commit_marker(attempt, signature)
    attempt.rename(target)
    print(f"Completed {config['segmentation']['method']}/{split}: exact + coverage", flush=True)


def evaluate_method(source, output, method, data_file, device, checkpoint_kind, splits, workers):
    config, suite, checkpoint, kind, inputs = load_source(source, method, data_file, device, checkpoint_kind)
    spec = new_identity(config, inputs)
    signature = digest(spec)
    folder = output / method
    manifest = folder / "evaluation_manifest.json"
    if manifest.exists():
        if read_json(manifest)["signature"] != signature:
            raise ValueError("Geometry evaluation inputs changed; choose a new output root")
    else:
        if folder.exists() and any(folder.iterdir()):
            raise ValueError("Nonempty unsigned evaluation directory")
        write_json(manifest, {"signature": signature, **spec, "source_path": str(source / method),
                              "seed": config["seed"], "checkpoint_kind": kind})
    protocol = Protocol(config)
    if max(config["evaluation"]["ks"]) > min(getattr(protocol.retrieval, s).candidates_per_query for s in splits):
        raise ValueError("K exceeds frozen candidate pool")
    with np.load(config["data_file"], allow_pickle=False) as z:
        maps, labels = z["arr_0"], z["arr_1"]
    # Check the source pool when available, in addition to Protocol's frozen files.
    pool_path = source / method / "valid_pool.json"
    if pool_path.exists() and read_json(pool_path) != validation_pool(protocol, suite["valid_pool"]):
        raise ValueError("Source valid pool differs from frozen protocol")
    for split in splits:
        feature_dir = build_features(folder, config, suite, checkpoint, protocol, maps, split, signature, workers)
        evaluate_split(folder, config, protocol, labels, feature_dir, split, kind, checkpoint["epoch"], signature)
    original.verify_files(source / method, inputs)
    if digest(original.identity(suite, config)) != read_json(source / method / "suite_manifest.json")["signature"]:
        raise RuntimeError("Source inputs changed during evaluation")


def summarize(output, methods):
    rows = []
    for method in methods:
        folder = output / method
        if not (folder / "evaluation_manifest.json").exists():
            continue
        manifest = read_json(folder / "evaluation_manifest.json")
        kind = manifest["checkpoint_kind"]
        for split in ("valid", "test"):
            target = folder / split / kind
            if not target.exists() or not completed(target, manifest["signature"]):
                continue
            for name in ("metrics.json", "coverage_metrics.json"):
                report = read_json(target / name)
                groups = {"all": report["metrics"], "class_macro": report["class_macro_metrics"],
                          "single_label": report["groups"]["single_label"]["metrics"],
                          "mixed_label": report["groups"]["mixed_label"]["metrics"]}
                groups.update({f"label_count_{k}": v["metrics"] for k, v in report["groups"]["by_label_count"].items()})
                for group, values in groups.items():
                    rows.extend({"method": method, "split": split, "seed": manifest["seed"], "checkpoint": kind,
                                 "metric_version": report["metric_version"], "group": group,
                                 "metric": key, "value": value} for key, value in values.items())
    if rows:
        write_csv(output / "comparison.csv", rows, list(rows[0]))
        write_json(output / "comparison.json", rows)
