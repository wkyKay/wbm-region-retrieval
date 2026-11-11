"""Bounded-memory F cache construction and resumable, immutable evaluation."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
import multiprocessing
from pathlib import Path
import tempfile

import numpy as np

from experiments.proposed._shared import ef_suite, ef_training
from wbm_retrieval.common import ROOT, digest, file_hash, get_device, read_json, write_csv, write_json
from wbm_retrieval.data import Protocol, source_fingerprint, validation_pool
from wbm_retrieval.exact_evaluation import is_exact
from wbm_retrieval.training import load_checkpoint
from .proposals import SETTINGS, segment_job


def load_suite(path, **kwargs):
    suite, config = ef_suite.load_suite(path, **kwargs)
    if config.get("data_protocol") != "deduplicated_v1" or not is_exact(config):
        raise ValueError("F segmentation comparisons require deduplicated_v1 and exact_v1")
    return suite, config


def method_config(config, method):
    result = copy.deepcopy(config)
    result["output_dir"] = str(Path(config["output_dir"]) / method)
    result["segmentation"] = {"method": method, "parameters": SETTINGS[method],
                              "version": "f_segmentation_v1"}
    return result


def identity(suite, config):
    excluded = {"device", "num_workers", "output_dir", "data_file", "split_manifest",
                "deduplication_metadata", "retrieval_protocol_dir"}
    code = ef_suite.code_hashes()
    code.update({str(p.relative_to(ROOT)): file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
    return {"format": "f_segmentation_v1", "sources": source_fingerprint(config),
            "settings": {k: v for k, v in config.items() if k not in excluded},
            "suite": {k: suite[k] for k in ("variants", "valid_pool", "selection_metric", "evaluate_every")},
            "code": code}


def verify_files(root, files):
    for name, expected in files.items():
        path = Path(root) / name
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Missing or modified signed artifact: {path}")


def initialize(suite, config):
    root = Path(config["output_dir"])
    # Only the new comparison parent is allowed; no nesting inside old suites.
    for parent in root.parents:
        if (parent / "suite_manifest.json").exists() or (parent / "resolved_config.json").exists():
            raise ValueError("Do not nest segmentation runs inside an existing experiment")
    payload = identity(suite, config)
    signature = digest(payload)
    manifest = root / "suite_manifest.json"
    root.mkdir(parents=True, exist_ok=True)
    if manifest.exists():
        if read_json(manifest)["signature"] != signature:
            raise ValueError("Inputs/config/code changed; choose a new --output-root")
    else:
        if any(p.name != ".ablation.lock" for p in root.iterdir()):
            raise ValueError("Nonempty method directory has no signed manifest")
        write_json(manifest, {"signature": signature, **payload})
    return signature


def segmented_maps(maps, ids, config, workers):
    """At most 2*workers wafer jobs in flight; deterministic input-order output."""
    jobs = ((maps[sid], config["segmentation"]["method"], config["proposal"]) for sid in ids)
    if not workers:
        yield from map(segment_job, jobs)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        pending = deque()
        for _ in range(workers * 2):
            job = next(jobs, None)
            if job is not None:
                pending.append(executor.submit(segment_job, job))
        while pending:
            yield pending.popleft().result()
            job = next(jobs, None)
            if job is not None:
                pending.append(executor.submit(segment_job, job))


def prepare(suite, config, signature, splits=("train", "valid"), workers=0):
    root = Path(config["output_dir"])
    cache = root / "cache"
    cache.mkdir(exist_ok=True)
    pending = []
    for split in splits:
        done = cache / f"{split}_prepared.json"
        if done.exists():
            marker = read_json(done)
            if marker["signature"] != signature:
                raise ValueError("Cache signature mismatch")
            verify_files(root, marker["files"])
            print(f"Verified {config['segmentation']['method']}/{split} cache", flush=True)
        else:
            pending.append(split)
    if not pending:
        return
    with np.load(config["data_file"], allow_pickle=False) as archive:
        maps, labels = archive["arr_0"], archive["arr_1"]
    protocol = Protocol(config, labels, maps=maps)
    pool = validation_pool(protocol, suite["valid_pool"])
    if (cache / "labels.npy").exists():
        if not np.array_equal(np.load(cache / "labels.npy"), labels):
            raise ValueError("Cached labels differ from source")
    else:
        np.save(cache / "labels.npy", labels)
    if (root / "valid_pool.json").exists():
        if read_json(root / "valid_pool.json") != pool:
            raise ValueError("Cached valid pool differs from frozen protocol")
    else:
        write_json(root / "valid_pool.json", pool)
    for split in pending:
        ids = protocol.ids[split]
        # One byte per original pixel; no list of all full-size region masks.
        regions = np.lib.format.open_memmap(cache / f"{split}_region_labels.npy", mode="w+",
                                           dtype=np.uint8, shape=(len(ids), *maps.shape[1:]))
        offsets, areas, rows = [0], [], []
        for index, (labels_map, audit) in enumerate(segmented_maps(maps, ids, config, workers)):
            regions[index] = labels_map
            counts = np.bincount(labels_map.ravel())[1:]
            areas.extend(counts.tolist())
            offsets.append(len(areas))
            rows.append({"sample_id": int(ids[index]), **audit})
            if (index + 1) % 500 == 0:
                print(f"{config['segmentation']['method']}/{split}: segmented {index+1}/{len(ids)}", flush=True)
        regions.flush()
        if not areas:
            raise ValueError(f"No regions remain in the entire {split} split; no silent CC8 fallback")
        patches = np.lib.format.open_memmap(cache / f"{split}_F.npy", mode="w+", dtype=np.uint8,
                                            shape=(len(areas), 3, config["input_size"], config["input_size"]))
        clipped, retained_window, min_fraction, sum_fraction = 0, 0, 1., 0.
        audit_path = cache / f"{split}_F_window_audit.csv"
        with audit_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["map_id", "token_id", "area", "retained_pixels",
                                    "retained_fraction", "clipped", "window_row", "window_col"])
            writer.writeheader()
            for index, sid in enumerate(ids):
                for token in range(offsets[index+1] - offsets[index]):
                    patch, audit = ef_suite.fixed_window_patch(maps[sid], regions[index] == token+1,
                                        config["input_size"], suite["variants"]["F"]["window_size"])
                    patches[offsets[index]+token] = patch
                    writer.writerow({"map_id": sid, "token_id": token, **audit})
                    clipped += int(audit["clipped"])
                    retained_window += audit["retained_pixels"]
                    min_fraction = min(min_fraction, audit["retained_fraction"])
                    sum_fraction += audit["retained_fraction"]
        patches.flush()
        del patches, regions
        np.savez_compressed(cache / f"{split}_regions.npz", map_ids=ids, offsets=offsets,
                            area=np.asarray(areas, dtype=np.int64))
        write_csv(cache / f"{split}_segmentation_audit.csv", rows, list(rows[0]))
        summary = {"maps": len(ids), "tokens": len(areas), "empty_maps": sum(r["empty"] for r in rows),
                   "empty_fraction": sum(r["empty"] for r in rows) / len(ids),
                   "original_pixels": sum(r["original_pixels"] for r in rows),
                   "retained_pixels": sum(areas), "clipped_tokens": clipped,
                   "window_retained_pixels": retained_window,
                   "mean_window_retained_fraction": sum_fraction / len(areas) if areas else None,
                   "min_window_retained_fraction": min_fraction if areas else None}
        write_json(cache / f"{split}_audit.json", summary)
        files = [cache / f"{split}_{name}" for name in ("region_labels.npy", "F.npy", "regions.npz",
                 "F_window_audit.csv", "segmentation_audit.csv", "audit.json")]
        files += [cache / "labels.npy", root / "valid_pool.json"]
        write_json(cache / f"{split}_prepared.json", {"signature": signature, "protocol": protocol.summary(),
                   "files": {str(p.relative_to(root)): file_hash(p) for p in files}})
        print(f"Prepared {config['segmentation']['method']}/{split}: {summary}", flush=True)


def evaluate_final(suite, config, signature, split, kind):
    """Commit reports atomically; retries verify completed output, preserve attempts."""
    run = Path(config["output_dir"]) / f"seed_{config['seed']}" / "F"
    if not read_json(run / "training_summary.json")["completed"]:
        raise ValueError("Complete training before final evaluation")
    checkpoint_path = run / f"{kind}.pt"
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint["signature"] != ef_training.run_signature(signature, config["seed"], "F"):
        raise ValueError("Checkpoint does not belong to this segmentation run")
    expected = {"signature": signature, "checkpoint_sha256": file_hash(checkpoint_path),
                "split": split, "checkpoint_kind": kind}
    target = run / split / kind
    if target.exists():
        marker = read_json(target / "completed.json")
        if any(marker.get(k) != v for k, v in expected.items()):
            raise ValueError("Completed evaluation signature mismatch")
        verify_files(target, marker["files"])
        print(f"Verified completed {config['segmentation']['method']}/{split}/{kind}", flush=True)
        return read_json(target / "metrics.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=f".{kind}_attempt_", dir=target.parent))
    device = get_device(config["device"])
    model = ef_suite.build_model(config, suite["variants"]["F"]).to(device)
    model.load_state_dict(checkpoint["model"])
    result = ef_suite.evaluate(config, suite, split, attempt, model=model,
                              spec=suite["variants"]["F"], device=device, rankings=True,
                              bootstrap=config["evaluation"]["bootstrap_samples"])
    result.update(epoch=checkpoint["epoch"], checkpoint=str(checkpoint_path),
                  segmentation=config["segmentation"], suite_signature=signature)
    write_json(attempt / "metrics.json", result)
    write_json(attempt / "completed.json", {**expected,
        "files": {p.name: file_hash(p) for p in sorted(attempt.iterdir()) if p.is_file()}})
    attempt.rename(target)
    print(f"{config['segmentation']['method']}/{split}/{kind}: {result['metrics']}", flush=True)
    return result
