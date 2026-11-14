"""Score-level fusion. Never train, truncate candidates, or select on test."""
from contextlib import ExitStack
import csv
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

from experiments.proposed.f_segmentation_geometry import pipeline as geo
from experiments.proposed.f_segmentation_geometry.scoring import pair_scores
from wbm_retrieval.common import digest, file_hash, read_json, write_csv, write_json
from wbm_retrieval.diagnostics import batched_map_scores
from wbm_retrieval.exact_metrics import evaluate_rankings, order_scores


DEFAULT_ALPHAS = (0., .1, .2, .3, .5, .75, 1.)


def validate_alphas(values):
    values = [float(v) for v in values]
    if (not values or not np.isfinite(values).all() or
            any(v < 0 or v > 1 for v in values) or len(set(values)) != len(values)):
        raise ValueError("Alphas must be distinct finite numbers in [0,1]")
    if 0. not in values or 1. not in values:
        raise ValueError("Include alpha=0 and alpha=1 as the two controls")
    return sorted(values)


def alpha_name(alpha):
    return "alpha_" + str(float(alpha))


def signed_manifest(path, *, compared=False):
    value = read_json(path)
    payload = {k: v for k, v in value.items() if k != "signature"}
    signature = (hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                 allow_nan=False, separators=(",", ":")).encode()).hexdigest() if compared else digest(payload))
    if value.get("signature") != signature:
        raise ValueError(f"Invalid manifest signature: {path}")
    return value


def checked_files(root, files):
    """Verify only contained artifact paths, including untrusted manifest paths."""
    root = Path(root).resolve()
    for name, expected in files.items():
        path = (root / name).resolve()
        if root not in path.parents or not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Missing or modified artifact: {path}")


def load_global(root, split, pool, sources, expected_checkpoint=None, expected_recipe=None):
    """Use completed, signed WaPIRL evaluations, independent of its source checkout."""
    root = Path(root)
    manifest = signed_manifest(root / "evaluation_manifest.json", compared=True)
    done = read_json(root / "completed.json")
    if (manifest.get("method") != "wapirl" or manifest.get("variant") != "global" or
            manifest.get("split") != split or manifest.get("smoke") is not False or
            manifest.get("candidate_rule") != "all_gallery_except_query" or
            manifest.get("derived_protocol") is not None or
            done.get("evaluation_signature") != manifest["signature"] or done.get("split") != split or
            done.get("smoke") is not False):
        raise ValueError("Expected a completed formal, full-gallery WaPIRL evaluation")
    expected_sources = {"data_file": sources["data_file"], "split_file": sources["split_manifest"],
                        "retrieval_protocol": {k.split("/", 1)[1]: v for k, v in sources.items()
                                               if k.startswith("retrieval_protocol/")}}
    if manifest["sources"] != expected_sources:
        raise ValueError("WaPIRL data/split/retrieval hashes differ from F")
    if (manifest["query_count"] != len(pool) or manifest["gallery_count"] != len(pool.gallery) or
            manifest["candidates_per_query"] != pool.candidates_per_query):
        raise ValueError("WaPIRL pool dimensions differ from frozen F pool")
    training = manifest["training_source"]
    checkpoint = training["files"]["best_nce.pt"]
    recipe = digest({"settings": manifest["settings"], "code": manifest["code"]})
    if expected_recipe is not None and recipe != expected_recipe:
        raise ValueError("WaPIRL valid/test inference settings or code differ")
    if (training.get("training_complete") is not True or training.get("smoke_training") is not False or
            training.get("selection") != "minimum_valid_nce" or training["sources"] != expected_sources or
            (expected_checkpoint is not None and checkpoint != expected_checkpoint)):
        raise ValueError("WaPIRL must use the same completed best_nce model on valid and test")
    relative = f"{split}/prepared/features.npz"
    if relative not in done["files"] or f"{split}/prepared/prepared_metadata.json" not in done["files"]:
        raise ValueError("WaPIRL feature cache is not signed")
    checked_files(root, done["files"])
    metadata = read_json(root / split / "prepared/prepared_metadata.json")
    if metadata.get("checkpoint_sha256") != checkpoint or metadata.get("similarity") != "clip(h512 cosine,0,1)":
        raise ValueError("WaPIRL feature checkpoint/scoring mismatch")
    with np.load(root / relative, allow_pickle=False) as data:
        ids, features = data["sample_ids"], data["features"]
    validate_features(ids, features, pool.gallery)
    if features.shape[1] != 512:
        raise ValueError("Expected WaPIRL h512 features")
    return ids, features, {"evaluation_signature": manifest["signature"], "checkpoint": checkpoint,
                           "inference_recipe": recipe,
                           "features_sha256": done["files"][relative],
                           "manifest_sha256": file_hash(root / "evaluation_manifest.json"),
                           "completed_sha256": file_hash(root / "completed.json")}


def validate_features(ids, features, gallery):
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or len(set(ids.tolist())) != len(ids) or
            set(ids.tolist()) != set(map(int, gallery)) or features.ndim != 2 or
            len(features) != len(ids) or not np.isfinite(features).all() or
            not np.allclose(np.linalg.norm(features, axis=1), 1., atol=2e-5)):
        raise ValueError("Invalid feature IDs, dimensions or L2 normalization")


def iter_rankings(path):
    with Path(path).open(newline="") as handle:
        for query, rows in itertools.groupby(csv.DictReader(handle), key=lambda r: int(r["query_id"])):
            yield query, np.asarray([int(r["candidate_id"]) for r in rows], dtype=np.int64)


class BranchScores:
    def __init__(self, feature_dir, global_ids, global_features, gallery, matching):
        with np.load(Path(feature_dir) / "regions.npz", allow_pickle=False) as data:
            self.meta = dict(data)
        self.features = np.load(Path(feature_dir) / "h512.npy", mmap_mode="r")
        ids, offsets = self.meta["map_ids"], self.meta["offsets"]
        if (ids.dtype.kind not in "iu" or len(set(ids.tolist())) != len(ids) or
                set(ids.tolist()) != set(map(int, gallery)) or len(offsets) != len(ids)+1 or
                offsets.dtype.kind not in "iu" or offsets[0] != 0 or
                np.any(np.diff(offsets) < 0) or offsets[-1] != len(self.features)):
            raise ValueError("F region IDs/offsets do not match the frozen gallery")
        n = len(self.features)
        if (self.features.ndim != 2 or not np.isfinite(self.features).all() or
                not np.allclose(np.linalg.norm(self.features, axis=1), 1., atol=2e-5) or
                self.meta["histograms"].shape != (n, 16) or
                any(self.meta[key].shape != (n,) for key in ("area", "relative_area")) or
                np.any(self.meta["area"] <= 0) or not np.isfinite(self.meta["area"]).all() or
                np.any(self.meta["relative_area"] <= 0) or np.any(self.meta["relative_area"] > 1) or
                not np.isfinite(self.meta["relative_area"]).all() or
                np.any(self.meta["histograms"] < 0) or
                not np.allclose(self.meta["histograms"].sum(axis=1), 1)):
            raise ValueError("Invalid F features or original-mask geometry")
        validate_features(global_ids, global_features, gallery)
        self.rows = {int(s): i for i, s in enumerate(ids)}
        self.global_rows = {int(s): i for i, s in enumerate(global_ids)}
        self.global_features, self.matching = global_features, matching
        self.padded = np.full((len(ids), int(np.diff(offsets).max(initial=0))), -1, dtype=np.int64)
        for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            self.padded[i, :end-start] = np.arange(start, end)

    def score(self, query, candidates):
        index = self.rows[int(query)]
        offsets, meta = self.meta["offsets"], self.meta
        tokens = np.arange(offsets[index], offsets[index+1])
        similarities = pair_scores(self.features[tokens], self.features, meta["histograms"][tokens],
                                   meta["histograms"], meta["relative_area"][tokens], meta["relative_area"])
        local = batched_map_scores(similarities, self.padded[[self.rows[int(c)] for c in candidates]],
                                   meta["area"][tokens], self.matching)
        rows = [self.global_rows[int(c)] for c in candidates]
        whole = np.clip(self.global_features[rows] @ self.global_features[self.global_rows[int(query)]], 0, 1)
        if not np.isfinite(local).all() or not np.isfinite(whole).all():
            raise ValueError("Nonfinite branch scores")
        return local, whole


def evaluate(output, split, alphas, signature, scorer, pool, bundle, ks, tie_seed):
    """Compute both branches once/query; keep only one query's full scores in RAM."""
    parent = Path(output) / split
    parent.mkdir(parents=True, exist_ok=True)
    pending = []
    for alpha in alphas:
        target = parent / alpha_name(alpha)
        if not geo.completed(target, signature):
            attempt = Path(tempfile.mkdtemp(prefix=f".{alpha_name(alpha)}_attempt_", dir=parent))
            pending.append((alpha, target, attempt))
    if not pending:
        print(f"Verified completed fusion {split}", flush=True)
        return
    with ExitStack() as stack:
        writers = []
        for alpha, _, attempt in pending:
            writer = csv.writer(stack.enter_context((attempt / "rankings.csv").open("w", newline="")))
            writer.writerow(["query_id", "rank", "candidate_id", "similarity_score", "f_score", "global_score"])
            writers.append((alpha, writer))
        for number, query in enumerate(pool, 1):
            candidates = pool[query]
            local, whole = scorer.score(query, candidates)
            for alpha, writer in writers:
                scores = (1-alpha)*local + alpha*whole
                order = order_scores(candidates, scores, int(query), tie_seed)[:max(ks)]
                writer.writerows((int(query), rank, int(candidates[i]), float(scores[i]),
                                   float(local[i]), float(whole[i])) for rank, i in enumerate(order, 1))
            if number % 100 == 0 or number == len(pool):
                print(f"Fusion {split}: {number}/{len(pool)} queries, {len(pending)} weights", flush=True)
    for alpha, target, attempt in pending:
        metadata = {"signature": signature, "alpha_global": alpha, "method": "cc8_wapirl_fusion",
                    "formula": "(1-alpha)*F_radial_area+alpha*WaPIRL_clipped_cosine",
                    "candidates_scored": "full_pool", "ranking_scope": "top_max_k", "tie_seed": tie_seed}
        evaluate_rankings(bundle, pool, iter_rankings(attempt / "rankings.csv"), ks, attempt,
                          split_name=split, metadata=metadata)
        geo.evaluate_rankings_file(bundle, pool, attempt / "rankings.csv", ks, attempt,
                                   split_name=split, metadata=metadata)
        geo.commit_marker(attempt, signature)
        attempt.rename(target)


def select(output, alphas, signature, metric="Precision@10", family="exact", group="all"):
    """Selection is exclusively a deterministic function of completed valid reports."""
    filename = "metrics.json" if family == "exact" else "coverage_metrics.json"
    candidates, hashes = [], {}
    for alpha in alphas:
        folder = Path(output) / "valid" / alpha_name(alpha)
        if not geo.completed(folder, signature):
            raise ValueError("Complete valid sweep before test")
        report = read_json(folder / filename)
        values = report["metrics"] if group == "all" else report["class_macro_metrics"]
        value = values.get(metric)
        if value is None or not np.isfinite(value):
            raise ValueError(f"Selection metric is unavailable: {family}/{group}/{metric}")
        candidates.append({"alpha": alpha, "value": value})
        hashes[alpha_name(alpha)] = file_hash(folder / "completed.json")
    best = max(candidates, key=lambda r: (r["value"], -r["alpha"]))
    payload = {"valid_signature": signature, "selection_split": "valid", "family": family,
               "group": group, "metric": metric, "tie_rule": "smaller alpha (more F)",
               "alpha": best["alpha"], "value": best["value"], "candidates": candidates,
               "valid_completed_sha256": hashes}
    return {**payload, "signature": digest(payload)}


def save_selection(output, selection, *, require_existing=False):
    path = Path(output) / "selection.json"
    if path.exists():
        if read_json(path) != selection:
            raise ValueError("Frozen valid selection changed; use a new output root")
    elif require_existing:
        raise ValueError("Run --stage valid before --stage test")
    else:
        write_json(path, selection)


def summarize(output):
    rows = []
    for split in ("valid", "test"):
        for folder in sorted((Path(output) / split).glob("alpha_*")):
            if not (folder / "completed.json").exists():
                continue
            for filename in ("metrics.json", "coverage_metrics.json"):
                report = read_json(folder / filename)
                groups = {"all": report["metrics"], "class_macro": report["class_macro_metrics"]}
                groups.update({g: report["groups"][g]["metrics"] for g in ("single_label", "mixed_label")})
                groups.update({f"label_count_{k}": v["metrics"] for k, v in report["groups"]["by_label_count"].items()})
                for group, metrics in groups.items():
                    rows.extend({"split": split, "alpha_global": float(folder.name[6:]), "group": group,
                                 "metric_version": report["metric_version"], "metric": key, "value": value}
                                for key, value in metrics.items())
    if rows:
        write_csv(Path(output) / "comparison.csv", rows, list(rows[0]))
        write_json(Path(output) / "comparison.json", rows)


def bundle_for(protocol, labels):
    _, classes = np.unique(labels, axis=0, return_inverse=True)
    return SimpleNamespace(labels=labels, class_ids=classes, valid_ids=protocol.ids["valid"],
                           reserved_ids=protocol.ids["test"])
