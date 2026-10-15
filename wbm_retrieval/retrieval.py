"""Shared feature caching, local ranking I/O and dispatch to method modules."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .common import budget_name, file_hash, read_json, write_json
from .matching import map_similarity


class Images(Dataset):
    def __init__(self, path):
        self.path, self.array = str(path), None
        self.length = len(np.load(self.path, mmap_mode="r"))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.array is None:
            self.array = np.load(self.path, mmap_mode="r")
        return torch.from_numpy(np.array(self.array[index], dtype=np.float32))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["array"] = None
        return state


@torch.no_grad()
def encode_images(encoder, path, config, device):
    encoder.eval()
    batches = DataLoader(Images(path), batch_size=config["inference_batch_size"], num_workers=config["num_workers"],
                         pin_memory=device.type == "cuda")
    values = [encoder(batch.to(device)).cpu().numpy() for batch in batches]
    return np.concatenate(values).astype(np.float32)


def feature_cache(encoder, source, checkpoint, target, config, device):
    metadata_path = target.with_suffix(".json")
    signature = {"checkpoint_sha256": file_hash(checkpoint),
                 "prepared_signature": read_json(Path(config["output_dir"]) / "cache/prepared.json")["signature"],
                 "input_array": Path(source).name}
    if target.exists() and metadata_path.exists() and read_json(metadata_path) == signature:
        return np.load(target, mmap_mode="r")
    representations = encode_images(encoder, source, config, device)
    np.save(target, representations)
    write_json(metadata_path, signature)
    return representations


def rank_local(name, groups, config, protocol):
    output = Path(config["output_dir"]) / "retrieval" / name
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    match_fields = ["query_id", "candidate_id", "candidate_rank", "query_token", "candidate_token", "score",
                    "shape", "position_affinity", "scale_affinity", "query_weight"]
    with (output / "rankings.csv").open("w", newline="") as ranking_file, (output / "matches.csv").open("w", newline="") as match_file:
        ranking_writer = csv.writer(ranking_file)
        ranking_writer.writerow(["query_id", "rank", "candidate_id", "similarity_score"])
        match_writer = csv.DictWriter(match_file, fieldnames=match_fields)
        match_writer.writeheader()
        for query_index, query_id in enumerate(protocol.queries, 1):
            scored = [(candidate_id, map_similarity(groups[query_id], groups[candidate_id], config["matching"]))
                      for candidate_id in protocol.candidates[query_id]]
            scored.sort(key=lambda value: (-value[1], value[0]))
            for rank, (candidate_id, score) in enumerate(scored, 1):
                ranking_writer.writerow([query_id, rank, candidate_id, score])
                if rank <= config["evaluation"]["save_matches_top_k"]:
                    _, matches = map_similarity(groups[query_id], groups[candidate_id], config["matching"], details=True)
                    match_writer.writerows({"query_id": query_id, "candidate_id": candidate_id, "candidate_rank": rank, **match}
                                           for match in matches)
            if query_index % 50 == 0:
                print(f"{name}: {query_index}/{len(protocol.queries)} queries", flush=True)
    write_json(output / "retrieval_info.json", {"method": name, "matching": config["matching"],
               "ranking_wall_seconds_excluding_encoding": time.perf_counter() - started,
               "num_queries": len(protocol.queries), "shape_only": True,
               "empty_query_maps": sum(not len(groups[q]["area"]) for q in protocol.queries)})


def retrieve(config, protocol, methods=None):
    """Dispatch retrieval while leaving method logic in its own implementation."""
    available = ["handcrafted", "proposed"] + [budget_name(b) for b in config["global"]["budgets"]]
    names = methods or available
    if not set(names) <= set(available):
        raise ValueError(f"Unknown retrieval methods: {sorted(set(names) - set(available))}")
    if "handcrafted" in names:
        from .methods import handcrafted
        handcrafted.retrieve(config, protocol)
    if "proposed" in names:
        from .methods import proposed
        proposed.retrieve(config, protocol)
    global_names = [name for name in names if name.startswith("global_")]
    if global_names:
        from .methods import supervised_global
        supervised_global.retrieve(config, protocol, global_names)
