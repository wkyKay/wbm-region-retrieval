"""Optional single-variable geometry examples for a trained local encoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import write_csv, write_json
from .data import load_regions
from .matching import map_similarity


def generate_examples(config):
    # This is a controlled feature-level experiment, separate from Experiment A.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .common import get_device
    from .regions import extract_regions
    from .methods.proposed import load_encoder
    from .retrieval import feature_cache

    output = Path(config["output_dir"]) / "geometry"
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(config["output_dir"]) / "cache"
    metadata = load_regions(config, "test")
    device = get_device(config["device"])
    encoder, checkpoint = load_encoder(config, device)
    embeddings = feature_cache(encoder, cache / "test_patches.npy", checkpoint, cache / "test_local_embeddings.npy", config, device)
    settings = {**config["matching"], "lambda_pos": config["geometry"]["lambda_pos"],
                "lambda_scale": config["geometry"]["lambda_scale"]}
    images = np.load(cache / "maps.npy", mmap_mode="r")
    examples, rows = 0, []
    for map_index, sample_id in enumerate(metadata["map_ids"]):
        start, end = metadata["offsets"][map_index:map_index+2]
        if start == end:
            continue
        # The single query token is kept fixed to isolate exactly one variable.
        q = {"embedding": np.array(embeddings[start:start+1]), "area": metadata["area"][start:start+1],
             "position": metadata["position"][start:start+1].copy(), "scale": metadata["scale"][start:start+1].copy()}
        mask = extract_regions(images[sample_id], **config["proposal"])[0]["mask"]
        figure, axes = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
        axes[0].imshow(mask, cmap="gray_r")
        axes[0].set_title(f"Query token, wafer {sample_id}")
        axes[0].axis("off")
        for axis, variable, magnitudes in ((axes[1], "position", [0., .1, .2, .4, .6]),
                                          (axes[2], "scale", [1., 1.25, 1.5, 2., 3.])):
            scores = []
            for magnitude in magnitudes:
                candidate = {key: np.array(value, copy=True) for key, value in q.items()}
                if variable == "position":
                    candidate["position"][0, 0] += magnitude
                else:
                    candidate["scale"] *= magnitude
                shape_only = map_similarity(q, candidate, config["matching"])
                geometry = map_similarity(q, candidate, settings)
                scores.append(geometry)
                rows.append({"sample_id": int(sample_id), "variable": variable, "magnitude": magnitude,
                             "shape_only_score": shape_only, "geometry_score": geometry})
            axis.plot(magnitudes, scores, marker="o", label="shape + geometry")
            axis.axhline(1, color="gray", linestyle="--", label="shape only")
            axis.set(xlabel="centroid shift / wafer radius" if variable == "position" else "PCA scale ratio",
                     ylabel="Similarity", ylim=(.7, 1.02), title=f"Only {variable} changes")
            axis.legend(fontsize=8)
        figure.savefig(output / f"wafer_{sample_id}.png", dpi=180)
        plt.close(figure)
        examples += 1
        if examples >= config["geometry"]["examples"]:
            break
    if rows:
        write_csv(output / "perturbations.csv", rows, list(rows[0]))
    write_json(output / "experiment.json", {"type": "controlled feature perturbation", "settings": settings,
               "examples": examples, "shape_embedding_fixed": True,
               "scope": "Tests scoring sensitivity only; this is not an image-level robustness or segmentation benchmark"})
    print(f"Geometry examples: {output}", flush=True)
