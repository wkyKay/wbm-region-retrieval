"""Conservative opening of a normalized region view, never of the proposal map."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .regions import augment


def validate_opening(settings):
    if settings is None:
        return
    if (not isinstance(settings.get("enabled", False), bool)
            or settings.get("kernel") != "cross3" or settings.get("iterations") != 1
            or not 0 <= settings.get("probability", -1) <= 1
            or not 0 < settings.get("min_retained_fraction", -1) <= 1):
        raise ValueError("Region opening requires cross3, one iteration, probability in [0,1] and retention in (0,1]")


def region_view(image, rng, augmentation, opening=None):
    """Return one composed view and audit information; original image is untouched.

    The kernel is measured in normalized input pixels. No recropping follows
    the opening. Failed retention checks revert only the opening; the usual
    rotation can still be applied. Disabled opening consumes no extra RNG draws.
    """
    output = np.array(image, dtype=np.float32, copy=True)
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError("Region views require a single-channel CxHxW patch")
    before = int(np.count_nonzero(output))
    info = {"attempted": 0, "accepted": 0, "changed": 0, "fallback": 0,
            "before_pixels": before, "after_pixels": before, "candidate_retention": 1.0}
    if opening and opening.get("enabled", False) and rng.random() < opening["probability"]:
        info["attempted"] = 1
        mask = output[0] > 0
        opened = ndimage.binary_opening(mask, structure=ndimage.generate_binary_structure(2, 1), iterations=1)
        after = int(opened.sum())
        fraction = after / before if before else 0.0
        info["candidate_retention"] = fraction
        if after and fraction >= opening["min_retained_fraction"]:
            output[0] = opened
            info.update(accepted=1, changed=int(not np.array_equal(mask, opened)), after_pixels=after)
        else:
            info["fallback"] = 1
    return augment(output, rng, **augmentation), info


def opening_summary(totals):
    count, attempted = totals["views"], totals["attempted"]
    return {**totals, "attempt_rate": attempted / count if count else 0.,
            "accepted_per_attempt": totals["accepted"] / attempted if attempted else None,
            "changed_view_rate": totals["changed"] / count if count else 0.,
            "fallback_per_attempt": totals["fallback"] / attempted if attempted else None,
            "foreground_retained_fraction": totals["after_pixels"] / totals["before_pixels"]
            if totals["before_pixels"] else None}


def save_view_examples(config, dataset):
    """Fixed valid examples of every available single-label class and mixed maps."""
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .common import write_csv
    from .data import load_regions

    folder = Path(config["output_dir"]) / "models/local"
    if (folder / "augmentation_examples.csv").is_file():
        return
    metadata = load_regions(config, "valid")
    labels = np.load(Path(config["output_dir"]) / "cache/labels.npy", mmap_mode="r")
    selected, quotas = [], {}
    for i, map_id in enumerate(metadata["map_ids"]):
        start, end = metadata["offsets"][i:i+2]
        if start == end:
            continue
        active = np.flatnonzero(labels[map_id])
        key = str(int(active[0])) if len(active) == 1 else "mixed"
        quota = 4 if key == "mixed" else 2
        if quotas.get(key, 0) < quota:
            selected.append((int(start), int(map_id), "+".join(map(str, active))))
            quotas[key] = quotas.get(key, 0) + 1
    if not selected:
        return
    figure, axes = plt.subplots(len(selected), 3, figsize=(9, 2.1 * len(selected)), squeeze=False)
    rows = []
    for row, (token_id, map_id, label) in enumerate(selected):
        _, original, view, info = dataset[token_id]
        rng = np.random.default_rng(np.random.SeedSequence([dataset.seed, dataset.epoch, token_id]))
        opened, _ = region_view(original.numpy(), rng,
                                {"rotation": False, "max_shift": 0, "dropout": 0., "noise": 0.}, dataset.opening)
        for ax, pixels, title in zip(axes[row], (original[0], opened[0], view[0]),
                                     ("original", "opening before rotation", "training view")):
            ax.imshow(pixels, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(f"map {map_id}, label {label}", fontsize=8)
        axes[row, 2].set_xlabel(f"attempt={info['attempted']}, change={info['changed']}, fallback={info['fallback']}", fontsize=8)
        rows.append({"map_id": map_id, "token_id": token_id, "labels": label, **info})
    figure.tight_layout()
    figure.savefig(folder / "augmentation_examples.png", dpi=110)
    plt.close(figure)
    write_csv(folder / "augmentation_examples.csv", rows, list(rows[0]))
