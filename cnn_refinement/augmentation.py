"""Original-grid context dropout; historical patch/view code remains read-only."""
from pathlib import Path

import numpy as np
import torch

from experiments.proposed._shared.ef_suite import RegionViews
from f_region_structure.state import read_signed, record, signed
from wbm_retrieval.common import digest, file_hash
from wbm_retrieval.regions import resize_nearest


def original_window(raw, target, window_size=52):
    """Mirror frozen fixed_window_patch before its nearest-neighbor resize."""
    points = np.argwhere(target)
    if raw.shape != target.shape or not len(points):
        raise ValueError("Expected a nonempty target on the original wafer grid")
    row, col = (int(round(float(v))) - window_size // 2 for v in points.mean(0))
    r0, c0 = max(row, 0), max(col, 0)
    r1, c1 = min(row + window_size, raw.shape[0]), min(col + window_size, raw.shape[1])
    channels = np.stack((raw == 2, target > 0, (raw == 1) | (raw == 2))).astype(np.uint8)
    result = np.zeros((3, window_size, window_size), dtype=np.uint8)
    result[:, r0-row:r1-row, c0-col:c1-col] = channels[:, r0:r1, c0:c1]
    return result


def context_dropout(window, rng, probability=.5):
    """Only delete D & ~T pixels; the draw is on the 52x52 original grid."""
    window = np.asarray(window)
    if window.shape != (3, 52, 52) or not np.isin(window, (0, 1)).all():
        raise ValueError("Context dropout requires a binary 3x52x52 window")
    if not 0 <= probability <= 1:
        raise ValueError("Drop probability must lie in [0, 1]")
    defect, target, support = window.astype(bool)
    if np.any(target & ~defect) or np.any(defect & ~support):
        raise ValueError("Expected T subset D subset W")
    context = defect & ~target
    removed = context & (rng.random(context.shape) < probability)
    result = window.copy()
    result[0, removed] = 0
    return result, {"context_pixels": int(context.sum()), "removed_pixels": int(removed.sum()),
                    "no_op": int(not removed.any()), "no_context": int(not context.any())}


def prepare_context_windows(inputs, destination):
    """Prepare train-only uint8 windows and audit EVERY token against frozen F.npy."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    path, marker = destination / "train_windows52.npy", destination / "train_windows52.json"
    data = inputs["datasets"]["train"]
    signature = digest({"version": 1, "prepared_marker": data["prepared_marker"],
                        "source_manifest": inputs["source_manifest"]["signature"],
                        "augmentation_code_sha256": file_hash(__file__), "window_size": 52})
    if marker.exists():
        receipt = read_signed(marker)
        if receipt["input_signature"] != signature or file_hash(path) != receipt["file_sha256"]:
            raise ValueError("Prepared context windows changed; use a new output root")
        return path
    recovering = path.exists()
    temporary = path.with_suffix(".building.npy")
    shape = (len(data["patches"]), 3, 52, 52)
    windows = (np.load(path, mmap_mode="r") if recovering else
               np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8, shape=shape))
    if windows.shape != shape or windows.dtype != np.uint8:
        raise ValueError("Unsealed context-window shape/dtype changed")
    context_total = retained_total = 0
    for owner, sid in enumerate(data["map_ids"]):
        start, end = map(int, data["offsets"][owner:owner+2])
        for index in range(start, end):
            target = data["region_masks"][owner] == int(data["token_region_ids"][index])
            if int(target.sum()) != int(data["area"][index]):
                raise ValueError(f"Original target area differs for token {index}")
            window = original_window(inputs["maps"][int(sid)], target)
            patch = np.stack([resize_nearest(channel, 96) for channel in window])
            if not np.array_equal(patch, data["patches"][index]):
                raise ValueError(f"Reconstructed context window differs from frozen token {index}")
            if recovering:
                if not np.array_equal(windows[index], window):
                    raise ValueError(f"Unsealed context window differs at token {index}")
            else:
                windows[index] = window
            context_total += int(((window[0] > 0) & (window[1] == 0)).sum())
            retained_total += int(window[1].sum())
        if (owner+1) % 5000 == 0:
            print(f"context windows audited: {owner+1}/{len(data['map_ids'])} wafers", flush=True)
    windows.flush()
    del windows
    if not recovering:
        temporary.replace(path)
    record(marker, signed({"input_signature": signature, "file_sha256": file_hash(path),
                           "tokens_checked": len(data["patches"]), "mismatched_tokens": 0,
                           "context_pixels_total": context_total, "retained_target_pixels_total": retained_total,
                           "dtype": "uint8", "shape": [len(data["patches"]), 3, 52, 52]}))
    return path


class ContextViews(RegionViews):
    """Preserve original rotation draws and original (first) view exactly."""
    def __init__(self, cache, split, spec, seed, windows_path, epoch=0):
        if split != "train" or spec.get("scale_range") is not None:
            raise ValueError("ContextViews is the fixed-window training augmentation only")
        super().__init__(cache, split, spec, seed, epoch)
        self.windows_path, self.windows = str(windows_path), None
        shape = np.load(self.windows_path, mmap_mode="r").shape
        if shape != (self.length, 3, 52, 52):
            raise ValueError("Context window count/order differs from cached train tokens")

    def __getstate__(self):
        return {**super().__getstate__(), "windows": None}

    def __getitem__(self, index):
        index, original, _, _ = super().__getitem__(index)
        if self.windows is None:
            self.windows = np.load(self.windows_path, mmap_mode="r")
        # A separate numpy stream leaves the historical per-token rotation unchanged.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index), 527]))
        changed, stats = context_dropout(self.windows[index], rng)
        resized = np.stack([resize_nearest(channel, 96) for channel in changed]).astype(np.float32)
        turns = int(np.random.default_rng(np.random.SeedSequence(
            [self.seed, self.epoch, int(index)])).integers(4))
        view = np.ascontiguousarray(np.rot90(resized, turns, axes=(-2, -1)))
        return index, original, torch.from_numpy(view), stats
