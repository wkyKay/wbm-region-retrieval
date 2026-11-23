"""Five single WaPIRL augmentations adapted to the frozen F D/T/W input.

Policies follow hgkahng/WaPIRL 10e24c3 and Albumentations 0.4.6.  The
NumPy per-token RNG is intentionally independent of training's global RNGs;
this reproduces distributions and operations, not the author's random stream.
"""
from pathlib import Path
import math

import cv2
import numpy as np
import torch

from experiments.proposed._shared.ef_suite import RegionViews

VARIANTS = ("crop", "cutout", "noise", "rotation", "shift")
AUGMENTATION_SPECS = {
    "crop": {"area_scale": [.5, 1.], "aspect_ratio": [.9, 1.1],
             "attempts": 10, "aspect_sampling": "uniform in log aspect ratio",
             "interpolation": "OpenCV INTER_NEAREST", "probability": 1.,
             "crop_start": "Albumentations0.4.6 int((side-size)*(index/(side-size+1e-10)))",
             "fallback": "central crop by ratio; full96 square for this fixed input"},
    "cutout": {"holes": 4, "side": 19, "probability": .5,
               "center_sampling": "integer uniform inclusive [0,96] per axis",
               "bounds": "clip(center-19//2), then clip(start+19), matching Albumentations0.4.6",
               "fill": 0, "channels": "all D/T/W together"},
    "noise": {"probability": .05, "selection": "independent Bernoulli on valid support",
              "replacement": "uniform normal/defect, including possible unchanged values",
              "target_update": "T_new=T_old & D_new", "support": "unchanged",
              "probability_apply_transform": 1.},
    "rotation": {"angle_degrees": [-180., 180.], "probability": 1.,
                 "center_xy": [48., 48.], "interpolation": "OpenCV INTER_NEAREST",
                 "border": "constant zero", "channels": "all D/T/W together"},
    "shift": {"fraction_x": [-.25, .25], "fraction_y": [-.25, .25],
              "probability": 1., "offset": "continuous fraction times96; no pre-rounding",
              "interpolation": "OpenCV INTER_NEAREST", "border": "constant zero",
              "channels": "all D/T/W together"},
}

# Same source coordinates as the three input planes; zero is reserved for fill.
_SOURCE_IDS = np.arange(1, 96*96+1, dtype=np.float32).reshape(96, 96)


def validate_patch(patch):
    value = np.asarray(patch)
    if value.shape != (3, 96, 96) or not np.all((value == 0) | (value == 1)):
        raise ValueError("Expected a binary frozen 3x96x96 F patch")
    if np.any(value[1] > value[0]) or np.any(value[0] > value[2]):
        raise ValueError("Target must be a subset of defects, and defects of support")
    return value


def _crop_parameters(rng):
    """Preserve the old library's float start conversion, including truncation."""
    for attempt in range(1, 11):
        area = float(rng.uniform(.5, 1.)) * 96 * 96
        aspect = math.exp(float(rng.uniform(math.log(.9), math.log(1.1))))
        width, height = int(round(math.sqrt(area*aspect))), int(round(math.sqrt(area/aspect)))
        if 0 < width <= 96 and 0 < height <= 96:
            row, col = int(rng.integers(0, 96-height+1)), int(rng.integers(0, 96-width+1))
            top = int((96-height) * (row / (96-height+1e-10)))
            left = int((96-width) * (col / (96-width+1e-10)))
            return top, left, height, width, attempt
    return 0, 0, 96, 96, 11


def augment_patch(patch, variant, rng):
    """Return an independent float32 view and scalar diagnostics.

    The fourth temporary plane tracks which original target pixels actually
    survive nearest-neighbor sampling.  It is never supplied to the network.
    Empty targets are reported, never silently rejected or replaced.
    """
    if variant not in VARIANTS:
        raise ValueError("Unknown single augmentation: "+str(variant))
    original = validate_patch(patch)
    planes = np.concatenate((original.astype(np.float32, copy=True), _SOURCE_IDS[None]), axis=0)
    image = np.ascontiguousarray(planes.transpose(1, 2, 0))
    parameters = {"crop_top": 0, "crop_left": 0, "crop_height": 96, "crop_width": 96,
                  "crop_attempts": 0, "angle_degrees": 0., "shift_fraction_x": 0.,
                  "shift_fraction_y": 0., "cutout_holes": 0}
    applied, noise_selected, noise_added, noise_removed = 1, 0, 0, 0
    if variant == "crop":
        top, left, height, width, attempts = _crop_parameters(rng)
        image = cv2.resize(image[top:top+height, left:left+width], (96, 96), interpolation=cv2.INTER_NEAREST)
        parameters.update(crop_top=top, crop_left=left, crop_height=height, crop_width=width,
                          crop_attempts=attempts)
    elif variant == "cutout":
        applied = int(rng.random() < .5)
        if applied:
            for _ in range(4):
                row, col = int(rng.integers(0, 97)), int(rng.integers(0, 97))
                top, left = int(np.clip(row-9, 0, 96)), int(np.clip(col-9, 0, 96))
                bottom, right = min(top+19, 96), min(left+19, 96)
                image[top:bottom, left:right] = 0
            parameters["cutout_holes"] = 4
    elif variant == "noise":
        selected = (rng.random((96, 96)) < .05) & (image[:, :, 2] > 0)
        replacements = rng.integers(0, 2, size=(96, 96), dtype=np.uint8)
        before = image[:, :, 0].copy()
        image[:, :, 0][selected] = replacements[selected]
        image[:, :, 1] *= image[:, :, 0]
        noise_selected = int(selected.sum())
        noise_added = int(((before == 0) & (image[:, :, 0] == 1)).sum())
        noise_removed = int(((before == 1) & (image[:, :, 0] == 0)).sum())
    else:
        if variant == "rotation":
            angle = float(rng.uniform(-180., 180.))
            matrix = cv2.getRotationMatrix2D((48., 48.), angle, 1.)
            parameters["angle_degrees"] = angle
        else:
            dx, dy = float(rng.uniform(-.25, .25)), float(rng.uniform(-.25, .25))
            matrix = cv2.getRotationMatrix2D((48., 48.), 0., 1.)
            matrix[0, 2] += dx*96
            matrix[1, 2] += dy*96
            parameters.update(shift_fraction_x=dx, shift_fraction_y=dy)
        image = cv2.warpAffine(image, matrix, (96, 96), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    result = np.ascontiguousarray(image[:, :, :3].transpose(2, 0, 1))
    validate_patch(result)
    target_before, target_after = int(original[1].sum()), int(result[1].sum())
    retained_ids = np.unique(image[:, :, 3][result[1] > 0]).astype(np.int64)
    retained_ids = retained_ids[retained_ids > 0]
    if len(retained_ids) and not np.all(original[1].reshape(-1)[retained_ids-1] == 1):
        raise AssertionError("Augmentation created a target without a source target pixel")
    retained = len(retained_ids)
    if retained > target_before:
        raise AssertionError("Source-space target retention exceeds original target")
    support_before, support_after = int(original[2].sum()), int(result[2].sum())
    support_ids = np.unique(image[:, :, 3][result[2] > 0]).astype(np.int64)
    support_ids = support_ids[support_ids > 0]
    if len(support_ids) and not np.all(original[2].reshape(-1)[support_ids-1] == 1):
        raise AssertionError("Augmentation created support without an original support pixel")
    stats = {"applied": applied, "no_op": int(np.array_equal(result, original)),
             "target_before_pixels": target_before, "target_after_pixels": target_after,
             "target_retained_source_pixels": retained,
             "target_retention_fraction": retained/target_before if target_before else 0.,
             "target_retention_eligible": int(target_before > 0),
             "empty_target_before": int(target_before == 0), "empty_target_after": int(target_after == 0),
             "newly_empty_target": int(target_before > 0 and target_after == 0),
             "support_before_pixels": support_before, "support_after_pixels": support_after,
             "support_retained_source_pixels": len(support_ids),
             "support_retention_fraction": len(support_ids)/support_before if support_before else 0.,
             "support_retention_eligible": int(support_before > 0),
             "changed_positions": int(np.any(result != original, axis=0).sum()),
             "noise_selected_pixels": noise_selected, "noise_added_defects": noise_added,
             "noise_removed_defects": noise_removed, "noise_changed_pixels": noise_added+noise_removed,
             **parameters}
    return result, stats


class PaperViews(RegionViews):
    """Replace only the training second view, without consuming shared RNGs."""
    def __init__(self, cache, split, spec, seed, variant, epoch=0):
        if split != "train" or variant not in VARIANTS or spec.get("scale_range") is not None:
            raise ValueError("PaperViews requires a named single-augmentation fixed-window train view")
        super().__init__(cache, split, spec, seed, epoch)
        self.variant = variant

    def __getitem__(self, index):
        if self.patches is None:
            self.patches = np.load(self.patch_path, mmap_mode="r", allow_pickle=False)
        original = np.array(self.patches[index], dtype=np.float32)
        # A stable integer, not Python's randomized hash, identifies each stream.
        rng = np.random.default_rng(np.random.SeedSequence(
            [self.seed, self.epoch, int(index), 731, VARIANTS.index(self.variant)+1]))
        view, stats = augment_patch(original, self.variant, rng)
        return int(index), torch.from_numpy(original), torch.from_numpy(view), stats
