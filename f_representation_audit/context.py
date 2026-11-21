"""Deterministic, read-only probes of a frozen F encoder's context sensitivity.

Channels are all defects D, the designated target T, and wafer support W.
The four default views preserve T <= D <= W; the optional target-removal
control deliberately violates T <= D and is not a valid region example.
It must never enter training, a retrieval gallery, or reported retrieval scores.

Prefer original 52x52 windows, byte-check their resized originals against the
frozen cache, and then perturb on that original grid.  ``model_input`` is an
explicit fallback whose pixel counts cannot be interpreted as die counts.
"""

from __future__ import annotations

import hashlib
from numbers import Integral, Real
from typing import Sequence

import numpy as np


VIEW_NAMES = ("original", "rot90", "target_only", "context_drop")
CONTROL_NAME = "target_removed_control"


def resize_windows_nearest(windows: np.ndarray, size: int) -> np.ndarray:
    """Match the historical F nearest-neighbor index rule without importing it."""
    if isinstance(size, bool) or not isinstance(size, Integral) or size < 1:
        raise ValueError("output_size must be a positive integer")
    data = np.asarray(windows)
    if data.ndim != 4 or data.shape[1] != 3 or min(data.shape[-2:]) < 1:
        raise ValueError("Expected windows shaped (N, 3, H, W)")
    h, w = data.shape[-2:]
    rr = np.minimum((np.arange(size) * h / size).astype(int), h - 1)
    cc = np.minimum((np.arange(size) * w / size).astype(int), w - 1)
    return np.array(data[:, :, rr[:, None], cc[None, :]], dtype=np.float32,
                    order="C", copy=True)


def _validate(windows, region_keys, seed, probability, input_grid):
    data = np.asarray(windows)
    if (data.ndim != 4 or data.shape[1] != 3 or data.shape[0] < 1
            or data.shape[-2] < 1 or data.shape[-2] != data.shape[-1]):
        raise ValueError("Expected nonempty square windows shaped (N, 3, H, H)")
    if data.dtype.kind not in "buif" or not np.isfinite(data).all():
        raise ValueError("Windows must contain finite numeric binary channels")
    if not np.logical_or(data == 0, data == 1).all():
        raise ValueError("Windows must contain binary channels, without interpolation")
    if np.any(data[:, 1] > data[:, 0]) or np.any(data[:, 0] > data[:, 2]):
        raise ValueError("Every input must satisfy target <= defects <= support")
    if np.any(data[:, 1].sum(axis=(1, 2)) == 0):
        raise ValueError("A context probe requires a nonempty designated target")
    if isinstance(region_keys, (str, bytes)):
        raise ValueError("region_keys must be a sequence of keys, not a single string")
    keys = list(region_keys)
    if (len(keys) != len(data) or any(not isinstance(key, str) or not key for key in keys)
            or len(set(keys)) != len(keys)):
        raise ValueError("region_keys must be unique nonempty strings, one per window")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (isinstance(probability, (bool, np.bool_)) or not isinstance(probability, Real)
            or not np.isfinite(probability) or not 0 <= probability <= 1):
        raise ValueError("dropout_probability must be a finite value in [0, 1]")
    if input_grid not in {"raw_window", "model_input"}:
        raise ValueError("input_grid must be raw_window or model_input")
    expected_side = 52 if input_grid == "raw_window" else 96
    if data.shape[-1] != expected_side:
        raise ValueError(f"Frozen F {input_grid} requires {expected_side}x{expected_side} input; "
                         "declare the actual grid rather than relabeling resized pixels as dies")
    return np.array(data, dtype=np.float32, order="C", copy=True), keys


def _rng(seed: int, key: str) -> np.random.Generator:
    payload = f"f_context_probe_v1\n{int(seed)}\n{key}".encode("utf-8")
    entropy = int.from_bytes(hashlib.sha256(payload).digest()[:16], "little")
    return np.random.Generator(np.random.PCG64(entropy))


def _counts(reference, changed):
    target = reference[1].astype(bool)
    context = reference[0].astype(bool) & ~target
    target_pixels, context_pixels = int(target.sum()), int(context.sum())
    context_changed = int(np.count_nonzero((reference[0] != changed[0]) & context))
    return {
        "target_pixels": target_pixels,
        "context_pixels": context_pixels,
        "context_changed_pixels": context_changed,
        "context_changed_fraction": context_changed / context_pixels if context_pixels else None,
        "target_mask_retained_fraction": float(changed[1][target].sum()) / target_pixels,
        "target_evidence_retained_fraction": float(changed[0][target].sum()) / target_pixels,
        "all_defect_changed_pixels": int(np.count_nonzero(reference[0] != changed[0])),
    }


def build_context_views(
    windows: np.ndarray,
    region_keys: Sequence[str],
    *,
    seed: int = 2026,
    dropout_probability: float = 0.5,
    output_size: int = 96,
    input_grid: str = "raw_window",
    include_target_removed_control: bool = False,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Return independent float32 view arrays and per-window/view audit records.

    ``region_keys`` should encode frozen sample ID and region index. Randomness
    depends on that key, not batch position, so chunking and reordering preserve
    the views. Context dropout removes whole input-grid defect pixels outside
    T. It never modifies target or support channels. Rot90 is applied AFTER
    resizing, exactly like F training, because resize and rotation need not
    commute. Audit counts align that view back before comparison.

    Deleting context is an input sensitivity intervention, not a claim that
    wafer labels or physical root causes remain unchanged. Context may include
    another fragment of the same semantic defect. The optional target removal
    keeps T solely as a location probe; it is explicitly outside the valid
    three-channel input relation and is not a matched semantic counterexample.
    """
    if not isinstance(include_target_removed_control, bool):
        raise ValueError("include_target_removed_control must be boolean")
    original, keys = _validate(windows, region_keys, seed, dropout_probability, input_grid)
    source = {"original": original, "target_only": original.copy(),
              "context_drop": original.copy()}
    source["target_only"][:, 0] = original[:, 1]
    for index, key in enumerate(keys):
        context = (original[index, 0] == 1) & (original[index, 1] == 0)
        removed = context & (_rng(seed, key).random(context.shape) < dropout_probability)
        source["context_drop"][index, 0, removed] = 0
    if include_target_removed_control:
        source[CONTROL_NAME] = original.copy()
        source[CONTROL_NAME][:, 0] *= 1 - original[:, 1]

    resized = {name: resize_windows_nearest(value, output_size) for name, value in source.items()}
    if np.any(resized["original"][:, 1].sum(axis=(1, 2)) == 0):
        raise ValueError("Resize erased a target; use an output size preserving target support")
    views = {"original": resized["original"],
             "rot90": np.ascontiguousarray(np.rot90(resized["original"], 1, axes=(-2, -1))),
             "target_only": resized["target_only"], "context_drop": resized["context_drop"]}
    if include_target_removed_control:
        views[CONTROL_NAME] = resized[CONTROL_NAME]

    audits = []
    for index, key in enumerate(keys):
        for name, batch in views.items():
            is_rotation = name == "rot90"
            source_view = original[index] if is_rotation else source[name][index]
            aligned = np.rot90(batch[index], -1, axes=(-2, -1)) if is_rotation else batch[index]
            before, after = _counts(original[index], source_view), _counts(resized["original"][index], aligned)
            audit = {
                "region_key": key, "view": name, "input_grid": input_grid,
                "source_height": int(original.shape[-2]), "source_width": int(original.shape[-1]),
                "output_height": int(output_size), "output_width": int(output_size),
                "rotation_quarter_turns": int(is_rotation),
                "valid_region_input": name != CONTROL_NAME, "diagnostic_only": True,
                "has_context": before["context_pixels"] > 0,
                "is_noop": bool(np.array_equal(batch[index], resized["original"][index])),
                "source_context_unchanged": before["context_changed_pixels"] == 0,
                "support_unchanged_after_alignment": bool(np.array_equal(aligned[2], resized["original"][index, 2])),
                "seed": int(seed), "dropout_probability": float(dropout_probability),
                "source_pixel_unit": "die" if input_grid == "raw_window" else "resized_pixel",
            }
            audit.update({"source_" + field: value for field, value in before.items()})
            audit.update({"model_" + field: value for field, value in after.items()})
            # Common metrics are source-grid values; explicit prefixed copies
            # keep downstream code from silently substituting model-grid counts.
            for field in ("context_changed_fraction", "target_mask_retained_fraction",
                          "target_evidence_retained_fraction"):
                audit[field] = before[field]
            audits.append(audit)
    return views, audits
