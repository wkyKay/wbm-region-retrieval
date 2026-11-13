"""Frozen radial + relative-area scoring from the 2026-11-11 valid selection."""
import numpy as np

SETTINGS = {"radial_bins": 16, "beta": 1.0, "scale_sigma": 1.5, "scale_weight": 0.15,
            "radial_center": "valid_support_bbox_center", "radius": "max_support_radius",
            "area_distance": "0.5*abs(log(relative_area_q/relative_area_c))",
            "radial_distance": "sum(abs(cdf_q-cdf_c))/bins"}


def geometry(mask, support):
    mask, support = np.asarray(mask, bool), np.asarray(support, bool)
    if mask.ndim != 2 or mask.shape != support.shape or np.any(mask & ~support):
        raise ValueError("Invalid mask/support")
    points, wafer = np.argwhere(mask), np.argwhere(support)
    if not len(points) or not len(wafer):
        raise ValueError("Empty region/support")
    center = (wafer.min(0) + wafer.max(0)) / 2
    radius = np.linalg.norm(wafer - center, axis=1).max()
    rho = np.linalg.norm(points - center, axis=1) / radius if radius > 0 else np.zeros(len(points))
    hist = np.histogram(np.clip(rho, 0, 1), bins=np.linspace(0, 1, SETTINGS["radial_bins"]+1))[0].astype(float)
    return hist / hist.sum(), len(points) / len(wafer)


def pair_scores(query_features, features, query_histograms, histograms, query_area, relative_area):
    """Q-token by gallery-token scores; callers expand one query at a time."""
    cosine = np.clip(query_features @ features.T, 0, 1)
    radial = np.abs(np.cumsum(query_histograms, axis=-1)[:, None] -
                    np.cumsum(histograms, axis=-1)[None]).sum(-1) / SETTINGS["radial_bins"]
    area_distance = .5 * np.abs(np.log(query_area)[:, None] - np.log(relative_area)[None])
    weight = SETTINGS["scale_weight"]
    return cosine * np.exp(-SETTINGS["beta"] * radial) * (
        1 - weight + weight * np.exp(-area_distance / SETTINGS["scale_sigma"]))
