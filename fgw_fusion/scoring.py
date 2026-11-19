"""Frozen F/G/WBBS formulas without an angular multiplier or new selection."""
from collections.abc import Mapping
from copy import deepcopy

import numpy as np

SETTINGS = {
    "schema_version": 1, "recipe": "fgw_fixed_v1",
    "ks": [1, 5, 10, 20, 50, 100, 500], "tie_seed": 2026,
    "training_required": False, "posthoc_research": True,
    "test_parameter_search": False,
    "parameter_selection": "none; frozen F/G weights and WBBS parameters",
}


def validate_settings(settings):
    if not isinstance(settings, Mapping) or set(settings) != set(SETTINGS):
        raise ValueError("Unknown FGW settings keys")
    if any(settings[key] != expected for key, expected in SETTINGS.items() if key != "ks"):
        raise ValueError("Only the frozen FGW recipe is supported")
    ks = settings["ks"]
    if (not isinstance(ks, list) or not ks or any(type(k) is not int or k < 1 for k in ks)
            or ks != sorted(set(ks))):
        raise ValueError("K values must be unique, sorted positive integers")
    return deepcopy(dict(settings))


def specs(split=None, settings=None):
    if split is not None and split not in ("valid", "test"):
        raise ValueError("FGW requires an explicit valid or test split")
    if settings is not None:
        validate_settings(settings)
    def spec(local, global_weight):
        return {"local_weight": local, "global_weight": global_weight,
                "wbbs_power": 1., "angular_gamma": 0., "local_mode": "optimal"}
    return {"FGW": spec(.5, .5), "GW": spec(0., 1.), "FW": spec(1., 0.)}


def validate_spec(spec):
    if (not isinstance(spec, Mapping) or not any(dict(spec) == known for known in specs().values())
            or any(isinstance(spec.get(key), bool) for key in
                   ("local_weight", "global_weight", "wbbs_power", "angular_gamma"))):
        raise ValueError("Unknown or modified frozen FGW specification")
    return dict(spec)


def unit_vector(value, name, shape=None):
    values = np.asarray(value, dtype=np.float64)
    if (values.ndim != 1 or (shape is not None and values.shape != shape)
            or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1)):
        raise ValueError("Expected aligned finite unit-interval vector: " + name)
    return values


def score_variant(values, wbbs_scores, spec):
    spec = validate_spec(spec)
    if not isinstance(values, Mapping) or not {"optimal_f_score", "global_score"} <= set(values):
        raise ValueError("Missing frozen F/G score components")
    wbbs = unit_vector(wbbs_scores, "WBBS")
    local = unit_vector(values["optimal_f_score"], "optimal F", wbbs.shape)
    global_score = unit_vector(values["global_score"], "global G", wbbs.shape)
    return (spec["local_weight"] * local + spec["global_weight"] * global_score) * wbbs
