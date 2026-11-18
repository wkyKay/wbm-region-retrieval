"""Fixed B1 component deletions; old model and descriptor code is read-only."""
from collections.abc import Mapping
import json
from pathlib import Path

import numpy as np

from b1_structure_fusion.scoring import score_variant as original_score_variant

CONFIG_PATH = Path(__file__).with_name("config.json")
DEFAULT_SETTINGS = {
    "schema_version": 1, "recipe": "b1_fixed_component_ablation_v1",
    "structure_strength": 2.0, "angular_gamma": 0.5, "fusion_weight": 0.5,
    "ks": [1, 5, 10, 20, 50, 100, 500], "tie_seed": 2026,
    "parameter_selection": "none; fixed deletion from frozen J_2",
    "training_required": False, "posthoc_research": True, "test_parameter_search": False,
    "valid_variants": ["no_local", "no_global", "no_angle", "greedy", "structure_only"],
    "test_additional_variants": ["I_2"],
}


NO_GLOBAL_ANGLE_SETTINGS = {
    **DEFAULT_SETTINGS,
    "recipe": "b1_no_global_angle_v1",
    "valid_variants": ["no_global_angle"],
    "test_additional_variants": [],
}


def validate_settings(settings):
    if not isinstance(settings, Mapping) or set(settings) != set(DEFAULT_SETTINGS):
        raise ValueError("Unknown ablation settings keys")
    result = dict(settings)
    expected = next((item for item in (DEFAULT_SETTINGS, NO_GLOBAL_ANGLE_SETTINGS)
                     if result.get("recipe") == item["recipe"]), None)
    if expected is None or any(result[key] != value for key, value in expected.items() if key != "ks"):
        raise ValueError("Only the frozen component ablation recipe is supported")
    ks = result["ks"]
    if (not isinstance(ks, list) or not ks or any(type(k) is not int or k < 1 for k in ks)
            or ks != sorted(set(ks))):
        raise ValueError("ks must be unique sorted positive integers")
    return result


def load_settings(path=None):
    with Path(path or CONFIG_PATH).open(encoding="utf-8") as handle:
        return validate_settings(json.load(handle))


def specs(split, settings=None):
    if split not in ("valid", "test"):
        raise ValueError("Ablations require an explicit valid/test split")
    settings = validate_settings(DEFAULT_SETTINGS if settings is None else settings)
    def spec(local, lw, gw, gamma=.5, family="J"):
        return {"family": family, "strength": 2., "local_mode": local,
                "local_weight": lw, "global_weight": gw, "angular_gamma": gamma}
    if settings["recipe"] == NO_GLOBAL_ANGLE_SETTINGS["recipe"]:
        return {"no_global_angle": spec("optimal", 1., 0., 0.)}
    result = {
        "no_local": spec("none", 0., 1.),
        "no_global": spec("optimal", 1., 0.),
        "no_angle": spec("optimal", .5, .5, 0.),
        "greedy": spec("greedy", .5, .5),
        "structure_only": spec("none", 0., 0.),
    }
    if split == "test":
        result["I_2"] = spec("optimal", .5, .5, family="I")
    return result


def validate_spec(spec):
    known = {**specs("test"), **specs("test", NO_GLOBAL_ANGLE_SETTINGS)}
    if (not isinstance(spec, Mapping) or not any(dict(spec) == item for item in known.values())
            or any(isinstance(spec.get(key), bool) for key in
                   ("strength", "local_weight", "global_weight", "angular_gamma"))):
        raise ValueError("Unknown or modified fixed ablation spec")
    return dict(spec)


def _unit_vector(value, name, shape=None):
    vector = np.asarray(value, dtype=np.float64)
    if (vector.ndim != 1 or (shape is not None and vector.shape != shape)
            or not np.isfinite(vector).all() or np.any(vector < 0) or np.any(vector > 1)):
        raise ValueError(f"Expected finite unit-interval candidate vector: {name}")
    return vector


def score_variant(values, distances, spec):
    """Score every supplied candidate; no input ranking or Top-K shortlist is used."""
    spec = validate_spec(spec)
    if not isinstance(values, Mapping) or not isinstance(distances, Mapping):
        raise ValueError("Score components and distances must be mappings")
    required = ("f_score", "optimal_f_score", "global_score", "angular_distance")
    if any(name not in values for name in required) or spec["family"] not in distances:
        raise ValueError("Missing required score component or structural distance")
    angular = _unit_vector(values["angular_distance"], "angular_distance")
    vectors = {name: _unit_vector(values[name], name, angular.shape) for name in required[:-1]}
    distance = _unit_vector(distances[spec["family"]], "structure_distance", angular.shape)
    local = vectors["optimal_f_score" if spec["local_mode"] == "optimal" else "f_score"]
    if spec["local_weight"] == 0 and spec["global_weight"] == 0:
        base = np.ones_like(angular)
    else:
        base = spec["local_weight"] * local + spec["global_weight"] * vectors["global_score"]
    return base * np.exp(-spec["angular_gamma"] * angular) * np.exp(-spec["strength"] * distance)


def full_b1_score(values, distances):
    """Read-only original J_2 anchor, preserving original arithmetic exactly."""
    return original_score_variant(values, distances, {"family": "J", "strength": 2.})
