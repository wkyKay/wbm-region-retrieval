"""Finite additive/product grids with a mandatory optimal-local F branch.

Each call scores a query's *complete* frozen candidate pool. In particular,
sum_minmax must never be applied to a saved Top-K or a candidate chunk.
"""
from collections.abc import Mapping
from copy import deepcopy
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np


SETTINGS = {
    "schema_version": 1,
    "recipe": "fgw_grid_search_v1",
    "denominator": 10,
    "min_f_ticks": 1,
    "weight_ticks": None,
    "families": ["sum_raw", "sum_minmax", "weighted_product"],
    "include_reference": True,
    "ks": [1, 5, 10, 20, 50, 100, 500],
    "tie_seed": 2026,
    "objective": "valid/exact/all/Precision@10",
    "selection_split": "valid",
    "tie_rule": ["active_branches_asc", "family_priority_asc", "F_weight_desc",
                 "WBBS_weight_asc", "variant_id_asc"],
    "family_priority": ["sum_raw", "weighted_product", "sum_minmax", "legacy_fgw"],
    "training_required": False,
    "posthoc_research": True,
    "test_parameter_search": False,
}
_CUSTOMIZABLE = {"denominator", "min_f_ticks", "weight_ticks", "families", "include_reference"}
_FAMILIES = ("sum_raw", "sum_minmax", "weighted_product")


def _same_type_and_value(value, expected):
    if type(value) is not type(expected):
        return False
    if isinstance(expected, list):
        return (len(value) == len(expected)
                and all(_same_type_and_value(v, e) for v, e in zip(value, expected)))
    return value == expected


def validate_settings(settings):
    """Validate known fields, fill defaults, and return an independent copy.

    Only the grid-definition fields are configurable. Evaluation and
    winner-selection rules remain frozen even when a custom grid is requested.
    """
    if not isinstance(settings, Mapping) or set(settings) - set(SETTINGS):
        raise ValueError("Unknown grid-search settings fields")
    result = deepcopy(SETTINGS)
    result.update(deepcopy(dict(settings)))
    for key in set(SETTINGS) - _CUSTOMIZABLE:
        if not _same_type_and_value(result[key], SETTINGS[key]):
            raise ValueError("Frozen grid-search setting changed: " + key)
    denominator = result["denominator"]
    if type(denominator) is not int or not 1 <= denominator <= 20:
        raise ValueError("denominator must be an integer in [1, 20]")
    minimum = result["min_f_ticks"]
    if type(minimum) is not int or not 1 <= minimum <= denominator:
        raise ValueError("min_f_ticks must retain F: integer in [1, denominator]")
    ticks = result["weight_ticks"]
    if ticks is not None:
        if not isinstance(ticks, list) or not ticks:
            raise ValueError("weight_ticks must be null or a nonempty list of tick triples")
        if any(not isinstance(row, list) or len(row) != 3
               or any(type(value) is not int or value < 0 for value in row)
               or row[0] < minimum or sum(row) != denominator for row in ticks):
            raise ValueError("Each weight_ticks triple must retain F and sum to denominator")
        if len({tuple(row) for row in ticks}) != len(ticks):
            raise ValueError("weight_ticks must not contain duplicate triples")
        # Equivalent user orderings must also have identical settings signatures.
        result["weight_ticks"] = sorted(ticks)
    families = result["families"]
    if (not isinstance(families, list) or not families
            or any(type(family) is not str or family not in _FAMILIES for family in families)
            or len(set(families)) != len(families)):
        raise ValueError("families must be a nonempty unique list of supported grid families")
    if type(result["include_reference"]) is not bool:
        raise ValueError("include_reference must be a JSON boolean")
    return result


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON setting: " + key)
        result[key] = value
    return result


def load_settings(path=None):
    path = Path(path) if path is not None else Path(__file__).with_name("config.json")
    with path.open(encoding="utf-8") as handle:
        return validate_settings(json.load(handle, object_pairs_hook=_unique_json_object))


def specs(settings=None):
    """Return deterministic IDs and exact tick-derived specifications."""
    settings = validate_settings(SETTINGS if settings is None else settings)
    denominator = settings["denominator"]
    ticks_grid = settings["weight_ticks"]
    if ticks_grid is None:
        ticks_grid = [[f, g, denominator - f - g]
                      for f in range(settings["min_f_ticks"], denominator + 1)
                      for g in range(denominator - f + 1)]
    result = {}
    for family in settings["families"]:
        for f_ticks, g_ticks, w_ticks in ticks_grid:
            name = (f"{family}_d{denominator:02d}_f{f_ticks:02d}"
                    f"_g{g_ticks:02d}_w{w_ticks:02d}")
            ticks = [f_ticks, g_ticks, w_ticks]
            result[name] = {"family": family,
                            "weights": [value / denominator for value in ticks],
                            "ticks": ticks, "denominator": denominator}
    if settings["include_reference"]:
        result["legacy_fgw"] = {"family": "legacy_fgw", "weights": [0.5, 0.5, 1.0]}
    return result


def validate_spec(spec):
    if not isinstance(spec, Mapping):
        raise ValueError("A fusion specification must be a mapping")
    family = spec.get("family")
    if type(family) is not str or family not in (*_FAMILIES, "legacy_fgw"):
        raise ValueError("Unsupported fusion family")
    weights = spec.get("weights")
    if (not isinstance(weights, list) or len(weights) != 3
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
                   or not math.isfinite(v) or not 0 <= v <= 1 for v in weights)):
        raise ValueError("weights must contain three finite real values in [0, 1]")
    if weights[0] <= 0:
        raise ValueError("F must have a strictly positive coefficient/exponent")
    if family == "legacy_fgw":
        if set(spec) != {"family", "weights"} or weights != [0.5, 0.5, 1.0]:
            raise ValueError("The legacy FGW reference formula is frozen")
    else:
        if set(spec) != {"family", "weights", "ticks", "denominator"}:
            raise ValueError("Unknown or missing fusion specification fields")
        denominator, ticks = spec["denominator"], spec["ticks"]
        if type(denominator) is not int or not 1 <= denominator <= 20:
            raise ValueError("Invalid specification denominator")
        if (not isinstance(ticks, list) or len(ticks) != 3
                or any(type(v) is not int or v < 0 for v in ticks)
                or ticks[0] < 1 or sum(ticks) != denominator):
            raise ValueError("Invalid fusion ticks or deleted F branch")
        if weights != [v / denominator for v in ticks]:
            raise ValueError("Weights differ from their exact integer-tick specification")
    result = deepcopy(dict(spec))
    result["weights"] = [float(v) for v in weights]
    return result


def unit_vector(value, name, shape=None):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError("Expected real score values: " + name)
    values = np.asarray(raw, dtype=np.float64)
    if (values.ndim != 1 or values.size == 0
            or (shape is not None and values.shape != shape)
            or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1)):
        raise ValueError("Expected nonempty aligned finite unit-interval vector: " + name)
    return values


def score_variant(components, spec):
    """Score full candidate vectors without mutating components.

    All three families preserve the pure-F endpoint bit-for-bit. For other
    min-max variants, constant branches contribute zero. Product branches with
    exponent zero are skipped; active zeros remain zero, with no epsilon floor.
    """
    spec = validate_spec(spec)
    if not isinstance(components, Mapping) or not {"F", "G", "WBBS"} <= set(components):
        raise ValueError("Missing F/G/WBBS score components")
    local = unit_vector(components["F"], "F")
    values = [local, unit_vector(components["G"], "G", local.shape),
              unit_vector(components["WBBS"], "WBBS", local.shape)]
    weights = spec["weights"]
    if weights == [1.0, 0.0, 0.0]:
        return local.copy()
    if spec["family"] == "legacy_fgw":
        return (0.5 * values[0] + 0.5 * values[1]) * values[2]
    if spec["family"] == "weighted_product":
        result = np.ones(local.shape, dtype=np.float64)
        for weight, vector in zip(weights, values):
            if weight > 0:
                result *= np.power(vector, weight)
        return result
    result = np.zeros(local.shape, dtype=np.float64)
    for weight, vector in zip(weights, values):
        if weight == 0:
            continue
        if spec["family"] == "sum_minmax":
            low, high = float(np.min(vector)), float(np.max(vector))
            if high == low:
                continue
            vector = (vector - low) / (high - low)
        result += weight * vector
    return result
