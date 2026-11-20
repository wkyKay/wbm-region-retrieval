"""Freeze ten test candidates using only the complete predefined valid grid."""
from __future__ import annotations

from collections.abc import Mapping
import copy

from .scoring import frozen_grid


BASELINE_VARIANT = "mul_a0p5_b0p15"
NUM_CANDIDATES = 29
NUM_SELECTED = 10

RULE = {
    "split": "valid",
    "metric": "exact/all/Precision@10",
    "objective": "integer_top10_hit_total",
    "num_candidates": NUM_CANDIDATES,
    "num_selected": NUM_SELECTED,
    "baseline_eligible": True,
    "baseline_variant": BASELINE_VARIANT,
    "tie_break": ["baseline_first", "lexicographic_variant_name"],
    "test_policy": "evaluate_only_frozen_valid_top10_without_test_selection",
    "posthoc_design": True,
    "test_parameter_search": False,
}


def tie_key(name):
    """Prefer the current baseline only when the integer objectives are tied."""
    return name != BASELINE_VARIANT, name


def select_top10(results, *, split="valid", expected_queries=500):
    """Return an ordered top ten and the complete ranking of all 29 candidates.

    The caller must verify report hashes, source identities and query/gallery
    completeness before freezing this value. A non-500 count is available only
    for explicitly synthetic checks and is marked in the returned metadata.
    Test scores and all secondary metrics are excluded from this decision.
    """
    if split != "valid":
        raise ValueError("Top-ten selection is only permitted on valid")
    if (isinstance(expected_queries, bool) or not isinstance(expected_queries, int)
            or expected_queries < 1):
        raise ValueError("Expected a positive query count")
    grid = frozen_grid()
    if len(grid) != NUM_CANDIDATES or BASELINE_VARIANT not in grid:
        raise ValueError("Expected the predefined 29-candidate grid and current baseline")
    if not isinstance(results, Mapping) or set(results) != set(grid):
        raise ValueError("Selection requires the complete frozen 29-candidate valid grid")
    checked = {}
    for name, result in results.items():
        if not isinstance(result, Mapping) or result.get("split", "valid") != "valid":
            raise ValueError("Every candidate must have valid results")
        hits, count = result.get("top10_hits"), result.get("num_queries")
        if (isinstance(hits, bool) or not isinstance(hits, int)
                or not 0 <= hits <= expected_queries * 10
                or isinstance(count, bool) or not isinstance(count, int)
                or count != expected_queries):
            raise ValueError("Selection needs integer Top-10 hits for every valid query")
        checked[name] = {"top10_hits": hits, "num_queries": count}
    ordered = sorted(grid, key=lambda name: (-checked[name]["top10_hits"], tie_key(name)))
    selected = ordered[:NUM_SELECTED]
    ranking = [{"rank": rank + 1, "variant": name, "spec": copy.deepcopy(grid[name]),
                **checked[name], "precision_at_10": checked[name]["top10_hits"] / (expected_queries * 10)}
               for rank, name in enumerate(ordered)]
    return {
        "schema_version": 1,
        "split": "valid",
        "winner": ordered[0],
        "selected_variants": list(selected),
        "selected_specs": {name: copy.deepcopy(grid[name]) for name in selected},
        "ranking": ranking,
        "selection_rule": copy.deepcopy(RULE),
        "num_candidates": NUM_CANDIDATES,
        "num_selected": NUM_SELECTED,
        "num_queries": expected_queries,
        "synthetic": expected_queries != 500,
        "posthoc_design": True,
        "test_parameter_search": False,
    }
