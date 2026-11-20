"""Complete-grid, valid-only selection for the frozen 15 scoring variants."""
from __future__ import annotations

from collections.abc import Mapping
import copy

from .scoring import frozen_grid


RULE = {
    "split": "valid",
    "metric": "exact/all/Precision@10",
    "objective": "integer_top10_hit_total",
    "baseline_eligible": True,
    "baseline_variant": "product",
    "tie_break": ["product_first", "smaller_eta", "alpha_closer_to_0.5",
                  "ungated_first", "lexicographic_variant_name"],
    "endpoint_tie_values": {"eta": 2.0, "alpha_distance_from_0.5": 1.0},
    "test_policy": "valid_only_no_test",
}


def tie_key(name, spec):
    """Endpoint controls sort after mixed variants on equal hits, then by name."""
    eta = float(spec.get("eta", 2.0))
    alpha_distance = abs(float(spec["alpha"]) - .5) if "alpha" in spec else 1.0
    return (name != "product", eta, alpha_distance, bool(spec.get("gate_zero", False)), name)


def select_best(results, *, split="valid", expected_queries=500):
    """Select once from integer hit totals for all frozen valid candidates.

    The caller verifies report/source hashes and fixed query/gallery identities.
    A non-500 expected count is only for explicitly marked synthetic tests.
    """
    if split != "valid":
        raise ValueError("Selection is only permitted on valid")
    if (isinstance(expected_queries, bool) or not isinstance(expected_queries, int)
            or expected_queries < 1):
        raise ValueError("Expected a positive query count")
    grid = frozen_grid()
    if not isinstance(results, Mapping) or set(results) != set(grid):
        raise ValueError("Selection requires the complete frozen 15-candidate grid")
    checked = {}
    for name, result in results.items():
        if not isinstance(result, Mapping) or result.get("split", "valid") != "valid":
            raise ValueError("Every candidate must have valid results")
        hits, count = result.get("top10_hits"), result.get("num_queries")
        if (isinstance(hits, bool) or not isinstance(hits, int)
                or hits < 0 or hits > expected_queries * 10
                or isinstance(count, bool) or not isinstance(count, int)
                or count != expected_queries):
            raise ValueError("Selection needs integer Top-10 hits for every valid query")
        checked[name] = {"top10_hits": hits, "num_queries": count}
    ordered = sorted(grid, key=lambda name: (-checked[name]["top10_hits"], tie_key(name, grid[name])))
    ranking = [{"rank": rank + 1, "variant": name, "spec": copy.deepcopy(grid[name]),
                **checked[name], "precision_at_10": checked[name]["top10_hits"] / (expected_queries * 10)}
               for rank, name in enumerate(ordered)]
    return {"schema_version": 1, "split": "valid", "winner": ordered[0],
            "selected_spec": copy.deepcopy(grid[ordered[0]]), "ranking": ranking,
            "selection_rule": copy.deepcopy(RULE), "num_candidates": len(grid),
            "num_queries": expected_queries, "synthetic": expected_queries != 500}
