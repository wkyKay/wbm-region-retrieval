"""One predeclared valid-only selection over the complete regional-J grid."""
from __future__ import annotations

from collections.abc import Mapping
import copy

from .scoring import frozen_grid


RULE = {
    "split": "valid", "metric": "exact/all/Precision@10", "objective": "integer_top10_hit_total",
    "baseline_eligible": True,
    "tie_break": ["baseline_first", "untrained_fusion_first", "fewer_input_branches",
                  "smaller_structure_strength_or_weight", "lexicographic_variant_name"],
    "test_policy": "one_global_winner_only; no_test_search",
}


def tie_key(name, spec):
    learned = spec["family"] in ("plain_mlp", "gated_mlp")
    branches = 4 if spec["family"] == "add_j" else (2 if not spec["area"] and not learned else 3)
    return (name != "baseline", learned, branches,
            float(spec.get("strength", spec.get("structure_weight", 0.))), name)


def select_best(results, *, split="valid", expected_queries=500):
    """Results must contain all 25 items with integer hits and query counts.

    Source hashes, complete report verification and frozen protocol identities
    belong to the caller; this function refuses partial grids and test selection.
    expected_queries is exposed only for explicitly marked synthetic unit tests.
    """
    if split != "valid":
        raise ValueError("Selection is only permitted on valid")
    if isinstance(expected_queries, bool) or not isinstance(expected_queries, int) or expected_queries < 1:
        raise ValueError("Expected a positive query count")
    grid = frozen_grid()
    if not isinstance(results, Mapping) or set(results) != set(grid):
        raise ValueError("Selection requires the complete frozen 25-candidate grid")
    checked = {}
    for name, result in results.items():
        if not isinstance(result, Mapping) or result.get("split", "valid") != "valid":
            raise ValueError("Each result must come from valid")
        hits, count = result.get("top10_hits"), result.get("num_queries")
        if (isinstance(hits, bool) or not isinstance(hits, int) or hits < 0 or hits > expected_queries*10
                or isinstance(count, bool) or not isinstance(count, int) or count != expected_queries):
            raise ValueError("Selection needs exact integer Top-10 hits for every valid query")
        checked[name] = {"top10_hits": hits, "num_queries": count}
    ordered = sorted(grid, key=lambda name: (-checked[name]["top10_hits"], tie_key(name, grid[name])))
    ranking = [{"rank": i+1, "variant": name, "spec": copy.deepcopy(grid[name]),
                **checked[name], "precision_at_10": checked[name]["top10_hits"]/(expected_queries*10)}
               for i, name in enumerate(ordered)]
    return {"schema_version": 1, "split": "valid", "winner": ordered[0],
            "selected_spec": copy.deepcopy(grid[ordered[0]]), "ranking": ranking,
            "selection_rule": copy.deepcopy(RULE), "num_candidates": len(grid),
            "num_queries": expected_queries, "synthetic": expected_queries != 500}
