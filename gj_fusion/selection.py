"""Freeze the whole-map G/J choice using the complete eight-role valid grid."""
from __future__ import annotations

from collections.abc import Mapping
import copy
from numbers import Integral

from .scoring import frozen_grid


NUM_CANDIDATES = 8
NUM_QUERIES = 500
K = 10
RULE = {
    "split": "valid",
    "metric": "exact/all/Precision@10",
    "objective": "integer_top10_hit_total",
    "num_candidates": NUM_CANDIDATES,
    "num_queries": NUM_QUERIES,
    "k": K,
    "tie_break": ["product_first", "J_only_second", "G_only_third",
                  "lexicographic_variant_name"],
    "family_endpoints": "both_families_share_J_only_and_G_only",
    "geometric_half_role": "product_G_times_J_same_mathematical_ranking",
    "test_policy": "deduplicated_global_and_family_winners_plus_fixed_G_J_product_references",
    "posthoc_design": True,
    "test_parameter_search": False,
}


def role_aliases():
    """Document shared endpoints and the geometric-half ranking alias.

    The product alias is mathematical ranking equivalence, not equal scores
    or an assertion of bit-identical sqrt rankings for every numeric backend.
    """
    return {
        "additive_alpha_0": {"variant": "J_only", "equivalence": "equal_scores"},
        "geometric_alpha_0": {"variant": "J_only", "equivalence": "equal_scores"},
        "additive_alpha_1": {"variant": "G_only", "equivalence": "equal_scores"},
        "geometric_alpha_1": {"variant": "G_only", "equivalence": "equal_scores"},
        "geometric_alpha_0p5": {
            "variant": "product", "equivalence": "same_mathematical_ranking",
            "alias_score": "sqrt(G*J)", "stored_score": "G*J",
        },
    }


def tie_key(name):
    return {"product": 0, "J_only": 1, "G_only": 2}.get(name, 3), name


def select_candidates(results, *, split="valid"):
    """Select one global winner and one winner per fusion family.

    Each result must explicitly identify valid, contain integer top10_hits,
    and cover all 500 fixed queries. Report/protocol hashes and per-query
    evidence must additionally be verified by the calling pipeline. No test
    values, secondary metrics, partial grids or synthetic query counts enter
    this choice.
    """
    if split != "valid":
        raise ValueError("G/J selection is only permitted on valid")
    grid = frozen_grid()
    if len(grid) != NUM_CANDIDATES or not isinstance(results, Mapping) or set(results) != set(grid):
        raise ValueError("Selection requires the complete frozen eight-role valid grid")
    checked = {}
    for name, result in results.items():
        if not isinstance(result, Mapping) or result.get("split") != "valid":
            raise ValueError("Every candidate must explicitly contain valid results")
        hits, count = result.get("top10_hits"), result.get("num_queries")
        if (isinstance(hits, bool) or not isinstance(hits, Integral)
                or not 0 <= hits <= NUM_QUERIES * K
                or isinstance(count, bool) or not isinstance(count, Integral)
                or count != NUM_QUERIES):
            raise ValueError("Selection requires integer Top-10 hits for all 500 valid queries")
        if "k" in result and (isinstance(result["k"], bool)
                              or not isinstance(result["k"], Integral) or result["k"] != K):
            raise ValueError("Selection only accepts exact Top-10 hits")
        checked[name] = {"top10_hits": int(hits), "num_queries": int(count)}
    ordered = sorted(grid, key=lambda name: (-checked[name]["top10_hits"], tie_key(name)))
    family_candidates = {
        "additive": [name for name in ordered if grid[name]["family"] in ("endpoint", "additive")],
        "geometric": [name for name in ordered if grid[name]["family"] in ("endpoint", "geometric", "product")],
    }
    family_winners = {family: names[0] for family, names in family_candidates.items()}
    roles = {
        "overall_winner": ordered[0],
        "additive_winner": family_winners["additive"],
        "geometric_winner": family_winners["geometric"],
        "product_reference": "product",
        "structure_reference": "J_only",
        "global_reference": "G_only",
    }
    selected = list(dict.fromkeys(roles.values()))
    variant_roles = {name: [role for role, variant in roles.items() if variant == name]
                     for name in grid}
    ranking = [{"rank": rank + 1, "variant": name, "spec": copy.deepcopy(grid[name]),
                **checked[name], "precision_at_10": checked[name]["top10_hits"] / (NUM_QUERIES * K)}
               for rank, name in enumerate(ordered)]
    return {
        "schema_version": 1, "split": "valid", "winner": ordered[0],
        "family_winners": family_winners, "family_candidates": family_candidates,
        "selected_variants": selected,
        "selected_specs": {name: copy.deepcopy(grid[name]) for name in selected},
        "ranking": ranking, "roles": roles, "variant_roles": variant_roles,
        "aliases": role_aliases(), "selection_rule": copy.deepcopy(RULE),
        "num_candidates": NUM_CANDIDATES, "num_selected": len(selected),
        "num_queries": NUM_QUERIES, "k": K,
        "posthoc_design": True, "test_parameter_search": False,
    }


# A readable alternate entry point for callers that focus on the three winners.
select_winners = select_candidates
