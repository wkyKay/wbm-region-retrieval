"""Choose a grid winner using only validation exact Top-10 hit counts."""
from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Real

from . import scoring


_ROW_FIELDS = {"variant", "split", "num_queries", "precision10_hits",
               "Precision@10", "mAP@100", "Coverage-nDCG@10"}


def choose_winner(rows, specs, settings):
    """Return a deterministic, JSON-signable record, never ranking by test.

    Every configured variant must appear once. Secondary metrics are retained
    as descriptive evidence and never enter either overall or family selection.
    """
    settings = scoring.validate_settings(settings)
    expected = scoring.specs(settings)
    if not isinstance(specs, Mapping) or set(specs) != set(expected):
        raise ValueError("Winner selection requires the complete configured variant grid")
    for name, spec in specs.items():
        if scoring.validate_spec(spec) != expected[name]:
            raise ValueError("Variant definition differs from the frozen grid: " + name)
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("Winner selection requires one valid row for every variant")
    normalized = []
    seen = set()
    num_queries = None
    for row in rows:
        if (not isinstance(row, Mapping) or not _ROW_FIELDS <= set(row)
                or set(row) - _ROW_FIELDS - {"family"}):
            raise ValueError("Unknown or missing winner-selection row fields")
        if row["split"] != "valid":
            raise ValueError("Only valid may select a winner; test rows are forbidden")
        name = row["variant"]
        if not isinstance(name, str) or name not in expected or name in seen:
            raise ValueError("Unknown or duplicate variant in selection rows")
        if "family" in row and row["family"] != expected[name]["family"]:
            raise ValueError("Selection-row family disagrees with variant specification")
        seen.add(name)
        n = row["num_queries"]
        hits = row["precision10_hits"]
        if type(n) is not int or n <= 0 or (num_queries is not None and n != num_queries):
            raise ValueError("All validation rows must have the same positive query count")
        num_queries = n
        if type(hits) is not int or not 0 <= hits <= 10 * n:
            raise ValueError("precision10_hits must be an integer in [0, 10*num_queries]")
        for key in ("Precision@10", "mAP@100", "Coverage-nDCG@10"):
            value = row[key]
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("Expected a finite proportion for " + key)
        if not math.isclose(row["Precision@10"], hits / (10 * n), rel_tol=0, abs_tol=1e-12):
            raise ValueError("Precision@10 disagrees with its exact total hit count")
        record = deepcopy(dict(row))
        record.update({"family": expected[name]["family"],
                       "weights": list(expected[name]["weights"]),
                       "active_branches": sum(w > 0 for w in expected[name]["weights"])})
        for key in ("Precision@10", "mAP@100", "Coverage-nDCG@10"):
            record[key] = float(record[key])
        normalized.append(record)
    family_priority = settings["family_priority"]

    def order(row):
        return (-row["precision10_hits"], row["active_branches"],
                family_priority.index(row["family"]), -row["weights"][0],
                row["weights"][2], row["variant"])

    ranking = sorted(normalized, key=order)
    family_winners = {}
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank
        family_winners.setdefault(row["family"], row["variant"])
    return {
        "schema_version": 1,
        "recipe": settings["recipe"],
        "selection_split": "valid",
        "objective": settings["objective"],
        "objective_statistic": "precision10_hits",
        "tie_rule": list(settings["tie_rule"]),
        "family_priority": list(family_priority),
        "num_queries": num_queries,
        "num_variants": len(ranking),
        "winner": ranking[0]["variant"],
        "family_winners": family_winners,
        "ranking": ranking,
        "settings": deepcopy(settings),
        "specifications": deepcopy(dict(expected)),
        "secondary_metrics_used_for_selection": False,
        "test_used_for_selection": False,
    }
