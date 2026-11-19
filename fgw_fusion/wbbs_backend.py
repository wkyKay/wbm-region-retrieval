"""Exact WBBS scoring with cached 3-D search trees and conservative tie replay.

Only a search acceleration: original retained point counts, parameters, CPU
float64 penalty and first-row nearest-neighbour ties remain unchanged. No labels
or candidate preselection are used. Trees use eps=0 and workers=1.

Repeated descriptor rows may be removed for matching only. With first-index
argmin, every mutual pair must use the first occurrence of each descriptor:
all opposite rows select that occurrence. Removing other occurrences therefore
preserves the BBP count. Normalisation and the count penalty still use the full
retained point counts, never the unique counts.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

SETTINGS = {"sigma_d": 9.0, "noise_threshold": 0.32, "alpha": 5.0,
            "mu": 1.0, "total_dies": 2061, "block_size": 256}
# Normalised descriptors have coordinates in [0,1]. This absolute safety floor
# is deliberately much larger than double roundoff of 3-D distance arithmetic.
NEAR_TIE_TOL = 128 * np.finfo(np.float64).eps


def original_module():
    """Load the unchanged sibling WBBS implementation for independent auditing."""
    name = "_fgw_original_wbbs_reference"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[2] / "compared_methods/methods/wbbs/wbbs.py"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load original WBBS: " + str(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _integer_ids(values, name):
    values = np.asarray(values)
    if (values.ndim != 1 or not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 0)):
        raise ValueError("Expected nonnegative integer " + name)
    return values.astype(np.int64, copy=False)


def _first_unique(rows):
    if not len(rows):
        return np.empty((0, 3), dtype=np.float64)
    # np.unique sorts values; re-sort first indices to preserve source row ties.
    first = np.unique(rows, axis=0, return_index=True)[1]
    return np.ascontiguousarray(rows[np.sort(first)], dtype=np.float64)


class Scorer:
    """Read-only feature storage and pre-built trees, suitable for fork workers."""

    def __init__(self, ids, offsets, descriptors, settings=None):
        self.settings = dict(SETTINGS if settings is None else settings)
        if (set(self.settings) != set(SETTINGS)
                or any(isinstance(self.settings[k], (bool, np.bool_))
                       or self.settings[k] != v for k, v in SETTINGS.items())):
            raise ValueError("Only the frozen original WBBS settings are supported")
        ids = _integer_ids(ids, "sample IDs")
        if len(set(map(int, ids))) != len(ids):
            raise ValueError("Sample IDs must be unique")
        offsets = _integer_ids(offsets, "offsets")
        values = np.asarray(descriptors, dtype=np.float64)
        if (values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all()
                or np.any(values < 0) or np.any(values > 1)):
            raise ValueError("Expected finite normalised (N,3) descriptors")
        if (len(offsets) != len(ids) + 1 or offsets[0] != 0
                or offsets[-1] != len(values) or np.any(np.diff(offsets) < 0)):
            raise ValueError("Offsets do not partition all descriptor rows")
        self.ids = ids.copy()
        self.offsets = offsets.copy()
        self.descriptors = np.ascontiguousarray(values).copy()
        for array in (self.ids, self.offsets, self.descriptors):
            array.flags.writeable = False
        self.rows = {int(sample): i for i, sample in enumerate(self.ids)}
        self.retained_counts = np.diff(self.offsets)
        self.retained_counts.flags.writeable = False
        unique, trees = [], []
        for i in range(len(ids)):
            rows = _first_unique(self._original_rows(i))
            rows.flags.writeable = False
            unique.append(rows)
            trees.append(cKDTree(rows, compact_nodes=True, balanced_tree=True,
                                 copy_data=False) if len(rows) else None)
        self.unique = tuple(unique)
        self.trees = tuple(trees)
        self.unique_counts = np.asarray([len(v) for v in unique], dtype=np.int64)
        self.unique_counts.flags.writeable = False
        self.fallback_rows = 0
        self._reference_features = {}

    def _row(self, sample):
        if isinstance(sample, (bool, np.bool_)) or not isinstance(sample, (int, np.integer)):
            raise ValueError("Sample ID must be an integer")
        try:
            return self.rows[int(sample)]
        except KeyError as error:
            raise ValueError("Unknown sample ID: " + str(sample)) from error

    def _original_rows(self, row):
        return self.descriptors[self.offsets[row]:self.offsets[row + 1]]

    def _nearest(self, queries, target_row):
        """Exact first-index nearest neighbours; replay any numerically close tie."""
        target = self.unique[target_row]
        if not len(target):
            raise ValueError("Cannot query an empty tree")
        if len(target) == 1:
            return np.zeros(len(queries), dtype=np.int64)
        distance, nearest = self.trees[target_row].query(queries, k=2, eps=0.0, p=2,
                                                        workers=1)
        distance = np.asarray(distance, dtype=np.float64).reshape(-1, 2)
        nearest = np.asarray(nearest, dtype=np.int64).reshape(-1, 2)
        scale = np.maximum(1.0, distance[:, 1])
        close = distance[:, 1] - distance[:, 0] <= NEAR_TIE_TOL * scale
        result = nearest[:, 0].copy()
        ambiguous = np.flatnonzero(close)
        self.fallback_rows += len(ambiguous)
        # Use the original scipy.cdist square-distance arithmetic and full row.
        # Bounded chunks avoid a dense wafer-by-wafer matrix allocation.
        for start in range(0, len(ambiguous), int(self.settings["block_size"])):
            selected = ambiguous[start:start + int(self.settings["block_size"])]
            exact = cdist(queries[selected], target, metric="sqeuclidean")
            result[selected] = np.argmin(exact, axis=1)
        return result

    def _count(self, qrow, crow):
        q, c = self.unique[qrow], self.unique[crow]
        if not len(q) or not len(c):
            return 0
        # BBP membership is symmetric. Start from the smaller unique set;
        # a target point never selected by this forward map cannot be mutual.
        # Reverse-query only selected target points, against the full first set.
        if len(q) > len(c):
            qrow, crow = crow, qrow
            q, c = c, q
        forward = self._nearest(q, crow)
        selected, inverse = np.unique(forward, return_inverse=True)
        backward = self._nearest(c[selected], qrow)
        return int(np.count_nonzero(backward[inverse] == np.arange(len(q))))

    def score_pair_ids(self, query_id, candidate_id):
        qrow, crow = self._row(query_id), self._row(candidate_id)
        n, m = int(self.retained_counts[qrow]), int(self.retained_counts[crow])
        if min(n, m) == 0:
            return 0.0
        count = self._count(qrow, crow)
        return float(count / min(n, m) * np.exp(-float(self.settings["alpha"])
                     * abs(n - m) / int(self.settings["total_dies"])))

    def scores(self, query_id, candidate_ids):
        self._row(query_id)
        candidates = _integer_ids(candidate_ids, "candidate IDs")
        if len(set(map(int, candidates))) != len(candidates) or int(query_id) in candidates:
            raise ValueError("Candidate IDs must be unique and exclude the query")
        return np.asarray([self.score_pair_ids(query_id, int(c)) for c in candidates],
                          dtype=np.float64)

    def audit_pairs(self, pairs, *, include_rows=False):
        """Require bit-for-bit scalar equality with the original CPU WBBS score."""
        if not isinstance(include_rows, bool):
            raise ValueError("include_rows must be boolean")
        original = original_module()
        checked, details = 0, []
        for query, candidate in pairs:
            qrow, crow = self._row(query), self._row(candidate)
            for row in (qrow, crow):
                if row not in self._reference_features:
                    values = self._original_rows(row)
                    points = np.column_stack((np.arange(len(values)), np.zeros(len(values))))
                    self._reference_features[row] = original.WaferFeatures(points, values, len(values))
            expected = original.score_pair(self._reference_features[qrow],
                                           self._reference_features[crow], self.settings)
            actual = self.score_pair_ids(query, candidate)
            if np.float64(actual).tobytes() != np.float64(expected).tobytes():
                raise AssertionError(f"WBBS exact audit mismatch: {query}, {candidate}, {actual}, {expected}")
            checked += 1
            if include_rows:
                details.append({"query_id": int(query), "candidate_id": int(candidate),
                                "score": actual, "reference_score": expected})
        result = {"status": "verified", "pairs_checked": checked,
                  "num_pairs": checked, "max_error": 0.0,
                  "scalar_equality": "float64 bit-for-bit", "max_abs_error": 0.0,
                  "fallback_rows_cumulative": int(self.fallback_rows)}
        if include_rows:
            result["pairs"] = details
        return result
