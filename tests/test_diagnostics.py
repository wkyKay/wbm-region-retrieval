"""Check diagnostic metric comparability and frozen-pool isolation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from wbm_retrieval.common import csv_rows
from wbm_retrieval.diagnostics import (batched_map_scores, freeze_protocol, guard_output, normalized,
                                       pair_auc, retrieve_variant, source_snapshot)
from wbm_retrieval.evaluation import evaluate_ranking
from wbm_retrieval.matching import group_tokens, map_similarity


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.settings = {"lambda_pos": 0., "lambda_scale": 0., "sigma_pos": .35, "sigma_scale": .5,
                         "shape_threshold": 0., "score_threshold": 0., "weight": "sqrt_area"}

    def test_batched_scores_match_official_greedy_including_ties_empty_and_thresholds(self):
        rng = np.random.default_rng(12)
        for raw in (rng.normal(size=(13, 8)), np.ones((13, 8)), np.zeros((13, 8))):
            features = normalized(raw)
            offsets = np.array([0, 3, 3, 5, 11, 13])
            metadata = {"map_ids": np.arange(5), "offsets": offsets, "area": rng.uniform(1, 8, 13),
                        "position": rng.random((13, 2)), "scale": rng.uniform(.1, 1, 13)}
            grouped = group_tokens(metadata, raw)
            padded = np.full((5, 6), -1, dtype=int)
            for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
                padded[i, :end-start] = np.arange(start, end)
            for weight in ("uniform", "sqrt_area", "area"):
                for threshold in (0, .5, 1):
                    settings = {**self.settings, "weight": weight, "shape_threshold": threshold, "score_threshold": .2}
                    for q in (0, 1, 3):
                        h = features[offsets[q]:offsets[q+1]]
                        actual = batched_map_scores(h @ features.T, padded, grouped[q]["area"], settings)
                        expected = [map_similarity(grouped[q], grouped[c], settings) for c in range(5)]
                        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_protocol_reused_excludes_self_and_never_leaves_valid(self):
        metadata = {"map_ids": np.array([2, 7, 11, 19]), "offsets": np.array([0, 1, 3, 4, 6])}
        labels = {2: np.array([1, 0]), 7: np.array([1, 1]), 11: np.array([0, 1]), 19: np.array([1, 1])}
        owners = np.repeat(metadata["map_ids"], np.diff(metadata["offsets"]))
        settings = {"seed": 42, "queries": 0, "candidates": 1000, "pair_samples": 20, "neighbor_anchors": 4}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol, manifest, positive, pairs, anchors = freeze_protocol(root, metadata, labels, owners, settings)
            initial = source_snapshot(root)
            repeated = freeze_protocol(root, metadata, labels, owners, settings)
            self.assertEqual(initial, source_snapshot(root))
            self.assertEqual(protocol.candidates, repeated[0].candidates)
            self.assertEqual(manifest["candidates_per_query"], 3)
            for q, values in protocol.candidates.items():
                self.assertEqual(set(values), set(labels) - {q})
            self.assertTrue(all(owners[int(row["query_token"])] != owners[int(row["candidate_token"])] for row in pairs))
            self.assertTrue(all(int(row["rotation_k"]) in (1, 2, 3) for row in positive))
            self.assertEqual(len(anchors), 4)
            path = root / "protocol/queries.csv"
            path.write_text("sample_id\n999\n")
            with self.assertRaises(ValueError):
                freeze_protocol(root, metadata, labels, owners, settings)

    def test_retrieval_metrics_equal_official_evaluator(self):
        rng = np.random.default_rng(31)
        metadata = {"map_ids": np.arange(5), "offsets": np.array([0, 1, 3, 3, 4, 6]), "area": np.ones(6)}
        labels = {i: np.array(value) for i, value in enumerate(([1, 0], [1, 0], [0, 1], [1, 1], [1, 1]))}
        owners = np.repeat(metadata["map_ids"], np.diff(metadata["offsets"]))
        settings = {"seed": 42, "queries": 0, "candidates": 0, "pair_samples": 20, "neighbor_anchors": 3, "bootstrap": 0}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol, *_ = freeze_protocol(root, metadata, labels, owners, settings)
            target = root / "variant"
            _, groups = retrieve_variant(normalized(rng.normal(size=(6, 8))), metadata, protocol, target,
                                          {"matching": self.settings, "evaluation": {"ks": [1, 5, 10, 20]}}, settings)
            reference = evaluate_ranking(target / "rankings.csv", protocol, {"ks": [1, 5, 10, 20], "bootstrap_samples": 0}, 42)
            self.assertEqual(groups[0]["metrics"], reference["metrics"])
            self.assertEqual(len(csv_rows(target / "rankings.csv")), 20)

    def test_output_cannot_overlap_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "experiment"
            for output in (source, source / "diagnostics", source.parent):
                with self.assertRaises(ValueError):
                    guard_output(source, output)
            guard_output(source, source.parent / "separate_diagnostics")

    def test_auc_ties_and_order(self):
        self.assertEqual(pair_auc([1, 1], [1, 1]), .5)
        self.assertEqual(pair_auc([2, 3], [0, 1]), 1.)
        self.assertEqual(pair_auc([-2, -1], [0, 1]), 0.)


if __name__ == "__main__":
    unittest.main()
