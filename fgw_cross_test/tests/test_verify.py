"""Synthetic numerical and provenance checks for the independent cross verifier."""
from collections.abc import Mapping
from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.proposed.f_segmentation_geometry import coverage_metrics
from fgw_grid_search import scoring
from fgw_cross_test import verify
from wbm_retrieval.exact_metrics import order_scores


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def spec(family="sum_minmax", ticks=(1, 9, 0)):
    return {"family": family, "weights": [n / sum(ticks) for n in ticks],
            "ticks": list(ticks), "denominator": sum(ticks)}


class BasePool(Mapping):
    def __init__(self, labels):
        self.labels = labels
        self.gallery = np.arange(len(labels), dtype=np.int64)
        self.queries = self.gallery.copy()
        self.split_name = "test"
        self.protocol_signature = "base_protocol"

    def __iter__(self):
        return iter(map(int, self.queries))

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, query):
        return self.gallery[self.gallery != query]


class Pool(BasePool):
    """Use the existing cross protocol so its serialization is exercised too."""
    def __init__(self, labels):
        base = BasePool(labels)
        path = Path(__file__).resolve().parents[3] / "compared_methods/shared/cross_combination.py"
        module_spec = importlib.util.spec_from_file_location("test_existing_cross_protocol", path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        self.wrapped = module.CrossCombinationPool(base, labels)
        self.labels = labels
        for name in ("gallery", "queries", "split_name", "metadata", "protocol_signature", "candidates_per_query"):
            setattr(self, name, getattr(self.wrapped, name))

    def __getitem__(self, query):
        return self.wrapped[query]


class Provider:
    def __init__(self, folder, pool):
        self.marker = Path(folder) / "source_completed.json"
        write_json(self.marker, {"source": "frozen synthetic full score cache"})
        self.signature = "source_signature"
        self.protocol_signature = "base_protocol"
        self.split = "test"
        self.query_ids, self.gallery_ids = pool.queries, pool.gallery
        q, c = np.meshgrid(self.query_ids, self.gallery_ids, indexing="ij")
        self.values = {name: ((q * 13 + c * (17 + i * 4) + i * 11) % 87 + 6) / 100.
                       for i, name in enumerate(("F", "G", "WBBS"))}
        # Removing exact candidate 1 changes F/G normalization for query 0.
        self.values["F"][0, 1] = 1.
        self.values["G"][0, 1] = 0.

    def scores(self, query, candidates):
        return {name: values[int(query), candidates].copy() for name, values in self.values.items()}


def fixture(root, *, wrong_minmax=False, tie_scores=False):
    root = Path(root)
    labels = np.array([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0],
                       [1, 0, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 0, 0],
                       [0, 0, 0, 1, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0],
                       [1, 1, 0, 0, 0], [0, 0, 0, 0, 1], [1, 1, 0, 0, 0],
                       [0, 0, 1, 1, 0]], dtype=int)
    taxonomy = {key: i for i, key in enumerate(sorted(set(map(tuple, labels))))}
    bundle = SimpleNamespace(labels=labels, class_ids=np.array([taxonomy[tuple(row)] for row in labels]))
    pool, ks = Pool(labels), [1, 3, 5]
    provider = Provider(root, pool)
    if tie_scores:
        for values in provider.values.values():
            values[:] = .4
    names = [f"sum_minmax_d10_f{i:02d}_g{10-i:02d}_w00" for i in range(1, 11)]
    specs = {name: spec(ticks=(i, 10-i, 0)) for i, name in enumerate(names, 1)}
    frozen = {"top_k": 10, "ranked_variants": names, "specs": specs,
              "primary_variant": names[0], "selection_split": "valid", "test_parameter_search": False,
              "valid_selection": {"ranking": [{"variant": name} for name in names]}}
    frozen["signature"] = verify._digest(frozen)
    frozen_path = root / "frozen_selection.json"
    write_json(frozen_path, frozen)
    folder = root / "evaluation"
    folder.mkdir()
    identity = {"run_id": "20261119T010203+0800__fixture", "started_at": "2026-11-19T01:02:03+08:00",
                "evaluation_signature": "evaluation_signature", "variant": names[0], "split": "test"}
    manifest = {**identity, "signature": "evaluation_signature", "spec": specs[names[0]],
                "ks": ks, "tie_seed": 2026, "num_queries": len(pool), "formal_protocol": False,
                "provider_source": {"signature": provider.signature, "completed_sha256": verify._sha(provider.marker)},
                "candidate_rule": verify.RULE, "derived_protocol": pool.metadata,
                "retrieval_protocol_signature": pool.protocol_signature,
                "base_protocol_signature": provider.protocol_signature,
                "frozen_selection": {"path": str(frozen_path), "sha256": verify._sha(frozen_path)}}
    write_json(folder / "run_identity.json", identity)
    write_json(folder / "evaluation_manifest.json", manifest)
    fields = ["query_id", "rank", "candidate_id", "similarity_score", "optimal_f_score", "global_score", "wbbs_score"]
    with (folder / "rankings.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for query in pool:
            candidates = pool[query]
            values = provider.scores(query, candidates)
            if wrong_minmax:
                full = provider.gallery_ids[provider.gallery_ids != query]
                full_scores = scoring.score_variant(provider.scores(query, full), manifest["spec"])
                scores = full_scores[np.isin(full, candidates)]
            else:
                scores = scoring.score_variant(values, manifest["spec"])
            order = order_scores(candidates, scores, query, 2026)[:max(ks)]
            for rank, i in enumerate(order, 1):
                writer.writerow((query, rank, int(candidates[i]), float(scores[i]),
                                 float(values["F"][i]), float(values["G"][i]), float(values["WBBS"][i])))
    coverage_metrics.evaluate_rankings_file(bundle, pool, folder / "rankings.csv", ks, folder,
        split_name="test", metadata={"signature": "evaluation_signature",
                                     "retrieval_protocol_signature": pool.protocol_signature})
    return folder, pool, bundle, ks, provider


def check(values):
    folder, pool, bundle, ks, provider = values
    return verify.verify_evaluation(folder, pool, bundle, ks, "evaluation_signature", provider)


def rewrite_csv(path, transform):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    transform(rows)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class CrossVerificationTests(unittest.TestCase):
    def test_original_linear_coverage_and_full_pool_denominators(self):
        query = np.array([1, 1, 1, 0])
        candidates = np.array([[1, 1, 1, 1], [1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 1, 0], [1, 1, 1, 1]])
        metrics, relevant = verify.independent_coverage(query, candidates, candidates[:4], [1, 2, 4, 5])
        self.assertEqual(relevant, 2)
        self.assertEqual(metrics["FullCoverageRecall@1"], .5)
        self.assertEqual(metrics["FullCoverageAP@1"], .5)
        self.assertAlmostEqual(metrics["MeanCoverage@4"], 2 / 3)
        self.assertEqual(metrics["FullCoveragePrecision@5"], .2)
        same = candidates.copy()
        same[[1, 2]] = same[[2, 1]]
        self.assertEqual(verify.independent_coverage(query, candidates, same[:4], [4])[0]["Coverage-nDCG@4"], metrics["Coverage-nDCG@4"])

    def test_zero_idcg_and_no_full_coverage_are_distinct(self):
        q = np.array([1, 1, 0])
        partial = np.array([[1, 0, 0], [0, 1, 0]])
        metrics, count = verify.independent_coverage(q, partial, partial, [1, 2])
        self.assertEqual(count, 0)
        self.assertEqual(metrics["Coverage-nDCG@1"], 1.)
        self.assertIsNone(metrics["FullCoverageRecall@1"])
        zero = np.array([[0, 0, 1]])
        metrics, _ = verify.independent_coverage(q, zero, zero, [1])
        self.assertIsNone(metrics["Coverage-nDCG@1"])
        self.assertEqual(metrics["MeanCoverage@1"], 0.)
        self.assertEqual(metrics["FullCoveragePrecision@1"], 0.)

    def test_all_groups_verified_without_production_scorer_or_evaluator(self):
        with tempfile.TemporaryDirectory() as root:
            values = fixture(root)
            with patch.object(scoring, "score_variant", side_effect=AssertionError("production scorer")), \
                 patch.object(coverage_metrics, "query_metrics", side_effect=AssertionError("production metrics")):
                result = check(values)
            self.assertEqual(result["status"], "verified")
            self.assertEqual(result["num_queries"], 13)
            self.assertEqual(result["ranking_rows"], 65)
            self.assertEqual(result["metric_query_counts"]["Coverage-nDCG@1"], 12)
            self.assertGreater(result["aggregation_values_checked"], 100)
            self.assertLess(result["max_metric_absolute_error"], 1e-12)

    def test_normalization_before_exclusion_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            values = fixture(root, wrong_minmax=True)
            with self.assertRaisesRegex(ValueError, "full-candidate ordering|score formula"):
                check(values)

    def test_tied_scores_use_frozen_hash_order_and_reject_swaps(self):
        with tempfile.TemporaryDirectory() as root:
            values = fixture(root, tie_scores=True)
            self.assertEqual(check(values)["status"], "verified")
            def swap(rows):
                for field in ("candidate_id", "similarity_score", "optimal_f_score", "global_score", "wbbs_score"):
                    rows[0][field], rows[1][field] = rows[1][field], rows[0][field]
            rewrite_csv(values[0] / "rankings.csv", swap)
            with self.assertRaisesRegex(ValueError, "full-candidate ordering"):
                check(values)

    def test_corrupt_metric_groups_effective_counts_and_exact_report_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            values = fixture(root)
            path = values[0] / "coverage_metrics.json"
            original = verify._json(path)
            damaged = deepcopy(original)
            damaged["groups"]["mixed_label"]["metrics"]["MeanCoverage@3"] += .01
            write_json(path, damaged)
            with self.assertRaisesRegex(ValueError, "Metric mismatch"):
                check(values)
            damaged = deepcopy(original)
            damaged["metric_query_counts"]["Coverage-nDCG@1"] = 13
            write_json(path, damaged)
            with self.assertRaisesRegex(ValueError, "effective query count"):
                check(values)
            damaged = deepcopy(original)
            damaged["metrics"]["UnrequestedMetric@1"] = 1.
            write_json(path, damaged)
            with self.assertRaisesRegex(ValueError, "metric keys"):
                check(values)
            write_json(path, original)
            write_json(values[0] / "metrics.json", {})
            with self.assertRaisesRegex(ValueError, "exact reports"):
                check(values)

    def test_frozen_spec_selection_and_source_hash_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            values = fixture(root)
            path = values[0] / "evaluation_manifest.json"
            original = verify._json(path)
            damaged = deepcopy(original)
            damaged["spec"] = spec(ticks=(2, 8, 0))
            write_json(path, damaged)
            with self.assertRaisesRegex(ValueError, "ten frozen valid"):
                check(values)
            write_json(path, original)
            values[-1].marker.write_text("changed source\n")
            with self.assertRaisesRegex(ValueError, "manifest/source"):
                check(values)

    def test_candidate_identity_and_unsaved_score_change_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            values = fixture(root)
            folder, pool, bundle, ks, provider = values
            original = Pool.__getitem__
            def missing(self, query):
                candidates = original(self, query)
                return candidates[1:] if query == 0 else candidates
            with patch.object(Pool, "__getitem__", missing):
                with self.assertRaisesRegex(ValueError, "candidate identities"):
                    check(values)
            with (folder / "rankings.csv").open(newline="") as stream:
                returned = {int(row["candidate_id"]) for row in csv.DictReader(stream) if int(row["query_id"]) == 0}
            absent = next(c for c in pool[0] if int(c) not in returned)
            for matrix in provider.values.values():
                matrix[0, absent] = 1.
            with self.assertRaisesRegex(ValueError, "full-candidate ordering"):
                check(values)

    def test_formal_size_and_frozen_seven_cutoffs_guard(self):
        with tempfile.TemporaryDirectory() as root:
            values = fixture(root)
            path = values[0] / "evaluation_manifest.json"
            manifest = verify._json(path)
            manifest["formal_protocol"] = True
            write_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "all 7199"):
                check(values)

    def test_frozen_formula_endpoints_constant_minmax_and_product_zeros(self):
        components = {"F": np.array([.2, .5, .8]), "G": np.full(3, .4), "WBBS": np.array([0., .2, .9])}
        np.testing.assert_allclose(verify.independent_score(components, spec(ticks=(5, 5, 0))), [0., .25, .5])
        np.testing.assert_allclose(verify.independent_score(components, spec("weighted_product", (5, 0, 5))), np.sqrt(components["F"] * components["WBBS"]))
        for family in ("sum_raw", "sum_minmax", "weighted_product"):
            np.testing.assert_array_equal(verify.independent_score(components, spec(family, (10, 0, 0))), components["F"])


if __name__ == "__main__":
    unittest.main()
