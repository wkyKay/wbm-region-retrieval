"""Real dual-evaluator fixtures and adversarial G/J verification failures."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from experiments.proposed.f_segmentation_geometry.coverage_metrics import evaluate_rankings_file
from gj_fusion.verify import RANKING_FIELDS, audit_components, independent_score, verify_evaluation
from wbm_retrieval.exact_metrics import evaluate_rankings, order_scores
from wbm_retrieval.retrieval_protocol import FullGalleryPool


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False))


def read_csv(path):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fixture(folder, *, only_single=False, spec=None):
    folder = Path(folder)
    ids = np.arange(2, 34, 2, dtype=np.int64)
    labels = np.zeros((34, 8), dtype=np.int64)
    first, second, unique = np.eye(8, dtype=np.int64)[[0, 1, 7]]
    mixed = first + second
    labels[ids] = [first, second, mixed, unique] + [first]*4 + [second]*4 + [mixed]*4
    _, classes = np.unique(labels, axis=0, return_inverse=True)
    bundle = SimpleNamespace(labels=labels, class_ids=classes, valid_ids=ids, reserved_ids=ids)
    queries = ids[[0, 1, 3]] if only_single else ids[:4]
    pool = FullGalleryPool(queries, ids, "valid", "tiny_complete_protocol")
    ks, signature = [1, 5, 10], "tiny_gj_evaluation"
    spec = spec or {"family": "additive", "alpha": .5}
    matrix = {(int(q), int(c)): [float(((int(c)//2 + int(q)) % 5)/5),
                              float(((int(c)//2 + int(q)) % 5)/7)]
              for q in pool for c in pool[q]}

    def components(query, candidates):
        values = np.asarray([matrix[int(query), int(c)] for c in candidates])
        return values[:, 0], values[:, 1]

    rankings, csv_rows = [], []
    for query in pool:
        candidates = pool[query][::-1]
        g, d = components(query, candidates)
        j = np.exp(-2*d)
        family, alpha = spec["family"], spec["alpha"]
        # Fixture scoring is intentionally explicit rather than calling verifier.
        if family == "endpoint":
            scores = g if alpha == 1 else j
        elif family == "additive":
            scores = alpha*g + (1-alpha)*j
        elif family == "product":
            scores = g*j
        else:
            scores = g**alpha * j**(1-alpha)
        order = order_scores(candidates, scores, query, 2026)[:10]
        ranked = candidates[order]
        rankings.append((query, ranked))
        csv_rows.extend({"query_id": query, "rank": rank, "candidate_id": int(candidates[i]),
                         "similarity_score": float(scores[i]), "global_score": float(g[i]),
                         "joint_distance": float(d[i])} for rank, i in enumerate(order, 1))
    write_csv(folder/"rankings.csv", RANKING_FIELDS, csv_rows)
    metadata = {"signature": signature, "retrieval_protocol_signature": pool.protocol_signature}
    evaluate_rankings(bundle, pool, rankings, ks, folder, metadata=metadata)
    evaluate_rankings_file(bundle, pool, folder/"rankings.csv", ks, folder, metadata=metadata)
    write_json(folder/"evaluation_manifest.json", {"signature": signature, "split": "valid", "ks": ks,
        "tie_seed": 2026, "variant": "synthetic_gj", "spec": spec, "num_queries": len(pool),
        "gallery_count": len(ids), "candidate_rule": "all_gallery_except_query",
        "retrieval_protocol_signature": pool.protocol_signature})
    write_json(folder/"run_identity.json", {"key": "synthetic_no_completion_required"})
    return pool, bundle, ks, signature, components, matrix


class GJVerificationTests(unittest.TestCase):
    def check(self, folder, values, components=None):
        pool, bundle, ks, signature, original, _ = values
        return verify_evaluation(folder, pool, bundle, ks, signature, components or original)

    def test_all_formula_families_full_gallery_and_every_report(self):
        specs = [{"family": "endpoint", "alpha": a} for a in (0., 1.)]
        specs += [{"family": "additive", "alpha": a} for a in (.25, .5, .75)]
        specs += [{"family": "geometric", "alpha": a} for a in (.25, .75)]
        specs += [{"family": "product", "alpha": .5}]
        for spec in specs:
            with self.subTest(spec=spec), tempfile.TemporaryDirectory() as tmp:
                result = self.check(tmp, fixture(tmp, spec=spec))
                self.assertEqual(result["status"], "verified")
                self.assertEqual(result["num_queries"], 4)
                self.assertEqual(result["ranking_rows"], 40)
                self.assertEqual(result["full_candidates_checked"], 60)
                self.assertTrue(result["independent_fusion_all_candidates"])
                self.assertLess(result["max_metric_absolute_error"], 1e-12)
                self.assertEqual(result["max_component_absolute_error"], 0.)
                self.assertIsInstance(result["top10_hits"], int)
                self.assertEqual(len(result["files"]), 9)
                self.assertNotIn("completed.json", result["files"])

    def test_empty_mixed_group_and_undefined_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.check(tmp, fixture(tmp, only_single=True))
            self.assertEqual(result["num_queries"], 3)
            report = read_json(Path(tmp)/"coverage_metrics.json")
            self.assertEqual(report["groups"]["mixed_label"]["num_queries"], 0)
            self.assertIsNone(report["groups"]["mixed_label"]["metrics"]["Coverage-nDCG@10"])

    def test_unsaved_stronger_candidate_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = fixture(tmp)
            pool, _, _, _, _, matrix = values
            _, rows = read_csv(Path(tmp)/"rankings.csv")
            query = next(iter(pool))
            saved = {int(r["candidate_id"]) for r in rows if int(r["query_id"]) == query}
            omitted = next(int(c) for c in pool[query] if c not in saved)
            matrix[query, omitted] = [1., 0.]
            with self.assertRaisesRegex(ValueError, "full-candidate ordering"):
                self.check(tmp, values)

    def test_incomplete_nonfinite_or_out_of_range_components_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = fixture(tmp)
            original = values[4]
            def incomplete(q, c):
                g, d = original(q, c)
                return g[:-1], d[:-1]
            with self.assertRaisesRegex(ValueError, "complete gallery"):
                self.check(tmp, values, incomplete)
            for value in (np.nan, np.inf, -.01, 1.01):
                def invalid(q, c, value=value):
                    g, d = original(q, c)
                    g[0] = value
                    return g, d
                with self.subTest(value=value), self.assertRaises(ValueError):
                    self.check(tmp, values, invalid)

    def test_saved_source_component_and_score_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = fixture(tmp)
            path = Path(tmp)/"rankings.csv"
            fields, original = read_csv(path)
            for field in ("global_score", "joint_distance", "similarity_score"):
                changed = deepcopy(original)
                changed[0][field] = str(float(changed[0][field])+.01)
                write_csv(path, fields, changed)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    self.check(tmp, values)

    def test_sha_tie_order_and_repeated_queries_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = fixture(tmp)
            path = Path(tmp)/"rankings.csv"
            fields, original = read_csv(path)
            changed = deepcopy(original)
            changed[0], changed[1] = changed[1], changed[0]
            changed[0]["rank"], changed[1]["rank"] = "1", "2"
            write_csv(path, fields, changed)
            with self.assertRaisesRegex(ValueError, "full-candidate ordering"):
                self.check(tmp, values)
            write_csv(path, fields, original+original[:10])
            with self.assertRaisesRegex(ValueError, "repeated"):
                self.check(tmp, values)

    def test_every_aggregate_level_and_counts_reject_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = fixture(tmp)
            path = Path(tmp)/"coverage_metrics.json"
            original = read_json(path)
            paths = [("metrics", "MeanCoverage@10"),
                     ("class_macro_metrics", "FullCoverage_mAP@10"),
                     ("groups", "single_label", "metrics", "Coverage-nDCG@5"),
                     ("groups", "mixed_label", "metrics", "FullCoveragePrecision@1"),
                     ("groups", "by_label_count", "2", "metrics", "MeanCoverage@5"),
                     ("metric_query_counts", "Coverage-nDCG@10"),
                     ("class_macro_metric_class_counts", "FullCoverageRecall@10"),
                     ("groups", "single_label", "num_classes")]
            for keys in paths:
                with self.subTest(path=keys):
                    changed = deepcopy(original)
                    target = changed
                    for key in keys[:-1]:
                        target = target[key]
                    target[keys[-1]] += 1
                    write_json(path, changed)
                    with self.assertRaises(ValueError):
                        self.check(tmp, values)

    def test_denominators_duplicates_and_per_class_metrics_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = fixture(tmp)
            path = Path(tmp)/"per_query.csv"
            fields, original = read_csv(path)
            changed = deepcopy(original)
            changed[0]["relevant_in_pool"] = str(int(changed[0]["relevant_in_pool"])+1)
            write_csv(path, fields, changed)
            with self.assertRaisesRegex(ValueError, "denominator"):
                self.check(tmp, values)
            write_csv(path, fields, original+original[:1])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                self.check(tmp, values)
            write_csv(path, fields, original)
            path = Path(tmp)/"per_class.csv"
            fields, rows = read_csv(path)
            rows[0]["Precision@10"] = ".999"
            write_csv(path, fields, rows)
            with self.assertRaisesRegex(ValueError, "Metric mismatch"):
                self.check(tmp, values)

    def test_manifest_parameters_and_protocol_cannot_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = fixture(tmp)
            path = Path(tmp)/"evaluation_manifest.json"
            original = read_json(path)
            for key, value in (("signature", "wrong"), ("split", "test"), ("tie_seed", 1),
                    ("ks", [1, 10]), ("num_queries", 500), ("candidate_rule", "shortlist"),
                    ("retrieval_protocol_signature", "wrong"),
                    ("spec", {"family": "additive", "alpha": .6})):
                write_json(path, dict(original, **{key: value}))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    self.check(tmp, values)

    def test_formula_endpoints_and_illegal_specifications(self):
        g, d = np.array([0., .5, 1.]), np.array([0., .5, 1.])
        np.testing.assert_array_equal(independent_score(g, d, {"family": "endpoint", "alpha": 1.}), g)
        np.testing.assert_allclose(independent_score(g, d, {"family": "endpoint", "alpha": 0.}), np.exp(-2*d))
        for spec in ({"family": "endpoint", "alpha": True}, {"family": "product", "alpha": 1.},
                     {"family": "geometric", "alpha": .5}, {"family": "additive", "alpha": np.nan},
                     {"family": "additive", "alpha": .5, "strength": 1.}):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                independent_score(g, d, spec)


class ComponentAuditTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(2026)
        ids = np.array([13, 2, 5, 9, 21, 19], dtype=np.int64)
        h = rng.normal(size=(6, 512)).astype(np.float32)
        h /= np.linalg.norm(h, axis=1, keepdims=True)
        joint = rng.uniform(size=(6, 128))
        joint /= joint.sum(1, keepdims=True)
        joint[[0, 1]] = 0
        empty = joint.sum(1) == 0
        classes = np.zeros(22, dtype=np.int64)
        classes[ids] = [2, 0, 1, 1, 2, 0]
        pool = FullGalleryPool(ids, ids, "valid", "audit_fixture")
        data = {"pool": pool, "bundle": SimpleNamespace(class_ids=classes),
                "global_ids": ids, "global_features": h,
                "descriptors": {"sample_ids": ids, "joint128": joint, "empty": empty}}
        lookup = {int(sample): i for i, sample in enumerate(ids)}
        def components(query, candidates):
            qi = lookup[int(query)]
            rows = np.array([lookup[int(c)] for c in candidates])
            g = np.clip(h[rows] @ h[qi], 0, 1).astype(np.float64)
            d = np.sqrt(.5*np.sum((np.sqrt(joint[rows])-np.sqrt(joint[qi]))**2, axis=1))
            d[np.logical_and(empty[qi], empty[rows])] = 0.
            d[np.logical_xor(empty[qi], empty[rows])] = 1.
            return g, d
        return data, components

    def test_minimum_class_query_and_candidate_ids_with_empty_semantics(self):
        data, components = self.fixture()
        result = audit_components(data, components, max_candidates=3)
        self.assertEqual(result["query_ids"], [2, 5, 13])
        self.assertEqual(result["candidate_ids"], {"2": [5, 9, 13], "5": [2, 9, 13], "13": [2, 5, 9]})
        self.assertEqual(result["candidate_pairs"], 9)
        self.assertEqual(result["empty_pairs_checked"], 8)
        self.assertLessEqual(result["max_global_absolute_error"], 2e-7)
        self.assertLessEqual(result["max_joint_distance_absolute_error"], 1e-12)

    def test_global_and_nonempty_hellinger_corruption_fail(self):
        data, original = self.fixture()
        for channel in (0, 1):
            def corrupted(q, c, channel=channel):
                g, d = original(q, c)
                if q == 5:
                    index = list(c).index(9)
                    (g if channel == 0 else d)[index] += 1e-4
                return g, d
            with self.subTest(channel=channel), self.assertRaises(ValueError):
                audit_components(data, corrupted)

    def test_empty_pair_error_is_not_hidden_by_numeric_tolerance(self):
        data, original = self.fixture()
        def corrupted(q, c):
            g, d = original(q, c)
            if q == 2:
                d[list(c).index(13)] = 1e-14
            return g, d
        with self.assertRaisesRegex(ValueError, "exactly zero or one"):
            audit_components(data, corrupted)

    def test_bad_sample_budget_and_bad_descriptor_mass_rejected(self):
        data, components = self.fixture()
        for size in (0, -1, .5, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                audit_components(data, components, max_candidates=size)
        data["descriptors"]["joint128"][2, 0] += .1
        with self.assertRaisesRegex(ValueError, "component arrays"):
            audit_components(data, components)


if __name__ == "__main__":
    unittest.main()
