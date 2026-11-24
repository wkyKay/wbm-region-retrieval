"""Synthetic corruption tests for independent samples and all-score replay."""
from copy import deepcopy
import csv
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.proposed.f_segmentation_geometry import coverage_metrics
from paper_c_rotation import scoring, state, verify
from paper_c_rotation.tests.test_scoring import fixture as region_fixture
from f_region_structure.tests.test_verify import fixture as ordinary_fixture
from f_region_j_cross_test.tests.test_verify import fixture as cross_fixture
from fgw_cross_test.tests.test_verify import rewrite_csv, write_json


def fixture(root, *, cross=False, ties=False):
    root = Path(root)
    if cross:
        folder, pool, bundle, ks, _, provider, binding = cross_fixture(root, ties=ties)
        stage = "cross_test"
    else:
        folder = root / "evaluation"; folder.mkdir()
        pool, bundle, ks, _, replay, matrix = ordinary_fixture(folder)
        provider = SimpleNamespace(signature="cache-signature", marker=root/"cache_completed.json",
            protocol_signature=pool.protocol_signature, query_ids=np.asarray(list(pool)), gallery_ids=pool.gallery)
        provider.scores = lambda q, ids: np.array([matrix[int(q), int(c)] for c in ids], dtype=np.float64)
        provider.values = matrix
        write_json(provider.marker, {"signature":provider.signature})
        source = root/"source.json"; write_json(source,{"frozen":True})
        binding = {str(source.resolve()):verify._sha(source)}
        stage = "valid"
    name, context = "primary", "context-signature"
    spec = state.grid()[name]
    signature = state.evaluation_signature(context, stage, name)
    identity = dict(key="paper_c_rotation_fixture_"+stage,run_id="20261120T210203+0800__fixture_"+stage,
                    started_at="2026-11-20T21:02:03+08:00",evaluation_signature=signature,
                    variant=name,split=pool.split_name,stage=stage,
                    protocol=verify.PROTOCOL if cross else "retrieval_v1")
    manifest = dict(**identity,signature=signature,context_signature=context,spec=spec,ks=ks,tie_seed=2026,
                    num_queries=len(pool),gallery_count=len(pool.gallery),
                    candidate_rule=verify.RULE if cross else "all_gallery_except_query",
                    retrieval_protocol_signature=pool.protocol_signature,
                    provider_source={"signature":provider.signature,"completed_sha256":verify._sha(provider.marker)},
                    source_binding_sha256=verify._digest(binding),formal_protocol=False)
    if cross:
        manifest["derived_protocol"] = pool.metadata
    write_json(folder/"evaluation_manifest.json",manifest)
    write_json(folder/"run_identity.json",identity)
    for filename in (["coverage_metrics.json"] if cross else ["metrics.json","coverage_metrics.json"]):
        path=folder/filename; value=verify._json(path); value["metadata"]=manifest; write_json(path,value)
    return folder,pool,bundle,ks,signature,provider,binding



def regional_provider(dtype=np.float64):
    cache, pool = region_fixture(dtype)
    labels = np.zeros((max(pool.gallery)+1, 8), dtype=int)
    classes = np.zeros(len(labels), dtype=int)
    for i, q in enumerate(pool.gallery):
        labels[q, i] = 1
        classes[q] = i
    return scoring.Provider(cache, pool), SimpleNamespace(labels=labels, class_ids=classes)


class FormulaVerificationTests(unittest.TestCase):
    def test_independent_frozen_grid_and_class_candidate_identity(self):
        self.assertEqual(verify.independent_grid(), state.grid())
        provider, bundle = regional_provider()
        receipt = verify.audit_formulas(provider, bundle, state.grid(), max_candidates=2)
        self.assertEqual(receipt["class_count"], 5)
        self.assertEqual(receipt["candidate_pairs"], 5*2*6)
        for detail in receipt["variants"].values():
            self.assertEqual(detail["query_ids"], list(provider.pool))
            for q in provider.pool:
                self.assertEqual(detail["candidate_ids_by_query"][str(q)], sorted(provider.pool[q])[:2])

    def test_scalar_product_preserves_joint_shape_marginal(self):
        joint = np.zeros((2, 128))
        joint[0, 0] = joint[0, 127] = .5
        joint[1, 15] = joint[1, 112] = .5
        independent = verify.independent_distribution(joint)
        np.testing.assert_array_equal(independent[0], independent[1])
        self.assertEqual(independent[0, 3], 0.)  # Shape is p(stick,ball), not p(stick)*p(ball).
        self.assertEqual(independent[0, 15], .25)
        np.testing.assert_array_equal(independent.reshape(-1,8,16).sum(1), joint.reshape(-1,8,16).sum(1))
        np.testing.assert_array_equal(independent.reshape(-1,8,16).sum(2), joint.reshape(-1,8,16).sum(2))
        bad = joint.copy(); bad[0] *= 2
        with self.assertRaises(ValueError): verify.independent_distribution(bad)

    def test_formal_sampling_and_original_token_ownership_are_mandatory(self):
        provider, bundle = regional_provider()
        pool = provider.pool
        class PartialFormalPool:
            def __getattr__(self, key): return getattr(pool, key)
            def __iter__(self): return iter(pool)
            def __len__(self): return 500
        provider.pool = PartialFormalPool()
        with self.assertRaisesRegex(ValueError, "37 classes times 20"):
            verify.audit_formulas(provider, bundle, state.grid())
        provider.pool = pool
        original = provider.tokens
        with patch.object(provider, "tokens", side_effect=lambda q: original(q)[::-1]):
            with self.assertRaisesRegex(ValueError, "map ownership"):
                verify.audit_formulas(provider, bundle, state.grid())
        provider.cache["owners"][0] += 1
        with self.assertRaisesRegex(ValueError, "token ownership"):
            verify.audit_formulas(provider, bundle, state.grid())

    def test_scalar_deletion_formulas_and_hellinger_scale(self):
        provider, _ = regional_provider()
        cache = deepcopy(provider.cache)
        cache["joint_all"][:] = 0.
        cache["joint_all"][:, 0] = 1.
        cache["joint_all"][1] = 0.; cache["joint_all"][1,127] = 1.
        parts = verify.independent_components(cache, np.array([0]), np.array([1]))
        self.assertEqual(parts["joint_distance"][0,0], 1.)
        self.assertEqual(parts["structure_similarity"][0,0], math.exp(-.5))
        self.assertAlmostEqual(parts["area_similarity"][0,0], math.exp(-abs(math.log(5/9))/3))
        parts = {"cosine": np.array([[.2]]), "structure_similarity": np.array([[.7]]),
                 "area_similarity": np.array([[.3]]), "independent_structure_similarity": np.array([[.9]])}
        expected = {"primary": .85*(.4*.2+.6*.7)+.15*.3,
                    "no_c": (.51*.7+.15*.3)/.66, "no_t": (.34*.2+.15*.3)/.49,
                    "no_a": .4*.2+.6*.7, "independent_j": .85*(.4*.2+.6*.9)+.15*.3,
                    "reuse_candidate": .85*(.4*.2+.6*.7)+.15*.3}
        for name, value in expected.items():
            self.assertAlmostEqual(verify.independent_score_components(parts, state.grid()[name])[0,0], value)

    def test_greedy_is_asymmetric_keeps_unmatched_denominator_and_tie_order(self):
        pair = np.array([[.9], [.9]])
        self.assertEqual(verify.independent_match(pair, [4.,9.]), .9*2/5)
        self.assertEqual(verify.independent_match(pair, [4.,9.], True), .9)
        self.assertEqual(verify.independent_match(pair.T, [16.]), .9)
        self.assertEqual(verify.independent_match(np.zeros((0,2)), []), 0.)
        self.assertEqual(verify.independent_match(np.zeros((2,0)), [4.,9.]), 0.)
        self.assertAlmostEqual(verify.independent_match(np.array([[.8],[0.]]), [4.,9.], True), .8*2/5)
        with self.assertRaises(ValueError): verify.independent_match([[float('nan')]], [1.])

    def test_all_six_formula_and_matching_checks_allow_source_float32_rounding(self):
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                provider, bundle = regional_provider(dtype)
                receipt = verify.audit_formulas(provider, bundle, state.grid())
                self.assertEqual(receipt["candidate_pairs"], 5*4*6)
                self.assertEqual(set(receipt["variants"]), set(state.grid()))
                self.assertLessEqual(receipt["max_raw_formula_absolute_error"], 2e-6)
                self.assertLessEqual(receipt["max_fusion_absolute_error"], 2e-6)
                self.assertLessEqual(receipt["max_matching_absolute_error"], 1e-12)
                self.assertIn("sampled formula audit", receipt["scope"])

    def test_corrupt_components_fusion_and_matching_are_detected_separately(self):
        for issue in ("cosine", "independent_structure_similarity", "fusion", "matching"):
            with self.subTest(issue=issue):
                provider, bundle = regional_provider()
                if issue in ("cosine", "independent_structure_similarity"):
                    original = provider.parts
                    def broken(q):
                        value = original(q)
                        value[issue] = np.maximum(0., value[issue]-.05)
                        return value
                    target, name, fn = provider, "parts", broken
                elif issue == "fusion":
                    original = scoring.score_components
                    def broken(parts, spec): return original(parts, spec)*.95
                    target, name, fn = scoring, "score_components", broken
                else:
                    original = provider.score
                    def broken(*args):
                        ids, values = original(*args)
                        return ids, values*.95
                    target, name, fn = provider, "score", broken
                with patch.object(target, name, side_effect=fn):
                    with self.assertRaisesRegex(ValueError, "formula mismatch|aggregation mismatch"):
                        verify.audit_formulas(provider, bundle, state.grid())


class CompleteCacheVerificationTests(unittest.TestCase):
    def test_all_ordinary_and_cross_queries_scores_metrics_and_groups(self):
        for cross in (False, True):
            with self.subTest(cross=cross), tempfile.TemporaryDirectory() as root:
                values = fixture(root, cross=cross)
                with patch.object(scoring, "score_components", side_effect=AssertionError("no formula recompute")), \
                     patch.object(coverage_metrics, "query_metrics", side_effect=AssertionError("no production metrics")):
                    receipt = verify.verify_evaluation(*values)
                self.assertEqual(receipt["exact_reports"], not cross)
                self.assertEqual(receipt["full_candidates_checked"], sum(len(values[1][q]) for q in values[1]))
                self.assertEqual(set(receipt["files"]), set(verify.CROSS_REPORT_FILES if cross else verify.REPORT_FILES))
                self.assertEqual(set(receipt["metric_query_counts"]), {"coverage"} if cross else {"exact","coverage"})
                self.assertLessEqual(receipt["max_metric_absolute_error"], 1e-12)
                self.assertIn("Not a second regional score computation", receipt["scope"])

    def test_high_unsaved_candidate_and_equal_score_tie_order_are_detected(self):
        for cross in (False, True):
            with self.subTest(cross=cross), tempfile.TemporaryDirectory() as root:
                values = fixture(root, cross=cross)
                folder, pool, _, _, _, provider, _ = values
                q = int(next(iter(pool)))
                with (folder/"rankings.csv").open() as stream:
                    returned = {int(row["candidate_id"]) for row in csv.DictReader(stream) if int(row["query_id"]) == q}
                omitted = next(int(c) for c in pool[q] if int(c) not in returned)
                provider.values[q, omitted] = 1.
                with self.assertRaisesRegex(ValueError, "full-candidate"): verify.verify_evaluation(*values)
        with tempfile.TemporaryDirectory() as root:
            values = fixture(root, cross=True, ties=True)
            verify.verify_evaluation(*values)
            def swap(rows):
                rows[0]["candidate_id"], rows[1]["candidate_id"] = rows[1]["candidate_id"], rows[0]["candidate_id"]
            rewrite_csv(values[0]/"rankings.csv", swap)
            with self.assertRaisesRegex(ValueError, "cache ordering"): verify.verify_evaluation(*values)

    def test_self_consistent_changed_science_or_formal_flag_is_rejected(self):
        for issue in ("spec", "formal"):
            with self.subTest(issue=issue), tempfile.TemporaryDirectory() as root:
                values = list(fixture(root)); folder = values[0]
                m = verify._json(folder/"evaluation_manifest.json")
                if issue == "spec": m["spec"]["distribution"] = "independent"
                else: m["formal_protocol"] = True
                sig = verify._digest({"context":m["context_signature"],"stage":m["stage"],"variant":m["variant"],"spec":m["spec"]})
                m["signature"] = m["evaluation_signature"] = sig; values[4] = sig
                identity = verify._json(folder/"run_identity.json"); identity["evaluation_signature"] = sig
                write_json(folder/"evaluation_manifest.json", m); write_json(folder/"run_identity.json", identity)
                with self.assertRaises(ValueError): verify.verify_evaluation(*values)

    def test_cache_identity_dtype_source_and_nonfinite_scores_are_rejected(self):
        for issue in ("binding", "marker", "query_order", "protocol", "float32", "nan"):
            with self.subTest(issue=issue), tempfile.TemporaryDirectory() as root:
                values = fixture(root); provider = values[5]
                if issue == "binding": values[6].clear()
                elif issue == "marker": write_json(provider.marker, {"changed":True})
                elif issue == "query_order": provider.query_ids = provider.query_ids[::-1]
                elif issue == "protocol": provider.protocol_signature = "changed"
                elif issue == "float32":
                    original = provider.scores; provider.scores = lambda q,c: original(q,c).astype(np.float32)
                else: provider.values[next(iter(provider.values))] = float("nan")
                with self.assertRaises(ValueError): verify.verify_evaluation(*values)

    def test_query_order_denominator_undefined_group_metadata_and_exact_in_cross(self):
        for issue in ("order", "denominator", "undefined", "group", "metadata", "exact"):
            with self.subTest(issue=issue), tempfile.TemporaryDirectory() as root:
                values = fixture(root, cross=True); folder = values[0]
                if issue == "order": rewrite_csv(folder/"rankings.csv", lambda rows: rows.__setitem__(slice(0,10), rows[5:10]+rows[:5]))
                elif issue == "denominator": rewrite_csv(folder/"coverage_per_query.csv", lambda rows: rows[0].update(full_coverage_in_pool=999))
                elif issue == "undefined":
                    def change(rows):
                        row = next(r for r in rows if r["Coverage-nDCG@1"] == ""); row["Coverage-nDCG@1"] = 0
                    rewrite_csv(folder/"coverage_per_query.csv", change)
                elif issue == "exact": write_json(folder/"metrics.json", {})
                else:
                    path = folder/"coverage_metrics.json"; report = verify._json(path)
                    if issue == "group": report["groups"]["single_label"]["num_queries"] += 1
                    else: report["metadata"]["source_binding_sha256"] = "changed"
                    write_json(path, report)
                with self.assertRaises(ValueError): verify.verify_evaluation(*values)

    def test_complete_label_combination_is_excluded_independently(self):
        with tempfile.TemporaryDirectory() as root:
            values = list(fixture(root, cross=True))
            pool = values[1]
            class AlteredPool:
                def __getattr__(self, key): return getattr(pool, key)
                def __iter__(self): return iter(pool)
                def __len__(self): return len(pool)
                def __getitem__(self, q):
                    candidates = pool[q].copy()
                    if q == 0: candidates[0] = 1  # Same complete combination, non-self.
                    return candidates
            values[1] = AlteredPool()
            with self.assertRaisesRegex(ValueError, "full label exclusion"): verify.verify_evaluation(*values)

    def test_ordinary_self_exclusion_and_candidate_order_are_checked(self):
        for issue in ("self", "order"):
            with self.subTest(issue=issue), tempfile.TemporaryDirectory() as root:
                values = list(fixture(root)); pool = values[1]
                class AlteredPool:
                    def __getattr__(self, key): return getattr(pool, key)
                    def __iter__(self): return iter(pool)
                    def __len__(self): return len(pool)
                    def __getitem__(self, q):
                        candidates = pool[q].copy()
                        if issue == "self": candidates[0] = q
                        else: candidates = candidates[::-1]
                        return candidates
                values[1] = AlteredPool()
                with self.assertRaises(ValueError): verify.verify_evaluation(*values)

    def test_mutating_cache_marker_mid_replay_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            values = fixture(root); provider = values[5]; original = provider.scores
            def changed(q, c):
                write_json(provider.marker, {"changed":True}); return original(q,c)
            provider.scores = changed
            with self.assertRaisesRegex(ValueError, "changed during verification"): verify.verify_evaluation(*values)


if __name__ == "__main__": unittest.main()
