"""Independent finite-difference and exact-count fixtures for NCE diagnostics."""
import importlib.util
import json
import math
import unittest

import numpy as np

from f_representation_audit.negative_audit import (
    audit_negative_sampling, nce_gradient_diagnostic, sameclass_crossmap_mask,
    singleton_prototypes, token_metadata,
)


class NegativeSamplingTests(unittest.TestCase):
    def test_source_ids_and_singleton_proxy_do_not_filter_bank(self):
        result = token_metadata([5, 5, 9, 12], {5: [0, 1], 9: [1, 1], 12: [0, 0]})
        self.assertEqual(result["singleton_labels"], [1, 1, -1, -1])
        self.assertEqual(result["map_ids"], [5, 5, 9, 12])
        self.assertEqual(result["token_indices"], [0, 1, 2, 3])

    def test_exact_batch_exclusion_keeps_same_map_siblings_and_mixed_maps(self):
        ids, labels = [10, 10, 20, 20, 30, 40], [0, 0, 0, 0, 1, -1]
        result = audit_negative_sampling(ids, labels, [[0, 4]], num_negatives=5)
        row = result["rows"][0]
        self.assertEqual(row["negative_pool_tokens"], 4)
        self.assertEqual(row["same_map_negative_tokens"], 1)
        self.assertEqual(row["sameclass_crossmap_negative_tokens"], 2)
        self.assertEqual(row["nonsingleton_negative_tokens"], 1)
        self.assertEqual(row["expected_sameclass_crossmap_count"], 2.5)
        self.assertAlmostEqual(row["probability_sameclass_crossmap_at_least_one"], 1 - .5 ** 5)
        self.assertEqual(result["rows"][1]["expected_sameclass_crossmap_count"], 0.)
        self.assertEqual(result["summary"]["bank_tokens"], 6)
        json.dumps(result, allow_nan=False)

    def test_full_bank_batch_uses_historical_self_only_fallback(self):
        result = audit_negative_sampling([1, 1, 2], [0, 0, 0], [[0, 1, 2]], num_negatives=5000)
        row = result["rows"][0]
        self.assertEqual(row["exclusion_rule"], "self_only_full_batch_fallback")
        self.assertEqual(row["same_map_negative_tokens"], 1)
        self.assertEqual(row["sameclass_crossmap_negative_tokens"], 1)
        self.assertEqual(row["expected_sameclass_crossmap_count"], 2500.)
        self.assertEqual(row["probability_sameclass_including_same_map_at_least_one"], 1.)

    def test_sameclass_mask_excludes_same_map_and_retains_repeated_draws(self):
        mask = sameclass_crossmap_mask([1, 1, 2, 3, 4], [0, 0, 0, 1, -1],
                                      [0, 4], [1, 2, 2, 3, 4])
        np.testing.assert_array_equal(mask, [[False, True, True, False, False], [False] * 5])

    def test_prototypes_are_crossmap_map_balanced(self):
        memory = [[1., 0.], [1., 0.], [1., 0.], [0., 2.], [-1., 0.]]
        proto = singleton_prototypes(memory, [1, 2, 2, 3, 4], [0, 0, 0, 0, 1], [0, 1, 4])
        np.testing.assert_allclose(proto, [[.5, .5], [.5, .5], [0., 0.]])

    def test_invalid_metadata_and_batches_rejected(self):
        for ids, labels, batches in (([1, 1], [0, 1], [[0]]), ([1, 2], [0, 0], [[0], [0]]),
                                      ([1, 2], [0, 0], [[2]]), ([1], [0], [[0]])):
            with self.assertRaises(ValueError):
                audit_negative_sampling(ids, labels, batches)
        with self.assertRaises(ValueError):
            token_metadata([1], {1: [0, .5]})


class NegativeGradientTests(unittest.TestCase):
    def fixture(self):
        return (np.array([[1.2, .4, -.1], [.2, .8, .3]]),
                np.array([[.2, 1., 0.], [.7, .3, .2]]),
                np.array([[0., 1., 0.], [-1., .1, .2], [.7, .2, .1]]),
                np.array([[True, False, True], [False, True, False]]),
                np.array([[.2, .9, .1], [-.5, .3, .2]]))

    def test_analytic_gradient_matches_independent_finite_difference(self):
        z, anchors, negatives, mask, proto = self.fixture()
        result = nce_gradient_diagnostic(z, anchors, negatives, mask, proto,
                                         temperature=.3, return_gradients=True)
        def loss(x, selected_mask):
            u = x / np.linalg.norm(x)
            a = anchors[0] / np.linalg.norm(anchors[0])
            n = negatives / np.linalg.norm(negatives, axis=1, keepdims=True)
            logits = np.r_[u @ a, n @ u] / .3
            logits[1:][selected_mask] = -np.inf
            return np.log(np.exp(logits - logits.max()).sum()) + logits.max() - logits[0]
        for name, selected in (("full", np.zeros(3, bool)), ("masked", mask[0])):
            numerical = []
            for d in range(3):
                plus, minus = z[0].copy(), z[0].copy()
                plus[d] += 1e-6
                minus[d] -= 1e-6
                numerical.append((loss(plus, selected) - loss(minus, selected)) / 2e-6)
            np.testing.assert_allclose(result["gradients"][name][0], numerical, atol=1e-8)
        self.assertLess(result["summary"]["gradient_decomposition_max_error"], 1e-14)
        json.dumps(result, allow_nan=False)

    def test_sameclass_repulsion_can_coexist_with_net_attraction(self):
        # Same-class negative term repels +y, but the memory positive attracts
        # more strongly: a diagnostic that reported only the term would mislead.
        result = nce_gradient_diagnostic([[1., 0.]], [[0., 1.]],
                                         [[0., 1.], [0., -1.]], [[True, False]], [[0., 1.]],
                                         temperature=1.)
        row = result["rows"][0]
        self.assertLess(row["sameclass_negative_term_descent_toward_class"], 0.)
        self.assertGreater(row["full_descent_toward_class"], 0.)
        self.assertGreater(row["masked_descent_toward_class"], 0.)

    def test_sameclass_negative_term_is_not_always_away_from_prototype(self):
        # A sampled same-class negative may be atypical relative to the full
        # class prototype. The label alone does not fix the derivative sign.
        result = nce_gradient_diagnostic([[1., 0.]], [[1., 0.]],
                                         [[0., -1.]], [[True]], [[0., 1.]], temperature=1.)
        self.assertGreater(result["rows"][0]["sameclass_negative_term_descent_toward_class"], 0.)

    def test_mask_counterfactual_is_not_direct_sameclass_term(self):
        z, anchors, negatives, mask, proto = self.fixture()
        result = nce_gradient_diagnostic(z, anchors, negatives, mask, proto,
                                         temperature=.3, return_gradients=True)
        self.assertFalse(np.allclose(result["gradients"]["full_minus_masked"],
                                     result["gradients"]["sameclass_negative_term"]))

    def test_undefined_tangent_and_no_sameclass_have_no_invented_effect(self):
        result = nce_gradient_diagnostic([[1., 0.]], [[0., 1.]], [[0., -1.]],
                                         [[False]], [[1., 0.]], temperature=.2)
        row = result["rows"][0]
        self.assertFalse(row["prototype_direction_defined"])
        self.assertIsNone(row["full_descent_toward_class"])
        self.assertEqual(row["sameclass_negative_term_gradient_norm"], 0.)
        self.assertEqual(row["full_minus_masked_gradient_norm"], 0.)
        self.assertEqual(row["loss_full"], row["loss_masked"])
        json.dumps(result, allow_nan=False)

    def test_all_negatives_removed_leaves_zero_loss_and_gradient(self):
        result = nce_gradient_diagnostic([[1., 0.]], [[0., 1.]],
                                         [[1., 1.]], [[True]], [[0., 1.]], temperature=.1)
        self.assertEqual(result["rows"][0]["loss_masked"], 0.)
        self.assertEqual(result["rows"][0]["masked_gradient_norm"], 0.)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch unavailable in this interpreter")
    def test_independent_historical_torch_forward_and_autograd(self):
        result = nce_gradient_diagnostic(*self.fixture(), temperature=.3, verify_torch=True)
        for value in result["summary"]["torch_verification"].values():
            self.assertLess(value, 1e-9)


if __name__ == "__main__":
    unittest.main()
