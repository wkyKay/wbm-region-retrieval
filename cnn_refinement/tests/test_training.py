import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cnn_refinement.training import FilteredNegativeSampler, filtered_nce_loss, _snapshot_last
from experiments.proposed.main.method import MemoryBank, nce_loss
from wbm_retrieval.training import optimization, restore, save_checkpoint


def graph(rows):
    return {"indptr": np.cumsum([0]+[len(row) for row in rows]),
            "indices": np.asarray([value for row in rows for value in row], dtype=np.int32)}


class FilteredNegativeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.bank = MemoryBank(12, 8, "cpu")
        self.bank.buffer.copy_(torch.randn(12, 8))
        self.anchors = torch.tensor([0, 1])

    def test_exact_5000_counts_exclusions_and_no_global_rng_consumption(self):
        rows = [[2, 3], [4, 5]] + [[] for _ in range(10)]
        sampler = FilteredNegativeSampler(graph(rows), 12, "cpu")
        base = self.bank.negative_indices(self.anchors, 5000)
        global_rng = torch.get_rng_state().clone()
        changes = sampler.replacements(self.anchors, base, self.anchors)
        self.assertTrue(torch.equal(torch.get_rng_state(), global_rng))
        actual = base.repeat(2, 1)
        actual[changes["rows"], changes["columns"]] = changes["ids"]
        self.assertEqual(actual.shape, (2, 5000))
        for row in range(2):
            self.assertFalse(np.isin(actual[row].numpy(), [0, 1]+rows[row]).any())
        self.assertGreater(changes["rejected"], 0)
        self.assertEqual(changes["eligible_anchor_count"], 2)
        self.assertEqual(changes["zero_filter_anchor_count"], 0)
        self.assertEqual(changes["base_duplicate_slots"], 5000-base.unique().numel())
        self.assertEqual(changes["eligible_negatives_min"], 8)
        changed_positions = torch.zeros_like(actual, dtype=torch.bool)
        changed_positions[changes["rows"], changes["columns"]] = True
        self.assertTrue(torch.equal(actual[~changed_positions], base.repeat(2, 1)[~changed_positions]))

    def test_empty_filter_loss_and_gradient_exactly_match_legacy(self):
        sampler = FilteredNegativeSampler(graph([[] for _ in range(12)]), 12, "cpu")
        base = self.bank.negative_indices(self.anchors, 5000)
        changes = sampler.replacements(self.anchors, base, self.anchors)
        self.assertEqual(changes["zero_filter_anchor_count"], 2)
        positives = torch.randn(2, 8, requires_grad=True)
        reference = positives.detach().clone().requires_grad_()
        anchors = self.bank.buffer[self.anchors]
        actual = filtered_nce_loss(anchors, positives, self.bank.buffer, base, changes, .07)
        expected = nce_loss(anchors, reference, self.bank.buffer[base], .07)
        self.assertTrue(torch.equal(actual, expected))
        actual.backward()
        expected.backward()
        self.assertTrue(torch.equal(positives.grad, reference.grad))

    def test_sparse_logit_replacements_match_explicit_per_anchor_objective(self):
        # Float64 separates objective/gradient equivalence from float32 GEMM
        # accumulation-order differences between a matrix and individual rows.
        self.bank.buffer = self.bank.buffer.double()
        sampler = FilteredNegativeSampler(graph([[2, 3], [4, 5]]+[[] for _ in range(10)]), 12, "cpu")
        base = self.bank.negative_indices(self.anchors, 5000)
        changes = sampler.replacements(self.anchors, base, self.anchors)
        positives = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
        reference = positives.detach().clone().requires_grad_()
        anchors = self.bank.buffer[self.anchors]
        actual = filtered_nce_loss(anchors, positives, self.bank.buffer, base, changes, .07)
        negative_ids = base.repeat(2, 1)
        negative_ids[changes["rows"], changes["columns"]] = changes["ids"]
        expected = torch.stack([nce_loss(anchors[i:i+1], reference[i:i+1],
                               self.bank.buffer[negative_ids[i]], .07) for i in range(2)]).mean()
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(positives.grad, reference.grad, atol=1e-10, rtol=1e-10)

    def test_supplementary_rng_state_roundtrip_is_exact(self):
        rows = [[2, 3], [4, 5]]+[[] for _ in range(10)]
        sampler = FilteredNegativeSampler(graph(rows), 12, "cpu")
        base = self.bank.negative_indices(self.anchors, 5000)
        saved = sampler.generator.get_state()
        a = sampler.replacements(self.anchors, base, self.anchors)
        other = FilteredNegativeSampler(graph(rows), 12, "cpu")
        other.generator.set_state(saved)
        b = other.replacements(self.anchors, base, self.anchors)
        for key in ("rows", "columns", "ids"):
            self.assertTrue(torch.equal(a[key], b[key]))

    def test_exhausted_pool_and_bad_base_fail(self):
        sampler = FilteredNegativeSampler(graph([list(range(2, 12)), []]+[[] for _ in range(10)]), 12, "cpu")
        with self.assertRaisesRegex(ValueError, "No permissible"):
            sampler.replacements(self.anchors, torch.tensor([2, 3]), self.anchors)
        empty = FilteredNegativeSampler(graph([[] for _ in range(12)]), 12, "cpu")
        with self.assertRaisesRegex(ValueError, "batch exclusion"):
            empty.replacements(self.anchors, torch.tensor([0, 3]), self.anchors)

    def test_checkpoint_restores_model_optimizer_and_all_rng_streams(self):
        import random
        np.random.seed(19)
        random.seed(18)
        model = torch.nn.Linear(8, 3)
        settings = {"learning_rate": .01, "momentum": .9, "weight_decay": .001, "epochs": 100}
        optimizer, scheduler = optimization(model, settings)
        model(torch.randn(2, 8)).sum().backward()
        optimizer.step()
        scheduler.step()
        sampler = FilteredNegativeSampler(graph([[2, 3], [4, 5]]+[[] for _ in range(10)]), 12, "cpu")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "last.pt"
            save_checkpoint(path, model, optimizer, scheduler, 15, .1, [{"epoch": 15}], "fixture",
                            negative_generator_rng=sampler.generator.get_state(), memory=self.bank.buffer)
            expected_torch, expected_numpy, expected_python = torch.rand(3), np.random.rand(3), random.random()
            torch.manual_seed(4)
            np.random.seed(3)
            random.seed(2)
            epoch, _, history, payload = restore(path, model, optimizer, scheduler, "fixture", True)
            self.assertEqual(epoch, 15)
            self.assertEqual(history, [{"epoch": 15}])
            self.assertTrue(torch.equal(torch.rand(3), expected_torch))
            np.testing.assert_array_equal(np.random.rand(3), expected_numpy)
            self.assertEqual(random.random(), expected_python)
            self.assertTrue(torch.equal(payload["memory"], self.bank.buffer))
            _snapshot_last(Path(folder), 15)
            self.assertEqual(path.read_bytes(), (Path(folder) / "checkpoints/epoch_015.pt").read_bytes())
            with self.assertRaisesRegex(ValueError, "mismatch"):
                restore(path, model, optimizer, scheduler, "different", True)


if __name__ == "__main__":
    unittest.main()
