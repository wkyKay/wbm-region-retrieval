from copy import deepcopy
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cnn_augmentation.augmentation import PaperViews, VARIANTS
from cnn_augmentation.training import (AUDIT_COUNTERS, INITIAL_HASH, _snapshot_last,
                                       _validate_recipe, audit_history)
from experiments.proposed._shared.ef_suite import RegionViews, build_model
from experiments.proposed.main.method import MemoryBank
from wbm_retrieval.common import seed_everything
from wbm_retrieval.training import (loader, optimization, parameter_hash, restore,
                                    save_checkpoint)


def recipe():
    return {"config": {"seed": 2026, "batch_size": 256, "embedding_dim": 512, "input_size": 96,
            "local": {"epochs": 100, "learning_rate": .01, "momentum": .9, "weight_decay": .001,
                      "projector_dim": 128, "temperature": .07, "num_negatives": 5000,
                      "memory_momentum": .5, "optimizer": "sgd", "scheduler": "wapirl_cosine",
                      "projector_type": "linear", "negative_sampling": "with_replacement"}},
            "spec": {"preprocess": "fixed_window", "input_channels": 3, "stem_stride": 1,
                     "scale_range": None, "window_size": 52, "max_shift": 0},
            "datasets": {"train": {"patches": range(99786)}}}


class TrainingControlTests(unittest.TestCase):
    @unittest.skipUnless(torch.__version__.split("+")[0] == "2.4.0",
                         "Exact historical initializer requires the source PyTorch 2.4.0 runtime")
    def test_original_initial_parameters_match_pinned_source_runtime(self):
        inputs = recipe()
        seed_everything(2026)
        model = build_model(inputs["config"], inputs["spec"])
        self.assertEqual(parameter_hash(model), INITIAL_HASH)

    def test_original_training_recipe_and_budget(self):
        inputs = recipe()
        _validate_recipe(inputs)
        self.assertEqual(len(range(0, 99786, 256))*100, 39000)
        for key, value in [("epochs", 99), ("num_negatives", 4000), ("projector_dim", 256),
                           ("negative_sampling", "without_replacement")]:
            altered = deepcopy(inputs)
            altered["config"]["local"][key] = value
            with self.assertRaisesRegex(ValueError, "recipe"):
                _validate_recipe(altered)
        altered = deepcopy(inputs)
        altered["datasets"]["train"]["patches"] = range(99785)
        with self.assertRaisesRegex(ValueError, "99,786"):
            _validate_recipe(altered)

    def test_bank_initialization_unchanged_for_every_augmentation(self):
        torch.manual_seed(2026)
        encoder = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3*96*96, 8))
        projector = torch.nn.Linear(8, 4)
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            rng = np.random.default_rng(9)
            patches = np.ones((4, 3, 96, 96), dtype=np.uint8)
            patches[:, 0] = rng.integers(0, 2, size=(4, 96, 96))
            patches[:, 1] = patches[:, 0]
            np.save(p/"train_F.npy", patches)
            spec = recipe()["spec"]
            cfg = {"batch_size": 2, "device": "cpu", "num_workers": 0}
            baseline = RegionViews(p, "train", spec, 2026)
            reference = MemoryBank(4, 4, "cpu")
            reference.initialize(encoder, projector, loader(baseline, cfg, 2026, False), "cpu")
            model_hash = parameter_hash(encoder)
            for variant in VARIANTS:
                data = PaperViews(p, "train", spec, 2026, variant)
                bank = MemoryBank(4, 4, "cpu")
                rng_before = torch.get_rng_state().clone()
                bank.initialize(encoder, projector, loader(data, cfg, 2026, False), "cpu")
                self.assertTrue(torch.equal(reference.buffer, bank.buffer))
                self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
                self.assertEqual(model_hash, parameter_hash(encoder))
                samples = bank.negative_indices(torch.tensor([0, 1]), 5000)
                self.assertEqual(samples.numel(), 5000)
                self.assertTrue(torch.all((samples == 2) | (samples == 3)))
                self.assertLess(samples.unique().numel(), samples.numel())

    def test_resume_keeps_optimizer_bank_and_global_rng_states(self):
        torch.manual_seed(2026)
        np.random.seed(11)
        random.seed(12)
        model = torch.nn.Linear(4, 3)
        optimizer, scheduler = optimization(model, recipe()["config"]["local"])
        model(torch.randn(2, 4)).sum().backward()
        optimizer.step()
        scheduler.step()
        bank = torch.randn(9, 3)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save_checkpoint(root/"last.pt", model, optimizer, scheduler, 15, 3.2,
                            [{"epoch": 15}], "fixture", memory=bank, optimizer_steps=5850)
            expected = (torch.rand(3), np.random.random(3), random.random())
            epoch, _, history, payload = restore(root/"last.pt", model, optimizer, scheduler, "fixture", True)
            self.assertEqual(epoch, 15)
            self.assertEqual(payload["optimizer_steps"], 5850)
            self.assertTrue(torch.equal(payload["memory"], bank))
            self.assertTrue(torch.equal(torch.rand(3), expected[0]))
            np.testing.assert_array_equal(np.random.random(3), expected[1])
            self.assertEqual(random.random(), expected[2])
            self.assertEqual(history, [{"epoch": 15}])
            _snapshot_last(root, 15)
            self.assertEqual((root/"last.pt").read_bytes(), (root/"checkpoints/epoch_015.pt").read_bytes())
            with self.assertRaisesRegex(ValueError, "mismatch"):
                restore(root/"last.pt", model, optimizer, scheduler, "changed", True)

    def test_history_audit_uses_eligible_denominator_and_source_retention(self):
        history = []
        for _ in range(2):
            row = {key: 0 for key in AUDIT_COUNTERS}
            row.update(sample_count=4, augmentation_applied_count=3, augmentation_no_op_count=1,
                       empty_target_before_count=1, empty_target_after_count=2, newly_empty_target_count=1,
                       target_retention_eligible_count=3, target_retention_fraction_sum=1.5,
                       target_before_pixels=30, target_retained_source_pixels=20, target_after_pixels=100,
                       support_retention_eligible_count=4, support_retention_fraction_sum=3.)
            history.append(row)
        result = audit_history(history, "crop")
        self.assertEqual(result["samples"], 8)
        self.assertEqual(result["epochs"], 2)
        self.assertEqual(result["totals"]["target_after_pixels"], 200)
        self.assertEqual(result["rates"]["mean_target_retention_fraction"], .5)
        self.assertEqual(result["rates"]["mean_support_retention_fraction"], .75)
        self.assertEqual(result["rates"]["applied_fraction"], .75)
        self.assertEqual(result["rates"]["newly_empty_target_fraction"], .25)


if __name__ == "__main__":
    unittest.main()
