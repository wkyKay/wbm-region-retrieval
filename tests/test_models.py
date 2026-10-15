"""Forward/gradient and inference integration checks; no optimizer steps or training."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from test_core import fixture_config

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "Install requirements.txt for model verification")
class ModelTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(123)

    def test_encoder_gradients_and_directional_scorer(self):
        import torch
        from wbm_retrieval.models import Encoder, DirectionalScorer, Projector
        encoder, projector = Encoder(16), Projector(16, 8)
        z = encoder(torch.rand(2, 1, 32, 32))
        self.assertEqual(z.shape, (2, 16))
        torch.testing.assert_close(z.norm(dim=1), torch.ones(2))
        projected = projector(z)
        scorer = DirectionalScorer(16, 16)
        a = scorer(z[:1], z[1:])
        b = scorer(z[1:], z[:1])
        self.assertFalse(torch.allclose(a, b))
        (a.sum() + projected.square().sum()).backward()
        self.assertIsNotNone(encoder.stem[0].weight.grad)

    def test_memory_excludes_anchors_and_normalizes_update(self):
        import torch
        from wbm_retrieval.methods.proposed import MemoryBank, nce_loss
        memory = MemoryBank(10, 8, "cpu")
        memory.buffer.copy_(torch.nn.functional.normalize(torch.rand(10, 8), dim=1))
        chosen = memory.negative_indices(torch.tensor([1, 3]), 1024)
        self.assertEqual(len(chosen), 8)
        self.assertFalse(any(i in {1, 3} for i in chosen.tolist()))
        memory.update(torch.tensor([0]), torch.rand(1, 8))
        torch.testing.assert_close(memory.buffer[0].norm(), torch.tensor(1.))
        value = torch.rand(2, 8, requires_grad=True)
        loss = nce_loss(memory.buffer[:2], value, memory.buffer[2:], .07)
        loss.backward()
        self.assertTrue(torch.isfinite(value.grad).all())

    def test_views_change_by_epoch(self):
        from wbm_retrieval.methods.proposed import PatchDataset
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "patch.npy"
            image = np.ones((1, 1, 32, 32), dtype=np.uint8)
            np.save(path, image)
            settings = {"rotation": True, "max_shift": 3, "dropout": .05, "noise": 0.}
            dataset = PatchDataset(path, settings, 2026)
            first = dataset[0][2].numpy()
            np.testing.assert_array_equal(first, dataset[0][2].numpy())
            dataset.epoch = 1
            self.assertFalse(np.array_equal(first, dataset[0][2].numpy()))

    def test_checkpoint_restores_weights_rng_and_rejects_mismatch(self):
        import torch
        from wbm_retrieval.training import restore, save_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5)
            initial = model.weight.detach().clone()
            path = Path(directory) / "last.pt"
            save_checkpoint(path, model, optimizer, scheduler, 1, .2, [{"epoch": 1}], "fixed-signature")
            expected_random = torch.rand(3)
            with torch.no_grad():
                model.weight.add_(10)
            epoch, best, _, _ = restore(path, model, optimizer, scheduler, "fixed-signature", True)
            self.assertEqual((epoch, best), (1, .2))
            torch.testing.assert_close(model.weight, initial)
            torch.testing.assert_close(torch.rand(3), expected_random)
            with self.assertRaises(ValueError):
                restore(path, model, optimizer, scheduler, "other-signature", True)
            with self.assertRaises(FileExistsError):
                restore(path, model, optimizer, scheduler, "fixed-signature", False)

    def test_preprocessing_to_five_rankings_without_training(self):
        import torch
        from wbm_retrieval.common import budget_name
        from wbm_retrieval.data import Protocol, prepare
        from wbm_retrieval.evaluation import compare_paired, evaluate_all
        from wbm_retrieval.geometry import generate_examples
        from wbm_retrieval.models import Encoder, GlobalRanker, Projector
        from wbm_retrieval.retrieval import retrieve
        from wbm_retrieval.training import training_signature
        with tempfile.TemporaryDirectory() as directory:
            config, labels = fixture_config(directory)
            prepare(config)
            protocol = Protocol(config, labels)
            root = Path(config["output_dir"])
            (root / "models/local").mkdir(parents=True)
            local = torch.nn.ModuleDict({"encoder": Encoder(16), "projector": Projector(16, config["local"]["projector_dim"])})
            torch.save({"model": local.state_dict(), "signature": training_signature(config, "local")}, root / "models/local/best.pt")
            del local
            global_model = GlobalRanker(16, config["global"]["pair_hidden_dim"])
            for budget in config["global"]["budgets"]:
                name = budget_name(budget)
                (root / "models" / name).mkdir()
                torch.save({"model": global_model.state_dict(), "signature": training_signature(config, name)}, root / "models" / name / "best.pt")
            retrieve(config, protocol)
            summary = evaluate_all(config, protocol)
            self.assertEqual(len(summary), 5)
            self.assertTrue((root / "comparison.csv").is_file())
            compare_paired(config)
            generate_examples(config)
            self.assertTrue(list((root / "geometry").glob("*.png")))


if __name__ == "__main__":
    unittest.main()
