"""Scientific parity, independent cache identity, and train→valid→test restart."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from test_deduplicated_protocol import fixture
from experiments.proposed._shared import ef_suite, ef_training
from experiments.proposed.f_segmentation import run, suite
from experiments.proposed.f_segmentation.proposals import METHODS, segment
from experiments.proposed.f_segmentation.tensor_voting import tensor_vote, retained_points
from wbm_retrieval.common import ROOT, csv_rows, file_hash, read_json, write_json
from wbm_retrieval.data import Protocol
from wbm_retrieval.models import Projector
from wbm_retrieval.regions import extract_regions


class TinyEncoder(torch.nn.Module):
    def __init__(self, channels, dimension):
        super().__init__()
        self.linear = torch.nn.Linear(channels, dimension)

    def forward(self, x):
        return torch.nn.functional.normalize(self.linear(x.mean((-2, -1))), dim=1)


def tiny_model(config, spec):
    return torch.nn.ModuleDict({"encoder": TinyEncoder(3, config["embedding_dim"]),
                               "projector": Projector(config["embedding_dim"], config["local"]["projector_dim"])})


class SegmentationTests(unittest.TestCase):
    def test_cc8_matches_original_masks_and_order(self):
        rng = np.random.default_rng(2026)
        for _ in range(30):
            raw = rng.choice([0, 1, 2, 3], (28, 28), p=[.1, .35, .5, .05])
            expected = extract_regions(raw, min_area=5, top_k=4)
            labels, _ = segment(raw, "cc8")
            self.assertEqual(int(labels.max()), len(expected))
            for i, region in enumerate(expected, 1):
                np.testing.assert_array_equal(labels == i, region["mask"])

    def test_diagonals_filtering_and_original_pixel_contract(self):
        raw = np.ones((15, 15), dtype=np.uint8)
        raw[2:5, 2:5] = raw[5:8, 5:8] = 2
        self.assertEqual(segment(raw, "cc8")[1]["regions"], 1)
        self.assertEqual(segment(raw, "cc4")[1]["regions"], 2)
        for method in METHODS:
            labels, audit = segment(raw, method)
            self.assertFalse(np.any((labels > 0) & (raw != 2)))
            self.assertLessEqual(audit["regions"], 4)
            self.assertTrue(all(np.count_nonzero(labels == i) >= 5 for i in range(1, labels.max()+1)))
            np.testing.assert_array_equal(labels, segment(raw, method)[0])
            self.assertEqual(segment(np.ones_like(raw), method)[1]["regions"], 0)
        line = np.ones_like(raw)
        line[7, 2:13] = 2
        self.assertEqual(segment(line, "cc8")[1]["regions"], 1)
        self.assertEqual(segment(line, "opening8")[1]["regions"], 0)
        self.assertEqual(segment(line, "median8")[1]["regions"], 0)

    def test_median_hole_is_support_only_and_raw_context_is_preserved(self):
        raw = np.ones((9, 9), dtype=np.uint8)
        raw[2:7, 2:7] = 2
        raw[4, 4] = 1
        labels, audit = segment(raw, "median8")
        self.assertEqual(audit["median_added_support_pixels"], 1)
        self.assertEqual(labels[4, 4], 0)
        patch_image, _ = ef_suite.fixed_window_patch(raw, labels == 1, 9, 9)
        self.assertEqual(int(patch_image[0].sum()), int((raw == 2).sum()))
        self.assertEqual(int(patch_image[1].sum()), int(np.count_nonzero(labels)))

    def test_npfc_merges_transitively_without_filling_gaps(self):
        raw = np.ones((10, 15), dtype=np.uint8)
        raw[4:7, 1] = raw[4:7, 3] = raw[4:7, 5] = 2
        raw[1, 12] = 2
        result, _ = segment(raw, "npfc_pre")
        self.assertEqual(int(result.max()), 1)
        self.assertEqual(int(np.count_nonzero(result)), 9)
        self.assertFalse(result[1, 12])
        raw[4:7, 5] = 1
        raw[4:7, 8] = 2
        # Only first two fragments reach min_area=5; distant third is discarded.
        self.assertEqual(segment(raw, "npfc_pre")[1]["retained_pixels"], 6)

    def test_tensor_voting_matches_explicit_pair_formula(self):
        points = np.array([[0, 0], [1, 0], [3, 2], [9, 8]], dtype=float)
        expected = np.zeros((4, 2, 2))
        for i, a in enumerate(points):
            for j, b in enumerate(points):
                if i != j:
                    d = a-b
                    r = d / np.linalg.norm(d)
                    expected[i] += np.exp(-np.dot(d, d)/9) * (np.eye(2) - .5*np.outer(r, r))
        np.testing.assert_allclose(tensor_vote(points, block_size=2), expected, atol=1e-14)
        np.testing.assert_array_equal(retained_points(np.array([[1., 1.], [.32, 0], [.1, .1]])),
                                      [True, True, False])

    def test_spawn_preparation_matches_serial(self):
        maps = np.ones((6, 12, 12), dtype=np.uint8)
        maps[:, 2:8, 3:8] = 2
        config = {"segmentation": {"method": "tv8"}, "proposal": {"min_area": 5, "top_k": 4}}
        serial = list(suite.segmented_maps(maps, range(6), config, 0))
        parallel = list(suite.segmented_maps(maps, range(6), config, 2))
        for (a, x), (b, y) in zip(serial, parallel):
            np.testing.assert_array_equal(a, b)
            self.assertEqual(x, y)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make_config(self, root):
        base, maps, labels, _, splits = fixture(root, per_class=60)
        base.update(input_size=96, batch_size=16, inference_batch_size=32, output_dir=str(root / "base_output"))
        base["evaluation"].update(metric_version="exact_v1", ks=[1, 5, 10], bootstrap_samples=0)
        base["local"]["num_negatives"] = 8
        write_json(root / "base.json", base)
        settings = read_json(ROOT / "experiments/proposed/f_segmentation/config.json")
        settings.update(base_config=str(root / "base.json"), output_root=str(root / "runs"), evaluate_every=1,
                        valid_pool={"seed": 2026, "queries": 6, "candidates": 17})
        settings["overrides"]["local"]["epochs"] = 1
        write_json(root / "suite.json", settings)
        return base, splits

    def test_all_six_train_valid_test_and_idempotent_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, splits = self.make_config(root)
            protected = [Path(base["data_file"]), *list((root / "deduplicated").iterdir()),
                         *list((root / "retrieval").iterdir())]
            before = {p: file_hash(p) for p in protected}
            argv = ["--config", str(root / "suite.json"), "--device", "cpu", "--num-workers", "0"]
            with patch.object(ef_training, "build_model", tiny_model), patch.object(ef_suite, "build_model", tiny_model):
                run.main(argv)
                artifacts = {p: file_hash(p) for p in (root / "runs").rglob("*")
                             if p.is_file() and (p.suffix == ".pt" or p.name == "metrics.json")}
                run.main(argv)
                self.assertEqual(artifacts, {p: file_hash(p) for p in artifacts})
            signatures, pools = set(), set()
            for method in METHODS:
                folder = root / "runs" / method
                signatures.add(read_json(folder / "suite_manifest.json")["signature"])
                pools.add(file_hash(folder / "valid_pool.json"))
                training = read_json(folder / "seed_2026/F/training_summary.json")
                self.assertTrue(training["completed"])
                self.assertGreater(training["optimizer_steps"], 0)
                self.assertNotEqual(training["initial_parameter_sha256"], training["final_parameter_sha256"])
                for split in splits:
                    with np.load(folder / f"cache/{split}_regions.npz") as z:
                        np.testing.assert_array_equal(z["map_ids"], splits[split])
                for split, nq, nc in (("valid", 6, 17), ("test", 36, 35)):
                    target = folder / f"seed_2026/F/{split}/best_nce"
                    report = read_json(target / "metrics.json")
                    self.assertEqual(report["num_queries"], nq)
                    self.assertEqual(report["test_evaluated"], split == "test")
                    rows = csv_rows(target / "per_query.csv")
                    self.assertTrue(all(int(row["pool_size"]) == nc for row in rows))
                    ranks = csv_rows(target / "rankings.csv")
                    self.assertEqual(len(ranks), nq * 10)
                    self.assertTrue(all(int(r["query_id"]) != int(r["candidate_id"]) for r in ranks))
            self.assertEqual(len(signatures), 6)
            self.assertEqual(len(pools), 1)
            self.assertEqual(len(csv_rows(root / "runs/comparison_seed_2026_best_nce.csv")), 12)
            self.assertEqual(before, {p: file_hash(p) for p in protected})
            with self.assertRaisesRegex(ValueError, "changed"):
                run.main(argv + ["--batch-size", "8"])
            corrupted = root / "runs/cc8/cache/valid_F.npy"
            with corrupted.open("r+b") as handle:
                handle.seek(-1, 2)
                handle.write(b"\xff")
            with self.assertRaisesRegex(ValueError, "modified"):
                run.main(argv + ["--methods", "cc8"])

    def test_prepare_omits_test_and_evaluation_retry_preserves_partial_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_config(root)
            argv = ["--config", str(root / "suite.json"), "--device", "cpu", "--methods", "opening8"]
            run.main(argv + ["--stage", "prepare"])
            folder = root / "runs/opening8"
            self.assertFalse((folder / "cache/test_F.npy").exists())
            self.assertFalse((folder / "seed_2026").exists())
            with patch.object(ef_training, "build_model", tiny_model), patch.object(ef_suite, "build_model", tiny_model):
                run.main(argv + ["--stage", "train"])
                def interrupted(*args, **kwargs):
                    target = args[3]
                    write_json(target / "interrupted.json", {"partial": True})
                    raise InterruptedError("evaluation interrupted")

                with patch.object(ef_suite, "evaluate", interrupted):
                    with self.assertRaises(InterruptedError):
                        run.main(argv + ["--stage", "valid"])
                partial = list((folder / "seed_2026/F/valid").glob(".best_nce_attempt_*/interrupted.json"))
                self.assertEqual(len(partial), 1)
                checksum = file_hash(partial[0])
                run.main(argv + ["--stage", "valid"])
                self.assertEqual(file_hash(partial[0]), checksum)
                self.assertTrue((folder / "seed_2026/F/valid/best_nce/completed.json").exists())
                self.assertFalse((folder / "cache/test_F.npy").exists())


if __name__ == "__main__":
    unittest.main()
