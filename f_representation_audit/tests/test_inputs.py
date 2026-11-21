"""Small read-only fixtures for source/label/cache identity rejection."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from f_representation_audit.inputs import (
    _load_dataset, _signed, _validate_split_rows, verify_checkpoint_metadata, verify_protection,
)
from wbm_retrieval.common import digest, file_hash, write_json


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.labels = np.zeros((4, 8), dtype=np.uint8)
        self.labels[0, 0] = self.labels[1, 5] = self.labels[2, 7] = 1
        self.maps = np.ones((4, 4, 4), dtype=np.uint8)
        self.maps[0, :2, :3] = 2
        self.maps[1, :2, :3] = 2

    def split_rows(self):
        return [{"sample_id": str(i), "original_index": str(i), "valid_index": str(i),
                 "split": split, "label_signature": str(int(self.labels[i].argmax())),
                 **{f"label_{j}": str(int(value)) for j, value in enumerate(self.labels[i])}}
                for i, split in enumerate(("train", "valid", "test"))]

    def fixture(self):
        cache = self.root / "cache"
        cache.mkdir()
        np.save(cache / "labels.npy", self.labels)
        regions = np.zeros((2, 4, 4), dtype=np.uint8)
        regions[:, :2, :3] = 1
        np.save(cache / "train_region_labels.npy", regions)
        np.save(cache / "train_F.npy", np.zeros((2, 3, 8, 8), dtype=np.uint8))
        np.savez(cache / "train_regions.npz", map_ids=[0, 1], offsets=[0, 1, 2], area=[6, 6])
        for suffix in ("F_window_audit.csv", "segmentation_audit.csv", "audit.json"):
            (cache / f"train_{suffix}").write_text("fixture\n")
        write_json(self.root / "valid_pool.json", {"fixture": True})
        self.resign()

    def resign(self):
        names = [f"cache/train_{suffix}" for suffix in
                 ("region_labels.npy", "regions.npz", "F.npy", "F_window_audit.csv", "segmentation_audit.csv", "audit.json")]
        names += ["cache/labels.npy", "valid_pool.json"]
        write_json(self.root / "cache/train_prepared.json", {
            "signature": "fixture", "files": {name: file_hash(self.root / name) for name in names}})

    def load(self, expected_ids=(0, 1)):
        return _load_dataset(self.root, "train", "fixture", np.asarray(expected_ids), self.maps, self.labels,
                             {"proposal": {"min_area": 5, "top_k": 4}, "input_size": 8}, {})

    def test_read_only_cache_identity_and_singleton_label_order(self):
        self.fixture()
        before = {str(p): file_hash(p) for p in self.root.rglob("*") if p.is_file()}
        result = self.load()
        np.testing.assert_array_equal(result["token_map_ids"], [0, 1])
        np.testing.assert_array_equal(result["singleton_labels"], [0, 5])
        np.testing.assert_array_equal(result["token_region_ids"], [1, 1])
        np.testing.assert_array_equal(result["rep_indices"], [0, 1])
        self.assertFalse(result["patches"].flags.writeable)
        self.assertFalse(result["region_masks"].flags.writeable)
        self.assertEqual(verify_protection(before)["status"], "verified")

    def test_split_order_mismatch_is_not_silently_sorted(self):
        self.fixture()
        with self.assertRaisesRegex(ValueError, "sample IDs/order"):
            self.load((1, 0))

    def test_invalid_offsets_are_rejected_even_with_updated_marker(self):
        self.fixture()
        np.savez(self.root / "cache/train_regions.npz", map_ids=[0, 1], offsets=[0, 2, 1], area=[6])
        self.resign()
        with self.assertRaisesRegex(ValueError, "offsets"):
            self.load()

    def test_label_cache_cannot_swap_near_full_and_random(self):
        self.fixture()
        changed = self.labels.copy()
        changed[:, [5, 7]] = changed[:, [7, 5]]
        np.save(self.root / "cache/labels.npy", changed)
        self.resign()
        with self.assertRaisesRegex(ValueError, "cache labels"):
            self.load()

    def test_mask_area_and_nondefect_membership_checked(self):
        self.fixture()
        masks = np.load(self.root / "cache/train_region_labels.npy")
        masks[0, 3, 3] = 1
        np.save(self.root / "cache/train_region_labels.npy", masks)
        self.resign()
        with self.assertRaisesRegex(ValueError, "nondefect"):
            self.load()

    def test_hash_tampering_and_missing_source_are_rejected(self):
        self.fixture()
        (self.root / "cache/train_audit.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.load()
        (self.root / "cache/train_audit.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.load()

    def test_split_label_and_source_id_checks(self):
        counts = {"train": 1, "valid": 1, "test": 1}
        result = _validate_split_rows(self.split_rows(), self.labels, counts)
        np.testing.assert_array_equal(result["valid"], [1])
        changed = self.split_rows()
        changed[1]["label_5"], changed[1]["label_7"] = "0", "1"
        with self.assertRaisesRegex(ValueError, "label vector"):
            _validate_split_rows(changed, self.labels, counts)
        changed = self.split_rows()
        changed[1]["original_index"] = "0"
        with self.assertRaisesRegex(ValueError, "original NPZ"):
            _validate_split_rows(changed, self.labels, counts)
        changed = self.split_rows()
        changed[1]["valid_index"] = "2"
        with self.assertRaisesRegex(ValueError, "valid_index"):
            _validate_split_rows(changed, self.labels, counts)

    def test_manifest_self_signature_and_protection(self):
        payload = {"format": "fixture", "settings": {"seed": 2026}}
        path = self.root / "manifest.json"
        write_json(path, {"signature": digest(payload), **payload})
        protection = {}
        self.assertEqual(_signed(path, protection)["settings"]["seed"], 2026)
        write_json(path, {"signature": digest(payload), **payload, "tampered": True})
        with self.assertRaisesRegex(ValueError, "manifest signature"):
            _signed(path, {})
        with self.assertRaisesRegex(ValueError, "Protected source changed"):
            verify_protection(protection)

    def test_checkpoint_metadata_rejects_wrong_epoch_or_signature(self):
        checkpoint_path = self.root / "best_nce.pt"
        checkpoint_path.write_bytes(b"fixture only; caller loads real torch checkpoint")
        run = {"signature": "run", "initial_parameter_sha256": "initial"}
        inputs = {"run_manifest": run, "spec": {"input_channels": 3},
                  "run_summary": {"best_nce_epoch": 100}, "checkpoint_paths": {"best_nce": checkpoint_path},
                  "protection": {str(checkpoint_path): file_hash(checkpoint_path)}}
        checkpoint = {**run, "variant": "F", "input_spec": inputs["spec"], "epoch": 100,
                      "model": {"encoder.weight": "fixture"}}
        self.assertEqual(verify_checkpoint_metadata(checkpoint, "best_nce", inputs)["epoch"], 100)
        changed = copy.deepcopy(checkpoint)
        changed["epoch"] = 15
        with self.assertRaisesRegex(ValueError, "checkpoint metadata"):
            verify_checkpoint_metadata(changed, "best_nce", inputs)
        changed = copy.deepcopy(checkpoint)
        changed["signature"] = "other"
        with self.assertRaisesRegex(ValueError, "checkpoint metadata"):
            verify_checkpoint_metadata(changed, "best_nce", inputs)


if __name__ == "__main__":
    unittest.main()
