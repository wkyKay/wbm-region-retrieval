from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gj_fusion import inputs
from wbm_retrieval.common import digest, file_hash, write_json


class InputTests(unittest.TestCase):
    def arrays(self):
        gallery = np.array([7, 19, 42], dtype=np.int64)
        features = np.zeros((3, 512), dtype=np.float32)
        features[:, 0] = 1.
        joint = np.zeros((3, 128), dtype=np.float64)
        joint[:2, 0] = 1.
        return gallery, features, {"sample_ids": gallery.copy(), "joint128": joint,
                                  "empty": np.array([False, False, True])}

    def test_different_global_order_is_valid_but_descriptor_order_is_frozen(self):
        gallery, features, descriptors = self.arrays()
        inputs.validate_arrays(gallery[::-1], features, descriptors, gallery)
        descriptors["sample_ids"] = gallery[::-1]
        with self.assertRaises(ValueError):
            inputs.validate_arrays(gallery, features, descriptors, gallery)

    def test_rejects_duplicate_ids_wrong_dimension_and_nonunit_features(self):
        gallery, features, descriptors = self.arrays()
        bad_ids = np.array([7, 7, 42])
        for ids, matrix in ((bad_ids, features), (gallery, features[:, :128]),
                            (gallery, features * .5)):
            with self.subTest(shape=matrix.shape):
                with self.assertRaises(ValueError):
                    inputs.validate_arrays(ids, matrix, descriptors, gallery)

    def test_rejects_invalid_j_probability_mass_or_empty_flags(self):
        for field, value in (("empty", np.array([False, False, False])),
                             ("empty", np.array([0, 0, 1])),
                             ("joint128", np.full((3, 128), np.nan)),
                             ("joint128", np.full((3, 128), 1.))):
            gallery, features, descriptors = self.arrays()
            descriptors[field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    inputs.validate_arrays(gallery, features, descriptors, gallery)

    def test_sealed_source_is_read_only_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "values.bin"
            data.write_bytes(b"frozen")
            write_json(root / "completed.json", {"signature": "science",
                       "files": {"values.bin": file_hash(data)}})
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir()}
            binding = {}
            inputs._sealed(root, "science", {"values.bin"}, binding)
            self.assertEqual(len(binding), 2)
            self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir()})
            data.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                inputs._sealed(root, "science", {"values.bin"}, {})

    def test_seal_rejects_path_escape_missing_files_and_wrong_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            outside = root.parent / "outside"
            outside.write_bytes(b"outside")
            for files, signature, required in (({"../outside": file_hash(outside)}, "ok", set()),
                                              ({}, "bad", set()), ({}, "ok", {"missing"})):
                write_json(root / "completed.json", {"signature": signature, "files": files})
                with self.assertRaises(ValueError):
                    inputs._sealed(root, "ok", required, {})

    def test_invalid_split_rejected_before_any_source_access(self):
        with patch.object(inputs, "_source", side_effect=AssertionError("opened source")):
            with self.assertRaises(ValueError):
                inputs.load_inputs(split="train")

    def global_control(self, root):
        def signed(name, payload):
            value = {**payload, "signature": digest(payload)}
            write_json(root / name, value)
            return value
        identity, sources = {"features_sha256": "frozen-global"}, {"data_file": "frozen-data"}
        manifest = signed("fusion_manifest.json", {"version": "cc8_wapirl_score_fusion_v1",
            "sources": sources, "formula": "(1-alpha)*F+alpha*WaPIRL", "tie_seed": 2026,
            "alphas": [0., 1.], "code": {}, "wapirl_valid": identity})
        stage = signed("valid_manifest.json", {"run_signature": manifest["signature"],
            "split": "valid", "global": identity, "weights": [0., 1.], "selection_signature": None})
        write_json(root / "status.json", {"status": "completed"})
        folder = root / "valid/alpha_1.0"
        folder.mkdir(parents=True)
        for family, filename in (("exact", "metrics.json"), ("coverage", "coverage_metrics.json")):
            report = {"split": "valid", "num_queries": 2, "metric_version": family + "_v1",
                "metadata": {"signature": stage["signature"], "alpha_global": 1.,
                             "candidates_scored": "full_pool", "tie_seed": 2026}}
            if family == "exact":
                report["retrieval_protocol_signature"] = "frozen-protocol"
            write_json(folder / filename, report)
        for name in ("per_query.csv", "coverage_per_query.csv", "rankings.csv"):
            (folder / name).write_text("frozen\n")
        write_json(folder / "completed.json", {"signature": stage["signature"],
            "files": {p.name: file_hash(p) for p in folder.iterdir()}})
        class Pool:
            protocol_signature = "frozen-protocol"
            def __len__(self):
                return 2
        return folder, Pool(), identity, sources

    def test_g_control_binds_the_matching_fusion_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder, pool, identity, sources = self.global_control(root)
            folder = folder.resolve()
            binding = {}
            with patch.object(inputs, "GLOBAL_CONTROL_SOURCE", root):
                self.assertEqual(inputs._global_control("valid", pool, identity, sources, binding), folder)
            self.assertIn(str(folder / "completed.json"), binding)
            self.assertIn(str(folder / "rankings.csv"), binding)

    def test_g_control_rejects_a_different_global_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, pool, _, sources = self.global_control(root)
            with patch.object(inputs, "GLOBAL_CONTROL_SOURCE", root):
                with self.assertRaisesRegex(ValueError, "incompatible sources"):
                    inputs._global_control("valid", pool, {"features_sha256": "different"}, sources, {})


if __name__ == "__main__":
    unittest.main()
