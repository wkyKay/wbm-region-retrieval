"""Subset isolation, original split provenance and repeatability checks."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts import run_mvp
from wbm_retrieval.common import ROOT, csv_rows, read_json, write_csv, write_json


class MVPSubsetTests(unittest.TestCase):
    def fixture(self, root):
        root = Path(root)
        signatures = [(i,) for i in range(8)] + [(0, 6), (1, 2)]
        labels, rows = [], []
        for split, single, mixed in (("train", 20, 25), ("valid", 2, 7), ("test", 4, 14)):
            for signature in signatures:
                for _ in range(single if len(signature) == 1 else mixed):
                    label = np.zeros(8, dtype=np.uint8)
                    label[list(signature)] = 1
                    sample_id = len(labels)
                    labels.append(label)
                    row = {"sample_id": sample_id, "original_index": sample_id,
                           "valid_index": sample_id, "split": split}
                    row.update({f"label_{i}": int(value) for i, value in enumerate(label)})
                    rows.append(row)
        np.savez(root / "source.npz", np.ones((300, 8, 8), dtype=np.uint8), np.array(labels))
        write_csv(root / "split.csv", rows, list(rows[0]))
        write_csv(root / "query.csv", [{"sample_id": 240}], ["sample_id"])
        write_csv(root / "candidate.csv", [{"query_id": 240, "candidate_id": 241}], ["query_id", "candidate_id"])
        source = copy.deepcopy(read_json(ROOT / "configs/experiment_a.json"))
        source.update(data_file=str(root / "source.npz"), split_manifest=str(root / "split.csv"),
                      query_manifest=str(root / "query.csv"), candidate_manifest=str(root / "candidate.csv"),
                      output_dir=str(root / "artifacts/formal"))
        write_json(root / "source.json", source)
        config = copy.deepcopy(read_json(ROOT / "configs/mvp_mps_300.json"))
        output = root / "artifacts/mvp"
        config.update(data_file=str(output / "data.npz"), split_manifest=str(output / "split.csv"),
                      query_manifest=str(output / "query.csv"), candidate_manifest=str(output / "candidate.csv"),
                      output_dir=str(output))
        config["mvp"]["source_config"] = str(root / "source.json")
        return config, {int(row["sample_id"]): row["split"] for row in rows}

    def test_subset_preserves_source_splits_and_formal_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, splits = self.fixture(root)
            with mock.patch.object(run_mvp, "ROOT", root):
                before = run_mvp.protected_files(root / "source.json")
                metadata = run_mvp.build_subset(config)
                again = run_mvp.build_subset(config)
                self.assertEqual(metadata, again)
                self.assertEqual(before, run_mvp.protected_files(root / "source.json"))
            self.assertEqual(metadata["protocol"]["split_sizes"], {"train": 210, "valid": 30, "test": 60})
            mapping = csv_rows(root / "artifacts/mvp/source_mapping.csv")
            self.assertEqual(len({row["source_map_id"] for row in mapping}), 300)
            for row in mapping:
                self.assertEqual(splits[int(row["source_map_id"])], row["source_split"])

    def test_subset_cannot_overwrite_formal_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = self.fixture(root)
            config["data_file"] = str(root / "source.npz")
            with mock.patch.object(run_mvp, "ROOT", root), self.assertRaises(ValueError):
                run_mvp.build_subset(config)


if __name__ == "__main__":
    unittest.main()
