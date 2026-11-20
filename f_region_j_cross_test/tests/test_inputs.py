from copy import deepcopy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from f_region_j_cross_test import inputs
from wbm_retrieval.common import digest, file_hash, write_json


class InputTests(unittest.TestCase):
    def fixture(self):
        root = Path("/frozen/valid")
        valid = {"context": {"stage": "valid", "fixed": "original"}}
        selection = {"winner": inputs.VARIANT, "selected_variants": [inputs.VARIANT],
            "selected_specs": {inputs.VARIANT: dict(inputs.SPEC)},
            "selection_rule": {"split": "valid", "metric": "exact/all/Precision@10"},
            "context_signature": digest(valid["context"])}
        choice = {"selection": selection, "selected_specs": selection["selected_specs"], "signature": "frozen"}
        matching = {**inputs.MATCHING, "sigma_pos": .35, "sigma_scale": .5}
        binding = {"/original/cache.npy": "original-sha"}
        context = {"split": "test", "source_run": str(root), "grid": choice["selected_specs"],
            "frozen_selection_signature": "frozen", "source_binding": binding,
            "cache_source_run": "/original", "matching": matching, "training": False,
            "test_parameter_search": False, "numpy": inputs.np.__version__,
            "torch": inputs.torch.__version__, "code": {"original.py": "code-hash"}}
        test = {"version": "f_region_area_fusion_v1", "source_run": str(root), "context": context}
        data = {"frozen_selection": choice, "source_binding": binding,
                "cache_source_run": Path("/original"), "config": {"matching": matching}}
        inp = {"source_binding": deepcopy(binding), "context_signature": digest(context),
               "source_run": str(root), "cache_source_run": "/original"}
        return data, valid, test, inp, deepcopy(choice), root

    def validate(self, values):
        with patch.object(inputs, "area_code", return_value={"original.py": "code-hash"}), \
             patch.dict(os.environ, {k: "1" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}):
            inputs._validate_source(*values)

    def test_accepts_exact_frozen_source_and_preserves_unused_matching_settings(self):
        values = self.fixture()
        before = deepcopy(values)
        self.validate(values)
        self.assertEqual(values, before)

    def test_rejects_different_valid_winner_or_regional_strength_spec(self):
        for field in ("winner", "spec", "selection_split"):
            values = self.fixture()
            choice = values[0]["frozen_selection"]
            if field == "winner": choice["selection"]["winner"] = "test_winner"
            elif field == "spec": choice["selected_specs"][inputs.VARIANT]["beta"] = .15
            else: choice["selection"]["selection_rule"]["split"] = "test"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "complete-valid winner"):
                self.validate(values)

    def test_rejects_original_test_binding_or_frozen_selection_drift(self):
        for field in ("binding", "frozen", "matching", "test_source"):
            values = self.fixture()
            if field == "binding": values[3]["source_binding"]["/original/cache.npy"] = "changed"
            elif field == "frozen": values[4]["signature"] = "changed"
            elif field == "matching": values[2]["context"]["matching"]["weight"] = "uniform"
            else: values[2]["source_run"] = "/another/valid"
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(values)

    def test_rejects_numerical_environment_or_scientific_code_change(self):
        for field in ("numpy", "torch", "code"):
            values = self.fixture()
            values[2]["context"][field] = "changed" if field != "code" else {"original.py": "changed"}
            values[3]["context_signature"] = digest(values[2]["context"])
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(values)

    def test_rejects_nonfrozen_blas_thread_environment(self):
        with patch.object(inputs, "area_code", return_value={"original.py": "code-hash"}), \
             patch.dict(os.environ, {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "8"}):
            with self.assertRaisesRegex(ValueError, "single-thread BLAS"):
                inputs._validate_source(*self.fixture())

    def test_seal_binds_every_file_readonly_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            file = root / "report.csv"
            file.write_text("frozen\n")
            write_json(root / "completed.json", {"status": "completed", "signature": "sig",
                "files": {file.name: file_hash(file)}})
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir()}
            binding = {}
            inputs._seal(root, binding, "sig", {"report.csv"})
            self.assertEqual(set(binding), {str(root / "completed.json"), str(file)})
            self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir()})
            file.write_text("changed\n")
            with self.assertRaises(ValueError): inputs._seal(root, {}, "sig")

    def test_seal_rejects_wrong_signature_missing_required_and_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "root"
            root.mkdir()
            outside = root.parent / "outside"
            outside.write_text("outside")
            for signature, files, required in (("wrong", {}, set()), ("sig", {}, {"report.csv"}),
                                               ("sig", {"../outside": file_hash(outside)}, set())):
                write_json(root / "completed.json", {"status": "completed", "signature": signature, "files": files})
                with self.assertRaises(ValueError): inputs._seal(root, {}, "sig", required)

    def test_seal_rejects_symlinked_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            actual = root / "actual"
            actual.write_text("actual")
            alias = root / "alias"
            alias.symlink_to(actual)
            write_json(root / "completed.json", {"status": "completed", "signature": "sig", "files": {"alias": file_hash(actual)}})
            with self.assertRaises(ValueError): inputs._seal(root, {}, "sig")


if __name__ == "__main__": unittest.main()
