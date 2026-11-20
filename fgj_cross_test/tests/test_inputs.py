from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from fgj_cross_test import inputs
from wbm_retrieval.common import file_hash, write_json
from wbm_retrieval.retrieval_protocol import FullGalleryPool


def fixture():
    # Two samples per each of 37 complete labels, preserving noncontiguous IDs.
    gallery = np.arange(1, 149, 2, dtype=np.int64)
    labels = np.zeros((149, 8), dtype=np.uint8)
    classes = np.full(149, -1, dtype=np.int64)
    for row, q in enumerate(gallery):
        label = row // 2 + 1
        labels[q] = [(label >> k) & 1 for k in range(8)]
        classes[q] = row // 2
    base = FullGalleryPool(gallery, gallery, "test", "test-base")
    pool = inputs.cross_pool(base, labels)
    rng = np.random.default_rng(2026)
    features = rng.normal(size=(len(gallery), 512)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    joint = rng.random((len(gallery), 128))
    joint /= joint.sum(axis=1, keepdims=True)
    empty = np.zeros(len(gallery), dtype=np.bool_)
    empty[[0, 2]] = True
    joint[empty] = 0.
    matrices = {"F": np.full((len(gallery), len(gallery)), .2, dtype=np.float64),
                "G": np.full((len(gallery), len(gallery)), np.nan, dtype=np.float64),
                "WBBS": np.full((len(gallery), len(gallery)), .7, dtype=np.float64)}
    rows = {int(q): i for i, q in enumerate(gallery)}
    for i, q in enumerate(gallery):
        c = base[int(q)]
        positions = [rows[int(x)] for x in c]
        matrices["G"][i, positions] = np.clip(features[positions] @ features[i], 0, 1)
        matrices["F"][i, i] = matrices["WBBS"][i, i] = np.nan
    return dict(bundle=SimpleNamespace(labels=labels, class_ids=classes), base_pool=base, pool=pool,
        cache=SimpleNamespace(query_ids=gallery, gallery_ids=gallery, matrices=matrices,
                              rows=rows, columns=rows.copy()),
        global_ids=gallery, global_features=features,
        descriptors=dict(sample_ids=gallery, joint128=joint, empty=empty))


class InputTests(unittest.TestCase):
    def test_reads_columns_in_requested_order_without_global_recomputation(self):
        data = fixture()
        provider = inputs.Components(data, "cpu")
        q = int(data["pool"].queries[0])
        candidates = data["pool"][q][[9, 1, 4]]
        # Deliberately change embeddings after constructing the immutable score
        # matrix: a filtered-input dot product would now give different G.
        data["global_features"][:] = 0.
        got = provider(q, candidates)
        for name in ("F", "G", "WBBS"):
            expected = data["cache"].matrices[name][data["cache"].rows[q],
                [data["cache"].columns[int(c)] for c in candidates]]
            np.testing.assert_array_equal(got[name], expected)
        self.assertEqual(set(got), {"F", "G", "WBBS", "joint_distance"})
        self.assertTrue(all(v.dtype == np.float64 for v in got.values()))

    def test_rejects_self_same_label_unknown_duplicate_and_noninteger_ids(self):
        data = fixture()
        provider = inputs.Components(data, "cpu")
        q, twin = map(int, data["pool"].queries[:2])
        eligible = int(data["pool"][q][0])
        for query, candidates in ((q, [q]), (q, [twin]), (q, [999]),
                                  (q, [eligible, eligible]), (q, [float(eligible)]),
                                  (True, [eligible]), (999, [eligible])):
            with self.subTest(query=query, candidates=candidates):
                with self.assertRaises(ValueError):
                    provider(query, np.asarray(candidates))

    def test_audit_independent_pairs_empty_semantics_and_exact_full_rows(self):
        data = fixture()
        audit = inputs.audit_components(data, inputs.Components(data, "cpu"))
        self.assertEqual(audit["status"], "verified")
        self.assertEqual(audit["num_queries"], 37)
        self.assertEqual(audit["candidate_pairs"], 740)
        self.assertEqual(audit["full_global_candidates_checked"], 37 * 73)
        self.assertEqual(audit["max_full_global_cache_error"], 0.)
        self.assertTrue(audit["full_global_cache_bitwise_identical"])
        self.assertGreater(audit["empty_pairs_checked"], 0)
        self.assertLessEqual(audit["max_global_absolute_error"], 2e-7)
        self.assertLessEqual(audit["max_joint_distance_absolute_error"], 1e-12)

    def test_audit_rejects_changed_global_cache_outside_sampled_twenty(self):
        data = fixture()
        q = int(data["pool"].queries[0])
        c = int(data["pool"][q][-1])
        data["cache"].matrices["G"][data["cache"].rows[q], data["cache"].columns[c]] += 1e-10
        with self.assertRaisesRegex(ValueError, "not bitwise identical"):
            inputs.audit_components(data, inputs.Components(data, "cpu"))

    def test_audit_rejects_bad_joint_distance(self):
        data = fixture()
        real = inputs.Components(data, "cpu")
        class BadProvider:
            device = "cpu"
            def __call__(self, q, c):
                result = real(q, c)
                result["joint_distance"][:] = .25
                return result
        with self.assertRaises(ValueError):
            inputs.audit_components(data, BadProvider())

    def test_provider_refuses_different_gallery_order(self):
        data = fixture()
        data["cache"].gallery_ids = data["cache"].gallery_ids[::-1]
        with self.assertRaisesRegex(ValueError, "cache IDs"):
            inputs.Components(data, "cpu")

    def test_seal_read_only_and_binds_actual_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            file = folder / "frozen.csv"
            file.write_bytes(b"frozen\n")
            write_json(folder / "completed.json", dict(signature="science", files={file.name: file_hash(file)}))
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in folder.iterdir()}
            binding = {}
            inputs._seal(folder, binding, signature="science", required={file.name})
            self.assertEqual(len(binding), 2)
            self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in folder.iterdir()})
            file.write_bytes(b"modified\n")
            with self.assertRaisesRegex(ValueError, "Modified frozen"):
                inputs._seal(folder, {}, signature="science", required={file.name})

    def test_seal_rejects_escape_missing_required_and_wrong_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "source"
            folder.mkdir()
            outside = folder.parent / "outside"
            outside.write_bytes(b"outside")
            for files, signature, required in (({"../outside": file_hash(outside)}, "science", set()),
                                              ({}, "wrong", set()), ({}, "science", {"required.csv"})):
                write_json(folder / "completed.json", dict(signature=signature, files=files))
                with self.assertRaises(ValueError):
                    inputs._seal(folder, {}, signature="science", required=required)

    def test_code_bind_rejects_outside_repository(self):
        with self.assertRaisesRegex(ValueError, "escapes repository"):
            inputs._code({"../other.py": "hash"}, {})

    def test_gj_test_binding_reproduces_exact_four_valid_additions_without_mutating_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            names = ("completed.json", "selection.json", "run_manifest.json", "evaluation_index_valid.json")
            for name in names:
                (root / name).write_text(name)
            base = {"/source/global.npy": "source-sha"}
            binding = dict(base)
            expected = {**base, **{str(root / name): file_hash(root / name) for name in names}}
            self.assertEqual(inputs._gj_test_binding(base, root, binding), expected)
            self.assertEqual(binding, expected)
            self.assertEqual(base, {"/source/global.npy": "source-sha"})
            (root / "selection.json").unlink()
            with self.assertRaisesRegex(ValueError, "Missing frozen source"):
                inputs._gj_test_binding(base, root, {})

    def test_gj_test_binding_rejects_changed_already_bound_valid_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for name in ("completed.json", "selection.json", "run_manifest.json", "evaluation_index_valid.json"):
                (root / name).write_text(name)
            with self.assertRaisesRegex(ValueError, "conflicts with base"):
                inputs._gj_test_binding({str(root / "completed.json"): "wrong-hash"}, root, {})


if __name__ == "__main__":
    unittest.main()
