"""Exercise frozen ten recipes through real filtering, metrics, verification and resume."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from fgw_cross_test import run, state
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from wbm_retrieval.retrieval_protocol import FullGalleryPool

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "artificial/fgw_topk_test_20261119T205138+0800/frozen_selection.json"


def fixture(folder):
    folder = Path(folder)
    frozen = read_json(FROZEN)
    write_json(folder / "frozen_selection.json", frozen)
    kinds = np.array([[1, 1, 1, 0], [1, 1, 1, 1], [1, 0, 1, 0],
                      [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.uint8)
    labels = np.tile(kinds, (140, 1))
    bundle = SimpleNamespace(labels=labels, class_ids=np.arange(700) % 5)
    base = FullGalleryPool(np.arange(5), np.arange(700), "test", "synthetic_base")
    pool = state.cross_pool(base, labels)
    marker = folder / "source_marker.json"
    write_json(marker, {"status": "completed", "source": "immutable synthetic cache"})
    rng = np.random.default_rng(2026)
    matrices = {name: rng.uniform(.2, .8, size=(5, 700)) for name in ("F", "G", "WBBS")}
    for i, q in enumerate(base):
        exact = np.all(labels == labels[q], axis=1)
        matrices["F"][i, exact] = 1.
        matrices["G"][i, exact] = 0.
        for value in matrices.values():
            value[i, q] = np.nan
    source = SimpleNamespace(signature="synthetic_source", marker=marker, split="test",
        query_ids=np.array(list(base)), gallery_ids=base.gallery, protocol_signature=base.protocol_signature,
        rows={int(q): i for i, q in enumerate(base)}, matrices=matrices)
    provider = run.FilteredScoreProvider(source, pool)
    return frozen, bundle, pool, provider


class RunnerTests(unittest.TestCase):
    def test_ten_recipes_full_metrics_and_readonly_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            frozen, bundle, pool, provider = fixture(output)
            matrix_before = {key: value.copy() for key, value in provider.source.matrices.items()}
            marker_before = (file_hash(provider.marker), Path(provider.marker).stat().st_mtime_ns)
            settings = frozen["valid_selection"]["settings"]
            index = run.evaluate_fixed(output, "parent_sig", provider, pool, bundle, frozen["specs"], settings)
            self.assertEqual(index["ranked_variants"], frozen["ranked_variants"])
            self.assertEqual(len(index["summary_rows"]), 10)
            for name in frozen["ranked_variants"]:
                folder = state.evaluation_folder(output, name)
                report = read_json(folder / "coverage_metrics.json")
                checked = read_json(folder / "verification.json")
                self.assertEqual(report["ks"], [1, 5, 10, 20, 50, 100, 500])
                self.assertEqual(len(report["metrics"]), 42)
                self.assertEqual(report["num_queries"], 5)
                self.assertEqual(checked["ranking_rows"], 2500)
                self.assertFalse((folder / "metrics.json").exists())
            before = {str(p): (file_hash(p), p.stat().st_mtime_ns)
                      for p in output.rglob("*") if p.is_file()}
            again = run.evaluate_fixed(output, "parent_sig", provider, pool, bundle, frozen["specs"], settings)
            self.assertEqual(index, again)
            after = {str(p): (file_hash(p), p.stat().st_mtime_ns)
                     for p in output.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            self.assertEqual(marker_before, (file_hash(provider.marker), Path(provider.marker).stat().st_mtime_ns))
            for key, value in provider.source.matrices.items():
                np.testing.assert_array_equal(value, matrix_before[key])

    def test_changed_spec_and_incomplete_formal_pool_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            frozen, bundle, pool, provider = fixture(temp)
            settings, specs = frozen["valid_selection"]["settings"], deepcopy(frozen["specs"])
            specs[next(iter(specs))]["weights"] = [.3, .3, .4]
            with self.assertRaisesRegex(ValueError, "original valid top ten"):
                run.evaluate_fixed(temp, "sig", provider, pool, bundle, specs, settings)
            with self.assertRaisesRegex(ValueError, "Formal test requires"):
                run.evaluate_fixed(temp, "sig", provider, pool, bundle, frozen["specs"], settings, formal=True)

    def test_filter_precedes_minmax_and_rejects_unfiltered_or_reordered_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            frozen, _, pool, provider = fixture(temp)
            candidates = pool[0]
            values = provider.scores(0, candidates)
            self.assertLess(values["F"].max(), .81)
            self.assertGreater(values["G"].min(), .19)
            spec = frozen["specs"]["sum_minmax_d10_f02_g02_w06"]
            expected = sum(w * (values[k] - values[k].min()) / (values[k].max() - values[k].min())
                           for k, w in zip(("F", "G", "WBBS"), spec["weights"]))
            np.testing.assert_allclose(run.scoring.score_variant(values, spec), expected, rtol=0, atol=1e-15)
            with self.assertRaisesRegex(ValueError, "entire filtered"):
                provider.scores(0, np.arange(1, 700))
            with self.assertRaisesRegex(ValueError, "entire filtered"):
                provider.scores(0, candidates[::-1])

    def test_original_protocol_signature_serialization(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, pool, _ = fixture(temp)
            self.assertEqual(state.derived_signature(pool.metadata), pool.protocol_signature)
            self.assertNotEqual(digest(pool.metadata), pool.protocol_signature)

    def test_output_cannot_replace_source_or_resume_unbound_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "artificial").mkdir()
            original = root / "artificial/fgw_topk_test_original"
            original.mkdir()
            with self.assertRaises(ValueError):
                state.choose_output(str(original), root=root)
            new = root / "artificial/fgw_cross_test_new"
            new.mkdir()
            with self.assertRaises(FileExistsError):
                state.choose_output(str(new), root=root)
            with self.assertRaisesRegex(ValueError, "original manifest"):
                state.choose_output(resume_root=str(new), root=root)


if __name__ == "__main__":
    unittest.main()
