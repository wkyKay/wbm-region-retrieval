"""Meaningful checks for paired IDs, raw-label strata and conditional intervals."""
from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from gj_fusion.analysis import (SourceBinding, _load_runs, comparison_plan,
                                label_strata, load_method, paired_bootstrap)
from gj_fusion.scoring import frozen_grid
from gj_fusion.selection import select_candidates
from gj_fusion.state import seal, signed
from wbm_retrieval.common import digest, file_hash, read_json, write_json


def reports(root, *, duplicate=False, wrong_label=False, wrong_mean=False):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    # Deliberately shuffled original IDs and unrelated class IDs between files.
    rows = [(4, [0, 1, 0, 0, 0, 0, 0, 0], .8, .2, .9),
            (1, [1, 0, 0, 0, 0, 0, 0, 0], .4, .1, .7)]
    for family in ("exact", "coverage"):
        prefix = "" if family == "exact" else "coverage_"
        data = []
        for q, label, precision, ap, coverage in rows:
            row = dict(query_id=q, class_id=999-q if family == "exact" else 600+q,
                       label_signature="|".join(map(str, np.flatnonzero(label))) if family == "exact" else "".join(map(str, label)),
                       label_count=1, pool_size=7198, returned=500)
            if wrong_label and q == 1:
                row["label_signature"] = "wrong"
            if family == "exact":
                row.update({"Precision@10": precision, "AP@100": ap})
            else:
                row.update({"Coverage-nDCG@10": coverage})
            data.append(row)
        if duplicate:
            data.append(data[0].copy())
        with (root / (prefix+"per_query.csv")).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
        metrics = ({"Precision@10": .6, "mAP@100": .15} if family == "exact"
                   else {"Coverage-nDCG@10": .8})
        if wrong_mean:
            metrics[next(iter(metrics))] += .1
        write_json(root / (prefix+"metrics.json"),
                   dict(split="test", num_queries=2, metric_version=family+"_v1", metrics=metrics))


class PairedReportsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.ids = np.array([1, 4], dtype=np.int64)
        self.labels = np.zeros((6, 8), dtype=np.int64)
        self.labels[1, 0] = 1; self.labels[4, 1] = 1

    def tearDown(self):
        self.temporary.cleanup()

    def test_aligns_original_ids_and_ignores_implementation_class_id(self):
        reports(self.root)
        actual = load_method(self.root, self.root, self.ids, self.labels, split="test", pool_size=7198)
        np.testing.assert_array_equal(actual["hits"], [4, 8])
        np.testing.assert_allclose(actual["continuous"], [[.1, .7], [.2, .9]])
        self.assertEqual(actual["top10_hits"], 12)
        np.testing.assert_allclose(actual["means"], [.6, .15, .8])

    def test_rejects_duplicate_queries_wrong_raw_labels_or_means(self):
        for option in ("duplicate", "wrong_label", "wrong_mean"):
            with self.subTest(option=option):
                reports(self.root, **{option: True})
                with self.assertRaises(ValueError):
                    load_method(self.root, self.root, self.ids, self.labels, split="test", pool_size=7198)

    def test_rejects_other_candidate_protocol(self):
        reports(self.root)
        with self.assertRaises(ValueError):
            load_method(self.root, self.root, self.ids, self.labels, split="test", pool_size=7000)

    def test_label_strata_follow_raw_patterns_not_id_magnitude(self):
        patterns, strata = label_strata(self.ids, self.labels)
        self.assertEqual(patterns, [(0, 1, 0, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0, 0, 0)])
        np.testing.assert_array_equal(strata[0], [1])
        np.testing.assert_array_equal(strata[1], [0])


class BootstrapTests(unittest.TestCase):
    def test_analytically_exact_stratified_differences(self):
        h = np.asarray([[1, -2], [1, -2], [3, 4], [3, 4]], dtype=np.int64)
        f = np.asarray([[[.01, .02], [-.01, -.02]], [[.01, .02], [-.01, -.02]],
                        [[.03, .04], [.05, .06]], [[.03, .04], [.05, .06]]])
        groups = [np.asarray([0, 1]), np.asarray([2, 3])]
        actual = paired_bootstrap(h, f, groups, repetitions=19, block_size=6)
        expected = np.asarray([[.2, .02, .03], [.1, .02, .02]])
        np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), rtol=0, atol=1e-15)

    def test_shared_draws_preserve_opposite_contrasts_and_seed(self):
        values = np.asarray([1, -1, 3, 2], dtype=np.int64)
        h = np.column_stack((values, -values))
        f = np.stack((np.column_stack((values*.01, values*.02)),
                      np.column_stack((-values*.01, -values*.02))), axis=1)
        groups = [np.asarray([0, 1]), np.asarray([2, 3])]
        first = paired_bootstrap(h, f, groups, repetitions=100, block_size=17)
        second = paired_bootstrap(h, f, groups, repetitions=100, block_size=17)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[:, 0], -first[:, 1])
        self.assertGreater(np.std(first[:, 0, 0]), 0)

    def test_identical_methods_have_zero_interval(self):
        samples = paired_bootstrap(np.zeros((4, 1), dtype=np.int64), np.zeros((4, 1, 2)),
                                   [np.arange(4)], repetitions=20)
        np.testing.assert_array_equal(samples, 0)

    def test_invalid_strata_and_array_shapes_rejected(self):
        h = np.zeros((4, 1), dtype=np.int64); f = np.zeros((4, 1, 2))
        for strata in ([np.array([0, 1]), np.array([1, 2, 3])], [np.array([0, 1, 2])],
                       [np.array([], dtype=int), np.arange(4)], [np.array([0., 1., 2., 3.])]):
            with self.assertRaises(ValueError):
                paired_bootstrap(h, f, strata, repetitions=2)
        with self.assertRaises(ValueError):
            paired_bootstrap(h.astype(float), f, [np.arange(4)], repetitions=2)
        with self.assertRaises(ValueError):
            paired_bootstrap(h, f, [np.arange(4)], repetitions=True)

    def test_plan_keeps_duplicate_roles_without_duplicate_computations(self):
        selection = {"family_winners": {"additive": "J_only", "geometric": "product"}}
        pairs, aliases = comparison_plan(selection)
        self.assertEqual(pairs, [("J_only", "product"), ("J_only", "J_only"),
                                 ("J_only", "G_only"), ("J_only", "WBBS")])
        self.assertEqual(len(aliases), 5)
        self.assertEqual(aliases[0]["pair_index"], aliases[1]["pair_index"])
        self.assertTrue(aliases[2]["identical_variant"])


class SourceEvidenceTests(unittest.TestCase):
    def test_modified_or_escaping_completion_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "report.txt").write_text("sealed")
            write_json(root / "completed.json", dict(status="completed", signature="sig",
                       files={"report.txt": file_hash(root / "report.txt")}))
            binding = SourceBinding(); binding.sealed(root, signature="sig", require_status=True)
            (root / "report.txt").write_text("changed")
            with self.assertRaises(ValueError):
                binding.recheck()
            with self.assertRaises(ValueError):
                SourceBinding().sealed(root, signature="sig", require_status=True)
            write_json(root / "completed.json", dict(files={"../outside": "bad"}))
            with self.assertRaises(ValueError):
                SourceBinding().sealed(root)

    def _runs(self, root):
        valid, test = root / "valid_run", root / "test_run"
        valid.mkdir(); test.mkdir()
        context = {"split": "valid", "source_binding": {}}
        manifest = signed(dict(version="gj_fusion_v1", split="valid", context=context))
        write_json(valid / "run_manifest.json", manifest)
        results = {name: dict(split="valid", top10_hits=2000, num_queries=500) for name in frozen_grid()}
        index = dict(results=results)
        write_json(valid / "evaluation_index_valid.json", index)
        chosen = signed(dict(**select_candidates(results), context_signature=digest(context),
                             valid_index_sha256=file_hash(valid / "evaluation_index_valid.json")))
        write_json(valid / "selection.json", chosen)
        test_context = {"split": "test", "frozen_selection_signature": chosen["signature"]}
        write_json(test / "run_manifest.json", signed(dict(version="gj_fusion_v1", split="test", context=test_context)))
        write_json(test / "frozen_selection.json", chosen)
        write_json(test / "evaluation_index_test.json", dict(variants=chosen["selected_specs"],
                   results={name: {} for name in chosen["selected_variants"]}))
        for folder, split, names in ((valid, "valid", frozen_grid()), (test, "test", chosen["selected_variants"])):
            write_json(folder / "status.json", {"status": "completed"})
            for name in names:
                write_json(folder / split / name / "verification.json", dict(status="verified", split=split,
                           variant=name, num_queries=500 if split == "valid" else 7199,
                           all_queries_full_candidate_topk_exact=True))
        seal(valid, digest(context)); seal(test, digest(test_context))
        return valid, test, chosen

    def test_loads_only_completed_matching_valid_frozen_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            valid, test, chosen = self._runs(Path(temporary))
            actual, _ = _load_runs(valid, test, SourceBinding())
            self.assertEqual(actual, chosen)
            bad = dict(chosen); bad["winner"] = "G_only"
            write_json(test / "frozen_selection.json", signed({k:v for k,v in bad.items() if k != "signature"}))
            with self.assertRaises(ValueError):
                _load_runs(valid, test, SourceBinding())


if __name__ == "__main__":
    unittest.main()
