"""Synthetic full-gallery, report, immutability and interruption checks."""
from contextlib import redirect_stdout
import csv
from io import StringIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from b1_ablation import pipeline, scoring
from b1_structure_fusion.verify import independent_query_metrics
from experiments.proposed.f_segmentation_geometry import pipeline as geo
from wbm_retrieval.common import file_hash, read_json
from wbm_retrieval.exact_metrics import order_scores
from wbm_retrieval.retrieval_protocol import FullGalleryPool


class Components:
    def __init__(self):
        self.calls = []

    def components(self, query, candidates):
        ids = np.asarray(candidates)
        self.calls.append((int(query), tuple(map(int, ids))))
        local = np.where(ids == 11, 0., 1.)
        glob = np.where(ids == 11, 1., .8)
        return {"f_score": .9*local, "optimal_f_score": local, "global_score": glob,
                "optimal": .5*local+.5*glob, "baseline": .45*local+.5*glob,
                "angular_distance": np.zeros(len(ids)), "density_distance": np.zeros(len(ids))}


def fixture(split="valid"):
    ids = np.arange(12, dtype=np.int64)
    labels = np.tile([0, 1], (12, 1))
    labels[[0, 11]] = [1, 0]
    labels[2] = [1, 1]
    _, classes = np.unique(labels, axis=0, return_inverse=True)
    bundle = SimpleNamespace(labels=labels, class_ids=classes, valid_ids=ids, reserved_ids=ids)
    pool = FullGalleryPool([0, 1, 2], ids, split, "synthetic_frozen_protocol")
    joint = np.zeros((12, 128))
    joint[:, 0] = 1.
    values = {"sample_ids": ids, "joint128": joint, "empty": np.zeros(12, dtype=bool)}
    settings = scoring.load_settings()
    settings["ks"] = [1, 5, 10]
    return pool, bundle, values, settings


def snapshot(folder):
    return {str(p.relative_to(folder)): (file_hash(p), p.stat().st_mtime_ns)
            for p in Path(folder).rglob("*") if p.is_file()}


class PipelineTests(unittest.TestCase):
    def test_full_gallery_candidate_enters_and_both_reports_use_same_ranking(self):
        pool, bundle, values, settings = fixture()
        scorer = Components()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            out = Path(tmp)/"ablation"
            reports = pipeline.evaluate(out, "source_valid", scorer, pool, bundle, settings["ks"], 2026,
                                        values, settings, device="cpu")
            self.assertEqual(set(reports), set(scoring.specs("valid")))
            self.assertEqual(len(scorer.calls), len(pool))
            self.assertEqual(scorer.calls[0][1], tuple(range(1, 12)))
            candidates = pool[0]
            v = Components().components(0, candidates)
            old_order = order_scores(candidates, scoring.full_b1_score(v, {"J": np.zeros(len(candidates))}), 0, 2026)[:10]
            self.assertNotIn(11, candidates[old_order])
            with (out/"valid/no_local/rankings.csv").open() as handle:
                rows = [r for r in csv.DictReader(handle) if r["query_id"] == "0"]
            self.assertEqual(int(rows[0]["candidate_id"]), 11)
            ranked = np.array([int(r["candidate_id"]) for r in rows])
            metrics = independent_query_metrics(bundle.labels[0], bundle.labels[candidates], bundle.labels[ranked], settings["ks"])
            for family, filename in (("exact", "per_query.csv"), ("coverage", "coverage_per_query.csv")):
                with (out/"valid/no_local"/filename).open() as handle:
                    row = next(r for r in csv.DictReader(handle) if r["query_id"] == "0")
                for key, val in metrics[family].items():
                    if val is not None:
                        self.assertAlmostEqual(float(row[key]), val, places=14)
            index = read_json(out/"evaluation_index_valid.json")
            run_ids = []
            for name in scoring.specs("valid"):
                target = out/"valid"/name
                self.assertTrue(geo.completed(target, index["variant_signatures"][name]))
                self.assertEqual(file_hash(target/"completed.json"), index["completed_sha256"][name])
                identity = read_json(target/"run_identity.json")
                self.assertTrue(identity["started_at"].endswith("+08:00"))
                run_ids.append(identity["run_id"])
            self.assertEqual(len(set(run_ids)), 5)
            valid_before = snapshot(out/"valid")
            test_pool, test_bundle, test_values, _ = fixture("test")
            test_reports = pipeline.evaluate(out, "source_test", Components(), test_pool, test_bundle,
                                             settings["ks"], 2026, test_values, settings, device="cpu")
            self.assertIn("I_2", test_reports)
            self.assertEqual(valid_before, snapshot(out/"valid"))
            with (out/"comparison.csv").open() as handle:
                self.assertEqual({r["split"] for r in csv.DictReader(handle)}, {"valid", "test"})

    def test_interrupted_attempts_preserved_then_completed_reused(self):
        pool, bundle, values, settings = fixture()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            out = Path(tmp)/"ablation"
            with patch.object(geo, "evaluate_rankings_file", side_effect=InterruptedError("after exact")):
                with self.assertRaises(InterruptedError):
                    pipeline.evaluate(out, "source", Components(), pool, bundle, settings["ks"], 2026,
                                      values, settings, device="cpu")
            attempts = list((out/"valid").glob(".*attempt_*"))
            before = {str(p): snapshot(p) for p in attempts}
            identities = snapshot(out/"identities")
            pipeline.evaluate(out, "source", Components(), pool, bundle, settings["ks"], 2026,
                              values, settings, device="cpu")
            self.assertEqual(before, {str(p): snapshot(p) for p in attempts})
            self.assertEqual(identities, snapshot(out/"identities"))
            completed = {name: snapshot(out/"valid"/name) for name in scoring.specs("valid")}
            with patch.object(Components, "components", side_effect=AssertionError("completed should be reused")):
                pipeline.evaluate(out, "source", Components(), pool, bundle, settings["ks"], 2026,
                                  values, settings, device="cpu")
            self.assertEqual(completed, {name: snapshot(out/"valid"/name) for name in scoring.specs("valid")})
            with self.assertRaisesRegex(ValueError, "signature changed"):
                pipeline.evaluate(out, "changed_source", Components(), pool, bundle, settings["ks"], 2026,
                                  values, settings, device="cpu")
            (out/"valid/no_local/rankings.csv").write_text("corrupt")
            with self.assertRaises(ValueError):
                pipeline.evaluate(out, "source", Components(), pool, bundle, settings["ks"], 2026,
                                  values, settings, device="cpu")

    def test_reject_wrong_gallery_settings_and_unsafe_pool(self):
        pool, bundle, values, settings = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            wrong = {**values, "sample_ids": values["sample_ids"][:-1]}
            with self.assertRaisesRegex(ValueError, "exactly the complete gallery"):
                pipeline.evaluate(tmp, "sig", Components(), pool, bundle, settings["ks"], 2026,
                                  wrong, settings, device="cpu")
            with self.assertRaisesRegex(ValueError, "K/tie"):
                pipeline.evaluate(tmp, "sig", Components(), pool, bundle, [1], 2026,
                                  values, settings, device="cpu")
            with self.assertRaisesRegex(ValueError, "K/tie"):
                pipeline.evaluate(tmp, "sig", Components(), pool, bundle, settings["ks"], 42,
                                  values, settings, device="cpu")
            class UnsafePool:
                split_name = "valid"
                gallery = pool.gallery
                candidates_per_query = 11
                protocol_signature = "unsafe"
                def __len__(self): return 1
                def __iter__(self): return iter([0])
                def __getitem__(self, key): return np.arange(11)
            with self.assertRaisesRegex(ValueError, "entire gallery except self"):
                pipeline.evaluate(tmp, "sig", Components(), UnsafePool(), bundle, settings["ks"], 2026,
                                  values, settings, device="cpu")
            class DuplicatePool(UnsafePool):
                def __getitem__(self, key): return np.array(list(range(1, 11))+[10])
            with self.assertRaisesRegex(ValueError, "entire gallery except self"):
                pipeline.evaluate(tmp, "sig", Components(), DuplicatePool(), bundle, settings["ks"], 2026,
                                  values, settings, device="cpu")


if __name__ == "__main__":
    unittest.main()
