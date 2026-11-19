"""Synthetic full-gallery evaluation, immutable resume and parallel equivalence."""
from copy import deepcopy
import csv
import multiprocessing
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from fgw_grid_search import pipeline, scoring, verify
from wbm_retrieval.common import file_hash, read_json, write_json
from wbm_retrieval.exact_metrics import order_scores
from wbm_retrieval.retrieval_protocol import FullGalleryPool


class Provider:
    def __init__(self, folder, pool, fail=None):
        self.signature = "synthetic_complete_cache"
        self.marker = Path(folder) / "source_completed.json"
        if not self.marker.exists():
            write_json(self.marker, {"signature": self.signature})
        self.split = pool.split_name
        self.protocol_signature = pool.protocol_signature
        self.query_ids = np.asarray(list(pool))
        self.gallery_ids = pool.gallery
        self.calls = []
        self.fail = fail

    def scores(self, query, candidates):
        if query == self.fail:
            raise InterruptedError("synthetic source interruption")
        self.calls.append(int(query))
        if set(map(int, candidates)) != set(map(int, self.gallery_ids)) - {int(query)}:
            raise ValueError("Provider requires every candidate")
        return {"F": np.where(candidates == 502, 1., .01),
                "G": np.where(candidates == 502, 1., .01),
                "WBBS": np.where(candidates == 502, .25, .8)}


def fixture(folder, split="valid", all_specs=False):
    ids = np.arange(503, dtype=np.int64)
    labels = np.zeros((503, 8), dtype=int)
    labels[:, 0] = 1
    labels[1] = [0, 1, 0, 0, 0, 0, 0, 0]
    labels[2] = [1, 1, 0, 0, 0, 0, 0, 0]
    labels[3:250] = labels[1]
    _, classes = np.unique(labels, axis=0, return_inverse=True)
    bundle = SimpleNamespace(labels=labels, class_ids=classes, valid_ids=ids, reserved_ids=ids)
    pool = FullGalleryPool([0, 1, 2], ids, split, "synthetic_protocol")
    settings = deepcopy(scoring.SETTINGS)
    settings["denominator"] = 2
    variants = scoring.specs(settings)
    if not all_specs:
        variants = {name: spec for name, spec in variants.items()
                    if spec["family"] == "legacy_fgw" or
                    spec.get("ticks") == ([1, 1, 0] if spec["family"] == "sum_minmax" else [1, 0, 1])}
    return pool, bundle, settings, variants, Provider(folder, pool)


def snapshot(folder):
    return {str(p.relative_to(folder)): (file_hash(p), p.stat().st_mtime_ns)
            for p in Path(folder).rglob("*") if p.is_file()}


def evaluate(output, fixture_values, **kwargs):
    pool, bundle, settings, variants, provider = fixture_values
    return pipeline.evaluate(output, "parent_source", provider, pool, bundle, variants, settings, **kwargs)


class PipelineTests(unittest.TestCase):
    def test_full_candidates_and_both_metric_families_verified_before_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = fixture(tmp)
            pool, bundle, settings, variants, provider = values
            candidates = pool[0]
            old_scores = provider.scores(0, candidates)["WBBS"]
            old_top = candidates[order_scores(candidates, old_scores, 0, 2026)[:500]]
            self.assertNotIn(502, old_top)
            output = Path(tmp) / "run"
            index = evaluate(output, values)
            self.assertEqual(set(index["variants"]), set(variants))
            self.assertEqual(len(index["summary_rows"]), len(variants))
            for name in variants:
                folder = output / "valid" / name
                with (folder / "rankings.csv").open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 1500)
                self.assertEqual(int(rows[0]["candidate_id"]), 502)
                checked = read_json(folder / "verification.json")
                self.assertEqual(checked["status"], "verified")
                self.assertEqual(checked["full_candidates_checked"], 3 * 502)
                self.assertTrue(checked["all_queries_full_candidate_topk_exact"])
                self.assertNotIn("completion_sha256", checked)
                self.assertNotIn("verification.json", checked["files"])
                marker = read_json(folder / "completed.json")
                self.assertEqual(marker["files"]["verification.json"], file_hash(folder / "verification.json"))
                self.assertEqual(index["completed_sha256"][name], file_hash(folder / "completed.json"))
                identity = read_json(folder / "run_identity.json")
                self.assertTrue(identity["key"].startswith("fgw_grid_run_valid_"))
                self.assertTrue(identity["started_at"].endswith("+08:00"))
                for file in ("metrics.json", "coverage_metrics.json"):
                    report = read_json(folder / file)
                    self.assertEqual(report["ks"], settings["ks"])
                    self.assertEqual(report["metadata"]["signature"], index["variant_signatures"][name])
            for row in index["summary_rows"]:
                self.assertIsInstance(row["precision10_hits"], int)
                self.assertAlmostEqual(row["precision10_hits"] / 30, row["Precision@10"])

    def test_completed_resume_zero_writes_and_changed_source_or_settings_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = fixture(tmp)
            output = Path(tmp) / "run"
            index = evaluate(output, values)
            before = snapshot(output)
            with patch.object(values[-1], "scores", side_effect=AssertionError("complete resume must be read-only")):
                self.assertEqual(evaluate(output, values), index)
            self.assertEqual(snapshot(output), before)
            pool, bundle, settings, variants, provider = values
            with self.assertRaisesRegex(ValueError, "changed"):
                pipeline.evaluate(output, "new_source", provider, pool, bundle, variants, settings)
            changed = deepcopy(settings)
            changed["include_reference"] = False
            with self.assertRaisesRegex(ValueError, "changed"):
                pipeline.evaluate(output, "parent_source", provider, pool, bundle, variants, changed)
            write_json(provider.marker, {"signature": provider.signature, "changed": True})
            with self.assertRaisesRegex(ValueError, "changed"):
                evaluate(output, values)
            self.assertEqual(snapshot(output), before)

    def test_interrupted_attempt_and_identity_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = fixture(tmp)
            output = Path(tmp) / "run"
            values[-1].fail = 1
            with self.assertRaises(InterruptedError):
                evaluate(output, values)
            identities = snapshot(output / "identities")
            attempts = {str(p): snapshot(p) for p in (output / "valid").glob(".*attempt_*")}
            self.assertTrue(attempts)
            values[-1].fail = None
            evaluate(output, values)
            self.assertEqual({name: snapshot(output / "identities")[name] for name in identities}, identities)
            self.assertEqual({p: snapshot(Path(p)) for p in attempts}, attempts)

    def test_stage_guards_and_invalid_spec_precede_any_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = fixture(tmp)
            output = Path(tmp) / "run"
            with self.assertRaisesRegex(ValueError, "search requires valid"):
                evaluate(output, values, stage="test")
            test_values = fixture(tmp, "test")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                evaluate(output, test_values, stage="test")
            with self.assertRaisesRegex(ValueError, "search requires valid"):
                evaluate(output, test_values)
            pool, bundle, settings, variants, provider = values
            bad = {"invalid": {"family": "sum_raw", "weights": [0., .5, .5], "ticks": [0, 1, 1], "denominator": 2}}
            with self.assertRaises(ValueError):
                pipeline.evaluate(output, "parent_source", provider, pool, bundle, bad, settings)
            self.assertFalse(output.exists())
            p, b, s, variants, provider = test_values
            only = {"selected": next(iter(variants.values()))}
            index = pipeline.evaluate(output, "test_source", provider, p, b, only, s, stage="test")
            self.assertEqual(index["stage"], "test")
            self.assertEqual(len(index["summary_rows"]), 1)

    def test_completed_report_corruption_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = fixture(tmp)
            output = Path(tmp) / "run"
            evaluate(output, values)
            name = next(iter(values[3]))
            (output / "valid" / name / "rankings.csv").write_text("corrupt")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                evaluate(output, values)

    @unittest.skipUnless("fork" in multiprocessing.get_all_start_methods(), "fork required")
    def test_sequential_and_fork_results_are_numerically_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = fixture(tmp)
            one, two = Path(tmp) / "one", Path(tmp) / "two"
            a = evaluate(one, values)
            b = evaluate(two, values, workers=2)
            self.assertEqual(a["summary_rows"], b["summary_rows"])
            for name in values[3]:
                for filename in ("rankings.csv", "per_query.csv", "coverage_per_query.csv", "per_class.csv", "coverage_per_class.csv"):
                    self.assertEqual(file_hash(one / "valid" / name / filename), file_hash(two / "valid" / name / filename))
            before = snapshot(two)
            evaluate(two, values, workers=2)
            self.assertEqual(snapshot(two), before)


if __name__ == "__main__":
    unittest.main()
