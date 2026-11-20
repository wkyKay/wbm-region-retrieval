"""Six-way full-candidate replay and independent coverage denominator checks."""
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.proposed.f_segmentation_geometry import coverage_metrics
from fgw_cross_test.tests.test_verify import Pool, Provider, rewrite_csv, write_json
from fgj_cross_test import verify
from wbm_retrieval.exact_metrics import order_scores


SPECS = {"GJ_ADD": {"formula": "0.5*G+0.5*J"}, "FJ_ADD": {"formula": "0.5*F+0.5*J"},
         "F_ONLY": {"formula": "F"}, "G_ONLY": {"formula": "G"},
         "J_ONLY": {"formula": "J"}, "WBBS": {"formula": "WBBS"}}


class Components(Provider):
    def __init__(self, root, pool):
        super().__init__(root, pool)
        q, c = np.meshgrid(self.query_ids, self.gallery_ids, indexing="ij")
        self.values["joint_distance"] = ((q * 7 + c * 3) % 19) / 20.
        self.calls = []

    def scores(self, query, candidates):
        self.calls.append(int(query))
        return super().scores(query, candidates)


def fixture(root, *, ties=False, wrong_formula=None):
    root = Path(root)
    labels = np.asarray([[1,1,1,0,0], [1,1,1,0,0], [1,1,1,1,0], [1,0,1,0,0],
                         [0,1,1,0,0], [0,0,1,0,0], [0,0,0,1,0], [1,0,0,0,0],
                         [0,1,0,0,0], [1,1,0,0,0], [0,0,0,0,1], [1,1,0,0,0],
                         [0,0,1,1,0]], dtype=int)
    labels = np.pad(labels, ((0,0),(0,3)))
    taxonomy = {pattern: i for i, pattern in enumerate(sorted(set(map(tuple, labels))))}
    bundle = SimpleNamespace(labels=labels, class_ids=np.asarray([taxonomy[tuple(row)] for row in labels]))
    pool, ks = Pool(labels), [1,3,5]
    provider = Components(root, pool)
    if ties:
        for values in provider.values.values():values[:] = .4
    folders, signatures = {}, {}
    binding = {str(provider.marker.resolve()):verify._sha(provider.marker)}
    for name, spec in SPECS.items():
        folder = root / name; folder.mkdir(); folders[name] = folder
        signatures[name] = "signature-" + name
        identity = dict(run_id="20261120T010203+0800__"+name, started_at="2026-11-20T01:02:03+08:00",
                        evaluation_signature=signatures[name], variant=name, split="test")
        manifest = dict(**identity, signature=signatures[name], context_signature="context",
                        spec=spec, ks=ks, tie_seed=2026, num_queries=len(pool), gallery_count=len(pool.gallery),
                        candidate_rule=verify.RULE, derived_protocol=pool.metadata,
                        retrieval_protocol_signature=pool.protocol_signature, formal_protocol=False)
        write_json(folder / "run_identity.json", identity)
        write_json(folder / "evaluation_manifest.json", manifest)
        with (folder / "rankings.csv").open("w", newline="") as stream:
            writer = csv.writer(stream); writer.writerow(verify.RANKING_FIELDS)
            for q in pool:
                candidates = pool[q]; values = provider.scores(q, candidates)
                j = np.exp(-2 * values["joint_distance"])
                choices = {"GJ_ADD": (values["G"]+j)/2, "FJ_ADD": (values["F"]+j)/2,
                           "F_ONLY": values["F"], "G_ONLY": values["G"], "J_ONLY":j, "WBBS":values["WBBS"]}
                scores = choices[name]
                if wrong_formula and name == "GJ_ADD":
                    if wrong_formula == "product":scores = values["G"] * j
                    if wrong_formula == "scale":scores = values["G"] + j
                    if wrong_formula == "exponent":scores = (values["G"] + np.exp(-values["joint_distance"]))/2
                    if wrong_formula == "F":scores = choices["FJ_ADD"]
                order = order_scores(candidates, scores, q, 2026)[:max(ks)]
                for rank, i in enumerate(order, 1):
                    writer.writerow((q, rank, int(candidates[i]), scores[i], values["F"][i], values["G"][i],
                                     values["joint_distance"][i], values["WBBS"][i]))
        coverage_metrics.evaluate_rankings_file(bundle, pool, folder / "rankings.csv", ks, folder,
                                               split_name="test", metadata=manifest)
    provider.calls.clear()
    return folders, pool, bundle, ks, signatures, provider, binding


def check(values):
    folders, pool, bundle, ks, signatures, provider, binding = values
    return verify.verify_evaluations(folders, pool, bundle, ks, signatures, provider.scores, binding)


class BatchVerificationTests(unittest.TestCase):
    def test_six_variants_share_one_component_and_tie_pass_per_query(self):
        with tempfile.TemporaryDirectory() as root:
            values = fixture(root)
            with patch.object(verify, "_tie_keys", wraps=verify._tie_keys) as ties, patch.object(
                    coverage_metrics, "query_metrics", side_effect=AssertionError("production metrics called")):
                result = check(values)
            self.assertEqual(values[5].calls, list(values[1]))
            self.assertEqual(ties.call_count, len(values[1]))
            self.assertEqual(result["shared"]["components_calls"], 13)
            self.assertEqual(set(result["variants"]), set(SPECS))
            self.assertEqual(result["shared"]["full_candidates_checked"], sum(len(values[1][q]) for q in values[1]))
            for name, receipt in result["variants"].items():
                self.assertEqual(receipt["status"], "verified")
                self.assertEqual(receipt["spec"], SPECS[name])
                self.assertEqual(receipt["metric_query_counts"]["Coverage-nDCG@1"], 12)
                self.assertEqual(receipt["ranking_rows"], 65)
                self.assertEqual(set(receipt["files"]), set(verify.REPORT_FILES))
                self.assertEqual(receipt["max_saved_score_absolute_error"], 0.)
                self.assertLess(receipt["max_metric_absolute_error"], 1e-12)

    def test_formula_endpoints_and_exponent_are_independent_and_do_not_mutate_inputs(self):
        values = {"F":np.array([0.,1.]), "G":np.array([1.,0.]),
                  "joint_distance":np.array([0.,1.]), "WBBS":np.array([.3,.7])}
        before = {k:v.copy() for k,v in values.items()}
        j = np.array([1., np.exp(-2.)])
        expected = {"GJ_ADD":(values["G"]+j)/2, "FJ_ADD":(values["F"]+j)/2,
                    "F_ONLY":values["F"], "G_ONLY":values["G"], "J_ONLY":j, "WBBS":values["WBBS"]}
        for name, spec in SPECS.items():
            np.testing.assert_array_equal(verify.independent_score(values, spec), expected[name])
        for key in values:np.testing.assert_array_equal(values[key], before[key])
        for wrong in ({"formula":"G*J"}, {"formula":"0.5*G+0.5*J", "alpha":.5}):
            with self.assertRaises(ValueError):verify.independent_score(values, wrong)

    def test_wrong_product_scale_exponent_or_local_substitution_is_rejected(self):
        for wrong in ("product", "scale", "exponent", "F"):
            with self.subTest(wrong=wrong), tempfile.TemporaryDirectory() as root:
                with self.assertRaisesRegex(ValueError, "ordering|formula/component"):
                    check(fixture(root, wrong_formula=wrong))

    def test_unsaved_candidate_score_change_is_detected_by_complete_pool_replay(self):
        with tempfile.TemporaryDirectory() as root:
            values = fixture(root); folders, pool, _, _, _, provider, _ = values
            with (folders["GJ_ADD"] / "rankings.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            returned = {int(r["candidate_id"]) for r in rows if int(r["query_id"]) == 0}
            absent = next(int(c) for c in pool[0] if int(c) not in returned)
            provider.values["G"][0, absent] = 1.
            provider.values["joint_distance"][0, absent] = 0.
            with self.assertRaisesRegex(ValueError, "full-candidate ordering"):check(values)

    def test_tie_hash_order_is_checked_for_every_variant(self):
        with tempfile.TemporaryDirectory() as root:
            values = fixture(root, ties=True); check(values)
            def swap(rows):
                for field in verify.RANKING_FIELDS[2:]:rows[0][field], rows[1][field] = rows[1][field], rows[0][field]
            rewrite_csv(values[0]["WBBS"] / "rankings.csv", swap)
            with self.assertRaisesRegex(ValueError, "full-candidate ordering"):check(values)

    def test_saved_component_tampering_is_rejected_even_if_variant_does_not_use_it(self):
        with tempfile.TemporaryDirectory() as root:
            values = fixture(root)
            rewrite_csv(values[0]["G_ONLY"] / "rankings.csv", lambda rows:rows[0].update(optimal_f_score=.0123))
            with self.assertRaisesRegex(ValueError, "component"):check(values)

    def test_undefined_metric_cannot_be_replaced_by_zero(self):
        with tempfile.TemporaryDirectory() as root:
            values = fixture(root)
            def alter(rows):
                row = next(r for r in rows if int(r["query_id"]) == 10)
                self.assertEqual(row["Coverage-nDCG@1"], "")
                row["Coverage-nDCG@1"] = 0.
            rewrite_csv(values[0]["J_ONLY"] / "coverage_per_query.csv", alter)
            with self.assertRaisesRegex(ValueError, "Undefined metric"):check(values)

    def test_metric_counts_groups_and_macro_are_independently_checked(self):
        for changed in ("count", "group", "macro", "extra"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as root:
                values = fixture(root); path = values[0]["FJ_ADD"] / "coverage_metrics.json"
                report = verify._json(path)
                if changed == "count":report["metric_query_counts"]["FullCoverageRecall@1"] += 1
                if changed == "group":report["groups"]["mixed_label"]["num_queries"] += 1
                if changed == "macro":report["class_macro_metrics"]["MeanCoverage@1"] += .01
                if changed == "extra":report["metrics"]["Unrequested@1"] = .1
                write_json(path, report)
                with self.assertRaises(ValueError):check(values)

    def test_same_combination_candidate_is_rejected_before_component_scoring(self):
        with tempfile.TemporaryDirectory() as root:
            values = fixture(root); pool = values[1]; original = type(pool).__getitem__
            def incorrect(self, query):
                candidates = original(self, query)
                return np.sort(np.r_[candidates, 1]) if query == 0 else candidates
            with patch.object(type(pool), "__getitem__", incorrect):
                with self.assertRaisesRegex(ValueError, "label exclusion"):check(values)
            self.assertEqual(values[5].calls, [])

    def test_empty_relative_or_invalid_source_bindings_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            values = fixture(root)
            for binding in ({}, {"relative":"0"*64}, {str(Path(root)/"source"):"bad"}):
                with self.assertRaisesRegex(ValueError, "Source binding"):
                    check((*values[:-1], binding))

    def test_missing_variant_extra_exact_report_or_wrong_manifest_is_rejected(self):
        for mistake in ("variant", "exact", "manifest", "columns"):
            with self.subTest(mistake=mistake), tempfile.TemporaryDirectory() as root:
                values = fixture(root); folder = values[0]["GJ_ADD"]
                if mistake == "variant":values[0].pop("WBBS")
                if mistake == "exact":write_json(folder / "metrics.json", {})
                if mistake == "manifest":
                    path = folder / "evaluation_manifest.json"; data = verify._json(path)
                    data["spec"] = {"formula":"G*J"}; write_json(path, data)
                if mistake == "columns":
                    path = folder / "rankings.csv"; path.write_text(path.read_text().replace("joint_distance", "wrong_distance", 1))
                with self.assertRaises(ValueError):check(values)

    def test_invalid_components_and_class_identity_are_rejected(self):
        for issue in ("nan", "negative", "shape", "class"):
            with self.subTest(issue=issue), tempfile.TemporaryDirectory() as root:
                values = fixture(root)
                if issue == "class":values[2].class_ids[1] += 100
                elif issue == "shape":values[5].values["F"] = np.zeros((13,13,1))
                else:values[5].values["joint_distance"][0,:] = np.nan if issue == "nan" else -.1
                with self.assertRaises(ValueError):check(values)

    def test_reports_changed_during_replay_cannot_receive_verified_receipt(self):
        with tempfile.TemporaryDirectory() as root:
            values = fixture(root); provider = values[5]; original = provider.scores
            path = values[0]["WBBS"] / "run_identity.json"
            def mutation(query, candidates):
                if query == 0:
                    data = verify._json(path); data["later_change"] = True; write_json(path, data)
                return original(query, candidates)
            provider.scores = mutation
            with self.assertRaisesRegex(ValueError, "Reports changed"):check(values)


if __name__ == "__main__":
    unittest.main()
