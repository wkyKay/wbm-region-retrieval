"""Analytic fixtures for independent representation geometry diagnostics."""
import copy
import json
import unittest
import warnings

import numpy as np

from f_representation_audit.geometry import audit_geometry, paired_bootstrap


class GeometryTests(unittest.TestCase):
    def test_negative_cosine_is_preserved_and_rank_is_correct(self):
        result = audit_geometry([[1., 0.], [1., 0.], [-1., 0.], [-1., 0.]],
                                [0, 0, 1, 1], [1, 2, 3, 4])
        row = result["rows"][0]
        self.assertEqual(row["intra_mean_cosine"], 1.)
        self.assertEqual(row["inter_mean_cosine"], -1.)
        self.assertEqual(row["cosine_margin"], 2.)
        self.assertEqual(row["nearest_same_rank"], 1)
        self.assertEqual(row["hit_at_1"], 1.)
        self.assertEqual(row["precision_at_5"], 1 / 3)
        self.assertEqual(row["effective_k_5"], 3)
        self.assertEqual(result["summary"]["knn_unit"], "map")
        json.dumps(result, allow_nan=False)

    def test_excludes_all_tokens_of_same_map_and_uses_map_balanced_inter(self):
        # Both first tokens belong to map 1. Map 3 has three token votes, while
        # map 4 has one. The balanced inter mean must give their classes 1/2.
        x = [[1., 0.], [1., 0.], [0., 1.], [-1., 0.], [-1., 0.], [-1., 0.], [1., 0.]]
        result = audit_geometry(x, [0, 0, 0, 1, 1, 1, 2], [1, 1, 2, 3, 3, 3, 4])
        row = result["rows"][0]
        self.assertEqual(row["candidate_count"], 5)
        self.assertEqual(row["same_class_candidates"], 1)
        self.assertEqual(row["intra_mean_cosine"], 0.)
        self.assertEqual(row["inter_mean_cosine"], -.5)
        self.assertEqual(row["balanced_inter_mean_cosine"], 0.)
        self.assertEqual(row["nearest_same_rank"], 2)
        self.assertEqual(result["summary"]["knn_unit"], "token")
        self.assertEqual([r["token_index"] for r in result["rows"]], [0, 1, 0, 0, 1, 2, 0])

    def test_summary_balances_query_maps_then_classes(self):
        result = audit_geometry([[1., 0.], [0., 1.], [1., 0.], [-1., 0.], [0., 1.]],
                                [0, 0, 0, 1, 1], [1, 1, 2, 3, 4])
        rows = result["rows"]
        expected_class0 = ((rows[0]["cosine_margin"] + rows[1]["cosine_margin"]) / 2
                           + rows[2]["cosine_margin"]) / 2
        expected_class1 = (rows[3]["cosine_margin"] + rows[4]["cosine_margin"]) / 2
        self.assertAlmostEqual(result["summary"]["class_balanced"]["metrics"]["cosine_margin"],
                               (expected_class0 + expected_class1) / 2)
        self.assertNotAlmostEqual(result["summary"]["query_macro"]["metrics"]["cosine_margin"],
                                  (expected_class0 + expected_class1) / 2)

    def test_missing_positive_and_empty_gallery_are_json_null(self):
        result = audit_geometry([[1., 0.], [-1., 0.]], [0, 1], [1, 2])
        for row in result["rows"]:
            for name in ("intra_mean_cosine", "cosine_margin", "nearest_same_rank", "hit_at_1", "precision_at_10"):
                self.assertIsNone(row[name])
        single = audit_geometry([[1., 0.]], [0], [1])
        self.assertEqual(single["rows"][0]["candidate_count"], 0)
        self.assertIsNone(single["rows"][0]["inter_mean_cosine"])
        self.assertEqual(single["summary"]["class_balanced"]["valid_classes"]["cosine_margin"], 0)
        json.dumps(single, allow_nan=False)
        effect = paired_bootstrap(single, single, repetitions=5)
        self.assertIsNone(effect["effects"]["cosine_margin"]["difference"])
        json.dumps(effect, allow_nan=False)

    def test_ties_and_chunk_size_are_deterministic(self):
        x = np.ones((4, 2))
        first = audit_geometry(x, [0, 0, 1, 1], [4, 3, 1, 2], chunk_size=1)
        second = audit_geometry(x, [0, 0, 1, 1], [4, 3, 1, 2], chunk_size=3)
        self.assertEqual(first, second)
        self.assertEqual(first["rows"][0]["nearest_same_rank"], 3)
        self.assertEqual(first["rows"][0]["hit_at_1"], 0.)

    def test_scaling_features_does_not_change_cosine_metrics(self):
        x = np.asarray([[1., 0.], [1., 1.], [-1., 0.], [-1., 1.]])
        a = audit_geometry(x, [0, 0, 1, 1], [1, 2, 3, 4])
        b = audit_geometry(x * np.array([2., 5., 3., 7.])[:, None], [0, 0, 1, 1], [1, 2, 3, 4])
        for name in a["summary"]["metric_names"]:
            self.assertAlmostEqual(a["summary"]["class_balanced"]["metrics"][name],
                                   b["summary"]["class_balanced"]["metrics"][name])

    def test_string_labels_are_json_safe_and_absolute_cosine_is_descriptive(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            result = audit_geometry([[1., 0.], [1., 1.]], np.asarray([b"center", b"center"]), [1, 2])
        self.assertEqual(result["rows"][0]["label"], "center")
        self.assertEqual(result["summary"]["metric_directions"]["intra_mean_cosine"], "descriptive_only")
        self.assertEqual(result["summary"]["metric_directions"]["cosine_margin"], "higher")
        json.dumps(result, allow_nan=False)
        with self.assertRaises(ValueError):
            audit_geometry([[1., 0.]], np.asarray([b""]), [1])

    def test_invalid_inputs_fail_instead_of_inventing_geometry(self):
        for x, labels, ids in (([[0., 0.]], [0], [1]), ([[float("nan"), 0.]], [0], [1]),
                               ([[1., 0.], [1., 0.]], [0, 1], [1, 1]),
                               ([[1., 0.]], [[1, 0]], [1])):
            with self.subTest(x=x, labels=labels):
                with self.assertRaises(ValueError):
                    audit_geometry(x, labels, ids)
        with self.assertRaises(ValueError):
            audit_geometry([[1., 0.]], [0], [1], ks=(1, 1))

    def test_paired_bootstrap_identity_and_reproducibility(self):
        x = [[1., 0.], [1., 1.], [-1., 0.], [-1., 1.]]
        result = audit_geometry(x, [0, 0, 1, 1], [1, 2, 3, 4])
        effect = paired_bootstrap(result, result, repetitions=99)
        self.assertEqual(effect, paired_bootstrap(result, result, repetitions=99))
        for value in effect["effects"].values():
            self.assertEqual(value["difference"], 0.)
            self.assertEqual(value["ci_low"], 0.)
            self.assertEqual(value["ci_high"], 0.)
            self.assertEqual(value["valid_maps"], 4)

    def test_absolute_intra_decline_can_coexist_with_better_margin_and_knn(self):
        # Initially every vector is identical, so similarity is high but has no
        # class information. Training separates the two class cones while each
        # class retains finite variation: intra falls and margin improves.
        ids = [4, 3, 1, 2]
        before = audit_geometry(np.ones((4, 2)), [0, 0, 1, 1], ids)
        after = audit_geometry([[1., .3], [1., -.3], [-1., .3], [-1., -.3]], [0, 0, 1, 1], ids)
        effect = paired_bootstrap(before, after, repetitions=100)
        self.assertLess(effect["effects"]["intra_mean_cosine"]["difference"], 0.)
        self.assertGreater(effect["effects"]["cosine_margin"]["difference"], 0.)
        self.assertGreater(effect["effects"]["hit_at_1"]["difference"], 0.)
        self.assertLess(effect["effects"]["nearest_same_rank"]["difference"], 0.)

    def test_bootstrap_pairs_rows_by_map_and_token_not_list_position(self):
        result = audit_geometry([[1., 0.], [1., 1.], [-1., 0.], [-1., 1.]],
                                [0, 0, 1, 1], [1, 2, 3, 4])
        reordered = copy.deepcopy(result)
        reordered["rows"].reverse()
        self.assertEqual(paired_bootstrap(result, result, repetitions=10),
                         paired_bootstrap(result, reordered, repetitions=10))
        changed = copy.deepcopy(result)
        changed["rows"][0]["label"] = 9
        with self.assertRaises(ValueError):
            paired_bootstrap(result, changed, repetitions=10)

    def test_bootstrap_resamples_maps_not_individual_tokens(self):
        # Duplicating all tokens of map 1 leaves its within-map mean unchanged.
        # For paired metrics copied from identical-map queries, the complete CI
        # must be unchanged too; a token bootstrap would narrow/reweight it.
        result = audit_geometry([[1., 0.], [1., 1.], [-1., 0.], [-1., 1.]],
                                [0, 0, 1, 1], [1, 2, 3, 4])
        candidate = copy.deepcopy(result)
        candidate["rows"][0]["cosine_margin"] += .4
        candidate["rows"][1]["cosine_margin"] -= .2
        before = paired_bootstrap(result, candidate, repetitions=300)
        duplicates = []
        for audit in (result, candidate):
            audit = copy.deepcopy(audit)
            row = copy.deepcopy(audit["rows"][0])
            row["token_index"] = 1
            audit["rows"].append(row)
            duplicates.append(audit)
        after = paired_bootstrap(*duplicates, repetitions=300)
        for key in ("difference", "ci_low", "ci_high", "valid_maps"):
            self.assertEqual(before["effects"]["cosine_margin"][key],
                             after["effects"]["cosine_margin"][key])
        self.assertEqual(after["effects"]["cosine_margin"]["valid_queries"], 5)


if __name__ == "__main__":
    unittest.main()
