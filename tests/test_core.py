"""Scientific/protocol invariants, runnable without training or Torch."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wbm_retrieval.common import ROOT, read_json, write_csv
from wbm_retrieval.data import Protocol, containment, sample_pairs, stratified_budget
from wbm_retrieval.evaluation import query_metrics
from wbm_retrieval.matching import greedy_assignment, map_similarity
from wbm_retrieval.methods.handcrafted import handcrafted_descriptor
from wbm_retrieval.regions import augment, extract_regions, token_patch


def fixture_config(root):
    root = Path(root)
    config = copy.deepcopy(read_json(ROOT / "configs/experiment_a.json"))
    config.update(data_file=str(root / "data.npz"), split_manifest=str(root / "split.csv"),
                  query_manifest=str(root / "query.csv"), candidate_manifest=str(root / "candidate.csv"),
                  output_dir=str(root / "output"), device="cpu", num_workers=0, batch_size=2,
                  inference_batch_size=4, input_size=32, embedding_dim=16, expected_candidates=3)
    config["proposal"].update(min_area=2, top_k=3, padding=2)
    config["evaluation"].update(bootstrap_samples=10, save_matches_top_k=2)
    config["geometry"]["examples"] = 1
    labels = np.array([[1, 0], [0, 1], [1, 1]] * 8, dtype=np.uint8)
    maps = np.ones((24, 16, 16), dtype=np.uint8)
    for i, label in enumerate(labels):
        if label[0]:
            maps[i, 3:5, 3:7] = 2
        if label[1]:
            maps[i, 9:13, 10:12] = 2
    np.savez(root / "data.npz", maps, labels)
    rows = []
    for i in range(24):
        split = "train" if i < 8 else "valid" if i < 16 else "test"
        rows.append({"sample_id": i, "original_index": i, "valid_index": i, "split": split,
                     "label_0": int(labels[i, 0]), "label_1": int(labels[i, 1])})
    write_csv(config["split_manifest"], rows, list(rows[0]))
    write_csv(config["query_manifest"], [{"sample_id": 16}, {"sample_id": 17}], ["sample_id"])
    edges = [{"query_id": q, "candidate_id": c, "jaccard": "deliberately ignored"} for q in [16, 17] for c in [18, 19, 20]]
    write_csv(config["candidate_manifest"], edges, list(edges[0]))
    return config, labels


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.matching = read_json(ROOT / "configs/experiment_a.json")["matching"]

    def test_containment_direction_and_extra_regions(self):
        abc, abcd = [1, 1, 1, 0], [1, 1, 1, 1]
        self.assertEqual(containment(abc, abcd), 1)
        self.assertEqual(containment(abcd, abc), .75)
        self.assertAlmostEqual(containment(abc, [0, 1, 1, 1]), 2/3)
        with self.assertRaises(ValueError):
            containment([0, 0], [1, 1])

    def test_eight_connectivity_area_and_top_k(self):
        raw = np.ones((10, 10), dtype=np.uint8)
        raw[1, 1] = raw[2, 2] = 2
        raw[6:8, 6:8] = 2
        raw[0, 9] = 2
        regions = extract_regions(raw, min_area=2, top_k=2)
        self.assertEqual([r["area"] for r in regions], [4, 2])
        self.assertEqual(len(extract_regions(raw, min_area=2, top_k=1)), 1)

    def test_patch_translation_and_handcrafted_rotation(self):
        raw = np.zeros((24, 24), dtype=np.uint8)
        raw[3:5, 3:10] = 1
        raw[5:10, 3:5] = 1
        shifted = np.roll(raw, (5, 4), axis=(0, 1))
        np.testing.assert_array_equal(token_patch(raw, 32, 3), token_patch(shifted, 32, 3))
        np.testing.assert_allclose(handcrafted_descriptor(raw), handcrafted_descriptor(np.rot90(raw)), atol=1e-6)

    def tokens(self, descriptors):
        count = len(descriptors)
        return {"embedding": np.asarray(descriptors, dtype=np.float32), "area": np.ones(count),
                "position": np.zeros((count, 2)), "scale": np.ones(count)}

    def test_one_to_one_and_query_denominator(self):
        q = self.tokens([[1, 0], [0, 1]])
        c = self.tokens([[1, 0], [0, 1], [-1, 0]])
        self.assertAlmostEqual(map_similarity(q, c, self.matching), 1)
        self.assertAlmostEqual(map_similarity(q, self.tokens([[1, 0]]), self.matching), .5)
        _, matches = map_similarity(q, c, self.matching, details=True)
        self.assertEqual(len({m["candidate_token"] for m in matches}), len(matches))
        self.assertEqual(map_similarity(q, self.tokens([]), self.matching), 0)

    def test_greedy_is_not_hungarian(self):
        matches = greedy_assignment(np.array([[.9, .89], [.88, 0]]))
        self.assertEqual(matches, [(0, 0, .9)])
        self.assertEqual(greedy_assignment(np.array([[.4]]), minimum=.5), [])

    def test_geometry_is_monotone_and_gated(self):
        q, c = self.tokens([[1, 0]]), self.tokens([[1, 0]])
        settings = {**self.matching, "lambda_pos": .125, "lambda_scale": .125}
        self.assertAlmostEqual(map_similarity(q, c, settings), 1)
        c["position"][0, 0] = 1
        self.assertLess(map_similarity(q, c, settings), 1)
        self.assertAlmostEqual(map_similarity(q, c, self.matching), 1)
        c["embedding"][0] = [0, 1]
        self.assertEqual(map_similarity(q, c, {**settings, "shape_threshold": .5}), 0)

    def test_pca_scale_rotation_invariance(self):
        raw = np.ones((24, 24), dtype=np.uint8)
        raw[5:7, 8:16] = 2
        a = extract_regions(raw)[0]
        b = extract_regions(np.rot90(raw))[0]
        self.assertGreater(a["scale"], 0)
        self.assertAlmostEqual(a["scale"], b["scale"])

    def test_perfect_metrics_and_missing_positive_queries(self):
        rel = np.array([1, 1, .5, 0])
        values = query_metrics(rel, rel, [1, 2, 10])
        self.assertEqual(values["ContainmentNDCG@2"], 1)
        self.assertEqual(values["FullCoverageRecall@2"], 1)
        self.assertEqual(values["mAP_full"], 1)
        self.assertIsNone(query_metrics([.5, 0], [.5, 0], [1])["mAP_full"])

    def test_protocol_ignores_candidate_jaccard_and_rejects_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            config, labels = fixture_config(directory)
            protocol = Protocol(config, labels)
            self.assertEqual(protocol.summary()["num_queries"], 2)
            write_csv(config["candidate_manifest"], [{"query_id": 16, "candidate_id": 0}], ["query_id", "candidate_id"])
            with self.assertRaises(ValueError):
                Protocol(config, labels)

    def test_nested_budget_and_pairs_stay_inside_selected_ids(self):
        labels = np.array([[1, 0], [0, 1], [1, 1]] * 20)
        small = stratified_budget(range(60), labels, .1, 2026)
        large = stratified_budget(range(60), labels, .5, 2026)
        self.assertLessEqual(set(small), set(large))
        pairs, targets = sample_pairs(small, labels, 100, 2026)
        self.assertLessEqual(set(pairs.ravel()), set(small))
        self.assertTrue((pairs[:, 0] != pairs[:, 1]).all())
        for (q, c), target in zip(pairs, targets):
            self.assertAlmostEqual(float(target), containment(labels[q], labels[c]))


if __name__ == "__main__":
    unittest.main()
