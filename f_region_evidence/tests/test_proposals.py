"""Pixel-graph oracle, frozen-token identity and isolated cache integration."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.proposed.f_segmentation.proposals import segment
from experiments.proposed.f_segmentation.tensor_voting import tensor_vote
from f_region_structure.descriptors import describe_regions
from b1_structure_fusion.descriptors import load_settings
from wbm_retrieval.common import file_hash, read_json
from wbm_retrieval.diagnostics import normalized
from f_region_evidence import prepare as preparation
from f_region_evidence.proposals import PROPOSALS, propose, propose_all, residual_groups


def wafer():
    raw = np.ones((48, 48), dtype=np.uint8)
    for row, col in ((3, 3), (3, 17), (17, 3), (17, 17)):
        raw[row:row+3, col:col+3] = 2
    raw[40, 30:36] = 2
    raw[33, 30:35] = 2
    for row, col in ((31, 5), (31, 7), (31, 9), (33, 5), (33, 7)):
        raw[row, col] = 2
    labels, _ = segment(raw, "cc8", 5, 4)
    return raw, labels


def graph_oracle(mask):
    """Independent exhaustive Euclidean point graph, deliberately quadratic."""
    todo = set(map(tuple, np.argwhere(mask)))
    groups = []
    while todo:
        first = min(todo)
        todo.remove(first)
        group, frontier = {first}, [first]
        while frontier:
            y, x = frontier.pop()
            neighbours = {p for p in todo if (p[0]-y)**2 + (p[1]-x)**2 <= 8}
            todo -= neighbours
            group |= neighbours
            frontier.extend(neighbours)
        groups.append(group)
    return {frozenset(g) for g in groups}


def source_fixture():
    raw, labels = wafer()
    empty = np.ones(raw.shape, dtype=np.uint8)
    empty[4, 4] = 2
    maps = np.stack((raw, empty))
    old_labels = np.stack((labels, np.zeros_like(labels)))
    parts = [describe_regions(maps[i], old_labels[i]) for i in range(2)]
    cache = {key: np.concatenate([p[key] for p in parts]) for key in preparation.DESCRIPTOR_FIELDS}
    h = normalized(np.random.default_rng(2026).normal(size=(4, 512)))
    cache.update(features=h, area=cache["original_count"], histograms=cache["radial_hist16"],
                 map_ids=np.array([0, 1]), offsets=np.array([0, 4, 4]), owners=np.zeros(4, dtype=np.int64))
    config = {"input_size": 96, "embedding_dim": 512, "inference_batch_size": 3,
              "proposal": {"min_area": 5, "top_k": 4, "padding": 4}}
    suite = {"variants": {"F": {"window_size": 52, "input_channels": 3}}}
    protocol = SimpleNamespace(ids={"valid": np.array([0, 1]), "test": np.array([0, 1])})
    return maps, old_labels, cache, config, suite, protocol


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, x):
        self.seen.append(x.detach().cpu().numpy())
        first = x.mean((2, 3))
        return torch.cat((first, torch.ones((len(x), 509), device=x.device)), dim=1)


class ProposalTests(unittest.TestCase):
    def test_base_and_top8_match_independent_original_segmentation(self):
        raw, old = wafer()
        baseline, _ = propose(raw, old, "base4")
        top8, audit = propose(raw, old, "top8")
        expected, _ = segment(raw, "cc8", 5, 8)
        self.assertEqual(baseline.tobytes(), old.tobytes())
        np.testing.assert_array_equal(top8, expected)
        np.testing.assert_array_equal(top8[old > 0], old[old > 0])
        self.assertEqual(audit["additional_regions"], 2)
        self.assertEqual(audit["added_pixels"], 11)

    def test_residual_restores_small_ccs_without_normal_pixels(self):
        raw, old = wafer()
        labels, audit = propose(raw, old, "residual8")
        self.assertEqual(audit["additional_regions"], 3)
        self.assertEqual(audit["added_pixels"], 16)
        self.assertEqual(audit["remaining_pixels"], 0)
        self.assertEqual(int(labels.max()), 7)
        self.assertFalse(np.any((labels > 0) & (raw != 2)))
        np.testing.assert_array_equal(labels[old > 0], old[old > 0])
        multi = [row for row in audit["region_trace"] if len(row["original_cc_ids"]) > 1]
        self.assertEqual(len(multi), 1)
        self.assertEqual(multi[0]["original_cc_areas"], [1] * 5)

    def test_trace_reconstructs_every_mask_and_has_no_overlap(self):
        raw, old = wafer()
        for name, (labels, audit) in propose_all(raw, old).items():
            restored = np.zeros(raw.size, dtype=np.uint8)
            for row in audit["region_trace"]:
                ids = row["flat_indices"]
                self.assertFalse(np.any(restored[ids]))
                self.assertEqual(len(ids), row["area"])
                self.assertEqual(min(ids), row["first_flat_index"])
                restored[ids] = row["token"]
            np.testing.assert_array_equal(restored.reshape(raw.shape), labels, err_msg=name)

    def test_graph_exact_boundary_and_transitive_closure(self):
        mask = np.zeros((12, 12), dtype=bool)
        for point in ((1, 1), (3, 3), (5, 5), (5, 8)):
            mask[point] = True
        groups = residual_groups(mask)
        self.assertEqual([int(m.sum()) for m in groups], [3, 1])
        self.assertEqual(sum(int(m.sum()) for m in groups), int(mask.sum()))
        self.assertFalse(any(m[2, 2] for m in groups), "Graph connections never fill gaps")

    def test_graph_matches_independent_oracle_and_tie_order(self):
        rng = np.random.default_rng(77)
        for density in (.03, .12, .4, .9):
            for _ in range(4):
                mask = rng.random((9, 11)) < density
                actual = residual_groups(mask)
                self.assertEqual({frozenset(map(tuple, np.argwhere(m))) for m in actual}, graph_oracle(mask))
                keys = [(-int(m.sum()), int(np.flatnonzero(m)[0])) for m in actual]
                self.assertEqual(keys, sorted(keys))

    def test_at_most_four_additions_and_area_ties_are_row_major(self):
        raw = np.ones((45, 45), dtype=np.uint8)
        for row in range(3, 43, 4):
            raw[row, 3:8] = 2
        old, _ = segment(raw, "cc8", 5, 4)
        for name in ("top8", "residual8"):
            labels, audit = propose(raw, old, name)
            self.assertEqual(audit["regions"], 8)
            self.assertEqual(audit["remaining_pixels"], 10)
            self.assertTrue(np.all(labels[31, 3:8] == 8))

    def test_empty_and_fewer_than_four_regions_remain_valid(self):
        for count in (0, 1, 3):
            raw = np.ones((15, 15), dtype=np.uint8)
            raw[3, 3:3+count] = 2
            old, _ = segment(raw, "cc8", 5, 4)
            for labels, audit in propose_all(raw, old).values():
                self.assertEqual(labels.max(), 0)
                self.assertEqual(audit["retained_fraction"], 0.)
                self.assertEqual(audit["original_pixels"], count)

    def test_additions_start_after_actual_base_count(self):
        raw = np.ones((25, 25), dtype=np.uint8)
        raw[3:6, 3:6] = 2
        for point in ((14, 14), (14, 16), (14, 18), (16, 14), (16, 16)):
            raw[point] = 2
        old, _ = segment(raw, "cc8", 5, 4)
        labels, audit = propose(raw, old, "residual8")
        self.assertEqual(audit["base_regions"], 1)
        self.assertEqual(audit["regions"], 2)
        self.assertEqual(labels[14, 14], 2)

    def test_wrong_original_masks_and_parameters_rejected(self):
        raw, old = wafer()
        bad = old.copy()
        bad[3, 3] = 0
        swapped = old.copy()
        swapped[old == 1], swapped[old == 2] = 2, 1
        for labels in (bad, swapped, old.astype(float), old[:, :-1], old.astype(bool)):
            with self.assertRaises(ValueError):
                propose_all(raw, labels)
        with self.assertRaises(ValueError):
            propose(raw, old, "top16")
        with self.assertRaises(ValueError):
            propose_all(np.zeros_like(raw), np.zeros_like(old))
        with self.assertRaises(ValueError):
            residual_groups(old)


class PrepareTests(unittest.TestCase):
    def test_one_original_vote_and_exact_existing_descriptors(self):
        raw, old = wafer()
        expected = describe_regions(raw, old)
        with patch("f_region_evidence.prepare.tensor_vote", wraps=tensor_vote) as mocked:
            variants = preparation._job((raw, old, load_settings()))
        self.assertEqual(mocked.call_count, 1)
        for _, _, descriptors in variants.values():
            for key in preparation.DESCRIPTOR_FIELDS:
                np.testing.assert_array_equal(descriptors[key][:4], expected[key], err_msg=key)

    def test_added_complete_cc_descriptors_match_historical_formula(self):
        raw, old = wafer()
        variants = preparation._job((raw, old, load_settings()))
        labels, _, descriptors = variants["top8"]
        for token in (5, 6):
            single = (labels == token).astype(np.uint8)
            expected = describe_regions(raw, single)
            for key in preparation.DESCRIPTOR_FIELDS:
                np.testing.assert_array_equal(descriptors[key][token-1], expected[key][0], err_msg=key)

    def test_spawn_job_matches_serial(self):
        raw, labels = wafer()
        settings = load_settings()
        expected = preparation._job((raw, labels, settings))
        actual = list(preparation._map_jobs(iter([(raw, labels, settings)]), workers=1))[0]
        for name in PROPOSALS:
            np.testing.assert_array_equal(actual[name][0], expected[name][0])
            self.assertEqual(actual[name][1], expected[name][1])
            for key in preparation.DESCRIPTOR_FIELDS:
                np.testing.assert_array_equal(actual[name][2][key], expected[name][2][key])

    def test_original_descriptor_corruption_rejected(self):
        _, _, cache, *_ = source_fixture()
        row = {key: np.array(cache[key]) for key in preparation.DESCRIPTOR_FIELDS}
        row["joint_all"][0, 0] += 1e-7
        with self.assertRaises(ValueError):
            preparation._preserve_base(row, cache, 0, 4)

    def test_prepare_preserves_original_bytes_and_encodes_only_additions(self):
        maps, labels, cache, config, suite, protocol = source_fixture()
        encoder = TinyEncoder()
        model = torch.nn.ModuleDict({"encoder": encoder})
        before = {key: np.asarray(value).tobytes() for key, value in cache.items()}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            source.write_text('{"frozen":true}')
            binding = {str(source): file_hash(source)}
            with patch("f_region_evidence.prepare.ef_suite.build_model", return_value=model) as build:
                result = preparation.prepare(root / "new", maps, protocol, cache, labels, config, suite,
                    {"model": model.state_dict()}, "context", binding, workers=0, device="cpu")
            self.assertEqual(build.call_count, 1)
            self.assertEqual(sum(len(batch) for batch in encoder.seen), 5)
            self.assertEqual(tuple(result), PROPOSALS)
            for name, current in result.items():
                indices = current["base_token_indices"]
                self.assertEqual(current["features"][indices].tobytes(), cache["features"].tobytes())
                for key in preparation.DESCRIPTOR_FIELDS:
                    self.assertEqual(current[key][indices].tobytes(), cache[key].tobytes())
                metadata = read_json(current["folder"] / "metadata.json")
                self.assertEqual(metadata["proposal"], name)
                self.assertFalse(metadata["training_required"])
                audit = read_json(current["folder"] / "audit.json")
                self.assertTrue(audit["old_features_bitwise_identical"])
                self.assertTrue(audit["old_descriptors_bitwise_identical"])
                traces = [json.loads(line) for line in (current["folder"] / "region_trace.jsonl").read_text().splitlines()]
                self.assertEqual([row["sample_id"] for row in traces], [0, 1])
            with patch("f_region_evidence.prepare._job", side_effect=AssertionError("must reuse")):
                reused = preparation.prepare(root / "new", maps, protocol, cache, labels, config, suite,
                    {"model": model.state_dict()}, "context", binding, workers=0, device="cpu")
            self.assertEqual(reused["base4"]["features"].tobytes(), cache["features"].tobytes())
            for key, value in cache.items():
                self.assertEqual(np.asarray(value).tobytes(), before[key], key)
            source.write_text('{"frozen":false}')
            with self.assertRaises(ValueError):
                preparation.prepare(root / "new", maps, protocol, cache, labels, config, suite,
                    {"model": model.state_dict()}, "context", binding, workers=0, device="cpu")
            feature_file = result["base4"]["folder"] / "h512.npy"
            with feature_file.open("r+b") as handle:
                handle.seek(-1, 2)
                old_byte = handle.read(1)
                handle.seek(-1, 2)
                handle.write(bytes([old_byte[0] ^ 1]))
            with self.assertRaises(ValueError):
                preparation.read_cache(result["base4"]["folder"])

    def test_preparation_rejects_wrong_split_and_non_original_encoder(self):
        maps, labels, cache, config, suite, protocol = source_fixture()
        for split in ("train", "smoke", "Test"):
            with self.assertRaises(ValueError):
                preparation._validate_inputs(maps, protocol, cache, labels, config, suite, split)
        for key, value in (("input_size", 64), ("embedding_dim", 128), ("proposal", {"min_area": 1, "top_k": 8})):
            bad = {**config, key: value}
            with self.assertRaises(ValueError):
                preparation._validate_inputs(maps, protocol, cache, labels, bad, suite, "valid")
        bad = deepcopy(cache)
        bad["map_ids"] = np.array([1, 0])
        with self.assertRaises(ValueError):
            preparation._validate_inputs(maps, protocol, bad, labels, config, suite, "valid")

    def test_empty_map_descriptor_shapes(self):
        raw = np.ones((8, 8), dtype=np.uint8)
        labels = np.zeros(raw.shape, dtype=np.uint8)
        for _, _, row in preparation._job((raw, labels, load_settings())).values():
            self.assertEqual(row["joint_all"].shape, (0, 128))
            self.assertEqual(row["radial_hist16"].shape, (0, 16))


if __name__ == "__main__":
    unittest.main()
