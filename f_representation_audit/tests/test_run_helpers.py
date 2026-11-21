"""Selection and raw-window replay checks without torch or source artifacts."""

from collections import Counter
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from f_representation_audit.run import context_selection, original_windows, select_singletons


def metadata(map_ids, counts, labels):
    map_ids = np.asarray(map_ids, dtype=np.int64)
    offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
    owners = np.repeat(map_ids, counts)
    singleton = np.asarray([
        int(np.flatnonzero(labels[sid])[0]) if labels[sid].sum() == 1 else -1
        for sid in owners
    ], dtype=np.int64)
    return {"map_ids": map_ids, "offsets": offsets, "token_map_ids": owners,
            "singleton_labels": singleton,
            "rep_indices": np.asarray([offsets[i] for i, n in enumerate(counts) if n], dtype=np.int64)}


def window_fixture():
    raw = np.ones((8, 40, 40), dtype=np.uint8)
    raw[6, 0, 0] = 3  # Neither wafer support nor a defect.
    raw[6, 39, 39] = 0
    masks = np.zeros((2, 40, 40), dtype=np.uint8)
    masks[0, 10:13, 10:14] = 1
    masks[0, 1:3, 3:7] = 2  # Centroid (1.5, 4.5): round-to-even is (2, 4).
    masks[1, 37:39, 34:38] = 1
    raw[6][masks[0] != 0] = 2
    raw[2][masks[1] != 0] = 2
    dataset = {
        "map_ids": np.array([6, 2]),  # Source NPZ IDs, not region-mask row numbers.
        "offsets": np.array([0, 2, 3]),
        "token_map_ids": np.array([6, 6, 2]),
        "token_region_ids": np.array([1, 2, 1]),
        "area": np.array([12, 8, 8]), "region_masks": masks,
    }
    return raw, dataset


class RunHelperTests(unittest.TestCase):
    def test_singletons_use_representatives_only_and_exclude_mixed_and_normal(self):
        labels = np.zeros((50, 8), dtype=np.uint8)
        labels[[40, 5], 0] = 1
        labels[13, 5] = 1
        labels[22, 7] = 1
        labels[7, [0, 5]] = 1
        labels[[1, 19], 1] = 1
        # Map 35 is Normal; map 1 has no retained regions. Neither can contribute.
        data = metadata([40, 5, 13, 22, 7, 35, 1, 19], [2, 1, 2, 1, 2, 1, 0, 1], labels)
        all_selected = select_singletons(data, cap=None, seed=2026)
        np.testing.assert_array_equal(all_selected, [0, 2, 3, 5, 9])
        chosen = select_singletons(data, cap=1, seed=2026)
        np.random.seed(17)
        np.testing.assert_array_equal(chosen, select_singletons(data, cap=1, seed=2026))
        self.assertEqual(Counter(data["singleton_labels"][chosen]), {0: 1, 1: 1, 5: 1, 7: 1})
        self.assertTrue(set(chosen).issubset(set(data["rep_indices"])))
        self.assertEqual(len(np.unique(data["token_map_ids"][chosen])), len(chosen))

    def test_context_caps_maps_per_full_combination_and_uses_only_supplied_valid_ids(self):
        labels = np.zeros((50, 8), dtype=np.uint8)
        labels[[17, 3], 0] = 1
        labels[np.ix_([8, 11], [0, 5])] = 1
        labels[23, [5, 7]] = 1
        labels[[12, 27], 7] = 1
        labels[48, [0, 5]] = 1  # An outside/test raw row; never part of this valid dataset.
        data = metadata([17, 3, 8, 11, 23, 12, 27], [2, 1, 3, 2, 1, 0, 2], labels)
        selected = context_selection(data, labels, maps_per_combination=1, seed=2026)
        np.testing.assert_array_equal(selected, context_selection(data, labels, 1, 2026))
        chosen_maps = np.unique(data["token_map_ids"][selected])
        self.assertEqual(len(chosen_maps), 4)
        self.assertNotIn(48, chosen_maps)
        self.assertNotIn(12, chosen_maps)
        combinations = Counter(tuple(labels[sid]) for sid in chosen_maps)
        self.assertEqual(set(combinations.values()), {1})
        self.assertEqual(sum(labels[sid].sum() > 1 for sid in chosen_maps), 2)
        self.assertEqual(sum(labels[sid].sum() == 1 for sid in chosen_maps), 2)
        # Once a map is selected, every one of its retained regions is included.
        for sid in chosen_maps:
            expected = np.flatnonzero(data["token_map_ids"] == sid)
            actual = selected[data["token_map_ids"][selected] == sid]
            np.testing.assert_array_equal(actual, expected)
        all_selected = context_selection(data, labels, maps_per_combination=20, seed=2026)
        np.testing.assert_array_equal(all_selected, np.arange(len(data["token_map_ids"])))

    def test_original_windows_preserve_raw_ids_padding_round_to_even_and_channel_meaning(self):
        raw, data = window_fixture()
        raw_before, masks_before = raw.copy(), data["region_masks"].copy()
        indices = np.array([1, 2, 0])
        actual = original_windows(raw, data, indices)
        # Known fixture origins are independent of the implementation's centroid calculation.
        origins = {0: (-15, -14), 1: (-24, -22), 2: (12, 10)}
        owner_row = {6: 0, 2: 1}
        expected = np.zeros((3, 3, 52, 52), dtype=np.float32)
        for out_index, token in enumerate(indices):
            sid, rid = data["token_map_ids"][token], data["token_region_ids"][token]
            start_row, start_col = origins[int(token)]
            for row in range(52):
                for col in range(52):
                    src_row, src_col = start_row + row, start_col + col
                    if 0 <= src_row < 40 and 0 <= src_col < 40:
                        pixel = raw[sid, src_row, src_col]
                        expected[out_index, :, row, col] = (
                            pixel == 2,
                            data["region_masks"][owner_row[int(sid)], src_row, src_col] == rid,
                            pixel in (1, 2),
                        )
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual.tobytes(), expected.tobytes())
        np.testing.assert_array_equal(raw, raw_before)
        np.testing.assert_array_equal(data["region_masks"], masks_before)
        self.assertTrue(np.all(actual[:, 1] <= actual[:, 0]))
        self.assertTrue(np.all(actual[:, 0] <= actual[:, 2]))

    def test_original_windows_reject_stale_area_metadata(self):
        raw, data = window_fixture()
        data["area"] = data["area"].copy()
        data["area"][1] += 1
        with self.assertRaisesRegex(ValueError, "area does not match mask"):
            original_windows(raw, data, np.array([1]))

    def test_run_helper_import_does_not_import_torch_or_open_inputs(self):
        root = Path(__file__).resolve().parents[2]
        script = (
            "import sys; from f_representation_audit import run; "
            "assert 'torch' not in sys.modules; "
            "assert 'f_representation_audit.inputs' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-B", "-c", script], cwd=root,
                                capture_output=True, text=True, timeout=20, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
