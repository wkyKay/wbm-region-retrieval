import unittest

import numpy as np

from f_representation_audit.context import build_context_views, resize_windows_nearest


def fixture():
    windows = np.zeros((3, 3, 52, 52), dtype=np.float32)
    windows[:, 2, 2:50, 2:50] = 1
    windows[:, 1, 8:12, 17:21] = 1
    windows[:, 0] = windows[:, 1]
    windows[0, 0, 25:35, 5:15] = 1
    windows[1, 0, 35:45, 30:42] = 1
    return windows, ["101:0", "102:1", "103:0"]


class ContextTests(unittest.TestCase):
    def test_target_support_invariants_and_source_immutability(self):
        windows, keys = fixture()
        before = windows.copy()
        windows.flags.writeable = False
        views, audits = build_context_views(windows, keys)
        np.testing.assert_array_equal(windows, before)
        for name, value in views.items():
            self.assertEqual(value.shape, (3, 3, 96, 96))
            self.assertEqual(value.dtype, np.float32)
            self.assertTrue(value.flags.c_contiguous)
            self.assertFalse(np.shares_memory(value, windows))
            self.assertTrue(np.all(value[:, 1] <= value[:, 0]))
            self.assertTrue(np.all(value[:, 0] <= value[:, 2]))
            if name != "rot90":
                np.testing.assert_array_equal(value[:, 1:], views["original"][:, 1:])
        for audit in audits:
            self.assertTrue(audit["valid_region_input"])
            self.assertTrue(audit["support_unchanged_after_alignment"])
            self.assertEqual(audit["target_mask_retained_fraction"], 1.0)
            self.assertEqual(audit["target_evidence_retained_fraction"], 1.0)
        views["target_only"][0, 1] = 0
        self.assertGreater(views["original"][0, 1].sum(), 0)

    def test_resize_matches_historical_index_rule_and_rotation_is_after_resize(self):
        windows, keys = fixture()
        views, _ = build_context_views(windows, keys)
        rr = np.minimum((np.arange(96) * 52 / 96).astype(int), 51)
        expected = np.stack([np.stack([c[np.ix_(rr, rr)] for c in w]) for w in windows])
        np.testing.assert_array_equal(views["original"], expected)
        np.testing.assert_array_equal(views["rot90"], np.rot90(expected, 1, axes=(-2, -1)))
        # A boundary pixel provides a counterexample to exchanging the operations.
        changed = windows.copy()
        changed[0, 0, 4, 6] = 1
        changed_views, _ = build_context_views(changed, keys)
        wrong = resize_windows_nearest(np.rot90(changed, 1, axes=(-2, -1)), 96)
        self.assertFalse(np.array_equal(changed_views["rot90"], wrong))

    def test_deterministic_across_reordering_chunks_and_global_rng(self):
        windows, keys = fixture()
        first, first_audit = build_context_views(windows, keys, seed=97)
        np.random.seed(912)
        order = [2, 0, 1]
        second, second_audit = build_context_views(windows[order], [keys[i] for i in order], seed=97)
        for name in first:
            np.testing.assert_array_equal(first[name][order], second[name])
        by_key = {(a["region_key"], a["view"]): a for a in first_audit}
        for audit in second_audit:
            self.assertEqual(audit, by_key[(audit["region_key"], audit["view"])])
        for i in range(len(keys)):
            single, _ = build_context_views(windows[i:i+1], keys[i:i+1], seed=97)
            for name in first:
                np.testing.assert_array_equal(first[name][i:i+1], single[name])

    def test_dropout_changes_whole_raw_grid_pixels_only_and_audits_match(self):
        windows, keys = fixture()
        source, audits = build_context_views(windows, keys, output_size=52)
        enlarged, _ = build_context_views(windows, keys, output_size=96)
        np.testing.assert_array_equal(enlarged["context_drop"], resize_windows_nearest(source["context_drop"], 96))
        for i, key in enumerate(keys):
            audit = next(a for a in audits if a["region_key"] == key and a["view"] == "context_drop")
            removed = (windows[i, 0] == 1) & (source["context_drop"][i, 0] == 0)
            self.assertFalse(np.any(removed & (windows[i, 1] == 1)))
            self.assertEqual(audit["source_context_changed_pixels"], int(removed.sum()))
            self.assertEqual(audit["source_pixel_unit"], "die")
        no_context = [a for a in audits if a["region_key"] == keys[2]]
        for audit in no_context:
            self.assertFalse(audit["has_context"])
            self.assertIsNone(audit["context_changed_fraction"])
            if audit["view"] != "rot90":
                self.assertTrue(audit["is_noop"])

    def test_dropout_endpoints_and_explicit_model_grid(self):
        windows, keys = fixture()
        zero, zero_audit = build_context_views(windows, keys, dropout_probability=0)
        one, _ = build_context_views(windows, keys, dropout_probability=1)
        np.testing.assert_array_equal(zero["context_drop"], zero["original"])
        np.testing.assert_array_equal(one["context_drop"], one["target_only"])
        self.assertTrue(all(a["is_noop"] for a in zero_audit if a["view"] == "context_drop"))
        _, audit = build_context_views(zero["original"], keys, input_grid="model_input")
        self.assertTrue(all(a["source_pixel_unit"] == "resized_pixel" for a in audit))

    def test_optional_target_removal_is_flagged_invalid_and_not_default(self):
        windows, keys = fixture()
        default, _ = build_context_views(windows, keys)
        self.assertNotIn("target_removed_control", default)
        views, audits = build_context_views(windows, keys, include_target_removed_control=True)
        control = views["target_removed_control"]
        np.testing.assert_array_equal(control[:, 1:], views["original"][:, 1:])
        np.testing.assert_array_equal(control[:, 0] * control[:, 1], 0)
        for audit in audits:
            if audit["view"] == "target_removed_control":
                self.assertFalse(audit["valid_region_input"])
                self.assertTrue(audit["diagnostic_only"])
                self.assertEqual(audit["target_mask_retained_fraction"], 1)
                self.assertEqual(audit["target_evidence_retained_fraction"], 0)
                self.assertEqual(audit["source_context_changed_pixels"], 0)

    def test_rejects_invalid_inputs(self):
        windows, keys = fixture()
        invalid = []
        bad = windows.copy(); bad[0, 0, 0, 0] = .5; invalid.append(bad)
        bad = windows.copy(); bad[0, 0, 0, 0] = np.nan; invalid.append(bad)
        bad = windows.copy(); bad[0, 0, 8, 17] = 0; invalid.append(bad)
        bad = windows.copy(); bad[0, 2, 8, 17] = 0; invalid.append(bad)
        bad = windows.copy(); bad[0, 1] = 0; invalid.append(bad)
        invalid.extend([windows[:0], windows[..., :30], windows[:, :2]])
        for bad in invalid:
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                build_context_views(bad, keys)
        for kwargs in ({"seed": -1}, {"seed": True}, {"dropout_probability": 2},
                       {"dropout_probability": np.nan}, {"dropout_probability": True},
                       {"dropout_probability": "0.5"},
                       {"input_grid": "unknown"}, {"output_size": 0}, {"output_size": True},
                       {"include_target_removed_control": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_context_views(windows, keys, **kwargs)
        for bad_keys in ([keys[0]] * 3, keys[:2], [1, 2, 3], "abc"):
            with self.assertRaises(ValueError):
                build_context_views(windows, bad_keys)
        cached = resize_windows_nearest(windows, 96)
        with self.assertRaisesRegex(ValueError, "declare the actual grid"):
            build_context_views(cached, keys)


if __name__ == "__main__":
    unittest.main()
