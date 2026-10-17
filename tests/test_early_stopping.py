"""Deterministic checks of stopping, best-loss independence and restoration."""

import unittest

from wbm_retrieval.early_stopping import EarlyStopping


class EarlyStoppingTests(unittest.TestCase):
    def test_patience_and_minimum_epochs(self):
        stopper = EarlyStopping(patience=2, min_delta=.01, min_epochs=3)
        self.assertFalse(stopper.step(1, 1))
        self.assertFalse(stopper.step(1, 2))
        self.assertFalse(stopper.step(1, 3))
        self.assertFalse(stopper.step(1, 4))
        self.assertTrue(stopper.step(1, 5))

    def test_meaningful_improvement_resets_counter(self):
        stopper = EarlyStopping(patience=2, min_delta=.01)
        stopper.step(1, 1)
        stopper.step(1.1, 2)
        self.assertEqual(stopper.bad_epochs, 1)
        self.assertFalse(stopper.step(.8, 3))
        self.assertEqual(stopper.bad_epochs, 0)

    def test_small_improvements_accumulate_against_reference(self):
        stopper = EarlyStopping(patience=3, min_delta=.1)
        stopper.step(1, 1)
        stopper.step(.96, 2)
        stopper.step(.92, 3)
        self.assertFalse(stopper.step(.88, 4))
        self.assertEqual(stopper.reference, .88)
        self.assertEqual(stopper.bad_epochs, 0)

    def test_resumed_patience_does_not_restart(self):
        original = EarlyStopping(patience=2)
        original.step(1, 1)
        original.step(1, 2)
        restored = EarlyStopping(patience=2)
        restored.load_state_dict(original.state_dict())
        self.assertTrue(restored.step(1, 3))
        stopped = EarlyStopping(patience=2)
        stopped.load_state_dict(restored.state_dict())
        self.assertTrue(stopped.stopped)
        with self.assertRaises(ValueError):
            stopped.step(1, 4)

    def test_disabled_and_invalid_metrics(self):
        stopper = EarlyStopping(enabled=False, patience=1)
        for epoch in range(1, 5):
            self.assertFalse(stopper.step(float(epoch), epoch))
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                EarlyStopping().step(value, 1)

    def test_restore_rejects_changed_policy(self):
        stopper = EarlyStopping(patience=2)
        with self.assertRaises(ValueError):
            EarlyStopping(patience=3).load_state_dict(stopper.state_dict())

    def test_defaults_do_not_change_max_epochs(self):
        local = {"epochs": 100}
        global_ = {"epochs": 50}
        self.assertEqual(EarlyStopping.from_settings(local, "local").patience, 10)
        self.assertEqual(EarlyStopping.from_settings(global_, "global").patience, 5)
        self.assertEqual(local, {"epochs": 100})
        self.assertEqual(global_, {"epochs": 50})


if __name__ == "__main__":
    unittest.main()
