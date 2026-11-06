"""Method routing checks that exercise the unified entry without training."""

import importlib.util
import sys
import unittest
from unittest import mock

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "Install requirements.txt for method dispatch verification")
class MethodDispatchTests(unittest.TestCase):
    def test_unified_all_stage_dispatches_to_separate_methods(self):
        from experiments._shared import main_cli as run_experiment

        config = {"output_dir": "unused", "global": {"budgets": [.05, .1, .5]}}
        protocol = object()
        with mock.patch.object(sys, "argv", ["experiments/_shared/main_cli.py"]), \
                mock.patch.object(run_experiment, "load_config", return_value=config), \
                mock.patch.object(run_experiment, "prepare") as prepare, \
                mock.patch.object(run_experiment, "Protocol", return_value=protocol), \
                mock.patch("experiments.proposed.main.method.train") as train_local, \
                mock.patch("experiments.supervised.a.method.train") as train_global, \
                mock.patch("wbm_retrieval.retrieval.retrieve") as retrieve, \
                mock.patch("wbm_retrieval.evaluation.evaluate_all") as evaluate, \
                mock.patch("wbm_retrieval.evaluation.compare_paired") as paired:
            run_experiment.main()
        prepare.assert_called_once_with(config)
        train_local.assert_called_once_with(config, resume=False)
        train_global.assert_called_once_with(config, protocol, resume=False)
        retrieve.assert_called_once_with(config, protocol, None)
        evaluate.assert_called_once_with(config, protocol, None)
        paired.assert_called_once_with(config)

    def test_retrieval_dispatch_only_runs_requested_method(self):
        from wbm_retrieval.retrieval import retrieve

        config = {"global": {"budgets": [.05, .1, .5]}}
        protocol = object()
        with mock.patch("experiments.handcrafted.a.descriptor.retrieve") as handcrafted, \
                mock.patch("experiments.proposed.main.method.retrieve") as proposed, \
                mock.patch("experiments.supervised.a.method.retrieve") as supervised:
            retrieve(config, protocol, ["handcrafted"])
            handcrafted.assert_called_once_with(config, protocol)
            proposed.assert_not_called()
            supervised.assert_not_called()
            handcrafted.reset_mock()
            retrieve(config, protocol, ["global_10pct"])
            supervised.assert_called_once_with(config, protocol, ["global_10pct"])
            handcrafted.assert_not_called()
            proposed.assert_not_called()
            with self.assertRaises(ValueError):
                retrieve(config, protocol, ["unknown"])


if __name__ == "__main__":
    unittest.main()
