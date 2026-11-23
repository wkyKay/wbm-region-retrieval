import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import pytest
from cnn_augmentation.run import require_valid_gate
from cnn_augmentation.followthrough import launch, gpu_free


def test_incomplete_valid_cannot_open_test_gate():
    with tempfile.TemporaryDirectory() as root:
        output=Path(root)
        index={'context_signature':'context','model':'crop','split':'valid',
               'evaluations':{'valid/cnn_only':{}}}
        with patch('cnn_augmentation.register.validate_model_split',return_value=index):
            with pytest.raises(ValueError,match='incomplete'):
                require_valid_gate(output,'crop','context')
        assert not (output/'valid_gate_crop.json').exists()


def test_stage_launcher_rejects_unplanned_model_and_stage():
    with tempfile.TemporaryDirectory() as root:
        output=Path(root)
        with pytest.raises(ValueError,match='authorized'):
            launch(output,'train_baseline','train',model='baseline',gpu='1')
        with pytest.raises(ValueError,match='authorized'):
            launch(output,'teacher','teacher',gpu='1')
        assert not list(output.iterdir())


def test_stage_launcher_refuses_duplicate_even_without_checkpoint():
    with tempfile.TemporaryDirectory() as root:
        output=Path(root)
        (output/'launch_train_crop.json').write_text('{}')
        with pytest.raises(ValueError,match='duplicate'):
            launch(output,'train_crop','train',model='crop',gpu='1')


def test_resource_guard_never_uses_gpu_occupied_by_another_process():
    with patch('cnn_augmentation.followthrough.subprocess.check_output',return_value='2314\n'):
        with pytest.raises(RuntimeError,match='another compute process'):
            gpu_free('1')


def test_partial_initialization_blocks_every_scientific_stage():
    from cnn_augmentation.run import main
    with tempfile.TemporaryDirectory() as root:
        with patch('cnn_augmentation.run.run_context',return_value={'signature':'fixture'}):
            for stage in ('audit','train','evaluate','report','finalize'):
                with pytest.raises(ValueError,match='Initialization incomplete'):
                    main(['--stage',stage,'--output-root',root])
        assert not list(Path(root).iterdir())
