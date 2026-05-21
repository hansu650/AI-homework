from scripts.train import parse_args
from src.lightning.lit_segmentation import LitSegmentation


def test_train_parse_args_for_fast_dev_cpu():
    args = parse_args(
        [
            "--data_dir",
            "data/nyu5",
            "--variant",
            "rgb",
            "--experiment_name",
            "unit_test",
            "--accelerator",
            "cpu",
            "--devices",
            "1",
            "--batch_size",
            "2",
            "--fast_dev_run",
        ]
    )

    assert args.data_dir == "data/nyu5"
    assert args.variant == "rgb"
    assert args.experiment_name == "unit_test"
    assert args.accelerator == "cpu"
    assert args.fast_dev_run is True


def test_lit_segmentation_constructs_with_variant():
    module = LitSegmentation(variant="rgbd_concat", learning_rate=1e-4)

    assert module.hparams.variant == "rgbd_concat"
    assert module.hparams.learning_rate == 1e-4
