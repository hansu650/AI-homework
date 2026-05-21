from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.config import IGNORE_INDEX
from src.datasets.nyu5_dataset import NYU5Dataset


def test_dataset_reads_synthetic_files_and_maps_labels(tmp_path: Path):
    rgb = np.zeros((8, 10, 3), dtype=np.uint8)
    rgb[..., 0] = 120
    rgb[..., 1] = 80
    rgb[..., 2] = 40

    depth = np.linspace(0, 65535, 80, dtype=np.uint16).reshape(8, 10)
    label = np.array(
        [
            [0, 1, 2, 5, 8, 255, 0, 1, 2, 5],
            [0, 1, 2, 5, 8, 255, 0, 1, 2, 5],
            [0, 1, 2, 5, 8, 255, 0, 1, 2, 5],
            [0, 1, 2, 5, 8, 255, 0, 1, 2, 5],
            [0, 1, 2, 5, 8, 255, 0, 1, 2, 5],
            [0, 1, 2, 5, 8, 255, 0, 1, 2, 5],
            [0, 1, 2, 5, 8, 255, 0, 1, 2, 5],
            [0, 1, 2, 5, 8, 255, 0, 1, 2, 5],
        ],
        dtype=np.uint8,
    )

    Image.fromarray(rgb).save(tmp_path / "rgb.png")
    Image.fromarray(depth).save(tmp_path / "depth.png")
    Image.fromarray(label).save(tmp_path / "label.png")
    (tmp_path / "split.txt").write_text("rgb.png depth.png label.png\n", encoding="utf-8")

    dataset = NYU5Dataset(tmp_path / "split.txt", image_size=(8, 10), training=False)
    sample = dataset[0]

    assert sample["rgb"].shape == (3, 8, 10)
    assert sample["depth"].shape == (1, 8, 10)
    assert sample["label"].shape == (8, 10)
    assert sample["depth"].min().item() == pytest.approx(0.0)
    assert sample["depth"].max().item() == pytest.approx(1.0)

    labels = sample["label"]
    assert labels[0, 0].item() == 0
    assert labels[0, 1].item() == 2
    assert labels[0, 2].item() == 1
    assert labels[0, 3].item() == 3
    assert labels[0, 4].item() == 4
    assert labels[0, 5].item() == IGNORE_INDEX


def test_dataset_missing_file_raises_clear_error(tmp_path: Path):
    (tmp_path / "split.txt").write_text("missing_rgb.png depth.png label.png\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="RGB file not found"):
        NYU5Dataset(tmp_path / "split.txt")
