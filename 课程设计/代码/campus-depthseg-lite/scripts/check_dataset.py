"""Check exported NYU5 splits through the project dataset class."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RGB_MEAN, RGB_STD
from src.datasets.nyu5_dataset import NYU5Dataset
from src.utils.visualization import colorize_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize exported NYU5 samples.")
    parser.add_argument("--data_dir", default="data/nyu5")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num_samples", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    split_file = data_dir / "splits" / f"{args.split}.txt"
    dataset = NYU5Dataset(
        split_file=split_file,
        data_dir=data_dir,
        training=False,
        label_mode="nyu5",
    )

    print(f"dataset length: {len(dataset)}")
    sample_count = min(args.num_samples, len(dataset))
    if sample_count <= 0:
        raise ValueError("num_samples must be positive and dataset must not be empty")

    rows = []
    for index in range(sample_count):
        sample = dataset[index]
        rgb = sample["rgb"]
        depth = sample["depth"]
        label = sample["label"]
        print(f"sample {index}:")
        print(f"  rgb: shape={tuple(rgb.shape)}, dtype={rgb.dtype}")
        print(f"  depth: shape={tuple(depth.shape)}, dtype={depth.dtype}")
        print(f"  label: shape={tuple(label.shape)}, dtype={label.dtype}")
        print(f"  label unique: {np.unique(label.numpy()).tolist()}")
        rows.append((rgb.numpy(), depth.numpy(), label.numpy()))

    output_path = PROJECT_ROOT / "outputs" / "figures" / "dataset_check.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_check_figure(rows, output_path)
    print(f"dataset check figure saved: {output_path}")


def _save_check_figure(
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(len(rows), 3, figsize=(10, 3.2 * len(rows)), squeeze=False)
    for row_index, (rgb, depth, label) in enumerate(rows):
        rgb_display = _unnormalize_rgb(rgb)
        depth_display = depth[0]
        label_display = label.copy()
        label_display[label_display == 255] = 0

        axes[row_index, 0].imshow(rgb_display)
        axes[row_index, 0].set_title("RGB")
        axes[row_index, 1].imshow(depth_display, cmap="viridis")
        axes[row_index, 1].set_title("Depth")
        axes[row_index, 2].imshow(colorize_mask(label_display))
        axes[row_index, 2].set_title("NYU5 label")
        for axis in axes[row_index]:
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _unnormalize_rgb(rgb: np.ndarray) -> np.ndarray:
    mean = np.array(RGB_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(RGB_STD, dtype=np.float32).reshape(3, 1, 1)
    image = np.clip(rgb * std + mean, 0.0, 1.0)
    return np.transpose(image, (1, 2, 0))


if __name__ == "__main__":
    main()
