"""Create RGB/depth/label gallery images for exported NYU5 splits."""

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
    parser = argparse.ArgumentParser(description="Make a NYU5 dataset gallery.")
    parser.add_argument("--data_dir", default="data/nyu5")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num_samples", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    split_file = data_dir / "splits" / f"{args.split}.txt"
    dataset = NYU5Dataset(split_file=split_file, data_dir=data_dir, training=False)
    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive")

    indices = _sample_indices(len(dataset), args.num_samples)
    rows = [dataset[index] for index in indices]

    output_path = PROJECT_ROOT / "outputs" / "figures" / f"nyu5_gallery_{args.split}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_gallery(rows, output_path)
    print(f"gallery saved: {output_path}")
    print(f"dataset length: {len(dataset)}")
    print(f"shown indices: {indices}")


def _sample_indices(dataset_length: int, num_samples: int) -> list[int]:
    if dataset_length <= 0:
        raise ValueError("dataset must not be empty")
    count = min(num_samples, dataset_length)
    if count == 1:
        return [0]
    return np.linspace(0, dataset_length - 1, count, dtype=np.int64).tolist()


def _save_gallery(rows: list[dict[str, object]], output_path: Path) -> None:
    fig, axes = plt.subplots(len(rows), 3, figsize=(10, 2.8 * len(rows)), squeeze=False)
    for row_index, sample in enumerate(rows):
        rgb = _unnormalize_rgb(sample["rgb"].numpy())
        depth = sample["depth"].numpy()[0]
        label = sample["label"].numpy().copy()
        label[label == 255] = 0

        axes[row_index, 0].imshow(rgb)
        axes[row_index, 0].set_title("RGB")
        axes[row_index, 1].imshow(depth, cmap="viridis")
        axes[row_index, 1].set_title("Depth")
        axes[row_index, 2].imshow(colorize_mask(label))
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
