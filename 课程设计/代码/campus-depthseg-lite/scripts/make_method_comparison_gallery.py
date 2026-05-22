"""Create side-by-side comparison galleries for the four model variants."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import IMAGE_SIZE, RGB_MEAN, RGB_STD
from src.datasets.nyu5_dataset import NYU5Dataset
from src.lightning.lit_segmentation import LitSegmentation
from src.utils.visualization import colorize_mask

METHODS = [
    ("RGB-only", "rgb", "exp01_rgb_e20"),
    ("RGBD-concat", "rgbd_concat", "exp02_rgbd_concat_e20"),
    ("RGBD-boundary", "rgbd_boundary", "exp03_rgbd_boundary_e20"),
    (
        "RGBD-concat-boundary",
        "rgbd_concat_boundary",
        "exp04_rgbd_concat_boundary_e20",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make method comparison panels.")
    parser.add_argument("--data_dir", default="data/nyu5")
    parser.add_argument("--split_file", default="data/nyu5/splits/test.txt")
    parser.add_argument("--num_samples", type=int, default=6)
    parser.add_argument(
        "--out_dir",
        default="outputs/report_assets/method_comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive")

    data_dir = _resolve_project_path(args.data_dir)
    split_file = _resolve_project_path(args.split_file)
    out_dir = _resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = NYU5Dataset(split_file=split_file, data_dir=data_dir, image_size=IMAGE_SIZE)
    models = _load_models()
    count = min(args.num_samples, len(dataset))
    rows = []

    with torch.no_grad():
        for index in range(count):
            sample = dataset[index]
            predictions = _predict_all(models, sample)
            row = _make_row(sample, predictions)
            rows.append(row)
            output_path = out_dir / f"comparison_{index + 1:03d}.png"
            _save_row(row, output_path)
            print(f"saved: {output_path}")

    _save_grid(rows, out_dir / "method_comparison_grid.png")
    print(f"grid saved: {out_dir / 'method_comparison_grid.png'}")


def _resolve_project_path(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def _load_models() -> list[tuple[str, LitSegmentation]]:
    models = []
    for label, variant, run_name in METHODS:
        checkpoint = (
            PROJECT_ROOT / "outputs" / "runs" / run_name / "checkpoints" / "best.ckpt"
        )
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        model = LitSegmentation.load_from_checkpoint(
            str(checkpoint),
            map_location=torch.device("cpu"),
            variant=variant,
        )
        model.eval()
        models.append((label, model))
    return models


def _predict_all(
    models: list[tuple[str, LitSegmentation]],
    sample: dict[str, object],
) -> list[tuple[str, np.ndarray]]:
    rgb = sample["rgb"].unsqueeze(0)
    depth = sample["depth"].unsqueeze(0)
    predictions = []
    for label, model in models:
        logits = model(rgb, depth)
        mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        predictions.append((label, mask))
    return predictions


def _make_row(
    sample: dict[str, object],
    predictions: list[tuple[str, np.ndarray]],
) -> list[tuple[str, np.ndarray, str | None]]:
    label = sample["label"].numpy().copy()
    label[label == 255] = 0
    row = [
        ("RGB", _unnormalize_rgb(sample["rgb"].numpy()), None),
        ("Depth", sample["depth"].numpy()[0], "viridis"),
        ("Ground Truth", colorize_mask(label), None),
    ]
    for name, mask in predictions:
        row.append((name, colorize_mask(mask), None))
    return row


def _save_row(row: list[tuple[str, np.ndarray, str | None]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(row), figsize=(3.0 * len(row), 3.2))
    for axis, (title, image, cmap) in zip(axes, row):
        axis.imshow(image, cmap=cmap)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_grid(rows: list[list[tuple[str, np.ndarray, str | None]]], output_path: Path) -> None:
    if len(rows) == 0:
        raise ValueError("No rows available for method comparison grid")
    column_count = len(rows[0])
    fig, axes = plt.subplots(len(rows), column_count, figsize=(3.0 * column_count, 3.0 * len(rows)))
    axes = np.atleast_2d(axes)
    for row_index, row in enumerate(rows):
        for col_index, (title, image, cmap) in enumerate(row):
            axis = axes[row_index, col_index]
            axis.imshow(image, cmap=cmap)
            if row_index == 0:
                axis.set_title(title, fontsize=9)
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _unnormalize_rgb(rgb: np.ndarray) -> np.ndarray:
    mean = np.array(RGB_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(RGB_STD, dtype=np.float32).reshape(3, 1, 1)
    image = np.clip(rgb * std + mean, 0.0, 1.0)
    image = np.transpose(image, (1, 2, 0))
    return np.round(image * 255.0).astype(np.uint8)


if __name__ == "__main__":
    main()
