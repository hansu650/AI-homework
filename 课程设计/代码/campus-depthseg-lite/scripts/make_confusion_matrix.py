"""Create confusion matrix assets for a trained checkpoint."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BATCH_SIZE, CLASS_NAMES, IMAGE_SIZE, NUM_CLASSES
from src.datasets.nyu5_dataset import NYU5Dataset
from src.lightning.lit_segmentation import LitSegmentation
from src.utils.metrics import confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make confusion matrix assets.")
    parser.add_argument("--data_dir", default="data/nyu5")
    parser.add_argument("--split_file", default="data/nyu5/splits/test.txt")
    parser.add_argument(
        "--checkpoint",
        default="outputs/runs/exp02_rgbd_concat_e20/checkpoints/best.ckpt",
    )
    parser.add_argument("--variant", default="rgbd_concat")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = _resolve_project_path(args.data_dir)
    split_file = _resolve_project_path(args.split_file)
    checkpoint = _validate_file(args.checkpoint, "Checkpoint")
    out_dir = PROJECT_ROOT / "outputs" / "report_assets" / "confusion_matrix"
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = _compute_matrix(
        data_dir=data_dir,
        split_file=split_file,
        checkpoint=checkpoint,
        variant=args.variant,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    normalized = _normalize_rows(matrix)
    _write_matrix_csv(out_dir / "confusion_matrix_raw.csv", matrix)
    _write_matrix_csv(out_dir / "confusion_matrix_normalized.csv", normalized)
    _save_matrix_plot(normalized, out_dir / "confusion_matrix_normalized.png")
    print(f"confusion matrix assets saved under: {out_dir}")


def _resolve_project_path(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def _validate_file(path: str, name: str) -> Path:
    file_path = _resolve_project_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{name} not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"{name} is not a file: {file_path}")
    return file_path


def _compute_matrix(
    data_dir: Path,
    split_file: Path,
    checkpoint: Path,
    variant: str,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    dataset = NYU5Dataset(split_file=split_file, data_dir=data_dir, image_size=IMAGE_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    model = LitSegmentation.load_from_checkpoint(
        str(checkpoint),
        map_location=torch.device("cpu"),
        variant=variant,
    )
    model.eval()

    matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["rgb"], batch["depth"])
            matrix += confusion_matrix(logits, batch["label"], NUM_CLASSES)
    return matrix.numpy()


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.zeros_like(matrix, dtype=np.float64)
    valid = row_sums[:, 0] > 0
    normalized[valid] = matrix[valid] / row_sums[valid]
    return normalized


def _write_matrix_csv(output_path: Path, matrix: np.ndarray) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target/pred", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, matrix):
            writer.writerow([class_name, *row.tolist()])


def _save_matrix_plot(matrix: np.ndarray, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=30, ha="right")
    axis.set_yticks(range(NUM_CLASSES), CLASS_NAMES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Target")
    axis.set_title("Normalized Confusion Matrix")
    for row in range(NUM_CLASSES):
        for col in range(NUM_CLASSES):
            axis.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
