"""Evaluate a trained checkpoint on a NYU5 split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BATCH_SIZE, CLASS_NAMES, IMAGE_SIZE, NUM_CLASSES
from src.datasets.nyu5_dataset import NYU5Dataset
from src.lightning.lit_segmentation import LitSegmentation
from src.utils.metrics import (
    confusion_matrix,
    mean_accuracy,
    mean_iou,
    per_class_iou,
    pixel_accuracy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint.")
    parser.add_argument("--data_dir", default="data/nyu5")
    parser.add_argument("--split_file", default="data/nyu5/splits/test.txt")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", default="rgbd_boundary")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--accelerator", default="cpu")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = _resolve_project_path(args.data_dir)
    split_file = _resolve_project_path(args.split_file)
    checkpoint = _validate_file(args.checkpoint, "Checkpoint")
    device = _select_device(args.accelerator)
    run_dir = _run_dir_from_checkpoint(checkpoint)

    dataset = NYU5Dataset(split_file=split_file, data_dir=data_dir, image_size=IMAGE_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    model = LitSegmentation.load_from_checkpoint(
        str(checkpoint),
        map_location=device,
        variant=args.variant,
    )
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_samples = 0
    matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            depth = batch["depth"].to(device)
            label = batch["label"].to(device)
            logits = model(rgb, depth)
            loss = model.loss_fn(logits, label)
            batch_size = rgb.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            matrix += confusion_matrix(logits, label, NUM_CLASSES).to(device)

    if total_samples == 0:
        raise ValueError("Evaluation split contains no samples")

    metrics = _metrics_to_dict(matrix, total_loss / total_samples)
    _save_metrics(metrics, run_dir)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


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


def _select_device(accelerator: str) -> torch.device:
    if accelerator == "cpu":
        return torch.device("cpu")
    if accelerator == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU accelerator requested, but CUDA is not available")
        return torch.device("cuda:0")
    raise ValueError("accelerator must be 'cpu' or 'gpu'")


def _run_dir_from_checkpoint(checkpoint: Path) -> Path:
    if checkpoint.parent.name != "checkpoints":
        raise ValueError(
            "Checkpoint must be under outputs/runs/{experiment_name}/checkpoints/"
        )
    return checkpoint.parent.parent


def _metrics_to_dict(matrix: torch.Tensor, test_loss: float) -> dict[str, object]:
    ious = per_class_iou(matrix).detach().cpu()
    per_class = {}
    for index, class_name in enumerate(CLASS_NAMES):
        value = ious[index]
        per_class[class_name] = None if torch.isnan(value) else float(value.item())
    return {
        "test_loss": test_loss,
        "test_mIoU": float(mean_iou(matrix).item()),
        "test_pixel_acc": float(pixel_accuracy(matrix).item()),
        "test_mean_acc": float(mean_accuracy(matrix).item()),
        "per_class_iou": per_class,
    }


def _save_metrics(metrics: dict[str, object], run_dir: Path) -> None:
    txt_path = run_dir / "test_metrics.txt"
    json_path = run_dir / "test_metrics.json"
    lines = [
        f"test_loss: {metrics['test_loss']}",
        f"test_mIoU: {metrics['test_mIoU']}",
        f"test_pixel_acc: {metrics['test_pixel_acc']}",
        f"test_mean_acc: {metrics['test_mean_acc']}",
        "per_class_iou:",
    ]
    for class_name, value in metrics["per_class_iou"].items():
        lines.append(f"  {class_name}: {value}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
