"""Create inspection panels for predictions from a trained checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import IMAGE_SIZE, RGB_MEAN, RGB_STD
from src.datasets.nyu5_dataset import NYU5Dataset
from src.lightning.lit_segmentation import LitSegmentation
from src.utils.boxes_from_mask import boxes_from_obstacle_mask
from src.utils.inspection import analyze_occupancy
from src.utils.visualization import save_inspection_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict panels for a NYU5 split.")
    parser.add_argument("--data_dir", default="data/nyu5")
    parser.add_argument("--split_file", default="data/nyu5/splits/test.txt")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", default="rgbd_boundary")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive")

    data_dir = _resolve_project_path(args.data_dir)
    split_file = _resolve_project_path(args.split_file)
    checkpoint = _validate_file(args.checkpoint, "Checkpoint")
    out_dir = _resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = NYU5Dataset(split_file=split_file, data_dir=data_dir, image_size=IMAGE_SIZE)
    model = LitSegmentation.load_from_checkpoint(
        str(checkpoint),
        map_location=torch.device("cpu"),
        variant=args.variant,
    )
    model.eval()

    sample_count = min(args.num_samples, len(dataset))
    with torch.no_grad():
        for index in range(sample_count):
            sample = dataset[index]
            rgb = sample["rgb"].unsqueeze(0)
            depth = sample["depth"].unsqueeze(0)
            logits = model(rgb, depth)
            mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            inspection = analyze_occupancy(mask)
            boxes = boxes_from_obstacle_mask(mask, min_area=80)
            output_path = out_dir / f"prediction_{index + 1:03d}.png"
            save_inspection_panel(
                rgb=_unnormalize_rgb(sample["rgb"].numpy()),
                depth=sample["depth"].numpy()[0],
                mask=mask,
                boxes=boxes,
                summary=inspection.summary,
                output_path=output_path,
            )
            print(f"saved: {output_path}")


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


def _unnormalize_rgb(rgb: np.ndarray) -> np.ndarray:
    mean = np.array(RGB_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(RGB_STD, dtype=np.float32).reshape(3, 1, 1)
    image = np.clip(rgb * std + mean, 0.0, 1.0)
    image = np.transpose(image, (1, 2, 0))
    return np.round(image * 255.0).astype(np.uint8)


if __name__ == "__main__":
    main()
