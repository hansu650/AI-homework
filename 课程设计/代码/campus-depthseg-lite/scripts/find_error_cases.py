"""Find best and worst prediction cases by per-image mIoU."""

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

from src.config import CLASS_NAMES, IMAGE_SIZE, NUM_CLASSES, RGB_MEAN, RGB_STD
from src.datasets.nyu5_dataset import NYU5Dataset
from src.lightning.lit_segmentation import LitSegmentation
from src.utils.metrics import confusion_matrix, mean_iou
from src.utils.visualization import colorize_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find best and worst prediction cases.")
    parser.add_argument("--data_dir", default="data/nyu5")
    parser.add_argument("--split_file", default="data/nyu5/splits/test.txt")
    parser.add_argument(
        "--checkpoint",
        default="outputs/runs/exp02_rgbd_concat_e20/checkpoints/best.ckpt",
    )
    parser.add_argument("--variant", default="rgbd_concat")
    parser.add_argument("--num_cases", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_cases <= 0:
        raise ValueError("num_cases must be positive")

    data_dir = _resolve_project_path(args.data_dir)
    split_file = _resolve_project_path(args.split_file)
    checkpoint = _validate_file(args.checkpoint, "Checkpoint")
    out_dir = PROJECT_ROOT / "outputs" / "report_assets" / "error_cases"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = NYU5Dataset(split_file=split_file, data_dir=data_dir, image_size=IMAGE_SIZE)
    model = LitSegmentation.load_from_checkpoint(
        str(checkpoint),
        map_location=torch.device("cpu"),
        variant=args.variant,
    )
    model.eval()

    cases = _score_cases(dataset, model)
    best = sorted(cases, key=lambda item: item["miou"], reverse=True)[: args.num_cases]
    worst = sorted(cases, key=lambda item: item["miou"])[: args.num_cases]
    _save_cases(best, out_dir / "best_cases.png", "Best cases")
    _save_cases(worst, out_dir / "worst_cases.png", "Worst cases")
    print(f"error case assets saved under: {out_dir}")


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


def _score_cases(dataset: NYU5Dataset, model: LitSegmentation) -> list[dict[str, object]]:
    cases = []
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            rgb = sample["rgb"].unsqueeze(0)
            depth = sample["depth"].unsqueeze(0)
            target = sample["label"]
            logits = model(rgb, depth)
            prediction = logits.argmax(dim=1).squeeze(0).cpu()
            matrix = confusion_matrix(prediction, target, NUM_CLASSES)
            score = float(mean_iou(matrix).item())
            cases.append(
                {
                    "index": index,
                    "miou": score,
                    "rgb": _unnormalize_rgb(sample["rgb"].numpy()),
                    "depth": sample["depth"].numpy()[0],
                    "target": target.numpy(),
                    "prediction": prediction.numpy().astype(np.uint8),
                }
            )
    return cases


def _save_cases(cases: list[dict[str, object]], output_path: Path, title: str) -> None:
    if len(cases) == 0:
        raise ValueError("No cases available to save")
    columns = ["RGB", "Depth", "Ground Truth", "Prediction", "Error Map"]
    fig, axes = plt.subplots(len(cases), len(columns), figsize=(3.1 * len(columns), 3.0 * len(cases)))
    axes = np.atleast_2d(axes)
    for row_index, case in enumerate(cases):
        target = case["target"].copy()
        prediction = case["prediction"]
        valid = target != 255
        target[target == 255] = 0
        error = np.zeros((*target.shape, 3), dtype=np.uint8)
        error[valid & (target == prediction)] = (40, 160, 80)
        error[valid & (target != prediction)] = (220, 60, 60)

        images = [
            (case["rgb"], None),
            (case["depth"], "viridis"),
            (colorize_mask(target), None),
            (colorize_mask(prediction), None),
            (error, None),
        ]
        for col_index, (image, cmap) in enumerate(images):
            axis = axes[row_index, col_index]
            axis.imshow(image, cmap=cmap)
            if row_index == 0:
                axis.set_title(columns[col_index], fontsize=9)
            if col_index == 0:
                axis.set_ylabel(f"idx {case['index']}\nmIoU {case['miou']:.3f}")
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle(title)
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
