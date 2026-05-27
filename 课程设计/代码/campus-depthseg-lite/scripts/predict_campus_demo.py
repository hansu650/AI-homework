"""Qualitative prediction panels for self-collected campus RGB images."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import IMAGE_SIZE
from src.datasets.transforms import rgb_to_tensor
from src.lightning.lit_segmentation import LitSegmentation
from src.utils.boxes_from_mask import boxes_from_obstacle_mask, draw_boxes
from src.utils.inspection import InspectionResult, analyze_occupancy
from src.utils.visualization import colorize_mask, occupancy_map

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
HEIC_EXTS = {".heic", ".heif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run qualitative campus RGB demo prediction panels.",
    )
    parser.add_argument("--rgb_dir", default="data/campus_demo/rgb")
    parser.add_argument(
        "--checkpoint",
        default="outputs/runs/exp01_rgb_e20/checkpoints/best.ckpt",
    )
    parser.add_argument("--variant", default="rgb")
    parser.add_argument("--out_dir", default="outputs/report_assets/campus_demo")
    parser.add_argument("--num_samples", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.variant != "rgb":
        raise ValueError(
            "Campus demo images have no real depth. Use --variant rgb with an RGB-only checkpoint."
        )
    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive")

    rgb_dir = _validate_dir(args.rgb_dir, "RGB directory")
    checkpoint = _validate_file(args.checkpoint, "Checkpoint")
    out_dir = _resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _collect_rgb_images(rgb_dir)
    selected_paths = image_paths[: args.num_samples]

    model = LitSegmentation.load_from_checkpoint(
        str(checkpoint),
        map_location=torch.device("cpu"),
        variant=args.variant,
    )
    model.eval()

    rows: list[dict[str, str | float]] = []
    panel_paths: list[Path] = []
    with torch.no_grad():
        for index, image_path in enumerate(selected_paths, start=1):
            rgb_original, rgb_resized, rgb_tensor = _load_rgb_for_inference(image_path)
            logits = model.model(rgb_tensor, None)
            mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            inspection = analyze_occupancy(mask)

            panel_path = out_dir / f"campus_demo_{index:03d}.png"
            _save_campus_panel(
                rgb_original=rgb_original,
                rgb_resized=rgb_resized,
                mask=mask,
                inspection=inspection,
                output_path=panel_path,
            )
            panel_paths.append(panel_path)
            rows.append(
                {
                    "file_name": image_path.name,
                    "floor_ratio": inspection.floor_ratio,
                    "obstacle_ratio": inspection.obstacle_ratio,
                    "lower_floor_ratio": inspection.lower_floor_ratio,
                    "risk_score": inspection.risk_score,
                    "risk_level": inspection.risk_level,
                }
            )
            print(f"saved: {panel_path}")

    gallery_path = out_dir / "campus_demo_gallery.png"
    _save_gallery(panel_paths, gallery_path)
    summary_path = out_dir / "campus_demo_summary.md"
    _write_summary(rows, summary_path)

    print(f"processed_images: {len(selected_paths)}")
    print(f"gallery: {gallery_path}")
    print(f"summary: {summary_path}")


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


def _validate_dir(path: str, name: str) -> Path:
    dir_path = _resolve_project_path(path)
    if not dir_path.exists():
        raise FileNotFoundError(f"{name} not found: {dir_path}")
    if not dir_path.is_dir():
        raise ValueError(f"{name} is not a directory: {dir_path}")
    return dir_path


def _collect_rgb_images(rgb_dir: Path) -> list[Path]:
    files = sorted(path for path in rgb_dir.iterdir() if path.is_file())
    heic_files = [path for path in files if path.suffix.lower() in HEIC_EXTS]
    if heic_files:
        names = ", ".join(path.name for path in heic_files)
        raise ValueError(f"HEIC/HEIF files must be converted to JPG before running: {names}")

    image_paths = [path for path in files if path.suffix.lower() in SUPPORTED_IMAGE_EXTS]
    if not image_paths:
        raise FileNotFoundError(f"No .jpg, .jpeg, or .png images found in: {rgb_dir}")
    return image_paths


def _load_rgb_for_inference(image_path: Path) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    rgb_original = np.asarray(image, dtype=np.uint8)

    height, width = IMAGE_SIZE
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    rgb_resized = np.asarray(resized, dtype=np.uint8)
    rgb_tensor = rgb_to_tensor(resized).unsqueeze(0)
    return rgb_original, rgb_resized, rgb_tensor


def _save_campus_panel(
    rgb_original: np.ndarray,
    rgb_resized: np.ndarray,
    mask: np.ndarray,
    inspection: InspectionResult,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prediction = colorize_mask(mask)
    occupancy = occupancy_map(mask)
    boxes = boxes_from_obstacle_mask(mask, min_area=80)
    risk_image = draw_boxes(rgb_resized, boxes)
    metrics_text = _metrics_text(inspection)
    summary_text = _english_summary(inspection)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    panels = [
        ("RGB image", rgb_original),
        ("Prediction mask", prediction),
        ("Occupancy map", occupancy),
        ("Risk boxes", risk_image),
    ]

    for axis, (title, image) in zip([axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]], panels):
        axis.set_title(title)
        axis.imshow(image)
        axis.axis("off")

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.95,
        metrics_text,
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02,
        0.95,
        f"Text summary\n\n{summary_text}",
        va="top",
        ha="left",
        fontsize=11,
        wrap=True,
    )

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _metrics_text(inspection: InspectionResult) -> str:
    return "\n".join(
        [
            "Metrics block",
            f"floor_ratio:       {inspection.floor_ratio:.4f}",
            f"obstacle_ratio:    {inspection.obstacle_ratio:.4f}",
            f"lower_floor_ratio: {inspection.lower_floor_ratio:.4f}",
            f"risk_score:        {inspection.risk_score:.4f}",
            f"risk_level:        {inspection.risk_level}",
        ]
    )


def _english_summary(inspection: InspectionResult) -> str:
    if inspection.lower_floor_ratio < 0.30 and inspection.lower_obstacle_ratio > 0.25:
        return "The lower view has limited visible floor and notable obstacle occupancy."
    if inspection.lower_obstacle_ratio > 0.35:
        return "The lower view contains many obstacle pixels; passage space should be checked."
    if inspection.lower_floor_ratio < 0.35:
        return "The lower view has limited visible floor, so passability needs review."
    if inspection.obstacle_ratio > 0.25:
        return "Obstacle occupancy is noticeable; on-site confirmation is recommended."
    return "Visible lower-floor area is relatively clear and the estimated risk is low."


def _save_gallery(panel_paths: list[Path], output_path: Path) -> None:
    if not panel_paths:
        raise ValueError("panel_paths must not be empty")

    columns = min(2, len(panel_paths))
    rows = math.ceil(len(panel_paths) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 7, rows * 4))
    axis_array = np.atleast_1d(axes).reshape(rows, columns)

    for axis in axis_array.flat:
        axis.axis("off")

    for axis, panel_path in zip(axis_array.flat, panel_paths):
        panel = np.asarray(Image.open(panel_path).convert("RGB"))
        axis.imshow(panel)
        axis.set_title(panel_path.stem)
        axis.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _write_summary(rows: list[dict[str, str | float]], output_path: Path) -> None:
    if not rows:
        raise ValueError("rows must not be empty")

    lines = [
        "# Campus Demo Summary",
        "",
        "These campus images are used only for qualitative demonstration. They are not used for training or quantitative evaluation because pixel-level ground truth labels are unavailable.",
        "",
        "| file name | floor_ratio | obstacle_ratio | lower_floor_ratio | risk_score | risk_level |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {file_name} | {floor_ratio:.4f} | {obstacle_ratio:.4f} | "
            "{lower_floor_ratio:.4f} | {risk_score:.4f} | {risk_level} |".format(
                **row
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
