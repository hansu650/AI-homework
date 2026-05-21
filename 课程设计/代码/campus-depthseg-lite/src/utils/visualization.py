"""Visualization helpers for inspection demos and predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from src.config import CLASS_NAMES
from src.utils.boxes_from_mask import Box, draw_boxes

PALETTE = np.array(
    [
        [80, 80, 80],
        [76, 175, 80],
        [96, 125, 139],
        [244, 67, 54],
        [33, 150, 243],
    ],
    dtype=np.uint8,
)


def _configure_chinese_font() -> None:
    font_candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got {mask.shape}")
    safe_mask = np.clip(mask, 0, len(PALETTE) - 1)
    return PALETTE[safe_mask]


def occupancy_map(mask: np.ndarray) -> np.ndarray:
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    canvas[mask == 1] = (76, 175, 80)
    canvas[mask == 3] = (244, 67, 54)
    canvas[mask == 4] = (33, 150, 243)
    return canvas


def save_inspection_panel(
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    boxes: Sequence[Box],
    summary: str,
    output_path: str | Path,
) -> Path:
    """Save a 2x3 visual panel for campus inspection."""

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must have shape [H, W, 3], got {rgb.shape}")
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2D, got {depth.shape}")
    if mask.shape != depth.shape or mask.shape != rgb.shape[:2]:
        raise ValueError("rgb, depth, and mask must share spatial size")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _configure_chinese_font()

    risk_image = draw_boxes(rgb, list(boxes))
    prediction = colorize_mask(mask)
    occupancy = occupancy_map(mask)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    panels = [
        ("RGB 原图", rgb),
        ("Depth / pseudo depth", depth),
        ("Prediction mask", prediction),
        ("Occupancy map", occupancy),
        ("Risk boxes", risk_image),
    ]

    for axis, (title, image) in zip(axes.flat[:5], panels):
        axis.set_title(title)
        if image.ndim == 2:
            axis.imshow(image, cmap="viridis")
        else:
            axis.imshow(image)
        axis.axis("off")

    text_axis = axes.flat[5]
    text_axis.axis("off")
    legend = "\n".join(f"{idx}: {name}" for idx, name in enumerate(CLASS_NAMES))
    text_axis.text(
        0.02,
        0.92,
        f"巡检摘要\n\n{summary}\n\n类别\n{legend}",
        va="top",
        ha="left",
        fontsize=11,
        wrap=True,
    )

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
