"""Synthetic-only inspection demo.

This script does not train or call a real model. It creates a small synthetic
mask to exercise occupancy analysis, boxes, and visualization.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.boxes_from_mask import boxes_from_obstacle_mask
from src.utils.inspection import analyze_occupancy
from src.utils.visualization import save_inspection_panel


def synthetic_demo_sample(
    height: int = 180,
    width: int = 240,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(110 + 60 * x, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(120 + 70 * y, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(135 + 40 * (1.0 - y), 0, 255).astype(np.uint8)

    depth = (0.25 + 0.75 * y).repeat(width, axis=1)

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[: height // 2, :] = 2
    mask[height // 2 :, :] = 1
    mask[95:170, 62:108] = 3
    mask[115:178, 145:205] = 3
    mask[42:105, 180:222] = 4
    return rgb, depth, mask


def main() -> None:
    rgb, depth, mask = synthetic_demo_sample()
    inspection = analyze_occupancy(mask)
    boxes = boxes_from_obstacle_mask(mask, min_area=80)
    output_path = PROJECT_ROOT / "demo" / "results" / "demo_panel.png"
    save_inspection_panel(
        rgb=rgb,
        depth=depth,
        mask=mask,
        boxes=boxes,
        summary=inspection.summary,
        output_path=output_path,
    )

    print(f"risk_level: {inspection.risk_level}")
    print(f"risk_score: {inspection.risk_score:.4f}")
    print(f"summary: {inspection.summary}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
