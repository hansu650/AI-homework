"""Bounding boxes derived from segmentation masks, not object detection."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int, int]


def boxes_from_obstacle_mask(
    mask: np.ndarray,
    obstacle_class: int = 3,
    min_area: int = 64,
) -> List[Box]:
    """Return connected-component boxes as (x1, y1, x2, y2, area)."""

    if mask.ndim != 2:
        raise ValueError(f"mask must be a 2D array, got shape {mask.shape}")
    obstacle = (mask == obstacle_class).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        obstacle,
        connectivity=8,
    )

    boxes: List[Box] = []
    for label_id in range(1, num_labels):
        x, y, width, height, area = stats[label_id]
        if int(area) < min_area:
            continue
        boxes.append(
            (
                int(x),
                int(y),
                int(x + width - 1),
                int(y + height - 1),
                int(area),
            )
        )
    return boxes


def draw_boxes(
    image: np.ndarray,
    boxes: List[Box],
    color: tuple[int, int, int] = (255, 64, 64),
    thickness: int = 2,
) -> np.ndarray:
    """Draw segmentation-derived boxes on an RGB image."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must have shape [H, W, 3], got {image.shape}")
    canvas = image.copy()
    for x1, y1, x2, y2, _ in boxes:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
    return canvas
