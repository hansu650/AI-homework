"""Spatial occupancy analysis from predicted segmentation masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InspectionResult:
    floor_ratio: float
    lower_floor_ratio: float
    obstacle_ratio: float
    lower_obstacle_ratio: float
    risk_score: float
    risk_level: str
    summary: str

    def to_dict(self) -> dict[str, float | str]:
        return {
            "floor_ratio": self.floor_ratio,
            "lower_floor_ratio": self.lower_floor_ratio,
            "obstacle_ratio": self.obstacle_ratio,
            "lower_obstacle_ratio": self.lower_obstacle_ratio,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "summary": self.summary,
        }


def analyze_occupancy(mask: np.ndarray) -> InspectionResult:
    """Analyze visible floor and obstacle occupancy from a 2D predicted mask."""

    if mask.ndim != 2:
        raise ValueError(f"mask must be a 2D array, got shape {mask.shape}")
    height, width = mask.shape
    if height == 0 or width == 0:
        raise ValueError("mask must have non-zero height and width")

    lower = mask[height // 2 :, :]
    total_pixels = float(height * width)
    lower_pixels = float(lower.size)

    floor_ratio = float((mask == 1).sum() / total_pixels)
    lower_floor_ratio = float((lower == 1).sum() / lower_pixels)
    obstacle_ratio = float((mask == 3).sum() / total_pixels)
    lower_obstacle_ratio = float((lower == 3).sum() / lower_pixels)

    risk_score = (
        0.45 * (1.0 - lower_floor_ratio)
        + 0.35 * lower_obstacle_ratio
        + 0.20 * obstacle_ratio
    )
    risk_level = _risk_level(risk_score)
    summary = _summary(lower_floor_ratio, lower_obstacle_ratio, obstacle_ratio)

    return InspectionResult(
        floor_ratio=floor_ratio,
        lower_floor_ratio=lower_floor_ratio,
        obstacle_ratio=obstacle_ratio,
        lower_obstacle_ratio=lower_obstacle_ratio,
        risk_score=float(risk_score),
        risk_level=risk_level,
        summary=summary,
    )


def _risk_level(score: float) -> str:
    if score < 0.35:
        return "low"
    if score < 0.65:
        return "medium"
    return "high"


def _summary(
    lower_floor_ratio: float,
    lower_obstacle_ratio: float,
    obstacle_ratio: float,
) -> str:
    if lower_floor_ratio < 0.30 and lower_obstacle_ratio > 0.25:
        return "画面下方地面可见率较低，疑似通道遮挡较明显。"
    if lower_obstacle_ratio > 0.35:
        return "画面下方障碍物占比较高，建议重点检查通行空间。"
    if lower_floor_ratio < 0.35:
        return "画面下方地面可见率偏低，通道可通行性需要复核。"
    if obstacle_ratio > 0.25:
        return "画面中障碍物较多，但下方通行区域仍需结合现场确认。"
    return "画面下方地面可见率较高，当前通行风险较低。"
