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
        return "\u753b\u9762\u4e0b\u65b9\u5730\u9762\u53ef\u89c1\u7387\u8f83\u4f4e\uff0c\u7591\u4f3c\u901a\u9053\u906e\u6321\u8f83\u660e\u663e\u3002"
    if lower_obstacle_ratio > 0.35:
        return "\u753b\u9762\u4e0b\u65b9\u969c\u788d\u7269\u5360\u6bd4\u8f83\u9ad8\uff0c\u5efa\u8bae\u91cd\u70b9\u68c0\u67e5\u901a\u884c\u7a7a\u95f4\u3002"
    if lower_floor_ratio < 0.35:
        return "\u753b\u9762\u4e0b\u65b9\u5730\u9762\u53ef\u89c1\u7387\u504f\u4f4e\uff0c\u901a\u9053\u53ef\u901a\u884c\u6027\u9700\u8981\u590d\u6838\u3002"
    if obstacle_ratio > 0.25:
        return "\u753b\u9762\u4e2d\u969c\u788d\u7269\u8f83\u591a\uff0c\u4f46\u4e0b\u65b9\u901a\u884c\u533a\u57df\u4ecd\u9700\u7ed3\u5408\u73b0\u573a\u786e\u8ba4\u3002"
    return "\u753b\u9762\u4e0b\u65b9\u5730\u9762\u53ef\u89c1\u7387\u8f83\u9ad8\uff0c\u5f53\u524d\u901a\u884c\u98ce\u9669\u8f83\u4f4e\u3002"
