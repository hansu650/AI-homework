import numpy as np

from src.utils.boxes_from_mask import boxes_from_obstacle_mask
from src.utils.inspection import analyze_occupancy


def test_inspection_high_risk_summary_and_boxes():
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[:10, :] = 2
    mask[10:, :] = 3
    mask[12:18, 4:14] = 1

    result = analyze_occupancy(mask)
    boxes = boxes_from_obstacle_mask(mask, min_area=20)

    assert result.risk_level == "high"
    assert result.lower_floor_ratio > 0.0
    assert result.lower_obstacle_ratio > 0.5
    assert "障碍物" in result.summary or "遮挡" in result.summary
    assert len(boxes) == 1
    x1, y1, x2, y2, area = boxes[0]
    assert (x1, y1) == (0, 10)
    assert x2 == 29
    assert y2 == 19
    assert area >= 20
