"""Label mapping helpers for NYU labels and the five indoor classes."""

from __future__ import annotations

import numpy as np

from src.config import CLASS_NAMES, IGNORE_INDEX, NYU_NAME_KEYWORDS_TO_5


def build_name_to_five_class_mapping(names: list[str]) -> dict[int, int]:
    """Build an explicit one-based raw-label-id to five-class-id mapping.

    The NYUDepthV2 labeled MAT file stores semantic labels as integer ids where
    id 1 corresponds to names[0]. Id 0 is unlabeled/other in the official file.
    """

    class_to_id = {name: index for index, name in enumerate(CLASS_NAMES)}
    id_to_five = {0: class_to_id["other"]}

    for index, raw_name in enumerate(names, start=1):
        name = raw_name.strip().lower()
        target = class_to_id["other"]
        for class_name, keywords in NYU_NAME_KEYWORDS_TO_5.items():
            if any(_matches_keyword(name, keyword, class_name) for keyword in keywords):
                target = class_to_id[class_name]
                break
        id_to_five[index] = target
    return id_to_five


def remap_label_by_id(label: np.ndarray, id_to_five: dict[int, int]) -> np.ndarray:
    """Map a raw label array to values 0..4, preserving IGNORE_INDEX."""

    if label.ndim != 2:
        raise ValueError(f"label must be 2D, got shape {label.shape}")

    mapped = np.zeros(label.shape, dtype=np.uint8)
    ignore_mask = (label == IGNORE_INDEX) & (IGNORE_INDEX not in id_to_five)
    for raw_id, five_id in id_to_five.items():
        if five_id < 0 or five_id >= len(CLASS_NAMES):
            raise ValueError(f"Invalid five-class id {five_id} for raw id {raw_id}")
        mapped[label == raw_id] = np.uint8(five_id)
    mapped[ignore_mask] = IGNORE_INDEX
    return mapped


def _matches_keyword(name: str, keyword: str, class_name: str) -> bool:
    if class_name == "floor":
        return name == keyword
    return keyword in name
