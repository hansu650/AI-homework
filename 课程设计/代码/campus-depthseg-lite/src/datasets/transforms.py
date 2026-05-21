"""Synchronized RGB-D-label transforms used by the NYU5 dataset."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image

from src.config import RGB_MEAN, RGB_STD


def resize_sample(
    rgb: Image.Image,
    depth: Image.Image,
    label: Image.Image,
    size: Tuple[int, int],
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Resize RGB, depth, and label with synchronized geometry."""

    height, width = size
    image_size = (width, height)
    rgb = rgb.resize(image_size, Image.Resampling.BILINEAR)
    depth = depth.resize(image_size, Image.Resampling.BILINEAR)
    label = label.resize(image_size, Image.Resampling.NEAREST)
    return rgb, depth, label


def horizontal_flip(
    rgb: Image.Image,
    depth: Image.Image,
    label: Image.Image,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Flip RGB, depth, and label in exactly the same way."""

    return (
        rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        depth.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        label.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
    )


def rgb_to_tensor(rgb: Image.Image) -> torch.Tensor:
    """Convert RGB PIL image to normalized float tensor [3, H, W]."""

    array = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(RGB_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(RGB_STD, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def depth_to_tensor(depth: Image.Image) -> torch.Tensor:
    """Convert a depth image to float tensor [1, H, W] normalized to [0, 1]."""

    array = np.asarray(depth, dtype=np.float32)
    if array.ndim == 3:
        array = array[..., 0]
    min_value = float(array.min())
    max_value = float(array.max())
    array = array - min_value
    scale = max_value - min_value
    if scale > 0.0:
        array = array / scale
    return torch.from_numpy(array).unsqueeze(0).float()


def label_to_tensor(label: Image.Image) -> torch.Tensor:
    """Convert a label image to long tensor [H, W]."""

    return torch.from_numpy(np.asarray(label, dtype=np.int64)).long()
