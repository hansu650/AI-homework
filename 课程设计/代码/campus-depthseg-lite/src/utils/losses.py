"""Loss functions for semantic segmentation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.config import DICE_WEIGHT, IGNORE_INDEX, NUM_CLASSES


class DiceLoss(nn.Module):
    """Multi-class Dice loss with ignore-index support."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        ignore_index: int = IGNORE_INDEX,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore_index
        safe_target = target.clone()
        safe_target[~valid] = 0

        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(safe_target, num_classes=self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).to(dtype=probs.dtype)

        mask = valid.unsqueeze(1).to(dtype=probs.dtype)
        probs = probs * mask
        one_hot = one_hot * mask

        intersection = (probs * one_hot).sum(dim=(0, 2, 3))
        denominator = (probs + one_hot).sum(dim=(0, 2, 3))
        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    """Cross entropy plus Dice loss."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        ignore_index: int = IGNORE_INDEX,
        dice_weight: float = DICE_WEIGHT,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, target) + self.dice_weight * self.dice(logits, target)
