"""Clear segmentation metrics with ignore-index support."""

from __future__ import annotations

import torch

from src.config import IGNORE_INDEX, NUM_CLASSES


def confusion_matrix(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Compute a [num_classes, num_classes] matrix, rows target and cols pred."""

    if prediction.ndim == target.ndim + 1:
        prediction = prediction.argmax(dim=1)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match after argmax")

    prediction = prediction.reshape(-1).long()
    target = target.reshape(-1).long()
    valid = target != ignore_index
    valid = valid & (target >= 0) & (target < num_classes)
    valid = valid & (prediction >= 0) & (prediction < num_classes)

    target = target[valid]
    prediction = prediction[valid]
    encoded = target * num_classes + prediction
    matrix = torch.bincount(encoded, minlength=num_classes * num_classes)
    return matrix.reshape(num_classes, num_classes)


def pixel_accuracy(matrix: torch.Tensor) -> torch.Tensor:
    total = matrix.sum()
    if total == 0:
        return torch.tensor(0.0, device=matrix.device)
    return matrix.diag().sum().float() / total.float()


def mean_accuracy(matrix: torch.Tensor) -> torch.Tensor:
    row_sum = matrix.sum(dim=1)
    valid = row_sum > 0
    if not bool(valid.any()):
        return torch.tensor(0.0, device=matrix.device)
    class_acc = matrix.diag().float() / row_sum.clamp_min(1).float()
    return class_acc[valid].mean()


def per_class_iou(matrix: torch.Tensor) -> torch.Tensor:
    intersection = matrix.diag().float()
    union = matrix.sum(dim=1).float() + matrix.sum(dim=0).float() - intersection
    iou = torch.zeros_like(intersection)
    valid = union > 0
    iou[valid] = intersection[valid] / union[valid]
    iou[~valid] = float("nan")
    return iou


def mean_iou(matrix: torch.Tensor) -> torch.Tensor:
    iou = per_class_iou(matrix)
    valid = ~torch.isnan(iou)
    if not bool(valid.any()):
        return torch.tensor(0.0, device=matrix.device)
    return iou[valid].mean()
