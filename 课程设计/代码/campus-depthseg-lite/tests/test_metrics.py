import torch

from src.utils.metrics import (
    confusion_matrix,
    mean_accuracy,
    mean_iou,
    per_class_iou,
    pixel_accuracy,
)


def test_metrics_ignore_index_and_iou():
    target = torch.tensor([[[0, 1, 1], [2, 255, 3]]])
    prediction = torch.tensor([[[0, 1, 2], [2, 0, 3]]])

    matrix = confusion_matrix(prediction, target, num_classes=5)

    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    assert torch.equal(matrix, expected)
    assert torch.isclose(pixel_accuracy(matrix), torch.tensor(0.8))
    assert torch.isclose(mean_accuracy(matrix), torch.tensor((1.0 + 0.5 + 1.0 + 1.0) / 4.0))

    iou = per_class_iou(matrix)
    assert torch.isclose(iou[0], torch.tensor(1.0))
    assert torch.isclose(iou[1], torch.tensor(0.5))
    assert torch.isclose(iou[2], torch.tensor(0.5))
    assert torch.isclose(iou[3], torch.tensor(1.0))
    assert torch.isnan(iou[4])
    assert torch.isclose(mean_iou(matrix), torch.tensor(0.75))
