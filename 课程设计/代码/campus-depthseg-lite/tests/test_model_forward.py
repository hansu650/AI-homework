import torch

from src.config import NUM_CLASSES
from src.models.campus_depthseg_lite import CampusDepthSegLite


def test_model_forward_shape_cpu():
    model = CampusDepthSegLite()
    model.eval()

    rgb = torch.rand(2, 3, 64, 80)
    depth = torch.rand(2, 1, 64, 80)

    with torch.no_grad():
        logits = model(rgb, depth)

    assert logits.shape == (2, NUM_CLASSES, 64, 80)
    assert torch.isfinite(logits).all()
