import pytest
import torch

from src.config import NUM_CLASSES
from src.models.campus_depthseg_lite import CampusDepthSegLite


@pytest.mark.parametrize(
    "variant",
    ["rgb", "rgbd_concat", "rgbd_boundary", "rgbd_concat_boundary"],
)
def test_model_variants_forward_cpu(variant: str):
    model = CampusDepthSegLite(variant=variant)
    model.eval()
    rgb = torch.rand(2, 3, 64, 80)
    depth = torch.rand(2, 1, 64, 80)

    with torch.no_grad():
        logits = model(rgb, depth)

    assert logits.shape == (2, NUM_CLASSES, 64, 80)
    assert torch.isfinite(logits).all()


def test_invalid_model_variant_raises():
    with pytest.raises(ValueError, match="variant must be one of"):
        CampusDepthSegLite(variant="bad_variant")
