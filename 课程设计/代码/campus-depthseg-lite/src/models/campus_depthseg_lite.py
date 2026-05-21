"""CampusDepthSegLite model."""

from __future__ import annotations

import torch
from torch import nn

from src.config import NUM_CLASSES
from src.models.depth_boundary_fusion import DepthBoundaryResidualFusion
from src.models.mini_encoder import MiniHierarchicalEncoder
from src.models.weighted_fpn_decoder import WeightedFPNDecoder


class CampusDepthSegLite(nn.Module):
    """Lightweight RGB-D semantic segmentation network."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        encoder_channels: tuple[int, int, int, int] = (48, 96, 192, 384),
        decoder_channels: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = MiniHierarchicalEncoder(channels=encoder_channels)
        self.depth_fusion = DepthBoundaryResidualFusion(encoder_channels)
        self.decoder = WeightedFPNDecoder(
            in_channels=encoder_channels,
            num_classes=num_classes,
            decoder_channels=decoder_channels,
        )

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"rgb must have shape [B, 3, H, W], got {tuple(rgb.shape)}")
        if depth.ndim != 4 or depth.shape[1] != 1:
            raise ValueError(
                f"depth must have shape [B, 1, H, W], got {tuple(depth.shape)}"
            )
        if rgb.shape[0] != depth.shape[0] or rgb.shape[-2:] != depth.shape[-2:]:
            raise ValueError("rgb and depth must share batch size and spatial size")

        features = self.encoder(rgb, depth)
        features = self.depth_fusion(features, depth)
        return self.decoder(features, output_size=rgb.shape[-2:])
