"""CampusDepthSegLite model."""

from __future__ import annotations

import torch
from torch import nn

from src.config import NUM_CLASSES
from src.models.depth_boundary_fusion import DepthBoundaryResidualFusion
from src.models.mini_encoder import MiniHierarchicalEncoder
from src.models.weighted_fpn_decoder import WeightedFPNDecoder

VALID_VARIANTS = ("rgb", "rgbd_concat", "rgbd_boundary")


class CampusDepthSegLite(nn.Module):
    """Lightweight RGB-D semantic segmentation network."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        encoder_channels: tuple[int, int, int, int] = (48, 96, 192, 384),
        decoder_channels: int = 128,
        variant: str = "rgbd_boundary",
    ) -> None:
        super().__init__()
        if variant not in VALID_VARIANTS:
            raise ValueError(f"variant must be one of {VALID_VARIANTS}, got {variant}")

        self.variant = variant
        in_channels = 4 if variant == "rgbd_concat" else 3
        self.encoder = MiniHierarchicalEncoder(
            in_channels=in_channels,
            channels=encoder_channels,
        )
        self.depth_fusion = (
            DepthBoundaryResidualFusion(encoder_channels)
            if variant == "rgbd_boundary"
            else None
        )
        self.decoder = WeightedFPNDecoder(
            in_channels=encoder_channels,
            num_classes=num_classes,
            decoder_channels=decoder_channels,
        )

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor | None = None) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"rgb must have shape [B, 3, H, W], got {tuple(rgb.shape)}")
        if self.variant in {"rgbd_concat", "rgbd_boundary"}:
            self._validate_depth(rgb, depth)

        if self.variant == "rgb":
            encoder_input = rgb
        elif self.variant == "rgbd_concat":
            encoder_input = torch.cat([rgb, depth], dim=1)
        else:
            encoder_input = rgb

        features = self.encoder(encoder_input)
        if self.depth_fusion is not None:
            features = self.depth_fusion(features, depth)
        return self.decoder(features, output_size=rgb.shape[-2:])

    @staticmethod
    def _validate_depth(rgb: torch.Tensor, depth: torch.Tensor | None) -> None:
        if depth is None:
            raise ValueError("depth is required for rgbd_concat and rgbd_boundary variants")
        if depth.ndim != 4 or depth.shape[1] != 1:
            raise ValueError(
                f"depth must have shape [B, 1, H, W], got {tuple(depth.shape)}"
            )
        if rgb.shape[0] != depth.shape[0] or rgb.shape[-2:] != depth.shape[-2:]:
            raise ValueError("rgb and depth must share batch size and spatial size")
