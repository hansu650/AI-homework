"""Depth boundary residual fusion for lightweight RGB-D segmentation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DepthBoundaryResidualFusion(nn.Module):
    """Inject fixed Sobel depth edges into each encoder stage."""

    def __init__(self, channels: tuple[int, int, int, int]) -> None:
        super().__init__()
        self.edge_projections = nn.ModuleList(
            nn.Conv2d(1, channel, kernel_size=1, bias=False) for channel in channels
        )
        self.alpha = nn.Parameter(torch.full((len(channels),), 0.1))

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(
        self,
        features: list[torch.Tensor],
        depth: torch.Tensor,
    ) -> list[torch.Tensor]:
        edge = self._depth_edge(depth)
        fused = []
        for index, feature in enumerate(features):
            edge_i = F.interpolate(
                edge,
                size=feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            residual = self.edge_projections[index](edge_i)
            fused.append(feature + self.alpha[index] * residual)
        return fused

    def _depth_edge(self, depth: torch.Tensor) -> torch.Tensor:
        grad_x = F.conv2d(depth, self.sobel_x, padding=1)
        grad_y = F.conv2d(depth, self.sobel_y, padding=1)
        edge = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
        max_value = edge.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return edge / max_value
