"""A compact four-stage encoder implemented from scratch."""

from __future__ import annotations

import torch
from torch import nn


class ConvNormAct(nn.Module):
    """Convolution followed by BatchNorm and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseResidualBlock(nn.Module):
    """Depthwise + pointwise convolution block with a residual connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = ConvNormAct(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            groups=channels,
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.activation(x + residual)


class EncoderStage(nn.Module):
    """One downsampling convolution followed by lightweight residual blocks."""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int) -> None:
        super().__init__()
        blocks = [ConvNormAct(in_channels, out_channels, kernel_size=3, stride=2)]
        blocks.extend(DepthwiseResidualBlock(out_channels) for _ in range(num_blocks))
        self.stage = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage(x)


class MiniHierarchicalEncoder(nn.Module):
    """Four-stage lightweight encoder."""

    def __init__(
        self,
        in_channels: int = 4,
        channels: tuple[int, int, int, int] = (48, 96, 192, 384),
        blocks_per_stage: tuple[int, int, int, int] = (1, 1, 2, 2),
    ) -> None:
        super().__init__()
        stages = []
        current_channels = in_channels
        for out_channels, num_blocks in zip(channels, blocks_per_stage):
            stages.append(EncoderStage(current_channels, out_channels, num_blocks))
            current_channels = out_channels
        self.stages = nn.ModuleList(stages)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features
