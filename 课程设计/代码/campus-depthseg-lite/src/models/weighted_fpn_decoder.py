"""Weighted FPN-style decoder for the lightweight segmentation model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class WeightedFPNDecoder(nn.Module):
    """Fuse four scales with learnable normalized positive weights."""

    def __init__(
        self,
        in_channels: tuple[int, int, int, int],
        num_classes: int,
        decoder_channels: int = 128,
    ) -> None:
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            nn.Conv2d(channel, decoder_channels, kernel_size=1)
            for channel in in_channels
        )
        self.scale_weights = nn.Parameter(torch.ones(len(in_channels)))
        self.smooth = nn.Sequential(
            nn.Conv2d(
                decoder_channels,
                decoder_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)

    def forward(
        self,
        features: list[torch.Tensor],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        resized_features = []
        for feature, lateral_conv in zip(features, self.lateral_convs):
            feature = lateral_conv(feature)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            resized_features.append(feature)

        weights = F.softplus(self.scale_weights)
        weights = weights / weights.sum().clamp_min(1e-6)

        fused = torch.zeros_like(resized_features[0])
        for weight, feature in zip(weights, resized_features):
            fused = fused + weight * feature

        logits = self.classifier(self.smooth(fused))
        return F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
