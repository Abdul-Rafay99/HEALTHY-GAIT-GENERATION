"""Encoder for gait sequence windows.

The dataset yields skeleton windows shaped [batch, time, joints, channels].
This encoder maps a window to a fixed-size latent vector that can be paired
with a decoder for reconstruction.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GaitEncoder(nn.Module):
    """Encode a gait window into a latent vector.

    Expected input shape:
        [batch, time, joints, channels]
        - raw skeleton input: channels = 3, joints = 22
        - normalized skeleton input: channels = 3, joints = 24

    Output shape:
        [batch, latent_dim]
    """

    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 256,
        base_channels: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.backbone = nn.Sequential(
            ConvBlock(input_channels, base_channels, dropout=dropout),
            ConvBlock(base_channels, base_channels, dropout=dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(base_channels, base_channels * 2, dropout=dropout),
            ConvBlock(base_channels * 2, base_channels * 2, dropout=dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(base_channels * 2, base_channels * 4, dropout=dropout),
            ConvBlock(base_channels * 4, base_channels * 4, dropout=dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"Expected input with shape [batch, time, joints, channels], got {tuple(x.shape)}"
            )

        # Convert to [batch, channels, time, joints] for convolution.
        x = x.permute(0, 3, 1, 2).contiguous()
        features = self.backbone(x)
        latent = self.projection(features)
        return latent
