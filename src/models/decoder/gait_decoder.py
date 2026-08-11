"""Decoder for gait sequence windows.

This module maps a latent vector back to a skeleton window shaped
[batch, time, joints, channels]. It is designed to mirror the encoder used
for the gait dataset.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


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


class GaitDecoder(nn.Module):
    """Decode a latent vector into a gait window.

    Expected latent shape:
        [batch, latent_dim]

    Output shape:
        [batch, time, joints, channels]
    """

    def __init__(
        self,
        latent_dim: int = 256,
        output_time_steps: int = 30,
        output_joints: int = 22,
        output_channels: int = 3,
        base_channels: int = 64,
        hidden_time_steps: int = 8,
        hidden_joints: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.output_time_steps = output_time_steps
        self.output_joints = output_joints
        self.output_channels = output_channels
        self.hidden_time_steps = hidden_time_steps
        self.hidden_joints = hidden_joints

        hidden_channels = base_channels * 4
        self.project = nn.Sequential(
            nn.Linear(latent_dim, hidden_channels * hidden_time_steps * hidden_joints),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            ConvBlock(hidden_channels, hidden_channels, dropout=dropout),
            ConvBlock(hidden_channels, hidden_channels // 2, dropout=dropout),
            ConvBlock(hidden_channels // 2, hidden_channels // 4, dropout=dropout),
            nn.Conv2d(hidden_channels // 4, output_channels, kernel_size=1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.dim() != 2:
            raise ValueError(f"Expected latent with shape [batch, latent_dim], got {tuple(latent.shape)}")

        batch_size = latent.shape[0]
        hidden = self.project(latent)
        hidden = hidden.view(batch_size, -1, self.hidden_time_steps, self.hidden_joints)
        hidden = F.interpolate(
            hidden,
            size=(self.output_time_steps, self.output_joints),
            mode="bilinear",
            align_corners=False,
        )
        output = self.decoder(hidden)
        output = output.permute(0, 2, 3, 1).contiguous()
        return output
