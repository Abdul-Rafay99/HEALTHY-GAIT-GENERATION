"""Decoder for gait sequence windows.

This module maps a latent vector back to a skeleton window shaped
[batch, time, joints, channels]. It mirrors the encoder: joints are treated
as a feature/channel dimension throughout, never a spatial axis, so only the
time axis is convolved over and upsampled.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout1d(dropout))
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
        output_joints: int = 24,
        output_channels: int = 3,
        base_channels: int = 128,
        hidden_time_steps: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.output_time_steps = output_time_steps
        self.output_joints = output_joints
        self.output_channels = output_channels
        self.hidden_time_steps = hidden_time_steps

        hidden_channels = base_channels * 2  # matches the encoder's final width

        self.project = nn.Sequential(
            nn.Linear(latent_dim, hidden_channels * hidden_time_steps),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            ConvBlock1d(hidden_channels, hidden_channels, dropout=dropout),
            ConvBlock1d(hidden_channels, base_channels, dropout=dropout),
            ConvBlock1d(base_channels, base_channels, dropout=dropout),
            nn.Conv1d(base_channels, output_joints * output_channels, kernel_size=1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.dim() != 2:
            raise ValueError(f"Expected latent with shape [batch, latent_dim], got {tuple(latent.shape)}")

        batch_size = latent.shape[0]
        hidden = self.project(latent)
        hidden = hidden.view(batch_size, -1, self.hidden_time_steps)  # [B, hidden_channels, hidden_time_steps]
        hidden = F.interpolate(hidden, size=self.output_time_steps, mode="linear", align_corners=False)
        output = self.decoder(hidden)  # [B, J*C, T]
        output = output.permute(0, 2, 1).contiguous()  # [B, T, J*C]
        output = output.view(batch_size, self.output_time_steps, self.output_joints, self.output_channels)
        return output
