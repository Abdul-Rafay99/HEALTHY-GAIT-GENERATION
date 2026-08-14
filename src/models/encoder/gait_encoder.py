"""Encoder for gait sequence windows.

The dataset yields skeleton windows shaped [batch, time, joints, channels].
This encoder maps a window to a fixed-size latent vector that can be paired
with a decoder for reconstruction.
"""

from __future__ import annotations

import torch
from torch import nn


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


class SafeAdaptiveAvgPool1d(nn.Module):
    """Adaptive average pool that also works on MPS.

    MPS's adaptive pooling only supports resizes where the output size evenly
    divides (or is divided by) the input size. Our time_steps -> hidden_time_steps
    resize generally isn't divisible, so on MPS this runs the pool on CPU and
    moves the result back - numerically identical to running it natively on
    CUDA/CPU, and it has no parameters so it doesn't affect checkpoint
    compatibility.
    """

    def __init__(self, output_size: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == "mps":
            return self.pool(x.cpu()).to(x.device)
        return self.pool(x)


class GaitEncoder(nn.Module):
    """Encode a gait window into a latent vector.

    Joints are treated as feature channels, not a spatial axis. A 2D
    convolution sliding a window over [time, joints] implicitly assumes
    nearby joint indices are related, but the joint ordering has no such
    structure - e.g. JOINT_NAMES index 4 and 5 are KNEE_RIGHT and KNEE_LEFT,
    opposite sides of the body, not neighbors. An earlier version of this
    encoder pooled the joints axis down to a single value before the latent,
    which discarded which body region was moving. Flattening [joints,
    channels] into one feature dimension per time step and convolving only
    over time avoids that entirely: every joint's signal is carried in full
    by the channel dimension through the whole network, and only the time
    axis - which actually has local structure, since motion is smooth and
    causal - gets pooled down.

    Expected input shape:
        [batch, time, joints, channels]
        - raw skeleton input: channels = 3, joints = 22
        - normalized skeleton input: channels = 3, joints = 24

    Output shape:
        [batch, latent_dim]
    """

    def __init__(
        self,
        input_joints: int = 24,
        input_channels: int = 3,
        latent_dim: int = 256,
        base_channels: int = 128,
        hidden_time_steps: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.input_joints = input_joints
        self.input_channels = input_channels
        self.hidden_time_steps = hidden_time_steps

        in_features = input_joints * input_channels
        widened_channels = base_channels * 2

        self.backbone = nn.Sequential(
            ConvBlock1d(in_features, base_channels, dropout=dropout),
            ConvBlock1d(base_channels, base_channels, dropout=dropout),
            nn.MaxPool1d(kernel_size=2, stride=2),
            ConvBlock1d(base_channels, widened_channels, dropout=dropout),
            ConvBlock1d(widened_channels, widened_channels, dropout=dropout),
            nn.MaxPool1d(kernel_size=2, stride=2),
            ConvBlock1d(widened_channels, widened_channels, dropout=dropout),
            ConvBlock1d(widened_channels, widened_channels, dropout=dropout),
            # Pool time down to a small but real resolution - never to a single
            # step, which would make the latent invariant to when things
            # happen (a shuffled or frozen window would then encode almost
            # identically to the real one).
            SafeAdaptiveAvgPool1d(hidden_time_steps),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(widened_channels * hidden_time_steps, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"Expected input with shape [batch, time, joints, channels], got {tuple(x.shape)}"
            )

        batch, time_steps, joints, channels = x.shape
        # [B, T, J, C] -> [B, T, J*C] -> [B, J*C, T] (channels-first for Conv1d)
        x = x.reshape(batch, time_steps, joints * channels).permute(0, 2, 1).contiguous()
        features = self.backbone(x)
        latent = self.projection(features)
        return latent
