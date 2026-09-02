"""
Spatial-Temporal Block (LST Block) — st_block.py

This module combines the spatial graph convolution, multi-scale temporal
convolution, and optional temporal attention into a single reusable block.
Three of these blocks are stacked per stream in the full LST-LA-GCN network.

Block Architecture
==================
                    ┌──────────────────────────────────┐
    Input ──┬──────►│  Spatial Graph Conv (LA-GCN)     │
            │       │  (B, C_in, T, N) → (B, C_out, T, N)  │
            │       └──────────────┬───────────────────┘
            │                      ▼
            │       ┌──────────────────────────────────┐
            │       │  Multi-Scale Temporal Conv       │
            │       │  (B, C_out, T, N) → (B, C_out, T', N) │
            │       └──────────────┬───────────────────┘
            │                      ▼
            │       ┌──────────────────────────────────┐
            │       │  Temporal Attention (optional)   │
            │       │  (B, C_out, T', N) → (B, C_out, T', N) │
            │       └──────────────┬───────────────────┘
            │                      ▼
            └──► residual ───►  (+)  ───► ReLU ───► Output
                  path

Residual Connection
===================
If C_in ≠ C_out or stride > 1, the residual path uses a 1×1 convolution
(+ BN) to match dimensions.  Otherwise, it's an identity shortcut.

The residual connection is essential for:
    1. Gradient flow in deeper stacks.
    2. Allowing the block to learn incremental refinements rather than
       full transformations (important when data is limited).

Channel Progression
===================
In the full network, 3 blocks are stacked with channels:
    Block 0:  3 →  64   (stride 1, T stays 150)
    Block 1: 64 → 128   (stride 2, T becomes 75)
    Block 2: 128 → 256  (stride 2, T becomes 38)

This mirrors the "narrow → wide" design of ResNet but with far fewer
layers, appropriate for the ~1300-sample UI-PRMD dataset.
"""

import torch
import torch.nn as nn

from .graph_conv import SpatialGraphConv
from .temporal_conv import MultiScaleTemporalConv
from .attention import TemporalAttention


class STBlock(nn.Module):
    """
    Spatial-Temporal Block combining graph conv, temporal conv, and attention.

    Parameters
    ----------
    in_channels : int
        Input feature channels.
    out_channels : int
        Output feature channels.
    num_joints : int
        Number of skeleton joints (default: 20).
    temporal_kernel_sizes : tuple
        Kernel sizes for multi-scale temporal conv. Default: (5, 7).
    stride : int
        Temporal stride for downsampling. Default: 1.
    dropout : float
        Dropout probability. Default: 0.0.
    use_attention : bool
        Whether to include temporal attention. Default: True.
    attention_reduction : int
        Reduction factor for attention embedding dim. Default: 4.

    Input
    -----
    x : (B, C_in, T, N)
    A : (N, N) — physical adjacency matrix.

    Output
    ------
    out : (B, C_out, T // stride, N)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_joints: int = 20,
        temporal_kernel_sizes: tuple = (5, 7),
        stride: int = 1,
        dropout: float = 0.0,
        use_attention: bool = True,
        attention_reduction: int = 4,
    ):
        super().__init__()

        # ── Spatial: graph convolution with local-adaptive adjacency ────
        self.gcn = SpatialGraphConv(
            in_channels=in_channels,
            out_channels=out_channels,
            num_joints=num_joints,
            use_attention=True,  # LA mechanism always on for graph conv
        )

        # ── Temporal: multi-scale 1D convolution ────────────────────────
        self.tcn = MultiScaleTemporalConv(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_sizes=temporal_kernel_sizes,
            stride=stride,
            dropout=dropout,
        )

        # ── Temporal attention (optional) ───────────────────────────────
        if use_attention:
            self.attention = TemporalAttention(
                channels=out_channels,
                reduction=attention_reduction,
                dropout=0.1,
            )
        else:
            self.attention = None

        # ── Block-level residual path ───────────────────────────────────
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : (B, C_in, T, N)
        A : (N, N)

        Returns
        -------
        out : (B, C_out, T // stride, N)
        """
        # Save input for the residual
        res = self.residual(x)

        # Spatial graph convolution (includes BN + ReLU internally)
        h = self.gcn(x, A)

        # Temporal convolution (includes BN + ReLU + dropout internally)
        h = self.tcn(h)

        # Optional temporal attention
        if self.attention is not None:
            h = self.attention(h)

        # Residual addition + final activation
        out = self.relu(h + res)
        return out
