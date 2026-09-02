"""
Multi-Scale Temporal Convolution — temporal_conv.py

This module implements the temporal modelling component of the LST block.
After the spatial graph convolution aggregates information across joints,
the temporal convolution captures motion dynamics along the time axis.

Multi-Scale Design
==================
Rehabilitation exercises have temporal patterns at multiple scales:
    - Fine-grained (kernel 5 ≈ 0.17s at 30fps): Joint jitter, tremor,
      short corrective movements.
    - Coarse (kernel 7 ≈ 0.23s): Phase transitions, speed changes during
      the movement arc.

The multi-scale branch processes the input through two parallel 1D
convolutions with kernels of size 5 and 7, then sums their outputs.
This is inspired by the Inception-style temporal module used in
MS-G3D (Liu et al., 2020) but kept minimal for the small UI-PRMD dataset.

Residual Connection
===================
A residual (skip) path ensures stable gradient flow through stacked blocks.
If input and output channels differ, a 1×1 convolution + BN adapts the
residual dimension to match.

Mathematical Formulation
========================
Let x ∈ ℝ^{B × C × T × N} be the input.

    branch_k(x) = BN(Conv1D_k(x))        for k ∈ {5, 7}
    h           = branch_5(x) + branch_7(x)
    out         = ReLU(h + residual(x))
    out         = Dropout(out)

Conv1D operates along the T dimension with padding = (k-1)//2 to
preserve temporal length (or stride > 1 for downsampling).
"""

import torch
import torch.nn as nn


class MultiScaleTemporalConv(nn.Module):
    """
    Multi-Scale Temporal Convolution block with residual connection.

    Parameters
    ----------
    in_channels : int
        Input feature channels.
    out_channels : int
        Output feature channels.
    kernel_sizes : tuple of int
        Kernel sizes for the parallel temporal branches. Default: (5, 7).
    stride : int
        Temporal stride. Use stride=2 in later blocks to halve the
        sequence length (150 → 75 → 38 frames), reducing compute and
        encouraging the model to learn higher-level temporal abstractions.
    dropout : float
        Dropout probability applied after the residual addition.

    Input
    -----
    x : (B, C_in, T, N)

    Output
    ------
    out : (B, C_out, T // stride, N)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple = (5, 7),
        stride: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # ── Parallel temporal convolution branches ──────────────────────
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            branch = nn.Sequential(
                # Conv2d with kernel (k, 1): convolves along T, point-wise on N
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(k, 1),
                    stride=(stride, 1),
                    padding=((k - 1) // 2, 0),
                ),
                nn.BatchNorm2d(out_channels),
            )
            self.branches.append(branch)

        # ── Residual path ───────────────────────────────────────────────
        if in_channels == out_channels and stride == 1:
            # Identity shortcut — cheapest option
            self.residual = nn.Identity()
        else:
            # 1×1 conv adapts channel count and/or temporal stride
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
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : (B, C_in, T, N)

        Returns
        -------
        out : (B, C_out, T // stride, N)
        """
        # Sum of multi-scale branches
        h = sum(branch(x) for branch in self.branches)

        # Residual addition + activation
        out = self.relu(h + self.residual(x))
        out = self.dropout(out)
        return out
