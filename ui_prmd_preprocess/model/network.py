"""
LST-LA-GCN Network — network.py

Full Lightweight Spatial-Temporal Local-Adaptive Graph Convolutional Network
for binary classification of rehabilitation exercises.

Architecture
============

    ┌─────────────────────────────────────────────────────────────────┐
    │                      JOINT STREAM                              │
    │                                                                │
    │   joint (B,3,T,N) → BN → STBlock(3→64) → STBlock(64→128,s=2) │
    │                        → STBlock(128→256,s=2) → GAP → Dropout │
    │                        → FC(256→num_classes) → logits_joint    │
    └───────────────────────────────┬─────────────────────────────────┘
                                    │  Average
    ┌───────────────────────────────┴─────────────────────────────────┐
    │                      BONE STREAM                               │
    │                                                                │
    │   bone (B,3,T,N) → BN → STBlock(3→64) → STBlock(64→128,s=2)  │
    │                       → STBlock(128→256,s=2) → GAP → Dropout  │
    │                       → FC(256→num_classes) → logits_bone      │
    └─────────────────────────────────────────────────────────────────┘

Late Fusion
===========
The final output is the average of the two streams' logits:

    logits = (logits_joint + logits_bone) / 2

Late fusion is preferred over early fusion for several reasons:
    1. Each stream can specialise: joint stream captures absolute pose,
       bone stream captures relative limb geometry.
    2. Averaging logits acts as an implicit ensemble, improving
       generalisation — critical for a dataset with only ~1300 samples.
    3. It adds zero extra parameters compared to a learned fusion layer.

Temporal Downsampling
=====================
Stride-2 in blocks 2 and 3 progressively reduces the sequence length:
    150 → 150 → 75 → 38  (approximate, depends on padding)

This is more parameter-efficient than processing all 150 frames at the
highest channel width, and encourages the later blocks to capture
longer-range temporal dependencies.

Parameter Count
===============
With the default configuration:
    - ~380K parameters per stream
    - ~760K total (both streams)
This is roughly 1/5 the size of a standard 10-layer ST-GCN (~3.1M),
making it appropriate for the small UI-PRMD dataset.
"""

import torch
import torch.nn as nn

from .st_block import STBlock


class StreamBackbone(nn.Module):
    """
    Single-stream backbone: input normalisation → 3× ST blocks → GAP → classifier.

    This is used twice in the full network (once for joints, once for bones).

    Parameters
    ----------
    in_channels : int
        Input channels (3 for XYZ coordinates).
    num_classes : int
        Number of output classes (2 for correct/incorrect).
    num_joints : int
        Number of skeleton joints.
    channels : tuple of int
        Channel widths for the 3 ST blocks. Default: (64, 128, 256).
    dropout : float
        Dropout rate before the classifier.
    use_attention : bool
        Whether to use temporal attention in ST blocks.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        num_joints: int = 20,
        channels: tuple = (64, 128, 256),
        dropout: float = 0.3,
        use_attention: bool = True,
    ):
        super().__init__()

        assert len(channels) == 3, "Expected exactly 3 channel widths"

        # ── Input batch normalisation ───────────────────────────────────
        # Normalises the raw (x, y, z) coordinates across the batch.
        # This is standard practice in ST-GCN-family models and helps
        # stabilise training when inputs have varying scales.
        self.input_bn = nn.BatchNorm2d(in_channels)

        # ── ST Blocks ──────────────────────────────────────────────────
        # Block 0: C_in → channels[0], stride 1 (keep full temporal res)
        # Block 1: channels[0] → channels[1], stride 2 (downsample T)
        # Block 2: channels[1] → channels[2], stride 2 (downsample T)
        c_in_list = [in_channels] + list(channels[:-1])
        strides = [1, 2, 2]

        self.blocks = nn.ModuleList()
        for i, (c_in, c_out, s) in enumerate(
            zip(c_in_list, channels, strides)
        ):
            self.blocks.append(
                STBlock(
                    in_channels=c_in,
                    out_channels=c_out,
                    num_joints=num_joints,
                    stride=s,
                    dropout=dropout if i > 0 else 0.0,  # No dropout in first block
                    use_attention=use_attention,
                )
            )

        # ── Classifier head ─────────────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool2d(1)        # Global Average Pooling
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for a single stream.

        Parameters
        ----------
        x : (B, 3, T, N) — joint or bone features.
        A : (N, N) — physical adjacency matrix.

        Returns
        -------
        logits : (B, num_classes)
        """
        # Input normalisation
        x = self.input_bn(x)

        # Stack of ST blocks
        for block in self.blocks:
            x = block(x, A)

        # Global average pooling: (B, C, T', N) → (B, C, 1, 1) → (B, C)
        x = self.gap(x).squeeze(-1).squeeze(-1)

        # Classifier
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


class LSTLAGCN(nn.Module):
    """
    LST-LA-GCN: Lightweight Spatial-Temporal Local-Adaptive GCN.

    Dual-stream architecture for binary classification of rehabilitation
    exercises from skeleton sequences.

    Parameters
    ----------
    in_channels : int
        Number of input channels per joint (default: 3 for XYZ).
    num_classes : int
        Number of output classes (default: 2 for correct/incorrect).
    num_joints : int
        Number of skeleton joints (default: 20).
    channels : tuple of int
        Channel widths for the 3 ST blocks. Default: (64, 128, 256).
    dropout : float
        Dropout rate. Default: 0.3.
    use_attention : bool
        Whether to use temporal attention in ST blocks. Default: True.

    Input (forward method)
    ------
    joint : (B, 3, T, N) — joint position features.
    bone  : (B, 3, T, N) — bone vector features.
    A     : (N, N) or (B, N, N) — physical adjacency matrix.
            If batched, only A[0] is used (all samples share the same skeleton).

    Output
    ------
    logits : (B, num_classes) — class scores (NOT softmax-normalised).
             Apply softmax or use nn.CrossEntropyLoss (which includes softmax).

    Example
    -------
    >>> model = LSTLAGCN(in_channels=3, num_classes=2, num_joints=20)
    >>> joint = torch.randn(4, 3, 150, 20)
    >>> bone  = torch.randn(4, 3, 150, 20)
    >>> A     = torch.randn(20, 20)
    >>> logits = model(joint, bone, A)
    >>> print(logits.shape)  # (4, 2)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        num_joints: int = 20,
        channels: tuple = (64, 128, 256),
        dropout: float = 0.3,
        use_attention: bool = True,
    ):
        super().__init__()

        self.joint_stream = StreamBackbone(
            in_channels=in_channels,
            num_classes=num_classes,
            num_joints=num_joints,
            channels=channels,
            dropout=dropout,
            use_attention=use_attention,
        )

        self.bone_stream = StreamBackbone(
            in_channels=in_channels,
            num_classes=num_classes,
            num_joints=num_joints,
            channels=channels,
            dropout=dropout,
            use_attention=use_attention,
        )

    def forward(
        self,
        joint: torch.Tensor,
        bone: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with late fusion of joint and bone streams.

        Parameters
        ----------
        joint : (B, 3, T, N)
        bone  : (B, 3, T, N)
        A     : (N, N) or (B, N, N)

        Returns
        -------
        logits : (B, num_classes)
        """
        # Handle batched adjacency: all samples share the same skeleton,
        # so we just take the first one.
        if A.dim() == 3:
            A = A[0]  # (B, N, N) → (N, N)

        # Run both streams
        logits_joint = self.joint_stream(joint, A)  # (B, num_classes)
        logits_bone = self.bone_stream(bone, A)     # (B, num_classes)

        # Late fusion: average logits
        logits = (logits_joint + logits_bone) / 2.0

        return logits

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
