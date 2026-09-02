"""
Spatial Graph Convolution with Local-Adaptive Adjacency — graph_conv.py

This module implements the core spatial graph convolution used in LST-LA-GCN.
Unlike vanilla ST-GCN which uses a fixed, hand-crafted adjacency matrix,
the local-adaptive mechanism learns residual corrections to the skeleton
topology, enabling the model to discover non-physical joint relationships
(e.g., left-hand ↔ right-hand coordination during bilateral exercises).

Adjacency Decomposition
========================
The effective adjacency matrix is a sum of three components:

    A_final = A_physical + A_adaptive + A_attention

where:
    A_physical  — (N, N) fixed, symmetric-normalised skeleton adjacency.
                  Encodes the anatomical bone connections (19 edges for 20 joints).
                  Provided externally and NOT learnable.

    A_adaptive  — (N, N) fully-learnable parameter initialised near zero.
                  Captures global, data-independent topology corrections.
                  Analogous to the "B" matrix in 2s-AGCN (Shi et al., 2019).

    A_attention — (N, N) data-dependent, computed per sample as:
                      A_att = softmax(θᵀ · φ)
                  where θ = W_θ · x ∈ (B, C', T, N) and φ = W_φ · x.
                  This gives sample-level (and implicitly frame-level)
                  attention over pairs of joints.  Inspired by the
                  self-attention mechanism in 2s-AGCN.

Graph Convolution Operation
============================
Given input features  x ∈ ℝ^{B × C_in × T × N}  and the combined
adjacency  A_final ∈ ℝ^{N × N}:

    out = σ(BN( (x · W) × A_final ))

Step-by-step:
    1. Linear projection:  x · W  via 1×1 convolution  → (B, C_out, T, N)
    2. Graph aggregation:  matmul with A_final on the joint axis
    3. Batch normalisation + ReLU activation

This corresponds to a first-order Chebyshev approximation on the graph
Laplacian, as described in Kipf & Welling (2017), extended to the
spatio-temporal setting by treating each frame independently for the
spatial step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGraphConv(nn.Module):
    """
    Spatial Graph Convolution with Local-Adaptive adjacency.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels per joint per frame.
    out_channels : int
        Number of output feature channels.
    num_joints : int
        Number of joints (nodes) in the skeleton graph (default: 20).
    adaptive_embed_dim : int
        Embedding dimension for the data-dependent attention branch.
        Smaller → fewer parameters.  Default 4 keeps it lightweight.
    use_attention : bool
        Whether to include the data-dependent A_attention term.
        Can be disabled to save compute on very small datasets.

    Input
    -----
    x : (B, C_in, T, N)   — per-joint features over time.
    A : (N, N)             — physical adjacency matrix (symmetric-normalised).

    Output
    ------
    out : (B, C_out, T, N)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_joints: int = 20,
        adaptive_embed_dim: int = 4,
        use_attention: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_joints = num_joints
        self.use_attention = use_attention

        # ── 1×1 convolution for the main feature transform ──────────────
        # Acts independently per joint per frame  →  W · x
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # ── Learnable adaptive adjacency (A_adaptive) ───────────────────
        # Initialised from N(0, 0.001) so the model starts close to the
        # physical topology and gradually learns corrections.
        self.A_adaptive = nn.Parameter(
            torch.randn(num_joints, num_joints) * 0.001
        )

        # ── Data-dependent attention adjacency (A_attention) ────────────
        if use_attention:
            # θ and φ project input features to a low-dim embedding space
            # for computing pairwise joint attention scores.
            self.theta = nn.Conv2d(
                in_channels, adaptive_embed_dim, kernel_size=1
            )
            self.phi = nn.Conv2d(
                in_channels, adaptive_embed_dim, kernel_size=1
            )

        # ── Post-aggregation layers ─────────────────────────────────────
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : (B, C_in, T, N)
        A : (N, N) — physical adjacency (same for every sample in the batch,
            taken from index [0] of the batched A).

        Returns
        -------
        out : (B, C_out, T, N)
        """
        B, C, T, N = x.shape

        # ── Build the combined adjacency ────────────────────────────────
        # A_physical: detached, no gradient.  Shape (N, N).
        A_combined = A + self.A_adaptive

        if self.use_attention:
            # θ(x): (B, d, T, N) → avg over T → (B, d, N) → transpose
            theta_out = self.theta(x).mean(dim=2)       # (B, d, N)
            phi_out = self.phi(x).mean(dim=2)            # (B, d, N)

            # Pairwise attention: (B, N, d) @ (B, d, N) → (B, N, N)
            A_att = torch.bmm(
                theta_out.permute(0, 2, 1),  # (B, N, d)
                phi_out,                      # (B, d, N)
            )
            # Softmax along the last dimension (source nodes)
            A_att = F.softmax(A_att, dim=-1)             # (B, N, N)

            # Average across the batch to get a single (N, N) attention
            # matrix, keeping it compatible with the static A_combined.
            A_att_mean = A_att.mean(dim=0)               # (N, N)
            A_combined = A_combined + A_att_mean

        # ── Feature transform ───────────────────────────────────────────
        # 1×1 conv: (B, C_in, T, N) → (B, C_out, T, N)
        h = self.conv(x)

        # ── Graph aggregation ───────────────────────────────────────────
        # h reshaped: (B*C_out*T, N) @ A^T → (B*C_out*T, N)
        # Equivalent to: for each (b, c, t):  h[b,c,t,:] = h[b,c,t,:] @ A^T
        h = torch.einsum("bctn,nm->bctm", h, A_combined)

        # ── Normalise + activate ────────────────────────────────────────
        out = self.relu(self.bn(h))
        return out
