"""
Temporal Attention Module — attention.py

This module implements a lightweight temporal attention mechanism that
allows the model to focus on the most discriminative phases of a
rehabilitation exercise.

Motivation
==========
In rehabilitation assessment, not all frames are equally informative:
    - The transition phase of a squat (frames ~40–90) reveals compensation
      patterns far more than the static standing phase (frames ~1–20).
    - Incorrect movements often differ from correct ones only during a
      brief critical window (e.g., knee valgus at max flexion).

A temporal attention layer lets the model learn to up-weight these high-
information frames and down-weight redundant ones, effectively acting as
a soft temporal selector.

Architecture
============
We use **single-head scaled dot-product self-attention** over the temporal
dimension.  Multi-head attention would add parameters; given the small
UI-PRMD dataset (~1300 samples), a single head is sufficient and reduces
overfitting risk.

    Q = W_Q · x_avg    ∈ ℝ^{B × T × d}
    K = W_K · x_avg    ∈ ℝ^{B × T × d}
    V = W_V · x_pool   ∈ ℝ^{B × T × C}

    Attn = softmax(Q · K^T / √d) · V     ∈ ℝ^{B × T × C}

where x_avg is the joint-averaged feature (B, C, T) reshaped to (B, T, C),
and d is the attention embedding dimension (default: C // 4).

The attended output is reshaped back to (B, C, T, 1) and broadcast-
multiplied with the original input to re-weight each frame.

This is cheaper than full spatio-temporal attention because we attend
only over T positions (up to 150) rather than T × N positions (up to 3000).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    Single-head temporal self-attention.

    Parameters
    ----------
    channels : int
        Number of feature channels (C) in the input tensor.
    reduction : int
        Reduction factor for the attention embedding dimension.
        d_attn = channels // reduction.  Default 4.
    dropout : float
        Dropout on the attention weights (default 0.1).

    Input
    -----
    x : (B, C, T, N)

    Output
    ------
    out : (B, C, T, N) — same shape, temporally re-weighted.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.channels = channels
        self.d_attn = max(channels // reduction, 1)

        # Linear projections for Q, K (low-dim) and V (full-dim)
        self.W_q = nn.Linear(channels, self.d_attn)
        self.W_k = nn.Linear(channels, self.d_attn)
        self.W_v = nn.Linear(channels, channels)

        self.attn_dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_attn)

        # Learnable gate: controls how much attention modifies the input.
        # Initialised to zero so early training behaves like identity.
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : (B, C, T, N)

        Returns
        -------
        out : (B, C, T, N) — attention-reweighted features.
        """
        B, C, T, N = x.shape

        # ── Pool over joints to get per-frame features ──────────────────
        # (B, C, T, N) → mean over N → (B, C, T) → permute → (B, T, C)
        x_pool = x.mean(dim=-1).permute(0, 2, 1)        # (B, T, C)

        # ── Q, K, V projections ─────────────────────────────────────────
        Q = self.W_q(x_pool)          # (B, T, d)
        K = self.W_k(x_pool)          # (B, T, d)
        V = self.W_v(x_pool)          # (B, T, C)

        # ── Scaled dot-product attention ────────────────────────────────
        # (B, T, d) @ (B, d, T) → (B, T, T)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)    # (B, T, T)
        attn_weights = self.attn_dropout(attn_weights)

        # (B, T, T) @ (B, T, C) → (B, T, C)
        attended = torch.bmm(attn_weights, V)            # (B, T, C)

        # ── Re-weight original input ────────────────────────────────────
        # attended → (B, C, T) → (B, C, T, 1)  broadcast over N
        attn_scale = attended.permute(0, 2, 1).unsqueeze(-1)  # (B, C, T, 1)

        # Gated residual: out = x + gate * attn_scale
        # gate starts at 0, so attention is gradually introduced.
        out = x + self.gate * attn_scale

        return out
