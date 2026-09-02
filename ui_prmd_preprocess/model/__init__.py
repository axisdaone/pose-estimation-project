"""
LST-LA-GCN Model — Lightweight Spatial-Temporal Local-Adaptive GCN.

A compact graph convolutional network for binary classification of
rehabilitation exercises (correct vs incorrect movement) on the UI-PRMD
skeleton dataset.

Architecture overview:
    ┌─────────────────────────────────────────┐
    │  Joint Stream  (B,3,T,N)                │
    │    → BN → ST-Block×3 → GAP → FC        ├──► Average
    │                                         │    Fusion  ──► logits (B,2)
    │  Bone Stream   (B,3,T,N)                │
    │    → BN → ST-Block×3 → GAP → FC        ├──►
    └─────────────────────────────────────────┘

Modules:
    - graph_conv:    Spatial graph convolution with local-adaptive adjacency
    - temporal_conv: Multi-scale 1D temporal convolution
    - attention:     Lightweight temporal attention
    - st_block:      Combined spatial-temporal (LST) block
    - network:       Full dual-stream LST-LA-GCN network
"""

from .network import LSTLAGCN

__all__ = ["LSTLAGCN"]
