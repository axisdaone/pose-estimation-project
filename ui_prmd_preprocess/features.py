"""
Features — Steps 5 & 6 of the UI-PRMD pipeline.

Step 5: Dual-stream feature construction (Joint positions + Bone vectors).
Step 6: Build and normalize the graph adjacency matrix.
"""

import numpy as np

from . import config


# ═════════════════════════════════════════════════════════════════════════════
# Step 5 — Bone Vector Computation
# ═════════════════════════════════════════════════════════════════════════════

def compute_bone_features(
    joint_seq: np.ndarray,
    edges: list = None,
) -> np.ndarray:
    """
    Compute bone vectors from joint positions.

    A bone vector at joint j is defined as the difference between the child
    joint and its parent joint:  bone[child] = joint[child] - joint[parent].
    The root joint has a zero bone vector (no parent).

    This gives the model complementary information: joint positions capture
    absolute pose configuration, while bone vectors capture relative limb
    direction and length.

    Args:
        joint_seq: (T, N, 3) array of joint positions.
        edges: List of (parent, child) tuples. Defaults to config.SKELETON_EDGES.

    Returns:
        bone_seq: (T, N, 3) bone vector array (zero vector for root joint).
    """
    if edges is None:
        edges = config.SKELETON_EDGES

    bone_seq = np.zeros_like(joint_seq)
    for parent, child in edges:
        bone_seq[:, child, :] = joint_seq[:, child, :] - joint_seq[:, parent, :]

    return bone_seq


def to_ctn(sequence_tnc: np.ndarray) -> np.ndarray:
    """
    Transpose a (T, N, C) array to (C, T, N) — PyTorch GCN convention.
    """
    return sequence_tnc.transpose(2, 0, 1).copy()


def to_tnc(sequence_ctn: np.ndarray) -> np.ndarray:
    """
    Transpose a (C, T, N) array back to (T, N, C).
    """
    return sequence_ctn.transpose(1, 2, 0).copy()


def build_dual_stream(joint_seq_tnc: np.ndarray) -> dict:
    """
    Build the dual-stream input (joint + bone) from a preprocessed joint sequence.

    Args:
        joint_seq_tnc: (T, N, 3) preprocessed joint positions.

    Returns:
        dict with:
            'joint': (3, T, N) joint position tensor.
            'bone':  (3, T, N) bone vector tensor.
    """
    bone_seq_tnc = compute_bone_features(joint_seq_tnc)
    return {
        "joint": to_ctn(joint_seq_tnc),   # (3, T, N)
        "bone": to_ctn(bone_seq_tnc),      # (3, T, N)
    }


# ═════════════════════════════════════════════════════════════════════════════
# Step 6 — Graph Adjacency Matrix
# ═════════════════════════════════════════════════════════════════════════════

def build_adjacency_matrix(
    edges: list = None,
    n_joints: int = config.N_JOINTS,
    normalize: bool = True,
    add_self_loops: bool = True,
) -> np.ndarray:
    """
    Build the adjacency matrix encoding the physical skeleton topology.

    ST-GCN and its variants (including LST-LA-GCN) operate on a graph
    defined by the skeleton's natural joint connections.  This function
    constructs the base adjacency matrix; the LA (local-adaptive) module
    in the network will learn residual corrections on top of it.

    Args:
        edges: List of (parent, child) tuples. Defaults to config.SKELETON_EDGES.
        n_joints: Number of joints in the graph.
        normalize: If True, apply symmetric normalization D^{-1/2} A D^{-1/2}.
        add_self_loops: If True, add identity (self-connections) to A.

    Returns:
        A: (n_joints, n_joints) adjacency matrix (float32).
    """
    if edges is None:
        edges = config.SKELETON_EDGES

    A = np.zeros((n_joints, n_joints), dtype=np.float32)

    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0  # Undirected graph

    if add_self_loops:
        A += np.eye(n_joints, dtype=np.float32)

    if normalize:
        A = _symmetric_normalize(A)

    return A


def _symmetric_normalize(A: np.ndarray) -> np.ndarray:
    """
    Symmetric normalization: D^{-1/2} A D^{-1/2}.

    This ensures that the spectral properties of the graph Laplacian are
    well-behaved, preventing joints with many connections from dominating
    the aggregation step in graph convolution.
    """
    d = A.sum(axis=1)  # degree vector
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


def build_partition_adjacency(
    edges: list = None,
    n_joints: int = config.N_JOINTS,
    root_index: int = config.ROOT_JOINT_INDEX,
) -> np.ndarray:
    """
    Build the 3-partition adjacency matrices used by spatial partitioning
    strategies in ST-GCN-family models (centripetal, centrifugal, root).

    Returns:
        A_parts: (3, n_joints, n_joints) — stacked partition matrices.
    """
    if edges is None:
        edges = config.SKELETON_EDGES

    # BFS to compute distances from root
    from collections import deque
    adj_list = {i: [] for i in range(n_joints)}
    for p, c in edges:
        adj_list[p].append(c)
        adj_list[c].append(p)

    dist = [-1] * n_joints
    dist[root_index] = 0
    queue = deque([root_index])
    while queue:
        node = queue.popleft()
        for nb in adj_list[node]:
            if dist[nb] == -1:
                dist[nb] = dist[node] + 1
                queue.append(nb)

    # Three partitions: self, centripetal (closer to root), centrifugal (farther)
    A_self = np.eye(n_joints, dtype=np.float32)
    A_centripetal = np.zeros((n_joints, n_joints), dtype=np.float32)
    A_centrifugal = np.zeros((n_joints, n_joints), dtype=np.float32)

    for p, c in edges:
        if dist[c] > dist[p]:
            # child is farther from root → centrifugal for parent, centripetal for child
            A_centrifugal[p, c] = 1.0
            A_centripetal[c, p] = 1.0
        else:
            A_centrifugal[c, p] = 1.0
            A_centripetal[p, c] = 1.0

    # Normalize each partition
    A_parts = np.stack([
        _symmetric_normalize(A_self),
        _symmetric_normalize(A_centripetal + np.eye(n_joints, dtype=np.float32)),
        _symmetric_normalize(A_centrifugal + np.eye(n_joints, dtype=np.float32)),
    ], axis=0)

    return A_parts
