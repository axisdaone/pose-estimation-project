"""
Preprocessing — Steps 1–4 of the UI-PRMD pipeline.

Step 1: Reconstruct absolute 3D coordinates from relative joints.
Step 2: Align axes from YXZ → XYZ convention.
Step 3: Center skeleton on spine/waist root per frame.
Step 4: Resample all sequences to a fixed temporal length.
"""

import logging

import numpy as np
from scipy.interpolate import interp1d

from . import config
from .filtering import kalman_smooth_sequence

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Step 1 — Coordinate Reconstruction (Absolute Joint Positions)
# ═════════════════════════════════════════════════════════════════════════════

def reconstruct_absolute_positions(
    positions: np.ndarray,
    kinematic_tree: list = None,
    root_index: int = config.ROOT_JOINT_INDEX,
) -> np.ndarray:
    """
    Convert relative joint coordinates to absolute coordinates.

    In the raw UI-PRMD CSV, the waist/SpineBase joint is stored in absolute
    (world) coordinates, and every other joint is stored *relative* to its
    parent in the kinematic tree.  This function accumulates the offsets
    along the tree to recover absolute world positions for every joint.

    Args:
        positions: (T, N, 3) array of raw positions (root is absolute,
                   others are relative to their parent).
        kinematic_tree: Ordered list of (child, parent) tuples.
                        Must be in topological order (parents before children).
                        Defaults to config.KINEMATIC_TREE.
        root_index: Index of the root joint (default 0 = Waist).

    Returns:
        abs_positions: (T, N, 3) array of absolute 3D positions.
    """
    if kinematic_tree is None:
        kinematic_tree = config.KINEMATIC_TREE

    T, N, C = positions.shape
    abs_positions = np.zeros_like(positions)

    # Root joint is already absolute
    abs_positions[:, root_index, :] = positions[:, root_index, :]

    # Walk the kinematic tree in topological order
    for child, parent in kinematic_tree:
        abs_positions[:, child, :] = abs_positions[:, parent, :] + positions[:, child, :]

    return abs_positions


# ═════════════════════════════════════════════════════════════════════════════
# Step 2 — Sensor Axis Alignment
# ═════════════════════════════════════════════════════════════════════════════

def align_axes(skeleton_sequence: np.ndarray) -> np.ndarray:
    """
    Reorder coordinates from UI-PRMD's (Y, X, Z) convention to the
    standard (X, Y, Z) convention expected by most GCN implementations.

    In the raw CSV, each position triplet is stored as (posY, posX, posZ).
    We swap columns to produce (posX, posY, posZ) so that:
        - X = width (lateral)
        - Y = height (vertical)
        - Z = depth (anterior-posterior)

    Args:
        skeleton_sequence: (T, N, 3) array in (Y, X, Z) order.

    Returns:
        (T, N, 3) array in (X, Y, Z) order.
    """
    # Column mapping: [Y, X, Z] → [X, Y, Z] = swap columns 0 and 1
    return skeleton_sequence[:, :, [1, 0, 2]].copy()


# ═════════════════════════════════════════════════════════════════════════════
# Step 3 — Root Normalization (Center on Spine/Waist)
# ═════════════════════════════════════════════════════════════════════════════

def center_on_root(
    skeleton_sequence: np.ndarray,
    root_index: int = config.ROOT_JOINT_INDEX,
) -> np.ndarray:
    """
    Center the skeleton on the root joint (Waist/SpineBase) per frame.

    Subtracts the root joint position from all joints in every frame,
    eliminating global translation and placing the skeleton at the origin.
    This removes inter-subject variation in camera distance/placement
    while preserving all relative joint motion.

    Args:
        skeleton_sequence: (T, N, 3) array of absolute joint positions.
        root_index: Index of the root joint (default 0 = Waist).

    Returns:
        (T, N, 3) centered skeleton sequence.
    """
    root = skeleton_sequence[:, root_index:root_index + 1, :]  # (T, 1, 3)
    return skeleton_sequence - root


# ═════════════════════════════════════════════════════════════════════════════
# Step 4 — Temporal Resampling to Fixed Length
# ═════════════════════════════════════════════════════════════════════════════

def resample_sequence(
    skeleton_sequence: np.ndarray,
    target_len: int = config.TARGET_SEQUENCE_LENGTH,
    kind: str = config.INTERPOLATION_KIND,
) -> np.ndarray:
    """
    Resample a variable-length skeleton sequence to a fixed number of frames
    using interpolation.

    Different subjects perform exercises at different speeds, producing
    sequences of varying length T. LST-LA-GCN's temporal convolution
    requires fixed-length input, so we temporally resample every sequence
    to `target_len` frames.

    Args:
        skeleton_sequence: (T, N, C) array — T frames, N joints, C coordinates.
        target_len: Desired number of output frames (default 150).
        kind: Interpolation method ('linear', 'cubic', etc.).

    Returns:
        (target_len, N, C) resampled skeleton sequence.
    """
    T, N, C = skeleton_sequence.shape

    if T == target_len:
        return skeleton_sequence.copy()

    if T < 2:
        # Edge case: single-frame sequence — just tile it
        logger.warning(f"Sequence has only {T} frame(s); tiling to {target_len}.")
        return np.tile(skeleton_sequence, (target_len, 1, 1))

    old_t = np.linspace(0.0, 1.0, T)
    new_t = np.linspace(0.0, 1.0, target_len)
    resampled = np.zeros((target_len, N, C), dtype=skeleton_sequence.dtype)

    for j in range(N):
        for c in range(C):
            interpolator = interp1d(old_t, skeleton_sequence[:, j, c], kind=kind)
            resampled[:, j, c] = interpolator(new_t)

    return resampled


def resample_ctn(
    skeleton_ctn: np.ndarray,
    target_len: int = config.TARGET_SEQUENCE_LENGTH,
    kind: str = config.INTERPOLATION_KIND,
) -> np.ndarray:
    """
    Resample a (C, T, N) tensor along the T axis.

    Convenience wrapper for sequences already transposed to GCN convention.
    """
    C, T, N = skeleton_ctn.shape
    tnc = skeleton_ctn.transpose(1, 2, 0)              # → (T, N, C)
    tnc_resampled = resample_sequence(tnc, target_len, kind)  # → (target_len, N, C)
    return tnc_resampled.transpose(2, 0, 1)             # → (C, target_len, N)


# ═════════════════════════════════════════════════════════════════════════════
# Combined: Run Steps 1–4 on a single sample
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_sample(
    raw_positions: np.ndarray,
    target_len: int = config.TARGET_SEQUENCE_LENGTH,
) -> np.ndarray:
    """
    Run the full geometric preprocessing pipeline (Steps 1–4) on one sample.

    Args:
        raw_positions: (T, N, 3) raw position data from the CSV (in YXZ order).
        target_len: Fixed temporal length for the output.

    Returns:
        (target_len, N, 3) preprocessed skeleton in (X, Y, Z), centered on root.
    """
    # Step 1: Reconstruct absolute coordinates
    abs_positions = reconstruct_absolute_positions(raw_positions)

    # Step 1.5: Kalman filter smoothing (reduces sensor noise & interpolation jumps)
    if config.ENABLE_KALMAN_FILTER:
        abs_positions = kalman_smooth_sequence(
            abs_positions,
            process_noise=config.KALMAN_PROCESS_NOISE,
            measurement_noise=config.KALMAN_MEASUREMENT_NOISE,
            fps=config.KALMAN_FPS,
        )

    # Step 2: Align axes YXZ → XYZ
    aligned = align_axes(abs_positions)

    # Step 3: Center on root joint
    centered = center_on_root(aligned)

    # Step 4: Resample to fixed temporal length
    resampled = resample_sequence(centered, target_len=target_len)

    return resampled
