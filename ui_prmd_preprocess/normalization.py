"""
Normalization — Step 7 of the UI-PRMD pipeline.

Per-channel z-score normalization, computed on training data only.
"""

from typing import Tuple, List

import numpy as np

from . import config


def compute_channel_stats(
    samples: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean and standard deviation across all samples.

    IMPORTANT: Call this ONLY on training-split samples to avoid leaking
    test-set distribution information into the normalization parameters.

    Args:
        samples: List of (C, T, N) arrays (C=3 channels: X, Y, Z).

    Returns:
        mean: (3, 1) per-channel mean.
        std:  (3, 1) per-channel standard deviation.
    """
    # Flatten each sample's spatial-temporal dims and stack horizontally
    # Each sample (3, T, N) → (3, T*N), then concatenate → (3, total_pixels)
    stacked = np.concatenate([s.reshape(3, -1) for s in samples], axis=1)

    mean = stacked.mean(axis=1, keepdims=True)  # (3, 1)
    std = stacked.std(axis=1, keepdims=True)    # (3, 1)

    return mean, std


def normalize(
    sample: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    epsilon: float = config.NORM_EPSILON,
) -> np.ndarray:
    """
    Apply z-score normalization to a single sample.

    Args:
        sample: (3, T, N) array.
        mean:   (3, 1) per-channel mean (from training set).
        std:    (3, 1) per-channel std (from training set).
        epsilon: Small constant to prevent division by zero.

    Returns:
        Normalized (3, T, N) array.
    """
    return (sample - mean[:, :, np.newaxis]) / (std[:, :, np.newaxis] + epsilon)


def denormalize(
    sample: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    epsilon: float = config.NORM_EPSILON,
) -> np.ndarray:
    """
    Reverse z-score normalization (useful for visualization).

    Args:
        sample: (3, T, N) normalized array.
        mean:   (3, 1) per-channel mean.
        std:    (3, 1) per-channel std.
        epsilon: Small constant used during normalization.

    Returns:
        Denormalized (3, T, N) array.
    """
    return sample * (std[:, :, np.newaxis] + epsilon) + mean[:, :, np.newaxis]
