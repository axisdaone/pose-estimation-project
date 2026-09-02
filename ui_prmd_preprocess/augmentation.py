"""
Data Augmentation — Step 9 of the UI-PRMD pipeline.

Online augmentations applied only to training data, per epoch.
Designed to combat overfitting on the small UI-PRMD dataset (~1000 samples).
"""

import numpy as np
from scipy.interpolate import interp1d

from . import config


def augment(
    skeleton_seq: np.ndarray,
    p: float = config.AUGMENT_PROBABILITY,
    target_len: int = config.TARGET_SEQUENCE_LENGTH,
) -> np.ndarray:
    """
    Apply a chain of stochastic augmentations to a skeleton sequence.

    Each augmentation is applied independently with probability `p`.
    All augmentations preserve the (C, T, N) shape of the input.

    Args:
        skeleton_seq: (C, T, N) joint or bone tensor.
                      C=3 channels correspond to (X, Y, Z) after axis alignment.
        p: Probability of applying each individual augmentation.
        target_len: Temporal length to resample back to after any temporal change.

    Returns:
        Augmented (C, T, N) array.
    """
    skeleton_seq = skeleton_seq.copy()

    # 1. Random yaw rotation (camera-angle invariance, rotates in XZ plane)
    if np.random.rand() < config.YAW_PROBABILITY:
        skeleton_seq = _random_yaw_rotation(skeleton_seq)

    # 2. Random temporal crop (keep 80–100% of frames, then resample back)
    if np.random.rand() < p:
        skeleton_seq = _random_temporal_crop(skeleton_seq, target_len)

    # 3. Gaussian joint jitter (small spatial perturbation)
    if np.random.rand() < p:
        skeleton_seq = _gaussian_jitter(skeleton_seq)

    # 4. Random speed variation (temporal scaling ±20%, then resample back)
    if np.random.rand() < p:
        skeleton_seq = _random_speed_variation(skeleton_seq, target_len)

    return skeleton_seq


# ─────────────────────────────────────────────────────────────────────────────
# Individual augmentation transforms
# ─────────────────────────────────────────────────────────────────────────────

def _random_yaw_rotation(
    seq: np.ndarray,
    max_degrees: float = config.YAW_MAX_DEGREES,
) -> np.ndarray:
    """
    Apply a random yaw (horizontal) rotation around the Y-axis.

    After axis alignment (Step 2), the coordinate channels are (X, Y, Z).
    Yaw rotation acts in the XZ (horizontal) plane, leaving Y unchanged:

        X' = X·cos(θ) - Z·sin(θ)
        Y' = Y                     (unchanged — vertical axis)
        Z' = X·sin(θ) + Z·cos(θ)

    This prevents the model from overfitting to the exact camera angle
    used during data collection (the Kinect was placed at a fixed position).

    Args:
        seq: (C, T, N) input sequence where C=0 is X, C=1 is Y, C=2 is Z.
        max_degrees: Maximum rotation angle in degrees (both directions).

    Returns:
        (C, T, N) rotated sequence.
    """
    angle_deg = np.random.uniform(-max_degrees, max_degrees)
    angle_rad = np.deg2rad(angle_deg)

    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    x = seq[0]  # (T, N) — X coordinates
    z = seq[2]  # (T, N) — Z coordinates

    seq_out = seq.copy()
    seq_out[0] = x * cos_a - z * sin_a   # X'
    seq_out[2] = x * sin_a + z * cos_a   # Z'
    # seq_out[1] = Y — unchanged

    return seq_out


def _random_temporal_crop(
    seq: np.ndarray,
    target_len: int,
    min_ratio: float = config.TEMPORAL_CROP_MIN_RATIO,
) -> np.ndarray:
    """
    Randomly crop a temporal sub-window and resample back to target length.

    This simulates partial observation of an exercise repetition and
    introduces temporal variability into the training data.

    Args:
        seq: (C, T, N) input sequence.
        target_len: Length to resample the cropped segment back to.
        min_ratio: Minimum fraction of frames to retain (default 0.8).

    Returns:
        (C, target_len, N) cropped and resampled sequence.
    """
    C, T, N = seq.shape
    crop_len = np.random.randint(int(min_ratio * T), T + 1)
    start = np.random.randint(0, T - crop_len + 1)
    cropped = seq[:, start:start + crop_len, :]
    return _resample_ctn(cropped, target_len)


def _gaussian_jitter(
    seq: np.ndarray,
    std: float = config.JITTER_STD,
) -> np.ndarray:
    """
    Add small Gaussian noise to joint positions.

    This acts as a regularizer, simulating sensor noise and making the model
    more robust to slight positional inaccuracies in real-world deployment.

    Args:
        seq: (C, T, N) input sequence.
        std: Standard deviation of the noise (default 0.01).

    Returns:
        (C, T, N) jittered sequence.
    """
    noise = np.random.normal(0, std, seq.shape).astype(seq.dtype)
    return seq + noise


def _random_speed_variation(
    seq: np.ndarray,
    target_len: int,
    speed_min: float = config.SPEED_VARIATION_MIN,
    speed_max: float = config.SPEED_VARIATION_MAX,
) -> np.ndarray:
    """
    Randomly scale the temporal duration, then resample to target length.

    Simulates people performing exercises at different speeds —
    stretching or compressing the motion timeline.

    Args:
        seq: (C, T, N) input sequence.
        target_len: Length to resample back to after scaling.
        speed_min: Minimum speed factor (0.8 = 20% slower).
        speed_max: Maximum speed factor (1.2 = 20% faster).

    Returns:
        (C, target_len, N) speed-varied sequence.
    """
    C, T, N = seq.shape
    scale = np.random.uniform(speed_min, speed_max)
    new_T = max(2, int(T * scale))  # Ensure at least 2 frames for interp
    rescaled = _resample_ctn(seq, new_T)
    return _resample_ctn(rescaled, target_len)


# ─────────────────────────────────────────────────────────────────────────────
# Resampling utility (operates on CTN arrays)
# ─────────────────────────────────────────────────────────────────────────────

def _resample_ctn(seq: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resample a (C, T, N) array along the temporal axis.

    Args:
        seq: (C, T, N) input.
        target_len: Desired number of temporal frames.

    Returns:
        (C, target_len, N) resampled sequence.
    """
    C, T, N = seq.shape

    if T == target_len:
        return seq.copy()
    if T < 2:
        return np.tile(seq, (1, target_len, 1))

    old_t = np.linspace(0, 1, T)
    new_t = np.linspace(0, 1, target_len)
    result = np.zeros((C, target_len, N), dtype=seq.dtype)

    for c in range(C):
        for n in range(N):
            f = interp1d(old_t, seq[c, :, n], kind="linear")
            result[c, :, n] = f(new_t)

    return result
