"""
Verification script for the Kalman filter implementation.

Tests:
  1. JointKalmanFilter — single joint tracking with noisy sine wave
  2. SkeletonKalmanTracker — full skeleton filtering
  3. Dropped frame handling — predict-only fills gaps smoothly
  4. kalman_smooth_sequence — offline full-sequence smoothing
  5. Integration with preprocess_sample — end-to-end check
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ui_prmd_preprocess.filtering import (
    JointKalmanFilter,
    SkeletonKalmanTracker,
    kalman_smooth_sequence,
)
from ui_prmd_preprocess import config


def test_joint_kalman_filter():
    """Test that a single joint Kalman filter smooths noisy observations."""
    print("\n[1] Testing JointKalmanFilter - noisy sine wave...")

    # Process noise must balance tracking (high) vs smoothing (low).
    # measurement_noise should approximate the actual observation variance.
    # Noise std = 0.05, so variance ~ 2.5e-3. Process noise = 0.1 allows
    # the filter to track the 0.5 Hz sine without excessive lag.
    kf = JointKalmanFilter(process_noise=0.1, measurement_noise=2.5e-3)

    T = 200
    dt = 1.0 / 30.0
    t = np.arange(T) * dt

    # Ground truth: slow sine wave (0.5 Hz) with realistic skeleton scale
    # Kinect positions are in meters — typical range 0-2m
    true_x = np.sin(2 * np.pi * 0.5 * t) * 0.3
    true_y = np.ones(T) * 1.0
    true_z = np.cos(2 * np.pi * 0.5 * t) * 0.3
    true_pos = np.stack([true_x, true_y, true_z], axis=-1)  # (T, 3)

    # Add measurement noise (~5cm std, realistic for Kinect extremity joints)
    np.random.seed(42)
    noise = np.random.normal(0, 0.05, true_pos.shape)
    noisy_pos = true_pos + noise

    # Run filter
    filtered_pos = np.zeros_like(true_pos)
    for i in range(T):
        filtered_pos[i] = kf.predict_and_update(noisy_pos[i], dt)

    # Compute errors (skip first 20 frames for filter warm-up)
    warmup = 20
    noisy_error = np.sqrt(np.mean((noisy_pos[warmup:] - true_pos[warmup:]) ** 2))
    filtered_error = np.sqrt(np.mean((filtered_pos[warmup:] - true_pos[warmup:]) ** 2))

    print(f"  Noisy RMSE:    {noisy_error:.4f}")
    print(f"  Filtered RMSE: {filtered_error:.4f}")
    print(f"  Improvement:   {(1 - filtered_error / noisy_error) * 100:.1f}%")

    assert filtered_error < noisy_error, (
        f"Filter made things worse! Filtered RMSE {filtered_error:.4f} > "
        f"Noisy RMSE {noisy_error:.4f}"
    )
    print("  [PASS] Kalman filter reduces RMSE vs raw noisy observations")


def test_skeleton_tracker():
    """Test SkeletonKalmanTracker with a full 20-joint skeleton."""
    print("\n[2] Testing SkeletonKalmanTracker - 20-joint skeleton...")

    n_joints = 20
    T = 60
    dt = 1.0 / 30.0

    tracker = SkeletonKalmanTracker(
        n_joints=n_joints,
        process_noise=1e-1,
        measurement_noise=1e-2,
    )

    # Generate random smooth skeleton trajectory + noise
    np.random.seed(123)
    true_seq = np.cumsum(np.random.randn(T, n_joints, 3) * 0.01, axis=0)
    noisy_seq = true_seq + np.random.normal(0, 0.03, true_seq.shape)

    smoothed = np.zeros_like(noisy_seq)
    for t in range(T):
        smoothed[t] = tracker.predict_and_update(noisy_seq[t], dt)

    # Check output shape
    assert smoothed.shape == (T, n_joints, 3), (
        f"Output shape mismatch: {smoothed.shape}"
    )

    # Check that smoothing reduces noise
    noisy_err = np.sqrt(np.mean((noisy_seq[10:] - true_seq[10:]) ** 2))
    smooth_err = np.sqrt(np.mean((smoothed[10:] - true_seq[10:]) ** 2))

    print(f"  Output shape:  {smoothed.shape}")
    print(f"  Noisy RMSE:    {noisy_err:.4f}")
    print(f"  Smoothed RMSE: {smooth_err:.4f}")
    print(f"  Improvement:   {(1 - smooth_err / noisy_err) * 100:.1f}%")

    assert smooth_err < noisy_err, "Tracker didn't reduce noise!"
    print("  [PASS] SkeletonKalmanTracker smooths all joints correctly")


def test_dropped_frames():
    """Test that predict_only produces smooth trajectories across gaps."""
    print("\n[3] Testing dropped frame handling...")

    n_joints = 20
    T = 90  # 3 seconds at 30 fps
    dt = 1.0 / 30.0

    tracker = SkeletonKalmanTracker(n_joints=n_joints)

    # Generate a smooth trajectory: joint 0 moves linearly in x
    true_seq = np.zeros((T, n_joints, 3))
    for t in range(T):
        true_seq[t, :, 0] = t * 0.01  # x increases linearly
        true_seq[t, :, 1] = 0.5       # y constant
        true_seq[t, :, 2] = 0.0       # z constant

    # Simulate: frames 30-39 are dropped (10 frames missing)
    output = np.zeros_like(true_seq)

    for t in range(T):
        if 30 <= t < 40:
            # Dropped frame — predict only
            output[t] = tracker.predict_only(dt)
        else:
            # Normal frame — predict and update
            output[t] = tracker.predict_and_update(true_seq[t], dt)

    # Check that the gap region is smoothly interpolated (not zeros or jumps)
    gap_positions = output[30:40, 0, 0]  # x-coord of joint 0 during gap
    before_gap = output[29, 0, 0]
    after_gap = output[40, 0, 0]

    print(f"  Position before gap (t=29): {before_gap:.4f}")
    print(f"  Positions during gap (t=30..39): {gap_positions}")
    print(f"  Position after gap (t=40):  {after_gap:.4f}")

    # Gap positions should be monotonically increasing (following the trend)
    diffs = np.diff(gap_positions)
    is_monotonic = np.all(diffs > 0)
    print(f"  Monotonically increasing during gap: {is_monotonic}")

    # Gap positions should not have sudden jumps
    max_jump = np.max(np.abs(diffs))
    print(f"  Max frame-to-frame jump in gap: {max_jump:.6f}")

    assert is_monotonic, "Gap predictions are not monotonically increasing!"
    assert max_jump < 0.1, f"Jump too large during gap: {max_jump}"
    print("  [PASS] Dropped frames are filled with smooth predictions")


def test_kalman_smooth_sequence():
    """Test the offline full-sequence smoothing function."""
    print("\n[4] Testing kalman_smooth_sequence — offline smoothing...")

    T, N = 150, 20
    np.random.seed(77)

    # Smooth ground truth + heavy noise
    t_axis = np.linspace(0, 2 * np.pi, T)
    true_seq = np.zeros((T, N, 3))
    for j in range(N):
        true_seq[:, j, 0] = np.sin(t_axis + j * 0.3) * 0.5
        true_seq[:, j, 1] = np.cos(t_axis + j * 0.3) * 0.5
        true_seq[:, j, 2] = j * 0.1

    noisy_seq = true_seq + np.random.normal(0, 0.04, true_seq.shape)

    smoothed = kalman_smooth_sequence(
        noisy_seq,
        process_noise=1e-2,
        measurement_noise=5e-4,
        fps=30.0,
    )

    assert smoothed.shape == (T, N, 3), f"Shape mismatch: {smoothed.shape}"

    noisy_rmse = np.sqrt(np.mean((noisy_seq[10:] - true_seq[10:]) ** 2))
    smooth_rmse = np.sqrt(np.mean((smoothed[10:] - true_seq[10:]) ** 2))

    print(f"  Input shape:   {noisy_seq.shape}")
    print(f"  Output shape:  {smoothed.shape}")
    print(f"  Noisy RMSE:    {noisy_rmse:.4f}")
    print(f"  Smoothed RMSE: {smooth_rmse:.4f}")
    print(f"  Improvement:   {(1 - smooth_rmse / noisy_rmse) * 100:.1f}%")

    assert smooth_rmse < noisy_rmse, "Full-sequence smoothing didn't help!"
    print("  [PASS] kalman_smooth_sequence reduces noise across full sequence")


def test_config_integration():
    """Verify that config constants are accessible."""
    print("\n[5] Testing config integration...")

    print(f"  ENABLE_KALMAN_FILTER:     {config.ENABLE_KALMAN_FILTER}")
    print(f"  KALMAN_PROCESS_NOISE:     {config.KALMAN_PROCESS_NOISE}")
    print(f"  KALMAN_MEASUREMENT_NOISE: {config.KALMAN_MEASUREMENT_NOISE}")
    print(f"  KALMAN_FPS:               {config.KALMAN_FPS}")

    assert hasattr(config, "ENABLE_KALMAN_FILTER")
    assert hasattr(config, "KALMAN_PROCESS_NOISE")
    assert hasattr(config, "KALMAN_MEASUREMENT_NOISE")
    assert hasattr(config, "KALMAN_FPS")
    print("  [PASS] All Kalman config constants are present")


def test_preprocessing_integration():
    """Test that preprocess_sample runs with Kalman filter enabled."""
    print("\n[6] Testing preprocessing pipeline integration...")

    # Temporarily ensure Kalman is enabled
    original_flag = config.ENABLE_KALMAN_FILTER
    config.ENABLE_KALMAN_FILTER = True

    try:
        from ui_prmd_preprocess.preprocessing import preprocess_sample

        # Synthetic raw positions: (T, N, 3) in YXZ order
        T, N = 80, 20
        np.random.seed(99)
        raw = np.random.randn(T, N, 3) * 0.1

        # This should not throw
        result = preprocess_sample(raw, target_len=150)

        assert result.shape == (150, N, 3), f"Shape mismatch: {result.shape}"
        assert not np.any(np.isnan(result)), "NaN in output!"
        assert not np.any(np.isinf(result)), "Inf in output!"

        print(f"  Input:  ({T}, {N}, 3)")
        print(f"  Output: {result.shape}")
        print("  [PASS] preprocess_sample runs correctly with Kalman filter enabled")

    finally:
        config.ENABLE_KALMAN_FILTER = original_flag


def main():
    print("=" * 60)
    print(" Kalman Filter Verification")
    print("=" * 60)

    tests = [
        ("JointKalmanFilter", test_joint_kalman_filter),
        ("SkeletonKalmanTracker", test_skeleton_tracker),
        ("Dropped frame handling", test_dropped_frames),
        ("kalman_smooth_sequence", test_kalman_smooth_sequence),
        ("Config integration", test_config_integration),
        ("Preprocessing integration", test_preprocessing_integration),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL]: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f" Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)
    else:
        print("\nAll Kalman filter tests passed!\n")


if __name__ == "__main__":
    main()
