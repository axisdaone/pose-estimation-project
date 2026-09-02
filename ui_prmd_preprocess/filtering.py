"""
Signal Filtering — Kalman filter for skeleton joint tracking.

Addresses the problem of FPS drops, frame lag, and sensor noise:
    - When frames are dropped, linear interpolation creates unrealistic
      jumps at gap boundaries.  A Kalman filter with a constant-velocity
      motion model produces physically plausible predictions across gaps.
    - Even on frames that are present, Kinect joint positions carry
      measurement noise.  The Kalman update step fuses predictions with
      observations, producing smoother trajectories without the phase
      distortion that simple moving-average filters introduce.

Two usage modes:
    Offline:   `kalman_smooth_sequence(skeleton_seq)` — run the filter
               forward through an entire (T, N, 3) sequence.
    Real-time: Instantiate a `SkeletonKalmanTracker` and call
               `predict_and_update(frame)` per frame, or `predict_only()`
               when frames are dropped.

Kalman Filter State Model
=========================
Each joint is tracked independently with a 6-dimensional state:

    state = [x, y, z, vx, vy, vz]^T

Transition (constant velocity):

    F = [[1, 0, 0, dt, 0,  0 ],
         [0, 1, 0, 0,  dt, 0 ],
         [0, 0, 1, 0,  0,  dt],
         [0, 0, 0, 1,  0,  0 ],
         [0, 0, 0, 0,  1,  0 ],
         [0, 0, 0, 0,  0,  1 ]]

Observation (we only measure position):

    H = [[1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0]]

    observation = [x_measured, y_measured, z_measured]^T

Process noise Q scales with dt and controls how much the filter trusts
the constant-velocity assumption.  Measurement noise R controls how
much the filter trusts the raw sensor reading.
"""

import logging
from typing import Optional

import numpy as np

from . import config

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Per-Joint Kalman Filter
# ═════════════════════════════════════════════════════════════════════════════

class JointKalmanFilter:
    """
    Constant-velocity Kalman filter for a single 3D joint.

    State vector:  [x, y, z, vx, vy, vz]^T   (6 × 1)
    Observation:   [x, y, z]^T                 (3 × 1)

    Parameters
    ----------
    process_noise : float
        Diagonal process noise magnitude.  Larger values make the filter
        trust its motion model less and react faster to observations.
        Default: config.KALMAN_PROCESS_NOISE.
    measurement_noise : float
        Diagonal measurement noise magnitude.  Larger values make the
        filter trust raw sensor readings less, producing smoother output
        but with more lag.  Default: config.KALMAN_MEASUREMENT_NOISE.
    """

    STATE_DIM = 6   # [x, y, z, vx, vy, vz]
    OBS_DIM = 3     # [x, y, z]

    def __init__(
        self,
        process_noise: float = config.KALMAN_PROCESS_NOISE,
        measurement_noise: float = config.KALMAN_MEASUREMENT_NOISE,
    ):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # ── State vector and covariance ─────────────────────────────────
        self.x = np.zeros(self.STATE_DIM)               # state estimate
        self.P = np.eye(self.STATE_DIM) * 1.0            # state covariance

        # ── Observation model (fixed) ───────────────────────────────────
        # H maps state → observation: we observe position only
        self.H = np.zeros((self.OBS_DIM, self.STATE_DIM))
        self.H[0, 0] = 1.0  # x
        self.H[1, 1] = 1.0  # y
        self.H[2, 2] = 1.0  # z

        # ── Measurement noise (fixed) ───────────────────────────────────
        self.R = np.eye(self.OBS_DIM) * measurement_noise

        self._initialized = False

    def _build_F(self, dt: float) -> np.ndarray:
        """Build the state-transition matrix for a given time step dt."""
        F = np.eye(self.STATE_DIM)
        F[0, 3] = dt   # x += vx * dt
        F[1, 4] = dt   # y += vy * dt
        F[2, 5] = dt   # z += vz * dt
        return F

    def _build_Q(self, dt: float) -> np.ndarray:
        """
        Build the process noise covariance matrix.

        Uses the piecewise white-noise jerk model, which produces a Q
        that scales naturally with dt — longer time steps get more
        uncertainty because the constant-velocity assumption becomes
        less reliable over longer intervals.
        """
        q = self.process_noise
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

        # Block structure for each position-velocity pair
        # Q_block = q * [[dt^3/3, dt^2/2],
        #                [dt^2/2, dt     ]]
        Q = np.zeros((self.STATE_DIM, self.STATE_DIM))
        for i in range(3):  # x, y, z
            pi = i       # position index
            vi = i + 3   # velocity index
            Q[pi, pi] = dt3 / 3.0 * q
            Q[pi, vi] = dt2 / 2.0 * q
            Q[vi, pi] = dt2 / 2.0 * q
            Q[vi, vi] = dt * q

        return Q

    def initialize(self, observation: np.ndarray) -> None:
        """
        Initialize the filter state from the first observation.

        Sets position to the observed values and velocity to zero.

        Args:
            observation: (3,) array [x, y, z].
        """
        self.x[:3] = observation
        self.x[3:] = 0.0  # initial velocity = zero
        self.P = np.eye(self.STATE_DIM) * 1.0
        # Lower initial uncertainty for position (we have an observation)
        self.P[0, 0] = self.measurement_noise
        self.P[1, 1] = self.measurement_noise
        self.P[2, 2] = self.measurement_noise
        self._initialized = True

    def predict(self, dt: float = 1.0 / 30.0) -> np.ndarray:
        """
        Prediction step: project state forward by dt seconds.

        Args:
            dt: Time step in seconds (default 1/30 for 30 fps).

        Returns:
            Predicted position (3,) — the position component of the
            predicted state vector.
        """
        if not self._initialized:
            return self.x[:3].copy()

        F = self._build_F(dt)
        Q = self._build_Q(dt)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        return self.x[:3].copy()

    def update(self, observation: np.ndarray) -> np.ndarray:
        """
        Update step: correct the predicted state with a new observation.

        Must be called after predict().

        Args:
            observation: (3,) array [x, y, z] — measured joint position.

        Returns:
            Updated (corrected) position (3,).
        """
        if not self._initialized:
            self.initialize(observation)
            return self.x[:3].copy()

        # Innovation (measurement residual)
        y = observation - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State correction
        self.x = self.x + K @ y

        # Covariance correction (Joseph form for numerical stability)
        I_KH = np.eye(self.STATE_DIM) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return self.x[:3].copy()

    def predict_and_update(
        self,
        observation: np.ndarray,
        dt: float = 1.0 / 30.0,
    ) -> np.ndarray:
        """
        Combined predict + update in one call (standard usage).

        Args:
            observation: (3,) measured position.
            dt: Time step in seconds.

        Returns:
            Filtered position (3,).
        """
        self.predict(dt)
        return self.update(observation)

    def get_position(self) -> np.ndarray:
        """Return the current estimated position (3,)."""
        return self.x[:3].copy()

    def get_velocity(self) -> np.ndarray:
        """Return the current estimated velocity (3,)."""
        return self.x[3:].copy()

    def reset(self) -> None:
        """Reset the filter to its uninitialized state."""
        self.x = np.zeros(self.STATE_DIM)
        self.P = np.eye(self.STATE_DIM) * 1.0
        self._initialized = False


# ═════════════════════════════════════════════════════════════════════════════
# Skeleton-Level Kalman Tracker
# ═════════════════════════════════════════════════════════════════════════════

class SkeletonKalmanTracker:
    """
    Manages one JointKalmanFilter per skeleton joint.

    Provides a unified interface for filtering an entire skeleton frame
    at once, handling both normal frames and dropped frames.

    Parameters
    ----------
    n_joints : int
        Number of joints in the skeleton (default: config.N_JOINTS = 20).
    process_noise : float
        Process noise for all joint filters.
    measurement_noise : float
        Measurement noise for all joint filters.
    """

    def __init__(
        self,
        n_joints: int = config.N_JOINTS,
        process_noise: float = config.KALMAN_PROCESS_NOISE,
        measurement_noise: float = config.KALMAN_MEASUREMENT_NOISE,
    ):
        self.n_joints = n_joints
        self.filters = [
            JointKalmanFilter(
                process_noise=process_noise,
                measurement_noise=measurement_noise,
            )
            for _ in range(n_joints)
        ]

    def predict_and_update(
        self,
        frame: np.ndarray,
        dt: float = 1.0 / 30.0,
    ) -> np.ndarray:
        """
        Run predict + update for all joints using observed positions.

        Args:
            frame: (N, 3) array of measured joint positions for one frame.
            dt: Time step in seconds.

        Returns:
            smoothed_frame: (N, 3) array of filtered joint positions.
        """
        smoothed = np.zeros_like(frame)
        for j in range(self.n_joints):
            smoothed[j] = self.filters[j].predict_and_update(frame[j], dt)
        return smoothed

    def predict_only(self, dt: float = 1.0 / 30.0) -> np.ndarray:
        """
        Run prediction without an observation (for dropped frames).

        When a frame is dropped, calling this produces a physically
        plausible estimate based on the constant-velocity model instead
        of leaving a gap for linear interpolation to fill with jumps.

        Args:
            dt: Time step in seconds.

        Returns:
            predicted_frame: (N, 3) array of predicted joint positions.
        """
        predicted = np.zeros((self.n_joints, 3))
        for j in range(self.n_joints):
            self.filters[j].predict(dt)
            predicted[j] = self.filters[j].get_position()
        return predicted

    def get_smoothed_frame(self) -> np.ndarray:
        """
        Return the current estimated positions for all joints.

        Returns:
            frame: (N, 3) array of current state estimates.
        """
        frame = np.zeros((self.n_joints, 3))
        for j in range(self.n_joints):
            frame[j] = self.filters[j].get_position()
        return frame

    def reset(self) -> None:
        """Reset all joint filters."""
        for f in self.filters:
            f.reset()


# ═════════════════════════════════════════════════════════════════════════════
# Offline convenience: smooth an entire sequence
# ═════════════════════════════════════════════════════════════════════════════

def kalman_smooth_sequence(
    skeleton_seq: np.ndarray,
    process_noise: float = config.KALMAN_PROCESS_NOISE,
    measurement_noise: float = config.KALMAN_MEASUREMENT_NOISE,
    fps: float = 30.0,
) -> np.ndarray:
    """
    Apply Kalman filtering to an entire skeleton sequence (offline).

    Runs a forward pass of the constant-velocity Kalman filter through
    every frame of the sequence.  This smooths sensor noise and produces
    physically plausible trajectories across any frames where the raw
    data has sudden jumps (e.g. from interpolated frame drops).

    Note: This is a forward-only filter, not a Rauch–Tung–Striebel
    smoother.  A forward-only filter is causal (each output depends
    only on past + current observations), which keeps the offline
    and real-time code paths consistent.

    Args:
        skeleton_seq: (T, N, 3) raw joint positions.
        process_noise: Kalman process noise.
        measurement_noise: Kalman measurement noise.
        fps: Frame rate of the sequence (used to compute dt = 1/fps).

    Returns:
        smoothed_seq: (T, N, 3) filtered joint positions.
    """
    T, N, C = skeleton_seq.shape
    assert C == 3, f"Expected 3 coordinate channels, got {C}"

    dt = 1.0 / fps

    tracker = SkeletonKalmanTracker(
        n_joints=N,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
    )

    smoothed = np.zeros_like(skeleton_seq)
    for t in range(T):
        smoothed[t] = tracker.predict_and_update(skeleton_seq[t], dt)

    return smoothed
