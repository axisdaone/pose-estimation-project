"""
Configuration constants for UI-PRMD preprocessing.

Contains skeleton topology, joint mappings, edge definitions,
and all tunable preprocessing hyperparameters.
"""

import numpy as np
from pathlib import Path

# =============================================================================
# Dataset Paths (override via CLI or environment)
# =============================================================================
DEFAULT_DATA_ROOT = Path("./data/UI-PRMD/Reduced_Dataset/Kinect")
DEFAULT_OUTPUT_DIR = Path("./output/preprocessed")

# =============================================================================
# Kinect Joint Definitions (25 joints tracked by Kinect v2)
# =============================================================================
# Full Kinect v2 joint names in order (index 0–24)
KINECT_V2_JOINTS = [
    "SpineBase",        # 0
    "SpineMid",         # 1
    "Neck",             # 2
    "Head",             # 3
    "ShoulderLeft",     # 4
    "ElbowLeft",        # 5
    "WristLeft",        # 6
    "HandLeft",         # 7
    "ShoulderRight",    # 8
    "ElbowRight",       # 9
    "WristRight",       # 10
    "HandRight",        # 11
    "HipLeft",          # 12
    "KneeLeft",         # 13
    "AnkleLeft",        # 14
    "FootLeft",         # 15
    "HipRight",         # 16
    "KneeRight",        # 17
    "AnkleRight",       # 18
    "FootRight",        # 19
    "SpineShoulder",    # 20
    "HandTipLeft",      # 21
    "ThumbLeft",        # 22
    "HandTipRight",     # 23
    "ThumbRight",       # 24
]

# =============================================================================
# UI-PRMD CSV Layout
# =============================================================================
# The raw CSV files store 22 joints (UI-PRMD omits 3 of the 25 Kinect v2 joints).
# Each joint has 6 columns: orient_Y, orient_X, orient_Z, pos_Y, pos_X, pos_Z
# Total columns per row: 22 joints × 6 values = 132

COLUMNS_PER_JOINT = 6  # 3 orientation (YXZ Euler) + 3 position (YXZ)
N_RAW_JOINTS = 22      # Joints in UI-PRMD CSV files

# Indices of the 22 joints present in UI-PRMD CSVs (out of the 25 Kinect v2 joints)
# UI-PRMD typically omits: HandTipLeft (21), ThumbLeft (22), HandTipRight (23), ThumbRight (24), SpineShoulder (20)
# But keeps 22 joints — the exact subset may vary by dataset version.
# Below is the commonly used 22-joint mapping:
RAW_JOINT_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

# Within each joint's 6-column block, position data is at offsets 3, 4, 5 (pos_Y, pos_X, pos_Z)
POS_OFFSET_Y = 3
POS_OFFSET_X = 4
POS_OFFSET_Z = 5

# =============================================================================
# 20-Joint Subset for GCN (dropping HandTipLeft, ThumbLeft, SpineShoulder)
# =============================================================================
# We use a 20-joint subset that matches the standard GCN skeleton topology.
# Map from 20-joint index → raw CSV joint index
JOINT_SUBSET_20 = list(range(20))  # Use first 20 of the 22 raw joints

N_JOINTS = 20  # Final joint count for the GCN

JOINT_NAMES_20 = [
    "Waist",            # 0  — SpineBase (root)
    "Spine",            # 1  — SpineMid
    "ShoulderCenter",   # 2  — Neck
    "Head",             # 3
    "ShoulderLeft",     # 4
    "ElbowLeft",        # 5
    "WristLeft",        # 6
    "HandLeft",         # 7
    "ShoulderRight",    # 8
    "ElbowRight",       # 9
    "WristRight",       # 10
    "HandRight",        # 11
    "HipLeft",          # 12
    "KneeLeft",        # 13
    "AnkleLeft",        # 14
    "FootLeft",         # 15
    "HipRight",         # 16
    "KneeRight",        # 17
    "AnkleRight",       # 18
    "FootRight",        # 19
]

ROOT_JOINT_INDEX = 0  # Waist / SpineBase is the root

# =============================================================================
# Kinematic Tree (for coordinate reconstruction — Step 1)
# =============================================================================
# Ordered list of (child_index, parent_index) for the 20-joint skeleton.
# Must be in topological order: parents appear before children.
KINEMATIC_TREE = [
    (1, 0),    # Spine ← Waist
    (2, 1),    # ShoulderCenter ← Spine
    (3, 2),    # Head ← ShoulderCenter
    (4, 2),    # ShoulderLeft ← ShoulderCenter
    (5, 4),    # ElbowLeft ← ShoulderLeft
    (6, 5),    # WristLeft ← ElbowLeft
    (7, 6),    # HandLeft ← WristLeft
    (8, 2),    # ShoulderRight ← ShoulderCenter
    (9, 8),    # ElbowRight ← ShoulderRight
    (10, 9),   # WristRight ← ElbowRight
    (11, 10),  # HandRight ← WristRight
    (12, 0),   # HipLeft ← Waist
    (13, 12),  # KneeLeft ← HipLeft
    (14, 13),  # AnkleLeft ← KneeLeft
    (15, 14),  # FootLeft ← AnkleLeft
    (16, 0),   # HipRight ← Waist
    (17, 16),  # KneeRight ← HipRight
    (18, 17),  # AnkleRight ← KneeRight
    (19, 18),  # FootRight ← AnkleRight
]

# =============================================================================
# Skeleton Edges (for bone features & adjacency — Steps 5, 6)
# =============================================================================
# Edges as (parent, child) pairs, 0-indexed into the 20-joint skeleton.
SKELETON_EDGES = [
    (0, 1),    # Waist → Spine
    (1, 2),    # Spine → ShoulderCenter
    (2, 3),    # ShoulderCenter → Head
    (2, 4),    # ShoulderCenter → ShoulderLeft
    (4, 5),    # ShoulderLeft → ElbowLeft
    (5, 6),    # ElbowLeft → WristLeft
    (6, 7),    # WristLeft → HandLeft
    (2, 8),    # ShoulderCenter → ShoulderRight
    (8, 9),    # ShoulderRight → ElbowRight
    (9, 10),   # ElbowRight → WristRight
    (10, 11),  # WristRight → HandRight
    (0, 12),   # Waist → HipLeft
    (12, 13),  # HipLeft → KneeLeft
    (13, 14),  # KneeLeft → AnkleLeft
    (14, 15),  # AnkleLeft → FootLeft
    (0, 16),   # Waist → HipRight
    (16, 17),  # HipRight → KneeRight
    (17, 18),  # KneeRight → AnkleRight
    (18, 19),  # AnkleRight → FootRight
]

# =============================================================================
# Preprocessing Hyperparameters
# =============================================================================
TARGET_SEQUENCE_LENGTH = 150  # Fixed temporal length after resampling
INTERPOLATION_KIND = "linear" # Interpolation method for resampling

# =============================================================================
# Augmentation Hyperparameters
# =============================================================================
AUGMENT_PROBABILITY = 0.5        # Probability of applying each augmentation
TEMPORAL_CROP_MIN_RATIO = 0.8    # Minimum fraction of frames to keep in crop
JITTER_STD = 0.01                # Std-dev of Gaussian noise for joint jitter
SPEED_VARIATION_MIN = 0.8        # Min temporal scale factor
SPEED_VARIATION_MAX = 1.2        # Max temporal scale factor
YAW_MAX_DEGREES = 30.0           # Max random yaw rotation angle (±degrees)
YAW_PROBABILITY = 0.5            # Probability of applying yaw rotation

# =============================================================================
# Data Splitting
# =============================================================================
N_SUBJECTS = 10
N_EXERCISES = 10
N_REPS = 10
N_FOLDS = 10  # Leave-one-subject-out cross-validation

# Cross-subject split: 6 train / 3 val / 1 test
TRAIN_SIZE = 6
VAL_SIZE = 3
TEST_SIZE = 1

# Weighted sampling — use inverse-frequency weights to balance class counts
# across folds (especially important for minority subjects like S7 and S10)
USE_WEIGHTED_SAMPLER = True

# =============================================================================
# Normalization
# =============================================================================
NORM_EPSILON = 1e-6  # Small constant to avoid division by zero in z-score

# =============================================================================
# Kalman Filter (signal smoothing for frame drops / sensor noise)
# =============================================================================
ENABLE_KALMAN_FILTER = True     # Enable Kalman smoothing in the offline pipeline
KALMAN_PROCESS_NOISE = 1e-4     # Trust in constant-velocity model (lower = smoother)
KALMAN_MEASUREMENT_NOISE = 1e-2 # Trust in raw sensor readings (lower = less smoothing)
KALMAN_FPS = 30.0               # Assumed frame rate for computing dt

# =============================================================================
# Labels
# =============================================================================
LABEL_CORRECT = 1
LABEL_INCORRECT = 0
