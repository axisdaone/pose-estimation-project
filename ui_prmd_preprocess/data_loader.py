"""
Data Loader — Discover and parse raw UI-PRMD Kinect CSV files.

Handles the file-system layout of the Reduced Dataset and extracts
per-frame joint position data from the raw 132-column CSV rows.
"""

import csv
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np

from . import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_files(data_root: Path) -> List[Dict]:
    """
    Walk the UI-PRMD Reduced Dataset directory and return a manifest of samples.
    Supports both hierarchical and flat directories with .csv and .txt extensions,
    correctly mapping sequential repetition numbers to subject IDs.
    """
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    manifest = []

    # Repetitions count mapping to subjects for each exercise in the Reduced Dataset
    REPS_MAP = {
        1: [9, 9, 9, 9, 9, 10, 8, 9, 8, 10],
        2: [6, 0, 8, 7, 7, 9, 0, 9, 9, 0],
        3: [6, 0, 0, 9, 9, 9, 0, 9, 9, 0],
        4: [9, 9, 9, 9, 9, 9, 0, 9, 7, 0],
        5: [9, 9, 9, 7, 9, 8, 9, 8, 7, 9],
        6: [9, 9, 7, 9, 10, 10, 0, 9, 10, 0],
        7: [9, 9, 9, 9, 9, 0, 0, 9, 9, 0],
        8: [5, 8, 9, 9, 8, 9, 0, 6, 9, 0],
        9: [0, 9, 9, 7, 10, 9, 0, 9, 7, 0],
        10: [9, 9, 9, 9, 0, 9, 0, 9, 0, 0]
    }

    def get_subject_id(exercise_num: int, rep_idx: int) -> Optional[int]:
        if exercise_num not in REPS_MAP:
            return None
        counts = REPS_MAP[exercise_num]
        current_sum = 0
        for s_idx, count in enumerate(counts):
            if current_sum < rep_idx <= current_sum + count:
                return s_idx + 1
            current_sum += count
        return None

    import re

    for label_dir_name, label in [("Correct movement", config.LABEL_CORRECT),
                                   ("Incorrect movement", config.LABEL_INCORRECT)]:
        label_dir = data_root / label_dir_name
        if not label_dir.exists():
            # Try alternative naming conventions
            alt_names = [label_dir_name.replace(" ", "_"),
                         label_dir_name.lower(),
                         label_dir_name.lower().replace(" ", "_"),
                         "correct" if label == 1 else "incorrect",
                         "Correct" if label == 1 else "Incorrect",
                         "Correct Movements" if label == 1 else "Incorrect Movements",
                         "correct_movements" if label == 1 else "incorrect_movements",
                         "Correct_Movements" if label == 1 else "Incorrect_Movements"]
            for alt in alt_names:
                alt_path = data_root / alt
                if alt_path.exists():
                    label_dir = alt_path
                    break
            else:
                logger.warning(f"Label directory not found: {label_dir}")
                continue

        # Check if there are any files directly inside label_dir (flat layout inside the label folder)
        files = list(label_dir.glob("*.txt")) + list(label_dir.glob("*.csv"))
        if files:
            # Flat layout inside the label folder
            for f in sorted(files):
                # Pattern: e{exercise}_r{rep}[_inc].txt or similar
                match = re.search(r"e(\d+)_r(\d+)", f.name, re.IGNORECASE)
                if match:
                    exercise_num = int(match.group(1))
                    rep_idx = int(match.group(2))
                    subject = get_subject_id(exercise_num, rep_idx)
                    if subject is not None:
                        manifest.append({
                            "filepath": f,
                            "subject": subject,
                            "exercise": exercise_num,
                            "rep": rep_idx,
                            "label": label,
                        })
                    else:
                        logger.warning(f"Could not map rep {rep_idx} to subject for exercise {exercise_num} in: {f.name}")
                else:
                    logger.warning(f"Could not parse filename: {f.name}")
        else:
            # Hierarchical layout: Exercise X/ folders
            for exercise_dir in sorted(label_dir.iterdir()):
                if not exercise_dir.is_dir():
                    continue
                exercise_num = _extract_number(exercise_dir.name)
                if exercise_num is None:
                    continue

                for csv_file in sorted(list(exercise_dir.glob("*.csv")) + list(exercise_dir.glob("*.txt"))):
                    subject, rep = _parse_sample_filename(csv_file.name)
                    if subject is None:
                        # Fallback to REPS_MAP if it's e.g. e01_r45.txt
                        match = re.search(r"e(\d+)_r(\d+)", csv_file.name, re.IGNORECASE)
                        if match:
                            rep = int(match.group(2))
                            subject = get_subject_id(exercise_num, rep)

                    if subject is not None:
                        manifest.append({
                            "filepath": csv_file,
                            "subject": subject,
                            "exercise": exercise_num,
                            "rep": rep,
                            "label": label,
                        })

    logger.info(f"Discovered {len(manifest)} samples in {data_root}")
    return manifest


def discover_files_flat(data_root: Path) -> List[Dict]:
    """
    Alternative file discovery for flat directory layouts.

    Expects CSV files named with a pattern like:
        e{exercise}_s{subject}_r{rep}_{correct|incorrect}.csv

    Or any pattern where subject, exercise, rep, and label can be extracted
    from the filename. Override `_parse_flat_filename` for custom patterns.
    """
    data_root = Path(data_root)
    manifest = []

    for csv_file in sorted(data_root.rglob("*.csv")):
        result = _parse_flat_filename(csv_file.name)
        if result is not None:
            subject, exercise, rep, label = result
            manifest.append({
                "filepath": csv_file,
                "subject": subject,
                "exercise": exercise,
                "rep": rep,
                "label": label,
            })

    logger.info(f"Discovered {len(manifest)} samples (flat mode) in {data_root}")
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# CSV Parsing
# ─────────────────────────────────────────────────────────────────────────────

def load_csv_positions(filepath: Path,
                       n_raw_joints: int = config.N_RAW_JOINTS,
                       cols_per_joint: int = config.COLUMNS_PER_JOINT,
                       joint_subset: Optional[List[int]] = None) -> np.ndarray:
    """
    Load a single UI-PRMD Kinect CSV/TXT file and extract 3D position data.

    Each row represents one frame with columns:
        [joint0_oY, joint0_oX, joint0_oZ, joint0_pY, joint0_pX, joint0_pZ,
         joint1_oY, joint1_oX, joint1_oZ, joint1_pY, joint1_pX, joint1_pZ,
         ...]

    We extract only the position columns (offsets 3, 4, 5 within each
    6-column block) to get raw (Y, X, Z) position triplets.

    Args:
        filepath: Path to the file.
        n_raw_joints: Number of joints (default 22).
        cols_per_joint: Columns per joint (default 6).
        joint_subset: Optional list of joint indices to keep (default: first 20).

    Returns:
        positions: ndarray of shape (T, N, 3) in raw (Y, X, Z) order,
                   where T = number of frames and N = number of selected joints.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Read all rows as float arrays
    frames = []
    with open(filepath, "r") as f:
        for row_idx, row_str in enumerate(f):
            row_str = row_str.strip()
            if not row_str:
                continue
            
            # Split by commas or whitespace depending on delimiter
            if "," in row_str:
                row = row_str.split(",")
            else:
                row = row_str.split()
                
            try:
                values = [float(v) for v in row]
            except ValueError:
                # Likely a header row — skip it
                if row_idx == 0:
                    continue
                raise

            expected_cols = n_raw_joints * cols_per_joint
            if len(values) < expected_cols:
                logger.warning(
                    f"Row {row_idx} in {filepath.name} has {len(values)} columns, "
                    f"expected {expected_cols}. Padding with zeros."
                )
                values.extend([0.0] * (expected_cols - len(values)))

            frames.append(values[:expected_cols])

    if not frames:
        raise ValueError(f"No valid data rows found in {filepath}")

    raw_data = np.array(frames, dtype=np.float64)  # (T, n_raw_joints * 6)
    T = raw_data.shape[0]

    # Reshape to (T, n_raw_joints, 6)
    raw_data = raw_data.reshape(T, n_raw_joints, cols_per_joint)

    # Extract position columns: indices 3 (Y), 4 (X), 5 (Z) within each joint block
    positions_yxz = raw_data[:, :, [config.POS_OFFSET_Y,
                                     config.POS_OFFSET_X,
                                     config.POS_OFFSET_Z]]  # (T, n_raw_joints, 3)

    # Select joint subset (default: first 20)
    if joint_subset is None:
        joint_subset = config.JOINT_SUBSET_20

    positions_yxz = positions_yxz[:, joint_subset, :]  # (T, N, 3)

    return positions_yxz


def load_sample(filepath: Path, **kwargs) -> np.ndarray:
    """
    Convenience wrapper — loads a CSV and returns raw (Y, X, Z) positions.

    Returns:
        ndarray of shape (T, N, 3) with N = config.N_JOINTS.
    """
    return load_csv_positions(filepath, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Filename Parsing Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_number(name: str) -> Optional[int]:
    """Extract the first integer from a string, e.g. 'Exercise 3' → 3."""
    import re
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else None


def _parse_sample_filename(filename: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse subject and rep from a CSV filename in the hierarchical layout.

    Handles common patterns:
        '01_01.csv'  → subject=1, rep=1
        's01_r01.csv' → subject=1, rep=1
        'Subject_1_rep_3.csv' → subject=1, rep=3
    """
    import re
    name = Path(filename).stem

    # Pattern 1: two numbers separated by underscore, e.g. '01_05'
    match = re.match(r"(\d+)_(\d+)", name)
    if match:
        return int(match.group(1)), int(match.group(2))

    # Pattern 2: s{num}_r{num} or s{num}_rep{num}
    match = re.search(r"s(\d+).*?r(?:ep)?(\d+)", name, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))

    # Pattern 3: Subject_{num}_rep_{num}
    match = re.search(r"[Ss]ubject[_\s]*(\d+).*?[Rr]ep[_\s]*(\d+)", name)
    if match:
        return int(match.group(1)), int(match.group(2))

    return None, None


def _parse_flat_filename(filename: str):
    """
    Parse subject, exercise, rep, and label from a flat-layout filename.

    Expected pattern: e{ex}_s{subj}_r{rep}_{correct|incorrect}.csv
    Returns (subject, exercise, rep, label) or None.
    """
    import re
    name = Path(filename).stem

    match = re.match(
        r"e(\d+)_s(\d+)_r(?:ep)?(\d+)_(correct|incorrect)",
        name,
        re.IGNORECASE,
    )
    if match:
        exercise = int(match.group(1))
        subject = int(match.group(2))
        rep = int(match.group(3))
        label = config.LABEL_CORRECT if match.group(4).lower() == "correct" else config.LABEL_INCORRECT
        return subject, exercise, rep, label

    return None
