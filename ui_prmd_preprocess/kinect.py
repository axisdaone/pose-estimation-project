"""Load UI-PRMD's separate 22-joint Kinect position/angle recordings.

Reconstruction follows the dataset visualization convention: Rz @ Ry @ Rx,
parent rotations along each chain, and chest-local rotations for arm roots.
Positions are XYZ; YXZ in the paper describes Euler angles, not a swap of
Cartesian position columns. The 20-node graph omits neck and head tip.
"""
from pathlib import Path
import re
import numpy as np
from scipy.spatial.transform import Rotation

SUBSET = [0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]


def discover_kinect_files(root):
    manifest = []
    for label_name, label in (("correct", 1), ("incorrect", 0)):
        directory = Path(root) / label_name / "positions"
        for path in sorted(directory.glob("*.txt")):
            match = re.fullmatch(r"m(\d+)_s(\d+)_e(\d+)_positions(_inc)?", path.stem)
            if not match:
                raise ValueError(f"Unrecognized Kinect filename: {path}")
            exercise, subject, rep = map(int, match.groups()[:3])
            if bool(match[4]) != (label == 0):
                raise ValueError(f"Label suffix mismatch: {path}")
            manifest.append({"filepath": path, "label": label, "subject": subject,
                             "exercise": exercise, "rep": rep, "coordinate_format": "kinect_pair"})
    return manifest


def load_triplets(path):
    values = np.loadtxt(path, delimiter=",", ndmin=2)
    if values.shape[1] != 66 or not np.isfinite(values).all() or len(values) < 2:
        raise ValueError(f"Expected finite (T,66) Kinect data: {path}, got {values.shape}")
    return values.reshape(-1, 22, 3)


def reconstruct_kinect(positions, angles):
    if positions.shape != angles.shape or positions.shape[1:] != (22, 3):
        raise ValueError("Position and angle recordings must share shape (T,22,3)")
    matrices = Rotation.from_euler("xyz", angles.reshape(-1, 3), degrees=True).as_matrix()
    matrices = matrices.reshape(len(positions), 22, 3, 3)
    absolute = positions.copy()

    def chain(indices, rotation):
        for child, parent in indices:
            absolute[:, child] = np.einsum("tij,tj->ti", rotation, positions[:, child]) + absolute[:, parent]
            rotation = rotation @ matrices[:, child]

    chain([(i, i-1) for i in range(1, 6)], matrices[:, 0])
    chain([(6, 2), (7, 6), (8, 7), (9, 8)], matrices[:, 2])
    chain([(10, 2), (11, 10), (12, 11), (13, 12)], matrices[:, 2])
    chain([(14, 0), (15, 14), (16, 15), (17, 16)], matrices[:, 0])
    chain([(18, 0), (19, 18), (20, 19), (21, 20)], matrices[:, 0])
    return absolute


def load_kinect_sample(position_path):
    position_path = Path(position_path)
    angle_path = position_path.parent.parent / "angles" / position_path.name.replace("_positions", "_angles")
    absolute = reconstruct_kinect(load_triplets(position_path), load_triplets(angle_path))
    # Unit-independent scale from the two leg chains, constant over the sequence.
    # This preserves movement depth while removing subject size/unit variation.
    lengths = []
    for hip, knee, ankle in ((14, 15, 16), (18, 19, 20)):
        lengths.append(np.linalg.norm(absolute[:, knee] - absolute[:, hip], axis=1) +
                       np.linalg.norm(absolute[:, ankle] - absolute[:, knee], axis=1))
    leg_length = float(np.median(lengths))
    if not np.isfinite(leg_length) or leg_length <= 1e-6:
        raise ValueError(f"Degenerate leg lengths in {position_path}")
    return (absolute[:, SUBSET] / leg_length).astype(np.float32)
