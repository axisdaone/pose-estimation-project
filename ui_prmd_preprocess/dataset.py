"""
PyTorch Dataset & DataLoaders — Step 10 of the UI-PRMD pipeline.

Provides:
  - UIPRMDDataset:   Core dataset wrapping preprocessed samples.
  - LOSOFoldManager: Loads saved .npy data and yields per-fold DataLoaders.
  - create_fold_dataloaders: One-call setup for a single LOSO fold with
    weighted sampling to handle minority subjects (S7, S10).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object

from . import config
from .features import compute_bone_features, to_ctn, to_tnc
from .normalization import compute_channel_stats, normalize
from .augmentation import augment

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Core Dataset
# ═════════════════════════════════════════════════════════════════════════════

class UIPRMDDataset(Dataset):
    """
    PyTorch Dataset for UI-PRMD skeleton sequences.

    Each __getitem__ call returns a dict with:
        - 'joint':  (3, T, N) float32 tensor — normalized joint positions.
        - 'bone':   (3, T, N) float32 tensor — bone vectors (computed on-the-fly).
        - 'A':      (N, N)   float32 tensor — precomputed adjacency matrix.
        - 'label':  scalar   long tensor     — 0 (incorrect) or 1 (correct).

    When augment=True (training mode), stochastic augmentations (including
    random yaw rotation) are applied to the joint stream before normalization
    and bone computation.
    """

    def __init__(
        self,
        samples: List[np.ndarray],
        labels: List[int],
        adjacency_matrix: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        do_augment: bool = False,
    ):
        """
        Args:
            samples: List of (3, T, N) joint position arrays (CTN format).
            labels: Parallel list of integer labels (0 or 1).
            adjacency_matrix: (N, N) normalized adjacency matrix.
            mean: (3, 1) per-channel mean from training set.
            std: (3, 1) per-channel std from training set.
            do_augment: Whether to apply stochastic augmentations.
        """
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for UIPRMDDataset. "
                "Install it with: pip install torch"
            )

        assert len(samples) == len(labels), (
            f"Mismatch: {len(samples)} samples vs {len(labels)} labels"
        )

        self.samples = samples
        self.labels = labels
        self.A = torch.tensor(adjacency_matrix, dtype=torch.float32)
        self.mean = mean
        self.std = std
        self.do_augment = do_augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        # Start with a copy of the joint stream (C, T, N)
        joint_ctn = self.samples[idx].copy()

        # Step 9: Apply augmentation (training only)
        if self.do_augment:
            joint_ctn = augment(joint_ctn)

        # Step 7: Normalize joint positions
        joint_ctn = normalize(joint_ctn, self.mean, self.std)

        # Step 5: Compute bone vectors from the (normalized) joint positions
        # Convert to (T, N, C), compute bones, convert back to (C, T, N)
        joint_tnc = to_tnc(joint_ctn)           # (T, N, 3)
        bone_tnc = compute_bone_features(joint_tnc)  # (T, N, 3)
        bone_ctn = to_ctn(bone_tnc)             # (3, T, N)

        return {
            "joint": torch.tensor(joint_ctn, dtype=torch.float32),
            "bone": torch.tensor(bone_ctn, dtype=torch.float32),
            "A": self.A,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ═════════════════════════════════════════════════════════════════════════════
# LOSO Fold Manager — loads from disk, iterates folds
# ═════════════════════════════════════════════════════════════════════════════

class LOSOFoldManager:
    """
    Manages Leave-One-Subject-Out cross-validation from saved preprocessed data.

    Loads samples.npy, labels.npy, subject_ids.npy, adjacency.npy, and
    folds.json once, then yields per-fold DataLoaders on demand.

    Usage:
        manager = LOSOFoldManager("./output/preprocessed")
        for fold_idx in range(manager.n_folds):
            loaders = manager.get_fold_loaders(fold_idx, batch_size=32)
            for batch in loaders["train"]:
                ...
    """

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Path to the directory containing preprocessed .npy files.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        data_dir = Path(data_dir)

        logger.info(f"Loading preprocessed data from {data_dir}...")
        self.samples = np.load(data_dir / "samples.npy")       # (N, 3, T, J)
        self.labels = np.load(data_dir / "labels.npy")          # (N,)
        self.subject_ids = np.load(data_dir / "subject_ids.npy")  # (N,)
        self.adjacency = np.load(data_dir / "adjacency.npy")    # (J, J)

        with open(data_dir / "folds.json") as f:
            self.folds = json.load(f)

        # Load metadata if available
        meta_path = data_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        self.n_folds = len(self.folds)
        self.n_samples = len(self.samples)

        logger.info(
            f"  Loaded {self.n_samples} samples, {self.n_folds} folds, "
            f"adjacency {self.adjacency.shape}"
        )

        # Log per-subject counts
        for s in sorted(set(self.subject_ids)):
            count = (self.subject_ids == s).sum()
            logger.info(f"  Subject {s}: {count} samples")

    def get_fold_split(self, fold_idx: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Split samples/labels by subject for a given fold.

        Returns:
            Dict with 'train', 'val', 'test' keys, each mapping to
            (samples_array, labels_array).
        """
        fold = self.folds[fold_idx]
        result = {}

        for split_name in ["train", "val", "test"]:
            split_subjects = set(fold[split_name])
            mask = np.array([s in split_subjects for s in self.subject_ids])
            result[split_name] = (self.samples[mask], self.labels[mask])

        return result

    def get_fold_loaders(
        self,
        fold_idx: int,
        batch_size: int = 32,
        num_workers: int = 0,
        use_weighted_sampler: bool = config.USE_WEIGHTED_SAMPLER,
    ) -> Dict[str, DataLoader]:
        """
        Create train/val/test DataLoaders for a specific LOSO fold.

        - Normalization stats are computed on the training split ONLY.
        - Training loader uses augmentation (including random yaw rotation).
        - Training loader uses WeightedRandomSampler when enabled, to
          handle class imbalance that arises from unequal subject sample
          counts (e.g., Subject 7 has only 34 samples vs Subject 8's 172).

        Args:
            fold_idx: 0-based fold index (0–9).
            batch_size: Batch size for all loaders.
            num_workers: DataLoader workers.
            use_weighted_sampler: If True, use inverse-frequency class
                weighting in the training sampler.

        Returns:
            Dict with 'train', 'val', 'test' DataLoaders and metadata:
                'train', 'val', 'test': DataLoader instances.
                'mean', 'std': normalization stats (from train only).
                'fold_info': the fold definition dict.
        """
        fold = self.folds[fold_idx]
        split_data = self.get_fold_split(fold_idx)

        train_samples, train_labels = split_data["train"]
        val_samples, val_labels = split_data["val"]
        test_samples, test_labels = split_data["test"]

        # ── Normalization: compute stats on training data ONLY ──────────
        mean, std = compute_channel_stats(list(train_samples))

        logger.info(
            f"Fold {fold_idx + 1}: "
            f"{len(train_samples)} train, "
            f"{len(val_samples)} val, "
            f"{len(test_samples)} test"
        )
        logger.info(f"  Norm mean: {mean.flatten()}")
        logger.info(f"  Norm std:  {std.flatten()}")

        # ── Create datasets ─────────────────────────────────────────────
        train_ds = UIPRMDDataset(
            list(train_samples), list(train_labels),
            self.adjacency, mean, std, do_augment=True,
        )
        val_ds = UIPRMDDataset(
            list(val_samples), list(val_labels),
            self.adjacency, mean, std, do_augment=False,
        )
        test_ds = UIPRMDDataset(
            list(test_samples), list(test_labels),
            self.adjacency, mean, std, do_augment=False,
        )

        # ── Training sampler with class weighting ───────────────────────
        train_sampler = None
        train_shuffle = True

        if use_weighted_sampler and len(train_ds) > 0:
            train_sampler, train_shuffle = _build_weighted_sampler(
                list(train_labels)
            )
            logger.info(
                f"  Weighted sampler enabled — "
                f"correct: {sum(1 for l in train_labels if l == 1)}, "
                f"incorrect: {sum(1 for l in train_labels if l == 0)}"
            )

        # ── Build DataLoaders ───────────────────────────────────────────
        loaders = {
            "train": DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=train_shuffle,
                sampler=train_sampler,
                num_workers=num_workers,
                drop_last=False,
            ),
            "val": DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            ),
            "test": DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            ),
            # Metadata for downstream use
            "mean": mean,
            "std": std,
            "fold_info": fold,
        }

        return loaders


# ═════════════════════════════════════════════════════════════════════════════
# Weighted Sampling Helper
# ═════════════════════════════════════════════════════════════════════════════

def _build_weighted_sampler(
    labels: List[int],
) -> Tuple[WeightedRandomSampler, bool]:
    """
    Build a WeightedRandomSampler that balances class frequencies.

    When subject sample counts vary dramatically (S7: 34 vs S8: 172),
    the class distribution within a training fold can become skewed.
    Inverse-frequency weighting ensures each class contributes equally
    to gradient updates per epoch.

    Args:
        labels: List of integer labels for all training samples.

    Returns:
        (sampler, shuffle) — sampler instance and False for shuffle
        (WeightedRandomSampler handles shuffling internally).
    """
    labels_arr = np.array(labels)
    classes, counts = np.unique(labels_arr, return_counts=True)

    # Inverse-frequency weight per class
    class_weights = {cls: 1.0 / count for cls, count in zip(classes, counts)}

    # Per-sample weight
    sample_weights = np.array([class_weights[l] for l in labels])
    sample_weights = torch.from_numpy(sample_weights).double()

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,  # Oversample minority class
    )

    logger.info(
        f"  Class weights: {dict(zip(classes.tolist(), [f'{class_weights[c]:.6f}' for c in classes]))}"
    )

    return sampler, False  # False = don't use shuffle when sampler is active


# ═════════════════════════════════════════════════════════════════════════════
# Convenience function (backward-compatible)
# ═════════════════════════════════════════════════════════════════════════════

def create_dataloaders(
    train_samples: List[np.ndarray],
    train_labels: List[int],
    val_samples: List[np.ndarray],
    val_labels: List[int],
    test_samples: List[np.ndarray],
    test_labels: List[int],
    adjacency_matrix: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 0,
    use_weighted_sampler: bool = config.USE_WEIGHTED_SAMPLER,
) -> dict:
    """
    Create train/val/test DataLoaders for one fold (backward-compatible API).

    Args:
        train_samples, val_samples, test_samples: Lists of (3, T, N) arrays.
        train_labels, val_labels, test_labels: Lists of int labels.
        adjacency_matrix: (N, N) normalized adjacency matrix.
        mean, std: Normalization stats from training set.
        batch_size: Batch size for all loaders.
        num_workers: Number of DataLoader workers.
        use_weighted_sampler: Whether to use weighted sampling for training.

    Returns:
        Dict with 'train', 'val', 'test' DataLoaders.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required. Install with: pip install torch")

    train_ds = UIPRMDDataset(
        train_samples, train_labels, adjacency_matrix, mean, std, do_augment=True
    )
    val_ds = UIPRMDDataset(
        val_samples, val_labels, adjacency_matrix, mean, std, do_augment=False
    )
    test_ds = UIPRMDDataset(
        test_samples, test_labels, adjacency_matrix, mean, std, do_augment=False
    )

    # Build weighted sampler for training
    train_sampler = None
    train_shuffle = True
    if use_weighted_sampler:
        train_sampler, train_shuffle = _build_weighted_sampler(train_labels)

    return {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=train_shuffle,
            sampler=train_sampler, num_workers=num_workers, drop_last=False,
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        ),
        "test": DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        ),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Quick sanity-check entry point
# ═════════════════════════════════════════════════════════════════════════════

def verify_fold_loaders(data_dir: str, fold_idx: int = 0, batch_size: int = 16):
    """
    Quick verification that DataLoaders produce correct shapes and dtypes.

    Run with:
        python -c "from ui_prmd_preprocess.dataset import verify_fold_loaders; verify_fold_loaders('./output/preprocessed')"
    """
    manager = LOSOFoldManager(data_dir)
    loaders = manager.get_fold_loaders(fold_idx, batch_size=batch_size)

    print(f"\n{'='*60}")
    print(f" DataLoader Verification — Fold {fold_idx + 1}")
    print(f"{'='*60}")

    for split_name in ["train", "val", "test"]:
        loader = loaders[split_name]
        batch = next(iter(loader))
        print(f"\n  [{split_name.upper()}] ({len(loader.dataset)} samples, "
              f"{len(loader)} batches)")
        print(f"    joint:  {batch['joint'].shape}  dtype={batch['joint'].dtype}")
        print(f"    bone:   {batch['bone'].shape}  dtype={batch['bone'].dtype}")
        print(f"    A:      {batch['A'].shape}  dtype={batch['A'].dtype}")
        print(f"    label:  {batch['label'].shape}  dtype={batch['label'].dtype}")
        print(f"    labels: {batch['label'].tolist()}")

    print(f"\n  Norm mean: {loaders['mean'].flatten()}")
    print(f"  Norm std:  {loaders['std'].flatten()}")
    print(f"  Fold info: {loaders['fold_info']}")
    print(f"{'='*60}\n")
