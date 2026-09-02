"""
Pipeline Orchestrator — Runs the full 10-step preprocessing pipeline.

This module ties together all preprocessing steps into a single callable
that transforms raw UI-PRMD CSV files into ready-to-train tensors.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config
from .data_loader import discover_files, discover_files_flat, load_sample
from .preprocessing import preprocess_sample
from .features import build_dual_stream, build_adjacency_matrix, to_ctn
from .normalization import compute_channel_stats, normalize
from .splits import generate_cross_subject_folds, split_samples_by_fold, print_fold_summary
from .augmentation import augment

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Orchestrates the full UI-PRMD preprocessing pipeline (Steps 1–10).

    Usage:
        pipeline = PreprocessingPipeline(data_root="./data/UI-PRMD/Kinect")
        pipeline.run()
        pipeline.save("./output/preprocessed")

        # Or, for training with a specific fold:
        fold_data = pipeline.get_fold_data(fold_index=0)
    """

    def __init__(
        self,
        data_root: str = str(config.DEFAULT_DATA_ROOT),
        target_len: int = config.TARGET_SEQUENCE_LENGTH,
        n_joints: int = config.N_JOINTS,
        flat_layout: bool = False,
    ):
        """
        Args:
            data_root: Path to the UI-PRMD Kinect data directory.
            target_len: Fixed temporal length for resampled sequences.
            n_joints: Number of joints in the GCN skeleton (default 20).
            flat_layout: If True, use flat file discovery instead of hierarchical.
        """
        self.data_root = Path(data_root)
        self.target_len = target_len
        self.n_joints = n_joints
        self.flat_layout = flat_layout

        # These are populated by run()
        self.manifest: List[Dict] = []
        self.samples: List[np.ndarray] = []       # List of (3, T, N) joint arrays
        self.labels: List[int] = []
        self.subject_ids: List[int] = []
        self.exercise_ids: List[int] = []
        self.adjacency_matrix: Optional[np.ndarray] = None
        self.folds: List[Dict] = []
        self.sequence_lengths: List[int] = []      # Original T per sample (for stats)

    def run(self) -> "PreprocessingPipeline":
        """
        Execute the full pipeline: discover files → preprocess → build features.

        Returns self for method chaining.
        """
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("UI-PRMD Preprocessing Pipeline - Starting")
        logger.info("=" * 60)

        # ── File Discovery ──────────────────────────────────────────────
        logger.info("[Discovery] Scanning data directory...")
        if self.flat_layout:
            self.manifest = discover_files_flat(self.data_root)
        else:
            self.manifest = discover_files(self.data_root)

        if not self.manifest:
            raise RuntimeError(
                f"No samples found in {self.data_root}. "
                f"Check the directory structure and file naming convention."
            )
        logger.info(f"[Discovery] Found {len(self.manifest)} samples.")

        # ── Steps 1–4: Per-sample geometric preprocessing ───────────────
        logger.info("[Steps 1-4] Running coordinate reconstruction, axis alignment, "
                     "centering, and temporal resampling...")

        self.samples = []
        self.labels = []
        self.subject_ids = []
        self.exercise_ids = []
        self.sequence_lengths = []
        n_errors = 0

        for i, entry in enumerate(self.manifest):
            try:
                # Load raw positions from CSV
                raw = load_sample(entry["filepath"])  # (T, N, 3) in YXZ
                self.sequence_lengths.append(raw.shape[0])

                # Run Steps 1-4
                processed = preprocess_sample(raw, target_len=self.target_len)  # (T, N, 3)

                # Convert to CTN format for GCN: (3, T, N)
                processed_ctn = to_ctn(processed)

                self.samples.append(processed_ctn)
                self.labels.append(entry["label"])
                self.subject_ids.append(entry["subject"])
                self.exercise_ids.append(entry["exercise"])

            except Exception as e:
                n_errors += 1
                logger.error(f"[Steps 1-4] Error processing {entry['filepath'].name}: {e}")
                continue

            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{len(self.manifest)} samples...")

        logger.info(f"[Steps 1-4] Done. {len(self.samples)} samples processed, "
                     f"{n_errors} errors.")

        # ── Print sequence length statistics ─────────────────────────────
        if self.sequence_lengths:
            lengths = np.array(self.sequence_lengths)
            logger.info(
                f"  Sequence lengths — min: {lengths.min()}, max: {lengths.max()}, "
                f"mean: {lengths.mean():.1f}, median: {np.median(lengths):.1f}"
            )

        # ── Step 6: Build adjacency matrix ──────────────────────────────
        logger.info("[Step 6] Building normalized adjacency matrix...")
        self.adjacency_matrix = build_adjacency_matrix(n_joints=self.n_joints)
        logger.info(f"  Adjacency matrix shape: {self.adjacency_matrix.shape}")

        # ── Step 8: Generate cross-subject folds ────────────────────────
        logger.info("[Step 8] Generating cross-subject folds...")
        self.folds = generate_cross_subject_folds()
        print_fold_summary(self.folds)

        elapsed = time.time() - t0
        logger.info(f"Pipeline completed in {elapsed:.1f}s")
        logger.info(f"  Total samples:    {len(self.samples)}")
        logger.info(f"  Label distribution: "
                     f"{sum(1 for l in self.labels if l == 1)} correct, "
                     f"{sum(1 for l in self.labels if l == 0)} incorrect")
        logger.info(f"  Sample shape:     {self.samples[0].shape if self.samples else 'N/A'}")
        logger.info("=" * 60)

        return self

    def get_fold_data(self, fold_index: int = 0) -> Dict:
        """
        Get train/val/test splits for a specific fold, with normalization stats
        computed on the training set only.

        Args:
            fold_index: 0-based fold index (0–9).

        Returns:
            Dict with:
                'train_samples', 'train_labels',
                'val_samples', 'val_labels',
                'test_samples', 'test_labels',
                'mean', 'std',
                'adjacency_matrix',
                'fold_info'
        """
        if not self.samples:
            raise RuntimeError("Pipeline has not been run yet. Call .run() first.")

        fold = self.folds[fold_index]
        split = split_samples_by_fold(
            self.samples, self.labels, self.subject_ids, fold
        )

        train_samples, train_labels = split["train"]
        val_samples, val_labels = split["val"]
        test_samples, test_labels = split["test"]

        # Step 7: Compute normalization stats on training data ONLY
        mean, std = compute_channel_stats(train_samples)

        logger.info(
            f"Fold {fold_index + 1}: "
            f"{len(train_samples)} train, "
            f"{len(val_samples)} val, "
            f"{len(test_samples)} test"
        )

        return {
            "train_samples": train_samples,
            "train_labels": train_labels,
            "val_samples": val_samples,
            "val_labels": val_labels,
            "test_samples": test_samples,
            "test_labels": test_labels,
            "mean": mean,
            "std": std,
            "adjacency_matrix": self.adjacency_matrix,
            "fold_info": fold,
        }

    def save(self, output_dir: str = str(config.DEFAULT_OUTPUT_DIR)) -> None:
        """
        Save all preprocessed data to disk as .npy files.

        Directory structure:
            output_dir/
            ├── samples.npy      (N_total, 3, T, N_joints)
            ├── labels.npy       (N_total,)
            ├── subject_ids.npy  (N_total,)
            ├── exercise_ids.npy (N_total,)
            ├── adjacency.npy    (N_joints, N_joints)
            ├── folds.json       fold definitions
            └── metadata.json    pipeline configuration
        """
        if not self.samples:
            raise RuntimeError("Pipeline has not been run yet. Call .run() first.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save arrays
        samples_array = np.array(self.samples, dtype=np.float32)
        np.save(output_dir / "samples.npy", samples_array)
        np.save(output_dir / "labels.npy", np.array(self.labels, dtype=np.int64))
        np.save(output_dir / "subject_ids.npy", np.array(self.subject_ids, dtype=np.int32))
        np.save(output_dir / "exercise_ids.npy", np.array(self.exercise_ids, dtype=np.int32))
        np.save(output_dir / "adjacency.npy", self.adjacency_matrix)

        # Save fold definitions
        with open(output_dir / "folds.json", "w") as f:
            json.dump(self.folds, f, indent=2)

        # Save metadata
        metadata = {
            "n_samples": len(self.samples),
            "sample_shape": list(self.samples[0].shape),
            "n_joints": self.n_joints,
            "target_sequence_length": self.target_len,
            "n_folds": len(self.folds),
            "label_distribution": {
                "correct": sum(1 for l in self.labels if l == 1),
                "incorrect": sum(1 for l in self.labels if l == 0),
            },
            "sequence_length_stats": {
                "min": int(np.min(self.sequence_lengths)),
                "max": int(np.max(self.sequence_lengths)),
                "mean": float(np.mean(self.sequence_lengths)),
                "median": float(np.median(self.sequence_lengths)),
            },
            "data_root": str(self.data_root),
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved preprocessed data to {output_dir}")
        logger.info(f"  samples.npy:  {samples_array.shape}")
        logger.info(f"  labels.npy:   ({len(self.labels)},)")
        logger.info(f"  adjacency.npy: {self.adjacency_matrix.shape}")

    @staticmethod
    def load(output_dir: str) -> Dict:
        """
        Load previously saved preprocessed data from disk.

        Returns:
            Dict with 'samples', 'labels', 'subject_ids', 'exercise_ids',
            'adjacency', 'folds', 'metadata'.
        """
        output_dir = Path(output_dir)

        data = {
            "samples": np.load(output_dir / "samples.npy"),
            "labels": np.load(output_dir / "labels.npy"),
            "subject_ids": np.load(output_dir / "subject_ids.npy"),
            "exercise_ids": np.load(output_dir / "exercise_ids.npy"),
            "adjacency": np.load(output_dir / "adjacency.npy"),
        }

        with open(output_dir / "folds.json") as f:
            data["folds"] = json.load(f)

        with open(output_dir / "metadata.json") as f:
            data["metadata"] = json.load(f)

        logger.info(f"Loaded preprocessed data from {output_dir}")
        logger.info(f"  Samples: {data['samples'].shape}")

        return data
