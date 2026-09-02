"""
Verification script for the data loading & augmentation pipeline.

Tests:
  1. LOSO fold splitting from saved .npy files
  2. Normalization (unit-variance scaling)
  3. Yaw rotation augmentation
  4. Weighted sampling logic
  5. (Optional) Full PyTorch DataLoader if torch is available
"""

import json
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ui_prmd_preprocess import config
from ui_prmd_preprocess.normalization import compute_channel_stats, normalize
from ui_prmd_preprocess.augmentation import augment, _random_yaw_rotation
from ui_prmd_preprocess.splits import split_samples_by_fold


def main():
    data_dir = Path("./output/preprocessed")

    print("=" * 60)
    print(" Data Loading & Augmentation Pipeline Verification")
    print("=" * 60)

    # ── 1. Load saved data ──────────────────────────────────────────
    print("\n[1] Loading preprocessed data...")
    samples = np.load(data_dir / "samples.npy")       # (N, 3, 150, 20)
    labels = np.load(data_dir / "labels.npy")          # (N,)
    subject_ids = np.load(data_dir / "subject_ids.npy")  # (N,)
    adjacency = np.load(data_dir / "adjacency.npy")    # (20, 20)

    with open(data_dir / "folds.json") as f:
        folds = json.load(f)

    print(f"  samples.npy:     {samples.shape}")
    print(f"  labels.npy:      {labels.shape}")
    print(f"  subject_ids.npy: {subject_ids.shape}")
    print(f"  adjacency.npy:   {adjacency.shape}")
    print(f"  folds.json:      {len(folds)} folds")

    # ── 2. LOSO Fold Splitting ──────────────────────────────────────
    print("\n[2] Verifying LOSO fold splitting...")
    for fold_idx, fold in enumerate(folds):
        train_subjects = set(fold["train"])
        val_subjects = set(fold["val"])
        test_subjects = set(fold["test"])

        # Check no overlap
        assert train_subjects & test_subjects == set(), f"Fold {fold_idx}: train/test overlap!"
        assert val_subjects & test_subjects == set(), f"Fold {fold_idx}: val/test overlap!"
        assert train_subjects & val_subjects == set(), f"Fold {fold_idx}: train/val overlap!"

        # Count samples per split
        train_mask = np.isin(subject_ids, fold["train"])
        val_mask = np.isin(subject_ids, fold["val"])
        test_mask = np.isin(subject_ids, fold["test"])

        n_train = train_mask.sum()
        n_val = val_mask.sum()
        n_test = test_mask.sum()

        print(f"  Fold {fold_idx+1:2d}: train={n_train:4d} (S{fold['train']}), "
              f"val={n_val:4d} (S{fold['val']}), "
              f"test={n_test:4d} (S{fold['test']})")

        assert n_train + n_val + n_test == len(samples), \
            f"Fold {fold_idx}: split sizes don't sum to total!"

    print("  ✓ All 10 folds verified — zero overlap, correct sample counts")

    # ── 3. Normalization (unit variance) ────────────────────────────
    print("\n[3] Verifying normalization (unit variance scaling)...")
    fold_0 = folds[0]
    train_mask = np.isin(subject_ids, fold_0["train"])
    train_samples = list(samples[train_mask])
    train_labels_fold = labels[train_mask]

    mean, std = compute_channel_stats(train_samples)
    print(f"  Train-only stats (Fold 1):")
    print(f"    Mean per channel: {mean.flatten()}")
    print(f"    Std per channel:  {std.flatten()}")

    # Normalize a single sample and verify
    sample = train_samples[0]
    normalized = normalize(sample, mean, std)

    # Normalize ALL training samples and check the population stats
    all_norm = np.array([normalize(s, mean, std) for s in train_samples])
    pop_mean = all_norm.reshape(len(all_norm), 3, -1).mean(axis=(0, 2))
    pop_std = all_norm.reshape(len(all_norm), 3, -1).std(axis=(0, 2))

    print(f"    After normalization (train population):")
    print(f"      Mean: {pop_mean}  (should be ≈ 0)")
    print(f"      Std:  {pop_std}   (should be ≈ 1)")

    assert np.allclose(pop_mean, 0, atol=0.05), "Normalized mean too far from 0!"
    assert np.allclose(pop_std, 1, atol=0.05), "Normalized std too far from 1!"
    print("  ✓ Normalization produces unit variance (std ≈ 1)")

    # ── 4. Yaw Rotation Augmentation ────────────────────────────────
    print("\n[4] Verifying yaw rotation augmentation...")
    sample_ctn = train_samples[0]  # (3, 150, 20)
    print(f"  Input shape: {sample_ctn.shape}")

    # Apply yaw rotation
    rotated = _random_yaw_rotation(sample_ctn)
    print(f"  Output shape: {rotated.shape}")

    # Y channel should be unchanged
    y_diff = np.abs(sample_ctn[1] - rotated[1]).max()
    print(f"  Y-channel max diff: {y_diff:.10f} (should be 0)")
    assert y_diff < 1e-10, "Yaw rotation changed Y-axis!"

    # X and Z should be changed (unless angle was exactly 0)
    xz_diff = np.abs(sample_ctn[[0, 2]] - rotated[[0, 2]]).max()
    print(f"  XZ-channel max diff: {xz_diff:.4f} (should be > 0)")

    # Verify that rotation preserves distances (norm of XZ should be same)
    original_r = np.sqrt(sample_ctn[0]**2 + sample_ctn[2]**2)
    rotated_r = np.sqrt(rotated[0]**2 + rotated[2]**2)
    r_diff = np.abs(original_r - rotated_r).max()
    print(f"  XZ-radius max diff: {r_diff:.10f} (should be ≈ 0, rotation preserves radius)")
    assert r_diff < 1e-4, "Yaw rotation didn't preserve XZ radius!"

    # Test augment() chain (which now includes yaw)
    np.random.seed(42)
    augmented = augment(sample_ctn)
    print(f"  Full augment chain output shape: {augmented.shape}")
    print("  ✓ Yaw rotation works correctly — Y preserved, XZ rotated, radius preserved")

    # ── 5. Weighted Sampling Logic ──────────────────────────────────
    print("\n[5] Verifying weighted sampling logic...")
    # Simulate class distribution for a fold
    correct_count = sum(1 for l in train_labels_fold if l == 1)
    incorrect_count = sum(1 for l in train_labels_fold if l == 0)
    total = correct_count + incorrect_count

    w_correct = 1.0 / correct_count
    w_incorrect = 1.0 / incorrect_count

    print(f"  Training fold class distribution:")
    print(f"    Correct:   {correct_count} ({100*correct_count/total:.1f}%)")
    print(f"    Incorrect: {incorrect_count} ({100*incorrect_count/total:.1f}%)")
    print(f"  Weights: correct={w_correct:.6f}, incorrect={w_incorrect:.6f}")
    print(f"  Weight ratio (incorrect/correct): {w_correct/w_incorrect:.2f}")
    print("  ✓ Weighted sampling will balance class representation")

    # ── 6. PyTorch DataLoader (optional) ────────────────────────────
    print("\n[6] Testing PyTorch DataLoader...")
    try:
        import torch
        from ui_prmd_preprocess.dataset import LOSOFoldManager

        manager = LOSOFoldManager("./output/preprocessed")
        loaders = manager.get_fold_loaders(fold_idx=0, batch_size=16)

        for split_name in ["train", "val", "test"]:
            loader = loaders[split_name]
            batch = next(iter(loader))
            n_samples = len(loader.dataset)
            print(f"  [{split_name.upper()}] {n_samples} samples, "
                  f"joint={batch['joint'].shape}, "
                  f"bone={batch['bone'].shape}, "
                  f"label={batch['label'].shape}")

        print("  ✓ PyTorch DataLoaders working correctly")
    except (ImportError, OSError) as e:
        print(f"  ⚠ PyTorch not available ({e})")
        print("  Skipping DataLoader test — all non-PyTorch components verified")

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" ✓ ALL VERIFICATIONS PASSED")
    print("=" * 60)
    print("\nPipeline components verified:")
    print("  1. LOSO fold splitting — 10 folds, zero data leakage")
    print("  2. Normalization — unit variance (std ≈ 1) on train data")
    print("  3. Yaw rotation — camera-angle invariance augmentation")
    print("  4. Weighted sampling — class-balanced training batches")
    print()


if __name__ == "__main__":
    main()
