#!/usr/bin/env python3
"""
Example training script showing how to use the preprocessed UI-PRMD data
with LST-LA-GCN.

This is NOT a complete LST-LA-GCN implementation — it demonstrates how to
wire the preprocessing pipeline output into a training loop.
"""

import logging
import sys

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

# ─────────────────────────────────────────────────────────────────────────────
# Option A: Run preprocessing from scratch
# ─────────────────────────────────────────────────────────────────────────────

def example_from_scratch():
    """Preprocess raw CSVs and create DataLoaders."""
    from ui_prmd_preprocess.pipeline import PreprocessingPipeline
    from ui_prmd_preprocess.dataset import create_dataloaders

    # 1. Run the full pipeline
    pipeline = PreprocessingPipeline(
        data_root="./data/UI-PRMD/Reduced_Dataset/Kinect",
        target_len=150,
    )
    pipeline.run()
    pipeline.save("./output/preprocessed")

    # 2. Get data for fold 1 (test subject = S1)
    fold_data = pipeline.get_fold_data(fold_index=0)

    # 3. Create PyTorch DataLoaders
    loaders = create_dataloaders(
        train_samples=fold_data["train_samples"],
        train_labels=fold_data["train_labels"],
        val_samples=fold_data["val_samples"],
        val_labels=fold_data["val_labels"],
        test_samples=fold_data["test_samples"],
        test_labels=fold_data["test_labels"],
        adjacency_matrix=fold_data["adjacency_matrix"],
        mean=fold_data["mean"],
        std=fold_data["std"],
        batch_size=32,
    )

    # 4. Iterate
    for batch in loaders["train"]:
        joint = batch["joint"]   # (B, 3, 150, 20)
        bone = batch["bone"]     # (B, 3, 150, 20)
        A = batch["A"]           # (B, 20, 20)
        label = batch["label"]   # (B,)

        print(f"Batch — joint: {joint.shape}, bone: {bone.shape}, "
              f"A: {A.shape}, label: {label.shape}")
        break  # Just show one batch


# ─────────────────────────────────────────────────────────────────────────────
# Option B: Load previously preprocessed data
# ─────────────────────────────────────────────────────────────────────────────

def example_from_saved():
    """Load preprocessed .npy files and create DataLoaders."""
    from ui_prmd_preprocess.pipeline import PreprocessingPipeline
    from ui_prmd_preprocess.normalization import compute_channel_stats
    from ui_prmd_preprocess.splits import split_samples_by_fold
    from ui_prmd_preprocess.dataset import create_dataloaders

    # 1. Load saved data
    data = PreprocessingPipeline.load("./output/preprocessed")

    samples = list(data["samples"])     # list of (3, 150, 20)
    labels = list(data["labels"])
    subject_ids = list(data["subject_ids"])
    A = data["adjacency"]
    folds = data["folds"]

    # 2. Split for fold 0
    fold = folds[0]
    split = split_samples_by_fold(samples, labels, subject_ids, fold)

    train_samples, train_labels = split["train"]
    val_samples, val_labels = split["val"]
    test_samples, test_labels = split["test"]

    # 3. Compute normalization stats on TRAINING DATA ONLY
    mean, std = compute_channel_stats(train_samples)

    # 4. Create DataLoaders
    loaders = create_dataloaders(
        train_samples, train_labels,
        val_samples, val_labels,
        test_samples, test_labels,
        A, mean, std,
        batch_size=32,
    )

    print(f"Train: {len(loaders['train'].dataset)} samples")
    print(f"Val:   {len(loaders['val'].dataset)} samples")
    print(f"Test:  {len(loaders['test'].dataset)} samples")


# ─────────────────────────────────────────────────────────────────────────────
# Option C: Full 10-fold cross-validation loop
# ─────────────────────────────────────────────────────────────────────────────

def example_10fold():
    """Run 10-fold cross-validation."""
    from ui_prmd_preprocess.pipeline import PreprocessingPipeline
    from ui_prmd_preprocess.dataset import create_dataloaders

    pipeline = PreprocessingPipeline(
        data_root="./data/UI-PRMD/Reduced_Dataset/Kinect"
    )
    pipeline.run()

    fold_results = []

    for fold_idx in range(10):
        print(f"\n{'='*40} FOLD {fold_idx + 1}/10 {'='*40}")

        fold_data = pipeline.get_fold_data(fold_index=fold_idx)
        loaders = create_dataloaders(
            train_samples=fold_data["train_samples"],
            train_labels=fold_data["train_labels"],
            val_samples=fold_data["val_samples"],
            val_labels=fold_data["val_labels"],
            test_samples=fold_data["test_samples"],
            test_labels=fold_data["test_labels"],
            adjacency_matrix=fold_data["adjacency_matrix"],
            mean=fold_data["mean"],
            std=fold_data["std"],
            batch_size=32,
        )

        # ─── Your LST-LA-GCN training loop goes here ───
        # model = LSTLAGCN(...)
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # for epoch in range(100):
        #     train_one_epoch(model, loaders['train'], optimizer)
        #     val_acc = evaluate(model, loaders['val'])
        # test_acc = evaluate(model, loaders['test'])
        # fold_results.append(test_acc)

        test_subject = fold_data["fold_info"]["test"]
        print(f"  Test subject: S{test_subject[0]}")
        print(f"  Train batches: {len(loaders['train'])}")
        print(f"  Val batches:   {len(loaders['val'])}")
        print(f"  Test batches:  {len(loaders['test'])}")

    # if fold_results:
    #     print(f"\nMean Test Accuracy: {np.mean(fold_results)*100:.2f}% "
    #           f"± {np.std(fold_results)*100:.2f}%")


if __name__ == "__main__":
    # Uncomment the example you want to run:
    example_from_scratch()
    # example_from_saved()
    # example_10fold()
