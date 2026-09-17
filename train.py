#!/usr/bin/env python3
# Fix OpenMP duplicate library crash on Windows/Anaconda (must be before imports)
import os as _os; _os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
"""
Training Script for LST-LA-GCN on UI-PRMD Dataset.

Runs 10-fold subject-independent rotating group-holdout evaluation with:
  - AdamW optimizer with weight decay
  - Cosine annealing LR scheduler with warm restarts
  - Early stopping based on validation accuracy
  - Comprehensive metrics: accuracy, precision, recall/sensitivity,
    specificity, F1, ROC-AUC, and confusion matrix
  - Per-fold and aggregated results
  - **Multithreaded data loading** (num_workers > 0)
  - **Parallel fold execution** via ProcessPoolExecutor
  - **Checkpoint/resume** for crash resilience

Usage:
    python train.py
    python train.py --num_folds 3 --epochs 50   # Quick test
    python train.py --num_folds 10 --epochs 100  # Full run
    python train.py --parallel_folds 2 --num_workers 4  # Parallel folds + threaded loading
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from ui_prmd_preprocess.model import LSTLAGCN
from ui_prmd_preprocess.dataset import LOSOFoldManager

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    all_preds: List[int],
    all_labels: List[int],
    all_scores: List[float] = None,
    num_classes: int = 2,
) -> Dict:
    """
    Compute classification metrics from predictions and ground truth.

    Returns paper-compatible binary metrics plus macro and support-weighted
    variants. Class 1 (correct/optimal movement) is the positive class.
    """
    preds = np.array(all_preds)
    labels = np.array(all_labels)

    if len(labels) == 0:
        raise ValueError("Cannot compute metrics for an empty evaluation set")

    # Overall accuracy
    accuracy = (preds == labels).mean()

    # Per-class metrics
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    support = np.zeros(num_classes, dtype=int)

    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        support[c] = (labels == c).sum()

        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[c] = (
            2 * precision[c] * recall[c] / (precision[c] + recall[c])
            if (precision[c] + recall[c]) > 0
            else 0.0
        )

    # Confusion matrix: rows = true, cols = predicted
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        confusion[t, p] += 1

    weights = support / max(support.sum(), 1)
    precision_weighted = float(np.sum(precision * weights))
    recall_weighted = float(np.sum(recall * weights))
    f1_weighted = float(np.sum(f1 * weights))

    # With class 1 as positive: confusion rows=true, columns=predicted.
    tn, fp = confusion[0, 0], confusion[0, 1]
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    roc_auc = None
    if all_scores is not None:
        scores = np.asarray(all_scores, dtype=float)
        positives = labels == 1
        n_pos = int(positives.sum())
        n_neg = int((~positives).sum())
        if len(scores) != len(labels):
            raise ValueError("all_scores must have the same length as labels")
        if n_pos > 0 and n_neg > 0:
            # Mann-Whitney/rank formulation of binary ROC-AUC; rankdata
            # correctly gives tied scores their average rank.
            ranks = stats.rankdata(scores, method="average")
            roc_auc = float(
                (ranks[positives].sum() - n_pos * (n_pos + 1) / 2)
                / (n_pos * n_neg)
            )

    return {
        "accuracy": float(accuracy),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "precision_positive": float(precision[1]),
        "recall_positive_sensitivity": float(recall[1]),
        "specificity": specificity,
        "f1_positive": float(f1[1]),
        "roc_auc": roc_auc,
        "support": support.tolist(),
        "n_samples": int(len(labels)),
        "confusion_matrix": confusion.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training & evaluation loops
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Returns:
        (avg_loss, accuracy) for the epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        joint = batch["joint"].to(device)    # (B, 3, T, N)
        bone = batch["bone"].to(device)      # (B, 3, T, N)
        A = batch["A"].to(device)            # (B, N, N)
        label = batch["label"].to(device)    # (B,)

        optimizer.zero_grad()
        logits = model(joint, bone, A)       # (B, num_classes)
        loss = criterion(logits, label)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item() * label.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == label).sum().item()
        total += label.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int], List[float]]:
    """
    Evaluate on a data loader.

    Returns:
        (avg_loss, accuracy, all_preds, all_labels, positive_class_scores)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_scores = []

    for batch in loader:
        joint = batch["joint"].to(device)
        bone = batch["bone"].to(device)
        A = batch["A"].to(device)
        label = batch["label"].to(device)

        logits = model(joint, bone, A)
        loss = criterion(logits, label)

        total_loss += loss.item() * label.size(0)
        preds = logits.argmax(dim=1)
        scores = torch.softmax(logits, dim=1)[:, 1]
        correct += (preds == label).sum().item()
        total += label.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(label.cpu().tolist())
        all_scores.extend(scores.cpu().tolist())

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy, all_preds, all_labels, all_scores


# ─────────────────────────────────────────────────────────────────────────────
# Single fold training
# ─────────────────────────────────────────────────────────────────────────────

def train_fold(
    fold_idx: int,
    data_dir: str,
    device_str: str,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
    patience: int = 15,
    channels: tuple = (64, 128, 256),
    num_workers: int = 4,
    pytorch_threads: int = None,
    use_smote_train: bool = True,
    smote_test_diagnostic: bool = False,
    seed: int = 42,
) -> Dict:
    """
    Train and evaluate one subject-independent 5/2/3 fold.

    This function is designed to be callable from a ProcessPoolExecutor.
    It loads data independently so each process has its own copy.

    Returns a dict with all metrics and training history.
    """
    # Configure PyTorch threading for this process
    if pytorch_threads:
        torch.set_num_threads(pytorch_threads)

    fold_seed = seed + fold_idx
    random.seed(fold_seed)
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fold_seed)

    device = torch.device(device_str)
    fold_start = time.time()

    # Each subprocess loads its own data to avoid pickling issues
    manager = LOSOFoldManager(data_dir)

    logger.info(f"\n{'='*70}")
    test_subjects = manager.folds[fold_idx]["test"]
    logger.info(
        f"  FOLD {fold_idx + 1}/{manager.n_folds} -- "
        f"Test Subjects: {', '.join('S'+str(s) for s in test_subjects)}"
    )
    logger.info(f"{'='*70}")

    # Get fold DataLoaders with multithreaded data loading
    loaders = manager.get_fold_loaders(
        fold_idx=fold_idx,
        batch_size=batch_size,
        num_workers=num_workers,
        use_smote_train=use_smote_train,
        smote_test_diagnostic=smote_test_diagnostic,
        smote_random_state=fold_seed,
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    logger.info(
        f"  Split sizes -- Train: {len(train_loader.dataset)}, "
        f"Val: {len(val_loader.dataset)}, "
        f"Test: {len(test_loader.dataset)}"
    )
    logger.info(f"  DataLoader num_workers: {num_workers}")

    # Initialize model
    model = LSTLAGCN(
        in_channels=3,
        num_classes=2,
        num_joints=20,
        channels=channels,
        dropout=dropout,
        use_attention=True,
    ).to(device)

    logger.info(f"  Model parameters: {model.count_parameters():,}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=2,
        eta_min=1e-6,
    )

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # ── Train ────────────────────────────────────────────────────────
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # ── Validate ─────────────────────────────────────────────────────
        val_loss, val_acc, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        # ── LR Scheduler step ────────────────────────────────────────────
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # ── Early stopping check ─────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Log progress every 10 epochs or on improvement
        if (epoch + 1) % 10 == 0 or epochs_without_improvement == 0:
            logger.info(
                f"  Epoch {epoch + 1:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Best: {best_val_acc:.4f} @ E{best_epoch}"
            )

        if epochs_without_improvement >= patience:
            logger.info(
                f"  ** Early stopping at epoch {epoch + 1} "
                f"(no improvement for {patience} epochs)"
            )
            break

    # ── Load best model and evaluate on test set ────────────────────────
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)

    test_loss, test_acc, test_preds, test_labels, test_scores = evaluate(
        model, test_loader, criterion, device
    )

    # Also get val metrics with best model
    _, _, val_preds, val_labels, val_scores = evaluate(
        model, val_loader, criterion, device
    )

    test_metrics = compute_metrics(test_preds, test_labels, test_scores)
    val_metrics = compute_metrics(val_preds, val_labels, val_scores)

    test_smote_metrics = None
    if smote_test_diagnostic:
        _, _, diag_preds, diag_labels, diag_scores = evaluate(
            model, loaders["test_smote_diagnostic"], criterion, device
        )
        test_smote_metrics = compute_metrics(diag_preds, diag_labels, diag_scores)

    fold_time = time.time() - fold_start

    # ── Log fold results ────────────────────────────────────────────────
    logger.info(f"\n  -- Fold {fold_idx + 1} Results --")
    logger.info(f"  Best Val Accuracy:  {best_val_acc:.4f} (epoch {best_epoch})")
    logger.info(f"  Test Accuracy:      {test_metrics['accuracy']:.4f}")
    logger.info(f"  Test Precision:     {test_metrics['precision_macro']:.4f}")
    logger.info(f"  Test Recall:        {test_metrics['recall_macro']:.4f}")
    logger.info(f"  Test F1 (macro):    {test_metrics['f1_macro']:.4f}")
    logger.info(f"  Test F1 (weighted): {test_metrics['f1_weighted']:.4f}")
    logger.info(f"  Test Specificity:   {test_metrics['specificity']:.4f}")
    auc_text = (
        f"{test_metrics['roc_auc']:.4f}"
        if test_metrics["roc_auc"] is not None else "N/A (single-class test set)"
    )
    logger.info(f"  Test ROC-AUC:       {auc_text}")
    logger.info(f"  Confusion Matrix:   {test_metrics['confusion_matrix']}")
    logger.info(f"  Fold time:          {fold_time:.1f}s")

    return {
        "fold_index": fold_idx,
        "seed": fold_seed,
        "train_subjects": manager.folds[fold_idx]["train"],
        "val_subjects": manager.folds[fold_idx]["val"],
        "test_subjects": test_subjects,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "epochs_trained": len(history["train_loss"]),
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "test_smote_diagnostic_metrics": test_smote_metrics,
        "fold_time_seconds": fold_time,
        "history": {
            "train_loss": [round(x, 6) for x in history["train_loss"]],
            "train_acc": [round(x, 6) for x in history["train_acc"]],
            "val_loss": [round(x, 6) for x in history["val_loss"]],
            "val_acc": [round(x, 6) for x in history["val_acc"]],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(checkpoint_path: Path, fold_results: List[Dict], config_dict: Dict):
    """Save incremental checkpoint after each fold completes."""
    checkpoint = {
        "config": config_dict,
        "completed_folds": [r["fold_index"] for r in fold_results],
        "per_fold": fold_results,
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f, indent=2)
    logger.info(f"  Checkpoint saved: {checkpoint_path} ({len(fold_results)} folds complete)")


def load_checkpoint(checkpoint_path: Path) -> Tuple[List[Dict], set]:
    """Load checkpoint and return (fold_results, completed_fold_indices)."""
    if not checkpoint_path.exists():
        return [], set()
    
    with open(checkpoint_path) as f:
        checkpoint = json.load(f)
    
    fold_results = checkpoint.get("per_fold", [])
    completed = set(checkpoint.get("completed_folds", []))
    logger.info(f"  Resumed from checkpoint: {len(completed)} folds already complete: {sorted(completed)}")
    return fold_results, completed


def _weighted_mean_std(values: List[float], weights: List[int]) -> Dict[str, float]:
    """Return sample-count-weighted mean and population standard deviation."""
    values_arr = np.asarray(values, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)
    valid = np.isfinite(values_arr) & np.isfinite(weights_arr) & (weights_arr > 0)
    if not np.any(valid):
        return {"mean": float("nan"), "std": float("nan")}
    values_arr = values_arr[valid]
    weights_arr = weights_arr[valid]
    mean = np.average(values_arr, weights=weights_arr)
    variance = np.average((values_arr - mean) ** 2, weights=weights_arr)
    return {"mean": float(mean), "std": float(np.sqrt(variance))}


def _subject_group(result: Dict) -> Tuple[int, ...]:
    """Normalize old and new result formats to a comparable test group."""
    if "test_subjects" in result:
        return tuple(sorted(int(s) for s in result["test_subjects"]))
    if "test_subject" in result:
        return (int(result["test_subject"]),)
    return tuple()


def compute_statistical_tests(
    fold_results: List[Dict], baseline_results_path: str = None
) -> Dict:
    """Compute chance-level and optional paired baseline accuracy t-tests."""
    accuracies = np.asarray(
        [r["test_metrics"]["accuracy"] for r in fold_results], dtype=float
    )
    output = {}

    if len(accuracies) >= 2:
        test = stats.ttest_1samp(accuracies, popmean=0.5, nan_policy="omit")
        output["accuracy_vs_chance_0_5"] = {
            "test": "two-sided one-sample t-test",
            "n_folds": int(len(accuracies)),
            "t_statistic": float(test.statistic),
            "p_value": float(test.pvalue),
            "significant_at_0_05": bool(test.pvalue < 0.05),
            "caveat": (
                "Rotating test groups overlap across folds, so fold accuracies "
                "are correlated; interpret this p-value as exploratory."
            ),
        }

    if baseline_results_path:
        with open(baseline_results_path) as stream:
            baseline = json.load(stream)
        baseline_by_group = {
            _subject_group(r): r["test_metrics"]["accuracy"]
            for r in baseline.get("per_fold", [])
            if _subject_group(r)
        }
        current_values = []
        baseline_values = []
        matched_groups = []
        for result in fold_results:
            group = _subject_group(result)
            if group in baseline_by_group:
                current_values.append(result["test_metrics"]["accuracy"])
                baseline_values.append(baseline_by_group[group])
                matched_groups.append(list(group))

        if len(current_values) >= 2:
            test = stats.ttest_rel(
                current_values, baseline_values, nan_policy="omit"
            )
            output["accuracy_vs_baseline"] = {
                "test": "two-sided paired t-test",
                "baseline_results": baseline_results_path,
                "matched_test_subject_groups": matched_groups,
                "n_pairs": len(current_values),
                "mean_difference": float(
                    np.mean(np.asarray(current_values) - np.asarray(baseline_values))
                ),
                "t_statistic": float(test.statistic),
                "p_value": float(test.pvalue),
                "significant_at_0_05": bool(test.pvalue < 0.05),
                "caveat": (
                    "The paired comparison is valid only when both models used "
                    "the same preprocessing and identical subject groups."
                ),
            }
        else:
            output["accuracy_vs_baseline"] = {
                "test": "two-sided paired t-test",
                "baseline_results": baseline_results_path,
                "status": "not_computed",
                "reason": "Fewer than two identical test-subject groups were found",
            }

    return output


# ─────────────────────────────────────────────────────────────────────────────
# Main: 10-fold cross-validation with parallelism
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train LST-LA-GCN using 5-train/2-validation/3-test "
            "subject-independent rotating group holdout"
        )
    )
    parser.add_argument("--data_dir", type=str, default="./output/preprocessed",
                        help="Path to preprocessed data directory")
    parser.add_argument("--num_folds", type=int, default=10,
                        help="Number of folds to run (1-10)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum epochs per fold")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="AdamW weight decay")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed; fold index is added per fold")
    parser.add_argument("--output_file", type=str, default="./output/training_results_5_2_3.json",
                        help="Path to save results JSON")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of DataLoader worker threads (0=main thread in-memory, recommended on Windows)")
    parser.add_argument("--parallel_folds", type=int, default=0,
                        help="Number of folds to train in parallel (0=auto: 2 for CPU, 1 for GPU)")
    parser.add_argument("--checkpoint_file", type=str, default="./output/training_checkpoint_5_2_3.json",
                        help="Path to checkpoint file for resume support")
    parser.add_argument(
        "--no_smote_train", action="store_true",
        help="Disable training-only SMOTE (enabled by default)",
    )
    parser.add_argument(
        "--smote_test_diagnostic", action="store_true",
        help=(
            "Also report metrics on a SMOTE-balanced synthetic test copy. "
            "These are diagnostic only; untouched-test metrics stay primary."
        ),
    )
    parser.add_argument(
        "--baseline_results", type=str, default=None,
        help=(
            "Optional prior results JSON with identical test-subject groups; "
            "enables a paired t-test on fold accuracies"
        ),
    )
    args = parser.parse_args()

    # ── Configure PyTorch threading ──────────────────────────────────────
    cpu_count = os.cpu_count() or 4
    torch.set_num_threads(cpu_count)
    logger.info(f"PyTorch threads: {torch.get_num_threads()} (CPUs: {cpu_count})")

    # ── Device selection ─────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ── Determine parallelism ────────────────────────────────────────────
    if args.parallel_folds == 0:
        # Auto: 2 folds on CPU, 1 on GPU (GPU already parallelizes internally)
        parallel_folds = 1 if device.type == "cuda" else min(2, cpu_count // 2)
    else:
        parallel_folds = args.parallel_folds

    # When running parallel folds, split PyTorch threads across processes
    if parallel_folds > 1:
        threads_per_fold = max(2, cpu_count // parallel_folds)
        # Reduce num_workers per fold to avoid thread oversubscription
        effective_num_workers = max(0, min(args.num_workers, cpu_count // parallel_folds - 1))
    else:
        threads_per_fold = cpu_count
        effective_num_workers = args.num_workers

    logger.info(f"Parallel folds: {parallel_folds}")
    logger.info(f"DataLoader workers per fold: {effective_num_workers}")
    logger.info(f"PyTorch threads per fold: {threads_per_fold}")

    # ── Load preprocessed data (quick check) ─────────────────────────────
    logger.info(f"\nVerifying preprocessed data at: {args.data_dir}")
    manager = LOSOFoldManager(args.data_dir)

    # ── Load checkpoint (resume support) ─────────────────────────────────
    checkpoint_path = Path(args.checkpoint_file)
    fold_results, completed_folds = load_checkpoint(checkpoint_path)

    # ── Print hyperparameters ────────────────────────────────────────────
    config_dict = {
        "evaluation_protocol": "rotating_subject_group_holdout_5_train_2_val_3_test",
        "num_folds": args.num_folds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "patience": args.patience,
        "seed": args.seed,
        "channels": [64, 128, 256],
        "device": str(device),
        "data_dir": args.data_dir,
        "num_workers": effective_num_workers,
        "parallel_folds": parallel_folds,
        "pytorch_threads_per_fold": threads_per_fold,
        "smote_train": not args.no_smote_train,
        "smote_test_diagnostic": args.smote_test_diagnostic,
        "baseline_results": args.baseline_results,
    }

    logger.info(f"\n{'='*70}")
    logger.info(f"  TRAINING CONFIGURATION")
    logger.info(f"{'='*70}")
    logger.info(f"  Folds:            {args.num_folds}")
    logger.info(f"  Epochs:           {args.epochs}")
    logger.info(f"  Batch size:       {args.batch_size}")
    logger.info(f"  Learning rate:    {args.lr}")
    logger.info(f"  Weight decay:     {args.weight_decay}")
    logger.info(f"  Dropout:          {args.dropout}")
    logger.info(f"  Patience:         {args.patience}")
    logger.info(f"  Base seed:        {args.seed}")
    logger.info(f"  Subject split:    5 train / 2 validation / 3 test")
    logger.info(f"  Training SMOTE:   {not args.no_smote_train}")
    logger.info(f"  Test SMOTE diag:  {args.smote_test_diagnostic}")
    logger.info(f"  Channels:         (64, 128, 256)")
    logger.info(f"  Device:           {device}")
    logger.info(f"  Parallel folds:   {parallel_folds}")
    logger.info(f"  DataLoader workers: {effective_num_workers}")
    logger.info(f"  PyTorch threads:  {threads_per_fold} per fold")
    if completed_folds:
        logger.info(f"  Resuming from:    {len(completed_folds)} completed folds")
    logger.info(f"{'='*70}\n")

    # ── Determine which folds to run ─────────────────────────────────────
    all_fold_indices = list(range(min(args.num_folds, manager.n_folds)))
    remaining_folds = [i for i in all_fold_indices if i not in completed_folds]

    if not remaining_folds:
        logger.info("All folds already completed! Skipping to results.")
    else:
        logger.info(f"Folds to run: {[i+1 for i in remaining_folds]}")

    # ── Run folds ────────────────────────────────────────────────────────
    total_start = time.time()

    # Common kwargs for train_fold
    fold_kwargs = dict(
        data_dir=args.data_dir,
        device_str=str(device),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        num_workers=effective_num_workers,
        pytorch_threads=threads_per_fold,
        use_smote_train=not args.no_smote_train,
        smote_test_diagnostic=args.smote_test_diagnostic,
        seed=args.seed,
    )

    if parallel_folds > 1 and len(remaining_folds) > 1:
        # ── Parallel fold execution ──────────────────────────────────────
        logger.info(f"\n*** PARALLEL MODE: Running {parallel_folds} folds concurrently ***\n")

        with ProcessPoolExecutor(max_workers=parallel_folds) as executor:
            future_to_fold = {}
            for fold_idx in remaining_folds:
                future = executor.submit(train_fold, fold_idx=fold_idx, **fold_kwargs)
                future_to_fold[future] = fold_idx

            for future in as_completed(future_to_fold):
                fold_idx = future_to_fold[future]
                try:
                    result = future.result()
                    fold_results.append(result)
                    logger.info(f"\n  ✓ Fold {fold_idx + 1} complete — "
                                f"Test Acc: {result['test_metrics']['accuracy']:.4f}, "
                                f"Time: {result['fold_time_seconds']:.1f}s")
                    # Save checkpoint after each fold
                    save_checkpoint(checkpoint_path, fold_results, config_dict)
                except Exception as e:
                    logger.error(f"\n  ✗ Fold {fold_idx + 1} FAILED: {e}")
                    import traceback
                    traceback.print_exc()
    else:
        # ── Sequential fold execution ────────────────────────────────────
        for fold_idx in remaining_folds:
            result = train_fold(fold_idx=fold_idx, **fold_kwargs)
            fold_results.append(result)
            # Save checkpoint after each fold
            save_checkpoint(checkpoint_path, fold_results, config_dict)

    # Sort results by fold index for consistent output
    fold_results.sort(key=lambda r: r["fold_index"])

    total_time = time.time() - total_start

    # ── Aggregate results ────────────────────────────────────────────────
    test_accs = [r["test_metrics"]["accuracy"] for r in fold_results]
    test_precisions = [r["test_metrics"]["precision_macro"] for r in fold_results]
    test_recalls = [r["test_metrics"]["recall_macro"] for r in fold_results]
    test_f1s = [r["test_metrics"]["f1_macro"] for r in fold_results]
    best_val_accs = [r["best_val_acc"] for r in fold_results]

    metric_keys = [
        "accuracy",
        "precision_positive",
        "recall_positive_sensitivity",
        "specificity",
        "f1_positive",
        "roc_auc",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]
    fold_weights = [r["test_metrics"]["n_samples"] for r in fold_results]
    sample_weighted = {
        key: _weighted_mean_std(
            [
                np.nan if r["test_metrics"][key] is None
                else r["test_metrics"][key]
                for r in fold_results
            ],
            fold_weights,
        )
        for key in metric_keys
    }

    aggregated = {
        "test_accuracy_mean": float(np.mean(test_accs)),
        "test_accuracy_std": float(np.std(test_accs)),
        "test_precision_mean": float(np.mean(test_precisions)),
        "test_precision_std": float(np.std(test_precisions)),
        "test_recall_mean": float(np.mean(test_recalls)),
        "test_recall_std": float(np.std(test_recalls)),
        "test_f1_mean": float(np.mean(test_f1s)),
        "test_f1_std": float(np.std(test_f1s)),
        "best_val_accuracy_mean": float(np.mean(best_val_accs)),
        "best_val_accuracy_std": float(np.std(best_val_accs)),
        "sample_count_weighted_metrics": sample_weighted,
        "total_test_evaluations": int(sum(fold_weights)),
        "total_time_seconds": total_time,
        "num_folds_run": len(fold_results),
    }

    diagnostic_results = [
        r["test_smote_diagnostic_metrics"]
        for r in fold_results
        if r.get("test_smote_diagnostic_metrics") is not None
    ]
    if diagnostic_results:
        diagnostic_weights = [m["n_samples"] for m in diagnostic_results]
        aggregated["test_smote_diagnostic"] = {
            "warning": (
                "Synthetic test-set metrics are diagnostic only and must not "
                "be reported as real-subject generalization performance."
            ),
            "sample_count_weighted_metrics": {
                key: _weighted_mean_std(
                    [np.nan if m[key] is None else m[key] for m in diagnostic_results],
                    diagnostic_weights,
                )
                for key in metric_keys
            },
        }
    statistical_tests = compute_statistical_tests(
        fold_results, args.baseline_results
    )

    # ── Print final summary ──────────────────────────────────────────────
    logger.info(f"\n\n{'='*70}")
    logger.info(f"  FINAL RESULTS -- {len(fold_results)}-Fold Cross-Validation")
    logger.info(f"{'='*70}")
    logger.info(f"")
    logger.info(f"  {'Metric':<25} {'Mean':>10}  {'± Std':>10}")
    logger.info(f"  {'-'*25} {'-'*10}  {'-'*10}")
    logger.info(f"  {'Test Accuracy':<25} {aggregated['test_accuracy_mean']*100:>9.2f}%  ±{aggregated['test_accuracy_std']*100:>8.2f}%")
    logger.info(f"  {'Test Precision (macro)':<25} {aggregated['test_precision_mean']*100:>9.2f}%  ±{aggregated['test_precision_std']*100:>8.2f}%")
    logger.info(f"  {'Test Recall (macro)':<25} {aggregated['test_recall_mean']*100:>9.2f}%  ±{aggregated['test_recall_std']*100:>8.2f}%")
    logger.info(f"  {'Test F1 (macro)':<25} {aggregated['test_f1_mean']*100:>9.2f}%  ±{aggregated['test_f1_std']*100:>8.2f}%")
    for label, key in [
        ("Accuracy (weighted)", "accuracy"),
        ("Precision+ (weighted)", "precision_positive"),
        ("Sensitivity (weighted)", "recall_positive_sensitivity"),
        ("Specificity (weighted)", "specificity"),
        ("F1+ (weighted)", "f1_positive"),
        ("ROC-AUC (weighted)", "roc_auc"),
    ]:
        summary = sample_weighted[key]
        logger.info(
            f"  {label:<25} {summary['mean']*100:>9.2f}%  "
            f"±{summary['std']*100:>8.2f}%"
        )
    logger.info(f"  {'Best Val Accuracy':<25} {aggregated['best_val_accuracy_mean']*100:>9.2f}%  ±{aggregated['best_val_accuracy_std']*100:>8.2f}%")
    logger.info(f"")
    logger.info(f"  Per-Fold Breakdown:")
    logger.info(f"  {'Fold':>6} {'Test Subjects':>16} {'Test Acc':>10} {'Test F1':>10} {'Val Acc':>10} {'Epochs':>8} {'Time':>8}")
    logger.info(f"  {'-'*6} {'-'*16} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for r in fold_results:
        logger.info(
            f"  {r['fold_index']+1:>6} "
            f"{','.join('S'+str(s) for s in r['test_subjects']):>16} "
            f"{r['test_metrics']['accuracy']*100:>9.2f}% "
            f"{r['test_metrics']['f1_macro']*100:>9.2f}% "
            f"{r['best_val_acc']*100:>9.2f}% "
            f"{r['epochs_trained']:>8} "
            f"{r['fold_time_seconds']:>7.1f}s"
        )
    logger.info(f"")
    logger.info(f"  Total training time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"  Parallelism: {parallel_folds} fold(s), {effective_num_workers} workers, {threads_per_fold} threads/fold")
    logger.info(f"{'='*70}")

    # ── Save results ─────────────────────────────────────────────────────
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "config": config_dict,
        "aggregated": aggregated,
        "statistical_tests": statistical_tests,
        "per_fold": fold_results,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    # ── Clean up checkpoint on successful completion ─────────────────────
    logger.info(f"Training complete! Checkpoint retained at: {checkpoint_path}")


if __name__ == "__main__":
    main()
