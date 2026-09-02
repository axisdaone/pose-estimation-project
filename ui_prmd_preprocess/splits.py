"""
Data Splitting — Step 8 of the UI-PRMD pipeline.

Cross-subject 10-fold evaluation protocol:
    6 subjects for training, 3 for validation, 1 for testing.
    Each fold uses a different subject as the test set.
"""

from typing import Dict, List, Tuple

from . import config


def generate_cross_subject_folds(
    n_subjects: int = config.N_SUBJECTS,
    train_size: int = config.TRAIN_SIZE,
    val_size: int = config.VAL_SIZE,
) -> List[Dict[str, List[int]]]:
    """
    Generate 10-fold cross-subject splits following the standard UI-PRMD protocol.

    In each fold, one subject is held out for testing. Of the remaining 9,
    the first 3 are used for validation and the last 6 for training.
    Over 10 folds, every subject serves as the test subject exactly once.

    Args:
        n_subjects: Total number of subjects (default 10).
        train_size: Number of training subjects per fold (default 6).
        val_size: Number of validation subjects per fold (default 3).

    Returns:
        List of 10 dicts, each with keys 'train', 'val', 'test',
        mapping to lists of 1-based subject IDs.
    """
    subjects = list(range(1, n_subjects + 1))
    folds = []

    for test_subj in subjects:
        remaining = [s for s in subjects if s != test_subj]
        val_subj = remaining[:val_size]
        train_subj = remaining[val_size:]

        assert len(train_subj) == train_size, (
            f"Expected {train_size} train subjects, got {len(train_subj)}"
        )

        folds.append({
            "train": train_subj,
            "val": val_subj,
            "test": [test_subj],
        })

    return folds


def split_samples_by_fold(
    samples: list,
    labels: list,
    subject_ids: list,
    fold: Dict[str, List[int]],
) -> Dict[str, Tuple[list, list]]:
    """
    Partition samples into train/val/test based on a fold's subject assignment.

    Args:
        samples: List of preprocessed sample arrays.
        labels: List of integer labels (0 or 1).
        subject_ids: List of subject IDs (1-based) parallel to samples.
        fold: Dict with 'train', 'val', 'test' keys mapping to subject ID lists.

    Returns:
        Dict with keys 'train', 'val', 'test', each mapping to a tuple
        (split_samples, split_labels).
    """
    result = {}

    for split_name in ["train", "val", "test"]:
        split_subjects = set(fold[split_name])
        indices = [i for i, s in enumerate(subject_ids) if s in split_subjects]
        split_samples = [samples[i] for i in indices]
        split_labels = [labels[i] for i in indices]
        result[split_name] = (split_samples, split_labels)

    return result


def print_fold_summary(folds: List[Dict[str, List[int]]]) -> None:
    """Pretty-print the fold assignments."""
    print(f"\n{'='*60}")
    print(f" Cross-Subject Folds ({len(folds)}-fold)")
    print(f"{'='*60}")
    for i, fold in enumerate(folds):
        print(
            f"  Fold {i+1:2d}  |  "
            f"Train: {fold['train']}  |  "
            f"Val: {fold['val']}  |  "
            f"Test: {fold['test']}"
        )
    print(f"{'='*60}\n")
