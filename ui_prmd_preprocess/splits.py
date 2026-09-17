"""
Data Splitting — Step 8 of the UI-PRMD pipeline.

Subject-independent 10-fold rotating group-holdout protocol:
    5 subjects for training, 2 for validation, 3 for testing.
    Every subject appears in three test folds, two validation folds, and five
    training folds. No subject appears in more than one partition in a fold.
"""

from typing import Dict, List, Tuple

from . import config


def generate_cross_subject_folds(
    n_subjects: int = config.N_SUBJECTS,
    train_size: int = config.TRAIN_SIZE,
    val_size: int = config.VAL_SIZE,
    test_size: int = config.TEST_SIZE,
) -> List[Dict[str, List[int]]]:
    """
    Generate deterministic rotating cross-subject splits.

    The subject order is rotated once per fold. The first ``test_size``
    subjects form the test set, the next ``val_size`` form validation, and the
    remainder form training. For 10 subjects with a 5/2/3 split this produces
    10 folds with balanced subject exposure across all roles.

    Args:
        n_subjects: Total number of subjects (default 10).
        train_size: Number of training subjects per fold (default 5).
        val_size: Number of validation subjects per fold (default 2).
        test_size: Number of test subjects per fold (default 3).

    Returns:
        List of 10 dicts, each with keys 'train', 'val', 'test',
        mapping to lists of 1-based subject IDs.
    """
    if train_size + val_size + test_size != n_subjects:
        raise ValueError(
            "train_size + val_size + test_size must equal n_subjects; "
            f"got {train_size} + {val_size} + {test_size} != {n_subjects}"
        )

    subjects = list(range(1, n_subjects + 1))
    folds = []

    for offset in range(n_subjects):
        rotated = subjects[offset:] + subjects[:offset]
        test_subj = rotated[:test_size]
        val_subj = rotated[test_size:test_size + val_size]
        train_subj = rotated[test_size + val_size:]

        assert len(train_subj) == train_size, (
            f"Expected {train_size} train subjects, got {len(train_subj)}"
        )

        folds.append({
            "train": train_subj,
            "val": val_subj,
            "test": test_subj,
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
