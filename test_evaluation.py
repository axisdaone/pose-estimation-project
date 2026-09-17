"""Regression tests for the subject-independent evaluation protocol."""

import unittest
from collections import Counter

import numpy as np

from train import _weighted_mean_std, compute_metrics, compute_statistical_tests
from ui_prmd_preprocess.dataset import smote_resample_sequences
from ui_prmd_preprocess.splits import generate_cross_subject_folds


class EvaluationProtocolTests(unittest.TestCase):
    def test_rotating_folds_are_disjoint_and_balanced(self):
        folds = generate_cross_subject_folds()
        role_counts = {
            subject: Counter()
            for subject in range(1, 11)
        }

        self.assertEqual(len(folds), 10)
        for fold in folds:
            train = set(fold["train"])
            val = set(fold["val"])
            test = set(fold["test"])
            self.assertEqual((len(train), len(val), len(test)), (5, 2, 3))
            self.assertFalse(train & val)
            self.assertFalse(train & test)
            self.assertFalse(val & test)
            self.assertEqual(train | val | test, set(range(1, 11)))
            for role in ("train", "val", "test"):
                for subject in fold[role]:
                    role_counts[subject][role] += 1

        for counts in role_counts.values():
            self.assertEqual(counts, {"train": 5, "val": 2, "test": 3})

    def test_binary_metrics_include_paper_metrics(self):
        metrics = compute_metrics(
            all_preds=[0, 0, 1, 1],
            all_labels=[0, 1, 1, 1],
            all_scores=[0.1, 0.4, 0.7, 0.9],
        )
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["specificity"], 1.0)
        self.assertAlmostEqual(metrics["recall_positive_sensitivity"], 2 / 3)
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertIn("f1_weighted", metrics)

    def test_cross_fold_weighted_mean(self):
        summary = _weighted_mean_std([0.5, 1.0], [1, 3])
        self.assertAlmostEqual(summary["mean"], 0.875)

    def test_accuracy_t_test_against_chance(self):
        fold_results = [
            {"test_metrics": {"accuracy": accuracy}}
            for accuracy in (0.70, 0.75, 0.80, 0.85)
        ]
        result = compute_statistical_tests(fold_results)
        chance_test = result["accuracy_vs_chance_0_5"]
        self.assertEqual(chance_test["n_folds"], 4)
        self.assertLess(chance_test["p_value"], 0.05)

    def test_smote_only_adds_minority_sequences(self):
        samples = np.arange(6 * 2 * 3 * 2, dtype=np.float32).reshape(6, 2, 3, 2)
        labels = np.array([0, 0, 0, 0, 1, 1])
        balanced_samples, balanced_labels = smote_resample_sequences(
            samples, labels, random_state=7
        )
        classes, counts = np.unique(balanced_labels, return_counts=True)
        self.assertEqual(dict(zip(classes, counts)), {0: 4, 1: 4})
        self.assertEqual(balanced_samples.shape, (8, 2, 3, 2))


if __name__ == "__main__":
    unittest.main()
