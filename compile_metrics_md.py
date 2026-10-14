import json
from pathlib import Path

def generate_md_report():
    input_path = Path("output/overnight_2026_10_09_kinect/results.json")
    with open(input_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    md = []
    md.append("# UI-PRMD Overnight Training & Cross-Validation Metrics Report\n\n")
    md.append("- **Date Generated:** October 2026\n")
    md.append("- **Results Source File:** [`output/overnight_2026_10_09_kinect/results.json`](file:///c:/Users/askra/.gemini/antigravity/scratch/ui-prmd-preprocessing/output/overnight_2026_10_09_kinect/results.json)\n")
    md.append("- **Model Architecture:** LST-LA-GCN (Spatiotemporal Graph Convolutional Network)\n")
    md.append("- **Dataset:** UI-PRMD (Kinect Data Preprocessed)\n\n---\n\n")

    # Section 1: Parameters
    md.append("## 1. Experimental Parameters & Hyperparameters\n\n")
    md.append("All hyperparameters and configuration settings used to train and validate the model:\n\n")
    md.append("| Parameter | Value | Description |\n")
    md.append("| :--- | :--- | :--- |\n")
    cfg = d.get("config", {})
    descriptions = {
        "evaluation_protocol": "Cross-validation protocol (rotating subject holdout)",
        "num_folds": "Total cross-validation splits (10 folds)",
        "epochs": "Maximum training epochs per fold",
        "batch_size": "Mini-batch size",
        "lr": "Initial learning rate (Adam optimizer)",
        "weight_decay": "L2 weight regularization penalty",
        "dropout": "Dropout rate in GCN layers",
        "patience": "Early stopping patience (validation accuracy)",
        "seed": "Base random seed for reproducibility",
        "channels": "GCN hidden layer channel dimensions",
        "device": "Compute device (GPU acceleration)",
        "smote_train": "Synthetic Minority Over-sampling Technique on train set",
        "smote_test_diagnostic": "SMOTE diagnostic on test set",
        "data_dir": "Preprocessed graph data directory",
        "model_dir": "Directory containing saved model weights"
    }
    for k, v in cfg.items():
        desc = descriptions.get(k, "")
        md.append(f"| `{k}` | `{v}` | {desc} |\n")

    md.append("\n---\n\n")

    # Section 2: Aggregated Metrics
    agg = d.get("aggregated", {})
    md.append("## 2. Aggregated Performance Summary (Across 10 Folds)\n\n")
    md.append("| Metric | Mean | Standard Deviation | Description |\n")
    md.append("| :--- | :---: | :---: | :--- |\n")
    md.append(f"| **Test Accuracy** | **{agg.get('test_accuracy_mean', 0)*100:.2f}%** | ±{agg.get('test_accuracy_std', 0)*100:.2f}% | Overall test subject classification accuracy |\n")
    md.append(f"| **Test Weighted F1** | **{agg.get('test_f1_mean', 0)*100:.2f}%** | ±{agg.get('test_f1_std', 0)*100:.2f}% | Harmonic mean of precision & recall (class-weighted) |\n")
    md.append(f"| **Test Weighted Precision** | **{agg.get('test_precision_mean', 0)*100:.2f}%** | ±{agg.get('test_precision_std', 0)*100:.2f}% | True positives / predicted positives (weighted) |\n")
    md.append(f"| **Test Weighted Recall** | **{agg.get('test_recall_mean', 0)*100:.2f}%** | ±{agg.get('test_recall_std', 0)*100:.2f}% | True positives / ground truth positives (weighted) |\n")
    md.append(f"| **Best Val Accuracy** | **{agg.get('best_val_accuracy_mean', 0)*100:.2f}%** | ±{agg.get('best_val_accuracy_std', 0)*100:.2f}% | Peak validation accuracy across folds |\n")

    weighted_metrics = agg.get("sample_count_weighted_metrics", {})
    if "roc_auc" in weighted_metrics:
        md.append(f"| **ROC-AUC** | **{weighted_metrics['roc_auc']['mean']*100:.2f}%** | ±{weighted_metrics['roc_auc']['std']*100:.2f}% | Area under ROC curve |\n")
    if "recall_positive_sensitivity" in weighted_metrics:
        md.append(f"| **Sensitivity** | **{weighted_metrics['recall_positive_sensitivity']['mean']*100:.2f}%** | ±{weighted_metrics['recall_positive_sensitivity']['std']*100:.2f}% | True positive rate / recall for positive class |\n")
    if "specificity" in weighted_metrics:
        md.append(f"| **Specificity** | **{weighted_metrics['specificity']['mean']*100:.2f}%** | ±{weighted_metrics['specificity']['std']*100:.2f}% | True negative rate |\n")

    stat = d.get("statistical_tests", {}).get("accuracy_vs_chance_0_5", {})
    md.append("\n### Statistical Significance (vs 50% Chance Baseline)\n\n")
    md.append(f"- **Statistical Test:** {stat.get('test')}\n")
    md.append(f"- **t-Statistic:** {stat.get('t_statistic', 0):.4f}\n")
    md.append(f"- **p-Value:** {stat.get('p_value', 0):.4e} (Significant at alpha = 0.05: `{stat.get('significant_at_0_05')}`)\n")
    md.append(f"- **Total Test Evaluations:** {agg.get('total_test_evaluations')} samples\n")
    t_sec = agg.get("total_time_seconds", 0)
    md.append(f"- **Total Wall-Clock Training Time:** {t_sec:.1f} seconds (~{t_sec/3600:.2f} hours)\n\n")
    md.append("---\n\n")

    # Section 3: Per-Fold Results
    md.append("## 3. Per-Fold Performance Breakdown\n\n")
    md.append("| Fold | Train Subjects | Val Subjects | Test Subjects | Best Epoch | Total Epochs | Time (s) | Best Val Acc | Test Acc | Test F1 (wt) | Test Prec | Test Rec | ROC-AUC | Sens | Spec |\n")
    md.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

    for f in d["per_fold"]:
        idx = f["fold_index"]
        tr_s = str(f["train_subjects"])
        va_s = str(f["val_subjects"])
        te_s = str(f["test_subjects"])
        b_ep = f["best_epoch"]
        tot_ep = f["epochs_trained"]
        t_s = f["fold_time_seconds"]
        bva = f["best_val_acc"]
        tm = f["test_metrics"]
        acc = tm["accuracy"]
        f1 = tm["f1_weighted"]
        prec = tm["precision_weighted"]
        rec = tm["recall_weighted"]
        auc = tm["roc_auc"]
        sens = tm["recall_positive_sensitivity"]
        spec = tm["specificity"]
        md.append(f"| **Fold {idx}** | `{tr_s}` | `{va_s}` | `{te_s}` | {b_ep} | {tot_ep} | {t_s:.1f} | {bva*100:.2f}% | **{acc*100:.2f}%** | {f1*100:.2f}% | {prec*100:.2f}% | {rec*100:.2f}% | {auc:.4f} | {sens*100:.2f}% | {spec*100:.2f}% |\n")

    md.append("\n---\n\n")

    # Section 4: Confusion Matrices
    md.append("## 4. Per-Fold Confusion Matrices\n\n")
    md.append("| Fold | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Total Test Samples |\n")
    md.append("| :---: | :---: | :---: | :---: | :---: | :---: |\n")
    for f in d["per_fold"]:
        idx = f["fold_index"]
        cm = f["test_metrics"]["confusion_matrix"]
        tn, fp = cm[0]
        fn, tp = cm[1]
        md.append(f"| **Fold {idx}** | {tn} | {fp} | {fn} | {tp} | {tn+fp+fn+tp} |\n")

    md.append("\n---\n\n")

    # Section 5: Per-Epoch Trajectories per fold
    md.append("## 5. Epoch-by-Epoch History per Fold\n\n")
    md.append("Metrics recorded at each epoch: Training Loss, Training Accuracy, Validation Loss, and Validation Accuracy.\n\n")
    for f in d["per_fold"]:
        idx = f["fold_index"]
        best_ep = f["best_epoch"]
        h = f["history"]
        n_epochs = len(h["train_loss"])
        md.append(f"### Fold {idx} (Best Epoch: {best_ep}, Total Epochs: {n_epochs})\n\n")
        md.append("| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Best Epoch? |\n")
        md.append("| :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for ep in range(n_epochs):
            is_best = "⭐ **Yes**" if (ep + 1) == best_ep else "No"
            tr_l = h["train_loss"][ep]
            tr_a = h["train_acc"][ep]
            va_l = h["val_loss"][ep]
            va_a = h["val_acc"][ep]
            md.append(f"| {ep+1} | {tr_l:.4f} | {tr_a*100:.2f}% | {va_l:.4f} | {va_a*100:.2f}% | {is_best} |\n")
        md.append("\n")

    content = "".join(md)
    out_path = Path("TRAINING_METRICS_REPORT.md")
    with open(out_path, "w", encoding="utf-8") as out:
        out.write(content)
    print(f"Report saved to {out_path.resolve()}")

if __name__ == "__main__":
    generate_md_report()
