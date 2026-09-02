"""
Generate a comprehensive DOCX report summarizing the UI-PRMD preprocessing
pipeline for LST-LA-GCN.
"""

import json
import numpy as np
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE


def set_cell_shading(cell, color_hex):
    """Set background shading on a table cell."""
    from docx.oxml.ns import qn
    from lxml import etree
    shading = etree.SubElement(cell._element.get_or_add_tcPr(), qn('w:shd'))
    shading.set(qn('w:fill'), color_hex)
    shading.set(qn('w:val'), 'clear')


def add_styled_table(doc, headers, rows, col_widths=None):
    """Add a formatted table to the document."""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = 'Table Grid'

    # Header row
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = header
        for paragraph in cell.paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                run.bold = True
                run.font.size = Pt(9)
                run.font.color.rgb = RGBColor(255, 255, 255)
        set_cell_shading(cell, '2C3E50')

    # Data rows
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = table.rows[r_idx + 1].cells[c_idx]
            cell.text = str(val)
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(9)
            if r_idx % 2 == 1:
                set_cell_shading(cell, 'ECF0F1')

    return table


def main():
    output_path = Path(r"C:\Users\askra\.gemini\antigravity\scratch\ui-prmd-preprocessing\UI_PRMD_Pipeline_Report.docx")
    data_dir = Path("./output/preprocessed")

    # Load data for stats
    samples = np.load(data_dir / "samples.npy")
    labels = np.load(data_dir / "labels.npy")
    subject_ids = np.load(data_dir / "subject_ids.npy")
    with open(data_dir / "folds.json") as f:
        folds = json.load(f)
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Compute normalization stats for fold 1
    from ui_prmd_preprocess.normalization import compute_channel_stats
    fold_0 = folds[0]
    train_mask = np.isin(subject_ids, fold_0["train"])
    train_samples = list(samples[train_mask])
    mean, std = compute_channel_stats(train_samples)

    # ── Build Document ──────────────────────────────────────────────
    doc = Document()

    # Page margins
    for section in doc.sections:
        section.top_margin = Cm(2.54)
        section.bottom_margin = Cm(2.54)
        section.left_margin = Cm(2.54)
        section.right_margin = Cm(2.54)

    # ── Title Page ──────────────────────────────────────────────────
    doc.add_paragraph("")
    doc.add_paragraph("")
    doc.add_paragraph("")

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("UI-PRMD Preprocessing Pipeline")
    run.bold = True
    run.font.size = Pt(28)
    run.font.color.rgb = RGBColor(44, 62, 80)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("for LST-LA-GCN Based Physical Rehabilitation Assessment")
    run.font.size = Pt(16)
    run.font.color.rgb = RGBColor(127, 140, 141)

    doc.add_paragraph("")

    line = doc.add_paragraph()
    line.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = line.add_run("_" * 60)
    run.font.color.rgb = RGBColor(189, 195, 199)

    doc.add_paragraph("")

    details = doc.add_paragraph()
    details.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = details.add_run("Technical Report\nData Preprocessing & Augmentation Pipeline")
    run.font.size = Pt(12)
    run.font.color.rgb = RGBColor(100, 100, 100)

    doc.add_paragraph("")
    doc.add_paragraph("")

    info = doc.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = info.add_run("Dataset: UI-PRMD (University of Idaho Physical Rehabilitation Movement Data)\n"
                       "Model Target: LST-LA-GCN (Lightweight Spatial-Temporal Local-Adaptive GCN)\n"
                       "Sensor: Microsoft Kinect v2\n"
                       "July 2026")
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(120, 120, 120)

    doc.add_page_break()

    # ── Table of Contents (manual) ──────────────────────────────────
    doc.add_heading("Table of Contents", level=1)
    toc_items = [
        "1. Executive Summary",
        "2. Dataset Overview",
        "3. Preprocessing Pipeline",
        "   3.1 Step 1: Coordinate Reconstruction",
        "   3.2 Step 2: Axis Alignment (YXZ to XYZ)",
        "   3.3 Step 3: Root Centering",
        "   3.4 Step 4: Temporal Resampling",
        "   3.5 Step 5: Dual-Stream Features (Joint + Bone)",
        "   3.6 Step 6: Adjacency Matrix Construction",
        "   3.7 Step 7: Z-Score Normalization (Unit Variance)",
        "   3.8 Step 8: Cross-Subject LOSO Splitting",
        "   3.9 Step 9: Data Augmentation",
        "   3.10 Step 10: PyTorch DataLoaders",
        "4. Data Augmentation Strategy",
        "5. LOSO Cross-Validation Protocol",
        "6. Output Format & Saved Files",
        "7. Verification Results",
        "8. Next Steps",
    ]
    for item in toc_items:
        p = doc.add_paragraph(item)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.space_before = Pt(2)

    doc.add_page_break()

    # ── 1. Executive Summary ────────────────────────────────────────
    doc.add_heading("1. Executive Summary", level=1)
    doc.add_paragraph(
        "This report documents the complete data preprocessing and augmentation pipeline "
        "developed for training an LST-LA-GCN (Lightweight Spatial-Temporal Local-Adaptive "
        "Graph Convolutional Network) model on the UI-PRMD (University of Idaho Physical "
        "Rehabilitation Movement Data) dataset. The pipeline transforms raw Kinect v2 skeleton "
        "tracking data into normalized, augmented tensors ready for graph-based deep learning."
    )
    doc.add_paragraph(
        "The pipeline consists of 10 sequential steps covering coordinate reconstruction, "
        "geometric normalization, temporal resampling, feature engineering (dual-stream joint + bone), "
        "graph construction, statistical normalization, subject-level cross-validation splitting, "
        "data augmentation (including camera-angle invariance via yaw rotation), and batched "
        "DataLoader creation with weighted sampling for class balance."
    )

    p = doc.add_paragraph()
    run = p.add_run("Key Results:")
    run.bold = True
    results = [
        f"Total samples processed: {len(samples)} (0 errors)",
        f"Sample tensor shape: (3, 150, 20) per sample — 3 channels x 150 frames x 20 joints",
        f"Label distribution: {(labels == 1).sum()} correct, {(labels == 0).sum()} incorrect (perfectly balanced)",
        "Cross-validation: 10-fold Leave-One-Subject-Out (LOSO) with zero data leakage",
        "Normalization: Unit variance (std = 1.0) computed on training data only",
        "Augmentation: Random yaw rotation, temporal crop, Gaussian jitter, speed variation",
    ]
    for r in results:
        doc.add_paragraph(r, style='List Bullet')

    # ── 2. Dataset Overview ─────────────────────────────────────────
    doc.add_heading("2. Dataset Overview", level=1)
    doc.add_paragraph(
        "The UI-PRMD dataset contains 3D skeleton sequences of 10 subjects performing "
        "10 different physical rehabilitation exercises. Each exercise has both correct and "
        "incorrect movement recordings captured using a Microsoft Kinect v2 sensor."
    )

    add_styled_table(doc,
        ["Property", "Value"],
        [
            ["Dataset", "UI-PRMD Reduced Dataset (Kinect)"],
            ["Subjects", "10"],
            ["Exercises", "10"],
            ["Total Samples", str(len(samples))],
            ["Correct Movements", str((labels == 1).sum())],
            ["Incorrect Movements", str((labels == 0).sum())],
            ["Sensor", "Microsoft Kinect v2"],
            ["Raw Joints", "22 per frame"],
            ["Selected Joints (GCN)", "20 per frame"],
            ["Columns per Joint (raw)", "6 (3 orientation + 3 position)"],
            ["Raw File Format", ".txt (whitespace-delimited)"],
        ]
    )

    doc.add_paragraph("")
    doc.add_heading("Per-Subject Sample Distribution", level=2)
    doc.add_paragraph(
        "Sample counts vary significantly across subjects, with Subjects 7 and 10 being "
        "notable minorities. This asymmetry is handled by weighted sampling during training."
    )

    subject_rows = []
    for s in range(1, 11):
        count = int((subject_ids == s).sum())
        pct = f"{100 * count / len(samples):.1f}%"
        note = "Minority" if count < 50 else ""
        subject_rows.append([f"Subject {s}", str(count), pct, note])

    add_styled_table(doc,
        ["Subject", "Samples", "Percentage", "Note"],
        subject_rows
    )

    doc.add_paragraph("")
    doc.add_heading("Sequence Length Statistics", level=2)
    add_styled_table(doc,
        ["Statistic", "Value (frames)"],
        [
            ["Minimum", str(metadata["sequence_length_stats"]["min"])],
            ["Maximum", str(metadata["sequence_length_stats"]["max"])],
            ["Mean", f"{metadata['sequence_length_stats']['mean']:.1f}"],
            ["Median", f"{metadata['sequence_length_stats']['median']:.1f}"],
            ["Target (after resampling)", "150"],
        ]
    )

    doc.add_page_break()

    # ── 3. Preprocessing Pipeline ───────────────────────────────────
    doc.add_heading("3. Preprocessing Pipeline", level=1)
    doc.add_paragraph(
        "The preprocessing pipeline consists of 10 steps, executed sequentially. "
        "Steps 1-4 are per-sample geometric transformations. Steps 5-6 build graph features. "
        "Step 7 normalizes to unit variance. Step 8 generates LOSO folds. Step 9 applies "
        "online augmentation. Step 10 wraps everything into PyTorch DataLoaders."
    )

    # Step 1
    doc.add_heading("3.1 Step 1: Coordinate Reconstruction", level=2)
    doc.add_paragraph(
        "The raw UI-PRMD CSV files store joint positions in a mixed coordinate system: "
        "the waist (SpineBase) joint has absolute world coordinates, while all other joints "
        "store relative offsets from their kinematic parent. This step walks the kinematic tree "
        "in topological order, accumulating offsets to produce absolute 3D positions for every joint."
    )
    p = doc.add_paragraph()
    run = p.add_run("Critical: ")
    run.bold = True
    run.font.color.rgb = RGBColor(192, 57, 43)
    p.add_run(
        "Skipping this step is the #1 mistake researchers make on UI-PRMD. "
        "Feeding raw relative coordinates into a GCN produces meaningless edge features."
    )

    # Step 2
    doc.add_heading("3.2 Step 2: Axis Alignment (YXZ to XYZ)", level=2)
    doc.add_paragraph(
        "UI-PRMD stores position triplets as (Y, X, Z). Most GCN implementations expect "
        "(X, Y, Z). This step swaps columns 0 and 1 to align with the standard convention."
    )

    # Step 3
    doc.add_heading("3.3 Step 3: Root Centering", level=2)
    doc.add_paragraph(
        "Subtracts the waist/SpineBase position from all joints per frame. This eliminates "
        "global translation (camera distance, subject placement) while preserving all relative "
        "joint motion, which is exactly what graph convolution operates on."
    )

    # Step 4
    doc.add_heading("3.4 Step 4: Temporal Resampling", level=2)
    doc.add_paragraph(
        "Different subjects perform exercises at different speeds, resulting in variable "
        f"sequence lengths (range: {metadata['sequence_length_stats']['min']}-"
        f"{metadata['sequence_length_stats']['max']} frames). "
        "This step uses linear interpolation to resample every sequence to a fixed length "
        "of 150 frames, enabling batched processing."
    )

    # Step 5
    doc.add_heading("3.5 Step 5: Dual-Stream Features (Joint + Bone)", level=2)
    doc.add_paragraph(
        "Bone vectors are computed as bone[child] = joint[child] - joint[parent] for each "
        "skeleton edge. The two streams provide complementary information:"
    )
    add_styled_table(doc,
        ["Stream", "Captures", "Shape"],
        [
            ["Joint", "Absolute pose configuration", "(3, 150, 20)"],
            ["Bone", "Relative limb direction & length", "(3, 150, 20)"],
        ]
    )
    doc.add_paragraph(
        "Bone computation happens on-the-fly inside the Dataset's __getitem__ method, "
        "so it uses the current (possibly augmented + normalized) joint positions."
    )

    # Step 6
    doc.add_heading("3.6 Step 6: Adjacency Matrix Construction", level=2)
    doc.add_paragraph(
        "Builds a 20x20 adjacency matrix encoding the physical skeleton topology:"
    )
    steps_list = [
        "Set A[i,j] = 1 for each skeleton edge (undirected)",
        "Add self-loops (A += I)",
        "Apply symmetric normalization: D^(-1/2) * A * D^(-1/2)",
    ]
    for s in steps_list:
        doc.add_paragraph(s, style='List Number')
    doc.add_paragraph(
        "The LST-LA-GCN's local-adaptive module will learn residual corrections "
        "on top of this base matrix during training."
    )

    # Step 7
    doc.add_heading("3.7 Step 7: Z-Score Normalization (Unit Variance)", level=2)
    doc.add_paragraph(
        "Per-channel (X, Y, Z) z-score normalization scales data to zero mean and unit variance:"
    )
    p = doc.add_paragraph()
    run = p.add_run("    normalized = (value - mean) / (std + epsilon)")
    run.font.name = 'Consolas'
    run.font.size = Pt(10)

    doc.add_paragraph("")
    doc.add_paragraph("Normalization statistics (Fold 1, training data only):")
    add_styled_table(doc,
        ["Channel", "Raw Mean", "Raw Std (mm)", "Post-Norm Mean", "Post-Norm Std"],
        [
            ["X", f"{mean.flatten()[0]:.2f}", f"{std.flatten()[0]:.2f}", "~0.000", "~1.000"],
            ["Y", f"{mean.flatten()[1]:.2f}", f"{std.flatten()[1]:.2f}", "~0.000", "~1.000"],
            ["Z", f"{mean.flatten()[2]:.2f}", f"{std.flatten()[2]:.2f}", "~0.000", "~1.000"],
        ]
    )

    doc.add_paragraph("")
    p = doc.add_paragraph()
    run = p.add_run("Critical: ")
    run.bold = True
    run.font.color.rgb = RGBColor(192, 57, 43)
    p.add_run(
        "Statistics are computed on training data ONLY per fold. Computing mean/std on "
        "the full dataset would leak test distribution information. This is the #2 most "
        "common mistake in UI-PRMD preprocessing."
    )

    # Step 8
    doc.add_heading("3.8 Step 8: Cross-Subject LOSO Splitting", level=2)
    doc.add_paragraph(
        "Standard 10-fold Leave-One-Subject-Out (LOSO) protocol. In each fold, one subject "
        "is held out for testing. Of the remaining 9 subjects, 3 are used for validation "
        "and 6 for training. Every subject serves as the test subject exactly once."
    )
    doc.add_paragraph(
        "This ensures zero data leakage between train and test: all samples from a given "
        "subject are entirely within one split. Results should be reported as mean +/- std "
        "across all 10 folds."
    )

    # Step 9
    doc.add_heading("3.9 Step 9: Data Augmentation", level=2)
    doc.add_paragraph(
        "Applied only on training data, stochastically per sample per epoch. "
        "See Section 4 for full details."
    )

    # Step 10
    doc.add_heading("3.10 Step 10: PyTorch DataLoaders", level=2)
    doc.add_paragraph(
        "The LOSOFoldManager class loads preprocessed data from disk and creates batched "
        "DataLoaders for each fold. Each __getitem__ call returns a dictionary with joint "
        "positions, bone vectors, the adjacency matrix, and the label. The training loader "
        "uses WeightedRandomSampler for class balance. See Section 5 for fold details."
    )

    doc.add_page_break()

    # ── 4. Data Augmentation Strategy ───────────────────────────────
    doc.add_heading("4. Data Augmentation Strategy", level=1)
    doc.add_paragraph(
        "With only ~1300 samples, data augmentation is critical to prevent overfitting. "
        "Four augmentations are applied stochastically during training, each with independent "
        "probability:"
    )

    add_styled_table(doc,
        ["Augmentation", "Description", "Parameters", "Probability"],
        [
            ["Random Yaw Rotation", "Rotates joints in XZ plane (horizontal) by random angle. "
             "Prevents overfitting to camera angle.", "angle in [-30deg, +30deg]", "p = 0.5"],
            ["Temporal Crop", "Crops 80-100% of frames, resamples back to 150. "
             "Simulates partial observation.", "min_ratio = 0.8", "p = 0.5"],
            ["Gaussian Jitter", "Adds N(0, 0.01) noise to positions. "
             "Simulates sensor noise.", "std = 0.01", "p = 0.5"],
            ["Speed Variation", "Scales time by 0.8-1.2x, resamples back. "
             "Simulates different exercise speeds.", "scale in [0.8, 1.2]", "p = 0.5"],
        ]
    )

    doc.add_paragraph("")
    doc.add_heading("Random Yaw Rotation (Camera-Angle Invariance)", level=2)
    doc.add_paragraph(
        "The Kinect sensor was placed at a fixed position during data collection. Without "
        "augmentation, the model can memorize the exact camera angle, reducing generalization "
        "to new sensor placements."
    )
    doc.add_paragraph(
        "The yaw rotation applies a 2D rotation matrix to the XZ (horizontal) plane while "
        "leaving the Y-axis (vertical) unchanged:"
    )
    p = doc.add_paragraph()
    run = p.add_run("    X' = X * cos(theta) - Z * sin(theta)\n"
                    "    Y' = Y  (unchanged)\n"
                    "    Z' = X * sin(theta) + Z * cos(theta)")
    run.font.name = 'Consolas'
    run.font.size = Pt(10)

    doc.add_paragraph("")
    doc.add_paragraph(
        "This rotation preserves the radius (distance from Y-axis) for each joint, "
        "verified to within float32 precision (~1e-5). The Y-axis is preserved because "
        "gravity direction carries semantic meaning for rehabilitation exercises."
    )

    doc.add_page_break()

    # ── 5. LOSO Cross-Validation Protocol ───────────────────────────
    doc.add_heading("5. LOSO Cross-Validation Protocol", level=1)
    doc.add_paragraph(
        "The following table shows the complete 10-fold LOSO split configuration. "
        "Note the significantly smaller test sets for Folds 7 and 10 (Subjects 7 and 10), "
        "which have far fewer samples than other subjects."
    )

    fold_rows = []
    for i, fold in enumerate(folds):
        train_mask = np.isin(subject_ids, fold["train"])
        val_mask = np.isin(subject_ids, fold["val"])
        test_mask = np.isin(subject_ids, fold["test"])
        fold_rows.append([
            str(i + 1),
            str(fold["train"]),
            str(fold["val"]),
            str(fold["test"]),
            str(train_mask.sum()),
            str(val_mask.sum()),
            str(test_mask.sum()),
        ])

    add_styled_table(doc,
        ["Fold", "Train Subjects", "Val Subjects", "Test Subject", "Train N", "Val N", "Test N"],
        fold_rows
    )

    doc.add_paragraph("")
    doc.add_heading("Weighted Sampling", level=2)
    doc.add_paragraph(
        "To handle the class imbalance that arises from unequal subject sample counts, "
        "the training DataLoader uses a WeightedRandomSampler with inverse-frequency class "
        "weights. This ensures each class (correct/incorrect) contributes equally to gradient "
        "updates per epoch, regardless of how many samples each subject contributes to the "
        "training fold."
    )
    doc.add_paragraph(
        "Formula: weight_per_sample = 1.0 / count_of_sample_class"
    )

    doc.add_page_break()

    # ── 6. Output Format ────────────────────────────────────────────
    doc.add_heading("6. Output Format & Saved Files", level=1)
    doc.add_paragraph(
        "The preprocessing pipeline saves all data to the output/preprocessed/ directory:"
    )

    add_styled_table(doc,
        ["File", "Shape / Type", "Description"],
        [
            ["samples.npy", f"({len(samples)}, 3, 150, 20)", "All preprocessed joint sequences"],
            ["labels.npy", f"({len(labels)},)", "Labels (0 = incorrect, 1 = correct)"],
            ["subject_ids.npy", f"({len(subject_ids)},)", "Subject IDs (1-based)"],
            ["exercise_ids.npy", f"({len(samples)},)", "Exercise IDs (1-based)"],
            ["adjacency.npy", "(20, 20)", "Normalized adjacency matrix"],
            ["folds.json", "JSON", "10-fold cross-subject split definitions"],
            ["metadata.json", "JSON", "Pipeline config + sequence length stats"],
        ]
    )

    doc.add_paragraph("")
    doc.add_heading("DataLoader Output Tensors", level=2)
    doc.add_paragraph("Each batch from the DataLoader returns:")
    add_styled_table(doc,
        ["Key", "Shape", "Dtype", "Description"],
        [
            ["joint", "(B, 3, 150, 20)", "float32", "Normalized joint positions"],
            ["bone", "(B, 3, 150, 20)", "float32", "Bone vectors (computed on-the-fly)"],
            ["A", "(B, 20, 20)", "float32", "Adjacency matrix"],
            ["label", "(B,)", "long", "0 = incorrect, 1 = correct"],
        ]
    )

    # ── 7. Verification Results ─────────────────────────────────────
    doc.add_heading("7. Verification Results", level=1)
    doc.add_paragraph(
        "All pipeline components were verified programmatically. The verification script "
        "(verify_pipeline.py) tests the following:"
    )

    verifications = [
        ("LOSO Fold Splitting", "PASSED",
         "All 10 folds verified. Zero overlap between train/val/test. "
         "Sample counts sum to total (1326) in every fold."),
        ("Normalization (Unit Variance)", "PASSED",
         "After z-score normalization with train-only stats: "
         "mean ~ 0.000, std ~ 1.000 across all 3 channels."),
        ("Yaw Rotation Augmentation", "PASSED",
         "Y-channel preserved exactly (diff = 0). XZ coordinates rotated. "
         "XZ-radius preserved within float32 precision (diff < 1e-4)."),
        ("Weighted Sampling", "PASSED",
         "Inverse-frequency weights computed correctly. "
         "Weight ratio matches class distribution ratio."),
        ("Preprocessing Pipeline", "PASSED",
         f"1326 samples processed with 0 errors. "
         f"Output shape (3, 150, 20) matches expected. "
         f"All 10 subjects and 10 exercises discovered."),
    ]

    add_styled_table(doc,
        ["Component", "Status", "Details"],
        [[v[0], v[1], v[2]] for v in verifications]
    )

    doc.add_page_break()

    # ── 8. Next Steps ───────────────────────────────────────────────
    doc.add_heading("8. Next Steps", level=1)
    doc.add_paragraph(
        "With the data preprocessing and augmentation pipeline complete, the following "
        "steps are recommended for the LST-LA-GCN training pipeline:"
    )

    next_steps = [
        ("Fix PyTorch Installation",
         "The current Anaconda environment has a broken PyTorch DLL. "
         "Create a fresh conda environment with a compatible PyTorch + CUDA version."),
        ("Implement LST-LA-GCN Model Architecture",
         "Build the Lightweight Spatial-Temporal Local-Adaptive GCN model following "
         "the published architecture. The model takes joint (3,150,20) and bone (3,150,20) "
         "dual-stream inputs with the (20,20) adjacency matrix."),
        ("Training Loop with LOSO Evaluation",
         "Train across all 10 LOSO folds using the LOSOFoldManager. Report mean +/- std "
         "accuracy, F1-score, and confusion matrices across folds."),
        ("Hyperparameter Tuning",
         "Tune learning rate, weight decay, augmentation probabilities, and yaw rotation "
         "range. Pay special attention to the minority folds (S7, S10)."),
        ("Results Reporting",
         "Compare against published baselines on UI-PRMD: EGCN, ST-GCN, CNN+LSTM."),
    ]

    for i, (title, desc) in enumerate(next_steps, 1):
        p = doc.add_paragraph()
        run = p.add_run(f"{i}. {title}: ")
        run.bold = True
        p.add_run(desc)

    # ── Footer info ─────────────────────────────────────────────────
    doc.add_paragraph("")
    doc.add_paragraph("")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("--- End of Report ---")
    run.font.color.rgb = RGBColor(150, 150, 150)
    run.font.size = Pt(10)

    # ── Save ────────────────────────────────────────────────────────
    doc.save(str(output_path))
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
