# UI-PRMD: Spatio-Temporal Graph Convolutional Network for Physical Rehabilitation Movement Assessment

A complete, end-to-end deep learning framework for automated evaluation and quality assessment of physical rehabilitation exercises using the **UI-PRMD (University of Idaho - Physical Rehabilitation Movement Dataset)**.

This repository implements a 10-step skeletal preprocessing pipeline alongside **LST-LA-GCN** (*Lightweight Spatio-Temporal Local-Adaptive Graph Convolutional Network*), featuring dual-stream spatial-temporal modeling, adaptive graph topology learning, cross-subject validation, and multi-level parallelized training.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Repository Architecture](#repository-architecture)
4. [Technologies & Frameworks](#technologies--frameworks)
5. [Preprocessing Pipeline (Steps 1–10)](#preprocessing-pipeline-steps-110)
6. [Model Architecture: LST-LA-GCN](#model-architecture-lst-la-gcn)
7. [Training, Validation & Evaluation Strategy](#training-validation--evaluation-strategy)
8. [Multithreading & Overnight Execution](#multithreading--overnight-execution)
9. [Installation & Getting Started](#installation--getting-started)
10. [Artifacts & Outputs](#artifacts--outputs)

---

## Project Overview

In physical therapy and telerehabilitation, patients often perform prescribed exercises at home without direct clinical supervision. Performing movements incorrectly can lead to ineffective recovery or injury. Automated assessment using optical motion capture or depth cameras (e.g., Microsoft Kinect v2) provides objective feedback on exercise execution.

This project addresses the challenge of **binary movement classification** (classifying an exercise repetition as **Correct / Optimal** vs. **Incorrect / Non-optimal**):
- Handles high inter-subject anatomical variations (height, limb proportions, movement speed).
- Models spatial relationships between physical joints and long-range kinematic dependencies.
- Evaluates models strictly under **Leave-One-Subject-Out (LOSO)** cross-validation to guarantee generalization to unseen patients.

---

## Dataset Description

The **UI-PRMD** dataset contains 3D skeletal data collected from 10 healthy human subjects performing 10 common physical rehabilitation exercises:
1. Deep Squat
2. Hurdle Step
3. Inline Lunge
4. Side Lunge
5. Sit to Stand
6. Standing Shoulder Abduction
7. Standing Shoulder Extension
8. Standing Shoulder Internal-External Rotation
9. Standing Hip Abduction
10. Standing Knee Flexion

Each subject performed approximately 10 repetitions per exercise in both correct and incorrect forms. The raw Kinect tracking records 22 joints with 6 degrees of data per row (3 Euler orientation angles + 3 position coordinates in YXZ sensor order).

---

## Repository Architecture

```text
ui-prmd-preprocessing/
│
├── data/                                 # Raw UI-PRMD dataset directory
│   └── UI-PRMD/
│       └── Reduced_Dataset/
│           └── Kinect/                   # Hierarchical or flat CSV files
│
├── output/                               # Pipeline outputs and results
│   ├── preprocessed/                     # Serialized preprocessed tensors (.npy)
│   │   ├── samples.npy                   # (N, 3, 150, 20) Joint positions
│   │   ├── labels.npy                    # (N,) Ground truth labels
│   │   ├── subject_ids.npy               # (N,) Subject identifiers (1-10)
│   │   ├── exercise_ids.npy              # (N,) Exercise identifiers (1-10)
│   │   ├── adjacency.npy                 # (20, 20) Normalized graph adjacency
│   │   ├── folds.json                    # LOSO cross-validation split definitions
│   │   └── metadata.json                 # Preprocessing configuration & stats
│   ├── training_checkpoint.json          # Incremental fold progress (crash resume)
│   ├── training_results.json             # Final 10-fold cross-validation metrics
│   └── overnight_training.log            # Timestamped training logs
│
├── ui_prmd_preprocess/                   # Core preprocessing & model package
│   ├── __init__.py
│   ├── config.py                         # Kinematic tree, hyperparameters, paths
│   ├── data_loader.py                    # CSV parsing & dataset discovery
│   ├── preprocessing.py                  # Steps 1–4: Geometric transformations
│   ├── filtering.py                      # Step 1.5: Kalman filter trajectory smoothing
│   ├── features.py                       # Steps 5–6: Dual-stream & graph adjacency
│   ├── normalization.py                  # Step 7: Leakage-free z-score normalization
│   ├── splits.py                         # Step 8: 10-fold LOSO partitioning
│   ├── augmentation.py                   # Step 9: Stochastic online augmentations
│   ├── dataset.py                        # Step 10: PyTorch Dataset & Weighted Sampler
│   ├── pipeline.py                       # Pipeline Orchestrator (Steps 1–10 runner)
│   └── model/                            # Neural network definition
│       ├── __init__.py
│       ├── attention.py                  # Temporal attention modules
│       ├── graph_conv.py                 # Local-Adaptive Graph Convolution (LA-GCN)
│       ├── st_block.py                   # Spatio-Temporal Block (GCN + TCN + Attn)
│       ├── temporal_conv.py              # Multi-scale temporal convolutions
│       ├── network.py                    # Dual-stream LST-LA-GCN model
│       └── test_model.py                 # Architecture unit tests
│
├── train.py                              # Main training script (10-fold LOSO, parallel)
├── run_overnight.py                      # Robust daemon wrapper with auto-restart
├── run_preprocessing.py                  # Standalone preprocessing CLI
├── test_kalman.py                        # Kalman filter validation & benchmarks
└── requirements.txt                      # Project dependencies
```

---

## Technologies & Frameworks

- **Deep Learning Framework:** [PyTorch](https://pytorch.org/) (Custom multi-stream Spatio-Temporal Graph Convolutions, Custom Autograd modules, WeightedRandomSampler).
- **Scientific Computing:** [NumPy](https://numpy.org/) (Vectorized array transformations, tensor reshaping, coordinate geometry).
- **Signal Processing & Kinematics:** [SciPy](https://scipy.org/) (1D linear and cubic spline interpolation, kinematic tree graph traversal).
- **Filtering & State Estimation:** Constant-velocity Kalman Filter formulation using discrete state-space modeling.
- **Concurrency & Multithreading:** Python `concurrent.futures.ProcessPoolExecutor`, PyTorch DataLoader multi-processing worker threads (`num_workers`), and OpenMP thread tuning (`torch.set_num_threads`).
- **OS & Environment Workarounds:** Intel OpenMP runtime duplicate resolution (`KMP_DUPLICATE_LIB_OK`).

---

## Preprocessing Pipeline (Steps 1–10)

Raw depth-sensor skeleton sequences suffer from missing coordinates, variable repetition speeds, sensor noise, non-standard axis ordering, and global translation. The pipeline transforms raw CSV motion capture data into standardized spatio-temporal graphs across 10 deterministic steps:

```text
Raw CSV (T, 22*6)
  │
  ├─► [Step 1] Kinematic Tree Coordinate Reconstruction (Relative -> Absolute 3D)
  │     └─► [Step 1.5] Constant-Velocity Kalman Filter Smoothing (Sensor De-noising)
  ├─► [Step 2] Axis Realignment (YXZ -> Standard Cartesian XYZ)
  ├─► [Step 3] Root-Centering / Translation Invariance (Subtract Waist per frame)
  ├─► [Step 4] Temporal Resampling (Interpolation to fixed T=150 frames)
  ├─► [Step 5] Dual-Stream Feature Extraction (Joint Coordinates + Bone Vectors)
  ├─► [Step 6] Normalized Physical Graph Adjacency Construction (N=20 joints, 19 edges)
  ├─► [Step 7] Strict Leakage-Free Z-score Normalization (Mean/Std on Train fold only)
  ├─► [Step 8] Cross-Subject Leave-One-Subject-Out (LOSO) Split Generation
  ├─► [Step 9] Online Stochastic Data Augmentation (Rotation, Jitter, Crop, Scaling)
  └─► [Step 10] PyTorch DataLoaders & Class-Imbalance Weighted Sampling
```

### Step 1: Kinematic Coordinate Reconstruction
In raw UI-PRMD recordings, only the root joint (`Waist` / `SpineBase`) is stored in absolute Cartesian coordinates; all subsequent child joints are recorded as relative positional offsets relative to their parent bone in the kinematic tree.
Using a topological tree traversal:
$$\mathbf{P}_{\text{child}} = \mathbf{P}_{\text{parent}} + \Delta \mathbf{P}_{\text{child}}$$
This recovers the full 3D body structure in metric space for the selected 20-joint subset.

### Step 1.5: Kalman Filter Trajectory Smoothing
Low-cost depth sensors exhibit high-frequency jitter and occlusions. An optional linear Kalman filter with a constant-velocity kinematic model is applied:
- **State vector:** $\mathbf{x}_t = [p_x, p_y, p_z, v_x, v_y, v_z]^T$
- **Process covariance ($Q$):** Parameterized by process noise ($10^{-4}$).
- **Measurement covariance ($R$):** Parameterized by sensor noise ($10^{-2}$).
Filters trajectory noise without introducing unnatural phase lag.

### Step 2: Sensor Axis Alignment
The Kinect sensor coordinate system stores positional data in $(Y, X, Z)$ order (Height, Width, Depth). The pipeline permutes the coordinate columns into standard Cartesian $(X, Y, Z)$ format:
- $X$: Lateral (Width)
- $Y$: Vertical (Height)
- $Z$: Anterior-Posterior (Depth)

### Step 3: Root Centering (Translation Invariance)
To eliminate variance caused by where the patient stood relative to the camera, the root joint ($\text{Waist}$ at index 0) is subtracted from all 20 joints in every frame:
$$\mathbf{P}'_{j}(t) = \mathbf{P}_{j}(t) - \mathbf{P}_{\text{root}}(t), \quad \forall j \in \{0, \dots, 19\}$$
The waist remains at $(0, 0, 0)$ across all frames, isolating pure body articulation.

### Step 4: Temporal Resampling
Patients perform exercises at different rates (sequence durations range from under 100 to over 600 frames). Convolutional architectures require uniform temporal dimensions. 
All sequences are resampled to a fixed temporal length of $T = 150$ frames using linear interpolation:
$$t_{\text{target}} \in [0, 1] \implies \mathbf{P}_{j}(t) \in \mathbb{R}^{150 \times 20 \times 3}$$

### Step 5: Dual-Stream Feature Extraction
Two complementary streams are extracted:
1. **Joint Stream:** Absolute 3D positions $\mathbf{J} \in \mathbb{R}^{3 \times T \times N}$.
2. **Bone Stream:** Vector offsets between adjacent joints connected by physical bones:
   $$\mathbf{B}_{child}(t) = \mathbf{J}_{child}(t) - \mathbf{J}_{parent}(t)$$
Joint positions capture absolute pose, while bone vectors directly capture limb angles and direction invariant to bone lengths.

### Step 6: Physical Graph Adjacency Matrix
The human body is modeled as an undirected graph $G = (V, E)$ with $N = 20$ joints and 19 biological edges. The base symmetric normalized adjacency matrix is computed with self-loops:
$$\tilde{A} = A + I_{N}, \quad \tilde{D}_{ii} = \sum_{j} \tilde{A}_{ij}, \quad \hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$$

### Step 7: Leakage-Free Z-Score Normalization
To prevent data snooping across subjects, normalization statistics (channel-wise mean $\mu$ and standard deviation $\sigma$) are computed **strictly on the training split** of each fold:
$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}} + \epsilon}$$
Validation and test sets are transformed using the training parameters.

### Step 8: Cross-Subject LOSO Splitting
To simulate deployment to a clinic where the model encounters new patients, evaluation uses 10-fold Leave-One-Subject-Out (LOSO):
- **Test:** 1 unseen subject ($10\%$).
- **Validation:** 3 subjects ($30\%$).
- **Train:** 6 subjects ($60\%$).

### Step 9: Online Stochastic Data Augmentation
Applied dynamically during training epochs to combat overfitting:
- **Random Yaw Rotation:** Rotates the skeleton around the vertical $Y$-axis within $[ -30^\circ, +30^\circ ]$ to simulate arbitrary camera viewing angles.
- **Random Temporal Crop:** Truncates between $80\%$ and $100\%$ of frames and resamples back to 150 frames.
- **Gaussian Joint Jitter:** Adds $\mathcal{N}(0, 0.01^2)$ spatial noise.
- **Random Speed Variation:** Temporally compresses or stretches the sequence by $\pm 20\%$.

### Step 10: PyTorch DataLoaders & Class Imbalance Handling
Because some subjects performed fewer repetitions (e.g., Subject 7 has 34 samples while Subject 8 has 172), classes can become imbalanced within training folds.
A `WeightedRandomSampler` assigns sample weights inversely proportional to class frequencies:
$$w_c = \frac{1}{N_c}, \quad W_i = w_{\text{label}_i}$$
Ensures that each batch presents balanced gradients for both correct and incorrect executions.

---

## Model Architecture: LST-LA-GCN

The **LST-LA-GCN** is optimized for high accuracy on skeletal sequences while remaining compact (~2.67M parameters per fold) to avoid overfitting small clinical datasets.

```text
                        ┌─────────────────────────────────────────────────────────┐
                        │                      JOINT STREAM                       │
joint (B, 3, 150, 20) ─►│  BN -> ST-Block (64) -> ST-Block (128, s=2)             │
                        │        -> ST-Block (256, s=2) -> GAP -> Dropout -> FC   │─┐
                        └─────────────────────────────────────────────────────────┘ │
                                                                                    ├─► Average Logits -> Pred
                        ┌─────────────────────────────────────────────────────────┐ │
                        │                      BONE STREAM                        │─┘
 bone (B, 3, 150, 20) ─►│  BN -> ST-Block (64) -> ST-Block (128, s=2)             │
                        │        -> ST-Block (256, s=2) -> GAP -> Dropout -> FC   │
                        └─────────────────────────────────────────────────────────┘
```

### 1. Local-Adaptive Graph Convolution (LA-GCN)
Unlike vanilla ST-GCN, which relies solely on a fixed physical graph, the spatial convolution dynamically adapts:
$$A_{\text{final}} = A_{\text{physical}} + A_{\text{adaptive}} + A_{\text{attention}}$$
- **$A_{\text{physical}}$:** Fixed symmetric-normalized anatomical bone connectivity.
- **$A_{\text{adaptive}}$:** Learnable $N \times N$ weight matrix capturing global non-physical joint correlations (e.g., wrist-to-wrist synchronization).
- **$A_{\text{attention}}$:** Sample-dependent dynamic self-attention matrix computed via $1 \times 1$ convolutions and softmax:
  $$A_{\text{att}} = \text{Softmax}\left(\theta(X)^T \phi(X)\right)$$

### 2. Multi-Scale Temporal Convolutions + Temporal Attention
Each spatio-temporal block uses multi-branch temporal convolutions with varying kernel sizes ($9, 5, 3$) and max-pooling, followed by a Temporal Attention module that identifies key transitional phases in the exercise repetition.

### 3. Late Fusion Logit Ensemble
Predictions from the Joint Stream and Bone Stream are averaged at inference:
$$\text{Logits} = \frac{\text{Logits}_{\text{joint}} + \text{Logits}_{\text{bone}}}{2}$$

---

## Training, Validation & Evaluation Strategy

### Optimization & Hyperparameters
- **Optimizer:** AdamW (Initial Learning Rate: $10^{-3}$, Weight Decay: $10^{-4}$)
- **Learning Rate Scheduler:** Cosine Annealing with Warm Restarts ($T_0 = 20$, $T_{\text{mult}} = 2$, $\eta_{\min} = 10^{-6}$)
- **Gradient Clipping:** Max norm $5.0$ to ensure stability
- **Loss Function:** Binary Cross Entropy / CrossEntropyLoss
- **Early Stopping:** Monitored on validation accuracy with a patience of 15 epochs

### Comprehensive Evaluation Metrics
At the conclusion of each fold, models are evaluated on the held-out test subject:
- **Accuracy**
- **Macro & Per-Class Precision**
- **Macro & Per-Class Recall**
- **Macro & Per-Class F1-Score**
- **Confusion Matrix**

Aggregated statistics across all 10 folds compute the mean and standard deviation ($\mu \pm \sigma$) for each metric.

---

## Multithreading & Overnight Execution

To allow fast and fault-tolerant full 10-fold training on local hardware, a 3-level concurrency hierarchy is implemented:

| Level | Mechanism | Description |
|---|---|---|
| **Level 1: Concurrent Folds** | `ProcessPoolExecutor` | Runs independent fold models simultaneously (e.g., Fold 1 & Fold 2 concurrently on multi-core CPUs). |
| **Level 2: Data Loading** | PyTorch `DataLoader(num_workers=4)` | Parallel background worker threads read and augment batches asynchronously. |
| **Level 3: PyTorch Threading** | `torch.set_num_threads()` | Distributes internal BLAS/matrix multiplication across CPU cores. |

### Overnight Fault-Tolerant Runner (`run_overnight.py`)
- **Auto-Restart Loop:** Catches uncaught process terminations and restarts training up to 5 times automatically.
- **State Checkpointing:** Incremental progress is committed to `output/training_checkpoint.json`. If execution is interrupted, the runner skips completed folds and resumes immediately from the remaining folds.
- **Log Streaming:** All output lines are timestamped and mirrored live to `output/overnight_training.log`.

---

## Installation & Getting Started

### 1. Prerequisites
Ensure Python 3.8+ is installed. Install required packages:
```bash
pip install -r requirements.txt
```

### 2. Run Preprocessing
Transform raw dataset CSVs into tensor archives:
```bash
python run_preprocessing.py --data_root ./data/UI-PRMD/Reduced_Dataset/Kinect --output_dir ./output/preprocessed
```

### 3. Run Standard Training
Run 10-fold cross-validation directly:
```bash
python train.py --num_folds 10 --epochs 100 --batch_size 32 --parallel_folds 2 --num_workers 4
```

### 4. Run Unattended Overnight Training
Launch the resilient runner with auto-restart:
```bash
python run_overnight.py
```

To stop all active training processes:
```powershell
Stop-Process -Name "python" -Force
```

---

## Artifacts & Outputs

All results are automatically generated in the `output/` directory:
- `output/preprocessed/`: Tensor arrays ready for immediate training.
- `output/training_checkpoint.json`: Checkpoint containing per-fold metrics and completed fold IDs.
- `output/training_results.json`: Final aggregated summary with cross-validation means, variances, and confusion matrices.
- `output/overnight_training.log`: Chronological log containing loss progression, validation benchmarks, and system events.
