#!/usr/bin/env python3
"""
CLI entry point for the UI-PRMD preprocessing pipeline.

Usage:
    python run_preprocessing.py --data_root ./data/UI-PRMD/Kinect --output_dir ./output/preprocessed
    python run_preprocessing.py --data_root ./data --flat_layout --target_len 120
    python run_preprocessing.py --help
"""

import argparse
import logging
import sys

from ui_prmd_preprocess.pipeline import PreprocessingPipeline
from ui_prmd_preprocess import config


def main():
    parser = argparse.ArgumentParser(
        description="UI-PRMD Preprocessing Pipeline for LST-LA-GCN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard run with hierarchical directory layout
  python run_preprocessing.py \\
      --data_root ./data/UI-PRMD/Reduced_Dataset/Kinect \\
      --output_dir ./output/preprocessed

  # Flat directory layout with custom sequence length
  python run_preprocessing.py \\
      --data_root ./data/flat_csvs \\
      --flat_layout \\
      --target_len 120

  # Verify output shapes only (dry run)
  python run_preprocessing.py \\
      --data_root ./data/UI-PRMD/Kinect \\
      --verify_only
        """,
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=str(config.DEFAULT_DATA_ROOT),
        help="Path to the UI-PRMD Kinect data directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(config.DEFAULT_OUTPUT_DIR),
        help="Directory to save preprocessed .npy files (default: %(default)s)",
    )
    parser.add_argument(
        "--target_len",
        type=int,
        default=config.TARGET_SEQUENCE_LENGTH,
        help="Fixed temporal sequence length (default: %(default)s)",
    )
    parser.add_argument(
        "--n_joints",
        type=int,
        default=config.N_JOINTS,
        help="Number of joints in the GCN skeleton (default: %(default)s)",
    )
    parser.add_argument(
        "--flat_layout",
        action="store_true",
        help="Use flat file discovery (all CSVs in one directory) instead of hierarchical",
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Run pipeline and print shapes/stats without saving",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    # Run the pipeline
    pipeline = PreprocessingPipeline(
        data_root=args.data_root,
        target_len=args.target_len,
        n_joints=args.n_joints,
        flat_layout=args.flat_layout,
    )

    pipeline.run()

    # Print verification info
    if pipeline.samples:
        print("\n" + "=" * 60)
        print(" Verification Summary")
        print("=" * 60)
        print(f"  Samples loaded:     {len(pipeline.samples)}")
        print(f"  Sample shape:       {pipeline.samples[0].shape}")
        print(f"  Expected shape:     (3, {args.target_len}, {args.n_joints})")
        print(f"  Adjacency shape:    {pipeline.adjacency_matrix.shape}")
        print(f"  Correct labels:     {sum(1 for l in pipeline.labels if l == 1)}")
        print(f"  Incorrect labels:   {sum(1 for l in pipeline.labels if l == 0)}")
        print(f"  Subjects:           {sorted(set(pipeline.subject_ids))}")
        print(f"  Exercises:          {sorted(set(pipeline.exercise_ids))}")

        # Show one fold
        fold_data = pipeline.get_fold_data(fold_index=0)
        print(f"\n  Fold 1 split:")
        print(f"    Train: {len(fold_data['train_samples'])} samples "
              f"(subjects {fold_data['fold_info']['train']})")
        print(f"    Val:   {len(fold_data['val_samples'])} samples "
              f"(subjects {fold_data['fold_info']['val']})")
        print(f"    Test:  {len(fold_data['test_samples'])} samples "
              f"(subjects {fold_data['fold_info']['test']})")
        print(f"    Norm mean: {fold_data['mean'].flatten()}")
        print(f"    Norm std:  {fold_data['std'].flatten()}")
        print("=" * 60)

    if not args.verify_only:
        pipeline.save(args.output_dir)
        print(f"\n[SUCCESS] Preprocessed data saved to: {args.output_dir}")
    else:
        print("\n[SUCCESS] Verification complete (--verify_only mode, nothing saved)")


if __name__ == "__main__":
    main()
