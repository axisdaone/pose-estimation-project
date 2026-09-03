#!/usr/bin/env python3
"""
Overnight Training Runner for LST-LA-GCN.

Wraps train.py with automatic restart on failure, logging to file,
and graceful shutdown. Designed to run unattended overnight.

Usage:
    python run_overnight.py
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 30  # Wait before restarting after a crash
LOG_FILE = Path("./output/overnight_training.log")
CHECKPOINT_FILE = Path("./output/training_checkpoint.json")

# Training arguments — full 10-fold LOSO with multithreading
TRAIN_ARGS = [
    sys.executable, "train.py",
    "--num_folds", "10",
    "--epochs", "100",
    "--batch_size", "32",
    "--lr", "1e-3",
    "--weight_decay", "1e-4",
    "--dropout", "0.3",
    "--patience", "15",
    "--num_workers", "0",  # In-memory RAM dataset: 0 is 2.5x faster on Windows than multiprocessing
    "--parallel_folds", "0",  # Auto-detect (2 for CPU, 1 for GPU)
    "--checkpoint_file", str(CHECKPOINT_FILE),
    "--output_file", "./output/training_results.json",
]


def get_completed_folds():
    """Read checkpoint to see how many folds are done."""
    import json
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE) as f:
                data = json.load(f)
            return data.get("completed_folds", [])
        except (json.JSONDecodeError, KeyError):
            return []
    return []


def log_message(msg: str, log_file=None):
    """Print and optionally write to log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    if log_file:
        log_file.write(line + "\n")
        log_file.flush()


def main():
    # Ensure output directory exists
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(LOG_FILE, "a", encoding="utf-8") as log_f:
        log_message("=" * 70, log_f)
        log_message("  OVERNIGHT TRAINING RUNNER — LST-LA-GCN / UI-PRMD", log_f)
        log_message(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_f)
        log_message(f"  Max retries: {MAX_RETRIES}", log_f)
        log_message(f"  Checkpoint: {CHECKPOINT_FILE}", log_f)
        log_message(f"  Log: {LOG_FILE}", log_f)
        log_message("=" * 70, log_f)

        attempt = 0

        while attempt < MAX_RETRIES:
            attempt += 1

            # Check progress
            completed = get_completed_folds()
            if len(completed) >= 10:
                log_message(
                    f"All 10 folds already completed! No training needed.", log_f
                )
                break

            log_message(f"\n--- Attempt {attempt}/{MAX_RETRIES} ---", log_f)
            log_message(
                f"Completed folds so far: {sorted(completed) if completed else 'none'}",
                log_f,
            )
            log_message(f"Remaining: {10 - len(completed)} folds", log_f)
            log_message(f"Command: {' '.join(TRAIN_ARGS)}", log_f)

            start_time = time.time()

            try:
                # Run training, streaming output to both console and log file
                process = subprocess.Popen(
                    TRAIN_ARGS,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,  # Line-buffered
                    cwd=str(Path(__file__).parent),
                    env={
                        **os.environ,
                        "PYTHONUNBUFFERED": "1",
                        "KMP_DUPLICATE_LIB_OK": "TRUE",
                    },
                )

                # Stream output line by line
                for line in process.stdout:
                    line = line.rstrip("\n")
                    print(line, flush=True)
                    log_f.write(line + "\n")
                    log_f.flush()

                return_code = process.wait()
                elapsed = time.time() - start_time

                if return_code == 0:
                    log_message(
                        f"\nTraining completed successfully in {elapsed:.1f}s "
                        f"({elapsed/60:.1f} min)",
                        log_f,
                    )
                    # Check final state
                    completed = get_completed_folds()
                    log_message(
                        f"Final completed folds: {sorted(completed)} "
                        f"({len(completed)}/10)",
                        log_f,
                    )
                    break
                else:
                    log_message(
                        f"\nTraining exited with code {return_code} after "
                        f"{elapsed:.1f}s",
                        log_f,
                    )
                    completed = get_completed_folds()
                    log_message(
                        f"Folds saved in checkpoint: {sorted(completed)}",
                        log_f,
                    )

            except KeyboardInterrupt:
                log_message("\nKeyboard interrupt received. Shutting down...", log_f)
                try:
                    process.terminate()
                    process.wait(timeout=10)
                except Exception:
                    process.kill()
                completed = get_completed_folds()
                log_message(
                    f"Progress saved: {len(completed)}/10 folds complete", log_f
                )
                log_message("You can resume by running this script again.", log_f)
                sys.exit(0)

            except Exception as e:
                elapsed = time.time() - start_time
                log_message(f"\nUnexpected error after {elapsed:.1f}s: {e}", log_f)

            # Retry logic
            if attempt < MAX_RETRIES:
                log_message(
                    f"Retrying in {RETRY_DELAY_SECONDS}s... "
                    f"(will resume from checkpoint)",
                    log_f,
                )
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                log_message(
                    f"\nMax retries ({MAX_RETRIES}) reached. Giving up.", log_f
                )
                completed = get_completed_folds()
                log_message(
                    f"Final progress: {len(completed)}/10 folds saved in "
                    f"checkpoint.",
                    log_f,
                )

        log_message("\n" + "=" * 70, log_f)
        log_message(
            f"  Overnight runner finished at "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            log_f,
        )
        log_message("=" * 70, log_f)


if __name__ == "__main__":
    main()
