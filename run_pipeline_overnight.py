"""Fresh preprocessing, validated 5/2/3 folds, and retried overnight training."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import ctypes
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def validate_raw_data():
    from ui_prmd_preprocess.data_loader import discover_files
    from ui_prmd_preprocess.config import DEFAULT_DATA_ROOT
    manifest = discover_files(ROOT / DEFAULT_DATA_ROOT)
    if not manifest:
        raise RuntimeError("No raw samples found")
    for entry in manifest:
        if entry.get("coordinate_format") == "kinect_pair":
            from ui_prmd_preprocess.kinect import load_triplets
            position = entry["filepath"]
            angle = position.parent.parent / "angles" / position.name.replace("_positions", "_angles")
            if load_triplets(position).shape != load_triplets(angle).shape:
                raise RuntimeError(f"Position/angle frame mismatch: {position}")
            continue
        with entry["filepath"].open() as source:
            for line in source:
                if line.strip():
                    columns = len(line.replace(",", " ").split())
                    if columns != 132:
                        raise RuntimeError(
                            f"{entry['filepath']}: {columns} columns; this loader expects "
                            "132 interleaved Kinect orientation/position columns. "
                            "Verify the source format before training; zero padding is invalid.")
                    break


def validate_data(directory):
    import numpy as np
    from ui_prmd_preprocess.data_loader import discover_files
    from ui_prmd_preprocess.config import DEFAULT_DATA_ROOT
    from ui_prmd_preprocess.splits import generate_cross_subject_folds
    samples = np.load(directory / "samples.npy", mmap_mode="r")
    labels = np.load(directory / "labels.npy")
    subjects = np.load(directory / "subject_ids.npy")
    folds = json.loads((directory / "folds.json").read_text())
    assert samples.shape == (len(labels), 3, 150, 20)
    assert len(labels) == len(subjects) == len(discover_files(ROOT / DEFAULT_DATA_ROOT))
    assert np.isfinite(samples).all()
    assert set(subjects) == set(range(1, 11))
    assert folds == generate_cross_subject_folds()
    for fold in folds:
        for role in ("train", "val", "test"):
            assert set(labels[np.isin(subjects, fold[role])]) == {0, 1}
    print(f"Validated {len(labels)} samples and ten disjoint 5/2/3 folds", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", default="output/overnight_2026_10_09")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    os.chdir(ROOT)
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    status = {"started": datetime.now().isoformat(), "runner_pid": os.getpid(),
              "python": sys.executable, "subject_split": [5, 2, 3]}

    def update(stage, **details):
        status.update(stage=stage, updated=datetime.now().isoformat(), **details)
        temp = run / "status.tmp"
        temp.write_text(json.dumps(status, indent=2), encoding="utf-8")
        temp.replace(run / "status.json")

    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
           "OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads)}
    data = run / "preprocessed"
    checkpoint = run / "checkpoint.json"
    awake = os.name == "nt"
    if awake:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        with (run / "pipeline.log").open("a", encoding="utf-8") as log:
            def execute(command):
                print("Running:", " ".join(map(str, command)), flush=True)
                process = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=env,
                                           stdout=log, stderr=subprocess.STDOUT)
                update(status["stage"], child_pid=process.pid)
                return process.wait()

            if not (run / "preprocessing_complete.json").exists():
                update("validating_raw_data")
                validate_raw_data()
                update("preprocessing")
                rc = execute([sys.executable, "run_preprocessing.py", "--output_dir", data])
                if rc:
                    raise RuntimeError(f"Preprocessing exited with code {rc}")
                validate_data(data)
                (run / "preprocessing_complete.json").write_text(json.dumps(status), encoding="utf-8")
            else:
                validate_data(data)

            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("GPU run requested but CUDA unavailable in this environment")
            update("training", device=torch.cuda.get_device_name(0), torch_version=torch.__version__,
                   epochs=args.epochs, batch_size=args.batch_size, threads=args.threads)
            command = [sys.executable, "train.py", "--data_dir", data,
                       "--num_folds", "10", "--epochs", args.epochs,
                       "--batch_size", args.batch_size, "--patience", "15",
                       "--num_workers", "0", "--parallel_folds", "1",
                       "--threads_per_fold", args.threads, "--checkpoint_file", checkpoint,
                       "--model_dir", run / "models", "--output_file", run / "results.json"]
            for attempt in range(1, 6):
                update("training", attempt=attempt)
                rc = execute(command)
                completed = json.loads(checkpoint.read_text())["completed_folds"] if checkpoint.exists() else []
                if rc == 0 and set(completed) == set(range(10)) and (run / "results.json").exists():
                    update("complete", completed_folds=completed)
                    return
                update("retrying", return_code=rc, completed_folds=completed)
                if attempt < 5:
                    time.sleep(30)
            raise RuntimeError("Training failed after five attempts; see pipeline.log")
    except Exception as exc:
        update("failed", error=str(exc))
        raise
    finally:
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
