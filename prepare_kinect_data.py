"""Fetch the original segmented UI-PRMD Kinect files from a pinned public mirror.

The source website currently returns HTTP 403. Each mirrored raw file is
verified against Git's blob SHA-1; filenames preserve actual subject IDs.
"""
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import quote
import requests

REPOSITORY = "jkaethee/UI-PRMD_Exercise_Model"
TREE_SHA = "475c66aba07731adbe1c90a15462fd30be564593"
ROOT = Path(__file__).resolve().parent


def main():
    response = requests.get(f"https://api.github.com/repos/{REPOSITORY}/git/trees/{TREE_SHA}?recursive=1", timeout=60)
    response.raise_for_status()
    tree = response.json()
    if tree.get("truncated"):
        raise RuntimeError("Incomplete source manifest")
    entries = [entry for entry in tree["tree"] if entry["type"] == "blob" and
               entry["path"].startswith(("Segmented Movements/Kinect/", "Incorrect Segmented Movements/Kinect/"))]
    if len(entries) != 4000:
        raise RuntimeError(f"Expected 4000 raw files, got {len(entries)}")
    destination = ROOT / "data/UI-PRMD/Kinect_Segmented"

    def download(entry):
        path = entry["path"]
        parts = path.split("/")
        target = destination / ("incorrect" if parts[0].startswith("Incorrect") else "correct") / parts[2].lower() / parts[-1]
        target.parent.mkdir(parents=True, exist_ok=True)

        def verified(content):
            return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest() == entry["sha"]

        if target.exists() and verified(target.read_bytes()):
            return
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{TREE_SHA}/" + quote(path)
        for attempt in range(5):
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                if not verified(response.content):
                    raise RuntimeError(f"File hash mismatch: {path}")
                target.write_bytes(response.content)
                return
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for count, _ in enumerate(pool.map(download, entries), 1):
            if count % 200 == 0:
                print(f"Verified {count}/{len(entries)} Kinect files", flush=True)
    (destination / "source.json").write_text(json.dumps({
        "dataset": "UI-PRMD segmented Kinect positions and Euler angles",
        "original_source": "https://webpages.uidaho.edu/ui-prmd/",
        "mirror": f"https://github.com/{REPOSITORY}", "tree_sha": TREE_SHA,
        "n_files": len(entries), "integrity": "Every file verified against its Git blob SHA-1",
    }, indent=2))


if __name__ == "__main__":
    main()
