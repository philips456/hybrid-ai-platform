"""
scripts/download_dataset.py
Télécharge le dataset NASA SMAP/MSL depuis Kaggle.

Usage:
    python scripts/download_dataset.py
"""
import os
import shutil
from pathlib import Path

import kagglehub

DATA_RAW = Path("data/raw/nasa_smap_msl")


def download():
    print("Downloading NASA SMAP/MSL dataset...")
    path = kagglehub.dataset_download(
        "patrickfleith/nasa-anomaly-detection-dataset-smap-msl"
    )
    print(f"Downloaded to: {path}")

    # Copier dans data/raw/
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for f in Path(path).rglob("*"):
        if f.is_file():
            dest = DATA_RAW / f.name
            shutil.copy2(f, dest)
            print(f"  Copied: {f.name}")

    print(f"\nDataset ready at: {DATA_RAW}")
    print("Files:")
    for f in sorted(DATA_RAW.iterdir()):
        print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    download()
