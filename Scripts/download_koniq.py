import os
import sys
import ssl
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm

# Fix for expired SSL certificate on the vqa.mmsp-kn.de server
ssl._create_default_https_context = ssl._create_unverified_context

project_root = Path(__file__).resolve().parent.parent
DATASET_URL = "http://datasets.vqa.mmsp-kn.de/archives/koniq10k_1024x768.zip"
DATA_DIR = project_root / "Data"
ZIP_FILE = DATA_DIR / "koniq10k_1024x768.zip"
EXTRACT_FOLDER = DATA_DIR / "koniq10k_1024x768_images"


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, blocks=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def download_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_FILE.exists():
        print(f"Downloading dataset from {DATASET_URL}...")
        with TqdmUpTo(
            unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=ZIP_FILE.name
        ) as t:
            urllib.request.urlretrieve(
                DATASET_URL, filename=str(ZIP_FILE), reporthook=t.update_to
            )
    else:
        print(f"{ZIP_FILE.name} already exists. Skipping download.")


def extract_dataset():
    if not EXTRACT_FOLDER.exists():
        print(f"Extracting {ZIP_FILE.name}...")
        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            EXTRACT_FOLDER.mkdir(parents=True, exist_ok=True)
            for member in tqdm(zip_ref.namelist(), desc="Extracting"):
                zip_ref.extract(member, path=str(EXTRACT_FOLDER))
    else:
        print(f"{EXTRACT_FOLDER.name} already extracted. Skipping extraction.")


def main():
    download_dataset()
    extract_dataset()
    print("Download and extraction complete.")


if __name__ == "__main__":
    main()
