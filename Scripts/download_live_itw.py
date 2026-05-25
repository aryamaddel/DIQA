import os
import sys
import ssl
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
# Box shared folder link. Note: downloading directly via script from Box
# shared links usually returns HTML instead of the zip file due to Box's
# anti-bot download protections.
DATASET_URL = "https://utexas.app.box.com/v/ChallengeDB-release"
DATA_DIR = project_root / "Data"
ZIP_FILE = DATA_DIR / "ChallengeDB_release.zip"
EXTRACT_FOLDER = DATA_DIR / "ChallengeDB_release"


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, blocks=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def download_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_FILE.exists():
        print(f"==================================================")
        print(f" ACTION REQUIRED for LIVE In the Wild (ChallengeDB)")
        print(f"==================================================")
        print(f"The dataset is hosted on Box, which blocks automated")
        print(f"script downloads of shared folder links.")
        print(f"\nPlease download the '{ZIP_FILE.name}' manually from:")
        print(f" {DATASET_URL}")
        print(f"\nPlace the downloaded zip file at:")
        print(f" {ZIP_FILE.resolve()}")
        print(f"==================================================\n")

        # Optional: prompt user to pause here while they download
        response = input(
            f"Press Enter once you have downloaded and placed the zip file, or type 'q' to quit: "
        )
        if response.lower() == "q":
            sys.exit("Make sure to download the dataset before extracting.")

        if not ZIP_FILE.exists():
            print(f"\n[Error] {ZIP_FILE.name} is still not found in Data directory.")
            sys.exit(1)
    else:
        print(f"{ZIP_FILE.name} already exists. Skipping download instructions.")


def extract_dataset():
    if not EXTRACT_FOLDER.exists() or not any(EXTRACT_FOLDER.iterdir()):
        print(f"Extracting {ZIP_FILE.name}...")
        try:
            with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
                EXTRACT_FOLDER.mkdir(parents=True, exist_ok=True)
                for member in tqdm(zip_ref.namelist(), desc="Extracting"):
                    zip_ref.extract(member, path=str(EXTRACT_FOLDER))
        except zipfile.BadZipFile:
            print(
                "\n[Error] Failed to extract! The zip file appears to be corrupted or invalid."
            )
            print(
                "If you used a script to download it, it might have downloaded an HTML page instead."
            )
            print("Please manually download the file from Box and try again.")
            sys.exit(1)
    else:
        print(f"{EXTRACT_FOLDER.name} already extracted. Skipping extraction.")


def main():
    download_dataset()
    extract_dataset()
    print("LIVE In the Wild dataset is ready!")


if __name__ == "__main__":
    main()
