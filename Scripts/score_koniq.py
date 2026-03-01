import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add the parent project directory to sys.path so we can import from `diqa`
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from diqa.training import get_scores

DATA_DIR = project_root / "Data"
EXTRACT_FOLDER = DATA_DIR / "koniq10k_1024x768_images"
CSV_OUTPUT = DATA_DIR / "koniq10k_scores.csv"


def main():
    if not EXTRACT_FOLDER.exists():
        print(
            f"Error: {EXTRACT_FOLDER.name} not found. Please run download_koniq.py first."
        )
        return

    print("Finding images...")
    image_paths = (
        list(EXTRACT_FOLDER.rglob("*.jpg"))
        + list(EXTRACT_FOLDER.rglob("*.png"))
        + list(EXTRACT_FOLDER.rglob("*.jpeg"))
    )

    if not image_paths:
        print(f"No images found in {EXTRACT_FOLDER}...")
        return

    print(f"Found {len(image_paths)} images.")

    print("Scoring images using DIQA...")
    scores_df = get_scores(image_paths)

    print(f"Saving scores to {CSV_OUTPUT}...")
    scores_df.to_csv(CSV_OUTPUT, index=False)
    print("Done!")


if __name__ == "__main__":
    main()
