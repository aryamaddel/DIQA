import os
import torch
import pyiqa
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# metrics to compute (name -> metric instance)
METRICS = {
    name.upper(): pyiqa.create_metric(name, device=device)
    for name in ("brisque", "piqe", "niqe")
}

# folder with images (same folder name used originally)
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "KaDiD Small")


def iter_image_files(root_dir):
    """Yield image file paths under root_dir."""
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".png"):
                yield os.path.join(root, f)


def score_image(path):
    """Return dict with image basename and computed metric scores.

    pyiqa accepts file paths directly, so we pass the path to each metric.
    """
    row = {"Image": os.path.basename(path)}
    for key, metric in METRICS.items():
        # metric returns a tensor; convert to Python float
        row[key] = float(metric(path).item())
    return row


def main():
    results = [score_image(p) for p in iter_image_files(IMAGES_DIR)]
    df = pd.DataFrame(results)
    print(df)
    print(f"Processed {len(results)} images")


if __name__ == "__main__":
    main()
