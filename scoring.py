import pyiqa
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def compute_all_scores(image_dir, output_csv="Data/iqa_raw_scores.csv"):
    """Compute IQA scores for all images in a directory."""
    iqa_methods = ["brisque", "niqe", "piqe", "maniqa", "hyperiqa"]
    metrics = {method: pyiqa.create_metric(method) for method in iqa_methods}
    image_files = list(Path(image_dir).glob("*.jpg")) + list(
        Path(image_dir).glob("*.bmp")
    )
    results = []

    for img_path in tqdm(image_files, desc="Computing IQA scores"):
        try:
            scores = {
                method: metrics[method](str(img_path)).item() for method in iqa_methods
            }
            results.append({"image_name": img_path.name, **scores})
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Saved IQA scores to {output_csv}")
