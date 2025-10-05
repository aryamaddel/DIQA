import pyiqa
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def compute_all_scores(image_dir, output_csv="iqa_raw_scores.csv"):
    iqa_methods = ["brisque", "niqe", "piqe", "maniqa", "hyperiqa"]

    # Load IQA metrics
    metrics = {
        m: pyiqa.create_metric(m)
        for m in iqa_methods
        if not print(f"Loaded {m}") or True
    }

    # Automatically list all image files in the directory
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob("*.jpg"))

    results = []
    for img_path in tqdm(image_files, desc="Computing IQA scores"):
        try:
            scores = {m: metrics[m](str(img_path)).item() for m in metrics}
            results.append({"image_name": img_path.name, **scores})
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Saved raw scores to {output_csv}")