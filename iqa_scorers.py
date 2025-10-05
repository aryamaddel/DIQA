import pyiqa
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def compute_all_scores(image_dir, output_csv="iqa_raw_scores.csv"):
    iqa_methods = ["brisque", "niqe", "piqe", "maniqa", "hyperiqa"]
    metrics = {m: pyiqa.create_metric(m) for m in iqa_methods}
    image_files = list(Path(image_dir).glob("*.jpg"))
    results = []
    for img_path in tqdm(image_files, desc="Computing IQA scores"):
        try:
            results.append({"image_name": img_path.name, **{m: metrics[m](str(img_path)).item() for m in metrics}})
        except Exception as e:
            print(f"Error: {img_path.name}: {e}")
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")