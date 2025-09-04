import os

import pandas as pd
import pyiqa
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METRICS = {
    name.upper(): pyiqa.create_metric(name, device=device)
    for name in ("brisque", "piqe", "niqe")
}
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "KaDiD Small")

image_files = []
for root, _, files in os.walk(IMAGES_DIR):
    for f in files:
        if f.lower().endswith(".png"):
            image_files.append(os.path.join(root, f))

results = []
for path in image_files:
    row = {"Image": os.path.basename(path)}
    for key, metric in METRICS.items():
        row[key] = float(metric(path).item())
    results.append(row)

df = pd.DataFrame(results)
print(df)
print(f"Processed {len(results)} images")
