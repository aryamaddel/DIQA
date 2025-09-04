import os
import torch
import pyiqa
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METRICS = {
    name.upper(): pyiqa.create_metric(name, device=device)
    for name in ("brisque", "piqe", "niqe")
}
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "KaDiD Small")

# Inline logic: gather image files
image_files = []
for root, _, files in os.walk(IMAGES_DIR):
    for f in files:
        if f.lower().endswith(".png"):
            image_files.append(os.path.join(root, f))

# Inline logic: score images
results = []
for path in image_files:
    row = {"Image": os.path.basename(path)}
    for key, metric in METRICS.items():
        row[key] = float(metric(path).item())
    results.append(row)

df = pd.DataFrame(results)
print(df)
print(f"Processed {len(results)} images")
