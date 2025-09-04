from pathlib import Path

import pandas as pd
import pyiqa
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGES_DIR = Path(__file__).with_name("KaDiD Small")
metric_names = ("brisque", "piqe", "niqe")
METRICS = {name.upper(): pyiqa.create_metric(name, device=device) for name in metric_names}

images = sorted(IMAGES_DIR.rglob("*.png"))

results = [
    {"Image": p.name, **{k: float(m(str(p)).item()) for k, m in METRICS.items()}}
    for p in images
]

df = pd.DataFrame(results)
print(df)
print(f"Processed {len(results)} images")
