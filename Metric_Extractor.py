from pathlib import Path

import pyiqa
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGES_DIR = Path("KaDiD Small")
metric_names = ("brisque", "piqe", "niqe")

METRICS = {
    name.upper(): pyiqa.create_metric(name, device=device) for name in metric_names
}

images = IMAGES_DIR.glob("*.png")

if not images:
    print(f"No PNG images found in '{IMAGES_DIR}'")
else:
    results = []
    for p in images:
        scores = {k: float(m(str(p)).item()) for k, m in METRICS.items()}
        results.append({"Image": p.name, **scores})

    header = ["Image"] + list(METRICS.keys())
    print("\t".join(header))

    for res in results:
        row = [str(res[h]) for h in header]
        print("\t".join(row))

    print(f"Processed {len(results)} images")
