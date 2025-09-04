from pathlib import Path
import pyiqa

IMAGES_DIR = Path("KaDiD Small")


metric = pyiqa.create_metric("piqe")
images = IMAGES_DIR.glob("*.png")

for p in images:
    try:
        score = float(metric(str(p)).item())
        print(f"{p.name}: {score:.4f}")
    except Exception as e:
        print(f"{p.name}: ERROR ({e})")
