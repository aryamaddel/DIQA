from pathlib import Path
import pyiqa

metric = pyiqa.create_metric("brisque")
for p in Path("KaDiD Small").glob("*.png"):
    print(f"{p.name}: {float(metric(str(p)).item()):.4f}")
