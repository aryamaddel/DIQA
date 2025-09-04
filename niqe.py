from pathlib import Path
import pyiqa

metric = pyiqa.create_metric("niqe")
for p in sorted(Path("KaDiD Small").iterdir()):
    print(f"{p.name}: {metric(str(p)).item():.3f}")