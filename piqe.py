from pathlib import Path
import pyiqa

metric = pyiqa.create_metric("piqe")
for p in sorted(Path("KaDiD Small").iterdir()):
    print(f"{p.name}: {metric(str(p)).item():.3f}")