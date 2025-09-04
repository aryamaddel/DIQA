from pathlib import Path
import pyiqa

for p in Path("KaDiD Small").glob("*.png"):
    print(f"{p.name}: {pyiqa.create_metric('piqe')(str(p)).item():.3f}")
