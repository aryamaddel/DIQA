import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pyiqa

image_folder = r"koniq10k_1024x768"
extensions = (".png", ".jpg", ".jpeg")


def process_image(image_path):
    # Create metrics inside the worker to avoid thread-safety issues
    brisque = pyiqa.create_metric("brisque")
    niqe = pyiqa.create_metric("niqe")
    piqe = pyiqa.create_metric("piqe")
    try:
        return {
            "brisque": brisque(image_path).item(),
            "niqe": niqe(image_path).item(),
            "piqe": piqe(image_path).item(),
        }
    except Exception:
        return None


images = [
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if f.lower().endswith(extensions)
]

if not images:
    print("No images found.")
    raise SystemExit(1)

results = {"brisque": [], "niqe": [], "piqe": []}

# Adjust max_workers as needed
with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
    futures = [executor.submit(process_image, img) for img in images]
    for fut in tqdm(as_completed(futures), total=len(futures), desc="Scoring"):
        scores = fut.result()
        if scores:
            for k, v in scores.items():
                results[k].append(v)

for name, values in results.items():
    if values:
        print(
            f"{name.upper()} - Min: {min(values):.4f}, Max: {max(values):.4f} (n={len(values)})"
        )
    else:
        print(f"{name.upper()} - No valid scores.")
