import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pyiqa
import threading

image_folder = "koniq10k_1024x768"
extensions = (".png", ".jpg", ".jpeg")
metrics = ["brisque", "niqe", "piqe"]

images = [
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if f.lower().endswith(extensions)
]
if not images:
    print("No images found.")
    exit(1)

results = {m: [] for m in metrics}
min_max = {
    m: {"min": float("inf"), "max": float("-inf"), "min_img": None, "max_img": None}
    for m in metrics
}
thread_local = threading.local()


def score_image(img_path):
    if not hasattr(thread_local, "metrics"):
        thread_local.metrics = {m: pyiqa.create_metric(m) for m in metrics}
    try:
        scores = {m: thread_local.metrics[m](img_path).item() for m in metrics}
        return img_path, scores
    except Exception:
        return img_path, None


processed_count = 0
with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    for img_path, scores in tqdm(
        executor.map(score_image, images), total=len(images), desc="Scoring"
    ):
        if scores:
            for m in metrics:
                v = scores[m]
                results[m].append(v)
                if v < min_max[m]["min"]:
                    min_max[m]["min"] = v
                    min_max[m]["min_img"] = os.path.basename(img_path)
                if v > min_max[m]["max"]:
                    min_max[m]["max"] = v
                    min_max[m]["max_img"] = os.path.basename(img_path)


print(f"\nFinal results after {processed_count} images:")
for m in metrics:
    print(
        f"{m.upper()} - Min: {min_max[m]['min']:.4f} ({min_max[m]['min_img']}), "
        f"Max: {min_max[m]['max']:.4f} ({min_max[m]['max_img']}) (n={len(results[m])})"
        if results[m]
        else f"{m.upper()} - No valid scores."
    )
