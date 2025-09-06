import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pyiqa

image_folder = "koniq10k_1024x768"
extensions = (".png", ".jpg", ".jpeg")
images = [
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if f.lower().endswith(extensions)
]
if not images:
    print("No images found.")
    raise SystemExit(1)

results = {k: [] for k in ("brisque", "niqe", "piqe")}
min_max_info = {
    k: {"min": float("inf"), "max": float("-inf"), "min_img": None, "max_img": None}
    for k in results
}
log_interval, processed_count = 500, 0

with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
    futures = {
        executor.submit(
            lambda img: (
                img,
                pyiqa.create_metric("brisque")(img).item(),
                pyiqa.create_metric("niqe")(img).item(),
                pyiqa.create_metric("piqe")(img).item(),
            ),
            img,
        ): img
        for img in images
    }
    for fut in tqdm(as_completed(futures), total=len(futures), desc="Scoring"):
        try:
            img_path, brisque_score, niqe_score, piqe_score = fut.result()
            for k, v in zip(results, (brisque_score, niqe_score, piqe_score)):
                results[k].append(v)
                if v < min_max_info[k]["min"]:
                    min_max_info[k]["min"], min_max_info[k]["min_img"] = (
                        v,
                        os.path.basename(img_path),
                    )
                if v > min_max_info[k]["max"]:
                    min_max_info[k]["max"], min_max_info[k]["max_img"] = (
                        v,
                        os.path.basename(img_path),
                    )
        except Exception:
            continue
        processed_count += 1
        if processed_count % log_interval == 0:
            print(f"\nAfter {processed_count} images:")
            for k in results:
                vals = results[k]
                if vals:
                    print(
                        f"{k.upper()} - Min: {min_max_info[k]['min']:.4f} ({min_max_info[k]['min_img']}), Max: {min_max_info[k]['max']:.4f} ({min_max_info[k]['max_img']}) (n={len(vals)})"
                    )
                else:
                    print(f"{k.upper()} - No valid scores.")

print(f"\nAfter {processed_count} images:")
for k in results:
    vals = results[k]
    if vals:
        print(
            f"{k.upper()} - Min: {min_max_info[k]['min']:.4f} ({min_max_info[k]['min_img']}), Max: {min_max_info[k]['max']:.4f} ({min_max_info[k]['max_img']}) (n={len(vals)})"
        )
    else:
        print(f"{k.upper()} - No valid scores.")

for k, vals in results.items():
    if vals:
        print(
            f"{k.upper()} - Min: {min_max_info[k]['min']:.4f} ({min_max_info[k]['min_img']}), Max: {min_max_info[k]['max']:.4f} ({min_max_info[k]['max_img']}) (n={len(vals)})"
        )
    else:
        print(f"{k.upper()} - No valid scores.")
