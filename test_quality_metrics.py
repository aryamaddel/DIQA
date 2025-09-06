import os
import random
import pandas as pd
import pyiqa
from tqdm import tqdm

image_folder = "koniq10k_512x384"
indicators_csv = "koniq10k_indicators.csv"
scores_csv = "koniq10k_scores_and_distributions.csv"

indicators_df = pd.read_csv(indicators_csv)
scores_df = pd.read_csv(scores_csv)

brisque_metric = pyiqa.create_metric("brisque")
niqe_metric = pyiqa.create_metric("niqe")
piqe_metric = pyiqa.create_metric("piqe")

results = {"image": [], "brisque": [], "niqe": [], "piqe": [], "actual_score": []}

all_images = [
    img
    for img in os.listdir(image_folder)
    if img.lower().endswith((".png", ".jpg", ".jpeg"))
]
random_images = random.sample(all_images, min(50, len(all_images)))

for image_name in tqdm(random_images):
    image_path = os.path.join(image_folder, image_name)

    try:
        brisque_score = brisque_metric(image_path).item()
        niqe_score = niqe_metric(image_path).item()
        piqe_score = piqe_metric(image_path).item()

    except Exception as e:
        print(f"Error processing {image_name}: {e}")
        continue

    actual_score_row = scores_df[scores_df["image_name"] == image_name]
    actual_score = (
        actual_score_row["MOS"].values[0] if not actual_score_row.empty else None
    )

    results["image"].append(image_name)
    results["brisque"].append(brisque_score)
    results["niqe"].append(niqe_score)
    results["piqe"].append(piqe_score)
    results["actual_score"].append(actual_score)

    # Convert results to a DataFrame and visualize
results_df = pd.DataFrame(results)

print("Quality Assessment Results:")
print(results_df)


