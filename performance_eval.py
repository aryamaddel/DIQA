import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
from pathlib import Path
import pyiqa

# Load LIVE In-the-Wild MOS scores from .mat files
print("Loading LIVE In-the-Wild dataset...")
mos_data = loadmat(r"LIVE In the Wild\Data\AllMOS_release.mat")
images_data = loadmat(r"LIVE In the Wild\Data\AllImages_release.mat")

# Extract image names and MOS scores
if "AllMOS_release" in mos_data:
    mos_scores = mos_data["AllMOS_release"].flatten()
else:
    mos_scores = list(mos_data.values())[3].flatten()

if "AllImages_release" in images_data:
    image_names = [str(img[0]) for img in images_data["AllImages_release"].flatten()]
else:
    image_names = [str(img[0]) for img in list(images_data.values())[3].flatten()]

# LIVE MOS is on 0-100 scale, normalize to 0-5 to match typical IQA output ranges
mos_scores_normalized = (mos_scores / 100.0) * 5.0

print(f"Original LIVE MOS range: [{mos_scores.min():.2f}, {mos_scores.max():.2f}]")
print(
    f"Normalized LIVE MOS range: [{mos_scores_normalized.min():.2f}, {mos_scores_normalized.max():.2f}]"
)

# Create DataFrame
live_mos_df = pd.DataFrame(
    {
        "image_name": image_names,
        "MOS_original": mos_scores,
        "MOS": mos_scores_normalized,
    }
)

# Get available images
live_image_dir = r"LIVE In the Wild\Images"
available_images = [
    f.name
    for f in Path(live_image_dir).iterdir()
    if f.suffix.lower() in [".bmp", ".jpg", ".jpeg"] and f.is_file()
]

# Filter to only available images
live_mos_df = live_mos_df[live_mos_df["image_name"].isin(available_images)]
print(f"\nEvaluating on {len(live_mos_df)} LIVE In-the-Wild images")

# Initialize IQA metrics
iqa_methods = {
    "brisque": pyiqa.create_metric("brisque"),
    "niqe": pyiqa.create_metric("niqe"),
    "piqe": pyiqa.create_metric("piqe"),
    "maniqa": pyiqa.create_metric("maniqa"),
    "hyperiqa": pyiqa.create_metric("hyperiqa"),
}

# Collect predictions for each method
results = {method: [] for method in iqa_methods.keys()}
valid_indices = []
ground_truth_list = []

print("\nComputing IQA scores...")
for idx, row in tqdm(live_mos_df.iterrows(), total=len(live_mos_df)):
    image_path = f"{live_image_dir}/{row['image_name']}"

    try:
        # Compute all IQA scores for this image
        all_scores_valid = True
        temp_scores = {}

        for method_name, metric in iqa_methods.items():
            try:
                score = metric(image_path).item()
                temp_scores[method_name] = score
            except Exception as e:
                print(f"Error with {method_name} on {row['image_name']}: {e}")
                all_scores_valid = False
                break

        if all_scores_valid:
            for method_name, score in temp_scores.items():
                results[method_name].append(score)
            valid_indices.append(idx)
            ground_truth_list.append(row["MOS"])

    except Exception as e:
        print(f"Error processing {row['image_name']}: {e}")

ground_truth = np.array(ground_truth_list)
print(f"\nSuccessfully processed {len(valid_indices)} images")

# Compute metrics for each method
print(f"\n{'=' * 80}")
print(f"LIVE IN-THE-WILD EVALUATION - INDIVIDUAL IQA METHODS")
print(f"{'=' * 80}")
print(f"Dataset: {len(valid_indices)} images")
print(f"Ground Truth MOS range: [{ground_truth.min():.2f}, {ground_truth.max():.2f}]")
print(f"{'=' * 80}\n")

evaluation_results = []

for method_name in iqa_methods.keys():
    predictions = np.array(results[method_name])

    # Compute correlation metrics (these are scale-invariant)
    srocc = spearmanr(predictions, ground_truth)[0]
    plcc = pearsonr(predictions, ground_truth)[0]

    # For RMSE/MAE, we need to map predictions to MOS scale
    # Use linear regression to map IQA scores to MOS
    from sklearn.linear_model import LinearRegression

    reg = LinearRegression()
    reg.fit(predictions.reshape(-1, 1), ground_truth)
    predictions_mapped = reg.predict(predictions.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(ground_truth, predictions_mapped))
    mae = mean_absolute_error(ground_truth, predictions_mapped)

    evaluation_results.append(
        {
            "Method": method_name.upper(),
            "SROCC": srocc,
            "PLCC": plcc,
            "RMSE": rmse,
            "MAE": mae,
            "Prediction_Range": f"[{predictions.min():.2f}, {predictions.max():.2f}]",
        }
    )

    print(
        f"{method_name.upper():12s} | SROCC: {srocc:.4f} | PLCC: {plcc:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}"
    )

print(f"\n{'=' * 80}")

# Save results to CSV
results_df = pd.DataFrame(evaluation_results)
results_df.to_csv("Data/live_individual_methods_evaluation.csv", index=False)

# Also save detailed predictions
detailed_results = pd.DataFrame(
    {
        "image_name": live_mos_df.loc[valid_indices, "image_name"].values,
        "ground_truth": ground_truth,
    }
)

for method_name in iqa_methods.keys():
    detailed_results[f"{method_name}_score"] = results[method_name]

detailed_results.to_csv("Data/live_individual_methods_predictions.csv", index=False)

print(f"\n✓ Evaluation complete!")
print(f"Summary saved to: Data/live_individual_methods_evaluation.csv")
print(f"Detailed predictions saved to: Data/live_individual_methods_predictions.csv")
