# %% Imports
import os
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier

from features import build_features, extract_features, feature_names
from router import predict
from scoring import compute_all_scores

# %% Extract Features
image_dir = "koniq10k_512x384"
print("Extracting simplified features:", feature_names)
df_features = build_features(image_dir, out_csv="features.csv")
print(f"Features extracted: {df_features.shape}\n{df_features.head()}")

# %% Visualize Features
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, feature in enumerate(feature_names):
    ax = axes.ravel()[i]
    ax.hist(df_features[feature], bins=30, alpha=0.7, edgecolor="black")
    ax.set_title(f'{feature.replace("_", " ").title()}')
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print("\nFeature Summary:\n", df_features[feature_names].describe())

# %% Compute Scores
compute_all_scores("koniq10k_512x384")
compute_all_scores("LIVE In the Wild\\Images", output_csv="live_scores.csv")

# %% Build MOS Mapping & Router Dataset
df_scores = pd.read_csv("iqa_raw_scores.csv")
df_mos = pd.read_csv("koniq10k_scores_and_distributions.csv")
df_features = pd.read_csv("features.csv")

if "image_name" not in df_scores.columns:
    df_scores.columns = ["image_name"] + list(df_scores.columns[1:])

df = df_scores.merge(df_mos[["image_name", "MOS"]], on="image_name").merge(
    df_features, left_on="image_name", right_on="image_path"
)

iqa_methods = ["brisque", "niqe", "piqe", "maniqa", "hyperiqa"]
regression_coefficients = {}

for method in iqa_methods:
    reg = LinearRegression()
    reg.fit(df[[method]].values, df["MOS"].values)
    regression_coefficients[method] = {
        "coef": float(reg.coef_[0]),
        "intercept": float(reg.intercept_),
    }
    print(f"{method}: R²={reg.score(df[[method]].values, df['MOS'].values):.4f}")

with open("mos_mapping_coefficients.json", "w") as f:
    json.dump(regression_coefficients, f, indent=2)

errors = pd.DataFrame(
    {
        method: np.abs(
            LinearRegression()
            .fit(df[[method]].values, df["MOS"].values)
            .predict(df[[method]].values)
            - df["MOS"].values
        )
        for method in iqa_methods
    }
)

df["best_method"] = errors.idxmin(axis=1)
df["best_method_label"] = df["best_method"].map(
    {m: i for i, m in enumerate(iqa_methods)}
)
df["best_method_error"] = errors.min(axis=1)

print(f"\n{df['best_method'].value_counts()}")
df.to_csv("router_training_data.csv", index=False)
print(f"Dataset: {df.shape}")

# %% Train Router
df = pd.read_csv("router_training_data.csv")
X = df[feature_names].values
y = df["best_method_label"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

router = XGBClassifier(n_estimators=100, random_state=42)
router.fit(X_scaled, y, verbose=50)

y_pred = router.predict(X_scaled)
print(f"Training Accuracy: {np.mean(y_pred == y):.4f}")

cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iqa_methods,
    yticklabels=iqa_methods,
)
plt.title("Confusion Matrix - Simplified Features")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

router.save_model("router_xgb.json")
joblib.dump(scaler, "scaler.pkl")
print("\n✓ Router saved!")

plt.figure(figsize=(10, 6))
plt.barh(feature_names, router.feature_importances_)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# %% Test on LIVE In-the-Wild
print("Loading LIVE In-the-Wild dataset...")
mos_data = loadmat(r"LIVE In the Wild\Data\AllMOS_release.mat")
images_data = loadmat(r"LIVE In the Wild\Data\AllImages_release.mat")

mos_scores = list(mos_data.values())[3].flatten()
image_names = [str(img[0]) for img in list(images_data.values())[3].flatten()]
mos_scores_normalized = (mos_scores / 100.0) * 5.0

live_mos_df = pd.DataFrame(
    {
        "image_name": image_names,
        "MOS_original": mos_scores,
        "MOS": mos_scores_normalized,
    }
)

live_image_dir = r"LIVE In the Wild\Images"
available_images = [
    f.name
    for f in Path(live_image_dir).iterdir()
    if f.suffix.lower() in [".bmp", ".jpg", ".jpeg"]
]
live_mos_df = live_mos_df[live_mos_df["image_name"].isin(available_images)]

predictions, confidences, selected_methods, times, valid_indices = [], [], [], [], []
for idx, row in tqdm(live_mos_df.iterrows(), total=len(live_mos_df), desc="Testing"):
    try:
        result = predict(f"{live_image_dir}/{row['image_name']}")
        predictions.append(result["MOS_estimate"])
        confidences.append(result["confidence"])
        selected_methods.append(result["selected_method"])
        times.append(result["timing"]["total_time_ms"])
        valid_indices.append(idx)
    except Exception as e:
        print(f"Error on {row['image_name']}: {e}")

ground_truth = live_mos_df.loc[valid_indices, "MOS"].values
srocc, plcc = (
    spearmanr(predictions, ground_truth)[0],
    pearsonr(predictions, ground_truth)[0],
)
rmse, mae = np.sqrt(mean_squared_error(ground_truth, predictions)), mean_absolute_error(
    ground_truth, predictions
)

print(f"\n{'='*60}\nLIVE IN-THE-WILD RESULTS\n{'='*60}")
print(f"SROCC: {srocc:.4f} | PLCC: {plcc:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
print(f"Avg Time: {np.mean(times):.2f}ms\n{'='*60}")

plt.figure(figsize=(10, 10))
plt.scatter(
    ground_truth, predictions, alpha=0.5, s=30, edgecolors="black", linewidths=0.5
)
plt.plot(
    [ground_truth.min(), ground_truth.max()],
    [ground_truth.min(), ground_truth.max()],
    "r--",
    linewidth=2,
)
plt.xlabel("Ground Truth MOS")
plt.ylabel("Predicted MOS")
plt.title(f"KonIQ→LIVE | SROCC={srocc:.4f}, PLCC={plcc:.4f}, RMSE={rmse:.4f}")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

pd.DataFrame(
    {
        "image_name": live_mos_df.loc[valid_indices, "image_name"].values,
        "ground_truth": ground_truth,
        "predicted": predictions,
        "method": selected_methods,
        "confidence": confidences,
    }
).to_csv("live_test_results.csv", index=False)


# %% Quick Assessment Function
def assess_image(image_path):
    """Quick image quality assessment."""
    result = predict(image_path)
    print(f"Image: {os.path.basename(image_path)}")
    print(f"MOS: {result['MOS_estimate']:.3f}/5.0")
    print(
        f"Method: {result['selected_method']} (confidence: {result['confidence']:.3f})"
    )
    print(f"Time: {result['timing']['total_time_ms']:.2f}ms")
    features = extract_features(image_path)
    print("\nFeatures:")
    for name, value in zip(feature_names, features):
        print(f"  {name.replace('_', ' ').title()}: {value:.4f}")
    return result


assess_image("me at night.jpg")
