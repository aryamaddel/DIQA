# %% cell 1
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.io import loadmat
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error
from xgboost import XGBClassifier
from tqdm import tqdm
import joblib
from features import build_features, feature_names
from scoring import compute_all_scores
from router import predict

# %% cell 2
image_dir = "koniq10k_512x384"
print("Extracting features:", feature_names)
df_features = build_features(image_dir, out_csv="Data/features.csv")
print(f"Extracted {df_features.shape[0]} images.")

# %% cell 3
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for i, f in enumerate(feature_names):
    ax = axes.ravel()[i]
    ax.hist(df_features[f], bins=30, alpha=0.7, edgecolor="black")
    ax.set(title=f.replace("_", " ").title(), xlabel="Value", ylabel="Freq")
plt.tight_layout()
plt.show()
print(df_features[feature_names].describe())

# %% cell 4
compute_all_scores(image_dir)
compute_all_scores(r"LIVE In the Wild\Images", output_csv="Data/live_scores.csv")

# %% cell 5
df = (
    pd.read_csv("Data/iqa_raw_scores.csv")
    .merge(
        pd.read_csv("Data/koniq10k_scores_and_distributions.csv")[
            ["image_name", "MOS"]
        ],
        on="image_name",
    )
    .merge(
        pd.read_csv("Data/features.csv"), left_on="image_name", right_on="image_path"
    )
)
iqa_methods = ["brisque", "niqe", "piqe", "maniqa", "hyperiqa"]

# Fit linear regressions to map each IQA → MOS
coeffs, errors = {}, {}
for m in iqa_methods:
    reg = LinearRegression().fit(df[[m]], df["MOS"])
    coeffs[m] = {"coef": float(reg.coef_[0]), "intercept": float(reg.intercept_)}
    errors[m] = np.abs(reg.predict(df[[m]]) - df["MOS"])
    print(f"{m}: R²={reg.score(df[[m]], df['MOS']):.4f}")
json.dump(coeffs, open("mos_mapping_coefficients.json", "w"), indent=2)

df["best_method"] = pd.DataFrame(errors).idxmin(axis=1)
df["best_method_label"] = df["best_method"].map(
    {m: i for i, m in enumerate(iqa_methods)}
)
df["best_method_error"] = pd.DataFrame(errors).min(axis=1)
df.to_csv("Data/router_training_data.csv", index=False)
print(df["best_method"].value_counts())

# %% cell 6
df = pd.read_csv("Data/router_training_data.csv")
X, y = df[feature_names].values, df["best_method_label"].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

router = XGBClassifier(n_estimators=100, random_state=42)
router.fit(X_scaled, y, verbose=50)
y_pred = router.predict(X_scaled)

print(f"Train Acc: {(y_pred == y).mean():.4f}")
conf_matrix = confusion_matrix(y.astype(int), y_pred.astype(int))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iqa_methods,
    yticklabels=iqa_methods,
)
plt.title("Confusion Matrix")
plt.show()

router.save_model("router_xgb.json")
joblib.dump(scaler, "scaler.pkl")
plt.barh(feature_names, router.feature_importances_)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# %% cell 7
live_dir = r"LIVE In the Wild\Images"
mos = loadmat(r"LIVE In the Wild\Data\AllMOS_release.mat")
imgs = loadmat(r"LIVE In the Wild\Data\AllImages_release.mat")

mos_scores = list(mos.values())[3].flatten()
img_names = [str(i[0]) for i in list(imgs.values())[3].flatten()]

available_images = {
    f.name
    for f in Path(live_dir).iterdir()
    if f.suffix.lower() in [".bmp", ".jpg", ".jpeg"]
}

df_live = pd.DataFrame({"image_name": img_names, "MOS": (mos_scores / 100.0) * 5.0})
df_live = df_live[df_live["image_name"].isin(available_images)].reset_index(drop=True)

# Predict on LIVE In-the-Wild
preds, confs, methods, times, valid = [], [], [], [], []
for idx, row in tqdm(df_live.iterrows(), total=len(df_live), desc="Testing"):
    try:
        r = predict(f"{live_dir}/{row.image_name}")
        preds.append(r["MOS_estimate"])
        confs.append(r["confidence"])
        methods.append(r["selected_method"])
        times.append(r["timing"]["total_time_ms"])
        valid.append(idx)
    except Exception as e:
        print("Error:", row.image_name, e)

# Ensure ground truth and predictions are numpy arrays
gt = df_live.loc[valid, "MOS"].to_numpy()
preds = np.array(preds)

# Compute metrics
srocc, plcc = spearmanr(preds, gt)[0], pearsonr(preds, gt)[0]
rmse, mae = np.sqrt(mean_squared_error(gt, preds)), mean_absolute_error(gt, preds)

print(
    f"\nLIVE RESULTS\nSROCC={srocc:.4f}, PLCC={plcc:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}"
)
print(f"Avg Time: {np.mean(times):.2f}ms")

# Scatter plot with numpy arrays
plt.scatter(gt, preds, alpha=0.6, edgecolor="k")
lims = [gt.min(), gt.max()]
plt.plot(lims, lims, "r--")
plt.xlabel("GT MOS")
plt.ylabel("Pred MOS")
plt.title(f"KonIQ→LIVE | SROCC={srocc:.3f}, PLCC={plcc:.3f}")
plt.tight_layout()
plt.show()

pd.DataFrame(
    {
        "image_name": df_live.loc[valid, "image_name"].values,
        "ground_truth": gt,
        "predicted": preds,
        "method": methods,
        "confidence": confs,
    }
).to_csv("Data/live_test_results.csv", index=False)


# %% cell 8
def assess_image(path):
    result = predict(path)
    print(
        f"\n{os.path.basename(path)} → MOS={result['MOS_estimate']:.3f}/5 | {result['selected_method']} ({result['confidence']:.2f}) | {result['timing']['total_time_ms']:.1f}ms"
    )
    return result


assess_image("me at night.jpg")
