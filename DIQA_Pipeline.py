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

# Define constants
IMAGE_DIR = "koniq10k_512x384"
FEATURES_CSV = "Data/features.csv"
KONIQ_SCORES_CSV = "Data/koniq10k_scores_and_distributions.csv"
IQA_RAW_SCORES_CSV = "Data/iqa_raw_scores.csv"
ROUTER_TRAINING_CSV = "Data/router_training_data.csv"
MOS_MAPPING_COEFFS_JSON = "mos_mapping_coefficients.json"
ROUTER_MODEL_JSON = "router_xgb.json"
SCALER_PKL = "scaler.pkl"
LIVE_DIR = r"LIVE In the Wild\Images"
LIVE_MOS_MAT = r"LIVE In the Wild\Data\AllMOS_release.mat"
LIVE_IMAGES_MAT = r"LIVE In the Wild\Data\AllImages_release.mat"
LIVE_TEST_RESULTS_CSV = "Data/live_test_results.csv"

# IQA methods
IQA_METHODS = ["brisque", "niqe", "piqe", "maniqa", "hyperiqa"]

# %% cell 2
print("Extracting features:", feature_names)
df_features = build_features(IMAGE_DIR, out_csv=FEATURES_CSV)
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
compute_all_scores(IMAGE_DIR)
compute_all_scores(r"LIVE In the Wild\Images", output_csv="Data/live_scores.csv")

# %% cell 5
df = (
    pd.read_csv(IQA_RAW_SCORES_CSV)
    .merge(
        pd.read_csv(KONIQ_SCORES_CSV)[["image_name", "MOS"]],
        on="image_name",
    )
    .merge(pd.read_csv(FEATURES_CSV), left_on="image_name", right_on="image_path")
)

# Fit linear regressions to map each IQA → MOS
coeffs, errors = {}, {}
for m in IQA_METHODS:
    reg = LinearRegression().fit(df[[m]], df["MOS"])
    coeffs[m] = {"coef": float(reg.coef_[0]), "intercept": float(reg.intercept_)}
    errors[m] = np.abs(reg.predict(df[[m]]) - df["MOS"])
    print(f"{m}: R²={reg.score(df[[m]], df['MOS']):.4f}")
json.dump(coeffs, open(MOS_MAPPING_COEFFS_JSON, "w"), indent=2)

df["best_method"] = pd.DataFrame(errors).idxmin(axis=1)
df["best_method_label"] = df["best_method"].map(
    {m: i for i, m in enumerate(IQA_METHODS)}
)
df["best_method_error"] = pd.DataFrame(errors).min(axis=1)
df.to_csv(ROUTER_TRAINING_CSV, index=False)
print(df["best_method"].value_counts())

# %% cell 6
df = pd.read_csv(ROUTER_TRAINING_CSV)
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
    xticklabels=IQA_METHODS,
    yticklabels=IQA_METHODS,
)
plt.title("Confusion Matrix")
plt.show()

router.save_model(ROUTER_MODEL_JSON)
joblib.dump(scaler, SCALER_PKL)
plt.barh(feature_names, router.feature_importances_)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# %% cell 7
mos = loadmat(LIVE_MOS_MAT)
imgs = loadmat(LIVE_IMAGES_MAT)

mos_scores = list(mos.values())[3].flatten()
img_names = [str(i[0]) for i in list(imgs.values())[3].flatten()]

available_images = {
    f.name
    for f in Path(LIVE_DIR).iterdir()
    if f.suffix.lower() in [".bmp", ".jpg", ".jpeg"]
}

df_live = pd.DataFrame({"image_name": img_names, "MOS": (mos_scores / 100.0) * 5.0})
df_live = df_live[df_live["image_name"].isin(available_images)].reset_index(drop=True)

# Predict on LIVE In-the-Wild
preds, confs, methods, times, valid = [], [], [], [], []
for idx, row in tqdm(df_live.iterrows(), total=len(df_live), desc="Testing"):
    try:
        r = predict(f"{LIVE_DIR}/{row.image_name}")
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
).to_csv(LIVE_TEST_RESULTS_CSV, index=False)


# %% cell 8
path = "koniq10k_512x384/826373.jpg"
result = predict(path)
print(
    f"\n{os.path.basename(path)} → MOS={result['MOS_estimate']:.3f}/5 | "
    f"{result['selected_method']} ({result['confidence']:.2f}) | "
    f"{result['timing']['total_time_ms']:.1f}ms"
)
