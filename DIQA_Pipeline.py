# %% Imports
import os, json
from pathlib import Path
import joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy.io import loadmat
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from tqdm import tqdm

from features import build_features, extract_features, feature_names
from router import predict
from scoring import compute_all_scores

# %% Extract Features
image_dir = "koniq10k_512x384"
print("Extracting features:", feature_names)
df_features = build_features(image_dir, out_csv="features.csv")
print(f"Extracted {df_features.shape[0]} images.\n{df_features.head()}")

# %% Visualize Features
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for i, f in enumerate(feature_names):
    ax = axes.ravel()[i]
    ax.hist(df_features[f], bins=30, alpha=0.7, edgecolor="black")
    ax.set(title=f.replace("_", " ").title(), xlabel="Value", ylabel="Freq")
plt.tight_layout()
plt.show()
print(df_features[feature_names].describe())

# %% Compute Scores
compute_all_scores(image_dir)
compute_all_scores(r"LIVE In the Wild\Images", output_csv="live_scores.csv")

# %% Build MOS Mapping & Router Dataset
df = (
    pd.read_csv("iqa_raw_scores.csv")
    .merge(
        pd.read_csv("koniq10k_scores_and_distributions.csv")[["image_name", "MOS"]],
        on="image_name",
    )
    .merge(pd.read_csv("features.csv"), left_on="image_name", right_on="image_path")
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
df.to_csv("router_training_data.csv", index=False)
print(df["best_method"].value_counts())

# %% Train Router
df = pd.read_csv("router_training_data.csv")
X, y = df[feature_names].values, df["best_method_label"].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

router = XGBClassifier(n_estimators=100, random_state=42)
router.fit(X_scaled, y, verbose=50)
y_pred = router.predict(X_scaled)

print(f"Train Acc: {(y_pred==y).mean():.4f}")
sns.heatmap(
    confusion_matrix(y, y_pred),
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

# %% Test on LIVE In-the-Wild
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

gt = df_live.loc[valid, "MOS"].values
srocc, plcc = spearmanr(preds, gt)[0], pearsonr(preds, gt)[0]
rmse, mae = np.sqrt(mean_squared_error(gt, preds)), mean_absolute_error(gt, preds)

print(
    f"\nLIVE RESULTS\nSROCC={srocc:.4f}, PLCC={plcc:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}"
)
print(f"Avg Time: {np.mean(times):.2f}ms")

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
).to_csv("live_test_results.csv", index=False)


# %% Quick Assessment
def assess_image(path):
    r = predict(path)
    print(
        f"\n{os.path.basename(path)} → MOS={r['MOS_estimate']:.3f}/5 | {r['selected_method']} ({r['confidence']:.2f}) | {r['timing']['total_time_ms']:.1f}ms"
    )
    feats = extract_features(path)
    for n, v in zip(feature_names, feats):
        print(f"  {n.title()}: {v:.4f}")
    return r


assess_image("me at night.jpg")
