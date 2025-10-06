import numpy as np, cv2, os, pandas as pd
from skimage.measure import shannon_entropy
from pathlib import Path
from tqdm import tqdm

TARGET_SIZE = (384, 512)
feature_names = [
    "mean_luminance",
    "std_luminance",
    "entropy_luminance",
    "colorfulness",
    "mean_saturation",
    "laplacian_variance",
    "tenengrad",
    "canny_edge_ratio",
    "noise_std_estimate",
    "rms_contrast",
    "percentile_contrast",
]


def extract_features(image_path):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"Unable to read: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != TARGET_SIZE:
        rgb = cv2.resize(
            rgb, (TARGET_SIZE[1], TARGET_SIZE[0]), interpolation=cv2.INTER_AREA
        )
    rgb = rgb.astype(np.float32) / 255.0
    gray = (
        cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(
            np.float32
        )
        / 255.0
    )
    flat = gray.ravel()
    feats = [np.mean(flat), np.std(flat), shannon_entropy(gray)]
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    colorfulness = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2) + 0.3 * np.sqrt(
        np.mean(rg) ** 2 + np.mean(yb) ** 2
    )
    hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    feats += [colorfulness, np.mean(hsv[:, :, 1].astype(float)) / 255.0]
    gray_u8 = (gray * 255).astype(np.uint8)
    lap = np.asarray(cv2.Laplacian(gray_u8, cv2.CV_64F), dtype=np.float32)
    gx = np.asarray(cv2.Sobel(gray_u8, cv2.CV_64F, 1, 0, ksize=3), dtype=np.float32)
    gy = np.asarray(cv2.Sobel(gray_u8, cv2.CV_64F, 0, 1, ksize=3), dtype=np.float32)
    feats += [float(np.var(lap)), float(np.mean(np.sqrt(gx**2 + gy**2)))]
    edges = cv2.Canny(gray_u8, 50, 150)
    feats.append(float(np.count_nonzero(edges) / edges.size))
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    resid = gray - blurred
    resid = np.asarray(resid, dtype=np.float32)
    med = float(np.median(resid))
    feats.append(float(np.median(np.abs(resid - med)) * 1.4826))
    mean_gray = np.mean(gray)
    feats += [
        float(np.sqrt(np.mean((gray - mean_gray) ** 2))),
        float(np.percentile(gray, 90) - np.percentile(gray, 10)),
    ]
    return np.array(feats, dtype=np.float32)


def build_features(image_dir, out_csv="features.csv"):
    imgs = [
        p
        for p in Path(image_dir).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]
    rows = []
    for p in tqdm(imgs, desc="Features"):
        try:
            rows.append([*extract_features(p), p.name])
        except Exception as e:
            print(f"skip {p.name}: {e}")
    df = pd.DataFrame(rows, columns=feature_names + ["image_path"])
    df.to_csv(out_csv, index=False)
    return df


if __name__ == "__main__":
    d = os.environ.get("IQA_IMAGE_DIR", "koniq10k_512x384")
    print(f"Writing features for {d}")
    build_features(d)
    print("done")
