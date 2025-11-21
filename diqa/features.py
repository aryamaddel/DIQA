import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

TARGET_SIZE = (384, 512)
FEATURE_NAMES = [
    "brightness",
    "contrast",
    "colorfulness",
    "sharpness",
    "saturation",
    "texture_noise",
]


def extract_features(image_path):
    """Extracts features from a single image."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"Unable to read: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Resize if necessary
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

    # Compute features
    brightness = np.mean(gray)
    contrast = float(np.sqrt(np.mean((gray - np.mean(gray)) ** 2)))

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    colorfulness = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2) + 0.3 * np.sqrt(
        np.mean(rg) ** 2 + np.mean(yb) ** 2
    )

    gray_u8 = (gray * 255).astype(np.uint8)
    sharpness = float(np.var(cv2.Laplacian(gray_u8, cv2.CV_64F)))

    hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    saturation = np.mean(hsv[:, :, 1].astype(float)) / 255.0

    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    resid = gray - blurred
    texture_noise = float(np.median(np.abs(resid - np.median(resid))) * 1.4826)

    return np.array(
        [brightness, contrast, colorfulness, sharpness, saturation, texture_noise],
        dtype=np.float32,
    )


def build_features(image_source, return_df=True):
    """
    Builds features for images.

    Args:
        image_source: Can be a directory path (str/Path) or a list of image paths.
        return_df: If True, returns a pandas DataFrame. Otherwise returns a list of dicts.
    """
    if isinstance(image_source, (str, Path)):
        path = Path(image_source)
        if path.is_dir():
            image_paths = (
                list(path.glob("*.jpg"))
                + list(path.glob("*.bmp"))
                + list(path.glob("*.png"))
            )
        else:
            image_paths = [path]
    else:
        image_paths = [Path(p) for p in image_source]

    rows = []
    for image_path in tqdm(image_paths, desc="Extracting Features"):
        try:
            features = extract_features(image_path)
            row = dict(zip(FEATURE_NAMES, features))
            row["image_path"] = str(image_path)
            row["image_name"] = image_path.name
            rows.append(row)
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")

    if return_df:
        return pd.DataFrame(rows)
    return rows
