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

    # Convert once and cache both uint8 and float32 versions to avoid redundant conversions
    rgb_uint8 = rgb
    rgb_float = rgb.astype(np.float32) / 255.0
    gray_uint8 = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    gray_float = gray_uint8.astype(np.float32) / 255.0

    # Compute features - using more efficient numpy operations
    brightness = float(np.mean(gray_float))
    contrast = float(
        np.std(gray_float)
    )  # std is more efficient than manual calculation

    r, g, b = rgb_float[:, :, 0], rgb_float[:, :, 1], rgb_float[:, :, 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    colorfulness = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2) + 0.3 * np.sqrt(
        np.mean(rg) ** 2 + np.mean(yb) ** 2
    )

    sharpness = float(np.var(cv2.Laplacian(gray_uint8, cv2.CV_64F)))

    # Use already converted uint8 for HSV conversion
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
    saturation = float(np.mean(hsv[:, :, 1])) / 255.0

    blurred = cv2.GaussianBlur(gray_float, (5, 5), 1.0)
    resid = gray_float - blurred
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
