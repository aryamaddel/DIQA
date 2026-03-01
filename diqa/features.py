import numpy as np, cv2, pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Union, Dict, Any

FEATURE_NAMES = [
    "brightness",
    "contrast",
    "colorfulness",
    "sharpness",
    "saturation",
    "texture_noise",
]


def extract_features(image_path: Union[str, Path]) -> List[float]:
    """
    Extracts key conceptual features from an image.

    Args:
        image_path: Path to the image file.

    Returns:
        A list of extracted features representing brightness, contrast, colorfulness,
        sharpness, saturation, and texture_noise.
    """
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(
            f"Could not load image at {image_path}. Please check the path and file integrity."
        )

    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_float = image_gray.astype(float) / 255.0
    channel_r, channel_g, channel_b = cv2.split(
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(float) / 255.0
    )
    red_green, yellow_blue = (
        channel_r - channel_g,
        0.5 * (channel_r + channel_g) - channel_b,
    )
    residual = gray_float - cv2.GaussianBlur(gray_float, (5, 5), 1)

    return [
        gray_float.mean(),
        gray_float.std(),
        np.sqrt(np.var(red_green) + np.var(yellow_blue))
        + 0.3 * np.sqrt(np.mean(red_green) ** 2 + np.mean(yellow_blue) ** 2),
        np.var(cv2.Laplacian(image_gray, cv2.CV_64F)),
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[:, :, 1].mean() / 255.0,
        np.median(np.abs(residual - np.median(residual))) * 1.4826,
    ]


def build_features(
    image_source: Union[str, Path, List[Union[str, Path]]], return_df: bool = True
) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Extracts features for a single image, a list of images, or a directory of images.

    Args:
        image_source: A single file path, list of file paths, or directory path.
        return_df: If True, returns a pandas DataFrame; if False, returns a list of dictionaries.

    Returns:
        Extracted features as a DataFrame or list of dicts.
    """
    source = Path(image_source) if not isinstance(image_source, list) else None
    if source and source.is_dir():
        paths = list(source.glob("*.*"))
    elif source:
        paths = [source]
    else:
        paths = [Path(p) for p in image_source]

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = [p for p in paths if p.suffix.lower() in valid_exts]
    rows = [
        {
            "image_name": p.name,
            "image_path": str(p),
            **dict(zip(FEATURE_NAMES, extract_features(p))),
        }
        for p in tqdm(paths, "Features")
    ]
    df = pd.DataFrame(rows)
    return df if return_df else df.to_dict("records")
