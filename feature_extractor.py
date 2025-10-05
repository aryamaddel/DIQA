import numpy as np
import cv2
from pathlib import Path
from scipy import stats
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy

TARGET_SIZE = (384, 512)  # KONIQ-10k 512×384 variant (H×W)

feature_names = [
    "mean_luminance",
    "std_luminance",
    "skewness_luminance",
    "kurtosis_luminance",
    "entropy_luminance",
    "median_luminance",
    "colorfulness",
    "mean_saturation",
    "laplacian_variance",
    "tenengrad",
    "canny_edge_ratio",
    "lbp_uniformity",
    "noise_std_estimate",
    "blockiness_energy",
    "rms_contrast",
    "percentile_contrast",
    "low_freq_energy",
    "mid_freq_energy",
    "high_freq_energy",
]


def extract_features(image_path: str | Path) -> np.ndarray:
    """Load image and extract all handcrafted features in one pass."""
    # Load and preprocess
    bgr = cv2.imread(str(Path(image_path)), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Unable to read image: {image_path}")

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

    features = []

    # Global stats (6 features)
    flat = gray.flatten()
    features.extend(
        [
            np.mean(flat),
            np.std(flat),
            stats.skew(flat),
            stats.kurtosis(flat),
            shannon_entropy(gray),
            np.median(flat),
        ]
    )

    # Colorfulness & saturation (2 features)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2) + 0.3 * np.sqrt(
        np.mean(rg) ** 2 + np.mean(yb) ** 2
    )
    hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    mean_saturation = np.mean(hsv[:, :, 1].astype(float)) / 255.0
    features.extend([colorfulness, mean_saturation])

    # Sharpness features (2 features)
    gray_uint8 = (gray * 255).astype(np.uint8)
    laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)
    grad_x = cv2.Sobel(gray_uint8, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_uint8, cv2.CV_64F, 0, 1, ksize=3)
    features.extend(
        [float(np.var(laplacian)), float(np.mean(np.sqrt(grad_x**2 + grad_y**2)))]
    )

    # Edge & texture features (2 features)
    edges = cv2.Canny(gray_uint8, 50, 150)
    edge_ratio = np.sum(edges > 0) / edges.size
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 9), density=True)
    lbp_hist = lbp_hist + 1e-10
    lbp_uniformity = -np.sum(lbp_hist * np.log(lbp_hist))
    features.extend([float(edge_ratio), float(lbp_uniformity)])

    # Noise estimate (1 feature)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    residual = gray - blurred
    noise_std = np.median(np.abs(residual - np.median(residual))) * 1.4826
    features.append(float(noise_std))

    # Blockiness energy (1 feature)
    h, w = gray.shape
    block_variances = []
    neighbor_variances = []
    for i in range(0, h - 16, 8):
        for j in range(0, w - 16, 8):
            if i + 16 < h and j + 8 < w:
                block = gray[i : i + 8, j : j + 8]
                neighbor = gray[i + 8 : i + 16, j : j + 8]
                block_variances.append(np.var(block))
                neighbor_variances.append(np.var(neighbor))
    blockiness = (
        np.abs(np.mean(block_variances) - np.mean(neighbor_variances))
        if block_variances
        else 0.0
    )
    features.append(float(blockiness))

    # Contrast features (2 features)
    mean_gray = np.mean(gray)
    rms_contrast = np.sqrt(np.mean((gray - mean_gray) ** 2))
    percentile_contrast = np.percentile(gray, 90) - np.percentile(gray, 10)
    features.extend([float(rms_contrast), float(percentile_contrast)])

    # Frequency features (3 features)
    center_size = min(64, h // 2, w // 2)
    if center_size >= 8:
        center_h, center_w = h // 2, w // 2
        crop = gray[
            center_h - center_size // 2 : center_h + center_size // 2,
            center_w - center_size // 2 : center_w + center_size // 2,
        ]
        dct = cv2.dct(crop.astype(np.float32))
        dct_abs = np.abs(dct)
        size = dct_abs.shape[0]
        third = max(1, size // 3)
        low_energy = np.mean(dct_abs[:third, :third])
        mid_energy = (
            np.mean(dct_abs[third : 2 * third, third : 2 * third])
            if 2 * third <= size
            else 0.0
        )
        high_energy = (
            np.mean(dct_abs[2 * third :, 2 * third :]) if 2 * third < size else 0.0
        )
        features.extend([float(low_energy), float(mid_energy), float(high_energy)])
    else:
        features.extend([0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32)
