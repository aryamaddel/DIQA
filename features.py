import numpy as np
import cv2
from skimage.measure import shannon_entropy

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
    features = []

    # Basic luminance statistics (3 features)
    flat = gray.flatten()
    features.extend(
        [
            np.mean(flat),
            np.std(flat),
            shannon_entropy(gray),
        ]
    )

    # Color features (2 features)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    colorfulness = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2) + 0.3 * np.sqrt(
        np.mean(rg) ** 2 + np.mean(yb) ** 2
    )
    hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    features.extend([colorfulness, np.mean(hsv[:, :, 1].astype(float)) / 255.0])

    # Sharpness features (2 features)
    gray_uint8 = (gray * 255).astype(np.uint8)
    laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)
    grad_x = cv2.Sobel(gray_uint8, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_uint8, cv2.CV_64F, 0, 1, ksize=3)
    features.extend(
        [float(np.var(laplacian)), float(np.mean(np.sqrt(grad_x**2 + grad_y**2)))]
    )

    # Edge density (1 feature)
    edges = cv2.Canny(gray_uint8, 50, 150)
    features.append(float(np.sum(edges > 0) / edges.size))

    # Noise estimate (1 feature)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    residual = gray - blurred
    features.append(float(np.median(np.abs(residual - np.median(residual))) * 1.4826))

    # Contrast features (2 features)
    mean_gray = np.mean(gray)
    features.extend(
        [
            float(np.sqrt(np.mean((gray - mean_gray) ** 2))),
            float(np.percentile(gray, 90) - np.percentile(gray, 10)),
        ]
    )

    return np.array(features, dtype=np.float32)
