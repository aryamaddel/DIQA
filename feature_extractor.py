import numpy as np
import cv2
from pathlib import Path
from scipy import stats
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy

TARGET_SIZE = (384, 512)
feature_names = ["mean_luminance", "std_luminance", "skewness_luminance", "kurtosis_luminance", 
                 "entropy_luminance", "median_luminance", "colorfulness", "mean_saturation", 
                 "laplacian_variance", "tenengrad", "canny_edge_ratio", "lbp_uniformity", 
                 "noise_std_estimate", "blockiness_energy", "rms_contrast", "percentile_contrast", 
                 "low_freq_energy", "mid_freq_energy", "high_freq_energy"]

def extract_features(image_path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(Path(image_path)), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Unable to read: {image_path}")
    
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != TARGET_SIZE:
        rgb = cv2.resize(rgb, (TARGET_SIZE[1], TARGET_SIZE[0]), interpolation=cv2.INTER_AREA)
    
    rgb = rgb.astype(np.float32) / 255.0
    gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    features = []
    
    flat = gray.flatten()
    features.extend([np.mean(flat), np.std(flat), stats.skew(flat), stats.kurtosis(flat), shannon_entropy(gray), np.median(flat)])
    
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    colorfulness = np.sqrt(np.std(rg)**2 + np.std(yb)**2) + 0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
    hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    features.extend([colorfulness, np.mean(hsv[:, :, 1].astype(float)) / 255.0])
    
    gray_uint8 = (gray * 255).astype(np.uint8)
    laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)
    grad_x = cv2.Sobel(gray_uint8, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_uint8, cv2.CV_64F, 0, 1, ksize=3)
    features.extend([float(np.var(laplacian)), float(np.mean(np.sqrt(grad_x**2 + grad_y**2)))])
    
    edges = cv2.Canny(gray_uint8, 50, 150)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 9), density=True)
    lbp_hist += 1e-10
    features.extend([float(np.sum(edges > 0) / edges.size), float(-np.sum(lbp_hist * np.log(lbp_hist)))])
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    residual = gray - blurred
    features.append(float(np.median(np.abs(residual - np.median(residual))) * 1.4826))
    
    h, w = gray.shape
    block_vars, neighbor_vars = [], []
    for i in range(0, h - 16, 8):
        for j in range(0, w - 16, 8):
            if i + 16 < h and j + 8 < w:
                block_vars.append(np.var(gray[i:i+8, j:j+8]))
                neighbor_vars.append(np.var(gray[i+8:i+16, j:j+8]))
    features.append(float(np.abs(np.mean(block_vars) - np.mean(neighbor_vars)) if block_vars else 0.0))
    
    mean_gray = np.mean(gray)
    features.extend([float(np.sqrt(np.mean((gray - mean_gray)**2))), float(np.percentile(gray, 90) - np.percentile(gray, 10))])
    
    center_size = min(64, h // 2, w // 2)
    if center_size >= 8:
        center_h, center_w = h // 2, w // 2
        crop = gray[center_h - center_size//2:center_h + center_size//2, center_w - center_size//2:center_w + center_size//2]
        dct_abs = np.abs(cv2.dct(crop.astype(np.float32)))
        size, third = dct_abs.shape[0], max(1, dct_abs.shape[0] // 3)
        low = np.mean(dct_abs[:third, :third])
        mid = np.mean(dct_abs[third:2*third, third:2*third]) if 2*third <= size else 0.0
        high = np.mean(dct_abs[2*third:, 2*third:]) if 2*third < size else 0.0
        features.extend([float(low), float(mid), float(high)])
    else:
        features.extend([0.0, 0.0, 0.0])
    
    return np.array(features, dtype=np.float32)
