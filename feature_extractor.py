import numpy as np
import cv2
from scipy import stats
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy
import warnings
warnings.filterwarnings('ignore')

class FeatureExtractor:
    def __init__(self):
        self.feature_names = [
            # Global stats (6)
            'mean_luminance', 'std_luminance', 'skewness_luminance', 
            'kurtosis_luminance', 'entropy_luminance', 'median_luminance',
            # Colorfulness & saturation (2)  
            'colorfulness', 'mean_saturation',
            # Sharpness / Blur (2)
            'laplacian_variance', 'tenengrad',
            # Edge density / texture (2)
            'canny_edge_ratio', 'lbp_uniformity',
            # Noise / smoothness (1)
            'noise_std_estimate',
            # Blockiness / compression (1)
            'blockiness_energy',
            # Contrast & dynamic range (2)
            'rms_contrast', 'percentile_contrast',
            # Frequency energy bands (3)
            'low_freq_energy', 'mid_freq_energy', 'high_freq_energy'
        ]
    
    def extract_features(self, preprocessed_data: dict) -> np.ndarray:
        """Extract all handcrafted features from preprocessed image data."""
        rgb = preprocessed_data['rgb']
        gray = preprocessed_data['gray']
        
        features = []
        
        # 1. Global stats (6 features)
        features.extend(self._global_stats(gray))
        
        # 2. Colorfulness & saturation (2 features)
        features.extend(self._color_features(rgb))
        
        # 3. Sharpness / Blur (2 features)
        features.extend(self._sharpness_features(gray))
        
        # 4. Edge density / texture (2 features)
        features.extend(self._edge_texture_features(gray))
        
        # 5. Noise / smoothness (1 feature)
        features.append(self._noise_estimate(gray))
        
        # 6. Blockiness / compression (1 feature)
        features.append(self._blockiness_energy(gray))
        
        # 7. Contrast & dynamic range (2 features)
        features.extend(self._contrast_features(gray))
        
        # 8. Frequency energy bands (3 features)
        features.extend(self._frequency_features(gray))
        
        return np.array(features, dtype=np.float32)
    
    def _global_stats(self, gray: np.ndarray) -> list:
        """Global luminance statistics."""
        flat = gray.flatten()
        
        mean_lum = np.mean(flat)
        std_lum = np.std(flat)
        skew_lum = stats.skew(flat)
        kurt_lum = stats.kurtosis(flat)
        entropy_lum = shannon_entropy(gray)
        median_lum = np.median(flat)
        
        return [mean_lum, std_lum, skew_lum, kurt_lum, entropy_lum, median_lum]
    
    def _color_features(self, rgb: np.ndarray) -> list:
        """Colorfulness and saturation metrics."""
        # Colorfulness (Hasler & Süsstrunk style)
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        rg = r - g
        yb = 0.5 * (r + g) - b
        
        std_rg = np.std(rg)
        std_yb = np.std(yb)
        mean_rg = np.mean(rg)
        mean_yb = np.mean(yb)
        
        colorfulness = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
        
        # Mean saturation (HSV)
        hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        mean_saturation = np.mean(hsv[:, :, 1]) / 255.0
        
        return [colorfulness, mean_saturation]
    
    def _sharpness_features(self, gray: np.ndarray) -> list:
        """Sharpness and blur indicators."""
        # Convert to uint8 for OpenCV operations
        gray_uint8 = (gray * 255).astype(np.uint8)
        
        # Variance of Laplacian
        laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)
        laplacian_var = np.var(laplacian)
        
        # Tenengrad (mean gradient magnitude)
        grad_x = cv2.Sobel(gray_uint8, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_uint8, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        
        return [float(laplacian_var), float(tenengrad)]
    
    def _edge_texture_features(self, gray: np.ndarray) -> list:
        """Edge density and texture features."""
        # Canny edge pixel ratio
        gray_uint8 = (gray * 255).astype(np.uint8)
        edges = cv2.Canny(gray_uint8, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        
        # LBP uniformity measure
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 9), density=True)
        # Add small epsilon to avoid log(0)
        lbp_hist = lbp_hist + 1e-10
        lbp_uniformity = -np.sum(lbp_hist * np.log(lbp_hist))  # entropy
        
        return [float(edge_ratio), float(lbp_uniformity)]
    
    def _noise_estimate(self, gray: np.ndarray) -> float:
        """Noise estimation via high-pass residual."""
        # High-pass filter (subtract Gaussian blur)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        residual = gray - blurred
        
        # Median absolute deviation estimate
        noise_std = np.median(np.abs(residual - np.median(residual))) * 1.4826
        
        return float(noise_std)
    
    def _blockiness_energy(self, gray: np.ndarray) -> float:
        """Block artifact energy indicator."""
        h, w = gray.shape
        block_size = 8
        
        # Sample 8x8 blocks
        block_variances = []
        neighbor_variances = []
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                if i + 2*block_size < h and j + 2*block_size < w:
                    # Current block
                    block = gray[i:i+block_size, j:j+block_size]
                    block_var = np.var(block)
                    
                    # Neighboring block
                    neighbor = gray[i+block_size:i+2*block_size, j:j+block_size]
                    neighbor_var = np.var(neighbor)
                    
                    block_variances.append(block_var)
                    neighbor_variances.append(neighbor_var)
        
        if len(block_variances) > 0:
            # Blockiness as difference in variance patterns
            blockiness = np.abs(np.mean(block_variances) - np.mean(neighbor_variances))
        else:
            blockiness = 0.0
            
        return float(blockiness)
    
    def _contrast_features(self, gray: np.ndarray) -> list:
        """Contrast and dynamic range measures."""
        # RMS contrast
        mean_gray = np.mean(gray)
        rms_contrast = np.sqrt(np.mean((gray - mean_gray)**2))
        
        # Percentile-based contrast
        p90 = np.percentile(gray, 90)
        p10 = np.percentile(gray, 10)
        percentile_contrast = p90 - p10
        
        return [float(rms_contrast), float(percentile_contrast)]
    
    def _frequency_features(self, gray: np.ndarray) -> list:
        """Frequency domain energy distribution."""
        # Take DCT of center crop to avoid edge effects
        h, w = gray.shape
        center_size = min(64, h//2, w//2)
        
        if center_size < 8:
            return [0.0, 0.0, 0.0]
            
        center_h = h // 2
        center_w = w // 2
        crop = gray[center_h-center_size//2:center_h+center_size//2, 
                   center_w-center_size//2:center_w+center_size//2]
        
        # Ensure crop is float32 for DCT
        crop_f32 = crop.astype(np.float32)
        
        # 2D DCT
        dct = cv2.dct(crop_f32)
        dct_abs = np.abs(dct)
        
        # Divide into frequency bands (low, mid, high)
        size = dct_abs.shape[0]
        third = max(1, size // 3)
        
        low_band = dct_abs[:third, :third]
        mid_band = dct_abs[third:2*third, third:2*third]
        high_band = dct_abs[2*third:, 2*third:]
        
        low_energy = np.mean(low_band) if low_band.size > 0 else 0.0
        mid_energy = np.mean(mid_band) if mid_band.size > 0 else 0.0
        high_energy = np.mean(high_band) if high_band.size > 0 else 0.0
        
        return [float(low_energy), float(mid_energy), float(high_energy)]
    
    def get_feature_names(self) -> list:
        """Return list of feature names."""
        return self.feature_names.copy()