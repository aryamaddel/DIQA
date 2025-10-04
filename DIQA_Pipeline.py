import numpy as np
from pathlib import Path
from preprocess import preprocess_image
from feature_extractor import extract_features, feature_names

image_path = "koniq10k_512x384/826373.jpg"

print(f"Processing image: {image_path}")

# Step 1: Preprocess (standardize the image)
print("Step 1: Preprocessing image...")
preprocessed = preprocess_image(image_path)
print(f"  ✓ RGB shape: {preprocessed['rgb'].shape}")
print(f"  ✓ Gray shape: {preprocessed['gray'].shape}")

# Step 2: Extract features (from the standardized image)
print("Step 2: Extracting features...")
features = extract_features(preprocessed)
print(f"  ✓ Extracted {len(features)} features")

print(f"Features extracted: {len(features)}")
print(f"Feature range: [{np.min(features):.3f}, {np.max(features):.3f}]")

print("\nFeature Summary:")
print("-" * 30)
for name, value in zip(feature_names, features):
    print(f"{name:20s}: {value:.6f}")
    
