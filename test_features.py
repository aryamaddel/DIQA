import numpy as np
from pathlib import Path
from preprocess import preprocess_image
from feature_extractor import FeatureExtractor

def test_feature_extraction():
    """Test feature extraction on a sample image."""
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Test with a sample image (you'll need to replace with actual image path)
    sample_image_path = "koniq10k_512x384/826373.jpg"  # Replace with actual path

    try:
        # Preprocess image
        preprocessed = preprocess_image(sample_image_path)
        print(f"✓ Image preprocessed successfully")
        print(f"  RGB shape: {preprocessed['rgb'].shape}")
        print(f"  Gray shape: {preprocessed['gray'].shape}")
        
        # Extract features
        features = extractor.extract_features(preprocessed)
        feature_names = extractor.get_feature_names()
        
        print(f"✓ Features extracted successfully")
        print(f"  Number of features: {len(features)}")
        print(f"  Expected features: {len(feature_names)}")
        
        # Display features
        print("\nExtracted Features:")
        print("-" * 50)
        for name, value in zip(feature_names, features):
            print(f"{name:20s}: {value:.6f}")
            
        # Basic sanity checks
        assert len(features) == len(feature_names), "Feature count mismatch"
        assert not np.any(np.isnan(features)), "NaN values detected"
        assert not np.any(np.isinf(features)), "Infinite values detected"
        
        print(f"\n✓ All checks passed!")
        return features
        
    except FileNotFoundError:
        print("❌ Sample image not found. Please update the sample_image_path.")
        print("You can test with any image from your dataset.")
        return None
    except Exception as e:
        print(f"❌ Error during feature extraction: {e}")
        return None

def test_synthetic_image():
    """Test with synthetic image data."""
    print("Testing with synthetic image...")
    
    extractor = FeatureExtractor()
    
    # Create synthetic preprocessed data
    synthetic_data = {
        'rgb': np.random.rand(384, 512, 3).astype(np.float32),
        'gray': np.random.rand(384, 512).astype(np.float32)
    }
    
    features = extractor.extract_features(synthetic_data)
    feature_names = extractor.get_feature_names()
    
    print(f"✓ Synthetic test passed")
    print(f"  Features extracted: {len(features)}")
    print(f"  Feature range: [{np.min(features):.3f}, {np.max(features):.3f}]")
    
    return features

if __name__ == "__main__":
    print("Testing Feature Extractor")
    print("=" * 50)
    
    # Test with synthetic data first
    test_synthetic_image()
    print()
    
    # Test with real image (if available)
    test_feature_extraction()