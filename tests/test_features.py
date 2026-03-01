import pytest
import numpy as np
import cv2
import pandas as pd
from pathlib import Path

from diqa.features import extract_features, build_features, FEATURE_NAMES


@pytest.fixture
def dummy_image(tmp_path):
    """Creates a temporary dummy image for testing."""
    img_path = tmp_path / "test_image.jpg"
    # Create a simple 10x10 RGB image
    img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    return img_path


def test_extract_features(dummy_image):
    """Test that extract_features returns a list of floats matching FEATURE_NAMES length."""
    features = extract_features(dummy_image)

    assert isinstance(features, list)
    assert len(features) == len(FEATURE_NAMES)
    assert all(isinstance(f, float) for f in features)


def test_build_features_single_image(dummy_image):
    """Test build_features with a single image path."""
    df = build_features(dummy_image, return_df=True)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "image_name" in df.columns
    assert "image_path" in df.columns
    for name in FEATURE_NAMES:
        assert name in df.columns


def test_build_features_directory(dummy_image):
    """Test build_features with a directory of images."""
    df = build_features(dummy_image.parent, return_df=True)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["image_name"] == dummy_image.name
