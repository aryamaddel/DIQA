import pytest
import numpy as np
import cv2
import pandas as pd
from pathlib import Path

from diqa.training import get_scores


@pytest.fixture
def dummy_image(tmp_path):
    img_path = tmp_path / "test_image.jpg"
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    return img_path


def test_get_scores(dummy_image):
    """Test get_scores correctly pulls out pyiqa metrics for list of methods."""
    # We use a very fast metric to not make the test slow.
    methods = ["brisque"]
    df = get_scores([dummy_image], methods=methods)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "image_name" in df.columns
    assert "brisque" in df.columns
    assert isinstance(df.iloc[0]["brisque"], float)
