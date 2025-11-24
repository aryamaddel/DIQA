import sys
import os
import pandas as pd
from pathlib import Path

# Ensure we can import the package locally
sys.path.append(os.getcwd())

from diqa import DIQA, DIQATrainer


def test_inference():
    print("Testing Inference...")
    diqa = DIQA()

    # Find a sample image
    image_dir = Path("koniq10k_512x384")
    if not image_dir.exists():
        print("Image directory not found, skipping inference test on real image.")
        return

    image_files = list(image_dir.glob("*.jpg"))
    if not image_files:
        print("No images found for testing.")
        return

    sample_image = image_files[0]
    print(f"Predicting on {sample_image}...")
    result = diqa.predict(sample_image)
    print("Result:", result)
    assert "MOS_estimate" in result
    assert "selected_method" in result


def test_training():
    print("\nTesting Training...")
    # We need a dummy MOS CSV for testing
    image_dir = Path("koniq10k_512x384")
    if not image_dir.exists():
        print("Image directory not found, skipping training test.")
        return

    image_files = list(image_dir.glob("*.jpg"))[:50]  # Use 50 images
    if not image_files:
        print("No images found.")
        return

    # Use real CSV
    real_csv = Path("Data/koniq10k_scores_and_distributions.csv")
    if not real_csv.exists():
        print("Real CSV not found, skipping training test.")
        return

    # Filter CSV to match our test images
    df = pd.read_csv(real_csv)
    image_names = [f.name for f in image_files]
    subset_df = df[df["image_name"].isin(image_names)]

    if subset_df.empty:
        print("No matching images found in CSV.")
        return

    temp_csv = "temp_test_mos.csv"
    subset_df.to_csv(temp_csv, index=False)

    try:
        trainer = DIQATrainer(image_dir, temp_csv, output_dir="test_models")
        trainer.prepare_data()
        trainer.train(test_split=0.2)
        print("Training test passed.")
    except Exception as e:
        print(f"Training test failed: {e}")
    finally:
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
        # Cleanup test_models if needed, or leave for inspection


if __name__ == "__main__":
    import pandas as pd  # Needed for test_training

    test_inference()
    test_training()
