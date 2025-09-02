import cv2
from skimage import io
from skimage.util import img_as_float
import pyiqa
import torch
import os
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create each metric
brisque = pyiqa.create_metric("brisque", device=device)
niqe = pyiqa.create_metric("niqe", device=device)
piqe = pyiqa.create_metric("piqe", device=device)

# Path to KaDiD 10k images folder
images_folder = os.path.join(os.path.dirname(__file__), "KaDiD Small")

# Get all image file paths
image_files = []
for root, dirs, files in os.walk(images_folder):
    for f in files:
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            image_files.append(os.path.join(root, f))

results = []

for img_path in image_files:
    img = io.imread(img_path)
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
    img_float = img_as_float(img_gray)

    # pyiqa's img utilities accept file paths, bytes, or PIL Images.
    # Pass the original file path so pyiqa can read/convert the image itself.
    brisque_score = brisque(img_path).item()
    piqe_score = piqe(img_path).item()
    niqe_score = niqe(img_path).item()

    results.append(
        {
            "Image": os.path.basename(img_path),
            "BRISQUE": brisque_score,
            "PIQE": piqe_score,
            "NIQE": niqe_score*10,
        }
    )


df = pd.DataFrame(results)
print(df)
print(f"Processed {len(results)} images")
