from pathlib import Path
import numpy as np
import cv2

TARGET_SIZE = (384, 512)  # KONIQ-10k 512×384 variant (H×W)

def preprocess_image(image_path: str | Path) -> dict[str, np.ndarray]:
    image_path = Path(image_path)

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Unable to read image: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if rgb.shape[:2] != TARGET_SIZE:
        rgb = cv2.resize(rgb, (TARGET_SIZE[1], TARGET_SIZE[0]), interpolation=cv2.INTER_AREA)

    rgb_float = rgb.astype(np.float32) / 255.0
    gray_float = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    return {"rgb": rgb_float, "gray": gray_float}