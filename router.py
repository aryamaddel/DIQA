import numpy as np
import pyiqa
import joblib
import json
import time
from xgboost import XGBClassifier
from features import extract_features

router = XGBClassifier()
router.load_model("router_xgb.json")
scaler = joblib.load("scaler.pkl")
with open("mos_mapping_coefficients.json", "r") as f:
    mos_mapping = json.load(f)
iqa_methods = ["brisque", "niqe", "piqe", "maniqa", "hyperiqa"]
metrics = {}


def load_metric(method):
    if method not in metrics:
        metrics[method] = pyiqa.create_metric(method)


def predict(image_path):
    """Predict the MOS and routing method for an image."""
    start_time = time.time()

    # Extract and scale features
    features = extract_features(image_path).reshape(1, -1)
    features_scaled = scaler.transform(features)

    # Predict routing method
    probabilities = router.predict_proba(features_scaled)[0]
    top_method_idx = np.argmax(probabilities)
    selected_method = iqa_methods[top_method_idx]
    confidence = probabilities[top_method_idx]

    # Compute IQA score
    load_metric(selected_method)
    raw_score = metrics[selected_method](str(image_path)).item()

    # Map raw score to MOS
    coef, intercept = mos_mapping[selected_method].values()
    mos_estimate = coef * raw_score + intercept

    return {
        "MOS_estimate": mos_estimate,
        "selected_method": selected_method,
        "confidence": confidence,
        "timing": {"total_time_ms": (time.time() - start_time) * 1000},
    }
