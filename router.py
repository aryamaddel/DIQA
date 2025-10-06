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
    """Simplified deterministic routing with MOS prediction."""
    start = time.time()

    # Extract and scale features
    features = extract_features(image_path).reshape(1, -1)
    features_scaled = scaler.transform(features)

    # Router prediction (deterministic top-1)
    probs = router.predict_proba(features_scaled)[0]
    top_idx = np.argmax(probs)
    selected_method = iqa_methods[top_idx]
    confidence = probs[top_idx]

    # Load and compute IQA score
    load_metric(selected_method)
    raw_score = metrics[selected_method](str(image_path)).item()

    # Map to MOS scale
    coef = mos_mapping[selected_method]["coef"]
    intercept = mos_mapping[selected_method]["intercept"]
    mos = coef * raw_score + intercept

    total_time = (time.time() - start) * 1000  # ms

    return {
        "MOS_estimate": mos,
        "selected_method": selected_method,
        "confidence": confidence,
        "timing": {"total_time_ms": total_time},
    }
