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
with open("mos_mapping_coefficients.json", 'r') as f:
    mos_mapping = json.load(f)
iqa_methods = ["brisque", "niqe", "piqe", "maniqa", "hyperiqa"]
metrics = {}

def load_metric(method):
    if method not in metrics:
        metrics[method] = pyiqa.create_metric(method)

def predict(image_path, confidence_threshold=0.5, use_top_k=2):
    start = time.time()
    features = extract_features(image_path).reshape(1, -1)
    features_scaled = scaler.transform(features)
    feature_time = time.time() - start
    
    router_start = time.time()
    probs = router.predict_proba(features_scaled)[0]
    top_idx = np.argmax(probs)
    confidence = probs[top_idx]
    selected_method = iqa_methods[top_idx]
    router_time = time.time() - router_start
    
    iqa_start = time.time()
    if confidence >= confidence_threshold:
        load_metric(selected_method)
        score = metrics[selected_method](str(image_path)).item()
        mos = mos_mapping[selected_method]['coef'] * score + mos_mapping[selected_method]['intercept']
        methods_used = [selected_method]
    else:
        top_k_idx = np.argsort(probs)[-use_top_k:][::-1]
        methods_used = [iqa_methods[i] for i in top_k_idx]
        scores = []
        for i in top_k_idx:
            method = iqa_methods[i]
            load_metric(method)
            s = metrics[method](str(image_path)).item()
            scores.append(mos_mapping[method]['coef'] * s + mos_mapping[method]['intercept'])
        weights = probs[top_k_idx] / probs[top_k_idx].sum()
        mos = np.sum(weights * scores)
    
    iqa_time = time.time() - iqa_start
    
    return {
        "image_path": str(image_path),
        "MOS_estimate": float(mos),
        "selected_method": selected_method,
        "confidence": float(confidence),
        "methods_used": methods_used,
        "timing": {
            "feature_extraction_ms": feature_time * 1000,
            "router_inference_ms": router_time * 1000,
            "iqa_computation_ms": iqa_time * 1000,
            "total_time_ms": (time.time() - start) * 1000
        },
        "router_probabilities": {iqa_methods[i]: float(probs[i]) for i in range(len(iqa_methods))}
    }
