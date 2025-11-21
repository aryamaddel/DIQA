import os
import json
import time
import joblib
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from .features import extract_features
from .scoring import ScoreEngine


class DIQA:
    def __init__(self, model_dir=None):
        if model_dir is None:
            # Default to models directory inside the package
            model_dir = Path(__file__).parent / "models"
        else:
            model_dir = Path(model_dir)

        self.router_path = model_dir / "router_xgb.json"
        self.scaler_path = model_dir / "scaler.pkl"
        self.mapping_path = model_dir / "mos_mapping_coefficients.json"

        self._load_models()
        self.score_engine = (
            ScoreEngine()
        )  # Initialize score engine (handles its own caching)

    def _load_models(self):
        if not self.router_path.exists():
            raise FileNotFoundError(f"Router model not found at {self.router_path}")

        self.router = XGBClassifier()
        self.router.load_model(str(self.router_path))

        self.scaler = joblib.load(self.scaler_path)

        with open(self.mapping_path, "r") as f:
            self.mos_mapping = json.load(f)

        self.iqa_methods = ["brisque", "niqe", "piqe", "maniqa", "hyperiqa"]

    def predict(self, image_path):
        """
        Predict the MOS and routing method for an image.
        """
        start_time = time.time()
        image_path = Path(image_path)

        # Extract and scale features
        features = extract_features(image_path).reshape(1, -1)
        features_scaled = self.scaler.transform(features)

        # Predict routing method
        probabilities = self.router.predict_proba(features_scaled)[0]
        top_method_idx = np.argmax(probabilities)
        selected_method = self.iqa_methods[top_method_idx]
        confidence = probabilities[top_method_idx]

        # Compute IQA score using the selected method
        # We use ScoreEngine but we only need one specific method here.
        # However, ScoreEngine is designed for batch processing and caching.
        # For single inference, we can just use the metric directly if we want speed,
        # OR we can use ScoreEngine to benefit from caching if this image was seen before.

        # Let's use ScoreEngine's internal metric loading to avoid code duplication,
        # but we might want to bypass the full 'get_scores' overhead if we just want one value.
        # Actually, for a single image, let's just use the metric directly via ScoreEngine's helper if possible,
        # or just instantiate it here.
        # To keep things clean, let's add a method to ScoreEngine or just use it as is.

        # For now, let's just use the metric directly to be fast and simple for single inference.
        # But wait, we want to support caching?
        # If the user is running inference on a new image, caching it might be useful.
        # Let's use ScoreEngine.

        # Actually, ScoreEngine computes ALL scores by default in get_scores.
        # That's wasteful if we only need one.
        # Let's just load the specific metric.

        self.score_engine._load_metric(selected_method)
        raw_score = self.score_engine.metrics[selected_method](str(image_path)).item()

        # Map raw score to MOS
        coef, intercept = self.mos_mapping[selected_method].values()
        mos_estimate = coef * raw_score + intercept

        return {
            "MOS_estimate": mos_estimate,
            "selected_method": selected_method,
            "confidence": confidence,
            "timing": {"total_time_ms": (time.time() - start_time) * 1000},
        }
