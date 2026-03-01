import time, json, numpy as np, pickle, pyiqa
from pathlib import Path
from typing import Dict, Union, Any, Optional
from xgboost import XGBClassifier
from .features import extract_features


class DIQA:
    def __init__(
        self, model_dir: Optional[Union[str, Path]] = None, preload: bool = True
    ):
        """
        Initializes the DIQA evaluation engine.

        Args:
            model_dir: Directory containing trained models (router, scaler, mapping coefficients).
            preload: If True, preloads pyiqa metrics to memory.
        """
        models_dir = Path(model_dir or Path(__file__).parent / "models")
        self.router = XGBClassifier()
        self.router.load_model(str(models_dir / "router_xgb.json"))
        self.mapping = json.load(open(models_dir / "mos_mapping_coefficients.json"))

        with open(models_dir / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        self.metrics = {}
        if preload:
            for method in self.mapping:
                self.metrics[method] = pyiqa.create_metric(method, as_loss=False)

    def predict(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Predicts the Mean Opinion Score (MOS) for a given image.

        Args:
            image_path: The file path to the image to evaluate.

        Returns:
            Dictionary containing MOS estimate, method used, confidence, and internal inference time.
        """
        start_time, image_path = time.time(), Path(image_path)

        # 1. Extract features of image
        raw_features = extract_features(image_path)
        scaled_features = self.scaler.transform([raw_features])[0]

        # 2. Give to router model and decide which image metric to use
        probabilities = self.router.predict_proba([scaled_features])[0]
        selected_method = list(self.mapping)[np.argmax(probabilities)]

        # 3. Run selected method to get raw score, and align to MOS scale
        raw_score = self.metrics.setdefault(
            selected_method, pyiqa.create_metric(selected_method, as_loss=False)
        )(str(image_path)).item()

        mapping_coeffs = self.mapping[selected_method]
        final_mos = mapping_coeffs["coef"] * raw_score + mapping_coeffs["intercept"]

        return {
            "MOS": float(final_mos),
            "method": selected_method,
            "confidence": float(max(probabilities)),
            "time_ms": (time.time() - start_time) * 1000,
        }
