import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


from .features import build_features, FEATURE_NAMES
from .scoring import ScoreEngine


class DIQATrainer:
    def __init__(self, image_dir, mos_csv, output_dir="DIQA/models"):
        self.image_dir = Path(image_dir)
        self.mos_csv = Path(mos_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.score_engine = ScoreEngine()
        self.iqa_methods = self.score_engine.iqa_methods

    def prepare_data(self):
        """
        Prepares data for training:
        1. Loads MOS data.
        2. Extracts features.
        3. Computes/Loads IQA scores.
        4. Merges everything.
        """
        print("Loading MOS data...")
        df_mos = pd.read_csv(self.mos_csv)
        # Expecting 'image_name' and 'MOS' columns
        if "image_name" not in df_mos.columns or "MOS" not in df_mos.columns:
            raise ValueError("MOS CSV must contain 'image_name' and 'MOS' columns.")

        image_paths = [self.image_dir / name for name in df_mos["image_name"]]

        # 1. Extract Features
        print("Extracting features...")
        df_features = build_features(image_paths, return_df=True)

        # 2. Compute Scores
        print("Computing IQA scores...")
        df_scores = self.score_engine.get_scores(image_paths)

        # 3. Merge
        print("Merging data...")
        self.df = df_mos.merge(df_features, on="image_name").merge(
            df_scores, on="image_name"
        )
        return self.df

    def train(self, test_split=0.2, random_seed=42):
        """
        Trains the router and mapping coefficients.
        """
        if not hasattr(self, "df"):
            raise RuntimeError("Data not prepared. Call prepare_data() first.")

        print("Training mapping coefficients...")
        coeffs = {}
        errors = {}

        # Fit linear regressions for each method
        for m in self.iqa_methods:
            reg = LinearRegression().fit(self.df[[m]], self.df["MOS"])
            coeffs[m] = {
                "coef": float(reg.coef_[0]),
                "intercept": float(reg.intercept_),
            }
            # Compute error for this method on all images
            # We want to find which method is BEST for each image
            pred_mos = reg.predict(self.df[[m]])
            errors[m] = np.abs(pred_mos - self.df["MOS"])

        # Save coefficients
        with open(self.output_dir / "mos_mapping_coefficients.json", "w") as f:
            json.dump(coeffs, f, indent=2)

        # Determine best method for each image
        self.df["best_method"] = pd.DataFrame(errors).idxmin(axis=1)
        self.df["best_method_label"] = self.df["best_method"].map(
            {m: i for i, m in enumerate(self.iqa_methods)}
        )

        # Prepare for XGBoost
        X = self.df[FEATURE_NAMES].values
        y = self.df["best_method_label"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=random_seed, stratify=y
        )

        print("Training XGBoost router...")
        router = XGBClassifier(n_estimators=100, random_state=random_seed)
        router.fit(X_train, y_train)

        # Evaluation
        train_acc = router.score(X_train, y_train)
        test_acc = router.score(X_test, y_test)
        print(f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

        # Save models
        router.save_model(self.output_dir / "router_xgb.json")

        return {
            "train_acc": train_acc,
            "test_acc": test_acc,
            "router": router,
            "X_test": X_test,
            "y_test": y_test,
        }
