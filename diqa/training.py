import json, pandas as pd, numpy as np, pickle, pyiqa
from pathlib import Path
from typing import Union, Dict, Any
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from tqdm import tqdm
from .features import build_features, FEATURE_NAMES


def get_scores(image_paths, methods=("brisque", "niqe", "piqe", "maniqa", "hyperiqa")):
    metrics = {m: pyiqa.create_metric(m) for m in methods}
    paths = [
        Path(p)
        for p in (image_paths if isinstance(image_paths, list) else [image_paths])
    ]
    return pd.DataFrame(
        [
            {"image_name": p.name, **{m: metrics[m](str(p)).item() for m in methods}}
            for p in tqdm(paths, "Scoring")
        ]
    )


class DIQATrainer:
    def __init__(
        self,
        image_dir: Union[str, Path],
        mos_data: Union[str, Path, pd.DataFrame, Any],
        output_dir: Union[str, Path] = "DIQA/models",
    ):
        """
        Initializes the trainer for custom dataset learning.

        Args:
            image_dir: Base directory for dataset images.
            mos_data: Path to CSV/Parquet containing Mean Opinion Scores, a Pandas DataFrame, or a Hugging Face Dataset (requires 'image_name' and 'MOS' columns).
            output_dir: Directory where the trained models and artifacts will be saved.
        """
        self.image_dir, self.output_dir = Path(image_dir), Path(output_dir)
        self.mos_data = mos_data
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.iqa_methods = ["brisque", "niqe", "piqe", "maniqa", "hyperiqa"]

    def _load_mos(self):
        if isinstance(self.mos_data, pd.DataFrame):
            return self.mos_data
        if hasattr(self.mos_data, "to_pandas"):
            return self.mos_data.to_pandas()
        return (
            pd.read_parquet(self.mos_data)
            if str(self.mos_data).endswith(".parquet")
            else pd.read_csv(self.mos_data)
        )

    def prepare_data(self) -> pd.DataFrame:
        """
        Extracts features and reference scores for all target images.
        """
        mos_dataframe = self._load_mos()
        image_paths = [self.image_dir / name for name in mos_dataframe["image_name"]]
        features_df, scores_df = build_features(image_paths), get_scores(
            image_paths, methods=self.iqa_methods
        )
        self.df = mos_dataframe.merge(features_df, on="image_name").merge(
            scores_df, on="image_name"
        )
        return self.df

    def train(self, test_size: float = 0.2, seed: int = 42) -> Dict[str, Any]:
        """
        Trains the linear regressors for raw metric mapping and the XGBoost router model.
        """
        coefficients, errors_to_mos = {}, {}

        # 1. Compute IQA values & align them to human given MOS scores
        # We need this because IQA metrics output random scales (e.g. 0-100 or 0-1)
        for method in self.iqa_methods:
            regression_model = LinearRegression().fit(self.df[[method]], self.df["MOS"])
            coefficients[method] = {
                "coef": float(regression_model.coef_[0]),
                "intercept": float(regression_model.intercept_),
            }
            # Calculate how far off this method is from the human MOS
            aligned_scores = regression_model.predict(self.df[[method]])
            errors_to_mos[method] = np.abs(aligned_scores - self.df["MOS"])

        json.dump(
            coefficients,
            open(self.output_dir / "mos_mapping_coefficients.json", "w"),
            indent=2,
        )

        # 2. Check which method is best for what type of features
        # The best method is simply the one with the lowest error against human MOS
        best_method_for_each_image = pd.DataFrame(errors_to_mos).idxmin(axis=1)

        # Map method names back to label IDs (0, 1, 2...) for XGBoost
        y_labels = best_method_for_each_image.map(
            {method: i for i, method in enumerate(self.iqa_methods)}
        )

        X_features = self.df[FEATURE_NAMES].values

        X_train, X_test, y_train, y_test = train_test_split(
            X_features,
            y_labels.values,
            test_size=test_size,
            random_state=seed,
            stratify=y_labels.values,
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        with open(self.output_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        # 3. Train router model to learn which type of images to pass to which model
        router = XGBClassifier(n_estimators=100, random_state=seed).fit(
            X_train, y_train
        )
        router.save_model(self.output_dir / "router_xgb.json")

        return {
            "train_acc": float(router.score(X_train, y_train)),
            "test_acc": float(router.score(X_test, y_test)),
            "router": router,
        }
