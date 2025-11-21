import pyiqa
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os


class ScoreEngine:
    def __init__(self, cache_csv="Data/iqa_raw_scores.csv"):
        self.cache_csv = Path(cache_csv)
        self.iqa_methods = ["brisque", "niqe", "piqe", "maniqa", "hyperiqa"]
        self.metrics = {}
        self._load_cache()

    def _load_cache(self):
        if self.cache_csv.exists():
            self.cache = pd.read_csv(self.cache_csv)
            # Ensure image_name is string for consistent querying
            self.cache["image_name"] = self.cache["image_name"].astype(str)
        else:
            self.cache = pd.DataFrame(columns=["image_name"] + self.iqa_methods)

    def _save_cache(self):
        self.cache_csv.parent.mkdir(exist_ok=True, parents=True)
        self.cache.to_csv(self.cache_csv, index=False)

    def _load_metric(self, method):
        if method not in self.metrics:
            print(f"Loading {method} model...")
            self.metrics[method] = pyiqa.create_metric(method)

    def get_scores(self, image_paths, update_cache=True):
        """
        Get scores for a list of images. Uses cache if available.
        """
        if isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]

        image_paths = [Path(p) for p in image_paths]
        results = []
        to_compute = []

        # Check cache first
        for img_path in image_paths:
            img_name = img_path.name
            cached_row = self.cache[self.cache["image_name"] == img_name]

            if not cached_row.empty:
                # Check if all methods are present (handling partial cache if we add methods later)
                if (
                    all(m in cached_row.columns for m in self.iqa_methods)
                    and not cached_row[self.iqa_methods].isnull().values.any()
                ):
                    results.append(cached_row.iloc[0].to_dict())
                    continue

            to_compute.append(img_path)

        # Compute missing scores
        if to_compute:
            print(f"Computing scores for {len(to_compute)} images...")
            new_rows = []

            # Load all metrics needed
            for method in self.iqa_methods:
                self._load_metric(method)

            for img_path in tqdm(to_compute, desc="Scoring"):
                try:
                    scores = {
                        method: self.metrics[method](str(img_path)).item()
                        for method in self.iqa_methods
                    }
                    row = {"image_name": img_path.name, **scores}
                    new_rows.append(row)
                    results.append(row)
                except Exception as e:
                    print(f"Error scoring {img_path.name}: {e}")

            if new_rows and update_cache:
                new_df = pd.DataFrame(new_rows)
                self.cache = pd.concat([self.cache, new_df], ignore_index=True)
                # Deduplicate keeping latest
                self.cache = self.cache.drop_duplicates(
                    subset=["image_name"], keep="last"
                )
                self._save_cache()

        return pd.DataFrame(results)
