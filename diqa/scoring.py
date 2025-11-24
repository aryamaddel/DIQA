import pyiqa
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os


class ScoreEngine:
    def __init__(self):
        self.iqa_methods = ["brisque", "niqe", "piqe", "maniqa", "hyperiqa"]
        self.metrics = {}

    def _load_metric(self, method):
        """Load a metric model if not already loaded."""
        if method not in self.metrics:
            self.metrics[method] = pyiqa.create_metric(method)

    def get_scores(self, image_paths):
        """
        Get scores for a list of images.
        """
        if isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]

        image_paths = [Path(p) for p in image_paths]
        results = []

        print(f"Computing scores for {len(image_paths)} images...")

        # Load all metrics needed
        for method in self.iqa_methods:
            self._load_metric(method)

        for img_path in tqdm(image_paths, desc="Scoring"):
            try:
                scores = {
                    method: self.metrics[method](str(img_path)).item()
                    for method in self.iqa_methods
                }
                row = {"image_name": img_path.name, **scores}
                results.append(row)
            except Exception as e:
                print(f"Error scoring {img_path.name}: {e}")

        return pd.DataFrame(results)
