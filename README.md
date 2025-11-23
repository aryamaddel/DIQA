# DIQA: No-Reference Image Quality Assessment Library

[![PyPI version](https://badge.fury.io/py/diqa.svg)](https://badge.fury.io/py/diqa)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DIQA is a production-ready Python library for No-Reference Image Quality Assessment (NR-IQA). It provides a unified interface for predicting image quality using multiple state-of-the-art methods and includes a routing mechanism to select the best method for a given image.

## Features

*   **Unified Inference**: Simple API to get MOS (Mean Opinion Score) estimates.
*   **Smart Routing**: Uses an XGBoost-based router to dynamically select the most accurate IQA method for each image.
*   **Training Support**: Built-in `DIQATrainer` to train the router and mapping coefficients on your own datasets.
*   **Extensible**: Easily integrates with `pyiqa` for underlying feature extraction.

## Installation

```bash
pip install diqa
```

## Usage

### 1. Basic Inference

Predict the quality score of an image:

```python
from diqa import DIQA

# Initialize the engine
diqa = DIQA()

# Predict
result = diqa.predict("path/to/image.jpg")

print(f"MOS Estimate: {result['MOS_estimate']}")
print(f"Method Used: {result['selected_method']}")
```

### 2. Training on Custom Data

Train the routing model on your own dataset (requires a CSV with `image_name` and `MOS` columns):

```python
from diqa import DIQATrainer

trainer = DIQATrainer(
    image_dir="path/to/images",
    mos_csv="path/to/scores.csv",
    output_dir="my_custom_models"
)

# Prepare data (extract features and compute base scores)
trainer.prepare_data()

# Train the router
trainer.train()
```

## License

MIT
