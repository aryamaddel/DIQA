import numpy as np
import pyiqa

# Input image
image = "koniq10k_1024x768/10325321.jpg"

# Stage 1: Statistical Metrics
metrics = {
    "brisque": pyiqa.create_metric("brisque"),
    "niqe": pyiqa.create_metric("niqe"),
    "piqe": pyiqa.create_metric("piqe"),
}

scores = {name: metric(image).item() for name, metric in metrics.items()}
for name, score in scores.items():
    print(f"{name.upper()} score:", score)

# Stage 1.5: Out-of-Bounds Escalation
bounds = {"brisque": (0, 100), "niqe": (0, 25), "piqe": (0, 100)}
out_of_bounds = any(
    not (bounds[name][0] <= score <= bounds[name][1]) for name, score in scores.items()
)

if out_of_bounds:
    print("Out-of-bounds score detected, escalating to deep IQA models.")
    # Stage 4: Deep IQA Models
    deep_metrics = {
        "maniqa": pyiqa.create_metric("maniqa"),
        "hyperiqa": pyiqa.create_metric("hyperiqa"),
    }
    for name, metric in deep_metrics.items():
        print(f"{name.upper()} score:", metric(image).item())
else:
    # Stage 2: Normalization
    normalized_scores = {
        "brisque": scores["brisque"] / 100,
        "niqe": scores["niqe"] / 25,
        "piqe": scores["piqe"] / 100,
    }
    for name, norm_score in normalized_scores.items():
        print(f"Normalized {name.upper()}:", norm_score)

    # Stage 3: Consistency Check
    norm_values = list(normalized_scores.values())
    variance = np.var(norm_values)
    score_range = max(norm_values) - min(norm_values)
    threshold = 0.1

    print("Variance:", variance)
    print("Range:", score_range)

    if variance > threshold:
        print("High disagreement detected, escalating to deep IQA models.")
        # Stage 4: Deep IQA Models
        deep_metrics = {
            "maniqa": pyiqa.create_metric("maniqa"),
            "hyperiqa": pyiqa.create_metric("hyperiqa"),
        }
        for name, metric in deep_metrics.items():
            print(f"{name.upper()} score:", metric(image).item())
    else:
        print("Variance within threshold. Returning traditional metric scores.")
        for name, score in scores.items():
            print(f"Final {name.upper()} score:", score)
