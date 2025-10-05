"""
Example usage of the simplified deterministic NR-IQA router.
This demonstrates how to use the router with the new simplified architecture.
"""

from router import predict

# Example 1: Assess a single image
print("=" * 60)
print("Example 1: Single Image Assessment")
print("=" * 60)

result = predict("me at night.jpg")

print(f"\nImage: {result['image_path']}")
print(f"MOS Estimate: {result['MOS_estimate']:.3f}")
print(f"Selected Method: {result['selected_method']}")
print(f"Router Confidence: {result['confidence']:.3f}")
print(f"\nTiming Breakdown:")
print(f"  - Feature Extraction: {result['timing']['feature_extraction_ms']:.2f}ms")
print(f"  - Router Inference: {result['timing']['router_inference_ms']:.2f}ms")
print(f"  - IQA Computation: {result['timing']['iqa_computation_ms']:.2f}ms")
print(f"  - Total Time: {result['timing']['total_time_ms']:.2f}ms")

print(f"\nRouter Probabilities:")
for method, prob in result["router_probabilities"].items():
    print(f"  - {method}: {prob:.4f}")

# Example 2: Batch processing
print("\n" + "=" * 60)
print("Example 2: Batch Processing")
print("=" * 60)

import os
from tqdm import tqdm

image_dir = "koniq10k_512x384"
image_files = [
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.lower().endswith(".jpg")
][
    :10
]  # First 10 images

results = []
for img_path in tqdm(image_files, desc="Processing images"):
    try:
        result = predict(img_path)
        results.append(result)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

# Summary statistics
import numpy as np

mos_scores = [r["MOS_estimate"] for r in results]
methods_used = [r["selected_method"] for r in results]
confidences = [r["confidence"] for r in results]
times = [r["timing"]["total_time_ms"] for r in results]

print(f"\nBatch Summary (n={len(results)}):")
print(f"  - MOS Range: {min(mos_scores):.3f} to {max(mos_scores):.3f}")
print(f"  - Mean MOS: {np.mean(mos_scores):.3f}")
print(f"  - Mean Confidence: {np.mean(confidences):.3f}")
print(f"  - Mean Processing Time: {np.mean(times):.2f}ms")
print(f"\nMethods Selected:")
from collections import Counter

for method, count in Counter(methods_used).most_common():
    print(f"  - {method}: {count} times ({count/len(results)*100:.1f}%)")

print("\n" + "=" * 60)
print("Deterministic routing ensures:")
print("  ✓ Always runs exactly 1 IQA method (fastest)")
print("  ✓ No confidence threshold tuning needed")
print("  ✓ Simplified code with single execution path")
print("  ✓ Linear regression MOS mapping preserved")
print("=" * 60)
