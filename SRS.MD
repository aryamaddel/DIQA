# Software Requirements Specification (SRS)
## Hybrid IQA Regressor: Classical + Deep Fusion

***

## 1. Introduction

### 1.1 Purpose
Define requirements for a lightweight image quality assessment (IQA) system that fuses a pretrained deep-IQA model’s score with classical no-reference metrics (BRISQUE, NIQE, PIQE) via a small regressor to predict perceptual quality.

### 1.2 Scope
The system computes four scalar inputs per image—deep model score, BRISQUE, NIQE, PIQE—and outputs a final quality score aligned with Mean Opinion Scores (MOS). It supports offline batch evaluation and real-time inference on resource-constrained devices.

### 1.3 Definitions, Acronyms, and Abbreviations
- IQA: Image Quality Assessment
- MOS: Mean Opinion Score
- BRISQUE: Blind/Referenceless Image Spatial Quality Evaluator
- NIQE: Naturalness Image Quality Evaluator
- PIQE: Perception-based Image Quality Evaluator
- CNN: Convolutional Neural Network
- MLP: Multi-Layer Perceptron
- SROCC/PCC: Spearman/Pearson correlation coefficient

### 1.4 References
- LIVE Dataset
- TID2013 Dataset
- KADID-10k Dataset

***

## 2. Overall Description

### 2.1 Product Perspective
A modular command-line and library-based tool. Classical IQA modules and a deep model feed scalar scores into a regression module.

### 2.2 Product Functions
- Image ingestion from disk or camera.
- Classical metrics extraction: BRISQUE, NIQE, PIQE.
- Deep model scoring via a pretrained lightweight CNN.
- Regressor fusion on normalized scalar inputs to produce final quality.
- Reporting of per-image scores and correlation vs. MOS.

### 2.3 User Characteristics
- Researchers in IQA and computer vision.
- Engineers deploying quality monitoring in media streaming, photography, or embedded vision.

### 2.4 Constraints
- Total inference per image ≤100ms on CPU-only ARM device.
- Memory footprint ≤200MB.

### 2.5 Assumptions and Dependencies
- Availability of pretrained deep IQA model and classical metric libraries.
- Ground-truth MOS available for offline evaluation.

***

## 3. Specific Requirements

### 3.1 Functional Requirements
- FR1: Accept images in JPEG, PNG, BMP.
- FR2: Compute BRISQUE, NIQE, PIQE per image.
- FR3: Load a pretrained CNN to produce a deep-IQA score per image.
- FR4: Normalize all four input scores to  before regression.
- FR5: Support at least two regressors: linear regression and MLP (1 hidden layer).
- FR6: Train the regressor on MOS-labeled datasets (LIVE, TID2013, KADID-10k).
- FR7: Output final scores in CSV or JSON with image identifiers.
- FR8: Report Pearson and Spearman correlations against MOS.
- FR9: Provide CLI with modes: train, evaluate, infer.

### 3.2 Non-Functional Requirements
- NFR1: Average inference time per image ≤100ms on ARM Cortex-A53 CPU.
- NFR2: Peak memory usage ≤200MB during inference.
- NFR3: Training utilizes GPU if available, otherwise CPU.
- NFR4: Achieve PCC ≥0.90 and SROCC ≥0.88 on validation sets.
- NFR5: Code follows PEP8 (for Python) or equivalent style guide.
- NFR6: Unit test coverage ≥90%.
- NFR7: Dependencies pinned with versions in requirements file.

***

## 4. System Architecture

### 4.1 High-Level Components
- Data Loader: Image I/O and ground-truth labels.
- Metric Extractors: BRISQUE, NIQE, PIQE.
- Deep IQA Model: Pretrained CNN inference.
- Regressor Module: Linear and MLP regressors.
- Evaluation & Reporting: Correlation metrics and exporters.
- CLI Interface: Pipeline orchestration.

### 4.2 Data Flow
Image → Metric Extractors + Deep Model → Normalizer → Regressor → Output → Evaluation

***

## 5. Data Requirements

### 5.1 Training Data
- LIVE, TID2013, KADID-10k with images and MOS.

### 5.2 Model and Artifacts
- pretrained_cnn.pth (deep model)
- scaler.pkl (MinMax/Standard scaler)
- regressor_linear.pkl, regressor_mlp.pkl

### 5.3 Outputs
- CSV/JSON with: image_id, brisque, niqe, piqe, deep_score, fused_score

***

## 6. Use Cases

### 6.1 Train Regressor
- Actors: Researcher
- Preconditions: Datasets and pretrained model available
- Flow:
  - Run: hyb_iqa train --data_dir data/ --model_dir models/ --regressor linear
  - System computes metrics and deep scores, trains regressor, saves artifacts.

### 6.2 Evaluate Performance
- Actors: Researcher
- Flow:
  - Run: hyb_iqa evaluate --data_dir data/ --model_dir models/ --output report.csv
  - System predicts scores, computes PCC/SROCC/RMSE, writes report.

### 6.3 Infer on New Images
- Actors: Engineer
- Flow:
  - Run: hyb_iqa infer --images imgs/ --model_dir models/ --output results.json
  - System outputs per-image fused quality scores.

***

## 7. Validation and Verification

- Unit Tests: Metric extractors, deep model loader, regressor, normalizer, CLI.
- Integration Tests: End-to-end training and inference on a small subset.
- Performance Tests: Inference time and memory on target hardware.
- Accuracy Tests: Verify PCC/SROCC targets on held-out datasets.

***

## 8. External Interfaces

### 8.1 CLI
- hyb_iqa train --data_dir --model_dir --regressor [linear|mlp] --epochs --lr
- hyb_iqa evaluate --data_dir --model_dir --output
- hyb_iqa infer --images --model_dir --output

### 8.2 Python API (example)
```python
from hyb_iqa import HybridIQA

model = HybridIQA.load(model_dir="models/")
score = model.infer("path/to/image.jpg")
```

***

## 9. Quality Attributes

- Performance: Meets latency and memory targets.
- Portability: Runs on Linux, Windows, macOS; supports ARM.
- Reliability: Deterministic outputs with fixed seeds; robust to missing metadata.
- Maintainability: Modular design, type hints, docs, tests.
- Usability: Clear CLI help, examples, and API docs.

***

## 10. Risks and Mitigations

- Risk: Classical metric implementations vary across libraries.
  - Mitigation: Version pinning; regression tests on known images.

- Risk: Domain shift across datasets reduces generalization.
  - Mitigation: Cross-dataset validation; optional domain-adaptive scaling.

- Risk: Deep model score ranges differ.
  - Mitigation: Robust normalization and calibration step.

***

## 11. Roadmap and Milestones

- M1: Prototype pipeline with BRISQUE+Deep+Linear regression (2 weeks).
- M2: Add NIQE/PIQE, normalization, evaluation metrics (2 weeks).
- M3: MLP regressor, ablations, cross-dataset validation (3 weeks).
- M4: Packaging, docs, tests, benchmarks on ARM (3 weeks).

***

## 12. Acceptance Criteria

- Trains and evaluates on at least two public IQA datasets.
- Achieves PCC ≥0.90 and SROCC ≥0.88 on validation.
- Inference latency ≤100ms/image and memory ≤200MB on target ARM device.
- CLI and Python API functional with documentation and ≥90% test coverage.

***

## 13. Appendices

### 13.1 Glossary
- See Section 1.3.

### 13.2 Versioning
- SRS v1.0 — Initial specification.
