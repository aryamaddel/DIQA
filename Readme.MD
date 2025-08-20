Here’s a **condensed and merged version** of your two drafts. I’ve unified the research motivation, objectives, and system requirements into a single structured document while removing repetition.

---

# Hybrid NR-IQA System: Classical + Deep Fusion

---

## 1. Introduction

The rapid growth of digital media has increased the need for reliable **Image Quality Assessment (IQA)**. Traditional full-reference methods require pristine originals, which are rarely available. **No-Reference IQA (NR-IQA)** addresses this by predicting perceptual quality without references.

However, existing NR-IQA models often struggle with **generalization across distortions and content types**. This project proposes a **hybrid IQA system** that fuses **classical NR metrics** (BRISQUE, NIQE, PIQE) with a **pretrained deep-IQA model** using a lightweight regressor. The goal is to align predictions with **human Mean Opinion Scores (MOS)** for applications such as adaptive streaming, surveillance, and mobile photography.

---

## 2. Motivation & Significance

* **Niche & High Impact:** NR-IQA deals with subjective human perception, a challenging and impactful subfield of computer vision.
* **Real-world Applications:**

  * Video streaming (adaptive bitrate control)
  * Surveillance (flagging poor frames)
  * Mobile cameras (real-time capture feedback)
  * Compression/storage (balancing size vs. perceptual quality)
* **Balanced Approach:** Combines handcrafted features and deep CNN features.
* **Scalability:** Can start with classical metrics, extend to advanced deep learning models.

---

## 3. Objectives

1. **Dataset Preparation**: Acquire and preprocess standard IQA datasets (e.g., **LIVE**, **TID2013**, **KonIQ-10k**, **KADID-10k**).
2. **Feature Extraction**: Compute BRISQUE, NIQE, PIQE and extract deep CNN features.
3. **Model Development**: Train regressors (Linear, Random Forest, SVR, MLP) to fuse features and predict MOS.
4. **Performance Evaluation**: Use MSE, MAE, PLCC, and SRCC to assess alignment with human perception.
5. **Deliverables**: A functional hybrid NR-IQA system, comparative analysis, and documented codebase.

---

## 4. System Overview

### 4.1 Product Perspective

A modular pipeline with classical metric extractors, a pretrained deep IQA model, and a regressor fusion stage.

### 4.2 Functional Flow

Image → Metric Extractors + Deep Model → Normalizer → Regressor → Quality Score → Evaluation

### 4.3 Core Functions

* Image ingestion (JPEG, PNG, BMP)
* Compute BRISQUE, NIQE, PIQE
* Generate pretrained CNN IQA score
* Fuse inputs with regressor (Linear / MLP)
* Output results in CSV/JSON with correlation metrics

---

## 5. Requirements

### 5.1 Functional

* Compute classical + deep features per image
* Normalize features before regression
* Train on MOS-labeled datasets
* Support multiple regressors (Linear, MLP, SVR, RF)
* Export results and correlations (Pearson/Spearman)

### 5.2 Non-Functional

* Inference ≤100 ms on ARM CPU
* Memory footprint ≤200 MB
* Achieve PLCC ≥0.90, SRCC ≥0.88 on validation sets
* Unit test coverage ≥90%
* Dependencies version-pinned

---

## 6. Datasets & Data Requirements

* **Training Data**: LIVE, TID2013, KonIQ-10k, KADID-10k (images + MOS)
* **Model Files**: pretrained CNN weights, normalization scaler, trained regressors
* **Outputs**: image\_id, brisque, niqe, piqe, deep\_score, fused\_score

---

## 7. Evaluation & Validation

* **Prediction Accuracy**: MSE, MAE
* **Perceptual Correlation**: PLCC, SRCC
* **Verification**: Unit tests, integration tests, hardware performance tests
* **Generalization Check**: Cross-dataset testing

---

## 8. Future Enhancements

* Feature fusion (handcrafted + deep)
* End-to-end deep NR-IQA models
* Attention mechanisms for perceptually important regions
* Optimized real-time deployment

---

## 9. References (Selected)

* BRISQUE (Mittal et al., 2012)
* NIQE (Mittal et al., 2013)
* HyperIQA (Su et al., 2020)
* MANIQA (Yang et al., 2022)
* TOPIQ (Cui et al., 2023)

---

✅ This version integrates the **research-driven structure** (intro, problem, significance) with the **practical SRS-style design** (requirements, flow, datasets, validation).

Do you want me to **keep it in SRS format** (strict requirements doc) or more of a **research proposal style** (for project/thesis)?
