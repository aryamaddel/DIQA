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

BRISQUE (Mittal et al., 2012)
No-Reference Image Quality Assessment in the Spatial Domain - Mittal, A., Moorthy, A. K., & Bovik, A. C.

NIQE (Mittal et al., 2013)
Making a 'Completely Blind' Image Quality Analyzer - Mittal, A., Soundararajan, R., & Bovik, A. C.

PIQE (Venkatanath et al., 2015)
Blind image quality evaluation using perception based features - Venkatanath, N., Praneeth, D., Bh, M. C., Channappayya, S. S., & Medasani, S. S.

HyperIQA (Su et al., 2020)
Blindly Assess Image Quality in the Wild Guided by a Self-Adaptive Hyper Network - Su, S., Yan, Q., Zhu, Y., Zhang, C., Ge, X., Sun, J., & Zhang, Y.

MANIQA (Yang et al., 2022)
MANIQA: Multi-dimension Attention Network for No-Reference Image Quality Assessment - Yang, S., Wu, T., Shi, S., Lao, S., Gong, Y., Cao, M., Wang, J., & Yang, Y.

TOPIQ (Cui et al., 2023)
TOPIQ: A Top-down Approach from Semantics to Distortions for Image Quality Assessment - Cui, C., Chen, C., Gu, K., Yuan, W., Chen, S., Zhang, L., & Liu, A.

SSIM (Wang & Bovik, 2004)
Image Quality Assessment: From Error Visibility to Structural Similarity - Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.

LIVE Database (Sheikh et al., 2006)
LIVE Image Quality Assessment Database Release 2 - Sheikh, H. R., Wang, Z., Cormack, L., & Bovik, A. C.

TID2013 Database (Ponomarenko et al., 2015)
Image database TID2013: Peculiarities, results and perspectives - Ponomarenko, N., Ieremeiev, O., Lukin, V., et al.

KonIQ-10k Database (Hosu et al., 2020)
KonIQ-10k: An ecologically valid database for deep learning of blind image quality assessment - Hosu, V., Lin, H., Sziranyi, T., & Saupe, D.

KADID-10k Database
KADID-10k Database - Visual Quality Assessment Databases, MMSP Konstanz

CSIQ Database (Larson & Chandler, 2010)
Categorical Image Quality (CSIQ) Database - Larson, E. C. & Chandler, D. M.

