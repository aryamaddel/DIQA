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

## 9. References

**BRISQUE (Mittal et al., 2012)**
[No-Reference Image Quality Assessment in the Spatial Domain](https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf) - Mittal, A., Moorthy, A. K., & Bovik, A. C.

**NIQE (Mittal et al., 2013)**
[Making a 'Completely Blind' Image Quality Analyzer](http://live.ece.utexas.edu/research/Quality/niqe_spl.pdf) - Mittal, A., Soundararajan, R., & Bovik, A. C.

**PIQE (Venkatanath et al., 2015)**
[Blind image quality evaluation using perception based features](https://www.semanticscholar.org/paper/Blind-image-quality-evaluation-using-perception-Venkatanath-Praneeth/1baf4fcf194bd10be327b1309543e55c1698e43d) - Venkatanath, N., Praneeth, D., Bh, M. C., Channappayya, S. S., & Medasani, S. S.

**HyperIQA (Su et al., 2020)**
[Blindly Assess Image Quality in the Wild Guided by a Self-Adaptive Hyper Network](https://openaccess.thecvf.com/content_CVPR_2020/papers/Su_Blindly_Assess_Image_Quality_in_the_Wild_Guided_by_a_CVPR_2020_paper.pdf) - Su, S., Yan, Q., Zhu, Y., Zhang, C., Ge, X., Sun, J., & Zhang, Y.

**MANIQA (Yang et al., 2022)**
[MANIQA: Multi-dimension Attention Network for No-Reference Image Quality Assessment](https://arxiv.org/abs/2204.08958) - Yang, S., Wu, T., Shi, S., Lao, S., Gong, Y., Cao, M., Wang, J., & Yang, Y.

**TOPIQ (Cui et al., 2023)**
[TOPIQ: A Top-down Approach from Semantics to Distortions for Image Quality Assessment](https://arxiv.org/abs/2308.03060) - Cui, C., Chen, C., Gu, K., Yuan, W., Chen, S., Zhang, L., & Liu, A.

**SSIM (Wang & Bovik, 2004)**
[Image Quality Assessment: From Error Visibility to Structural Similarity](https://www.cns.nyu.edu/pub/lcv/wang03-preprint.pdf) - Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.

**LIVE Database (Sheikh et al., 2006)**
[LIVE Image Quality Assessment Database Release 2](https://qualinet.github.io/databases/image/live_image_quality_assessment_database/) - Sheikh, H. R., Wang, Z., Cormack, L., & Bovik, A. C.

**TID2013 Database (Ponomarenko et al., 2015)**
[Image database TID2013: Peculiarities, results and perspectives](https://qualinet.github.io/databases/image/tampere_image_database_tid2013/) - Ponomarenko, N., Ieremeiev, O., Lukin, V., et al.

**KonIQ-10k Database (Hosu et al., 2020)**
[KonIQ-10k: An ecologically valid database for deep learning of blind image quality assessment](https://arxiv.org/abs/1910.06180) - Hosu, V., Lin, H., Sziranyi, T., & Saupe, D.

**KADID-10k Database**
[KADID-10k Database](https://database.mmsp-kn.de/kadid-10k-database.html) - Visual Quality Assessment Databases, MMSP Konstanz

**CSIQ Database (Larson & Chandler, 2010)**
[Categorical Image Quality (CSIQ) Database](https://qualinet.github.io/databases/image/categorical_image_quality_csiq_database/) - Larson, E. C. & Chandler, D. M.

[1] https://applied-informatics-j.springeropen.com/articles/10.1186/s40535-015-0010-x
[2] https://www.mathworks.com/help/images/ref/niqe.html
[3] https://openaccess.thecvf.com/content_CVPR_2020/papers/Su_Blindly_Assess_Image_Quality_in_the_Wild_Guided_by_a_CVPR_2020_paper.pdf
[4] https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf
[5] https://www.mathworks.com/help/images/ref/niqemodel.html
[6] https://www.scribd.com/document/895245965/Su-Blindly-Assess-Image-Quality-in-the-Wild-Guided-by-a-CVPR-2020-Paper
[7] https://pmc.ncbi.nlm.nih.gov/articles/PMC9921252/
[8] http://live.ece.utexas.edu/research/Quality/niqe_spl.pdf
[9] https://openaccess.thecvf.com/content_CVPR_2020/html/Su_Blindly_Assess_Image_Quality_in_the_Wild_Guided_by_a_CVPR_2020_paper.html
[10] https://arxiv.org/pdf/2210.10249.pdf
[11] https://arxiv.org/abs/2204.08958
[12] https://arxiv.org/abs/2308.03060
[13] https://quality.nfdi4ing.de/en/latest/image_quality/PIQE.html
[14] https://www.bohrium.com/paper-details/maniqa-multi-dimension-attention-network-for-no-reference-image-quality-assessment/867751619557589653-108597
[15] https://openaccess.thecvf.com/content/CVPR2023/papers/Chahine_An_Image_Quality_Assessment_Dataset_for_Portraits_CVPR_2023_paper.pdf
[16] https://www.mathworks.com/help/images/ref/piqe.html
[17] https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Yang_MANIQA_Multi-Dimension_Attention_Network_for_No-Reference_Image_Quality_Assessment_CVPRW_2022_paper.pdf
[18] https://dl.acm.org/doi/10.1109/TIP.2024.3378466
[19] https://pypi.org/project/pypiqe/
[20] https://github.com/IIGROUP/MANIQA
[21] https://www.mathworks.com/help/images/image-quality-metrics.html
[22] https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12035/1203515/Performance-evaluation-of-image-quality-metrics-for-perceptual-assessment-of/10.1117/12.2612541.full
[23] https://qualinet.github.io/databases/image/live_image_quality_assessment_database/
[24] https://qualinet.github.io/databases/image/tampere_image_database_tid2013/
[25] https://database.mmsp-kn.de/koniq-10k-database.html
[26] https://www.mathworks.com/matlabcentral/answers/464038-live-database-for-image-quality-assessment
[27] https://www.repository.cam.ac.uk/items/a5044faf-d078-47d2-aa28-31d22491c039
[28] https://arxiv.org/abs/1910.06180
[29] http://live.ece.utexas.edu/research/Quality/
[30] https://github.com/hcl14/AVA-and-TID2013-image-quality-assessment
[31] https://arxiv.org/pdf/2209.10451.pdf
[32] https://live.ece.utexas.edu/research/quality/subjective.htm
[33] https://huggingface.co/datasets/xmba15/TID2013
[34] https://www.sciencedirect.com/science/article/pii/S0923596514001490
[35] https://pmc.ncbi.nlm.nih.gov/articles/PMC11457998/
[36] https://en.wikipedia.org/wiki/Structural_similarity_index_measure
[37] https://www.semanticscholar.org/paper/Blind-image-quality-evaluation-using-perception-Venkatanath-Praneeth/1baf4fcf194bd10be327b1309543e55c1698e43d
[38] https://qualinet.github.io/databases/image/categorical_image_quality_csiq_database/
[39] https://www.cns.nyu.edu/pub/lcv/wang03-preprint.pdf
[40] http://live.ece.utexas.edu/research/twostep/index.html
[41] https://pmc.ncbi.nlm.nih.gov/articles/PMC5527267/
[42] https://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf

